// rdma_ar.cu: RDMA all-reduce / all-gather for small tensors, one GPU per node (DGX Spark / GB10).
//
// NCCL's net transport costs one CPU-proxy hop per ring step (14 hops for an 8-rank ring all-reduce,
// ~150-190 us inside a CUDA graph). Three modes here, all one kernel launch:
//   one-shot reduce : every rank writes its whole input into each peer's receive slot (one hop),
//                     then sums the world contributions locally in rank order
//   two-shot reduce : reduce-scatter (chunk q of my input -> rank q), rank q sums chunk q in rank
//                     order and broadcasts it (two hops, 4x fewer bytes on the wire than one-shot)
//   all-gather      : one-shot copy, out[r] = input of rank r
// Every element of a reduction is summed by one fixed rule, so all ranks see bit-identical results.
//
// Data path (all buffers are cudaHostAlloc'd = the same LPDDR5X on GB10, registered once per HCA):
//   GPU kernel: stage input in the send slot, last CTA publishes ctrl.send_seq
//   proxy thread: sees send_seq, posts per peer {RDMA WRITE payload, RDMA WRITE 8-byte seq flag}
//                 (same RC QP, so the flag lands after the payload)
//   GPU kernel: spins until every peer's flag for this slot reaches seq, then reduces / copies
// The sequence number lives in device memory and is advanced by the kernel itself, so the ops are
// safe to capture in CUDA graphs (kernel args never change between replays).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <sys/mman.h>
#include <fstream>
#include <ctime>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kSlots = 4;
constexpr int kThreads = 512;
constexpr int kSignalEvery = 32;  // request a CQE every N posts per QP so the SQ drains

enum Mode : int { kOneShot = 0, kGather = 1, kTwoShot = 2 };

#define CHECK_CUDA(x)                                                                        \
  do {                                                                                       \
    cudaError_t e_ = (x);                                                                    \
    if (e_ != cudaSuccess)                                                                   \
      throw std::runtime_error(std::string("rdma_ar CUDA: ") + cudaGetErrorString(e_) +      \
                               " at " #x);                                                   \
  } while (0)
#define CHECK_IB(cond, what)                                                                 \
  do {                                                                                       \
    if (!(cond))                                                                             \
      throw std::runtime_error(std::string("rdma_ar ibverbs: ") + (what) +                   \
                               " errno=" + std::to_string(errno) + " " + strerror(errno));  \
  } while (0)

struct alignas(64) Ctrl {
  uint64_t send_seq;    // GPU -> proxy: phase-1 message `seq` is staged in send slot seq % kSlots
  uint64_t send_bytes;  // whole message bytes (written before send_seq)
  uint64_t mode;        // Mode
  uint64_t chunk_vecs;  // two-shot chunk size in 16-byte vectors
  uint64_t pad0[4];
  uint64_t posted_seq;  // proxy -> GPU: phase-1 writes for this seq have been posted
  uint64_t pad1[7];
  uint64_t send2_seq;   // GPU -> proxy: two-shot reduced chunk staged in send2 slot
  uint64_t send2_bytes;
  uint64_t pad2[6];
  uint64_t posted2_seq;  // proxy -> GPU: phase-2 writes posted
  uint64_t pad3[7];
  uint64_t pub_time;    // GPU globaltimer at the phase-1 publish (for proxy hand-off timing)
  uint64_t pad4[7];
  uint64_t ping;        // GPU -> proxy hand-off probe
  uint64_t pad5[7];
  uint64_t pong;        // proxy echo
  uint64_t pad6[7];
};

struct DevState {
  unsigned long long seq;  // last completed op
  unsigned int arrive;     // CTA arrival counter, phase 1 staging
  unsigned int arrive2;    // CTA arrival counter, phase 2 staging
  unsigned int depart;     // CTA departure counter
  // timing (ns, globaltimer; recorded by thread 0 of CTA 0 / the publishing thread)
  unsigned long long t_start, t_pub;
  long long sum_stage, sum_net, sum_tail, sum_total;  // start->pub, pub->flags, flags->end, start->end
  unsigned long long n_timed;
};

struct Params {
  const uint4* in;
  uint4* out;
  long long nvec;        // message size in 16-byte vectors
  long long chunk;       // two-shot chunk size in vectors
  char* send;            // kSlots * max_bytes
  char* send2;           // kSlots * max_chunk
  const char* recv;      // kSlots * world * max_bytes
  const char* recv2;     // kSlots * world * max_chunk
  const uint64_t* flags;   // kSlots * world
  const uint64_t* flags2;  // kSlots * world
  Ctrl* ctrl;
  DevState* st;
  long long max_bytes, max_chunk;
  int rank, world;
  long long timeout_ns;
  int timing;  // record per-call stage timings (RDMA_AR_TIMING=1)
};

__device__ __forceinline__ uint64_t ld_acquire_sys(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(uint64_t* p, uint64_t v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void spin_until(const uint64_t* f, uint64_t seq, const Params& p, int who, int phase) {
  const unsigned long long t0 = now_ns();
  while (ld_acquire_sys(f) < seq) {
    if (now_ns() - t0 > (unsigned long long)p.timeout_ns) {
      printf("rdma_ar rank %d: timeout waiting for %d phase %d (seq %llu)\n", p.rank, who, phase,
             (unsigned long long)seq);
      asm volatile("trap;");
    }
  }
}

// True for exactly one thread: thread 0 of the last CTA through a grid-wide arrival counter (all CTAs'
// prior writes, fenced at system scope, are visible to it). That thread then publishes to the proxy.
__device__ __forceinline__ bool last_cta(unsigned int* arrive, int* s_last) {
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) *s_last = (atomicAdd(arrive, 1u) == gridDim.x - 1);
  __syncthreads();
  if (*s_last && threadIdx.x == 0) {
    *arrive = 0;
    __threadfence_system();
    return true;
  }
  return false;
}

__device__ __forceinline__ void acc8(float* acc, const uint4& v) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float2 f = __bfloat1622float2(h[k]);
    acc[2 * k] += f.x;
    acc[2 * k + 1] += f.y;
  }
}
__device__ __forceinline__ uint4 pack8(const float* acc) {
  uint4 o;
  __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int k = 0; k < 4; ++k) oh[k] = __floats2bfloat162_rn(acc[2 * k], acc[2 * k + 1]);
  return o;
}

// GPU<->proxy round trip through host-pinned memory, timed on the GPU clock only.
__global__ void pingpong_kernel(Ctrl* ctrl, int n, unsigned long long base, unsigned long long* out_ns) {
  const unsigned long long t0 = now_ns();
  for (int i = 1; i <= n; ++i) {
    st_release_sys(&ctrl->ping, base + i);
    while (ld_acquire_sys(&ctrl->pong) != base + i) {
    }
  }
  *out_ns = now_ns() - t0;
}

template <int kMode>
__global__ void __launch_bounds__(kThreads) ar_kernel(Params p) {
  __shared__ unsigned long long s_seq;
  __shared__ int s_last;
  if (threadIdx.x == 0) s_seq = *reinterpret_cast<volatile unsigned long long*>(&p.st->seq) + 1;
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) p.st->t_start = now_ns();
  __syncthreads();
  const unsigned long long seq = s_seq;
  const int slot = static_cast<int>(seq % kSlots);
  const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long stride = (long long)gridDim.x * blockDim.x;

  // 1) stage the input in this rank's send slot and hand it to the proxy
  uint4* send4 = reinterpret_cast<uint4*>(p.send + slot * p.max_bytes);
  for (long long i = tid; i < p.nvec; i += stride) send4[i] = p.in[i];
  if (last_cta(&p.st->arrive, &s_last)) {
    // the proxy must have consumed the previous message on this channel before its descriptor is
    // overwritten (compare with what was last published, not seq-1: modes interleave)
    spin_until(&p.ctrl->posted_seq, ld_acquire_sys(&p.ctrl->send_seq), p, -1, 0);
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send_bytes) = (uint64_t)p.nvec * 16;
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->mode) = (uint64_t)kMode;
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->chunk_vecs) = (uint64_t)p.chunk;
    if (p.timing) {
      const unsigned long long tp = now_ns();
      p.st->t_pub = tp;
      *reinterpret_cast<volatile uint64_t*>(&p.ctrl->pub_time) = tp;
    }
    st_release_sys(&p.ctrl->send_seq, seq);
  }

  // 2) wait for every peer's phase-1 contribution
  if (threadIdx.x < p.world && threadIdx.x != p.rank)
    spin_until(p.flags + slot * p.world + threadIdx.x, seq, p, threadIdx.x, 1);
  __syncthreads();
  unsigned long long t_flags = 0;
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) t_flags = now_ns();

  const char* rbase = p.recv + (long long)slot * p.world * p.max_bytes;
  if constexpr (kMode == kGather) {
    for (int r = 0; r < p.world; ++r) {
      const uint4* src = reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes);
      uint4* dst = p.out + (long long)r * p.nvec;
      for (long long i = tid; i < p.nvec; i += stride) dst[i] = (r == p.rank) ? p.in[i] : __ldcv(src + i);
    }
  } else if constexpr (kMode == kOneShot) {
    for (long long i = tid; i < p.nvec; i += stride) {
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      for (int r = 0; r < p.world; ++r)
        acc8(acc, (r == p.rank) ? p.in[i]
                                : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes) + i));
      p.out[i] = pack8(acc);
    }
  } else {
    // two-shot: peers wrote chunk `rank` of their input at offset 0 of recv[slot][peer]
    const long long b = (long long)p.rank * p.chunk;
    const long long len = max(0LL, min(p.chunk, p.nvec - b));
    uint4* send2 = reinterpret_cast<uint4*>(p.send2 + slot * p.max_chunk);
    for (long long i = tid; i < len; i += stride) {
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      for (int r = 0; r < p.world; ++r)
        acc8(acc, (r == p.rank) ? p.in[b + i]
                                : __ldcv(reinterpret_cast<const uint4*>(rbase + (long long)r * p.max_bytes) + i));
      const uint4 o = pack8(acc);
      p.out[b + i] = o;
      send2[i] = o;
    }
    if (last_cta(&p.st->arrive2, &s_last)) {
      spin_until(&p.ctrl->posted2_seq, ld_acquire_sys(&p.ctrl->send2_seq), p, -2, 0);
      *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send2_bytes) = (uint64_t)len * 16;
      st_release_sys(&p.ctrl->send2_seq, seq);
    }
    if (threadIdx.x < p.world && threadIdx.x != p.rank)
      spin_until(p.flags2 + slot * p.world + threadIdx.x, seq, p, threadIdx.x, 2);
    __syncthreads();
    const char* r2 = p.recv2 + (long long)slot * p.world * p.max_chunk;
    for (int r = 0; r < p.world; ++r) {
      if (r == p.rank) continue;
      const long long br = (long long)r * p.chunk;
      const long long lr = max(0LL, min(p.chunk, p.nvec - br));
      const uint4* src = reinterpret_cast<const uint4*>(r2 + (long long)r * p.max_chunk);
      for (long long i = tid; i < lr; i += stride) p.out[br + i] = __ldcv(src + i);
    }
  }

  // 3) last CTA out advances the sequence for the next call
  __syncthreads();
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) {
    __threadfence();
    const unsigned long long t_end = now_ns();
    const unsigned long long ts = p.st->t_start;
    const unsigned long long tp = *reinterpret_cast<volatile unsigned long long*>(&p.st->t_pub);
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_stage), (unsigned long long)(long long)(tp - ts));
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_net), (unsigned long long)(long long)(t_flags - tp));
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_tail), (unsigned long long)(long long)(t_end - t_flags));
    atomicAdd(reinterpret_cast<unsigned long long*>(&p.st->sum_total), (unsigned long long)(long long)(t_end - ts));
    atomicAdd(&p.st->n_timed, 1ull);
  }
  if (threadIdx.x == 0 && atomicAdd(&p.st->depart, 1u) == gridDim.x - 1) {
    p.st->depart = 0;
    __threadfence();
    p.st->seq = seq;
  }
}

struct Hca {
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_mr* mr = nullptr;
  ibv_gid gid{};
  ibv_mtu mtu = IBV_MTU_1024;
};

struct Peer {
  int hca = -1;
  ibv_qp* qp = nullptr;
  uint64_t remote_base = 0;
  uint32_t rkey = 0;
  uint64_t posts = 0;
};

class RdmaAR {
 public:
  RdmaAR(int rank, int world, int64_t max_bytes, int64_t max_reduce_bytes, std::vector<std::string> hca_names,
         int gid_index, int ib_port, int proxy_core, int64_t timeout_ms, int device)
      : rank_(rank), world_(world), max_bytes_(max_bytes), gid_index_(gid_index), ib_port_(ib_port),
        proxy_core_(proxy_core), timeout_ns_(timeout_ms * 1000000LL), device_(device) {
    if (max_bytes % 16 || max_reduce_bytes % 16) throw std::runtime_error("rdma_ar: sizes must be multiples of 16");
    if (world < 2 || world > kThreads) throw std::runtime_error("rdma_ar: bad world size");
    max_reduce_bytes_ = std::min(max_reduce_bytes, max_bytes);
    max_chunk_ = align((size_t)((max_reduce_bytes_ / 16 + world - 1) / world) * 16, 64);
    c10::cuda::CUDAGuard guard(device_);
    // one host-pinned arena: [ctrl | flags | flags2 | send | send2 | recv | recv2], registered on every HCA
    const size_t fl = align(sizeof(uint64_t) * kSlots * world, 4096);
    flags_off_ = 4096;
    flags2_off_ = flags_off_ + fl;
    send_off_ = flags2_off_ + fl;
    send2_off_ = send_off_ + align(kSlots * max_bytes, 4096);
    recv_off_ = send2_off_ + align(kSlots * max_chunk_, 4096);
    recv2_off_ = recv_off_ + align((size_t)kSlots * world * max_bytes, 4096);
    arena_bytes_ = recv2_off_ + align((size_t)kSlots * world * max_chunk_, 4096);
    // Back the arena with 2 MB transparent huge pages (THP is in madvise mode on DGX OS, so a user
    // process may ask for it) and pin it with cudaHostRegister. cudaHostAlloc takes whatever pages
    // the allocator has; on a fragmented node that is 4 KB pages, and the GPU's staging copies and
    // peer reads then miss the TLB constantly (measured 3x slower stage/reduce on such ranks, and
    // every other rank waits for them). RDMA_AR_THP=0 falls back to cudaHostAlloc.
    const size_t kHuge = 2u << 20;
    arena_bytes_ = align(arena_bytes_, kHuge);
    const char* thp = getenv("RDMA_AR_THP");
    if (!thp || atoi(thp) != 0) {
      void* raw = mmap(nullptr, arena_bytes_ + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (raw != MAP_FAILED) {
        char* a = reinterpret_cast<char*>(align((size_t)(uintptr_t)raw, kHuge));
        madvise(a, arena_bytes_, MADV_HUGEPAGE);
        memset(a, 0, arena_bytes_);  // fault in now, as huge pages where the kernel can
        if (cudaHostRegister(a, arena_bytes_, cudaHostRegisterMapped | cudaHostRegisterPortable) == cudaSuccess) {
          arena_ = a;
          arena_registered_ = true;
        } else {
          cudaGetLastError();
          munmap(raw, arena_bytes_ + kHuge);
        }
      }
    }
    if (!arena_registered_) {
      CHECK_CUDA(cudaHostAlloc(&arena_, arena_bytes_, cudaHostAllocMapped | cudaHostAllocPortable));
      memset(arena_, 0, arena_bytes_);
    }
    CHECK_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&arena_dev_), arena_, 0));
    thp_bytes_ = huge_backed_bytes(arena_, arena_bytes_);
    CHECK_CUDA(cudaMalloc(&st_, sizeof(DevState)));
    CHECK_CUDA(cudaMemset(st_, 0, sizeof(DevState)));
    ctrl_ = reinterpret_cast<Ctrl*>(arena_);

    int ndev = 0;
    ibv_device** list = ibv_get_device_list(&ndev);
    CHECK_IB(list != nullptr, "ibv_get_device_list");
    for (auto& name : hca_names) {
      Hca h;
      for (int i = 0; i < ndev; ++i)
        if (name == ibv_get_device_name(list[i])) h.ctx = ibv_open_device(list[i]);
      CHECK_IB(h.ctx != nullptr, "open device " + name);
      h.pd = ibv_alloc_pd(h.ctx);
      CHECK_IB(h.pd, "alloc_pd");
      h.cq = ibv_create_cq(h.ctx, 4096, nullptr, nullptr, 0);
      CHECK_IB(h.cq, "create_cq");
      h.mr = ibv_reg_mr(h.pd, arena_, arena_bytes_, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      CHECK_IB(h.mr, "reg_mr " + name);
      CHECK_IB(ibv_query_gid(h.ctx, ib_port_, gid_index_, &h.gid) == 0, "query_gid");
      ibv_port_attr pa{};
      CHECK_IB(ibv_query_port(h.ctx, ib_port_, &pa) == 0, "query_port");
      h.mtu = pa.active_mtu;
      hcas_.push_back(h);
    }
    ibv_free_device_list(list);

    peers_.resize(world);
    for (int p = 0; p < world; ++p) {
      if (p == rank) continue;
      Peer& pr = peers_[p];
      pr.hca = (rank + p) % (int)hcas_.size();  // both ends of a pair agree on the HCA
      Hca& h = hcas_[pr.hca];
      ibv_qp_init_attr ia{};
      ia.send_cq = h.cq;
      ia.recv_cq = h.cq;
      ia.qp_type = IBV_QPT_RC;
      ia.cap.max_send_wr = 512;
      ia.cap.max_recv_wr = 1;
      ia.cap.max_send_sge = 1;
      ia.cap.max_recv_sge = 1;
      ia.cap.max_inline_data = 64;
      ia.sq_sig_all = 0;
      pr.qp = ibv_create_qp(h.pd, &ia);
      CHECK_IB(pr.qp, "create_qp");
      ibv_qp_attr a{};
      a.qp_state = IBV_QPS_INIT;
      a.pkey_index = 0;
      a.port_num = ib_port_;
      a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
      CHECK_IB(ibv_modify_qp(pr.qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0,
               "qp INIT");
    }
  }

  ~RdmaAR() {
    stop_.store(true);
    if (proxy_.joinable()) proxy_.join();
    // QPs/MRs are left to process teardown; the arena may still be referenced by in-flight graphs.
  }

  // [gid_hi, gid_lo] per HCA, then rkey per HCA, then arena addr, then qpn per rank (0 for self)
  std::vector<int64_t> local_info() const {
    std::vector<int64_t> v;
    for (auto& h : hcas_) {
      uint64_t hi, lo;
      memcpy(&hi, h.gid.raw, 8);
      memcpy(&lo, h.gid.raw + 8, 8);
      v.push_back((int64_t)hi);
      v.push_back((int64_t)lo);
    }
    for (auto& h : hcas_) v.push_back((int64_t)h.mr->rkey);
    v.push_back((int64_t)(uintptr_t)arena_);
    for (int p = 0; p < world_; ++p) v.push_back(p == rank_ ? 0 : (int64_t)peers_[p].qp->qp_num);
    return v;
  }

  void connect(const std::vector<std::vector<int64_t>>& infos) {
    const int nh = (int)hcas_.size();
    for (int p = 0; p < world_; ++p) {
      if (p == rank_) continue;
      Peer& pr = peers_[p];
      const auto& r = infos[p];
      Hca& h = hcas_[pr.hca];
      ibv_gid rgid{};
      uint64_t hi = (uint64_t)r[2 * pr.hca], lo = (uint64_t)r[2 * pr.hca + 1];
      memcpy(rgid.raw, &hi, 8);
      memcpy(rgid.raw + 8, &lo, 8);
      pr.rkey = (uint32_t)r[2 * nh + pr.hca];
      pr.remote_base = (uint64_t)r[3 * nh];
      uint32_t rqpn = (uint32_t)r[3 * nh + 1 + rank_];

      ibv_qp_attr a{};
      a.qp_state = IBV_QPS_RTR;
      a.path_mtu = h.mtu;
      a.dest_qp_num = rqpn;
      a.rq_psn = 0;
      a.max_dest_rd_atomic = 1;
      a.min_rnr_timer = 12;
      a.ah_attr.is_global = 1;
      a.ah_attr.grh.dgid = rgid;
      a.ah_attr.grh.sgid_index = gid_index_;
      a.ah_attr.grh.hop_limit = 64;
      a.ah_attr.port_num = ib_port_;
      CHECK_IB(ibv_modify_qp(pr.qp, &a,
                             IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                 IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0,
               "qp RTR");
      ibv_qp_attr b{};
      b.qp_state = IBV_QPS_RTS;
      b.timeout = 14;
      b.retry_cnt = 7;
      b.rnr_retry = 7;
      b.sq_psn = 0;
      b.max_rd_atomic = 1;
      CHECK_IB(ibv_modify_qp(pr.qp, &b,
                             IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                 IBV_QP_MAX_QP_RD_ATOMIC) == 0,
               "qp RTS");
    }
    proxy_ = std::thread([this] { proxy_loop(); });
  }

  torch::Tensor all_reduce(torch::Tensor x, bool two_shot) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
                "rdma_ar: need contiguous bf16 CUDA");
    TORCH_CHECK(x.numel() * 2 <= max_reduce_bytes_, "rdma_ar: all-reduce too large");
    auto out = torch::empty_like(x);
    if (two_shot)
      launch<kTwoShot>(x, out);
    else
      launch<kOneShot>(x, out);
    return out;
  }

  // out = concat over ranks (dim 0) of x; any dtype, out must be contiguous with world * x.numel() elements
  void all_gather_into(torch::Tensor out, torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && out.is_contiguous() && out.scalar_type() == x.scalar_type() &&
                    out.numel() == x.numel() * world_,
                "rdma_ar: bad all_gather args");
    launch<kGather>(x, out);
  }

  int64_t max_bytes() const { return max_bytes_; }
  // average GPU->proxy->GPU round trip (ns) through the control block
  double pingpong(int n) {
    unsigned long long* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(unsigned long long)));
    static unsigned long long base = 1ull << 40;
    base += 1ull << 20;
    pingpong_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<Ctrl*>(arena_dev_), n, base, d_out);
    unsigned long long h = 0;
    CHECK_CUDA(cudaMemcpyAsync(&h, d_out, sizeof(h), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream()));
    CHECK_CUDA(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
    CHECK_CUDA(cudaFree(d_out));
    return (double)h / n;
  }

  // [arena bytes, bytes backed by huge pages, 1 if THP+cudaHostRegister path was used]
  std::vector<int64_t> arena_info() const { return {(int64_t)arena_bytes_, thp_bytes_, arena_registered_ ? 1 : 0}; }
  int64_t max_reduce_bytes() const { return max_reduce_bytes_; }

 private:
  static size_t align(size_t x, size_t a) { return (x + a - 1) / a * a; }

  // bytes of [p, p+n) backed by anonymous huge pages, from /proc/self/smaps
  static int64_t huge_backed_bytes(void* p, size_t n) {
    std::ifstream f("/proc/self/smaps");
    std::string line;
    const uintptr_t lo = (uintptr_t)p, hi = lo + n;
    bool in = false;
    int64_t kb = 0;
    while (std::getline(f, line)) {
      unsigned long a, b;
      if (sscanf(line.c_str(), "%lx-%lx", &a, &b) == 2) {  // mapping header line
        in = a < hi && b > lo;
        continue;
      }
      long v;
      if (in && sscanf(line.c_str(), "AnonHugePages: %ld kB", &v) == 1) kb += v;
    }
    return kb * 1024;
  }

  template <int kMode>
  void launch(const torch::Tensor& x, torch::Tensor& out) {
    const int64_t nbytes = x.numel() * x.element_size();
    TORCH_CHECK(nbytes % 16 == 0 && nbytes <= max_bytes_, "rdma_ar: bad size ", nbytes);
    TORCH_CHECK(!proxy_error_.load(), "rdma_ar: proxy reported an RDMA error");
    if (nbytes == 0) return;
    Params p;
    p.in = reinterpret_cast<const uint4*>(x.data_ptr());
    p.out = reinterpret_cast<uint4*>(out.data_ptr());
    p.nvec = nbytes / 16;
    p.chunk = (p.nvec + world_ - 1) / world_;
    p.send = arena_dev_ + send_off_;
    p.send2 = arena_dev_ + send2_off_;
    p.recv = arena_dev_ + recv_off_;
    p.recv2 = arena_dev_ + recv2_off_;
    p.flags = reinterpret_cast<const uint64_t*>(arena_dev_ + flags_off_);
    p.flags2 = reinterpret_cast<const uint64_t*>(arena_dev_ + flags2_off_);
    p.ctrl = reinterpret_cast<Ctrl*>(arena_dev_);
    p.st = st_;
    p.max_bytes = max_bytes_;
    p.max_chunk = (long long)max_chunk_;
    p.rank = rank_;
    p.world = world_;
    p.timeout_ns = timeout_ns_;
    p.timing = timing_;
    // a few CTAs are enough: the cost is latency, not bandwidth (RDMA_AR_VEC_PER_CTA tunes it)
    long long blocks = (p.nvec + vec_per_cta_ - 1) / vec_per_cta_;
    blocks = std::max(1LL, std::min(blocks, 16LL));
    auto stream = at::cuda::getCurrentCUDAStream();
    ar_kernel<kMode><<<(int)blocks, kThreads, 0, stream>>>(p);
    CHECK_CUDA(cudaGetLastError());
  }

  void post(Peer& pr, uint64_t src, uint32_t len, uint64_t dst, uint64_t flag_dst, uint64_t flag) {
    Hca& h = hcas_[pr.hca];
    ibv_sge sge{};
    sge.addr = src;
    sge.length = len;
    sge.lkey = h.mr->lkey;
    ibv_sge fsge{};
    fsge.addr = (uint64_t)(uintptr_t)&flag;
    fsge.length = 8;
    fsge.lkey = 0;
    ibv_send_wr wr[2]{};
    wr[0].opcode = IBV_WR_RDMA_WRITE;
    wr[0].sg_list = &sge;
    wr[0].num_sge = 1;
    wr[0].wr.rdma.remote_addr = dst;
    wr[0].wr.rdma.rkey = pr.rkey;
    wr[0].next = &wr[1];
    wr[1].opcode = IBV_WR_RDMA_WRITE;
    wr[1].sg_list = &fsge;
    wr[1].num_sge = 1;
    wr[1].send_flags = IBV_SEND_INLINE | ((++pr.posts % kSignalEvery) == 0 ? IBV_SEND_SIGNALED : 0);
    wr[1].wr.rdma.remote_addr = flag_dst;
    wr[1].wr.rdma.rkey = pr.rkey;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(pr.qp, len ? &wr[0] : &wr[1], &bad)) {
      fprintf(stderr, "rdma_ar rank %d: ibv_post_send failed errno=%d\n", rank_, errno);
      proxy_error_.store(true);
    }
  }

  void proxy_loop() {
    if (proxy_core_ >= 0) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(proxy_core_, &set);
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    const uint64_t base = (uint64_t)(uintptr_t)arena_;
    uint64_t last1 = 0, last2 = 0;
    ibv_wc wc[32];
    // Spin while traffic flows (and for RDMA_AR_SPIN_US after the last message, which covers the
    // CPU gaps between decode steps); after that, back off to short sleeps so an idle server does
    // not pin a 3.9 GHz core (on GB10 CPU heat is shared with the GPU). Worst case the first
    // all-reduce after an idle period waits RDMA_AR_MAX_SLEEP_US.
    const char* e1 = getenv("RDMA_AR_SPIN_US");
    const char* e2 = getenv("RDMA_AR_MAX_SLEEP_US");
    const auto spin_window = std::chrono::microseconds(e1 ? atol(e1) : 20000);
    const long max_sleep_us = e2 ? atol(e2) : 1000;
    auto last_active = std::chrono::steady_clock::now();
    long backoff_us = 8;
    unsigned idle_iters = 0;
    uint64_t last_ping = 0;
    while (!stop_.load(std::memory_order_relaxed)) {
      bool idle = true;
      const uint64_t pg = __atomic_load_n(&ctrl_->ping, __ATOMIC_ACQUIRE);
      if (pg != last_ping) {
        __atomic_store_n(&ctrl_->pong, pg, __ATOMIC_RELEASE);
        last_ping = pg;
        idle = false;
      }
      const uint64_t s1 = __atomic_load_n(&ctrl_->send_seq, __ATOMIC_ACQUIRE);
      if (s1 != last1) {
        idle = false;
        timespec ts0;
        clock_gettime(CLOCK_REALTIME, &ts0);
        const int64_t t_seen = (int64_t)ts0.tv_sec * 1000000000LL + ts0.tv_nsec;
        const int64_t t_pub = (int64_t)__atomic_load_n(&ctrl_->pub_time, __ATOMIC_ACQUIRE);
        const uint64_t nbytes = __atomic_load_n(&ctrl_->send_bytes, __ATOMIC_ACQUIRE);
        const uint64_t mode = __atomic_load_n(&ctrl_->mode, __ATOMIC_ACQUIRE);
        const uint64_t cv = __atomic_load_n(&ctrl_->chunk_vecs, __ATOMIC_ACQUIRE);
        const uint64_t nvec = nbytes / 16;
        const int slot = (int)(s1 % kSlots);
        const uint64_t src0 = base + send_off_ + (uint64_t)slot * max_bytes_;
        for (int i = 1; i < world_; ++i) {
          const int q = (rank_ + i) % world_;
          Peer& pr = peers_[q];
          uint64_t src = src0, len = nbytes;
          if (mode == kTwoShot) {  // peer q gets chunk q of my input
            const uint64_t b = (uint64_t)q * cv;
            len = b < nvec ? std::min(cv, nvec - b) * 16 : 0;
            src = src0 + b * 16;
          }
          post(pr, src, (uint32_t)len, pr.remote_base + recv_off_ + ((uint64_t)slot * world_ + rank_) * max_bytes_,
               pr.remote_base + flags_off_ + ((uint64_t)slot * world_ + rank_) * 8, s1);
        }
        __atomic_store_n(&ctrl_->posted_seq, s1, __ATOMIC_RELEASE);
        timespec ts1;
        clock_gettime(CLOCK_REALTIME, &ts1);
        const int64_t t_done = (int64_t)ts1.tv_sec * 1000000000LL + ts1.tv_nsec;
        px_n_.fetch_add(1, std::memory_order_relaxed);
        px_handoff_.fetch_add(t_seen - t_pub, std::memory_order_relaxed);
        px_post_.fetch_add(t_done - t_seen, std::memory_order_relaxed);
        if (t_done - t_seen > px_post_max_.load(std::memory_order_relaxed))
          px_post_max_.store(t_done - t_seen, std::memory_order_relaxed);
        last1 = s1;
      }
      const uint64_t s2 = __atomic_load_n(&ctrl_->send2_seq, __ATOMIC_ACQUIRE);
      if (s2 != last2) {
        idle = false;
        const uint64_t len = __atomic_load_n(&ctrl_->send2_bytes, __ATOMIC_ACQUIRE);
        const int slot = (int)(s2 % kSlots);
        const uint64_t src = base + send2_off_ + (uint64_t)slot * max_chunk_;
        for (int i = 1; i < world_; ++i) {
          const int q = (rank_ + i) % world_;
          Peer& pr = peers_[q];
          post(pr, src, (uint32_t)len, pr.remote_base + recv2_off_ + ((uint64_t)slot * world_ + rank_) * max_chunk_,
               pr.remote_base + flags2_off_ + ((uint64_t)slot * world_ + rank_) * 8, s2);
        }
        __atomic_store_n(&ctrl_->posted2_seq, s2, __ATOMIC_RELEASE);
        last2 = s2;
      }
      if (idle) {
        for (auto& h : hcas_) {
          int n = ibv_poll_cq(h.cq, 32, wc);
          for (int i = 0; i < n; ++i)
            if (wc[i].status != IBV_WC_SUCCESS) {
              fprintf(stderr, "rdma_ar rank %d: completion error %s (qp %u)\n", rank_,
                      ibv_wc_status_str(wc[i].status), wc[i].qp_num);
              proxy_error_.store(true);
            }
        }
        if ((++idle_iters & 255) == 0 && std::chrono::steady_clock::now() - last_active > spin_window) {
          std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
          backoff_us = std::min(backoff_us * 2, max_sleep_us);
        }
      } else {
        idle_iters = 0;
        backoff_us = 8;
        last_active = std::chrono::steady_clock::now();
      }
    }
  }

  int rank_, world_;
  int64_t max_bytes_, max_reduce_bytes_ = 0;
  int gid_index_, ib_port_, proxy_core_;
  long long timeout_ns_;
  int device_;
  size_t max_chunk_ = 0;
  void* arena_ = nullptr;
  char* arena_dev_ = nullptr;
  size_t arena_bytes_ = 0, flags_off_ = 0, flags2_off_ = 0, send_off_ = 0, send2_off_ = 0, recv_off_ = 0,
         recv2_off_ = 0;
  Ctrl* ctrl_ = nullptr;
  DevState* st_ = nullptr;
  std::vector<Hca> hcas_;
  std::vector<Peer> peers_;
  std::thread proxy_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> proxy_error_{false};
  std::atomic<int64_t> px_n_{0}, px_handoff_{0}, px_post_{0}, px_post_max_{0};
  bool arena_registered_ = false;
  int64_t thp_bytes_ = 0;
  int timing_ = getenv("RDMA_AR_TIMING") ? atoi(getenv("RDMA_AR_TIMING")) : 0;
  long long vec_per_cta_ = getenv("RDMA_AR_VEC_PER_CTA") ? std::max(64LL, atoll(getenv("RDMA_AR_VEC_PER_CTA"))) : 512;

 public:
  // [n_kernel, avg stage, avg pub->flags, avg flags->end, avg total, n_proxy, avg handoff, avg post, max post]
  // all in ns; GPU globaltimer vs CPU CLOCK_REALTIME for the hand-off (aligned on Linux drivers).
  std::vector<double> stats(bool reset) {
    DevState h{};
    CHECK_CUDA(cudaMemcpy(&h, st_, sizeof(DevState), cudaMemcpyDeviceToHost));
    const double n = h.n_timed ? (double)h.n_timed : 1.0;
    const double pn_raw = (double)px_n_.load();
    const double pn = pn_raw > 0 ? pn_raw : 1.0;
    std::vector<double> v{(double)h.n_timed, h.sum_stage / n, h.sum_net / n, h.sum_tail / n, h.sum_total / n,
                          pn_raw, px_handoff_.load() / pn, px_post_.load() / pn, (double)px_post_max_.load()};
    if (reset) {
      h.sum_stage = h.sum_net = h.sum_tail = h.sum_total = 0;
      h.n_timed = 0;
      CHECK_CUDA(cudaMemcpy(st_, &h, sizeof(DevState), cudaMemcpyHostToDevice));
      px_n_ = 0;
      px_handoff_ = 0;
      px_post_ = 0;
      px_post_max_ = 0;
    }
    return v;
  }
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<RdmaAR>(m, "RdmaAR")
      .def(py::init<int, int, int64_t, int64_t, std::vector<std::string>, int, int, int, int64_t, int>(),
           py::arg("rank"), py::arg("world"), py::arg("max_bytes"), py::arg("max_reduce_bytes"), py::arg("hcas"),
           py::arg("gid_index"), py::arg("ib_port"), py::arg("proxy_core"), py::arg("timeout_ms"), py::arg("device"))
      .def("local_info", &RdmaAR::local_info)
      .def("connect", &RdmaAR::connect)
      .def("all_reduce", &RdmaAR::all_reduce, py::arg("x"), py::arg("two_shot") = false)
      .def("all_gather_into", &RdmaAR::all_gather_into)
      .def("stats", &RdmaAR::stats, py::arg("reset") = false)
      .def("arena_info", &RdmaAR::arena_info)
      .def("pingpong", &RdmaAR::pingpong, py::arg("n") = 1000)
      .def("max_bytes", &RdmaAR::max_bytes)
      .def("max_reduce_bytes", &RdmaAR::max_reduce_bytes);
}
