// skinny_gemm.cu: y[M, N] = x[M, K] @ W[N, K]^T for decode-sized M (1..8), bf16 in/out, fp32 accumulate.
//
// Decode GEMMs are pure weight streaming. cuBLAS picks a 16x16 tensor-core tile for M=4 on GB10 and
// reaches ~180 GB/s on the o_proj shape (N=6144, K=2048); a GEMV-style kernel that keeps x in shared
// memory and streams W with 16-byte evict-first loads gets close to the ~270 GB/s DRAM ceiling.
//
// Layout: each warp owns ROWS consecutive output rows and walks K with 16-byte vectors (8 bf16), lanes
// interleaved, UNROLL vectors in flight per row; partial sums are reduced with warp shuffles. For narrow
// N (e.g. the 384-row MoE router) K is split across SPLIT warps of a block and combined in shared memory.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ float dot8(const uint4& a, const uint4& b) {
  const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&a);
  const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&b);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 fa = __bfloat1622float2(ha[i]);
    float2 fb = __bfloat1622float2(hb[i]);
    s = fmaf(fa.x, fb.x, s);
    s = fmaf(fa.y, fb.y, s);
  }
  return s;
}

// grid.x = ceil(N / (ROWS * WARPS / SPLIT)); block = WARPS * 32 threads. Warp w handles rows
// row0 + (w / SPLIT) * ROWS .. +ROWS over the K slice (w % SPLIT).
template <int M, int ROWS, int UNROLL, int WARPS, int SPLIT>
__global__ void __launch_bounds__(WARPS * 32) skinny_kernel(const __nv_bfloat16* __restrict__ x,
                                                             const __nv_bfloat16* __restrict__ w,
                                                             __nv_bfloat16* __restrict__ y, int N, int K) {
  extern __shared__ uint4 smem[];
  uint4* xs = smem;  // M * K/8 vectors
  const int kv = K / 8;
  for (int i = threadIdx.x; i < M * kv; i += blockDim.x) xs[i] = reinterpret_cast<const uint4*>(x)[i];
  __syncthreads();

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int group = warp / SPLIT, part = warp % SPLIT;
  const int row0 = (blockIdx.x * (WARPS / SPLIT) + group) * ROWS;
  const int kv_per = (kv + SPLIT - 1) / SPLIT;
  const int kb = part * kv_per, ke = min(kv, kb + kv_per);

  float acc[ROWS][M];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;

  if (row0 < N) {
    const uint4* wr[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) wr[r] = reinterpret_cast<const uint4*>(w + (size_t)min(row0 + r, N - 1) * K);
    int k = kb + lane;
    for (; k + 32 * (UNROLL - 1) < ke; k += 32 * UNROLL) {
      uint4 wv[UNROLL][ROWS];
#pragma unroll
      for (int u = 0; u < UNROLL; ++u)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) wv[u][r] = ld_stream(wr[r] + k + 32 * u);
#pragma unroll
      for (int u = 0; u < UNROLL; ++u)
#pragma unroll
        for (int m = 0; m < M; ++m) {
          const uint4 xv = xs[m * kv + k + 32 * u];
#pragma unroll
          for (int r = 0; r < ROWS; ++r) acc[r][m] += dot8(wv[u][r], xv);
        }
    }
    for (; k < ke; k += 32) {
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const uint4 wv = ld_stream(wr[r] + k);
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] += dot8(wv, xs[m * kv + k]);
      }
    }
  }

#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = acc[r][m];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      acc[r][m] = v;
    }

  if constexpr (SPLIT == 1) {
    if (lane == 0 && row0 < N) {
#pragma unroll
      for (int r = 0; r < ROWS; ++r)
        if (row0 + r < N)
#pragma unroll
          for (int m = 0; m < M; ++m) y[(size_t)m * N + row0 + r] = __float2bfloat16(acc[r][m]);
    }
  } else {
    // combine the SPLIT partial sums of each row group in a fixed order (deterministic)
    __shared__ float part_sums[WARPS][ROWS][M];
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) part_sums[warp][r][m] = acc[r][m];
    }
    __syncthreads();
    if (part == 0 && lane == 0 && row0 < N) {
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        if (row0 + r >= N) continue;
#pragma unroll
        for (int m = 0; m < M; ++m) {
          float s = 0.f;
          for (int q = 0; q < SPLIT; ++q) s += part_sums[group * SPLIT + q][r][m];
          y[(size_t)m * N + row0 + r] = __float2bfloat16(s);
        }
      }
    }
  }
}

// FP8 (e4m3) weights [N, K] with 128x128 block scales s[ceil(N/128), ceil(K/128)] (fp32, multiply to
// dequantize), bf16 activations: y = x @ (w * s)^T, fp32 accumulation. 16 fp8 per 16-byte load, all inside
// one 128-wide K block, so each load's dot product is scaled once.
__device__ __forceinline__ float dot16_fp8(const uint4& w, const uint4& xa, const uint4& xb) {
  const __nv_fp8x2_storage_t* wp = reinterpret_cast<const __nv_fp8x2_storage_t*>(&w);
  const __nv_bfloat162* xpa = reinterpret_cast<const __nv_bfloat162*>(&xa);
  const __nv_bfloat162* xpb = reinterpret_cast<const __nv_bfloat162*>(&xb);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(wp[i], __NV_E4M3);
    float2 wf = __half22float2(*reinterpret_cast<__half2*>(&hr));
    float2 xf = __bfloat1622float2(i < 4 ? xpa[i] : xpb[i - 4]);
    s = fmaf(wf.x, xf.x, s);
    s = fmaf(wf.y, xf.y, s);
  }
  return s;
}

template <int M, int ROWS, int UNROLL, int WARPS, int SPLIT>
__global__ void __launch_bounds__(WARPS * 32) skinny_fp8_kernel(const __nv_bfloat16* __restrict__ x,
                                                                 const uint8_t* __restrict__ w,
                                                                 const float* __restrict__ sc,
                                                                 __nv_bfloat16* __restrict__ y, int N, int K,
                                                                 int sc_stride) {
  extern __shared__ uint4 smem[];
  uint4* xs = smem;  // M * K/8 vectors of 8 bf16
  const int kv8 = K / 8;
  for (int i = threadIdx.x; i < M * kv8; i += blockDim.x) xs[i] = reinterpret_cast<const uint4*>(x)[i];
  __syncthreads();

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int group = warp / SPLIT, part = warp % SPLIT;
  const int row0 = (blockIdx.x * (WARPS / SPLIT) + group) * ROWS;
  const int kv = K / 16;  // 16-byte weight vectors per row
  const int kv_per = (kv + SPLIT - 1) / SPLIT;
  const int kb = part * kv_per, ke = min(kv, kb + kv_per);

  float acc[ROWS][M];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;

  if (row0 < N) {
    const uint4* wr[ROWS];
    const float* sr[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const int row = min(row0 + r, N - 1);
      wr[r] = reinterpret_cast<const uint4*>(w + (size_t)row * K);
      sr[r] = sc + (size_t)(row / 128) * sc_stride;
    }
    int k = kb + lane;
    for (; k + 32 * (UNROLL - 1) < ke; k += 32 * UNROLL) {
      uint4 wv[UNROLL][ROWS];
#pragma unroll
      for (int u = 0; u < UNROLL; ++u)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) wv[u][r] = ld_stream(wr[r] + k + 32 * u);
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        const int kk = k + 32 * u;
        float s_r[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) s_r[r] = __ldg(sr[r] + (kk * 16) / 128);
#pragma unroll
        for (int m = 0; m < M; ++m) {
          const uint4 xa = xs[m * kv8 + 2 * kk], xb = xs[m * kv8 + 2 * kk + 1];
#pragma unroll
          for (int r = 0; r < ROWS; ++r) acc[r][m] = fmaf(s_r[r], dot16_fp8(wv[u][r], xa, xb), acc[r][m]);
        }
      }
    }
    for (; k < ke; k += 32) {
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const uint4 wv = ld_stream(wr[r] + k);
        const float s_r = __ldg(sr[r] + (k * 16) / 128);
#pragma unroll
        for (int m = 0; m < M; ++m)
          acc[r][m] = fmaf(s_r, dot16_fp8(wv, xs[m * kv8 + 2 * k], xs[m * kv8 + 2 * k + 1]), acc[r][m]);
      }
    }
  }

#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = acc[r][m];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      acc[r][m] = v;
    }

  if constexpr (SPLIT == 1) {
    if (lane == 0 && row0 < N) {
#pragma unroll
      for (int r = 0; r < ROWS; ++r)
        if (row0 + r < N)
#pragma unroll
          for (int m = 0; m < M; ++m) y[(size_t)m * N + row0 + r] = __float2bfloat16(acc[r][m]);
    }
  } else {
    __shared__ float part_sums[WARPS][ROWS][M];
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) part_sums[warp][r][m] = acc[r][m];
    }
    __syncthreads();
    if (part == 0 && lane == 0 && row0 < N) {
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        if (row0 + r >= N) continue;
#pragma unroll
        for (int m = 0; m < M; ++m) {
          float t = 0.f;
          for (int q = 0; q < SPLIT; ++q) t += part_sums[group * SPLIT + q][r][m];
          y[(size_t)m * N + row0 + r] = __float2bfloat16(t);
        }
      }
    }
  }
}

int max_smem() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    v -= 1024;  // headroom for the static part_sums buffer
  }
  return v;
}

template <int M, int ROWS, int UNROLL, int WARPS, int SPLIT>
void launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y, cudaStream_t stream) {
  const int N = (int)w.size(0), K = (int)w.size(1);
  const size_t smem = (size_t)M * K * sizeof(__nv_bfloat16);
  auto kern = skinny_kernel<M, ROWS, UNROLL, WARPS, SPLIT>;
  static bool attr_set = false;  // per instantiation
  if (!attr_set) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem()) == cudaSuccess,
                "skinny_gemm: cannot raise shared memory limit");
    attr_set = true;
  }
  const int rows_per_block = ROWS * (WARPS / SPLIT);
  const int grid = (N + rows_per_block - 1) / rows_per_block;
  kern<<<grid, WARPS * 32, smem, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                                           reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
                                           reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), N, K);
}

// variant: 0 = wide N (ROWS=2, no split), 1 = narrow N (ROWS=1, split K over 4 warps)
template <int M>
void dispatch_variant(int variant, const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y, cudaStream_t s) {
  if (variant == 1)
    launch<M, 1, 4, 8, 4>(x, w, y, s);
  else if (variant == 2)
    launch<M, 1, 4, 8, 1>(x, w, y, s);
  else
    launch<M, 2, 4, 8, 1>(x, w, y, s);
}

template <int M, int ROWS, int UNROLL, int WARPS, int SPLIT>
void launch_fp8(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s, torch::Tensor& y,
                cudaStream_t stream) {
  const int N = (int)w.size(0), K = (int)w.size(1);
  const size_t smem = (size_t)M * K * sizeof(__nv_bfloat16);
  auto kern = skinny_fp8_kernel<M, ROWS, UNROLL, WARPS, SPLIT>;
  static bool attr_set = false;
  if (!attr_set) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem()) == cudaSuccess,
                "skinny_fp8: cannot raise shared memory limit");
    attr_set = true;
  }
  const int rows_per_block = ROWS * (WARPS / SPLIT);
  const int grid = (N + rows_per_block - 1) / rows_per_block;
  kern<<<grid, WARPS * 32, smem, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                                           reinterpret_cast<const uint8_t*>(w.data_ptr()),
                                           s.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), N, K,
                                           (int)s.size(1));
}

template <int M>
void dispatch_fp8(int variant, const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s,
                  torch::Tensor& y, cudaStream_t st) {
  if (variant == 1)
    launch_fp8<M, 1, 4, 8, 4>(x, w, s, y, st);
  else if (variant == 2)
    launch_fp8<M, 1, 4, 8, 1>(x, w, s, y, st);
  else
    launch_fp8<M, 2, 4, 8, 1>(x, w, s, y, st);
}

// y[M, N] = x[M, K] (bf16) @ (w[N, K] fp8 e4m3 * s[ceil(N/128), ceil(K/128)])^T
torch::Tensor skinny_gemm_fp8(torch::Tensor x, torch::Tensor w, torch::Tensor s, int64_t variant) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && s.is_cuda() && x.scalar_type() == at::kBFloat16 &&
                  w.element_size() == 1 && s.scalar_type() == at::kFloat,
              "skinny_gemm_fp8: bf16 x, fp8 w, fp32 block scales");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && s.dim() == 2 && x.size(1) == w.size(1), "skinny_gemm_fp8: shapes");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && s.is_contiguous() && w.size(1) % 128 == 0,
              "skinny_gemm_fp8: contiguous, K % 128 == 0");
  TORCH_CHECK(s.size(0) == (w.size(0) + 127) / 128 && s.size(1) == w.size(1) / 128, "skinny_gemm_fp8: scale shape");
  const int M = (int)x.size(0);
  TORCH_CHECK(M >= 1 && M <= 8 && (int64_t)M * w.size(1) * 2 <= max_smem(), "skinny_gemm_fp8: M");
  c10::cuda::CUDAGuard guard(x.device());
  auto y = torch::empty({M, w.size(0)}, x.options());
  auto st = at::cuda::getCurrentCUDAStream();
  switch (M) {
    case 1: dispatch_fp8<1>(variant, x, w, s, y, st); break;
    case 2: dispatch_fp8<2>(variant, x, w, s, y, st); break;
    case 3: dispatch_fp8<3>(variant, x, w, s, y, st); break;
    case 4: dispatch_fp8<4>(variant, x, w, s, y, st); break;
    case 5: dispatch_fp8<5>(variant, x, w, s, y, st); break;
    case 6: dispatch_fp8<6>(variant, x, w, s, y, st); break;
    case 7: dispatch_fp8<7>(variant, x, w, s, y, st); break;
    default: dispatch_fp8<8>(variant, x, w, s, y, st); break;
  }
  return y;
}

bool fits(int64_t m, int64_t k) { return m >= 1 && m <= 8 && k % 8 == 0 && m * k * 2 <= max_smem(); }

torch::Tensor skinny_gemm(torch::Tensor x, torch::Tensor w, int64_t variant) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16,
              "skinny_gemm: bf16 CUDA tensors");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "skinny_gemm: shape mismatch");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && w.size(1) % 8 == 0, "skinny_gemm: contiguous, K % 8 == 0");
  const int M = (int)x.size(0);
  TORCH_CHECK(M >= 1 && M <= 8, "skinny_gemm: M must be 1..8");
  TORCH_CHECK((int64_t)M * w.size(1) * 2 <= max_smem(), "skinny_gemm: x too large for shared memory");
  c10::cuda::CUDAGuard guard(x.device());
  auto y = torch::empty({M, w.size(0)}, x.options());
  auto s = at::cuda::getCurrentCUDAStream();
  switch (M) {
    case 1: dispatch_variant<1>(variant, x, w, y, s); break;
    case 2: dispatch_variant<2>(variant, x, w, y, s); break;
    case 3: dispatch_variant<3>(variant, x, w, y, s); break;
    case 4: dispatch_variant<4>(variant, x, w, y, s); break;
    case 5: dispatch_variant<5>(variant, x, w, y, s); break;
    case 6: dispatch_variant<6>(variant, x, w, y, s); break;
    case 7: dispatch_variant<7>(variant, x, w, y, s); break;
    default: dispatch_variant<8>(variant, x, w, y, s); break;
  }
  return y;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fits", &fits, "whether skinny_gemm handles [m, k] activations");
  m.def("skinny_gemm_fp8", &skinny_gemm_fp8, "y = x @ (w_fp8 * block_scale).T for M <= 8", py::arg("x"),
        py::arg("w"), py::arg("s"), py::arg("variant") = 0);
  m.def("skinny_gemm", &skinny_gemm, "y = x @ w.T for M <= 8 (bf16)", py::arg("x"), py::arg("w"),
        py::arg("variant") = 0);
}
