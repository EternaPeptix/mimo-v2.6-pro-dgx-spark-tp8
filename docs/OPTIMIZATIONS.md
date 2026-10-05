# What we changed and why

Numbers are per GPU (one TP rank) unless stated otherwise. "Step" means one decode iteration of
multi-layer EAGLE: three MTP draft steps plus one target verify of 4 tokens. Each step emits
`accept_len` tokens (≈2.6–3.5 depending on content), so `tok/s = accept_len / step time`.

Model facts that drive everything below: 70 layers (10 full attention + 60 sliding-window attention with a
128-token window and attention sinks), hidden 6144, 128 query heads / 8 KV heads (head dim 192 for Q/K,
128 for V), 384 experts top-8 with 2048-wide experts, a dense first layer, vocabulary 152,576. At TP8 each
GPU holds 16 query heads that share **one** KV head.

---

## 1. Getting it to start (Dockerfile, patch 05)

| Symptom | Cause | Fix |
|---|---|---|
| `No processor registered for architecture: ['MiMoV2ForCausalLM']` | the MiMo-V2 module imports its audio processor, which needs `torchcodec` | `pip install torchcodec==0.17.0` in the image |
| `AssertionError: Hidden size mismatch` in Triton fused MoE | auto-selected Triton MoE does not understand the MXFP4 expert layout | `--moe-runner-backend marlin` |
| `AttributeError: 'NoneType' object has no attribute 'type'` while capturing the draft-extend CUDA graph | Triton backend never fills `window_kv_indices` for draft-extend graphs, but the MTP layers are sliding-window | patch 05 fills the window buffers in `_update_draft_extend_buffers` and passes them to the graph metadata |

## 2. RDMA all-reduce instead of NCCL (`kernels/rdma_ar`, patch 04)

**Problem.** Every layer needs two TP all-reduces (after attention, after the MoE): 150 per step, 12 KB
(draft) or 48 KB (verify) each. NCCL picks its ring/LL algorithm: 14 sequential network hops for 8 ranks,
each hop through NCCL's CPU proxy, with GPUDirect RDMA unavailable on GB10. Inside a CUDA graph that is
~130–190 µs per all-reduce; in the first profile, NCCL kernels were **50% of the decode step** (~48 ms).
NCCL's other algorithms are no better (tree/LL: 137–445 µs).

**Fix.** A small ibverbs extension that keeps everything in one CUDA kernel launch per collective:

- **one-shot** (≤ 32 KB): every rank RDMA-writes its whole input into a slot on each peer, then sums
  the 8 contributions locally.
- **two-shot** (> 32 KB): reduce-scatter (chunk *q* of my input → rank *q*), rank *q* sums its chunk and
  broadcasts it. Two hops, 4× fewer bytes on the wire.
- **all-gather** (logits): one-shot copy.
- The GPU stages data in host-pinned memory and bumps a sequence number; a **CPU proxy thread** posts
  the RDMA writes (payload + 8-byte flag on the same RC QP, so the flag lands last); the receiving GPU
  spins on the flags. GPU→proxy hand-off is ~0.8 µs; posting to 7 peers ~1.3 µs.
- **CUDA-graph safe**: the sequence number lives in device memory and is advanced by the kernel itself, so
  the kernel arguments never change between replays.
- **Bit-identical on every rank**: each output element is summed by one fixed rule (rank order; in
  two-shot, by the chunk's owner), so all TP ranks stay in lock-step.
- The buffer arena is backed by **2 MB transparent huge pages** (`mmap` + `MADV_HUGEPAGE` +
  `cudaHostRegister`). With `cudaHostAlloc` on a fragmented node the GPU's accesses to it went through 4 KB
  pages and stage/reduce took 3× longer on that rank — which every other rank then waited for.
- Pairs of ranks use one of two RoCE devices (one per PCIe x4 link); the proxy backs off to sleeping after
  20 ms of silence so an idle server does not spin a core.

| Message | one-shot | two-shot | NCCL (graph) |
|---|---|---|---|
| 12 KB (1 token) | **21 µs** | 31 µs | 120–130 µs |
| 24 KB | **26 µs** | 32 µs | 132 µs |
| 48 KB (4-token verify) | 37 µs | **34 µs** | 146–154 µs |
| 96 KB | 55 µs | **39 µs** | 151 µs |
| 192 KB | 98 µs | **49 µs** | 211–224 µs |
| logits all-gather, fp32 [4, 19072] | **146 µs** | | 870 µs |

Per stage (one-shot 12 KB): 3 µs staging, ~12 µs publish-to-flags (NIC + wire + remote DMA, 7 peers),
~5 µs local reduce. In the server the all-reduce runs at p50 ≈ 38 µs (p90 ≈ 46 µs).

## 3. Grouped-head split-KV verify attention (patches 03, 05)

**Problem.** The 4-token verify step runs through Triton's prefill "extend" kernel: one program per
(query block, head) walking the whole prefix serially. With 16 query heads sharing one KV head and only
4 queries, that is 16 programs on a 48-SM GPU, each re-reading the same KV — **6 GB/s** effective. At 62K
context decode ran at 19 tok/s (45.8 tok/s with this fix); at 1M it would be ~1.1 s of attention per step.

**Fix.** SGLang already ships a grouped-head split-KV verify kernel (`verify_mla.py`, tuned for AMD
gfx95 and gated to it). It handles any GQA layer whose local KV heads form power-of-two groups; for MiMo the
Q·K split into "nope/rope" parts is just a 128 + 64 split of the 192-dim head. Patch 05 enables it on CUDA
(`SGLANG_VERIFY_SHARED_KV`, sliding-window/sink layers still fall back), patch 03 adds a GB10 config: all 16
heads in one block (each KV tile read once), BLOCK_N 64, 4 warps, up to 96 KV splits.

| Context | extend kernel | grouped-head split-KV |
|---|---|---|
| 64K | 6.7 ms | 0.19 ms |
| 256K | 27.2 ms | 0.69 ms |
| 1M | 109 ms | **2.9 ms** (234 GB/s) |

(per layer; outputs match to < 2e-4.)

## 4. Fused sliding-window + sink verify attention (patches 02, 05)

**Problem.** The other 60 layers use a 128-token window with attention sinks; their verify step also went
through the extend kernel and cost ~28 µs each (~40 µs in the server) for a few KB of KV — latency of a
dependent chain of index → K → V loads, each a likely TLB miss into the multi-GB KV pool.

**Fix.** `swa_verify.py`: one program per (request, 4-head block) covering all draft tokens, every index,
K and V load of the window (two 64-key tiles) and of the draft block issued before any math (~3 dependent
round trips per layer), one exact softmax including the sink term. Same masking semantics as the extend
kernel (window relative to the window list, causal among draft tokens).

| | extend | swa_verify |
|---|---|---|
| 1 request, window 128 | 28 µs | **8.5 µs** |
| 4 requests | 56 µs | **9.2 µs** |

## 5. Decode-sized GEMMs (`kernels/skinny_gemm`, patches 07, 08)

For M ≤ 8 tokens every linear layer is pure weight streaming. cuBLAS picks a 16×16 tensor-core tile on GB10
(~180 GB/s for o_proj at M=1), and the Triton W8A8 block-FP8 kernel quantizes activations and reaches
~190–200 GB/s. `skinny_gemm.cu` keeps x in shared memory and streams the weights with 16-byte non-allocating
loads, warp-reducing in fp32:

- **BF16** (`SGLANG_SKINNY_GEMM`, o_proj — kept in BF16 by the checkpoint): o_proj M=1 138 → 99 µs,
  M=4 111 → 101 µs.
- **FP8 e4m3 with 128×128 block scales** (`SGLANG_SKINNY_FP8`, qkv, dense layer, MTP layers): W8A16 —
  activations stay BF16, so it is also more accurate (rel. error 2–3.5e-3 vs 2.2–2.8e-2 for W8A8). qkv M=1
  191 → 241 GB/s, down-proj 203 → 243 GB/s. Larger M (prefill) still uses Triton W8A8.

## 6. Drafter (patches 06, 10)

- **`eh_proj` sharded.** The MTP layer's 2H→H projection (151 MB BF16) was a replicated `nn.Linear` that
  every rank read in full on all three draft steps (660 µs each). Column-parallel with a gathered output:
  ~90 µs + a 3 KB all-gather. **−1.8 ms per step.**
- **FP8 copy of the draft lm_head** (`SGLANG_MTP_FP8_HEAD`). The drafter only proposes tokens; the target
  verifies every one with its own BF16 head, so drafter precision can change acceptance but never the
  output. Halving its 234 MB/rank head read saves **~1.3 ms per step**; acceptance was unchanged in our
  runs (code 3.27, math 3.45, essay 2.70).

## 7. Prefill attention tiles on sm_121 (patch 01)

The extend kernel uses 64-row tiles on sm_12x for 192-dim heads. On GB10 a 128-row tile with 2 pipeline
stages still fits in the 99 KB of shared memory and halves KV bytes per FLOP: an 8K-token chunk against a
64K prefix goes from 362 to 177 ms per layer (16 → 33 TFLOPS), sliding-window layers unchanged.

## 8. Memory hygiene

On GB10 the GPU allocates from the same LPDDR5X as the OS. If its allocations land on fragmented memory,
some ranks run 1.5–8× slower (TLB pressure) and all ranks wait for them.

- `--weight-loader-drop-cache-after-load`: every rank streams the whole 566 GB checkpoint (keeping 1/8);
  without dropping each file's page cache the load churns memory while 70 GB of weights are allocated.
- Evict model files from the page cache before launching (`scripts/dropcache.py`, no root needed).
- `SGLANG_PREFILL_POOL_GB` (patch 09): reserve the prefill activation memory right after start-up instead
  of on the first long prompt.
- **`vm.compaction_proactiveness=0`** — see [PITFALLS.md](PITFALLS.md).
- Never run other GPU jobs on a node while the server is loading.

## 9. CPU side

- `--scheduler-recv-interval 8`: with multi-node TP, rank 0 gloo-broadcasts new requests to all ranks on
  every scheduler iteration (~1 ms per step); checking every 8th iteration removes most of it.
- `--sleep-on-idle` plus the RDMA proxy's back-off: idle nodes do not spin cores (on GB10 CPU heat shares the
  SoC's budget with the GPU).

## 10. Measuring decode at long context without a long prefill (patch 05)

`SGLANG_FAKE_VERIFY_CTX_FILE=1` (inert unless a flag file exists): with `N` in `/tmp/fake_verify_ctx`
inside each rank's container, the 10 full-attention layers of each single-request verify step attend over
`N` real slots of the KV pool. That is exactly the KV traffic and kernel work of an `N`-token context; the
sliding-window and drafter layers have fixed windows and cost the same beyond ~1K tokens anyway. Outputs
are garbage while it is on, so only the step time is meaningful. `bench/longctx-decode.sh` drives it.

## What we tried and dropped

- **NCCL tuning** (tree vs ring, LL/LL128/Simple, fewer channels; `bench/arbench.py`, one process per node
  with RANK/WORLD_SIZE/MASTER_ADDR set): nothing below ~130 µs for 12 KB in a graph.
- **Marlin W8A16 for the FP8 linears** (`SGLANG_FORCE_FP8_MARLIN=1`): ~11% faster than Triton at decode but
  very slow at prefill sizes (1.8 s per 8K-token chunk for the qkv projections alone); the skinny FP8 kernel
  is faster still at decode, so Marlin is off.
- **A user-space "balloon"** that absorbs small free fragments so kcompactd's trigger score stays low: works
  while large free blocks exist, fails on nodes where none are left. The sysctl is the real fix.
- **Custom MXFP4 MoE kernel**: not done. Marlin streams experts at ~228–240 GB/s on GB10 (measured 10.4 µs per
  extra distinct expert), so the remaining gain is ~2.6 ms per step.
