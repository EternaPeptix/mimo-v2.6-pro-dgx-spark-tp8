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
64K prefix goes from 362 to 177 ms per layer (16 → 33 TFLOPS), sliding-window layers unchanged. (The
full-attention layers now prefill through FlashInfer, §12; these tiles still serve the sliding-window layers.)

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

---

# Prefill

An 8K-token prefill chunk (the scheduler's chunk size) costs ~3.7 s. Before the changes below a ~31K-token
prompt took 19.6 s to its first token; now 14.3 s (see [RESULTS.md](RESULTS.md#prefill)). The largest
remaining item is the network: each chunk does 150 TP all-reduces of 96 MB, ~28% of its time.

## 11. Prefill MoE GEMMs on Marlin's MXFP4 layout (patches 11, 12)

**Problem.** Marlin is built for decode. For an 8K-token chunk (65,536 token×expert rows over 384 experts) it
streams, and re-decodes, every expert's FP4 weights once per 64-token block — about 3 times per expert — and
re-reads the activations for every 256-column tile: an estimated ~3.7 GB of DRAM traffic for the gate_up GEMM
of one layer, 15.2 ms (26.6 TFLOPS); down 8.5 ms (24 TFLOPS). Marlin's GEMMs alone were 37% of a chunk.
The experts are 66 GB per rank, so a second copy in another layout is not an option: a prefill kernel has to
read Marlin's.

**Fix.** `kernels/ops/moe/mxfp4_marlin_prefill.py` (Triton), used from `fused_marlin_moe` when
`M·top_k ≥ 20·E` (≈1K tokens here); decode-sized batches keep Marlin.

- Grouped GEMM over `moe_align_block_size`'s 64-token expert blocks, BF16 tensor-core MMAs with FP32
  accumulation, gathered activation rows, CTAs ordered so the column tiles of one token block run together
  (they share its activations through the 24 MB L2).
- Marlin packs each 16×64 (k×n) tile into 128 int32 words, and which (k, n) a nibble holds is a pure bit
  permutation of (word, nibble) — so the tile is loaded as contiguous words and unscrambled with a
  reshape/permute. Its E8M0 scales are stored 8×8-transposed with a pair swap; the kernel reads them in place.
- FP4 → BF16 in inline PTX, Marlin's own trick: `(q & 0x80008000) | ((q & 0x70007000) >> 6)` turns two
  nibbles into a bf16x2 equal to value·2⁻¹²⁶ (0.5 becomes a bf16 subnormal), then two `mul.bf16x2` apply 2¹²⁶
  and the scale — exact, ~3 instructions per weight. A first version decoding with per-element integer ops
  reached 36 TFLOPS; Nsight Compute showed it issue-bound with 237 registers per thread (2 warps per
  scheduler). The PTX decode cut that to 124 registers.

| Tokens | Marlin w13 + w2 | Triton w13 + w2 |
|---|---|---|
| 1,024 | 3.79 + 2.04 = 5.83 ms | 3.39 + 2.23 = 5.63 ms |
| 2,048 | 5.63 + 2.78 = 8.42 ms | 3.82 + 2.70 = 6.52 ms |
| 4,096 | 8.83 + 4.91 = 13.75 ms | 6.12 + 4.33 = 10.44 ms |
| 8,192 | 15.13 + 8.51 = 23.64 ms | **9.22 + 7.47 = 16.70 ms** |

(one rank, 384 experts, random routing; gate_up at 8K: 26.6 → 45 TFLOPS, against 86–90 TFLOPS for a dense
cuBLAS BF16 GEMM on GB10.) Max relative error against an FP32 reference: gate_up 3.0e-3 for both, down
2.0e-3 (new) vs 4.6e-3 (Marlin). `kernels/tests/test_mxfp4_prefill_moe.py` reproduces the table;
`SGLANG_MXFP4_PREFILL_MOE=0` turns the kernel off.

## 12. Full-attention prefill through FlashInfer (patch 05)

**Problem.** The 10 full-attention layers prefill through Triton's extend kernel, which reaches 51 TFLOPS on an
empty prefix but falls to 36 TFLOPS at a 131K prefix (Nsight Compute: one 255-register CTA per SM, stalled on
loads). Attention is quadratic, so at long context it is most of a prefill.

**Fix.** For plain causal EXTEND batches the Triton backend runs FlashInfer's FA2 paged prefill kernel (already
in the image, JIT-compiled for head dims 192/128) on the same KV pool. Its page table comes from the same
token-index stream the Triton kernels read: with page size 64, every 64th token index divided by 64 is the
page id. It is planned once per batch; sliding-window/sink layers, target-verify and drafter passes are
untouched.

| Prefix (chunk of 8K tokens) | Triton extend | FlashInfer FA2 |
|---|---|---|
| 0 | 6.7 ms (51 TFLOPS) | 5.7 ms (60) |
| 8K | 21.6 ms (48) | 16.2 ms (64) |
| 32K | 82.1 ms (38) | 47.6 ms (65) |
| 131K | 315 ms (36) | **173 ms (65)** |

(per layer and rank; both kernels have the same error against an FP32 reference, 1.8e-3 / 3.4e-3.)
`SGLANG_FI_FULL_PREFILL=1` enables it (default in the launcher). `SGLANG_FI_FULL_PREFILL_CHECK=N` also runs
the Triton kernel for the first N calls and logs the difference (2–7e-3 relative in the server: BF16
rounding). A `/tmp/fi_prefill_off` file in a rank's container switches it off at runtime; there are no
collectives involved, which makes A/B runs easy (`FI_AB=1 bench/prefill-ttft.sh`). Long-context retrieval
(`bench/needle.py`, 5 facts at 5 depths) recalled 15/15 at 53K tokens and 5/5 at 120K.
`kernels/tests/bench_prefill_attention.py` reproduces the table.

## 13. Sliding-window prefill: start at the first visible key tile (patch 01)

For the 60 sliding-window layers the extend kernel walked every key tile of the chunk for each 128-query block
and masked away all but the last ~128 keys — for the last block of an 8K chunk, 128 key tiles visited for 4
useful ones. The loop
now starts at the first tile the block can see (plain causal case only): **4.06 → 1.19 ms** per layer per
8K chunk (−0.17 s per chunk), bit-identical outputs. `kernels/tests/test_swa_prefill.py` checks it against an
FP32 reference including the sink term.

## 14. No zero-fill of the MoE scratch buffer (patch 11)

`fused_marlin_moe` zeroed its `[M·top_k, max(2N, K)]` scratch (805 MB per layer for an 8K chunk) because Marlin
skips masked rows under expert parallelism. Without EP every row is written by both GEMMs, so it is now
`torch.empty` (`SGLANG_MARLIN_MOE_ZERO_FILL=1` restores the fill): 33.1 → 28.8 ms per MoE layer with Marlin at
8K tokens; outputs identical, checked with a garbage-filled allocator.

## 15. NCCL for prefill's large all-reduces (launcher)

Prefill all-reduces are 96 MB (8K tokens × 6144 × BF16), far above what the RDMA kernel is for, so they go to
NCCL's ring. `NCCL_PROTO=Simple` with exactly 8 channels: **7.9 → 6.8 ms** per all-reduce (12/16/24 channels:
6.9–7.0 ms; tree: 52 ms; more QPs or larger buffers: no change). That is ~25 GB/s of ring bus bandwidth per
rank; we found no setting that goes faster, so the remaining lever is overlapping it with compute (see below).

## What we tried and dropped

- **NCCL tuning** (tree vs ring, LL/LL128/Simple, fewer channels; `bench/arbench.py`, one process per node
  with RANK/WORLD_SIZE/MASTER_ADDR set): nothing below ~130 µs for 12 KB in a graph.
- **Marlin W8A16 for the FP8 linears** (`SGLANG_FORCE_FP8_MARLIN=1`): ~11% faster than Triton at decode but
  very slow at prefill sizes (1.8 s per 8K-token chunk for the qkv projections alone); the skinny FP8 kernel
  is faster still at decode, so Marlin is off.
- **A user-space "balloon"** that absorbs small free fragments so kcompactd's trigger score stays low: works
  while large free blocks exist, fails on nodes where none are left. The sysctl is the real fix.
- **Custom MXFP4 MoE kernel for decode**: not done. Marlin streams experts at ~228–240 GB/s on GB10 (measured
  10.4 µs per extra distinct expert), so the remaining gain is ~2.6 ms per step. (For prefill see §11.)
- **Two-stream half-batch prefill** to hide the 96 MB all-reduces behind compute: each chunk ran as two halves
  on two CUDA streams (the second half's all-reduces on a second NCCL communicator; the second half released
  after the first half's attention of the same layer). 14.6–14.7 s vs 14.3 s for a 31K-token prompt. In the
  traces only ~0.6 of ~2.7 s of NCCL time overlapped compute: when both streams compute they share the SMs and
  finish together, so their all-reduces coincide (each half-size all-reduce took as long as a full one), and
  half-size MoE GEMMs are ~25% less efficient. Forcing the halves' MoE phases to alternate, with 16K-token
  chunks to keep the GEMMs large, is the next thing to try.
- **Tile tuning of Triton's block-FP8 GEMM** (the qkv projection at prefill, 4.4 ms per layer for an 8K chunk): no
  configuration beats the default (~79 TFLOPS) while staying bit-identical. GB10 does ~165 TFLOPS FP8 in
  cuBLAS, but this stack has no native 128×128 block-scaled FP8 GEMM for sm_121 (none in sgl-kernel; PyTorch's
  blockwise `scaled_mm` is SM90-only).
- **SiLU-and-mul fused into the gate_up epilogue** (one CTA computes matching gate and up columns): 11.0 ms vs
  9.1 + 0.4 ms unfused. **Down projection keeping its activation tile in registers** across output tiles:
  8.5–10.9 ms vs 7.5 ms.
