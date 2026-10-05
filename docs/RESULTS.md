# Results

All numbers: one request at a time, greedy decoding (temperature 0), 512 generated tokens with
`ignore_eos`, measured from the first to the last streamed token (`bench/bench.py`). MTP speculative
decoding: 3 draft steps, top-1, 4 tokens verified per step; `accept` is SGLang's average accepted tokens per
verify step, and **step time = accept / decode tok/s**. Numbers below were taken with the image built from
this repository (`scripts/build-image.sh`) and the repository's launch and bench scripts.

## Decode, short context

`bench/run-bench.sh 3`:

| Prompt | Decode tok/s | Acceptance | Step |
|---|---|---|---|
| code | 62.93 / 61.38 / 58.40 | 3.35 / 3.26 / 3.12 | 53.2 / 53.1 / 53.5 ms |
| math | 66.82 / 63.44 / 64.32 | 3.53 / 3.35 / 3.39 | 52.8 / 52.7 / 52.7 ms |
| essay | 48.37 / 53.53 / 51.24 | 2.56 / 2.83 / 2.71 | 52.9 / 52.8 / 52.9 ms |

The step time is flat (52.7–53.5 ms); tok/s moves only with how predictable the text is for the drafter.

## Decode vs context length

`bench/longctx-decode.sh` (synthetic KV, see [OPTIMIZATIONS.md §10](OPTIMIZATIONS.md#10-measuring-decode-at-long-context-without-a-long-prefill-patch-05)):

| Context | Step time | tok/s at code acceptance (3.25) |
|---|---|---|
| short | 53.4 ms | ~61 |
| 128K | 54.4 ms | ~60 |
| 256K | 58.0 ms | ~56 |
| 512K | 64.1 ms | ~51 |
| **1M** | **77.5 ms** | **~42** |

The extra 24 ms at 1M context is the KV-cache read of the 10 full-attention layers: 10 × 640 MB per GPU per
step, i.e. ~270 GB/s — the memory bandwidth limit. (A real 1M-token request also rebuilds a 1M-entry index
list per step, ~0.1 ms, which the synthetic mode skips.)

## How close to the memory-bandwidth limit

Bytes each GPU must read per decode step (TP8):

| Item | GB |
|---|---|
| MoE experts: ~28 distinct experts per layer for 4 verified tokens × 2.51 MB × 69 layers | ~4.9 |
| qkv (FP8) + dense layer | 1.5 |
| o_proj (BF16, kept unquantized by the checkpoint) | 1.76 |
| router (BF16, replicated) + target lm_head | 0.56 |
| drafter: 3 × (MTP layer + FP8 lm_head + eh_proj shard) | ~0.7 |
| **total** | **~9.4** |

At the ~260 GB/s these kernels can sustain that is **~36 ms**; we run 53 ms, i.e. **~68%** of the
roofline. At 1M context (+6.4 GB of KV) the floor is ~61 ms and we run 77.5 ms (**~79%**).

For comparison, plain decoding without a drafter reads ~5.2 GB per GPU per token, so even a perfect kernel
stack could not exceed ~53 tok/s on 8 Sparks (273 GB/s spec) — speculative decoding is what gets us above it.

## Where the 53 ms go

Approximate split from torch-profiler traces (taken one build before the last three changes, at 56.5 ms
per step; those changes — FP8 skinny GEMM, scheduler receive interval, FP8 draft head — removed ~3.5 ms
from qkv, the draft heads and the CPU gaps):

| Part | ms per step |
|---|---|
| MoE (Marlin MXFP4, 138 kernels) | ~21 |
| o_proj (skinny BF16, 73 calls) | ~8.4 |
| TP all-reduce (150 × p50 ~38 µs) + logits all-gather | ~6.5 |
| qkv / dense / MTP linears (FP8) | ~6 |
| drafter + target lm_heads | ~2.5 |
| router GEMM | ~1.6 |
| attention (10 full + 60 SWA layers + drafter) | ~1.3 |
| norms, activations, KV stores, small kernels | ~2 |
| GPU idle between graph replays (CPU work) | ~2 |

## History

Same benchmark, as changes were added (step times when no compaction was happening):

| Build | Step | code tok/s |
|---|---|---|
| Stock SGLang v0.5.21 with only the start-up fixes, NCCL | ~80 ms | ~35 (median of all prompts, 31–43) |
| + RDMA one-shot all-reduce, page-cache drop during load | ~63 ms | ~46 (median of all prompts) |
| + two-shot + all-gather, grouped-head verify attention, skinny BF16 GEMM, prefill tiles | ~60 ms | ~54 |
| + SWA verify kernel, sharded drafter eh_proj, huge-page RDMA arena | ~56.5 ms | ~57 |
| + skinny FP8 GEMM, scheduler receive interval | ~55 ms | ~59 |
| + FP8 draft lm_head | ~53 ms | ~61 |
| + `vm.compaction_proactiveness=0` | **53 ms in every run** (was bimodal 53 / 70–77) | **~61** |

## Prefill (not optimized yet)

| Prompt | Time to first token |
|---|---|
| 62K tokens, warm caches | 83–150 s |
| 247K tokens | 881 s |

Prefill attention is bandwidth-bound at ~33 TFLOPS on GB10 even with the larger tiles, the prefill MoE uses
Marlin (built for small batches), and long prefills are the most exposed to stragglers. This is the next
thing we are working on.
