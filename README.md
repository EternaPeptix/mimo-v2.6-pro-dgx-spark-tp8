# MiMo-V2.6-Pro on 8× DGX Spark — TP8 + MTP speculative decoding, 1M context

Serving **[XiaomiMiMo/MiMo-V2.6-Pro-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD)**
(~1T-parameter MoE, MXFP4 experts / FP8 attention, 566 GB checkpoint) on **eight NVIDIA DGX Spark (GB10)**
nodes with tensor parallelism 8 over a 200 Gb/s RoCE fabric, using the model's built-in 3-layer MTP drafter,
with a **1,048,576-token context window**.

Stock SGLang does not start on this setup, and once it does, single-stream decode is held back by the
interconnect (NCCL's ring all-reduce over 8 nodes), by attention kernels that were never tuned for sm_121,
and by Linux memory compaction fighting the GPU's pinned memory. This repository contains everything needed
to reproduce our deployment: a Dockerfile (SGLang v0.5.21 + 12 patches), two custom CUDA extensions
(an RDMA all-reduce and decode-sized GEMMs), Triton prefill MoE GEMMs that read the decode kernel's MXFP4
layout, the launch scripts, the benchmarks and the measurements.

## Results (single request, greedy, 512 output tokens)

| | Stock SGLang (after startup fixes) | This repo |
|---|---|---|
| Decode step (verify of 4 tokens + 3 draft steps) | ~80 ms, bimodal | **53 ms**, stable (52.7–53.7) |
| Decode, `code` prompt (acceptance ≈ 3.25 tokens/step) | 31–43 tok/s (median 35.5, all prompts) | **~60.5 tok/s** |
| Decode, `math` prompt (acceptance ≈ 3.45) | | **~65.5 tok/s** |
| Decode, `essay` prompt (acceptance ≈ 2.65) | | **~50 tok/s** |
| Decode step at 1M-token context | ~1.1 s of attention alone (est.) | **77 ms** (≈42 tok/s at code acceptance) |
| TP all-reduce, 48 KB, inside a CUDA graph | 150–190 µs (NCCL) | **34 µs** (one-shot 12 KB: **21 µs**) |

How close is that to the hardware? A plain (non-speculative) decode of this model must read ~5.2 GB per GPU
per token, so 8 Sparks at their 273 GB/s spec cannot exceed **~53 tok/s** without speculation — the MTP
drafter takes us past that. One verify step reads ~9.5 GB per GPU (≈36 ms at the ~260 GB/s the kernels can
reach), so the 53 ms step runs at ~68% of the memory-bandwidth roofline; at 1M context (15.9 GB per step) it
is ~79%, with the KV-cache read itself streaming at ~270 GB/s. Details: [docs/RESULTS.md](docs/RESULTS.md).

**Prefill** (one request): a 31K-token prompt reaches its first token in **14.3 s** (19.6 s before the prefill
work), 53K tokens in ~27 s, 120K tokens in ~65 s — about 2K tokens/s. The largest remaining part is the TP
all-reduce traffic (~28% of prefill time). Details: [docs/RESULTS.md#prefill](docs/RESULTS.md#prefill).

## What is in here

```
docker/Dockerfile            SGLang v0.5.21 (pinned digest) + torchcodec + patches + compiled kernels
patches/sglang-v0.5.21/      12 patches against SGLang v0.5.21 (10 modified files, 2 new kernels)
kernels/rdma_ar/             one-/two-shot RDMA all-reduce + all-gather over ibverbs (CUDA + C++ proxy)
kernels/skinny_gemm/         decode-sized BF16 and FP8 block-scaled GEMMs (M <= 8)
kernels/tests/               correctness + speed tests for every kernel change
scripts/                     host setup, image build, model distribution, launch / wait / stop
bench/                       decode and prefill (TTFT) benchmarks, decode-at-long-context, long-context
                             retrieval check, RDMA test, profiling helpers
docs/                        SETUP, OPTIMIZATIONS, RESULTS, PITFALLS
```

## Quick start

Full walkthrough with explanations: **[docs/SETUP.md](docs/SETUP.md)**.

```bash
# 0. 8 DGX Sparks on a RoCEv2 fabric (all 4 RoCE netdevs addressed, MTU 9000), passwordless ssh from a
#    control machine, docker on every node. Then, on EVERY node (proactive compaction off, fabric firewall):
sudo bash scripts/setup-host.sh 10.77.0.0/24

# 1. Model on every node at the same path (566 GB): download once, copy over the fabric.
hf download XiaomiMiMo/MiMo-V2.6-Pro-MOPD --local-dir ~/models/MiMo-V2.6-Pro-MOPD     # on node 1
scripts/distribute-model.sh spark-1 10.77.0.21 10.77.0.31 ...                         # from the control machine

# 2. Describe the cluster, build the image on every node.
cp scripts/cluster.env.example scripts/cluster.env && $EDITOR scripts/cluster.env
scripts/build-image.sh
scripts/warm-flashinfer.sh     # pre-compile FlashInfer's prefill kernel into each node's JIT cache

# 3. Launch (weights load in ~13 min) and benchmark.
scripts/launch-all.sh && scripts/wait-ready.sh
bench/run-bench.sh
bench/longctx-decode.sh        # decode step time at 128K..1M context, no long prefill needed
bench/prefill-ttft.sh 32768    # time to first token of a ~31K-token prompt
```

The OpenAI-compatible API listens on `127.0.0.1:30000` on rank 0 (`ssh -L 30000:127.0.0.1:30000 spark-1`).

## The changes, briefly

Each is explained, with measurements, in **[docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md)**.

| Change | Where | Effect |
|---|---|---|
| Make it start: torchcodec, Marlin MXFP4 MoE, SWA metadata for the drafter's CUDA graphs | Dockerfile, patch 05 | server starts with TP8 + multi-layer EAGLE |
| RDMA one-/two-shot all-reduce + all-gather, CUDA-graph safe, bit-identical across ranks | `kernels/rdma_ar`, patch 04 | 150 all-reduces/step: ~190 µs → ~35 µs each |
| Grouped-head split-KV verify attention enabled on CUDA, tuned for GB10 | patches 03, 05 | 1M-context verify: 109 → 2.9 ms per layer |
| Fused sliding-window + sink verify attention | patches 02, 05 | 60 SWA layers: ~28 → ~9 µs each |
| Skinny BF16 GEMM (o_proj) and FP8 block-scaled GEMM (qkv, dense, MTP) | `kernels/skinny_gemm`, patches 07, 08 | 10–28% faster than cuBLAS / Triton W8A8 at M ≤ 4 |
| Drafter: shard the replicated `eh_proj`; FP8 copy of its lm_head | patches 06, 10 | −1.8 ms and −1.3 ms per step |
| Prefill attention tiles for sm_121 (128×64, 2 stages) | patch 01 | 2× prefill attention throughput |
| Prefill MoE GEMMs (Triton) on Marlin's MXFP4 weights, FP4 decoded in inline PTX | patches 11, 12 | 8K-token chunk: 23.6 → 16.7 ms per layer |
| FlashInfer FA2 for the full-attention layers' prefill | patch 05 | 1.2× (empty prefix) to 1.8× (131K prefix) |
| Sliding-window prefill starts at the first visible key tile; no MoE scratch zero-fill | patches 01, 11 | −0.17 s and −0.30 s per 8K chunk |
| NCCL Simple protocol on 8 channels for prefill's 96 MB all-reduces | launcher | 7.9 → 6.8 ms each |
| Memory hygiene: drop checkpoint page cache during load, reserve prefill activations, compaction off | launcher, patch 09, setup-host | removes 1.5–8× "slow rank" stragglers |
| Scheduler: `--scheduler-recv-interval 8`, `--sleep-on-idle`; RDMA proxy idle back-off | launcher, `kernels/rdma_ar` | less per-step CPU work; no spinning cores when idle |
| Decode-at-long-context measurement switch | patch 05, `bench/longctx-decode.sh` | time 1M-context decode without a 1M prefill |

## Things that will bite you

Read **[docs/PITFALLS.md](docs/PITFALLS.md)** before debugging performance. The big one: **set
`vm.compaction_proactiveness=0`**. With the default (20), kcompactd periodically tries to migrate pages the GPU
holds, fails, and stalls that node's GPU for seconds; with TP8 every node waits for it, and decode swings
between 53 and 70–77 ms per step.

## Limitations

- **Prefill runs at ~2K tokens/s** (31K tokens in 14.3 s). The TP all-reduces (150 × 96 MB per 8K chunk,
  ~28% of the time) are not overlapped with compute yet — our two-stream attempt was slower, see
  [OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md#what-we-tried-and-dropped). A 1M-token prompt would take an
  estimated 20–25 minutes, two thirds of it in full attention.
- Tuned for one request at a time (`--max-running-requests 4`); batched throughput was not a goal.
- The patches target **SGLang v0.5.21** exactly and several choices are MiMo-V2- and GB10-specific
  (head dims 192/128, one KV head per rank at TP8, 128-token SWA window, sm_121 tile sizes).
- Decode at long context was measured with the synthetic-KV switch (same bytes and kernels as a real
  context, see `bench/longctx-decode.sh`); real long prompts also pay the prefill.

## License

Apache-2.0 (see [LICENSE](LICENSE)). The files in `patches/` modify [SGLang](https://github.com/sgl-project/sglang)
(Apache-2.0); see [NOTICE](NOTICE). The model weights are not included — see the
[model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD) for their license.
