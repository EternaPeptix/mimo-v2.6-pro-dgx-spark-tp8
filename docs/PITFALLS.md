# Pitfalls

Things that cost us hours, roughly in order of how much they hurt.

## 1. Proactive memory compaction

**Symptom.** Decode is bimodal: some requests at ~53 ms per step, others at 70–77 ms, with no change in
acceptance. Profiles show every kernel on *one* rank 30–150% slower and the other seven ranks waiting in
the all-reduce. Which rank is slow changes over time.

**Cause.** With `vm.compaction_proactiveness=20` (the default), kcompactd runs a background pass whenever more
than ~90% of a node's free memory is in fragments smaller than 2 MB, which is the normal state of a Spark
whose RAM is mostly GPU allocations. It tries to migrate pages the GPU holds, fails (millions of
`pgmigrate_fail` per minute), and while it does, that node's GPU stalls. On an idle cluster we measured
kcompactd active 22–45% of the time on five of eight nodes.

**Diagnose.** `python3 bench/vmwatch.py 60` on a node prints `/proc/vmstat` deltas every 0.5 s; look for
`compact_isolated` / `pgmigrate_fail` bursts.

**Fix.** `sysctl -w vm.compaction_proactiveness=0` (persisted by `scripts/setup-host.sh`). On-demand
compaction for high-order allocations still works. After the change: zero compaction, every run 52.7–53.7 ms.

## 2. Slow ranks from where memory was allocated

**Symptom.** Right after a launch, one or more ranks are consistently 1.5–8× slower in specific kernels
(attention, o_proj, prefill MoE) while the hardware tests identical in isolation.

**Cause.** GB10's GPU allocates from the same physical memory as the OS. Allocations made while memory is
fragmented (page cache churn from streaming the checkpoint, another GPU job running during the load) end
up with poor page sizes and thrash the GPU's TLB.

**Fix.** Evict page cache before launching (`scripts/dropcache.py`), keep
`--weight-loader-drop-cache-after-load`, reserve prefill memory early (`SGLANG_PREFILL_POOL_GB`), and
**never run benchmarks or other GPU work on a node while the server loads**. Our RDMA arena is explicitly
backed by 2 MB huge pages for the same reason.

## 3. NCCL over this fabric

GPUDirect RDMA is reported disabled on GB10 (`NET/IB : GPU Direct RDMA Disabled`), NCCL chooses ring/LL for
small messages, and every hop goes through its CPU proxy: ~95–115 µs eager and 130–190 µs inside CUDA
graphs for 12–48 KB across 8 nodes. Tree/LL is no better. That is why `kernels/rdma_ar` exists. NCCL is still
used for large messages (prefill), where its ring is bandwidth-efficient: 96 MB in ~7.9 ms with NCCL's
defaults, 6.8 ms with `NCCL_PROTO=Simple` and exactly 8 channels (the launcher's default).

## 4. The torch profiler distorts CUDA-graph launches

With the profiler (CUPTI) active, `cudaGraphLaunch` of the ~1500-node verify graph took ~550 µs instead of
~3 µs, and it stayed slower for a while after stopping the profiler. Don't read CPU launch gaps off a
profile; benchmark without it. Also, the first all-reduce in every rank's trace includes the time until
the *last* rank started profiling (tens of ms): drop it from latency statistics.

## 5. Every rank reads the whole checkpoint

With TP each rank streams all 566 GB of safetensors and keeps 1/8, so a cold start is ~13 minutes at
~700 MB/s NVMe. Weight files must be byte-identical on all nodes; a truncated shard fails late.

## 6. First requests compile kernels

The first long prompt after a start triggers Triton JIT compilation (we saw a 92 s first prefill chunk).
`scripts/mimo-tp8.sh` persists `/root/.triton` and `/root/.cache` to `~/.cache/mimo-gb10-jit` on each node,
which also cuts CUDA-graph capture from ~3.5 to ~1 minute on later starts. FlashInfer's prefill kernel is
compiled on first use too (~1–2 min with 8 ranks at once): `scripts/warm-flashinfer.sh` builds it into that
cache ahead of time. Even warm, the first long prompt after a launch is ~3 s slower than the next ones.

## 7. The radix cache flatters repeated long prompts

Re-sending a long prompt that shares a prefix with an earlier one skips most of the prefill. `bench/bench.py`
puts a random nonce at the start of every long prompt.

## 8. Other GPU services that restart themselves

**Symptom.** A launch that worked before fails on every rank with "The memory capacity is unbalanced. Some GPUs
may be occupied by other processes", or one rank shows ~20 GB less free memory than the others.

**Cause.** Another inference container on one node with a `--restart unless-stopped` (or `always`) policy. While
our server holds the memory it crash-loops (its restart count keeps climbing); in the few minutes between
stopping the server and the next launch it finally starts and keeps its memory.

**Fix.** Keep other GPU services off the 8 nodes while this runs, or `docker update --restart=no <name>` them for
the duration; check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` on every node before launching.

## 9. Smaller things

- `pkill -f <pattern>` inside `ssh host '...'` matches the remote shell running that very command and kills
  your session (exit 255). Anchor the pattern, e.g. `pkill -f "^python3 /path/script.py"`.
- A `FROM image@sha256:...` pin makes BuildKit contact the registry even when the image is local; on nodes
  without registry access build once and `docker save | docker load`.
- Containers run as root: anything they write into bind mounts (build directories, caches) is root-owned on
  the host.
- The long-context measurement switch is re-read every 32 verify steps, so the request right after you turn
  it off still decodes a few steps in measurement mode and continues from that text.
- In this SGLang build's GPU-only scheduling path, `ForwardBatch.seq_lens_cpu` / `extend_seq_lens_cpu` can be
  `None`. Anything that plans on the host (FlashInfer's `plan()`) has to fall back to the device tensors
  (one sync per prefill batch); our first FlashInfer integration silently never ran because of this.
- FP8 checkpoints give every attention layer `k_scale` / `v_scale` tensors equal to 1.0, not `None`: gate on
  the values (`k_scale_float == 1.0`), not on presence.
- Nsight Compute ships in the image and works on GB10 with `docker run --cap-add SYS_ADMIN ... ncu
  --kernel-name regex:<kernel> --launch-skip 1 --launch-count 1 python3 <test>`; the DRAM byte counters are not
  available on this unified-memory part, but occupancy, issue and stall statistics are.
