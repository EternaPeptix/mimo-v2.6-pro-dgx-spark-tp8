# Setup: MiMo-V2.6-Pro-MOPD on 8× DGX Spark

Everything below runs from a **control machine** (any host that can `ssh` to all eight Sparks without a
password) except the one-time host setup, which needs `sudo` on each node.

## 0. What we ran on

| Component | Version / detail |
|---|---|
| Nodes | 8× NVIDIA DGX Spark (GB10: sm_121, 128 GB unified LPDDR5X, ~273 GB/s) |
| OS | DGX OS 7.4.0 (Ubuntu 24.04), kernel 7.0.0-1019-nvidia, driver 580.178.04 |
| NIC | built-in ConnectX-7, firmware 28.45.4028, both QSFP ports cabled at 200 Gb/s |
| Switch | Cisco 8101-32FH-O running SONiC 202411, 400G ports split 2×200G; plain lossy Ethernet (no PFC/ECN) |
| Container | `lmsysorg/sglang:v0.5.21-cu130` (CUDA 13.0, torch 2.13.0, NCCL 2.30.7) + this repo's patches |
| Model | `XiaomiMiMo/MiMo-V2.6-Pro-MOPD` (MXFP4 experts, FP8 e4m3 attention, BF16 o_proj; 566 GB) |

Any RoCEv2-capable switch with enough 200G ports should work; nothing here depends on the switch beyond
plain L2/L3 forwarding and jumbo frames.

## 1. Fabric

Each Spark's ConnectX-7 shows up as **four** RoCE devices: two ports, each reachable over two PCIe Gen5 x4
links (`rocep1s0f0`, `rocep1s0f1`, `roceP2p1s0f0`, `roceP2p1s0f1`, netdevs `enp1s0f0np0`, `enp1s0f1np1`,
`enP2p1s0f0np0`, `enP2p1s0f1np1`). Give **all four** an address in one subnet and set MTU 9000 (also on
the switch). We used `10.77.0.<node*10 + k>/24`, e.g. node 2 = `10.77.0.21`..`10.77.0.24`.

Check on every node:

```bash
ip -br addr | grep -E 'enp1s0f|enP2p1s0f'                       # 4 addresses, UP, mtu 9000
cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/5   # "RoCE v2" -> GID index 5 = RoCEv2/IPv4
ib_write_lat -d rocep1s0f0 -x 5 <peer-ip>                         # ~3 us between two nodes (perftest)
```

## 2. Host settings (every node, once)

```bash
sudo bash scripts/setup-host.sh 10.77.0.0/24
```

This sets `vm.compaction_proactiveness=0` (live and in `/etc/sysctl.d/`) and, if `ufw` is active, allows
the fabric subnet. **Do not skip the compaction setting**; see [PITFALLS.md](PITFALLS.md#1-proactive-memory-compaction).

We also ran with `iommu.passthrough=1` on the kernel command line (the ConnectX-7 then DMAs without IOMMU
translation). We did not A/B it for this workload; to set it, add it to `GRUB_CMDLINE_LINUX_DEFAULT` in
`/etc/default/grub`, run `sudo update-grub`, reboot.

Your user needs to be in the `docker` group on every node.

## 3. Model weights (566 GB per node)

Download once on one node, then copy over the fabric:

```bash
# on node 1
pip install -U huggingface_hub
hf download XiaomiMiMo/MiMo-V2.6-Pro-MOPD --local-dir ~/models/MiMo-V2.6-Pro-MOPD

# from the control machine: node 1 -> the other seven (fabric IPs), ~1 GB/s each, resumable.
# Needs ssh from node 1 to the others (uses agent forwarding).
scripts/distribute-model.sh spark-1 10.77.0.21 10.77.0.31 10.77.0.41 10.77.0.51 10.77.0.61 10.77.0.71 10.77.0.81
```

Verify every node has all 155 files with the same sizes before launching (a truncated shard fails late,
after minutes of loading).

## 4. Cluster description

```bash
cp scripts/cluster.env.example scripts/cluster.env
```

Edit `NODES` (ssh names in TP-rank order; rank 0 serves the API), `MASTER_ADDR` (rank 0's fabric IP),
the interface/HCA names if yours differ, and `MODEL_DIRS`.

## 5. Image

```bash
scripts/build-image.sh       # copies the repo to every node and runs docker build there, in parallel
```

The Dockerfile starts from the SGLang image **pinned by digest**, installs `torchcodec` and the ibverbs
headers, applies `patches/sglang-v0.5.21/*.patch` with `patch -p1`, and compiles the two extensions for
sm_121 (no GPU needed at build time). Nodes without registry access can't resolve the pinned base digest;
build on one node and stream it to the others instead (~8 min each, over the fabric):

```bash
ssh -A spark-2 'docker save mimo-gb10:v0.5.21 | ssh 10.77.0.11 docker load'
```

Then compile FlashInfer's prefill kernel for MiMo's head dims into every node's JIT cache (otherwise the first
long prompt after a launch waits ~1–2 min while all 8 ranks compile it at once):

```bash
scripts/warm-flashinfer.sh   # needs the GPU: run it while the server is stopped
```

Optional, but a good check that the fabric and the RDMA extension work (stop any running server first):

```bash
bench/rdma-test.sh
# expect: identical_across_ranks=True, bad_samples_all_ranks=0, one-shot 12 KB ~21 us, two-shot 48 KB ~34 us
```

## 6. Launch

```bash
scripts/launch-all.sh        # ranks 1..7, then rank 0
scripts/wait-ready.sh        # ~13 min of weight loading + 1-4 min of CUDA-graph capture
```

What the launcher does (see `scripts/mimo-tp8.sh`): TP8 across the 8 nodes, Marlin MXFP4 MoE, Triton
attention, multi-layer EAGLE with the built-in MTP layers (3 draft steps, top-1, 4 verified tokens),
1,048,576-token context, a 1.11M-token full-attention KV pool and a small sliding-window pool, page size 64,
plus the feature switches from [OPTIMIZATIONS.md](OPTIMIZATIONS.md) (all on by default; each can be turned
off with an env var, e.g. `RDMA_AR=0 scripts/launch-all.sh`). The prefill ones are `MXFP4_PREFILL_MOE`
(prefill MoE GEMMs), `FI_FULL_PREFILL` (FlashInfer for the full-attention layers) and `NCCL_PROTO` /
`NCCL_CHANNELS` (defaults `Simple` / 8, for prefill's large all-reduces).

Keep other GPU services off the 8 nodes. In particular, a container with a `--restart` policy that crash-loops
while the server holds the memory will come up in the gap between a stop and the next launch, and the launch
then fails with "The memory capacity is unbalanced" (see [PITFALLS.md](PITFALLS.md)).

Triton/JIT caches persist in `~/.cache/mimo-gb10-jit` on each node, so later starts skip most compilation.

## 7. Use it

The OpenAI-compatible server listens on `127.0.0.1:30000` on rank 0:

```bash
ssh -L 30000:127.0.0.1:30000 spark-1
curl -s localhost:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"mimo-v2.6-pro","messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":64}'
```

`--reasoning-parser mimo` and `--tool-call-parser mimo` are enabled; pass
`"chat_template_kwargs": {"enable_thinking": true}` to turn on thinking.

## 8. Benchmark

```bash
bench/run-bench.sh             # code / math / essay prompts: tok/s, MTP acceptance, ms per verify step
bench/longctx-decode.sh        # step time at 0 / 128K / 256K / 512K / 1M context (no long prefill)
bench/prefill-ttft.sh 32768 3  # time to first token of a ~31K-token prompt (FI_AB=1: also without FlashInfer)
ssh spark-1 "python3 - 120000 1" < bench/needle.py   # long-context retrieval check (5 facts at 5 depths)
```

Kernel-level tests and micro-benchmarks are in the image under `/opt/mimo-gb10/tests` (run them on a node
whose server is stopped), e.g.
`docker run --rm --gpus all mimo-gb10:v0.5.21 python3 /opt/mimo-gb10/tests/test_mxfp4_prefill_moe.py`.

See [RESULTS.md](RESULTS.md) for what to expect and how to read the numbers.

## 9. Stop

```bash
scripts/stop-all.sh
```
