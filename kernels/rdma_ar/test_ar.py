#!/usr/bin/env python3
"""test_ar.py: 8-node check of the RDMA all-reduce against NCCL (one process per Spark).

1. correctness vs an fp32 NCCL reference, and bit-identical output across ranks
2. stress: many back-to-back calls on fresh random data (catches stale/out-of-order reads)
3. latency, eager and inside a CUDA graph, next to NCCL
"""
import os, sys, time
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdma_ar import RdmaAllReduce  # noqa: E402

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
dist.init_process_group("nccl", device_id=dev)
cpu = dist.new_group(backend="gloo")
ar = RdmaAllReduce(cpu, rank, world, dev)
H = 6144
TOKENS = [int(t) for t in os.environ.get("AR_TOKENS", "1,2,4,8,16").split(",")]
log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)


def checksum(t):
    return t.view(torch.int16).to(torch.int64).sum().item()


# 1) correctness
for n in TOKENS:
    g = torch.Generator(device=dev).manual_seed(1234 + rank)
    x = torch.randn((n, H), generator=g, device=dev).to(torch.bfloat16)
    ref = x.float().clone(); dist.all_reduce(ref)
    for two in (False, True):
        y = ar.all_reduce(x, two_shot=two); torch.cuda.synchronize()
        err = (y.float() - ref).abs().max().item()
        sums = [None] * world
        dist.all_gather_object(sums, checksum(y), group=cpu)
        log(f"correct {'two' if two else 'one'}-shot tokens={n:2d} max|rdma-fp32ref|={err:.4f} identical_across_ranks={len(set(sums)) == 1}")

# 2) stress on fresh data each call, verified against NCCL fp32 on a sample of iterations
bad = 0
iters = int(os.environ.get("AR_STRESS", "3000"))
for i in range(iters):
    n = TOKENS[i % len(TOKENS)]
    x = torch.randn((n, H), device=dev).to(torch.bfloat16)
    y = ar.all_reduce(x, two_shot=bool(i % 3))
    if i % 50 == 0:
        ref = x.float(); dist.all_reduce(ref)
        if (y.float() - ref).abs().max().item() > 0.5:
            bad += 1
torch.cuda.synchronize()
tot = torch.tensor([bad], device=dev); dist.all_reduce(tot)
log(f"stress iters={iters} bad_samples_all_ranks={int(tot.item())}")


def t_eager(f, it=500):
    for _ in range(20): f()
    torch.cuda.synchronize(); dist.barrier()
    t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / it * 1e6


def t_graph(f, reps=50, it=10):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): f()
    g.replay(); torch.cuda.synchronize(); dist.barrier()
    t = time.perf_counter()
    for _ in range(it): g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / (it * reps) * 1e6


# 3) latency
for n in TOKENS:
    x = torch.randn((n, H), device=dev).to(torch.bfloat16)
    o_g = t_graph(lambda: ar.all_reduce(x, two_shot=False))
    w_g = t_graph(lambda: ar.all_reduce(x, two_shot=True))
    def nc(): dist.all_reduce(x)
    n_g = t_graph(nc)
    log(f"latency tokens={n:2d} ({n*H*2//1024:3d} KB) graph: one-shot={o_g:6.1f} two-shot={w_g:6.1f} nccl={n_g:6.1f} us")

# all-gather (logits-shaped fp32 and bf16): correctness + latency vs NCCL
for dt, n in ((torch.float32, 4), (torch.bfloat16, 4), (torch.float32, 1)):
    V = 19072
    x = torch.full((n, V), float(rank), dtype=dt, device=dev) + torch.arange(V, device=dev, dtype=dt) * 0
    out = torch.empty((world * n, V), dtype=dt, device=dev)
    ar.all_gather_into(out, x); torch.cuda.synchronize()
    ok = all(bool((out[r * n:(r + 1) * n] == float(r)).all()) for r in range(world))
    ref = torch.empty_like(out)
    r_g = t_graph(lambda: ar.all_gather_into(out, x))
    n_g = t_graph(lambda: dist.all_gather_into_tensor(ref, x))
    log(f"allgather {str(dt).split('.')[-1]:8s} [{n},{V}] ({x.numel()*x.element_size()//1024} KB) ok={ok} rdma graph={r_g:6.1f} us | nccl graph={n_g:6.1f} us")

# arena backing and GPU<->proxy hand-off
if hasattr(ar.h, "arena_info"):
    info = ar.h.arena_info(); rtt = ar.h.pingpong(2000)
    allinfo = [None] * world
    dist.all_gather_object(allinfo, (info, rtt), group=cpu)
    for r, (inf, rt) in enumerate(allinfo):
        log(f"arena rank{r}: {inf[0]/2**20:.0f} MiB, huge-page backed {inf[1]/2**20:.0f} MiB, thp_path={inf[2]} | GPU->proxy->GPU round trip {rt/1e3:.2f} us")

# latency breakdown (instrumented build): one-shot 12 KB and two-shot 48 KB inside graphs
if hasattr(ar.h, "stats"):
    for n, two in ((1, False), (4, True)):
        x = torch.randn((n, H), device=dev).to(torch.bfloat16)
        torch.cuda.synchronize(); dist.barrier(); ar.h.stats(True)
        t_graph(lambda: ar.all_reduce(x, two_shot=two), reps=50, it=20)
        s = ar.h.stats(True)
        allst = [None] * world
        dist.all_gather_object(allst, s, group=cpu)
        if rank == 0:
            for r, v in enumerate(allst):
                print(f"breakdown {'two' if two else 'one'}-shot {n*H*2//1024}KB rank{r}: n={v[0]:.0f} stage={v[1]/1e3:.1f} pub->flags={v[2]/1e3:.1f} flags->end={v[3]/1e3:.1f} total={v[4]/1e3:.1f} us | proxy n={v[5]:.0f} handoff={v[6]/1e3:.1f} post={v[7]/1e3:.1f} max_post={v[8]/1e3:.1f} us", flush=True)

# graph replay correctness: values change between replays, result must track them
x = torch.zeros((4, H), device=dev, dtype=torch.bfloat16)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    ar.all_reduce(x)
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    y = ar.all_reduce(x)
ok = True
for k in range(50):
    x.fill_(float(rank + k % 7))
    g.replay(); torch.cuda.synchronize()
    want = sum(float(r + k % 7) for r in range(world))
    ok &= abs(y.float().mean().item() - want) < 1e-2
oks = [None] * world
dist.all_gather_object(oks, ok, group=cpu)
log(f"graph replay tracking inputs: {all(oks)}")
dist.barrier()
log("TEST_DONE")
dist.destroy_process_group()
