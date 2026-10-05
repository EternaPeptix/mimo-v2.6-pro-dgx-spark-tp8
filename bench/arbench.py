#!/usr/bin/env python3
"""arbench.py: small-message all-reduce latency across the 8 Sparks (one GPU each).

Run one copy per node (RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT in env). Compares:
  nccl      dist.all_reduce (algorithm/protocol from NCCL_ALGO / NCCL_PROTO env)
  ag_sum    all_gather_into_tensor + local sum (one collective, no reduction hops)
  p2p_sum   one-shot: grouped isend/irecv to all peers, then local sum
Each is timed eagerly and inside a CUDA graph (the server replays graphs).
"""
import os, time
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(0)
dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
dev = torch.device("cuda", 0)
H = 6144
SIZES = [int(s) for s in os.environ.get("AR_TOKENS", "1,4,8").split(",")]
ITERS = int(os.environ.get("AR_ITERS", "300"))
METHODS = os.environ.get("AR_METHODS", "nccl,ag_sum,p2p_sum").split(",")


def make(method, x):
    n = x.shape[0]
    if method == "nccl":
        def f():
            dist.all_reduce(x)
            return x
    elif method == "ag_sum":
        buf = torch.empty((world, n, H), dtype=x.dtype, device=dev)
        def f():
            dist.all_gather_into_tensor(buf, x)
            torch.sum(buf, dim=0, out=x)
            return x
    elif method == "p2p_sum":
        buf = torch.empty((world, n, H), dtype=x.dtype, device=dev)
        def f():
            ops = []
            for p in range(world):
                if p == rank:
                    continue
                ops.append(dist.P2POp(dist.isend, x, p))
                ops.append(dist.P2POp(dist.irecv, buf[p], p))
            for r in dist.batch_isend_irecv(ops):
                r.wait()
            buf[rank].copy_(x)
            torch.sum(buf, dim=0, out=x)
            return x
    else:
        raise ValueError(method)
    return f


def check(method, n):
    # correctness: rank r contributes (r+1); sum = world*(world+1)/2
    x = torch.full((n, H), float(rank + 1), dtype=torch.bfloat16, device=dev)
    make(method, x)()
    torch.cuda.synchronize()
    want = world * (world + 1) / 2
    return abs(x.float().mean().item() - want) < 1e-3


def time_eager(f):
    for _ in range(20):
        f()
    torch.cuda.synchronize(); dist.barrier()
    t = time.perf_counter()
    for _ in range(ITERS):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / ITERS * 1e6


def time_graph(f, reps=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            f()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize(); dist.barrier()
    t = time.perf_counter()
    k = max(1, ITERS // reps)
    for _ in range(k):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / (k * reps) * 1e6


res = []
for m in METHODS:
    for n in SIZES:
        ok = check(m, n)
        x = torch.randn((n, H), dtype=torch.bfloat16, device=dev)
        f = make(m, x)
        e = time_eager(f)
        try:
            gr = time_graph(f)
        except Exception as ex:  # p2p capture may be unsupported in some builds
            gr = float("nan")
            if rank == 0:
                print(f"  graph capture failed for {m}: {type(ex).__name__}: {str(ex)[:120]}")
        res.append((m, n, ok, e, gr))
        dist.barrier()
if rank == 0:
    tag = f"ALGO={os.environ.get('NCCL_ALGO','default')} PROTO={os.environ.get('NCCL_PROTO','default')}"
    for m, n, ok, e, gr in res:
        print(f"{tag:32s} {m:8s} tokens={n:2d} ({n*H*2//1024:3d} KB) ok={ok} eager={e:7.1f} us  graph={gr:7.1f} us")
dist.destroy_process_group()
