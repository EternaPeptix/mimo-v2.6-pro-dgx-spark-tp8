#!/usr/bin/env python3
"""bench_skinny.py: skinny_gemm vs torch (cuBLAS) on the MiMo TP8 decode shapes; checks accuracy too."""
import os, sys, time
import torch
sys.path.insert(0, os.environ.get("SKINNY_GEMM_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")))
import skinny_gemm_ext as sk
dev = torch.device("cuda", 0)
SHAPES = [("o_proj", 6144, 2048, 70), ("router", 384, 6144, 69), ("lm_head", 19072, 6144, 2)]

def bench(fn, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters

for name, N, K, copies in SHAPES:
    W = [torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.02 for _ in range(copies)]
    nbytes = sum(w.numel() * 2 for w in W)
    for M in (1, 2, 4, 8):
        if not sk.fits(M, K):
            continue
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        ref = (x.float() @ W[0].float().T)
        res = {}
        t = bench(lambda: [torch.nn.functional.linear(x, w) for w in W])
        res["torch"] = nbytes / t / 1e9
        for v in (0, 1, 2):
            y = sk.skinny_gemm(x, W[0], v)
            err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
            t = bench(lambda: [sk.skinny_gemm(x, w, v) for w in W])
            res[f"v{v}"] = nbytes / t / 1e9
            res[f"err{v}"] = err
        us = {k: (nbytes / copies) / (v * 1e9) * 1e6 for k, v in res.items() if not k.startswith("err")}
        print(f"{name:8s} N={N:5d} K={K} M={M}: " + " ".join(f"{k}={res[k]:.0f}GB/s({us[k]:.0f}us)" for k in ("torch", "v0", "v1", "v2"))
              + f"  relerr={max(res['err0'], res['err1'], res['err2']):.1e}", flush=True)
