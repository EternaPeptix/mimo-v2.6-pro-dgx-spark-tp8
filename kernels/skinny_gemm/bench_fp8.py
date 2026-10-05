#!/usr/bin/env python3
"""bench_fp8.py: skinny FP8 (block-scaled) GEMM vs dequantized fp32 reference and SGLang's Triton W8A8."""
import os, sys, time
import torch
sys.path.insert(0, os.environ.get("SKINNY_GEMM_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")))
import skinny_gemm_ext as sk
from sglang.srt.layers.quantization.fp8_utils import triton_w8a8_block_fp8_linear
dev = torch.device("cuda", 0)
SHAPES = [("qkv", 3392, 6144, 30), ("dense_gate_up", 4096, 6144, 20), ("down", 6144, 2048, 40)]
def bench(fn, iters=10):
    for _ in range(2): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters
for name, N, K, copies in SHAPES:
    Ws, Ss = [], []
    for _ in range(copies):
        w = (torch.randn(N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        s = torch.rand((N + 127) // 128, K // 128, device=dev) * 0.02 + 0.001
        Ws.append(w); Ss.append(s)
    nbytes = sum(w.numel() for w in Ws)
    for M in (1, 4):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        sfull = Ss[0].repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)[:, :K]
        ref = x.float() @ (Ws[0].float() * sfull).T
        res = {}
        t = bench(lambda: [triton_w8a8_block_fp8_linear(x, w, [128, 128], s) for w, s in zip(Ws, Ss)])
        res["triton"] = t
        tr = triton_w8a8_block_fp8_linear(x, Ws[0], [128, 128], Ss[0]).float()
        err_t = ((tr - ref).abs().max() / ref.abs().max()).item()
        errs = []
        for v in (0, 1, 2):
            y = sk.skinny_gemm_fp8(x, Ws[0], Ss[0], v).float()
            errs.append(((y - ref).abs().max() / ref.abs().max()).item())
            res[f"v{v}"] = bench(lambda: [sk.skinny_gemm_fp8(x, w, s, v) for w, s in zip(Ws, Ss)])
        line = " ".join(f"{k}={nbytes/res[k]/1e9:.0f}GB/s({res[k]/copies*1e6:.0f}us)" for k in ("triton", "v0", "v1", "v2"))
        print(f"{name:14s} N={N} K={K} M={M}: {line} relerr skinny={max(errs):.1e} triton={err_t:.1e}", flush=True)
