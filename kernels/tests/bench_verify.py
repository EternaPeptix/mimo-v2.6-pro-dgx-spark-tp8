#!/usr/bin/env python3
"""bench_verify.py: MiMo full-attention verify shape on one TP8 rank (16 q heads, 1 kv head,
d_qk=192, d_v=128, 4 draft tokens). Compares the serial-prefix extend kernel (what SGLang runs on
CUDA today) with the grouped-head split-KV verify kernel, for numerics and time vs context length."""
import itertools, os, sys, time
import torch
from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention import verify_mla as vm

dev = torch.device("cuda", 0)
H_Q, H_KV, D, DV, L = 16, 1, 192, 128, 4
bf = torch.bfloat16
sm = 1.0 / D ** 0.5
PAGE = 64

def timeit(fn, iters):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3

configs = [(4, 64, 8, 32, 512), (16, 64, 8, 64, 512), (16, 32, 4, 96, 512), (8, 64, 8, 64, 512), (16, 64, 4, 96, 1024)]
for S in [int(s) for s in os.environ.get("CTX", "16384,65536,262144,1048576").split(",")]:
    npages = (S + PAGE - 1) // PAGE + 4
    k_buf = torch.randn(npages * PAGE, H_KV, D, dtype=bf, device=dev)
    v_buf = torch.randn(npages * PAGE, H_KV, DV, dtype=bf, device=dev)
    # shuffled pages, like a real paged KV pool
    perm = torch.randperm(npages, device=dev)[: (S + PAGE - 1) // PAGE]
    kv_indices = (perm[:, None] * PAGE + torch.arange(PAGE, device=dev)[None, :]).reshape(-1)[:S].to(torch.int64)
    q = torch.randn(L, H_Q, D, dtype=bf, device=dev)
    ke = torch.randn(L, H_KV, D, dtype=bf, device=dev)
    ve = torch.randn(L, H_KV, DV, dtype=bf, device=dev)
    qo = torch.tensor([0, L], dtype=torch.int32, device=dev)
    kvp = torch.tensor([0, S], dtype=torch.int32, device=dev)
    o_ref = torch.empty(L, H_Q, DV, dtype=bf, device=dev)
    run_ref = lambda: extend_attention_fwd(q, ke, ve, o_ref, k_buf, v_buf, qo, kvp, kv_indices, None, True, None, L, 1.0, 1.0, sm)
    it = 3 if S >= 262144 else 10
    t_ref = timeit(run_ref, it)
    gb = S * (D + DV) * 2 / 1e9
    print(f"S={S:8d} KV={gb*1e3:6.0f} MB  extend(serial prefix): {t_ref:8.2f} ms ({gb/t_ref*1e3:5.1f} GB/s)", flush=True)
    for bh, bn, nw, maxs, tgt in configs:
        vm._BLOCK_CONFIG[D] = (bh, bn, nw); vm.MAX_N_SPLITS = maxs; vm.TARGET_PROGRAMS = tgt; vm._VMLA_CACHE.clear()
        o = torch.empty_like(o_ref)
        ok = vm.verify_shared_kv_fwd(q, ke, ve, o, k_buf, v_buf, qo, kvp, kv_indices, None, True, None, L, 1.0, 1.0, sm)
        err = (o.float() - o_ref.float()).abs().max().item()
        t = timeit(lambda: vm.verify_shared_kv_fwd(q, ke, ve, o, k_buf, v_buf, qo, kvp, kv_indices, None, True, None, L, 1.0, 1.0, sm), it * 3)
        print(f"    shared-kv BLOCK_H={bh:2d} BLOCK_N={bn:3d} warps={nw} splits<={maxs:3d}: {t:7.3f} ms ({gb/t*1e3:5.1f} GB/s) ran={ok} maxerr={err:.4f}", flush=True)
    del k_buf, v_buf
