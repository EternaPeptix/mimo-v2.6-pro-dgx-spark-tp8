#!/usr/bin/env python3
"""bench_prefill2.py: extend-kernel tile configs for MiMo full attention (d_qk=192, d_v=128) on GB10.
Monkeypatches the block-size picker; checks every config against the stock (64,64,8w,1st) output."""
import os, time
import torch
import sglang.kernels.ops.attention.extend_attention as ea
dev = torch.device("cuda", 0); bf = torch.bfloat16
H_Q, H_KV, D, DV, T, PAGE = 16, 1, 192, 128, 8192, 64
sm = 1.0 / D ** 0.5
orig_bs, orig_st = ea._get_block_sizes_for_extend_attention, ea._get_num_stages_for_extend_attention
CFGS = [(64, 64, 8, 1), (128, 64, 8, 1), (128, 32, 8, 1), (128, 64, 8, 2), (64, 64, 4, 2), (128, 32, 4, 2)]
for P in [int(s) for s in os.environ.get("PREFIX", "65536").split(",")]:
    npages = (P + PAGE - 1) // PAGE + 2
    k_buf = torch.randn(npages * PAGE, H_KV, D, dtype=bf, device=dev); v_buf = torch.randn(npages * PAGE, H_KV, DV, dtype=bf, device=dev)
    perm = torch.randperm(npages, device=dev)[: (P + PAGE - 1) // PAGE]
    kv_indices = (perm[:, None] * PAGE + torch.arange(PAGE, device=dev)[None, :]).reshape(-1)[:P].to(torch.int64)
    q = torch.randn(T, H_Q, D, dtype=bf, device=dev); ke = torch.randn(T, H_KV, D, dtype=bf, device=dev)
    ve = torch.randn(T, H_KV, DV, dtype=bf, device=dev)
    qo = torch.tensor([0, T], dtype=torch.int32, device=dev); kvp = torch.tensor([0, P], dtype=torch.int32, device=dev)
    ref = None
    for bm, bn, nw, ns in CFGS:
        ea._get_block_sizes_for_extend_attention = lambda Lq, Lv, bm=bm, bn=bn, nw=nw: (128, 64, 128, bm, bn, nw)
        ea._get_num_stages_for_extend_attention = lambda Lq, Lv, block_n=None, ns=ns: ns
        o = torch.empty(T, H_Q, DV, dtype=bf, device=dev)
        f = lambda: ea.extend_attention_fwd(q, ke, ve, o, k_buf, v_buf, qo, kvp, kv_indices, None, True, None, T, 1.0, 1.0, sm)
        try:
            f(); torch.cuda.synchronize()
        except Exception as e:
            print(f"P={P} BLOCK_M={bm} BLOCK_N={bn} warps={nw} stages={ns}: FAILED {type(e).__name__}: {str(e)[:100]}", flush=True); continue
        if ref is None: ref = o.clone()
        err = (o.float() - ref.float()).abs().max().item()
        t = time.perf_counter(); it = 3
        for _ in range(it): f()
        torch.cuda.synchronize(); ms = (time.perf_counter() - t) / it * 1e3
        flops = 2 * T * H_Q * (D + DV) * (P + T / 2)
        print(f"P={P} BLOCK_M={bm:3d} BLOCK_N={bn:3d} warps={nw} stages={ns}: {ms:8.1f} ms {flops/ms/1e9:6.1f} TFLOPS maxerr_vs_stock={err:.4f}", flush=True)
