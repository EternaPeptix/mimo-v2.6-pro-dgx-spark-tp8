#!/usr/bin/env python3
"""test_swa_prefill.py: sliding-window (128) + sink prefill attention through the patched Triton extend kernel
(patch 01: the chunk's key loop starts at the first tile the query block can see) against an FP32 reference,
at MiMo-V2 TP8 shapes (16 query heads, 1 KV head, head dims 192/128), with timings.

Semantics checked: the window list holds the last min(prefix, 128) prefix tokens; a key at absolute position k
is visible to the query at absolute position q iff k <= q and q - k <= 128; the sink logit adds exp(sink) to
the softmax denominator only.
"""
import time
import torch
import sglang.kernels.ops.attention.extend_attention as ea

dev = torch.device("cuda", 0)
bf = torch.bfloat16
H, HKV, DQK, DV, W = 16, 1, 192, 128, 128
SCALE = DQK ** -0.5
torch.manual_seed(0)

for T, P in ((8192, 8192), (8192, 0), (2048, 500)):
    wl = min(P, W)
    k_buf = torch.randn(P + 64, HKV, DQK, dtype=bf, device=dev)   # prefix keys at their absolute positions
    v_buf = torch.randn(P + 64, HKV, DV, dtype=bf, device=dev)
    win_idx = torch.arange(P - wl, P, device=dev, dtype=torch.int64)
    q = torch.randn(T, H, DQK, dtype=bf, device=dev)
    k_ext = torch.randn(T, HKV, DQK, dtype=bf, device=dev)
    v_ext = torch.randn(T, HKV, DV, dtype=bf, device=dev)
    sinks = torch.randn(H, dtype=torch.float32, device=dev)
    o = torch.empty(T, H, DV, dtype=bf, device=dev)
    run = lambda: ea.extend_attention_fwd(
        q, k_ext, v_ext, o, k_buf, v_buf, torch.tensor([0, T], dtype=torch.int32, device=dev),
        torch.tensor([0, wl], dtype=torch.int32, device=dev), win_idx, None, True, None, T, 1.0, 1.0, SCALE,
        sliding_window_size=W, sinks=sinks, window_kv_offsets=torch.tensor([P - wl], dtype=torch.int64, device=dev))
    run(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        run()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 10 * 1e3

    k_all = torch.cat([k_buf[:P], k_ext]).float()[:, 0]           # absolute positions 0 .. P+T-1
    v_all = torch.cat([v_buf[:P], v_ext]).float()[:, 0]
    err = 0.0
    for i in (0, 1, 127, 128, 129, T // 2, T - 1):
        qa = P + i
        lo = max(qa - W, P - wl)
        for h in (0, 5, 15):
            s = (q[i, h].float() @ k_all[lo:qa + 1].T) * SCALE
            m = torch.maximum(s.max(), sinks[h])
            p = torch.exp(s - m)
            ref = (p @ v_all[lo:qa + 1]) / (p.sum() + torch.exp(sinks[h] - m))
            err = max(err, ((o[i, h].float() - ref).abs().max() / ref.abs().max()).item())
    status = "OK" if err < 2e-2 else "MISMATCH"
    print(f"SWA prefill T={T:5d} prefix={P:5d}: {ms:7.3f} ms   max rel err vs fp32 {err:.1e}  {status}", flush=True)
