#!/usr/bin/env python3
"""test_swa_verify.py: swa_verify_fwd vs extend_attention_fwd on MiMo SWA verify shapes (TP8 rank:
16 q heads, 1 kv head, d=192/128, W=128, sinks, 4 draft tokens)."""
import time
import torch
from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention import swa_verify as sv  # added by patches/02-swa_verify.patch
dev = torch.device("cuda", 0); bf = torch.bfloat16
H, HKV, Dq, Dv, L, W = 16, 1, 192, 128, 4, 128
sm = 1.0 / Dq ** 0.5
def run(prefix_lens):
    bs = len(prefix_lens); pool = 4096 * 4
    k_buf = torch.randn(pool, HKV, Dq, dtype=bf, device=dev); v_buf = torch.randn(pool, HKV, Dv, dtype=bf, device=dev)
    perm = torch.randperm(pool, device=dev)
    win_lens = [min(p, W) for p in prefix_lens]
    win_indptr = torch.tensor([0] + list(torch.tensor(win_lens).cumsum(0).tolist()), dtype=torch.int32, device=dev)
    win_indices = perm[: sum(win_lens)].to(torch.int64)
    offs = torch.tensor([p - w for p, w in zip(prefix_lens, win_lens)], dtype=torch.int64, device=dev)
    q = torch.randn(bs * L, H, Dq, dtype=bf, device=dev); ke = torch.randn(bs * L, HKV, Dq, dtype=bf, device=dev)
    ve = torch.randn(bs * L, HKV, Dv, dtype=bf, device=dev)
    qo = torch.arange(0, (bs + 1) * L, L, dtype=torch.int32, device=dev)
    sinks = torch.randn(H, dtype=torch.float32, device=dev)
    o_ref = torch.empty(bs * L, H, Dv, dtype=bf, device=dev); o = torch.empty_like(o_ref)
    ref = lambda: extend_attention_fwd(q, ke, ve, o_ref, k_buf, v_buf, qo, win_indptr, win_indices, None, True, None, L,
                                       1.0, 1.0, sm, sliding_window_size=W, sinks=sinks, window_kv_offsets=offs)
    new = lambda: sv.swa_verify_fwd(q, ke, ve, o, k_buf, v_buf, qo, win_indptr, win_indices, L, W, sm, sinks=sinks)
    ref(); assert new(); torch.cuda.synchronize()
    err = (o.float() - o_ref.float()).abs().max().item()
    def t(f, reps=100):
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3): f()
        torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps): f()
        g.replay(); torch.cuda.synchronize(); s = time.perf_counter()
        for _ in range(10): g.replay()
        torch.cuda.synchronize(); return (time.perf_counter() - s) / (10 * reps) * 1e6
    print(f"prefix={prefix_lens}: maxerr={err:.5f}  extend={t(ref):6.1f} us  swa_verify={t(new):6.1f} us", flush=True)
for pl in ([50], [128], [1000], [5000, 37], [300, 128, 2000, 90]):
    run(pl)
