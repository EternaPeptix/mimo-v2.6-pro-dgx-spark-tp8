#!/usr/bin/env python3
"""bench_prefill_attention.py: full-attention prefill at MiMo-V2 TP8 shapes (16 query heads sharing 1 KV head,
head dims 192/128), one 8K-token chunk against growing prefixes: FlashInfer's FA2 paged prefill (patch 05,
SGLANG_FI_FULL_PREFILL) vs SGLang's Triton extend kernel, on the same paged KV (pages in random order), with
each kernel's max relative error against an FP32 reference on sampled rows.

usage: python3 bench_prefill_attention.py [chunk_tokens] [prefix ...]   (default: 8192  0 8192 32768 131072)
"""
import sys
import torch
import flashinfer
from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd

dev = torch.device("cuda", 0)
torch.manual_seed(0)
H, HKV, DQK, DV, PS = 16, 1, 192, 128, 64
T = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
prefixes = [int(p) for p in sys.argv[2:]] or [0, 8192, 32768, 131072]
SCALE = DQK ** -0.5


def bench(fn, it=5):
    fn(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it


workspace = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
print(f"flashinfer {flashinfer.__version__}; chunk {T} tokens; per layer, per rank")
for P in prefixes:
    N = P + T
    npages = (N + PS - 1) // PS
    k_cache = torch.randn(npages, PS, HKV, DQK, device=dev, dtype=torch.bfloat16)
    v_cache = torch.randn(npages, PS, HKV, DV, device=dev, dtype=torch.bfloat16)
    pages = torch.randperm(npages, device=dev).int()            # logical page i lives in physical page pages[i]
    q = torch.randn(T, H, DQK, device=dev, dtype=torch.bfloat16)

    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
    w.plan(torch.tensor([0, T], dtype=torch.int32), torch.tensor([0, npages], dtype=torch.int32), pages,
           torch.tensor([N - (npages - 1) * PS], dtype=torch.int32), H, HKV, DQK, PS, causal=True, head_dim_vo=DV,
           q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, sm_scale=SCALE)
    o_fi = w.run(q, (k_cache, v_cache))
    t_fi = bench(lambda: w.run(q, (k_cache, v_cache)))

    # the same tokens through the Triton extend kernel: prefix via token indices, chunk K/V passed directly
    k_flat, v_flat = k_cache.view(-1, HKV, DQK), v_cache.view(-1, HKV, DV)
    tok = (pages.long()[:, None] * PS + torch.arange(PS, device=dev)[None, :]).reshape(-1)[:N]
    k_ext, v_ext = k_flat[tok[P:]].contiguous(), v_flat[tok[P:]].contiguous()
    o_tr = torch.empty(T, H, DV, device=dev, dtype=torch.bfloat16)
    qo = torch.tensor([0, T], device=dev, dtype=torch.int32)
    kv_indptr = torch.tensor([0, P], device=dev, dtype=torch.int32)
    run_tr = lambda: extend_attention_fwd(q, k_ext, v_ext, o_tr, k_flat, v_flat, qo, kv_indptr, tok[:P].contiguous(),
                                          None, True, None, T, 1.0, 1.0, sm_scale=SCALE)
    t_tr = bench(run_tr)

    kk, vv = k_flat[tok].float()[:, 0], v_flat[tok].float()[:, 0]
    err_fi = err_tr = 0.0
    for r in (0, T // 3, T - 1):
        for h in (0, 7, 15):
            ref = torch.softmax((q[r, h].float() @ kk[:P + r + 1].T) * SCALE, -1) @ vv[:P + r + 1]
            err_fi = max(err_fi, ((o_fi[r, h].float() - ref).abs().max() / ref.abs().max()).item())
            err_tr = max(err_tr, ((o_tr[r, h].float() - ref).abs().max() / ref.abs().max()).item())
    flops = 2 * H * T * (P + T / 2) * (DQK + DV)
    print(f"prefix {P:7d}: FlashInfer {t_fi:8.2f} ms ({flops / t_fi / 1e9:5.1f} TFLOPS, err {err_fi:.1e}) | "
          f"Triton {t_tr:8.2f} ms ({flops / t_tr / 1e9:5.1f} TFLOPS, err {err_tr:.1e})", flush=True)
