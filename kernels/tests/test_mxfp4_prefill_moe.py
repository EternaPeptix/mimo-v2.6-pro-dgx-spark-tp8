#!/usr/bin/env python3
"""test_mxfp4_prefill_moe.py: the prefill MoE GEMMs (patches 11, 12) against Marlin and an FP32 reference.

One MiMo-V2 MoE layer as a TP8 rank sees it (384 experts, hidden 6144, 256-wide expert slice, top-8), random
MXFP4 weights in checkpoint layout repacked by SGLang's own Marlin preparation, so both kernels read the very
same tensors. For several prefill sizes it prints the gate_up (w13) and down (w2) GEMM times of Marlin and of
the new kernel, and each kernel's max relative error against an FP32 dequantized reference on sampled rows.

usage: python3 test_mxfp4_prefill_moe.py [tokens ...]      (default: 1024 2048 4096 8192)
"""
import sys, types
import torch
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe import mxfp4_marlin_prefill as P
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import get_scalar_type
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin

dev = torch.device("cuda", 0)
torch.manual_seed(0)
E, H, I, TOPK = 384, 6144, 256, 8
sizes = [int(a) for a in sys.argv[1:]] or [1024, 2048, 4096, 8192]

# checkpoint layout: FP4 nibbles [E, N, K/2] (low nibble = even k), E8M0 scales [E, N, K/32]
w13_q = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
w2_q = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
w13_s = torch.randint(118, 123, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev)
w2_s = torch.randint(118, 123, (E, H, I // 32), dtype=torch.uint8, device=dev)
L = types.SimpleNamespace(orig_dtype=torch.bfloat16)
for name, t in (("w13_weight", w13_q), ("w2_weight", w2_q), ("w13_weight_scale", w13_s), ("w2_weight_scale", w2_s)):
    setattr(L, name, torch.nn.Parameter(t.clone(), requires_grad=False))
prepare_moe_mxfp4_layer_for_marlin(L)
st1 = get_scalar_type(4, False, L.w13_weight_scale, None)
st2 = get_scalar_type(4, False, L.w2_weight_scale, None)

FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)


def dequant(q, s, e):  # -> fp32 [N, K]
    q = q[e]
    w = torch.stack([FP4[(q & 15).long()], FP4[(q >> 4).long()]], -1).reshape(q.shape[0], -1)
    return w * torch.exp2(s[e].float() - 127).repeat_interleave(32, dim=1)


def bench(fn, it=10):
    fn(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it


def rel(out, ref):
    return ((out.float() - ref).abs().max() / ref.abs().max()).item()


print(f"{'tokens':>6} | {'Marlin w13 + w2':>22} | {'new w13 + w2':>22} | err w13 Marlin/new | err w2 Marlin/new")
for M in sizes:
    x = (torch.randn(M, H, device=dev) * 0.5).to(torch.bfloat16)
    tw, ids = torch.topk(torch.softmax(torch.randn(M, E, device=dev), -1), TOPK, dim=-1)
    ids = ids.to(torch.int32)
    tw = (tw / tw.sum(-1, keepdim=True)).float().contiguous()
    for bs in [8, 16, 32, 48, 64]:  # Marlin's own block-size choice (fused_marlin_moe)
        if M * TOPK / E / bs < 0.9:
            break
    m_sorted, m_eids, m_ntpp = moe_align_block_size(ids, bs, E)

    def marlin(a, w, s, size_m, n, k, topk, mul, st):
        out = torch.empty(size_m * topk, n, device=dev, dtype=torch.bfloat16)
        return moe_wna16_marlin_gemm(a, out, w, None, s, None, None, None, None, L.workspace, m_sorted, m_eids, m_ntpp,
                                     tw, moe_block_size=bs, top_k=topk, mul_topk_weights=mul, is_ep=False, b_q_type=st,
                                     size_m=size_m, size_n=n, size_k=k, is_k_full=True, use_atomic_add=False,
                                     use_fp32_reduce=True, is_zp_float=False)

    c1m = marlin(x, L.w13_weight, L.w13_weight_scale, M, 2 * I, H, TOPK, False, st1)
    inter = torch.empty(M * TOPK, I, device=dev, dtype=torch.bfloat16)
    silu_and_mul(c1m, inter)
    c2m = marlin(inter, L.w2_weight, L.w2_weight_scale, M * TOPK, H, I, 1, True, st2)
    tm1 = bench(lambda: marlin(x, L.w13_weight, L.w13_weight_scale, M, 2 * I, H, TOPK, False, st1))
    tm2 = bench(lambda: marlin(inter, L.w2_weight, L.w2_weight_scale, M * TOPK, H, I, 1, True, st2))

    sorted_ids, eids, ntpp = moe_align_block_size(ids, P.BLOCK_M, E)
    c1 = torch.empty_like(c1m)
    c2 = torch.empty_like(c2m)
    f1 = lambda: P.mxfp4_marlin_moe_gemm(x, L.w13_weight, L.w13_weight_scale, c1, tw, sorted_ids, eids, ntpp, TOPK, False)
    f2 = lambda: P.mxfp4_marlin_moe_gemm(inter, L.w2_weight, L.w2_weight_scale, c2, tw, sorted_ids, eids, ntpp, TOPK, True)
    t1, t2 = bench(f1), bench(f2)

    rows = torch.randperm(M * TOPK, device=dev)[:256].tolist()
    ref1 = torch.stack([x[r // TOPK].float() @ dequant(w13_q, w13_s, int(ids.view(-1)[r])).T for r in rows])
    ref2 = torch.stack([inter[r].float() @ dequant(w2_q, w2_s, int(ids.view(-1)[r])).T * tw.view(-1)[r] for r in rows])
    print(f"{M:6d} | {tm1:7.2f} + {tm2:6.2f} = {tm1 + tm2:6.2f} ms | {t1:7.2f} + {t2:6.2f} = {t1 + t2:6.2f} ms | "
          f"{rel(c1m[rows], ref1):.1e} / {rel(c1[rows], ref1):.1e} | {rel(c2m[rows], ref2):.1e} / {rel(c2[rows], ref2):.1e}",
          flush=True)
