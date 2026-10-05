#!/usr/bin/env python3
"""bench_moe.py: Marlin MXFP4 MoE on one MiMo-V2 layer as a TP8 rank sees it (384 experts, H=6144,
I=256 per rank), random weights in checkpoint layout prepared by SGLang's own Marlin repack.
Times the fused kernel for 4 tokens x top-8 while forcing E distinct experts, inside a CUDA graph,
and converts to effective weight bandwidth (2.51 MB per expert per rank incl. scales)."""
import time, types
import torch
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe

dev = torch.device("cuda", 0)
E, H, I, T, K = 384, 6144, 256, 4, 8
layer = types.SimpleNamespace()
layer.w13_weight = torch.nn.Parameter(torch.randint(0, 255, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev), requires_grad=False)
layer.w2_weight = torch.nn.Parameter(torch.randint(0, 255, (E, H, I // 2), dtype=torch.uint8, device=dev), requires_grad=False)
# E8M0 scales around 2^-4 (126..128 => 2^-1..2^1 * 2^-127 offset: value byte 123 ~ 2^-4)
layer.w13_weight_scale = torch.nn.Parameter(torch.randint(120, 124, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev), requires_grad=False)
layer.w2_weight_scale = torch.nn.Parameter(torch.randint(120, 124, (E, H, I // 32), dtype=torch.uint8, device=dev), requires_grad=False)
layer.orig_dtype = torch.bfloat16
prepare_moe_mxfp4_layer_for_marlin(layer)
bytes_per_expert = (2 * I * H // 2 + 2 * I * H // 32) + (H * I // 2 + H * I // 32)
x = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
gating = torch.randn(T, E, dtype=torch.float32, device=dev)


def ids_for(distinct):
    # T x K expert ids covering exactly `distinct` experts (no repeats within a token)
    perm = torch.randperm(E)[:distinct].tolist()
    rows = []
    for t in range(T):
        row = [perm[(t * K + k) % distinct] for k in range(K)]
        if len(set(row)) < K:  # distinct < K: pad with fresh experts
            row = list(dict.fromkeys(row))
            extra = [e for e in perm if e not in row]
            row += extra[: K - len(row)]
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int32, device=dev)


def run(ids):
    w = torch.full((T, K), 1.0 / K, dtype=torch.float32, device=dev)
    return lambda: fused_marlin_moe(x, layer.w13_weight, layer.w2_weight, layer.w13_weight_scale,
                                    layer.w2_weight_scale, gating, w, ids, workspace=layer.workspace,
                                    num_bits=4, is_k_full=True)


for distinct in (8, 16, 24, 32):
    ids = ids_for(distinct)
    real = int(torch.unique(ids).numel())
    f = run(ids)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20): f()
    g.replay(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10): g.replay()
    torch.cuda.synchronize(); us = (time.perf_counter() - t) / 200 * 1e6
    gbs = real * bytes_per_expert / (us * 1e-6) / 1e9
    print(f"distinct experts={real:2d}: {us:6.1f} us per MoE layer  -> {gbs:5.1f} GB/s of expert weights", flush=True)
