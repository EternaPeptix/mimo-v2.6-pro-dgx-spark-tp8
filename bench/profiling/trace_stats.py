#!/usr/bin/env python3
"""trace_stats.py <trace.json[.gz]> [top_n]: GPU kernel time breakdown of a torch-profiler trace.

Groups kernels into coarse buckets (moe, fp8 gemm, bf16 gemm, attention, nccl, ...) and prints
per-bucket and per-kernel totals, plus the GPU-busy vs wall span so idle gaps are visible.
"""
import collections, gzip, json, re, sys

path = sys.argv[1]
top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
    trace = json.load(f)
events = trace["traceEvents"] if isinstance(trace, dict) else trace

kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
if not kernels:
    sys.exit("no kernel events")

BUCKETS = [
    ("nccl", r"nccl|allreduce|all_reduce|AllReduce|AllGather|all_gather"),
    ("moe_marlin", r"[Mm]arlin.*[Mm]oe|moe.*marlin|marlin_moe"),
    ("marlin_gemm", r"[Mm]arlin"),
    ("moe_other", r"moe|topk|expert|align_block|fused_moe|router|gating"),
    ("fp8_gemm", r"w8a8|fp8|_w8a8_block|per_token_group_quant|block_fp8|cutlass.*fp8|e4m3"),
    ("attention", r"attn|attention|_fwd_kernel|flash|extend|decode_|_decode|split_kv|sink"),
    ("bf16_gemm", r"gemm|gemv|cutlass|cublas|sm90_xmma|sm80_xmma|nvjet|ampere|Kernel2|splitK"),
    ("norm_act", r"norm|rms|silu|act_and_mul|gelu|layernorm"),
    ("rope", r"rope|rotary"),
    ("sampling", r"sample|argmax|softmax|logits|top_k|top_p|multinomial"),
    ("copy/elementwise", r"elementwise|copy|Memcpy|Memset|fill|cat|index|gather|scatter|reduce|vectorized"),
]

def bucket(name):
    for b, pat in BUCKETS:
        if re.search(pat, name):
            return b
    return "other"

def short(name, n=110):
    name = re.sub(r"\s+", " ", name)
    return name if len(name) <= n else name[: n - 3] + "..."

by_bucket = collections.Counter()
by_kernel = collections.Counter()
count_kernel = collections.Counter()
for e in kernels:
    d = e.get("dur", 0)
    by_bucket[bucket(e["name"])] += d
    by_kernel[e["name"]] += d
    count_kernel[e["name"]] += 1

start = min(e["ts"] for e in kernels)
end = max(e["ts"] + e.get("dur", 0) for e in kernels)
busy = sum(e.get("dur", 0) for e in kernels)
# union of intervals (streams can overlap)
iv = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in kernels)
union, cs, ce = 0.0, iv[0][0], iv[0][1]
for s, t in iv[1:]:
    if s > ce:
        union += ce - cs
        cs, ce = s, t
    else:
        ce = max(ce, t)
union += ce - cs

print(f"kernels={len(kernels)} span={(end-start)/1e3:.1f} ms  busy(sum)={busy/1e3:.1f} ms  busy(union)={union/1e3:.1f} ms  idle={(end-start-union)/1e3:.1f} ms")
print("\n== buckets ==")
for b, d in by_bucket.most_common():
    print(f"  {b:18s} {d/1e3:9.2f} ms  {100*d/busy:5.1f}%")
print(f"\n== top {top_n} kernels ==")
for k, d in by_kernel.most_common(top_n):
    print(f"  {d/1e3:9.2f} ms {100*d/busy:5.1f}% n={count_kernel[k]:6d} avg={d/count_kernel[k]:8.1f}us  {short(k)}")
