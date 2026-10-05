#!/usr/bin/env python3
"""trace_tail.py <trace.json.gz> [kernel_substring]: duration percentiles for one kernel family and
per-step intervals (step boundary = first MoE kernel after a gap in MoE kernels)."""
import gzip, json, sys

path = sys.argv[1]
pat = sys.argv[2] if len(sys.argv) > 2 else "ar_kernel"
ev = json.load(gzip.open(path, "rt"))
ev = ev["traceEvents"] if isinstance(ev, dict) else ev
ks = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"), key=lambda e: e["ts"])
d = sorted(e["dur"] for e in ks if pat in e["name"])
if d:
    q = lambda f: d[min(len(d) - 1, int(f * len(d)))]
    print(f"{pat}: n={len(d)} p50={q(.5):.1f} p90={q(.9):.1f} p99={q(.99):.1f} max={d[-1]:.1f} us  "
          f"sum>1ms={sum(x for x in d if x > 1000)/1e3:.1f} ms")
moe = [e for e in ks if "marlin_moe" in e["name"]]
starts = [moe[0]["ts"]] + [b["ts"] for a, b in zip(moe, moe[1:]) if b["ts"] - (a["ts"] + a["dur"]) > 3000]
iv = [(b - a) / 1e3 for a, b in zip(starts, starts[1:])]
if iv:
    print("step intervals (ms):", " ".join(f"{x:.1f}" for x in iv))
gaps = sorted(((b["ts"] - (a["ts"] + a["dur"])) / 1e3, a["name"][:50], b["name"][:50]) for a, b in zip(ks, ks[1:]))[-5:]
print("largest GPU gaps (ms):", [(round(g, 2), x, y) for g, x, y in gaps])
