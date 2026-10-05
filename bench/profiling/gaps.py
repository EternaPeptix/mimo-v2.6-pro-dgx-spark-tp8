#!/usr/bin/env python3
"""gaps.py <trace.json.gz> [min_ms]: GPU idle gaps (no kernel running) longer than min_ms, plus the
longest single kernels, with times relative to the first kernel."""
import gzip, json, sys
path = sys.argv[1]; thr = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
ev = json.load(gzip.open(path, "rt"))["traceEvents"]
k = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"), key=lambda e: e["ts"])
t0 = k[0]["ts"]; end = k[0]["ts"] + k[0]["dur"]; gaps = []
for a, b in zip(k, k[1:]):
    end = max(end, a["ts"] + a["dur"])
    if b["ts"] - end > thr * 1000:
        gaps.append(((end - t0) / 1e3, (b["ts"] - end) / 1e3, a["name"][:40], b["name"][:40]))
print(f"gaps>{thr}ms: n={len(gaps)} total={sum(g[1] for g in gaps):.1f} ms")
for g in sorted(gaps, key=lambda g: -g[1])[:6]:
    print(f"  at {g[0]:8.1f} ms gap {g[1]:7.1f} ms after [{g[2]}] before [{g[3]}]")
long = sorted(k, key=lambda e: -e["dur"])[:3]
for e in long:
    print(f"  long kernel at {(e['ts']-t0)/1e3:8.1f} ms dur {e['dur']/1e3:7.1f} ms {e['name'][:50]}")
