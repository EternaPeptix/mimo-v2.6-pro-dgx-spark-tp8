#!/usr/bin/env python3
"""vmwatch.py SECONDS: every 0.5 s print /proc/vmstat counters (migration/compaction/reclaim/thp) that moved."""
import sys, time
KEYS = ("pgmigrate", "compact_", "thp_", "pgscan", "pgsteal", "numa_", "kcompactd", "pgdemote", "pgpromote",
        "nr_isolated", "workingset_", "allocstall", "pgfault", "pgmajfault", "pgrefill", "drop_")
def snap():
    d = {}
    for line in open("/proc/vmstat"):
        k, v = line.split()
        if k.startswith(KEYS):
            d[k] = int(v)
    return d
end = time.time() + float(sys.argv[1]); prev = snap(); t0 = time.time()
while time.time() < end:
    time.sleep(0.5)
    cur = snap()
    delta = {k: cur[k] - prev[k] for k in cur if cur[k] != prev.get(k)}
    big = {k: v for k, v in delta.items() if abs(v) >= 50 and k not in ("pgfault",)}
    if big:
        print(f"t={time.time()-t0:5.1f} " + " ".join(f"{k}={v:+d}" for k, v in sorted(big.items())), flush=True)
    prev = cur
