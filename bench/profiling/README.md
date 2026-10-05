# Profiling helpers

Capture a GPU trace of `N` forward steps on every rank while a request is decoding (rank 0 shown; each
rank writes `<output_dir>/<id>-TP-<rank>.trace.json.gz` inside its own container):

```bash
docker exec mimo-tp8 sh -c 'rm -rf /tmp/prof; mkdir -p /tmp/prof'      # on every node
curl -s localhost:30000/start_profile -H 'Content-Type: application/json' \
  -d '{"output_dir":"/tmp/prof","num_steps":25,"activities":["GPU"]}'   # on rank 0, during a long request
docker cp mimo-tp8:/tmp/prof ~/prof                                     # on every node, afterwards
```

- `trace_stats.py <trace> [top_n]` — GPU time by kernel family and the top kernels (count, average).
- `trace_tail.py <trace> <kernel substring>` — duration percentiles for one kernel family (e.g.
  `"ar_kernel<2>"` for the two-shot all-reduce) and the per-step intervals.
- `gaps.py <trace> [ms]` — GPU idle gaps longer than `ms`, and the longest kernels.

Compare ranks, not just rank 0: in TP8 a slow rank shows up as *long all-reduce waits on every other rank*.
Ignore each trace's first all-reduce (it waits for the last rank to start profiling), and don't trust CPU
launch gaps measured under the profiler (see `docs/PITFALLS.md`).
