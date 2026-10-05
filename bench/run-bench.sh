#!/bin/bash
# run-bench.sh [rounds]: single-stream decode benchmark on rank 0 (code / math / essay prompts, 512 tokens,
# greedy). Prints decode tok/s, MTP acceptance (tokens per verify step) and the verify step time.
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../scripts/cluster.env"
ROUNDS=${1:-3}
for i in $(seq 1 "$ROUNDS"); do
  for p in code math essay; do
    ssh -o BatchMode=yes "${NODES[0]}" "python3 - --n 1 --max-tokens 512 --prompt $p" < "$HERE/bench.py" 2>&1 | tail -1 |
      awk '{for(i=1;i<=NF;i++){if($i ~ /^decode=/) d=substr($i,8); if($i ~ /^accept_len=/) a=substr($i,12)};
            printf "%-6s %6.2f tok/s  accept=%5.3f  step=%5.1f ms\n", $1, d, a, 1000*a/d}'
  done
done
