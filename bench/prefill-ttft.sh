#!/bin/bash
# prefill-ttft.sh [ctx_tokens] [runs]: time to first token of a long prompt on rank 0 (bench.py puts a random
# nonce first, so nothing is served from the prefix cache). The first run after a launch includes Triton/JIT
# compilation and is reported separately.
# FI_AB=1: also measure with the FlashInfer full-attention prefill switched off, via the per-rank flag file
# /tmp/fi_prefill_off (patch 05), then switch it back on.
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../scripts/cluster.env"
CTX=${1:-32768}; RUNS=${2:-3}
fi_switch() {  # fi_switch on|off
  local cmd="touch /tmp/fi_prefill_off"; [ "$1" = on ] && cmd="rm -f /tmp/fi_prefill_off"
  for n in "${NODES[@]}"; do ssh -n -o BatchMode=yes "$n" "docker exec mimo-tp8 $cmd" & done
  wait
}
ttft() { ssh -o BatchMode=yes "${NODES[0]}" "python3 - --n 1 --max-tokens 8 --ctx-tokens $CTX --prompt code" < "$HERE/bench.py" 2>&1 |
         tail -1 | grep -oE "in=[0-9]+|ttft=[0-9.]+s" | tr '\n' ' '; }
modes=(on); [ "${FI_AB:-0}" = 1 ] && modes=(on off)
trap 'fi_switch on' EXIT
for m in "${modes[@]}"; do
  fi_switch "$m"
  echo "flashinfer_prefill=$m  warm-up: $(ttft)"
  for i in $(seq 1 "$RUNS"); do echo "flashinfer_prefill=$m  run $i: $(ttft)"; done
done
