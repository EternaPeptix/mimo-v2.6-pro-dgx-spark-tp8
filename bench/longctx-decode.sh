#!/bin/bash
# longctx-decode.sh: decode step time vs context length WITHOUT long prefills.
#
# Writing N to /tmp/fake_verify_ctx in every rank's container makes the 10 full-attention layers of each
# single-request verify step attend over N real slots of the KV pool instead of the request's own prefix:
# exactly the KV bytes and kernel work of an N-token context (the sliding-window and drafter layers have
# fixed windows, so they cost the same at any length beyond ~1K). Outputs are garbage while it is on, so
# acceptance collapses; the step time (= accept_len / decode tok/s) is what this measures. Convert to
# real tok/s with the acceptance you measure at short context.
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../scripts/cluster.env"
setctx() {
  for n in "${NODES[@]}"; do
    ssh -n -o BatchMode=yes "$n" "docker exec mimo-tp8 sh -c 'echo $1 > /tmp/fake_verify_ctx'" &
  done
  wait
}
bench() { ssh -o BatchMode=yes "${NODES[0]}" "python3 - --n 1 --max-tokens $1 --prompt code" < "$HERE/bench.py" 2>&1 | tail -1; }
trap 'setctx 0' EXIT
for N in ${CTXS:-0 131072 262144 524288 1048576}; do
  setctx "$N"
  bench 64 >/dev/null  # every rank re-reads the file within 32 verify steps
  for i in 1 2 3; do
    bench 512 | awk -v n="$N" '{for(i=1;i<=NF;i++){if($i ~ /^decode=/) d=substr($i,8); if($i ~ /^accept_len=/) a=substr($i,12)};
                                 printf "ctx=%8d  step=%6.1f ms  (accept=%5.3f is meaningless when ctx>0)\n", n, 1000*a/d, a}'
  done
done
setctx 0
bench 64 >/dev/null  # flush: the first request after switching back still decodes some steps in fake mode
echo "measurement switch is off again"
