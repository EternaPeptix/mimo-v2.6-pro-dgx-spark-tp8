#!/bin/bash
# launch-all.sh [extra sglang args]: (re)start all 8 ranks from the control machine, ranks 1..7 first and
# rank 0 last. Optional overrides (CTX, MAX_TOKENS, RDMA_AR=0, SKINNY=0, ...) are forwarded to every node.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/cluster.env"
pass="MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT SOCKET_IFNAME=$SOCKET_IFNAME"
pass+=" NCCL_IB_HCA=$NCCL_IB_HCA IB_GID_INDEX=$IB_GID_INDEX RDMA_AR_HCAS=$RDMA_AR_HCAS IMAGE=$IMAGE"
pass+=" MODEL_DIRS='$MODEL_DIRS'"
for v in CTX MAX_TOKENS SWA_RATIO MEM_FRAC RDMA_AR RDMA_AR_PROXY_CORE SKINNY SKINNY_FP8 FP8_MARLIN \
         VERIFY_SHARED_KV SWA_VERIFY SHARD_EH_PROJ MTP_FP8_HEAD PREFILL_POOL_GB FAKE_VERIFY_CTX_FILE; do
  [ -n "${!v:-}" ] && pass+=" $v=${!v}"
done
[ -n "${SPEC_ARGS+x}" ] && pass+=" SPEC_ARGS='$SPEC_ARGS'"
[ -n "${EXTRA_ENV:-}" ] && pass+=" EXTRA_ENV='$EXTRA_ENV'"
for r in 1 2 3 4 5 6 7 0; do
  ssh -o BatchMode=yes "${NODES[$r]}" "$pass bash -s -- $r $*" < "$HERE/mimo-tp8.sh" >/dev/null
  echo "rank $r on ${NODES[$r]}: $(ssh -n -o BatchMode=yes "${NODES[$r]}" "docker ps --filter name=mimo-tp8 --format '{{.Status}}'")"
done
echo "started; run scripts/wait-ready.sh (weights load in ~13 min, then ~1-3 min of CUDA-graph capture)"
