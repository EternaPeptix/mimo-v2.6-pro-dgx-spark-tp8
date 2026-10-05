#!/bin/bash
# rdma-test.sh [KEY=VAL ...]: run the RDMA all-reduce correctness + latency test on all 8 nodes (one process
# per Spark, in the mimo-gb10 image). Compares against NCCL, checks bit-identical results across ranks,
# stress-tests, and prints latency inside CUDA graphs. Stop the server first (it needs a few GB per node).
#   bench/rdma-test.sh RDMA_AR_TIMING=1     # adds a per-stage latency breakdown
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../scripts/cluster.env"
EXTRA=""; for kv in "$@"; do EXTRA="$EXTRA -e $kv"; done
LOGDIR=$(mktemp -d)
for r in 7 6 5 4 3 2 1 0; do
  ssh -n -o BatchMode=yes "${NODES[$r]}" "docker rm -f artest >/dev/null 2>&1; docker run --rm --name artest --gpus all \
    --network host --ipc host --ulimit memlock=-1 --device /dev/infiniband --cap-add IPC_LOCK \
    -e RANK=$r -e WORLD_SIZE=8 -e MASTER_ADDR=$MASTER_ADDR -e MASTER_PORT=29561 \
    -e NCCL_SOCKET_IFNAME=$SOCKET_IFNAME -e GLOO_SOCKET_IFNAME=$SOCKET_IFNAME -e NCCL_IB_HCA=$NCCL_IB_HCA \
    -e NCCL_IB_GID_INDEX=$IB_GID_INDEX -e RDMA_AR_HCAS=$RDMA_AR_HCAS -e RDMA_AR_GID_INDEX=$IB_GID_INDEX $EXTRA \
    $IMAGE python3 /opt/mimo-gb10/rdma_ar/test_ar.py" > "$LOGDIR/rank$r.log" 2>&1 &
done
wait
grep -E "correct|stress|latency|allgather|arena|breakdown|graph replay|TEST_DONE|Traceback|rdma_ar rank" "$LOGDIR/rank0.log"
for r in 1 2 3 4 5 6 7; do
  grep -qE "Traceback|rdma_ar rank" "$LOGDIR/rank$r.log" && { echo "--- rank $r:"; grep -E "Traceback|rdma_ar rank|Error" "$LOGDIR/rank$r.log" | head -5; }
done
echo "logs: $LOGDIR"
