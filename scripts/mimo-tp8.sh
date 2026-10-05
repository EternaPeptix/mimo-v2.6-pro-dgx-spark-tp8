#!/bin/bash
# mimo-tp8.sh <rank> [extra sglang args]: start one TP rank of MiMo-V2.6-Pro-MOPD on this Spark.
# Runs ON the node. launch-all.sh streams it over ssh and passes the cluster.env values as env vars.
set -euo pipefail
RANK=${1:?usage: mimo-tp8.sh <rank 0-7> [extra sglang args]}; shift
: "${MASTER_ADDR:?}" "${SOCKET_IFNAME:?}" "${NCCL_IB_HCA:?}" "${RDMA_AR_HCAS:?}"
MASTER_PORT=${MASTER_PORT:-20000}
IB_GID_INDEX=${IB_GID_INDEX:-5}
IMAGE=${IMAGE:-mimo-gb10:v0.5.21}

MODEL_DIR=""
IFS=: read -ra dirs <<< "${MODEL_DIRS:-~/models/MiMo-V2.6-Pro-MOPD}"
for d in "${dirs[@]}"; do
  d=${d/#\~/$HOME}
  [ -f "$d/config.json" ] && { MODEL_DIR=$d; break; }
done
[ -n "$MODEL_DIR" ] || { echo "model not found in MODEL_DIRS=${MODEL_DIRS:-}" >&2; exit 1; }

CTX=${CTX:-1048576}                # context length served
MAX_TOKENS=${MAX_TOKENS:-1114112}  # full-attention KV pool: one 1M-token request + 64K slack
SWA_RATIO=${SWA_RATIO:-0.06}       # the 60 sliding-window layers only keep window(128)+chunk per request
MEM_FRAC=${MEM_FRAC:-0.85}
SPEC_ARGS=${SPEC_ARGS:---speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-multi-layer-eagle}

# Feature switches (1 = on, 0 = stock behaviour). docs/OPTIMIZATIONS.md explains each one.
ENVS=(
  -e "SGLANG_RDMA_AR=${RDMA_AR:-1}"                    # one/two-shot RDMA all-reduce instead of NCCL ring
  -e "RDMA_AR_HCAS=$RDMA_AR_HCAS" -e "RDMA_AR_GID_INDEX=$IB_GID_INDEX"
  -e "RDMA_AR_PROXY_CORE=${RDMA_AR_PROXY_CORE:-19}"   # a Cortex-X925 core on GB10 (5-9, 15-19)
  -e "SGLANG_SKINNY_GEMM=${SKINNY:-1}"                 # decode-sized bf16 GEMM (o_proj)
  -e "SGLANG_SKINNY_FP8=${SKINNY_FP8:-1}"              # decode-sized FP8 block-scaled GEMM (qkv, dense MLP, MTP)
  -e "SGLANG_FORCE_FP8_MARLIN=${FP8_MARLIN:-0}"
  -e "SGLANG_VERIFY_SHARED_KV=${VERIFY_SHARED_KV:-1}"  # grouped-head split-KV verify attention (long context)
  -e "SGLANG_SWA_VERIFY=${SWA_VERIFY:-1}"              # fused sliding-window+sink verify attention
  -e "SGLANG_MTP_SHARD_EH_PROJ=${SHARD_EH_PROJ:-1}"    # shard the drafter's 151 MB eh_proj over TP
  -e "SGLANG_MTP_FP8_HEAD=${MTP_FP8_HEAD:-1}"          # FP8 copy of the drafter-only lm_head
  -e "SGLANG_PREFILL_POOL_GB=${PREFILL_POOL_GB:-6}"    # reserve prefill activations before RAM fragments
  -e "SGLANG_FAKE_VERIFY_CTX_FILE=${FAKE_VERIFY_CTX_FILE:-1}"  # bench/longctx-decode.sh switch (inert by default)
  -e "NCCL_SOCKET_IFNAME=$SOCKET_IFNAME" -e "GLOO_SOCKET_IFNAME=$SOCKET_IFNAME"
  -e "NCCL_IB_HCA=$NCCL_IB_HCA" -e "NCCL_IB_GID_INDEX=$IB_GID_INDEX" -e NCCL_DEBUG=WARN
)
for kv in ${EXTRA_ENV:-}; do ENVS+=(-e "$kv"); done

# Persistent JIT caches (Triton kernels, FlashInfer/TVM-FFI JIT): the container is recreated on every
# launch, so without these each start recompiles everything.
JIT=$HOME/.cache/mimo-gb10-jit
mkdir -p "$JIT/triton" "$JIT/root-cache"

docker rm -f mimo-tp8 >/dev/null 2>&1 || true
exec docker run -d --name mimo-tp8 --gpus all --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 --device /dev/infiniband --cap-add IPC_LOCK \
  -v "$MODEL_DIR":/model:ro -v "$JIT/triton":/root/.triton -v "$JIT/root-cache":/root/.cache \
  "${ENVS[@]}" "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path /model --served-model-name mimo-v2.6-pro --trust-remote-code \
    --tp 8 --nnodes 8 --node-rank "$RANK" --dist-init-addr "$MASTER_ADDR:$MASTER_PORT" \
    --moe-runner-backend marlin --attention-backend triton \
    --context-length "$CTX" --page-size 64 --max-total-tokens "$MAX_TOKENS" --swa-full-tokens-ratio "$SWA_RATIO" \
    --mem-fraction-static "$MEM_FRAC" --max-running-requests 4 --chunked-prefill-size 8192 \
    --weight-loader-drop-cache-after-load --sleep-on-idle --scheduler-recv-interval 8 \
    $SPEC_ARGS \
    --reasoning-parser mimo --tool-call-parser mimo \
    --host 127.0.0.1 --port 30000 "$@"
