#!/bin/bash
# warm-flashinfer.sh: compile FlashInfer's FA2 prefill kernel for MiMo's head dims (192/128) into every node's
# persistent JIT cache, so the first long prompt after a launch does not wait for 8 ranks compiling it at once
# (~1-2 min). Run once after scripts/build-image.sh, while the server is NOT running (it needs the GPU).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/cluster.env"
WARM='
import torch, flashinfer
dev = torch.device("cuda", 0)
w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(64 << 20, dtype=torch.uint8, device=dev), "NHD", backend="fa2")
w.plan(torch.tensor([0, 100], dtype=torch.int32), torch.tensor([0, 3], dtype=torch.int32),
       torch.tensor([2, 0, 1], dtype=torch.int32, device=dev), torch.tensor([36], dtype=torch.int32),
       16, 1, 192, 64, causal=True, head_dim_vo=128, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
       sm_scale=192 ** -0.5)
o = w.run(torch.randn(100, 16, 192, device=dev, dtype=torch.bfloat16),
          (torch.randn(3, 64, 1, 192, device=dev, dtype=torch.bfloat16), torch.randn(3, 64, 1, 128, device=dev, dtype=torch.bfloat16)))
torch.cuda.synchronize(); print("ok", tuple(o.shape))
'
for n in "${NODES[@]}"; do
  ssh -o BatchMode=yes "$n" "mkdir -p \$HOME/.cache/mimo-gb10-jit/root-cache && docker run --rm -i --gpus all \
      -v \$HOME/.cache/mimo-gb10-jit/root-cache:/root/.cache $IMAGE python3 -" <<< "$WARM" 2>&1 |
    grep -E "^ok|Error" | sed "s/^/$n: /" &
done
wait
