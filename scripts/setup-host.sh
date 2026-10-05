#!/bin/bash
# setup-host.sh [fabric-cidr]: one-time host settings on a DGX Spark. Run on EVERY node with sudo:
#   sudo bash setup-host.sh 10.77.0.0/24
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
CIDR=${1:-}

# 1. Turn off proactive memory compaction (the single most important setting here).
#    The GPU's memory is pinned; kcompactd's background passes cannot move it, keep retrying, and stall
#    that node's GPU for seconds at a time - which stalls all 8 TP ranks (decode 53 -> 70-77 ms/step).
sysctl -w vm.compaction_proactiveness=0
echo "vm.compaction_proactiveness = 0" > /etc/sysctl.d/90-no-proactive-compaction.conf

# 2. Let the ranks reach each other over TCP on the fabric (torch.distributed / gloo / NCCL bootstrap).
if [ -n "$CIDR" ] && command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
  ufw allow from "$CIDR" comment "MiMo TP8 fabric"
fi

# 3. Report the rest of the expected host state (not changed by this script).
echo "--- checks"
echo "kernel cmdline iommu: $(grep -o 'iommu[^ ]*' /proc/cmdline | tr '\n' ' ') (we ran with iommu.passthrough=1)"
echo "THP mode: $(cat /sys/kernel/mm/transparent_hugepage/enabled)  (madvise is fine)"
for d in /sys/class/infiniband/*; do
  dev=$(basename "$d")
  echo "$dev: state=$(cat "$d/ports/1/state" 2>/dev/null) gid5=$(cat "$d/ports/1/gids/5" 2>/dev/null) type5=$(cat "$d/ports/1/gid_attrs/types/5" 2>/dev/null)"
done
id -nG "${SUDO_USER:-root}" | grep -qw docker && echo "docker group: ok" || echo "WARNING: ${SUDO_USER:-user} is not in the docker group"
