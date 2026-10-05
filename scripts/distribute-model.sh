#!/bin/bash
# distribute-model.sh <src-node> <dst-fabric-ip>...: copy the 566 GB checkpoint from one Spark to others over
# the RoCE fabric's TCP path with parallel rsync streams (~1 GB/s per destination in our setup).
# Requires node-to-node ssh (it uses agent forwarding: ssh -A). Resumable: rerun to finish an interrupted copy.
#   scripts/distribute-model.sh spark-1 10.77.0.21 10.77.0.31 ...
set -euo pipefail
SRC=${1:?usage: distribute-model.sh <src-node> <dst-fabric-ip>...}; shift
STREAMS=${STREAMS:-6}
DIR=${MODEL_SUBDIR:-models/MiMo-V2.6-Pro-MOPD}   # relative to $HOME on every node
for DEST in "$@"; do
  ssh -A -o BatchMode=yes "$SRC" "DEST=$DEST STREAMS=$STREAMS DIR=$DIR bash -s" <<'EOF'
set -uo pipefail
cd ~/"$DIR" || exit 1
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -c aes128-gcm@openssh.com -o Compression=no"
$SSH "$DEST" "mkdir -p ~/$DIR"
list=$(mktemp); find . -type f -not -path "./.cache/*" -printf "%s %P\n" | sort -rn | awk '{print $2}' > "$list"
split -n r/"$STREAMS" "$list" "$list.part."
t0=$(date +%s)
for p in "$list".part.*; do rsync -a --size-only --files-from="$p" -e "$SSH" ./ "$DEST:$DIR/" & done
wait
echo "$DEST: done in $(( $(date +%s) - t0 )) s, $($SSH "$DEST" "find ~/$DIR -type f -not -path '*/.cache/*' | wc -l") files"
rm -f "$list" "$list".part.*
EOF
done
