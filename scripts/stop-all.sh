#!/bin/bash
# stop-all.sh: stop the server on every node.
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/cluster.env"
for n in "${NODES[@]}"; do
  ssh -n -o BatchMode=yes "$n" 'docker stop -t 20 mimo-tp8 >/dev/null 2>&1; echo "$(hostname): stopped"' &
done
wait
