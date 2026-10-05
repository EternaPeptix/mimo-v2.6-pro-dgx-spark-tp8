#!/bin/bash
# build-image.sh: copy this repo to every node and build the image there, all nodes in parallel
# (~5 min; each node needs internet for the base image, apt and pip). Logs: /tmp/mimo-gb10-build.<node>.log
# Alternative with one build: docker save $IMAGE | ssh <node> docker load
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
source "$HERE/cluster.env"
for n in "${NODES[@]}"; do
  (
    rsync -a --delete --exclude .git --exclude scripts/cluster.env "$ROOT/" "$n:mimo-gb10-src/" &&
    ssh -n -o BatchMode=yes "$n" "cd mimo-gb10-src && docker build -f docker/Dockerfile -t $IMAGE ." \
      > "/tmp/mimo-gb10-build.$n.log" 2>&1 && echo "$n: built $IMAGE" || echo "$n: FAILED (see /tmp/mimo-gb10-build.$n.log)"
  ) &
done
wait
