#!/bin/bash
# wait-ready.sh: poll rank 0 until the server is ready; stop early (with that rank's log) if any rank exits.
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/cluster.env"
while true; do
  st=$(ssh -n -o ConnectTimeout=8 -o BatchMode=yes "${NODES[0]}" \
    'l=$(docker logs mimo-tp8 2>&1 | tail -300)
     if echo "$l" | grep -qE "fired up and ready|Uvicorn running on"; then echo READY
     elif [ "$(docker inspect -f {{.State.Running}} mimo-tp8 2>/dev/null)" != true ]; then echo EXITED; fi' 2>/dev/null)
  [ "$st" = READY ] && { echo "READY $(date +%T)"; exit 0; }
  for r in 0 1 2 3 4 5 6 7; do
    run=$(ssh -n -o ConnectTimeout=8 -o BatchMode=yes "${NODES[$r]}" 'docker inspect -f {{.State.Running}} mimo-tp8 2>/dev/null')
    if [ "$run" = "false" ]; then
      echo "rank $r (${NODES[$r]}) exited $(date +%T):"
      ssh -n -o BatchMode=yes "${NODES[$r]}" 'docker logs mimo-tp8 2>&1 | grep -vE "^\s*$|[Ww]arn" | tail -30 | cut -c1-240'
      exit 1
    fi
  done
  sleep 20
done
