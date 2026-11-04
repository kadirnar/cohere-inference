#!/bin/bash
# arena_level_effect.py for local model copies: a second instance of the service on port 8087 per model, then stopped.
#   cd $ARENA_REPO && bash /path/to/norm_study/run_level_effect_local.sh KEY=/path/to/model [KEY=/path ...]
ROOT=$(cd "$(dirname "$0")/.." && pwd)
for spec in "$@"; do
  key=${spec%%=*}; path=${spec#*=}
  echo "== $(date -u '+%T UTC') $key"
  ( cd "$ROOT" && MODEL=$path PORT=8087 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=/dev/shm/s86k-api-tmp \
    exec .venv/bin/python server.py > logs/level_effect_server.log 2>&1 ) &
  pid=$!
  for i in $(seq 1 90); do sleep 2; curl -s -m 3 http://127.0.0.1:8087/health | grep -q '"ok"' && break; done
  /root/arena-venv/bin/python "$ROOT/norm_study/arena_level_effect.py" "$key" http://127.0.0.1:8087 2>&1 | grep -v Warning
  kill $pid; wait $pid 2>/dev/null
done
echo "== $(date -u '+%T UTC') LEVEL_EFFECT_DONE"
