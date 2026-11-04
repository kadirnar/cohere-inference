#!/bin/bash
# Start the endpoint in a tmux session: server.py on 127.0.0.1:$PORT (restarted if it exits) and, unless TUNNEL=0, a
# public Cloudflare quick tunnel to it (no account, no key; its https://*.trycloudflare.com URL is printed in
# logs/tunnel.log and changes whenever the tunnel restarts).
#
# Settings come from the environment or from ./local.env (not in git), read again at every server restart:
#   MODEL        Hub id (default Cohere-AR/cohere-transcribe-ar-sft-v9-full-s86k, gated: set HF_TOKEN) or a local copy
#   PORT         default 8086;  SESSION  tmux session name, default cohere-inference
#   CLOUDFLARED  path of the cloudflared binary, default cloudflared;  TUNNEL=0  no tunnel
#   TMPDIR       where uploads are spooled (a RAM disk keeps audio off the disk), HF_HUB_OFFLINE=1 with a local MODEL
#
# Stop with: tmux kill-session -t $SESSION
cd "$(dirname "$0")" || exit 1
[ -f local.env ] && set -a && . ./local.env && set +a
SESSION=${SESSION:-cohere-inference}
PORT=${PORT:-8086}
mkdir -p logs
tmux has-session -t "$SESSION" 2>/dev/null && { echo "already running (tmux session $SESSION)"; exit 1; }
tmux new-session -d -s "$SESSION" -n server -c "$PWD" "while true; do \
  ( set -a; [ -f local.env ] && . ./local.env; set +a; [ -n \"\$TMPDIR\" ] && mkdir -p \"\$TMPDIR\"; \
    PORT=$PORT exec .venv/bin/python server.py ) >> logs/server.log 2>&1; \
  echo \"\$(date -u '+%F %T') server exited, restarting in 10 s\" >> logs/server.log; sleep 10; done"
if [ "${TUNNEL:-1}" != 0 ]; then
  tmux new-window -t "$SESSION" -n tunnel -c "$PWD" \
    "${CLOUDFLARED:-cloudflared} tunnel --no-autoupdate --url http://127.0.0.1:$PORT >> logs/tunnel.log 2>&1"
fi
echo "started tmux session $SESSION (server on 127.0.0.1:$PORT; tunnel URL in logs/tunnel.log)"
