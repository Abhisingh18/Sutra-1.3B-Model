#!/usr/bin/env bash
# Keep the model server and its tunnel up.
#
#   tmux new -d -s sutra 'bash deploy/serve.sh'
#   tmux attach -t sutra          # watch it
#   tmux kill-session -t sutra    # stop it
#
# Both processes are supervised: if either exits, this restarts it within ten
# seconds. The tunnel URL is written to deploy/tunnel_url.txt every time it
# changes, because a quick tunnel gets a new hostname on every start and the
# frontend has to be pointed at it.
set -u

cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="$ROOT/.venv/bin/python"
CLOUDFLARED="${CLOUDFLARED:-$ROOT/deploy/cloudflared}"
PORT="${PORT:-8000}"
LOG="$ROOT/logs_serve.txt"

# GPUs 7-10 are the ones this project is allowed to touch, and PCI_BUS_ID is
# required or the indices do not mean what nvidia-smi says they mean.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

start_server() {
  say "starting model server on :$PORT"
  "$PY" -m deploy.server --port "$PORT" >>"$ROOT/logs_server.txt" 2>&1 &
  SERVER_PID=$!
}

start_tunnel() {
  say "starting tunnel"
  : >"$ROOT/logs_tunnel.txt"
  "$CLOUDFLARED" tunnel --url "http://localhost:$PORT" \
    >>"$ROOT/logs_tunnel.txt" 2>&1 &
  TUNNEL_PID=$!

  # The hostname only appears once the tunnel is registered.
  for _ in $(seq 1 40); do
    URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' \
      "$ROOT/logs_tunnel.txt" | head -1)
    [ -n "${URL:-}" ] && break
    sleep 3
  done

  if [ -n "${URL:-}" ]; then
    echo "$URL" >"$ROOT/deploy/tunnel_url.txt"
    say "tunnel URL: $URL"
    say "  -> set NEXT_PUBLIC_API_URL to this in Vercel, then redeploy"
  else
    say "tunnel did not report a URL; will retry"
  fi
}

cleanup() {
  say "shutting down"
  kill "${SERVER_PID:-0}" "${TUNNEL_PID:-0}" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

start_server
start_tunnel

while true; do
  sleep 10

  if ! kill -0 "${SERVER_PID:-0}" 2>/dev/null; then
    say "model server died, restarting"
    start_server
  fi

  if ! kill -0 "${TUNNEL_PID:-0}" 2>/dev/null; then
    say "tunnel died, restarting"
    start_tunnel
  fi

  # A live process is not the same as a working one: the server can hang while
  # still holding its port, and the tunnel can stay up after losing its edge
  # connection. Health is what actually matters.
  if ! curl -fsS -m 10 "http://localhost:$PORT/health" >/dev/null 2>&1; then
    say "health check failed, restarting model server"
    kill "${SERVER_PID:-0}" 2>/dev/null
    sleep 3
    start_server
    sleep 40
  fi
done
