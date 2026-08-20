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

# PCI_BUS_ID is required or the indices do not mean what nvidia-smi says they
# mean. The card is a default, not a fixture: 8 filled up with someone else's
# 46 GB job and every restart then died on OOM, so override it when the box is
# busy rather than letting the supervisor loop.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Search key, kept outside the repo. Its absence is not an error -- the server
# simply starts without the web toggle, and /health reports web:false so the
# frontend hides the control rather than offering one that does nothing.
TAVILY_KEY_FILE="${TAVILY_KEY_FILE:-$HOME/.sutra_tavily_key}"
WEB_FLAG=""
if [ -s "$TAVILY_KEY_FILE" ]; then
  export TAVILY_API_KEY="$(tr -d '\n' <"$TAVILY_KEY_FILE")"
  WEB_FLAG="--web"
fi

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

start_server() {
  # Claim the port first. Without this an orphan from an earlier run keeps
  # holding it, every new server exits with EADDRINUSE, and the supervisor
  # restarts it forever while the health check passes against the orphan.
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "$PORT/tcp" >/dev/null 2>&1 || true
  else
    pkill -f "deploy.server --port $PORT" 2>/dev/null || true
  fi
  sleep 2

  say "starting model server on :$PORT"
  "$PY" -m deploy.server --port "$PORT" $WEB_FLAG >>"$ROOT/logs_server.txt" 2>&1 &
  SERVER_PID=$!
  # Loading 5.3 GB of weights takes the better part of a minute. Health checks
  # must not run before then, or every start is killed mid-load and the
  # supervisor restarts forever without the server once coming up.
  READY_AT=$(( $(date +%s) + 90 ))
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
    # Publish it so the site picks the new address up on its own. Without this
    # every tunnel restart needs a Vercel redeploy, which is the surest way to
    # end up pointing at a dead hostname.
    bash "$ROOT/deploy/publish_url.sh" "$URL" 2>&1 | tee -a "$LOG"
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

FAILS=0
TFAILS=0

start_server
start_tunnel

while true; do
  sleep 10

  if ! kill -0 "${SERVER_PID:-0}" 2>/dev/null; then
    FAILS=$((FAILS + 1))
    say "model server died (restart #$FAILS)"
    if [ "$FAILS" -ge 5 ]; then
      say "five restarts in a row -- backing off 60s; check logs_server.txt"
      sleep 60
      FAILS=0
    fi
    start_server
    sleep 30
  else
    FAILS=0
  fi

  if ! kill -0 "${TUNNEL_PID:-0}" 2>/dev/null; then
    say "tunnel died, restarting"
    start_tunnel
  fi

  # A live process is not the same as a working one: the server can hang while
  # still holding its port, and the tunnel can stay up after losing its edge
  # connection. Health is what actually matters.
  # Still inside the startup grace period; nothing to check yet.
  if [ "$(date +%s)" -lt "${READY_AT:-0}" ]; then
    continue
  fi

  if ! curl -fsS -m 10 "http://localhost:$PORT/health" >/dev/null 2>&1; then
    say "health check failed, restarting model server"
    kill "${SERVER_PID:-0}" 2>/dev/null
    sleep 3
    start_server
    continue
  fi

  # The server can be healthy while the tunnel is not. A quick tunnel
  # sometimes hangs after its preflight without ever registering, leaving a
  # live process and a hostname that resolves to nothing -- which looks
  # identical to "model offline" from the browser.
  PUBLIC=$(cat "$ROOT/deploy/tunnel_url.txt" 2>/dev/null || true)
  if [ -n "$PUBLIC" ]; then
    if ! curl -fsS -m 20 "$PUBLIC/health" >/dev/null 2>&1; then
      TFAILS=$((TFAILS + 1))
      say "tunnel unreachable at $PUBLIC ($TFAILS/2)"
      if [ "$TFAILS" -ge 2 ]; then
        say "restarting tunnel"
        kill "${TUNNEL_PID:-0}" 2>/dev/null
        sleep 3
        start_tunnel
        TFAILS=0
      fi
    else
      TFAILS=0
    fi
  fi
done
