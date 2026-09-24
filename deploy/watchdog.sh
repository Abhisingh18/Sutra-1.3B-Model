#!/usr/bin/env bash
# Make sure the supervisor is running, and start it if it is not.
#
# serve.sh already restarts the model server and the tunnel. What it cannot do
# is survive its own supervisor going away -- and on 2026-09-15 the whole
# systemd user manager was killed, which took sutra.service with it and left
# the site offline for hours with nobody watching.
#
# So this runs from cron, which is a system service and keeps running when the
# user manager does not. It prefers systemd when that is available, because a
# service started there is supervised and survives logout; it falls back to
# starting serve.sh directly when it is not.
#
# serve.sh holds a lock, so a second copy started by mistake waits as a standby
# rather than fighting for the port.
#
#   */10 * * * * /speech/abhishek/Abhi-LLM-3B/deploy/watchdog.sh
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/logs_watchdog.txt"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

say() { echo "[$(date '+%F %T')] $*" >>"$LOG"; }

# Already up: nothing to do. This is the common case, so it stays silent --
# a log line every ten minutes buries the ones that matter.
if pgrep -u "$(id -u)" -f '^bash deploy/serve\.sh' >/dev/null 2>&1; then
    exit 0
fi

if systemctl --user is-system-running >/dev/null 2>&1 \
   || systemctl --user is-active sutra.service >/dev/null 2>&1; then
    say "supervisor missing; starting sutra.service"
    systemctl --user start sutra.service >>"$LOG" 2>&1 && exit 0
    say "systemctl start failed; falling back to a direct launch"
fi

say "supervisor missing; launching serve.sh directly"
cd "$ROOT" || exit 1
setsid nohup bash deploy/serve.sh </dev/null >>"$ROOT/logs_serve_nohup.txt" 2>&1 &
