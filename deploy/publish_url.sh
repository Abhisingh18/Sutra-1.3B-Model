#!/usr/bin/env bash
# Publish the current tunnel hostname so the site can find it on its own.
#
#   deploy/publish_url.sh https://something.trycloudflare.com
#
# A quick tunnel gets a new hostname every time it starts, and baking that into
# NEXT_PUBLIC_API_URL means a Vercel redeploy after every restart -- which is
# the single thing most likely to leave the site pointing at a dead address.
# Writing it to a file in the repo instead lets the frontend read the current
# address at runtime and recover by itself.
#
# The token lives outside the repo, in ~/.sutra_gh_token, and is never
# committed.
set -euo pipefail

URL="${1:-}"
[ -n "$URL" ] || { echo "usage: $0 <tunnel-url>" >&2; exit 1; }

TOKEN_FILE="${SUTRA_GH_TOKEN_FILE:-$HOME/.sutra_gh_token}"
[ -f "$TOKEN_FILE" ] || { echo "no token at $TOKEN_FILE; skipping publish" >&2; exit 0; }
TOKEN=$(tr -d '\n' <"$TOKEN_FILE")

REPO="${SUTRA_REPO:-Abhisingh18/Sutra-1.3B-Model}"
PATH_IN_REPO="web/public/backend.json"
BODY="{\"url\":\"$URL\",\"updated\":\"$(date -u +%FT%TZ)\"}"
B64=$(printf '%s' "$BODY" | base64 -w0)

SHA=$(curl -fsS -H "Authorization: Bearer $TOKEN" \
  "https://api.github.com/repos/$REPO/contents/$PATH_IN_REPO" 2>/dev/null \
  | grep -oE '"sha": *"[a-f0-9]+"' | head -1 | grep -oE '[a-f0-9]{6,}' || true)

PAYLOAD="{\"message\":\"Point the site at the current tunnel\",\"content\":\"$B64\""
[ -n "$SHA" ] && PAYLOAD="$PAYLOAD,\"sha\":\"$SHA\""
PAYLOAD="$PAYLOAD}"

if curl -fsS -X PUT -H "Authorization: Bearer $TOKEN" \
  "https://api.github.com/repos/$REPO/contents/$PATH_IN_REPO" \
  -d "$PAYLOAD" >/dev/null; then
  echo "published $URL"
else
  echo "failed to publish $URL" >&2
fi
