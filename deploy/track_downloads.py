"""Record HuggingFace download counts once a day, forever.

The Hub reports a number, never a history: the API returns downloadsAllTime as
it stands today and the model page shows a rolling 30-day figure. Neither can
be asked what last week looked like. A day that is not recorded is gone.

So this appends today's totals to a file in the repo, building the series the
Hub does not keep. Run it daily; the site reads the file and draws the curve.

    python -m deploy.track_downloads              # record and publish
    python -m deploy.track_downloads --dry-run    # print, change nothing

Idempotent per day: running it twice overwrites today's entry rather than
appending a second one, so a retry after a failure cannot double-count.
"""

import argparse
import base64
import datetime
import json
import os
import urllib.request

REPOS = [
    ("model", "Abhisingh-18/Sutra-1.3B-Chat"),
    ("dataset", "Abhisingh-18/Sutra-1.3B-Data"),
]

GH_REPO = os.environ.get("SUTRA_REPO", "Abhisingh18/Sutra-1.3B-Model")
PATH_IN_REPO = "web/public/downloads.json"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "sutra-tracker"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fetch():
    """Today's totals. downloadsAllTime only appears if explicitly expanded."""
    out = {}
    for kind, repo_id in REPOS:
        base = "models" if kind == "model" else "datasets"
        d = _get(f"https://huggingface.co/api/{base}/{repo_id}"
                 f"?expand[]=downloadsAllTime&expand[]=downloads&expand[]=likes")
        out[kind] = {
            "all_time": d.get("downloadsAllTime", 0),
            "last_30d": d.get("downloads", 0),
            "likes": d.get("likes", 0),
        }
    return out


def _gh(method, url, token, payload=None):
    req = urllib.request.Request(
        url, method=method,
        data=json.dumps(payload).encode() if payload else None,
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "sutra-tracker"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    today = datetime.date.today().isoformat()
    entry = {"date": today, **fetch()}
    print(json.dumps(entry, indent=2))
    if args.dry_run:
        return

    token_file = os.environ.get("SUTRA_GH_TOKEN_FILE",
                                os.path.expanduser("~/.sutra_gh_token"))
    if not os.path.exists(token_file):
        print(f"no token at {token_file}; recorded nothing")
        return
    token = open(token_file).read().strip()

    url = f"https://api.github.com/repos/{GH_REPO}/contents/{PATH_IN_REPO}"
    history, sha = [], None
    try:
        cur = _gh("GET", url, token)
        sha = cur["sha"]
        history = json.loads(base64.b64decode(cur["content"]))
    except Exception:
        pass                     # first run: the file does not exist yet

    # Replace rather than append, so a retry cannot record the day twice.
    history = [h for h in history if h.get("date") != today] + [entry]
    history.sort(key=lambda h: h["date"])

    body = json.dumps(history, indent=1).encode()
    payload = {"message": f"Download counts for {today}",
               "content": base64.b64encode(body).decode()}
    if sha:
        payload["sha"] = sha
    _gh("PUT", url, token, payload)
    print(f"recorded {today} ({len(history)} days on file)")


if __name__ == "__main__":
    main()
