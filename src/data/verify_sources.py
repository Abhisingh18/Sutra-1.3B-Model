"""Verify every source in the mixture before spending days on tokenization.

    python -m src.data.verify_sources          # API check only, no downloads
    python -m src.data.verify_sources --live   # also pull real rows

HuggingFace dataset ids, config names, split names and column names drift, and a
wrong string does not fail fast -- it fails after the streaming run has already
been going for hours. Two minutes here saves that.

The API check needs only `requests`/urllib, not `datasets`, so it works before
anything is installed.
"""

import argparse
import json
import sys
import urllib.request
import urllib.parse

API = "https://datasets-server.huggingface.co"
HUB = "https://huggingface.co/api/datasets"

from .mixture import MIXTURE


def _get(url, timeout=30):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return json.load(e)
        except Exception:
            return {"error": f"HTTP {e.code}"}
    except Exception as e:
        return {"error": str(e)[:80]}


def check_api(src):
    """Confirm dataset/config/split/field without downloading data."""
    meta = _get(f"{HUB}/{src.name}")
    if "error" in meta:
        return False, f"dataset not found ({meta['error']})"
    gated = meta.get("gated")
    gate_note = f" [gated={gated}]" if gated else ""

    q = urllib.parse.urlencode({
        "dataset": src.name, "config": src.config or "default", "split": src.split})
    rows = _get(f"{API}/first-rows?{q}")
    if "error" in rows:
        return False, f"config/split rejected: {str(rows['error'])[:60]}{gate_note}"

    fields = [f["name"] for f in rows.get("features", [])]
    if src.text_field not in fields:
        return False, f"field '{src.text_field}' missing; has {fields}{gate_note}"

    # Confirm the field actually carries text, not a nested structure.
    sample = rows.get("rows", [{}])[0].get("row", {}).get(src.text_field)
    if not isinstance(sample, str):
        return False, f"field '{src.text_field}' is {type(sample).__name__}, not str"

    return True, f"ok, sample {len(sample)} chars{gate_note}"


def check_live(src, n=3):
    """Actually open the stream and pull rows -- the real test."""
    from datasets import load_dataset
    try:
        ds = load_dataset(src.name, src.config, split=src.split, streaming=True)
        it = iter(ds)
        chars = 0
        for _ in range(n):
            row = next(it)
            if src.text_field not in row:
                return False, f"field '{src.text_field}' missing from row"
            chars += len(row[src.text_field] or "")
        return True, f"ok, {n} rows, {chars} chars"
    except Exception as e:
        return False, str(e)[:90]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="open real streams (needs `datasets` installed)")
    args = ap.parse_args()

    print(f"verifying {len(MIXTURE)} sources "
          f"({'live streams' if args.live else 'API only'})\n")

    failures = []
    for s in MIXTURE:
        label = f"{s.name}:{s.config}:{s.split}"
        ok, msg = check_live(s, ) if args.live else check_api(s)
        print(f"  [{'OK ' if ok else 'FAIL'}] {label:60s} {msg}")
        if not ok:
            failures.append((label, msg))

    print()
    if failures:
        print(f"{len(failures)} source(s) failed -- fix src/data/mixture.py "
              f"before running prepare.py:")
        for label, msg in failures:
            print(f"  {label}: {msg}")
        sys.exit(1)
    print("all sources verified")


if __name__ == "__main__":
    main()
