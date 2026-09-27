"""Self-serve API keys for the public /v1 endpoints, free and paid.

Anyone can request a key with no signup: POST /v1/keys returns one instantly,
on the free tier. That has to be paired with limits, because this runs on one
home GPU shared by the web chat, the demo, and every key holder at once -- one
key with no cap could starve everyone else.

Two limits on the free tier, for two different abuses:
  - per IP,  how many keys can be minted a day (stops one visitor from
    farming unlimited keys to dodge the per-key limit below)
  - per key, how many requests it may spend a day (the actual quota)

There is deliberately NO payment gateway wired in here -- that would need a
merchant account (Razorpay, Stripe, ...) with its own KYC and bank details,
which only the account owner can set up, and the wrong integration is worse
than none. What exists instead is a manual upgrade: once payment is settled
by whatever means (UPI, bank transfer), running

    python -m src.rag.apikeys upgrade <raw_key> --limit 5000

raises that key's daily cap and marks it paid. A paid key is exempt from the
per-IP mint cap (it is no longer trying to dodge anything) and its own
DAILY_REQUESTS_PER_KEY of 100 no longer applies -- --limit is now its cap.

Keys are stored HASHED (sha256) in a local JSON file, the same pattern the
download tracker uses for its history -- plain JSON, no database needed at
this scale. The raw key is shown to the caller exactly once, at creation; it
is not recoverable from the file, the same way a password would not be.
"""

import argparse
import hashlib
import json
import os
import secrets
import threading
import time

PATH = os.environ.get("SUTRA_APIKEYS_PATH",
                      os.path.join(os.path.dirname(__file__), "..", "..",
                                   "deploy", "api_keys.json"))

DAILY_KEYS_PER_IP = 3          # new keys one visitor may mint per day (free tier)
DAILY_REQUESTS_PER_KEY = 100   # free tier: /v1/chat calls one key may spend a day

_lock = threading.Lock()


def _today():
    return time.strftime("%Y-%m-%d", time.gmtime())


def _load():
    if not os.path.exists(PATH):
        return {"keys": {}, "ip_mints": {}}
    with open(PATH) as f:
        return json.load(f)


def _save(data):
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    tmp = PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, PATH)          # atomic: readers never see a half-written file


def _hash(key):
    return hashlib.sha256(key.encode()).hexdigest()


def create_key(ip: str):
    """Mint a new key for `ip`, or refuse if that IP minted too many today.

    Returns (raw_key, None) on success, (None, error_message) on refusal.
    The raw key is returned once and never stored -- only its hash is.
    """
    today = _today()
    with _lock:
        data = _load()
        mints = data["ip_mints"].get(ip, {"date": today, "count": 0})
        if mints["date"] != today:
            mints = {"date": today, "count": 0}
        if mints["count"] >= DAILY_KEYS_PER_IP:
            return None, (f"limit of {DAILY_KEYS_PER_IP} keys per IP per day "
                          "reached; try again tomorrow")

        raw = "sk-sutra-" + secrets.token_urlsafe(24)
        data["keys"][_hash(raw)] = {
            "created": today, "date": today, "count": 0, "ip": ip,
            "tier": "free", "limit": DAILY_REQUESTS_PER_KEY,
        }
        mints["count"] += 1
        data["ip_mints"][ip] = mints
        _save(data)
        return raw, None


def check_and_spend(raw_key: str):
    """Validate a key and charge one request against its daily quota.

    Returns (True, None) if the request may proceed, (False, reason) if not.
    Charging happens here, atomically with validation, so two concurrent
    requests on a key at its last unit of quota cannot both slip through --
    the lock covers read, check and write as one step.

    The cap is the key's own `limit` field, not the module constant: a paid
    key's limit was raised by upgrade_key() and lives on the record, so a
    change to DAILY_REQUESTS_PER_KEY later can't accidentally reset it.
    """
    if not raw_key or not raw_key.startswith("sk-sutra-"):
        return False, "invalid API key"

    today = _today()
    with _lock:
        data = _load()
        rec = data["keys"].get(_hash(raw_key))
        if rec is None:
            return False, "invalid API key"

        if rec["date"] != today:
            rec["date"] = today
            rec["count"] = 0
        limit = rec.get("limit", DAILY_REQUESTS_PER_KEY)
        if rec["count"] >= limit:
            tier = rec.get("tier", "free")
            hint = (" -- request a paid key for a higher limit"
                   if tier == "free" else "")
            return False, (f"daily limit of {limit} requests reached for "
                          f"this {tier} key; resets at 00:00 UTC{hint}")

        rec["count"] += 1
        data["keys"][_hash(raw_key)] = rec
        _save(data)
        return True, None


def usage(raw_key: str):
    """Requests used today, the daily cap and tier, or None if unknown."""
    data = _load()
    rec = data["keys"].get(_hash(raw_key))
    if rec is None:
        return None
    used = rec["count"] if rec["date"] == _today() else 0
    return {"used_today": used,
            "daily_limit": rec.get("limit", DAILY_REQUESTS_PER_KEY),
            "tier": rec.get("tier", "free")}


def upgrade_key(raw_key: str, limit: int):
    """Mark a key paid and raise its daily cap. Run by hand after payment.

    There is no gateway here to trigger this automatically -- see the module
    docstring. Returns True if the key existed and was upgraded, False if the
    key is unknown (nothing is created; upgrading requires the visitor to have
    minted a free key first, which is also how their identity/IP is on file).
    """
    with _lock:
        data = _load()
        h = _hash(raw_key)
        rec = data["keys"].get(h)
        if rec is None:
            return False
        rec["tier"] = "paid"
        rec["limit"] = limit
        data["keys"][h] = rec
        _save(data)
        return True


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upgrade", help="mark a key paid, after payment is settled")
    up.add_argument("key", help="the raw sk-sutra-... key the caller was given")
    up.add_argument("--limit", type=int, required=True,
                    help="new daily request cap for this key")

    st = sub.add_parser("status", help="show a key's tier, limit and usage")
    st.add_argument("key")

    args = ap.parse_args()
    if args.cmd == "upgrade":
        if upgrade_key(args.key, args.limit):
            print(f"upgraded: paid tier, {args.limit} requests/day")
        else:
            print("no such key -- the caller must mint one with POST /v1/keys first")
    elif args.cmd == "status":
        u = usage(args.key)
        print(u if u else "no such key")


if __name__ == "__main__":
    _cli()
