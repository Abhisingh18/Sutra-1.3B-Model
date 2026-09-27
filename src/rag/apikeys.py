"""Self-serve API keys for the public /v1 endpoints.

Anyone can request a key with no signup: POST /v1/keys returns one instantly.
That has to be paired with limits, because this runs on one home GPU shared
by the web chat, the demo, and every key holder at once -- one key with no
cap could starve everyone else.

Two limits, for two different abuses:
  - per IP,  how many keys can be minted a day (stops one visitor from
    farming unlimited keys to dodge the per-key limit below)
  - per key, how many requests it may spend a day (the actual quota)

Keys are stored HASHED (sha256) in a local JSON file, the same pattern the
download tracker uses for its history -- plain JSON, no database needed at
this scale. The raw key is shown to the caller exactly once, at creation; it
is not recoverable from the file, the same way a password would not be.
"""

import hashlib
import json
import os
import secrets
import threading
import time

PATH = os.environ.get("SUTRA_APIKEYS_PATH",
                      os.path.join(os.path.dirname(__file__), "..", "..",
                                   "deploy", "api_keys.json"))

DAILY_KEYS_PER_IP = 3          # new keys one visitor may mint per day
DAILY_REQUESTS_PER_KEY = 100   # /v1/chat calls one key may spend per day

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
        if rec["count"] >= DAILY_REQUESTS_PER_KEY:
            return False, (f"daily limit of {DAILY_REQUESTS_PER_KEY} requests "
                          "reached for this key; resets at 00:00 UTC")

        rec["count"] += 1
        data["keys"][_hash(raw_key)] = rec
        _save(data)
        return True, None


def usage(raw_key: str):
    """Requests used today and the daily cap, or None if the key is unknown."""
    data = _load()
    rec = data["keys"].get(_hash(raw_key))
    if rec is None:
        return None
    used = rec["count"] if rec["date"] == _today() else 0
    return {"used_today": used, "daily_limit": DAILY_REQUESTS_PER_KEY}
