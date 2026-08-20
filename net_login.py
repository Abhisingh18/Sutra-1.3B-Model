#!/usr/bin/env python3

import requests
import urllib3
import logging

urllib3.disable_warnings()

USERNAME="ic43815"
PASSWORD="hr5678uikjm"

LOGIN_URL="https://cc.iitm.ac.in/netaccess/account/login"
APPROVE_URL="https://cc.iitm.ac.in/netaccess/account/approve"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

session=requests.Session()

# -------------------
# LOGIN
# -------------------

logging.info("Opening login page")
session.get(LOGIN_URL,verify=False)

logging.info("Logging in")

login_payload={
    "username":USERNAME,
    "password":PASSWORD
}

session.post(LOGIN_URL,data=login_payload,verify=False)

# -------------------
# OPEN APPROVE PAGE
# -------------------

logging.info("Opening approve page")

session.get(APPROVE_URL,verify=False)

# -------------------
# AUTHORIZE MACHINE
# -------------------

logging.info("Authorizing machine")

approve_payload={
    "duration":"day",
    "use_policy":"on",
    "approveBtn":"Authorize"
}

session.post(APPROVE_URL,data=approve_payload,verify=False)

logging.info("Authorization request sent")
