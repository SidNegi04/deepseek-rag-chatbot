import json
import os

import firebase_admin
from firebase_admin import auth as fb_auth, credentials
from fastapi import Header, HTTPException


def _init_firebase():
    try:
        firebase_admin.get_app()
        return
    except ValueError:
        pass
    raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT")  # used on Render
    if raw:
        cred = credentials.Certificate(json.loads(raw))
    else:
        cred = credentials.Certificate(os.environ["FIREBASE_KEY_PATH"])  # used locally
    firebase_admin.initialize_app(cred)


_init_firebase()


def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    try:
        return fb_auth.verify_id_token(authorization.split(" ", 1)[1])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
