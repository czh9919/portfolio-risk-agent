import hmac
import os
import secrets
from datetime import datetime, timedelta

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# No hardcoded fallback: an unset WEB_SECRET_KEY gets a random per-process key,
# so tokens stay unforgeable (they just won't survive a restart).
SECRET_KEY = os.environ.get("WEB_SECRET_KEY") or secrets.token_hex(32)
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def _web_pass_hash() -> str:
    return os.environ.get("WEB_PASS_HASH", "")


def verify_password(plain: str) -> bool:
    h = _web_pass_hash()
    if h:
        return _pwd.verify(plain, h)
    pw = os.environ.get("WEB_PASS", "")
    if not pw:
        # Neither WEB_PASS_HASH nor WEB_PASS configured → deny everything;
        # the old fallback let "" == "" authenticate with empty credentials.
        return False
    return hmac.compare_digest(plain, pw)


def create_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    err = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid or expired token",
                        headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise err
        return username
    except JWTError:
        raise err
