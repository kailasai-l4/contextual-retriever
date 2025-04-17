from fastapi import Request, HTTPException, status
from fastapi.security.utils import get_authorization_scheme_param
import os

API_KEY = os.getenv("API_KEY")
API_KEY_HEADER = "X-API-Key"

async def verify_api_key(request: Request):
    api_key = request.headers.get(API_KEY_HEADER)
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
