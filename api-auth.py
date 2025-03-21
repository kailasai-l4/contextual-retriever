from fastapi import Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
import os
from typing import Optional

# Create API key header schema
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Get API keys from environment variable or a secure source
# For production, consider using a more secure storage method
def get_api_keys():
    """Get list of valid API keys"""
    # Priority:
    # 1. Comma-separated list of keys in RAG_API_KEYS env var
    # 2. Single key in RAG_API_KEY env var
    # 3. Default key for development only
    
    keys_list = os.environ.get("RAG_API_KEYS", "")
    if keys_list:
        return [key.strip() for key in keys_list.split(",")]
    
    single_key = os.environ.get("RAG_API_KEY", "")
    if single_key:
        return [single_key]
    
    # Development fallback (not recommended for production)
    return ["dev_api_key_replace_in_production"]

async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    """Validate API key"""
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    valid_keys = get_api_keys()
    
    if api_key_header not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    return api_key_header

# To use this in an endpoint, add the dependency:
# @app.get("/protected-endpoint")
# async def protected_endpoint(api_key: str = Depends(get_api_key)):
#     return {"message": "This is a protected endpoint"}
