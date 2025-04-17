from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import logging
from api.api_key_auth import verify_api_key

router = APIRouter(prefix="/collections", tags=["collections"])
logger = logging.getLogger("api.collections")

class CreateCollectionRequest(BaseModel):
    collection_name: str
    vector_size: int = 1024
    distance: str = "cosine"

@router.post("/", status_code=201)
async def create_collection(request: Request, body: CreateCollectionRequest):
    await verify_api_key(request)
    qdrant_manager = request.app.state.qdrant_manager
    try:
        qdrant_manager.create_collection(
            collection_name=body.collection_name,
            vector_size=body.vector_size,
            distance=body.distance
        )
        return {"status": "ok", "collection": body.collection_name}
    except Exception as e:
        logger.error(f"Create collection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def list_collections(request: Request):
    await verify_api_key(request)
    qdrant_manager = request.app.state.qdrant_manager
    try:
        collections = qdrant_manager.list_collections()
        return {"collections": collections}
    except Exception as e:
        logger.error(f"List collections error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{collection_name}")
async def get_collection(request: Request, collection_name: str):
    await verify_api_key(request)
    qdrant_manager = request.app.state.qdrant_manager
    try:
        info = qdrant_manager.get_collection(collection_name)
        return info
    except Exception as e:
        logger.error(f"Get collection error: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{collection_name}")
async def delete_collection(request: Request, collection_name: str):
    await verify_api_key(request)
    qdrant_manager = request.app.state.qdrant_manager
    try:
        qdrant_manager.delete_collection(collection_name)
        return {"status": "deleted", "collection": collection_name}
    except Exception as e:
        logger.error(f"Delete collection error: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))
