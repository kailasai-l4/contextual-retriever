from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Optional
import os
import io
import csv
import json
import logging

from processing.chunker import Chunker
from processing.processor import Processor

router = APIRouter(prefix="/process", tags=["process"])
logger = logging.getLogger("api.process")

# Helper to read file content based on type
SUPPORTED_TYPES = ["text/plain", "text/markdown", "application/json", "text/csv"]

def read_file_content(upload_file: UploadFile):
    content = upload_file.file.read()
    content_type = upload_file.content_type
    filename = upload_file.filename or "uploaded"
    if content_type in ("text/plain", "text/markdown"):
        return content.decode("utf-8"), filename
    elif content_type == "application/json":
        return json.dumps(json.loads(content.decode("utf-8")), indent=2), filename
    elif content_type == "text/csv":
        return content.decode("utf-8"), filename
    else:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {content_type}")

@router.post("/")
async def process_file(
    request: Request,
    file: UploadFile = File(...),
    collection_name: str = Form(...),
    metadata: Optional[str] = Form(None),
    chunk_size: Optional[int] = Form(1000),
    overlap_size: Optional[int] = Form(100)
):
    try:
        text, filename = read_file_content(file)
        meta = json.loads(metadata) if metadata else {}
        meta["filename"] = filename
        # Get providers and managers from app state
        embedding_provider = request.app.state.embedding_provider
        storage_manager = request.app.state.qdrant_manager
        chunker = Chunker(max_tokens=chunk_size, overlap_tokens=overlap_size)
        processor = Processor(chunker, embedding_provider, storage_manager)
        result = processor.process_document(
            document=text,
            metadata={"collection_name": collection_name, **meta}
        )
        return JSONResponse(content={"status": "ok", "result": result})
    except Exception as e:
        logger.error(f"Process file error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
