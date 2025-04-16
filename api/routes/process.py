from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Body
from fastapi.responses import JSONResponse
from typing import Optional
import os
import io
import csv
import json
import logging
import threading
import uuid

from processing.chunker import Chunker
from processing.processor import Processor

router = APIRouter(prefix="/process", tags=["process"])
logger = logging.getLogger("api.process")

# Helper to read file content based on type
SUPPORTED_TYPES = ["text/plain", "text/markdown", "application/json", "text/csv"]

# In-memory progress store
ingest_progress_store = {}
store_lock = threading.Lock()

def read_file_content(upload_file: UploadFile):
    content = upload_file.file.read()
    content_type = upload_file.content_type
    filename = upload_file.filename or "uploaded"
    # Debug print/log
    print(f"[DEBUG] Received file: {filename}, content_type: {content_type}")
    # Accept text/plain, text/markdown, application/json, text/csv as before
    if content_type in ("text/plain", "text/markdown"):
        return content.decode("utf-8"), filename
    elif content_type == "application/json":
        return json.dumps(json.loads(content.decode("utf-8")), indent=2), filename
    elif content_type == "text/csv":
        return content.decode("utf-8"), filename
    # Accept application/octet-stream and empty content_type for text-based files (e.g., .md, .txt, .csv, .json)
    elif content_type in ("application/octet-stream", None, ""):
        ext = os.path.splitext(filename)[1].lower()
        if ext in [".md", ".txt", ".csv", ".json"]:
            try:
                text = content.decode("utf-8")
                if ext == ".json":
                    return json.dumps(json.loads(text), indent=2), filename
                return text, filename
            except Exception:
                raise HTTPException(status_code=415, detail=f"Could not decode file {filename} as text")
    # Otherwise, unsupported
    raise HTTPException(status_code=415, detail=f"Unsupported file type: {content_type}, filename: {filename}")

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
        embedding_provider = request.app.state.embedding_provider
        storage_manager = request.app.state.qdrant_manager
        chunker = Chunker(max_tokens=chunk_size, overlap_tokens=overlap_size)
        processor = Processor(chunker, embedding_provider, storage_manager)
        task_id = str(uuid.uuid4())
        # Chunk the document up front to get total
        chunks = chunker.chunk(text, metadata=meta)
        total_chunks = len(chunks)
        with store_lock:
            ingest_progress_store[task_id] = {"processed": 0, "total": total_chunks, "percent": 0, "done": False}
        def progress_callback(progress):
            with store_lock:
                ingest_progress_store[task_id].update(progress)
            logger.info(f"[Progress] Task {task_id}: {progress}")
        def run_ingest():
            try:
                logger.info(f"[Ingest] Background thread started for task {task_id}")
                processor.process_document(
                    document=text,
                    metadata={"collection_name": collection_name, **meta},
                    progress_callback=progress_callback
                )
                logger.info(f"[Ingest] Background thread finished for task {task_id}")
            except Exception as e:
                logger.error(f"[Ingest] Error in background thread for task {task_id}: {e}", exc_info=True)
                with store_lock:
                    ingest_progress_store[task_id]["error"] = str(e)
                    ingest_progress_store[task_id]["done"] = True
        thread = threading.Thread(target=run_ingest, daemon=True)
        thread.start()
        logger.info(f"[Process] Started ingestion thread for task {task_id} (total_chunks={total_chunks})")
        return JSONResponse(content={"status": "started", "task_id": task_id})
    except Exception as e:
        logger.error(f"Process file error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": str(e)})

@router.get("/ingest-progress/{task_id}")
async def ingest_progress(task_id: str):
    with store_lock:
        progress = ingest_progress_store.get(task_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Task not found")
        return progress
