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
from .process_utils import find_supported_files

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
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ingest-progress/{task_id}")
async def ingest_progress(task_id: str):
    with store_lock:
        progress = ingest_progress_store.get(task_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Task not found")
        return progress

@router.post("/directory/")
async def process_directory(
    request: Request,
    directory_path: str = Body(..., embed=True),
    collection_name: str = Body(..., embed=True),
    metadata: Optional[dict] = Body(None, embed=True),
    chunk_size: Optional[int] = Body(1000, embed=True),
    overlap_size: Optional[int] = Body(100, embed=True)
):
    """
    Recursively process all supported files in the given directory and subdirectories.
    Now runs in a background thread and returns a task_id for progress tracking.
    """
    import threading
    import uuid
    task_id = str(uuid.uuid4())
    file_paths = find_supported_files(directory_path)
    if not file_paths:
        raise HTTPException(status_code=404, detail="No supported files found in directory.")
    total_files = len(file_paths)
    with store_lock:
        ingest_progress_store[task_id] = {"processed": 0, "total": total_files, "percent": 0, "done": False, "errors": [], "files": []}
    def progress_callback(file_idx, file_path, status, error=None):
        with store_lock:
            ingest_progress_store[task_id]["processed"] = file_idx + 1
            ingest_progress_store[task_id]["percent"] = round(100 * (file_idx + 1) / total_files, 1)
            ingest_progress_store[task_id]["files"].append({"file": file_path, "status": status, "error": error})
            if file_idx + 1 == total_files:
                ingest_progress_store[task_id]["done"] = True
    def run_directory_ingest():
        embedding_provider = request.app.state.embedding_provider
        storage_manager = request.app.state.qdrant_manager
        for idx, file_path in enumerate(file_paths):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                meta = dict(metadata or {})
                meta["filename"] = os.path.basename(file_path)
                file_chunker = Chunker(max_tokens=chunk_size, overlap_tokens=overlap_size)
                file_processor = Processor(file_chunker, embedding_provider, storage_manager)
                file_processor.process_document(
                    document=text,
                    metadata={"collection_name": collection_name, **meta}
                )
                progress_callback(idx, file_path, "success")
            except Exception as e:
                progress_callback(idx, file_path, "error", str(e))
    thread = threading.Thread(target=run_directory_ingest, daemon=True)
    thread.start()
    logger.info(f"[Process] Started directory ingestion thread for task {task_id} (total_files={total_files})")
    return JSONResponse(content={"status": "started", "task_id": task_id, "total_files": total_files})
