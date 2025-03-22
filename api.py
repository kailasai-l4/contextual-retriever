#!/usr/bin/env python3
"""
RAG Content Retriever API
-------------------------
A REST API service exposing the RAG Content Retriever functionality.
"""

import os
import sys
import time
import json
from typing import Dict, List, Optional, Any
import logging
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile, Body, Query, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
from datetime import datetime

# Load .env file
load_dotenv()

# Import our modules
from config import get_config
from content_processor import ContentProcessor
from advanced_retriever import AdvancedRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_api")

# Initialize the API
app = FastAPI(
    title="RAG Content Retriever API",
    description="API for retrieving and processing content using RAG (Retrieval Augmented Generation) techniques",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify for production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models for request/response ---

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: Optional[int] = Field(20, description="Maximum number of results to return")
    use_optimized_retrieval: Optional[bool] = Field(True, description="Whether to use optimized retrieval strategy")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filter criteria (e.g., {\"source_type\": \".md\"})")

class ProcessRequest(BaseModel):
    directory_path: str = Field(..., description="Path to directory containing documents")
    recursive: Optional[bool] = Field(True, description="Process directories recursively")
    file_types: Optional[List[str]] = Field(None, description="File extensions to process (e.g., [\".md\", \".json\"])")

class ApiResponse(BaseModel):
    success: bool
    data: Any
    message: Optional[str] = None
    duration_ms: float

# --- Dependencies ---

def get_api_components():
    """Creates and returns processor and retriever components with configuration"""
    config = get_config()

    # Get API keys from environment or config
    jina_key = os.environ.get("JINA_API_KEY") or config.get('jina', 'api_key')
    gemini_key = os.environ.get("GEMINI_API_KEY") or config.get('gemini', 'api_key')
    qdrant_url = config.get('qdrant', 'url')
    qdrant_port = config.get('qdrant', 'port')

    # Check for required API keys
    if not jina_key:
        raise ValueError("Jina API key not provided in config or environment")

    if not gemini_key:
        raise ValueError("Gemini API key not provided in config or environment")

    # Initialize processor and retriever
    processor = ContentProcessor(
        jina_api_key=jina_key,
        gemini_api_key=gemini_key,
        qdrant_url=qdrant_url,
        qdrant_port=qdrant_port
    )

    retriever = AdvancedRetriever(
        jina_api_key=jina_key,
        gemini_api_key=gemini_key,
        qdrant_url=qdrant_url,
        qdrant_port=qdrant_port
    )

    return processor, retriever

API_KEY = os.environ.get("API_KEY") or get_config().get('api', 'api_key')
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

# --- API Endpoints ---

@app.get("/")
async def root(api_key: str = Depends(get_api_key)):
    """Root endpoint providing information about the API"""
    return {
        "name": "RAG Content Retriever API",
        "version": "1.0.0",
        "description": "API for retrieving and processing content using RAG techniques",
        "endpoints": [
            {"path": "/search", "method": "POST", "description": "Search for content"},
            {"path": "/process", "method": "POST", "description": "Process documents into vector database"},
            {"path": "/stats", "method": "GET", "description": "Get statistics about the vector database"},
            {"path": "/sources/{source_id}", "method": "GET", "description": "Get all content from a specific source"},
            {"path": "/dashboard", "method": "GET", "description": "Get a comprehensive dashboard view of the database statistics"}
        ]
    }

@app.post("/search", response_model=ApiResponse)
async def search_content(request: SearchRequest, api_key: str = Depends(get_api_key)):
    """
    Search for content based on the provided query

    Parameters:
    - query: Search query
    - limit: Maximum number of results to return
    - use_optimized_retrieval: Whether to use optimized retrieval strategy
    - filters: Filter criteria (e.g., {\"source_type\": \".md\"})

    Returns:
    - success: Whether the request was successful
    - data: Search results
    - message: Additional information
    - duration_ms: Request duration in milliseconds
    """
    start_time = time.time()

    try:
        _, retriever = get_api_components()

        if request.filters:
            results = retriever.filter_search(
                query=request.query,
                filters=request.filters,
                limit=request.limit,
                use_optimized_retrieval=request.use_optimized_retrieval
            )
        else:
            results = retriever.search(
                query=request.query,
                limit=request.limit,
                use_optimized_retrieval=request.use_optimized_retrieval
            )

        duration_ms = (time.time() - start_time) * 1000
        return ApiResponse(
            success=True,
            data={
                "query": request.query,
                "count": len(results),
                "results": results
            },
            message=f"Found {len(results)} results",
            duration_ms=duration_ms
        )

    except Exception as e:
        logger.error(f"Error during search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", response_model=ApiResponse)
async def process_documents(request: ProcessRequest, background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)):
    """
    Process documents into the vector database

    This endpoint starts document processing in the background and returns immediately.

    Parameters:
    - directory_path: Path to directory containing documents
    - recursive: Process directories recursively
    - file_types: File extensions to process (e.g., [".md", ".json"])

    Returns:
    - success: Whether the request was started successfully
    - data: Request details
    - message: Additional information
    """
    start_time = time.time()

    try:
        processor, _ = get_api_components()

        # Define the background task
        def process_task(path, recursive, file_types):
            try:
                logger.info(f"Starting background processing of {path}")

                # Convert file types to proper format
                if file_types:
                    file_types = [ft if ft.startswith('.') else f'.{ft}' for ft in file_types]

                # Process the directory
                processor.process_directory(
                    directory_path=path,
                    recursive=recursive,
                    file_types=file_types
                )

                logger.info(f"Completed background processing of {path}")
            except Exception as e:
                logger.error(f"Error during background processing: {str(e)}", exc_info=True)

        # Add the task to background tasks
        background_tasks.add_task(
            process_task,
            request.directory_path,
            request.recursive,
            request.file_types
        )

        duration_ms = (time.time() - start_time) * 1000
        return ApiResponse(
            success=True,
            data={
                "directory_path": request.directory_path,
                "recursive": request.recursive,
                "file_types": request.file_types
            },
            message="Document processing started in the background",
            duration_ms=duration_ms
        )

    except Exception as e:
        logger.error(f"Error starting document processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=ApiResponse)
async def get_stats(api_key: str = Depends(get_api_key)):
    """
    Get statistics about the vector database

    Returns:
    - success: Whether the request was successful
    - data: Database statistics
    - duration_ms: Request duration in milliseconds
    """
    start_time = time.time()

    try:
        processor, _ = get_api_components()
        raw_stats = processor.get_collection_stats()
        
        # Format statistics in a more human-readable way
        stats = {
            "summary": {
                "collection_name": raw_stats.get("collection_name", "content_library"),
                "status": raw_stats.get("status", "unknown"),
                "total_chunks": raw_stats.get("points_count", 0),
                "unique_documents": raw_stats.get("unique_sources", 0),
                "avg_chunks_per_document": raw_stats.get("avg_chunks_per_source", 0)
            },
            "database_info": {
                "vectors_count": raw_stats.get("vectors_count"),
                "segments_count": raw_stats.get("segments_count"),
                "vector_dimension": raw_stats.get("vector_dimension", 1024)
            },
            "content": {
                "file_type_distribution": raw_stats.get("file_types", {}),
                "recently_processed": raw_stats.get("recently_processed", [])
            },
            "raw_statistics": raw_stats  # Include original stats for debugging
        }

        duration_ms = (time.time() - start_time) * 1000
        return ApiResponse(
            success=True,
            data=stats,
            message="Database statistics retrieved successfully",
            duration_ms=duration_ms
        )

    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources/{source_id}", response_model=ApiResponse)
async def get_source_content(source_id: str, api_key: str = Depends(get_api_key)):
    """
    Get all content from a specific source

    Parameters:
    - source_id: Source ID to retrieve

    Returns:
    - success: Whether the request was successful
    - data: Source content
    - duration_ms: Request duration in milliseconds
    """
    start_time = time.time()

    try:
        _, retriever = get_api_components()
        chunks = retriever.get_source_content(source_id)

        duration_ms = (time.time() - start_time) * 1000
        return ApiResponse(
            success=True,
            data={
                "source_id": source_id,
                "chunks": chunks,
                "count": len(chunks)
            },
            duration_ms=duration_ms
        )

    except Exception as e:
        logger.error(f"Error getting source content: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=ApiResponse)
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...), process_now: bool = Query(True), api_key: str = Depends(get_api_key)):
    """
    Upload and optionally process a file

    Parameters:
    - file: File to upload
    - process_now: Whether to process the file immediately

    Returns:
    - success: Whether the upload was successful
    - data: Upload details
    - message: Additional information
    """
    start_time = time.time()

    try:
        processor, _ = get_api_components()

        # Create uploads directory if it doesn't exist
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)

        # Save the file
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Process the file if requested
        if process_now:
            def process_file_task(path):
                try:
                    logger.info(f"Processing uploaded file: {path}")
                    processor.process_file(path)
                    logger.info(f"Completed processing of uploaded file: {path}")
                except Exception as e:
                    logger.error(f"Error processing uploaded file: {str(e)}", exc_info=True)

            background_tasks.add_task(process_file_task, file_path)
            message = "File uploaded and processing started in the background"
        else:
            message = "File uploaded successfully"

        duration_ms = (time.time() - start_time) * 1000
        return ApiResponse(
            success=True,
            data={
                "filename": file.filename,
                "path": file_path,
                "size": file.size,
                "content_type": file.content_type,
                "processed": process_now
            },
            message=message,
            duration_ms=duration_ms
        )

    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimized_retrieval", response_model=ApiResponse)
async def optimized_retrieval(topic: str = Body(...), limit: int = Body(20), api_key: str = Depends(get_api_key)):
    """
    Retrieve optimized content for a specific topic

    Parameters:
    - topic: Topic to retrieve content for
    - limit: Maximum number of chunks to retrieve

    Returns:
    - success: Whether the request was successful
    - data: Optimized content
    - duration_ms: Request duration in milliseconds
    """
    start_time = time.time()

    try:
        _, retriever = get_api_components()
        results = retriever.retrieve_optimized_content(topic, limit)

        duration_ms = (time.time() - start_time) * 1000
        return ApiResponse(
            success=True,
            data=results,
            duration_ms=duration_ms
        )

    except Exception as e:
        logger.error(f"Error during optimized retrieval: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard", response_model=ApiResponse)
async def get_dashboard(api_key: str = Depends(get_api_key)):
    """
    Get a comprehensive dashboard view of the database statistics
    
    Returns:
    - success: Whether the request was successful
    - data: Dashboard statistics with formatting hints for UI
    - duration_ms: Request duration in milliseconds
    """
    start_time = time.time()

    try:
        processor, _ = get_api_components()
        raw_stats = processor.get_collection_stats()
        
        # Calculate additional metrics
        total_points = raw_stats.get("points_count", 0)
        total_sources = raw_stats.get("unique_sources", 0)
        avg_chunks = 0 if total_sources == 0 else round(total_points / total_sources, 1)
        
        # Prepare dashboard data with formatting hints
        dashboard = {
            "metrics": [
                {
                    "name": "Total Documents",
                    "value": total_sources,
                    "format": "number",
                    "icon": "document",
                    "description": "Number of unique documents processed"
                },
                {
                    "name": "Total Chunks",
                    "value": total_points,
                    "format": "number",
                    "icon": "database",
                    "description": "Total number of vector chunks in the database"
                },
                {
                    "name": "Avg. Chunks per Document",
                    "value": avg_chunks,
                    "format": "decimal",
                    "icon": "chart-bar",
                    "description": "Average number of chunks per document"
                },
                {
                    "name": "Database Status",
                    "value": raw_stats.get("status", "unknown"),
                    "format": "status",
                    "icon": "status-circle",
                    "description": "Current status of the database"
                }
            ],
            "file_types": {
                "chart_type": "pie",
                "data": [
                    {"name": file_type, "value": count}
                    for file_type, count in raw_stats.get("file_types", {}).items()
                ],
                "title": "Document Types Distribution"
            },
            "collection_info": {
                "name": raw_stats.get("collection_name", "content_library"),
                "segments": raw_stats.get("segments_count", 0),
                "vector_dimension": raw_stats.get("vector_dimension", 1024)
            },
            "recent_activity": [
                {
                    "title": doc.get("title", "Unknown Document"),
                    "id": doc.get("source_id", ""),
                    "timestamp": doc.get("processed_at", "Unknown")
                }
                for doc in raw_stats.get("recently_processed", [])
            ]
        }
        
        # Add database health indicator
        dashboard["health"] = {
            "status": "healthy" if raw_stats.get("status") == "green" else "warning",
            "last_checked": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "details": {
                "vectors_status": "available" if raw_stats.get("vectors_count") is not None else "unknown",
                "segments_status": "optimized" if raw_stats.get("segments_count", 0) < 20 else "needs_optimization"
            }
        }
        
        duration_ms = (time.time() - start_time) * 1000
        return ApiResponse(
            success=True,
            data=dashboard,
            message="Dashboard data retrieved successfully",
            duration_ms=duration_ms
        )

    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Server ---

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='RAG Content Retriever API')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind the server to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    args = parser.parse_args()

    # Start the server
    logger.info(f"Starting API server at http://{args.host}:{args.port}")
    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)