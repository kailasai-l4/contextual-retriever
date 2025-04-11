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
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile, Body, Query, Header, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
from datetime import datetime
import traceback

# Load .env file
load_dotenv()

# Import our modules
from config import get_config, get_qdrant_client
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
    collection: Optional[str] = Field(None, description="Specific collection to search")
    search_all_collections: Optional[bool] = Field(False, description="Whether to search all available collections")

class ProcessRequest(BaseModel):
    directory_path: str = Field(..., description="Path to directory containing documents")
    recursive: Optional[bool] = Field(True, description="Process directories recursively")
    file_types: Optional[List[str]] = Field(None, description="File extensions to process (e.g., [\".md\", \".json\"])")
    collection: Optional[str] = Field(None, description="Target collection for processed documents")

class BulkDataRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="List of data items to process")
    collection: Optional[str] = Field(None, description="Target collection for processed data")

class BulkEmbeddingRequest(BaseModel):
    embeddings: List[Dict[str, Any]] = Field(..., description="List of embedding records with id, vector, and payload")
    collection: Optional[str] = Field(None, description="Target collection for embeddings")

class CollectionRequest(BaseModel):
    name: str = Field(..., description="Name of the collection")
    description: Optional[str] = Field(None, description="Description of the collection")
    vector_size: Optional[int] = Field(None, description="Vector dimension size")

class HealthResponse(BaseModel):
    status: str
    details: Dict[str, Any]
    timestamp: str

class ApiResponse(BaseModel):
    success: bool
    data: Any
    message: Optional[str] = None
    duration_ms: float

# --- Error handling ---

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Global exception handler to provide consistent error responses"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "message": "An unexpected error occurred",
            "details": traceback.format_exc() if app.debug else None
        }
    )

# --- Dependencies ---

def get_api_components(collection: Optional[str] = None):
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

    # Initialize processor and retriever (they will share the Qdrant client)
    processor = ContentProcessor(
        jina_api_key=jina_key,
        gemini_api_key=gemini_key,
        qdrant_url=qdrant_url,
        qdrant_port=qdrant_port,
        collection=collection
    )

    retriever = AdvancedRetriever(
        jina_api_key=jina_key,
        gemini_api_key=gemini_key,
        qdrant_url=qdrant_url,
        qdrant_port=qdrant_port,
        collection=collection
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

# --- Health Check Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring systems"""
    try:
        # Check Qdrant connection
        qdrant_client = get_qdrant_client()
        qdrant_status = "ok"
        qdrant_details = "Connected"
        
        try:
            # Try to get collection list
            collections = qdrant_client.get_collections()
            collection_count = len(collections.collections)
            qdrant_details = f"Connected: {collection_count} collections available"
        except Exception as e:
            qdrant_status = "degraded"
            qdrant_details = f"Connection issue: {str(e)}"
    except Exception as e:
        qdrant_status = "error"
        qdrant_details = f"Failed to connect: {str(e)}"
    
    # Overall status depends on Qdrant
    status = "ok" if qdrant_status == "ok" else "degraded" 
    
    return HealthResponse(
        status=status,
        details={
            "api": "ok",
            "qdrant": {
                "status": qdrant_status,
                "details": qdrant_details
            }
        },
        timestamp=datetime.now().isoformat()
    )

@app.get("/readiness")
async def readiness_probe():
    """Kubernetes readiness probe endpoint"""
    # Only return 200 OK if system is fully operational
    health = await health_check()
    if health.status != "ok":
        raise HTTPException(status_code=503, detail="System not ready")
    return {"status": "ready"}

@app.get("/liveness")
async def liveness_probe():
    """Kubernetes liveness probe endpoint"""
    # Just ensure the API is running 
    return {"status": "alive"}

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
            {"path": "/collections", "method": "GET", "description": "List available collections"},
            {"path": "/collections/{name}", "method": "POST", "description": "Create a new collection"},
            {"path": "/collections/{name}", "method": "GET", "description": "Get collection details"},
            {"path": "/bulk/process", "method": "POST", "description": "Process bulk data"},
            {"path": "/bulk/embeddings", "method": "POST", "description": "Upload pre-computed embeddings"},
            {"path": "/sources/{source_id}", "method": "GET", "description": "Get all content from a specific source"},
            {"path": "/dashboard", "method": "GET", "description": "Get a comprehensive dashboard view of the database statistics"},
            {"path": "/health", "method": "GET", "description": "Health check endpoint for monitoring systems"},
            {"path": "/readiness", "method": "GET", "description": "Kubernetes readiness probe endpoint"},
            {"path": "/liveness", "method": "GET", "description": "Kubernetes liveness probe endpoint"}
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
    - collection: Specific collection to search
    - search_all_collections: Whether to search all available collections

    Returns:
    - success: Whether the request was successful
    - data: Search results
    - message: Additional information
    - duration_ms: Request duration in milliseconds
    """
    start_time = time.time()

    try:
        _, retriever = get_api_components(request.collection)

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
                use_optimized_retrieval=request.use_optimized_retrieval,
                collection=request.collection,
                search_all_collections=request.search_all_collections
            )

        duration_ms = (time.time() - start_time) * 1000
        return ApiResponse(
            success=True,
            data={
                "query": request.query,
                "count": len(results),
                "results": results,
                "collections_searched": [request.collection] if request.collection else 
                                     (retriever.available_collections if request.search_all_collections else [retriever.collection_name])
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
    - collection: Target collection for processed documents

    Returns:
    - success: Whether the request was started successfully
    - data: Request details
    - message: Additional information
    """
    start_time = time.time()

    try:
        processor, _ = get_api_components(request.collection)

        # Define the background task
        def process_task(path, recursive, file_types, collection=None):
            try:
                logger.info(f"Starting background processing of {path}" + 
                           (f" to collection {collection}" if collection else ""))

                # Convert file types to proper format
                if file_types:
                    file_types = [ft if ft.startswith('.') else f'.{ft}' for ft in file_types]

                # Set collection if specified
                if collection and collection != processor.collection_name:
                    processor.set_collection(collection)

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
            request.file_types,
            request.collection
        )

        duration_ms = (time.time() - start_time) * 1000
        return ApiResponse(
            success=True,
            data={
                "directory_path": request.directory_path,
                "recursive": request.recursive,
                "file_types": request.file_types,
                "collection": request.collection or processor.collection_name
            },
            message="Document processing started in the background",
            duration_ms=duration_ms
        )

    except Exception as e:
        logger.error(f"Error starting document processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bulk/process", response_model=ApiResponse)
async def process_bulk_data(request: BulkDataRequest, background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)):
    """
    Process bulk data directly from structured content
    
    Parameters:
    - data: List of data items to process. Each item should have at least 'text' field
            and can optionally include 'metadata', 'source_id', etc.
    - collection: Target collection for processed data
            
    Returns:
    - success: Whether the request was started successfully
    - data: Request details
    - message: Information about the bulk processing task
    """
    start_time = time.time()
    
    try:
        processor, _ = get_api_components(request.collection)
        
        # Validate data format
        if not request.data:
            raise HTTPException(status_code=400, detail="No data provided for processing")
        
        for i, item in enumerate(request.data):
            if "text" not in item:
                raise HTTPException(status_code=400, detail=f"Item at index {i} missing required 'text' field")
        
        # Process in background
        def process_bulk_task(data, collection):
            try:
                logger.info(f"Starting bulk processing of {len(data)} items" +
                           (f" to collection {collection}" if collection else ""))
                
                # Set collection if specified
                if collection and collection != processor.collection_name:
                    processor.set_collection(collection)
                    
                # Process the data
                stats = processor.process_bulk_data(data, collection)
                
                logger.info(f"Completed bulk processing: {stats['successful_items']}/{stats['total_items']} items processed")
            except Exception as e:
                logger.error(f"Error during bulk processing: {str(e)}", exc_info=True)
        
        # Start background task
        background_tasks.add_task(
            process_bulk_task,
            request.data,
            request.collection
        )
        
        duration_ms = (time.time() - start_time) * 1000
        return ApiResponse(
            success=True,
            data={
                "items_count": len(request.data),
                "collection": request.collection or processor.collection_name,
                "task_id": f"bulk_{int(time.time())}"
            },
            message=f"Bulk processing of {len(request.data)} items started in the background",
            duration_ms=duration_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting bulk processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bulk/embeddings", response_model=ApiResponse)
async def upload_embeddings(request: BulkEmbeddingRequest, background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)):
    """
    Upload pre-computed embeddings directly to the database
    
    Parameters:
    - embeddings: List of embedding records. Each record should have:
                - 'id': Unique identifier for the point
                - 'vector': Vector embedding (List[float])
                - 'payload': Metadata to store with the vector
    - collection: Target collection for embeddings
            
    Returns:
    - success: Whether the request was started successfully
    - data: Request details
    - message: Information about the upload task
    """
    start_time = time.time()
    
    try:
        processor, _ = get_api_components(request.collection)
        
        # Validate data format
        if not request.embeddings:
            raise HTTPException(status_code=400, detail="No embeddings provided for upload")
        
        for i, emb in enumerate(request.embeddings):
            if "id" not in emb:
                raise HTTPException(status_code=400, detail=f"Embedding at index {i} missing required 'id' field")
            if "vector" not in emb:
                raise HTTPException(status_code=400, detail=f"Embedding at index {i} missing required 'vector' field")
        
        # Process in background
        def upload_embeddings_task(embeddings, collection):
            try:
                logger.info(f"Starting upload of {len(embeddings)} embeddings" +
                           (f" to collection {collection}" if collection else ""))
                
                # Set collection if specified
                if collection and collection != processor.collection_name:
                    processor.set_collection(collection)
                    
                # Upload the embeddings
                stats = processor.bulk_upload_embeddings(embeddings, collection)
                
                logger.info(f"Completed embeddings upload: {stats['successful_uploads']}/{stats['total_embeddings']} embeddings uploaded")
            except Exception as e:
                logger.error(f"Error during embeddings upload: {str(e)}", exc_info=True)
        
        # Start background task
        background_tasks.add_task(
            upload_embeddings_task,
            request.embeddings,
            request.collection
        )
        
        duration_ms = (time.time() - start_time) * 1000
        return ApiResponse(
            success=True,
            data={
                "embeddings_count": len(request.embeddings),
                "collection": request.collection or processor.collection_name,
                "task_id": f"emb_upload_{int(time.time())}"
            },
            message=f"Upload of {len(request.embeddings)} embeddings started in the background",
            duration_ms=duration_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting embeddings upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=ApiResponse)
async def get_stats(collection: Optional[str] = None, api_key: str = Depends(get_api_key)):
    """
    Get statistics about the vector database

    Parameters:
    - collection: Optional collection name to get stats for

    Returns:
    - success: Whether the request was successful
    - data: Database statistics
    - duration_ms: Request duration in milliseconds
    """
    start_time = time.time()

    try:
        processor, _ = get_api_components(collection)
        
        # If collection is specified, set it
        if collection and collection != processor.collection_name:
            processor.set_collection(collection)
            
        raw_stats = processor.get_collection_stats()
        
        # Format statistics in a more human-readable way
        stats = {
            "summary": {
                "collection_name": raw_stats.get("collection_name", processor.collection_name),
                "status": raw_stats.get("status", "unknown"),
                "total_chunks": raw_stats.get("points_count", 0),
                "unique_documents": raw_stats.get("unique_sources", 0),
                "avg_chunks_per_document": raw_stats.get("avg_chunks_per_source", 0)
            },
            "database_info": {
                "vectors_count": raw_stats.get("vectors_count"),
                "segments_count": raw_stats.get("segments_count"),
                "vector_dimension": raw_stats.get("vector_dimension", processor.vector_size)
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
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    process_now: bool = Query(True),
    collection: Optional[str] = Query(None, description="Target collection for the uploaded file"),
    api_key: str = Depends(get_api_key)
):
    """
    Upload and optionally process a file

    Parameters:
    - file: File to upload
    - process_now: Whether to process the file immediately
    - collection: Target collection for the uploaded file (defaults to current collection)

    Returns:
    - success: Whether the upload was successful
    - data: Upload details
    - message: Additional information
    """
    start_time = time.time()

    try:
        processor, _ = get_api_components(collection)

        # Create uploads directory if it doesn't exist
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)

        # Save the file
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Process the file if requested
        if process_now:
            def process_file_task(path, target_collection=None):
                try:
                    logger.info(f"Processing uploaded file: {path}" + 
                               (f" to collection {target_collection}" if target_collection else ""))
                    
                    # Set collection if specified
                    if target_collection and target_collection != processor.collection_name:
                        processor.set_collection(target_collection)
                        
                    processor.process_file(path)
                    logger.info(f"Completed processing of uploaded file: {path}")
                except Exception as e:
                    logger.error(f"Error processing uploaded file: {str(e)}", exc_info=True)

            background_tasks.add_task(process_file_task, file_path, collection)
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
                "processed": process_now,
                "collection": collection or processor.collection_name
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
        
        # Ensure we're working with numeric values
        if isinstance(total_points, str):
            total_points = int(total_points) if total_points.isdigit() else 0
        if isinstance(total_sources, str):
            total_sources = int(total_sources) if total_sources.isdigit() else 0
            
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

# --- Collection Management ---

@app.get("/collections", response_model=ApiResponse)
async def list_collections(api_key: str = Depends(get_api_key)):
    """
    List all available collections
    
    Returns:
        List of available collections with basic info
    """
    start_time = time.time()
    
    try:
        # Get Qdrant client
        qdrant_client = get_qdrant_client()
        
        # Get collections from Qdrant
        collections_info = qdrant_client.get_collections().collections
        
        # Enhanced with details
        collections = []
        for coll in collections_info:
            try:
                # Get detailed info for each collection
                details = qdrant_client.get_collection(coll.name)
                
                collections.append({
                    "name": coll.name,
                    "vectors_count": details.vectors_count,
                    "points_count": details.points_count,
                    "vector_size": details.config.params.vectors.size if hasattr(details.config, 'params') else None,
                    "distance_metric": str(details.config.params.vectors.distance) if hasattr(details.config, 'params') else None
                })
            except Exception as e:
                # Add basic info if detailed retrieval fails
                collections.append({
                    "name": coll.name,
                    "error": str(e)
                })
        
        duration_ms = (time.time() - start_time) * 1000
        return ApiResponse(
            success=True,
            data=collections,
            message=f"Found {len(collections)} collections",
            duration_ms=duration_ms
        )
        
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/{name}", response_model=ApiResponse)
async def create_collection(
    name: str, 
    request: CollectionRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Create a new collection with specified parameters
    
    Parameters:
    - name: Collection name
    - vector_size: Size of vector embedding
    
    Returns:
        Collection creation status
    """
    start_time = time.time()
    
    try:
        processor, _ = get_api_components()
        
        # Use specified vector size or default from config
        vector_size = request.vector_size or processor.vector_size
        
        # Create the collection
        success = processor.set_collection(name)
        
        duration_ms = (time.time() - start_time) * 1000
        return ApiResponse(
            success=success,
            data={
                "name": name,
                "vector_size": vector_size,
                "description": request.description
            },
            message=f"Collection '{name}' {'created or accessed successfully' if success else 'creation failed'}",
            duration_ms=duration_ms
        )
        
    except Exception as e:
        logger.error(f"Error creating collection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/{name}", response_model=ApiResponse)
async def get_collection(name: str, api_key: str = Depends(get_api_key)):
    """
    Get detailed information about a specific collection
    
    Parameters:
    - name: Collection name
    
    Returns:
        Detailed collection information
    """
    start_time = time.time()
    
    try:
        # Get Qdrant client
        qdrant_client = get_qdrant_client()
        
        if not qdrant_client.collection_exists(name):
            raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
        
        # Get collection details
        details = qdrant_client.get_collection(name)
        
        # Sample a few points to get payload schema
        try:
            sample = qdrant_client.scroll(
                collection_name=name,
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            sample_points = sample[0] if isinstance(sample, tuple) else []
            
            # Extract payload schema from sample if available
            payload_schema = {}
            if sample_points and hasattr(sample_points[0], 'payload'):
                payload = sample_points[0].payload
                payload_schema = {k: type(v).__name__ for k, v in payload.items()}
        except Exception as e:
            logger.warning(f"Error sampling collection points: {str(e)}")
            payload_schema = {"error": str(e)}
        
        # Format collection info
        collection_info = {
            "name": name,
            "vectors_count": details.vectors_count,
            "points_count": details.points_count,
            "vector_size": details.config.params.vectors.size if hasattr(details.config, 'params') else None,
            "distance_metric": str(details.config.params.vectors.distance) if hasattr(details.config, 'params') else None,
            "segments_count": details.segments_count,
            "payload_schema": payload_schema,
            "status": "available" if details.status == "green" else details.status
        }
        
        duration_ms = (time.time() - start_time) * 1000
        return ApiResponse(
            success=True,
            data=collection_info,
            message=f"Collection '{name}' details retrieved",
            duration_ms=duration_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collection details: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/collections/{name}", response_model=ApiResponse)
async def delete_collection(name: str, api_key: str = Depends(get_api_key)):
    """
    Delete a collection
    
    Parameters:
    - name: Collection name
    
    Returns:
        Deletion status
    """
    start_time = time.time()
    
    try:
        # Get Qdrant client
        qdrant_client = get_qdrant_client()
        
        if not qdrant_client.collection_exists(name):
            raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
        
        # Delete the collection
        qdrant_client.delete_collection(name)
        
        duration_ms = (time.time() - start_time) * 1000
        return ApiResponse(
            success=True,
            data={"name": name},
            message=f"Collection '{name}' deleted successfully",
            duration_ms=duration_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}", exc_info=True)
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