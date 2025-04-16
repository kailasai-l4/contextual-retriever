import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from storage.qdrant_manager import QdrantManager

# Load environment variables from .env
load_dotenv()

# Import config and provider factories
from core.config import Config
from embedding import get_embedding_provider
from expansion import get_expansion_provider
from reranking import get_reranker_provider

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Create FastAPI app
app = FastAPI(title="RAG System API", version="1.0.0")

# Allow CORS for all origins (customize for prod if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config and providers at startup
@app.on_event("startup")
def startup_event():
    config = Config.from_env()
    logger.info(f"Loaded config: default_expansion_provider={config.default_expansion_provider}")
    app.state.config = config
    app.state.embedding_provider = get_embedding_provider(config)
    app.state.expansion_provider = get_expansion_provider(config)
    app.state.reranker_provider = get_reranker_provider(config)
    # Initialize Qdrant client and manager
    qdrant_url = config.qdrant.url
    qdrant_port = config.qdrant.port
    qdrant_client = QdrantClient(host=qdrant_url, port=qdrant_port, timeout=120)
    app.state.qdrant_manager = QdrantManager(qdrant_client)
    logger.info("API startup: config, providers, and Qdrant manager loaded.")

# Exception handler for clean error responses
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."}
    )

# Health check endpoint
@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}

# Example: Add your route modules here
from api.routes import search
from api.routes import collections
from api.routes import process
app.include_router(search.router)
app.include_router(collections.router)
app.include_router(process.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=int(os.getenv("API_PORT", 8000)), reload=True) 