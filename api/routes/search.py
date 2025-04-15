from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

# Import Retriever from retrieval
from retrieval.retriever import Retriever

logger = logging.getLogger("api.search")

router = APIRouter(prefix="/search", tags=["search"])

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    use_expansion: Optional[bool] = True
    collection_name: Optional[str] = "content_library"
    expansion_model: Optional[str] = None

class SearchResult(BaseModel):
    text: str
    score: float
    source_id: Optional[str]
    source_path: Optional[str]
    metadata: Optional[dict]
    keywords: Optional[List[str]]
    rerank_score: Optional[float] = None
    rerank_position: Optional[int] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    expanded_query: Optional[str] = None
    expansion_model: Optional[str] = None

@router.post("/", response_model=SearchResponse)
async def search_endpoint(request: Request, body: SearchRequest):
    try:
        config = request.app.state.config
        embedding_provider = request.app.state.embedding_provider
        reranker_provider = request.app.state.reranker_provider
        qdrant_manager = request.app.state.qdrant_manager
        # Expansion provider selection
        expansion_model = body.expansion_model or getattr(config, "default_expansion_provider", None)
        expansion_provider = None
        expanded_query = None
        if body.use_expansion and expansion_model:
            from expansion import get_expansion_provider
            expansion_provider = get_expansion_provider(config, provider_name=expansion_model)
            expanded_query = expansion_provider.expand_query(body.query)
            search_query = expanded_query
        else:
            search_query = body.query
        retriever = Retriever(
            embedding_provider=embedding_provider,
            reranker_provider=reranker_provider,
            storage_manager=qdrant_manager
        )
        results = retriever.search(
            query=search_query,
            limit=body.limit,
            use_expansion=False,  # expansion already applied
            collection_name=body.collection_name
        )
        if isinstance(results, dict) and "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        return {"results": results, "expanded_query": expanded_query, "expansion_model": expansion_model}
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))