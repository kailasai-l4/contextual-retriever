class Retriever:
    def __init__(self, embedding_provider, reranker_provider, storage_manager):
        self.embedding_provider = embedding_provider
        self.reranker_provider = reranker_provider
        self.storage_manager = storage_manager

    def search(self, query, limit=10, use_expansion=True, collection_name="content_library"):
        import logging
        logger = logging.getLogger("Retriever")
        try:
            logger.info(f"Searching for query: {query} in collection: {collection_name}")
            query_vector = self.embedding_provider.get_query_embedding(query)
            logger.info(f"Query embedding shape: {len(query_vector)}")
            results = self.storage_manager.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit
            )
            logger.info(f"Search results: {results}")
            # Return results as a list of dicts (for API response)
            return results
        except Exception as e:
            logger.error(f"Retriever.search error: {e}", exc_info=True)
            return {"error": str(e)}