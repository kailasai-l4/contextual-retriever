import os
import json
import time
import hashlib
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import numpy as np
import backoff

# External dependencies
import requests
from tqdm.auto import tqdm
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse

# Import configuration
from config import get_config, get_qdrant_client

# Initialize logging
logger = logging.getLogger("advanced_retriever")

class AdvancedRetriever:
    def __init__(self,
                 jina_api_key: str,
                 gemini_api_key: str,
                 qdrant_url: str = "localhost",
                 qdrant_port: int = 6333,
                 config_path: str = None,
                 collection: str = None):
        """
        Initialize the advanced retriever

        Args:
            jina_api_key: API key for Jina AI
            gemini_api_key: API key for Gemini model
            qdrant_url: URL for Qdrant vector database
            qdrant_port: Port for Qdrant vector database
            config_path: Path to configuration file
            collection: Name of the collection to use (defaults to config's default collection)
        """
        # Load configuration
        self.config = get_config(config_path)

        # API keys
        self.jina_api_key = jina_api_key

        # Get collection name
        self.default_collection = self.config.get('qdrant', 'default_collection', default="content_library")
        self.collection_name = collection or self.default_collection

        # Available collections
        self.available_collections = self.config.get('qdrant', 'collections', default=[self.default_collection])

        # Get retrieval parameters from config
        self.max_chunk_tokens = self.config.get('retrieval', 'max_consolidated_tokens', default=4000)

        # Initialize Gemini
        self.gemini_api_key = gemini_api_key
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel(self.config.get('gemini', 'model', default="gemini-1.5-flash-latest"))

        # Initialize client using the shared client
        self.client = get_qdrant_client(qdrant_url, qdrant_port)

        # Set up logging
        log_file = self.config.get('logging', 'file', default="rag_retriever.log")
        # Ensure the log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure file handler for this module's logger if not already set up
        if not logger.handlers:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)

        # Set log level from config
        log_level = getattr(logging, self.config.get('logging', 'level', default="INFO"))
        logger.setLevel(log_level)

        logger.debug(f"Advanced Retriever initialized with Qdrant at {qdrant_url}:{qdrant_port}, collection: {self.collection_name}")

        # Check if collection exists and has data
        self._check_qdrant_collection(self.collection_name)

    def set_collection(self, collection_name: str) -> bool:
        """
        Change the active collection

        Args:
            collection_name: Name of the collection to use

        Returns:
            bool: True if collection exists and was set, False otherwise
        """
        if self._check_qdrant_collection(collection_name):
            self.collection_name = collection_name
            logger.info(f"Switched to collection: {collection_name}")
            return True
        return False

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=backoff.full_jitter)
    def _check_qdrant_collection(self, collection_name: str) -> bool:
        """
        Check if Qdrant collection exists and has vectors with improved error handling

        Args:
            collection_name: Name of the collection to check

        Returns:
            bool: True if collection exists and has data, False otherwise
        """
        try:
            # Check if collection exists
            if not self.client.collection_exists(collection_name):
                logger.warning(f"Collection '{collection_name}' does not exist")
                return False

            # Get collection info
            collection_info = self.client.get_collection(collection_name)
            logger.debug(f"Collection info: vectors_count={collection_info.vectors_count}, points_count={collection_info.points_count}")

            if collection_info.vectors_count == 0:
                logger.warning(f"Collection '{collection_name}' is empty (0 vectors)")
                return False

            # Try to get a sample point to verify access
            try:
                scroll_results = self.client.scroll(
                    collection_name=collection_name,
                    limit=1,
                    with_payload=True,
                    with_vectors=False
                )

                # Handle different scroll_results formats
                if isinstance(scroll_results, tuple):
                    # Direct tuple response (points, next_offset)
                    points = scroll_results[0]
                elif hasattr(scroll_results, '__iter__'):
                    # Iterator response (older API)
                    try:
                        points, _ = next(scroll_results)
                    except Exception:
                        # If iterator fails, try treating as a tuple
                        points = scroll_results[0] if len(scroll_results) > 0 else []
                else:
                    # Unknown format
                    logger.warning(f"Unknown scroll_results format: {type(scroll_results)}")
                    points = []

                # Check if we got any results
                if not points:
                    logger.warning(f"No points found in collection '{collection_name}'")
                    return False

                logger.debug(f"Successfully retrieved sample point from collection")
                return True

            except Exception as e:
                logger.error(f"Error retrieving sample point: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"Error checking collection: {str(e)}", exc_info=True)
            return False

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=backoff.full_jitter)
    def search(self, query: str, limit: int = 20, use_optimized_retrieval: bool = True, 
               collection: str = None, search_all_collections: bool = False) -> List[Dict[str, Any]]:
        """
        Search for content with advanced optimization options

        Args:
            query: Search query
            limit: Maximum number of results to return
            use_optimized_retrieval: Whether to use optimized retrieval strategy
            collection: Optional specific collection to search
            search_all_collections: Whether to search all available collections

        Returns:
            List of content chunks
        """
        logger.debug(f"Search initiated: query='{query}', limit={limit}, optimized={use_optimized_retrieval}")

        # Determine which collections to search
        collections_to_search = []
        if search_all_collections:
            # Get available collections from config
            collections_to_search = self.available_collections
        else:
            # Use specified collection or default
            target_collection = collection or self.collection_name
            collections_to_search = [target_collection]

        # If no collections to search, return empty results
        if not collections_to_search:
            logger.warning("No collections available to search")
            return []

        logger.debug(f"Searching collections: {collections_to_search}")

        # Search each collection and merge results
        all_results = []
        for coll_name in collections_to_search:
            # Skip collections that don't exist
            if not self.client.collection_exists(coll_name):
                logger.warning(f"Collection '{coll_name}' does not exist. Skipping.")
                continue

            try:
                # Get collection info
                collection_info = self.client.get_collection(coll_name)
                
                if collection_info.vectors_count == 0:
                    logger.warning(f"Collection '{coll_name}' is empty. Skipping.")
                    continue
                
                # Set temporary collection name
                orig_collection = self.collection_name
                self.collection_name = coll_name
                
                # Perform search
                if use_optimized_retrieval:
                    try:
                        result = self.retrieve_optimized_content(query, limit)
                        collection_results = result["chunks"]
                    except Exception as e:
                        logger.error(f"Error in optimized retrieval for collection '{coll_name}': {str(e)}", exc_info=True)
                        logger.info(f"Falling back to standard search for collection '{coll_name}'")
                        collection_results = self._simple_search(query, limit=limit, use_expansion=True)
                else:
                    collection_results = self._simple_search(query, limit=limit, use_expansion=True)
                
                # Add collection name to results
                for item in collection_results:
                    item["collection"] = coll_name
                
                all_results.extend(collection_results)
                
                # Restore original collection name
                self.collection_name = orig_collection
                
            except Exception as e:
                logger.error(f"Error searching collection '{coll_name}': {str(e)}", exc_info=True)
                continue

        # Sort combined results by score
        all_results.sort(key=lambda x: x.get("rerank_score", x.get("score", 0)), reverse=True)
        
        # Limit to requested number
        return all_results[:limit]

    def _simple_search(self, query: str, limit: int = 100, use_expansion: bool = True):
        """
        Basic search implementation for content using the vector database

        Args:
            query: Search query
            limit: Maximum number of results to return
            use_expansion: Whether to use query expansion

        Returns:
            Search results with metadata
        """
        # Ensure limit is at least 1 to prevent Qdrant API errors
        limit = max(1, limit)
        
        try:
            search_query = query

            # Expand query if requested
            if use_expansion:
                search_query = self._expand_query(query)
                logger.info(f"Expanded query: {len(search_query.split())} terms")

            # Generate embedding for the query
            try:
                response = requests.post(
                    "https://api.jina.ai/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.jina_api_key}",
                        "Accept": "application/json"
                    },
                    json={
                        "model": self.config.get('jina', 'embedding_model', default="jina-embeddings-v3"),
                        "input": [search_query],
                        "task": "retrieval.query"  # Important: optimization for query embedding
                    },
                    timeout=60
                )

                if response.status_code != 200:
                    response_text = response.text[:500] if len(response.text) > 500 else response.text
                    raise Exception(f"Embedding API error: {response.status_code} {response_text}")

                query_embedding = response.json()["data"][0]["embedding"]
            except Exception as e:
                logger.error(f"Error generating query embedding: {str(e)}", exc_info=True)
                return []

            # Search in Qdrant with oversampling
            try:
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=limit * 2,  # Oversample for post-processing
                    score_threshold=0.5  # Minimum similarity threshold
                )
            except Exception as e:
                logger.error(f"Error searching Qdrant: {str(e)}", exc_info=True)
                return []

            # Extract result information and deduplicate
            results = []
            seen_content = set()

            for hit in search_results:
                text = hit.payload.get("text", "")
                content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

                # Skip near-duplicates
                if content_hash in seen_content:
                    continue
                seen_content.add(content_hash)

                results.append({
                    "text": text,
                    "score": hit.score,
                    "source_id": hit.payload.get("source_id", ""),
                    "source_path": hit.payload.get("source_path", ""),
                    "metadata": hit.payload.get("metadata", {}),
                    "keywords": hit.payload.get("topic_keywords", [])
                })

            # Apply reranking for better relevance
            if len(results) > 1:
                texts = [result["text"] for result in results]

                try:
                    rerank_response = requests.post(
                        "https://api.jina.ai/v1/rerank",
                        headers={
                            "Authorization": f"Bearer {self.jina_api_key}",
                            "Accept": "application/json"
                        },
                        json={
                            "model": self.config.get('jina', 'reranker_model', default="jina-reranker-v2-base-multilingual"),
                            "query": query,  # Use original query for reranking
                            "documents": texts,
                            "top_n": min(limit, len(texts))
                        },
                        timeout=60
                    )

                    if rerank_response.status_code == 200:
                        # Create a mapping of text to new rank
                        new_scores = {}
                        for idx, item in enumerate(rerank_response.json()["results"]):
                            new_scores[item["document"]["text"]] = {
                                "new_score": item["relevance_score"],
                                "new_rank": idx
                            }

                        # Apply new scores to our results
                        for result in results:
                            if result["text"] in new_scores:
                                result["rerank_score"] = new_scores[result["text"]]["new_score"]
                                result["rerank_position"] = new_scores[result["text"]]["new_rank"]

                        # Sort by reranker position
                        results.sort(key=lambda x: x.get("rerank_position", 999))
                except Exception as e:
                    logger.warning(f"Error during reranking: {str(e)}", exc_info=True)
                    # Fall back to sorting by original score
                    results.sort(key=lambda x: x.get("score", 0), reverse=True)

            # Limit to requested number
            return results[:limit]

        except Exception as e:
            logger.error(f"Search error: {str(e)}", exc_info=True)
            return []

    def retrieve_optimized_content(self, topic: str, limit: int = 100) -> Dict[str, Any]:
        """
        Retrieve and optimize content for the given topic

        This implements the advanced retrieval techniques based on the research paper
        findings about embedding model limitations

        Args:
            topic: Topic to retrieve content for
            limit: Maximum number of chunks to retrieve

        Returns:
            Dictionary with optimized content
        """
        logger.debug(f"Retrieving optimized content for topic: '{topic}', limit={limit}")

        # 1. Generate query variations with different levels of specificity
        logger.debug("Generating query variations")
        query_variations = self._generate_query_variations(topic)
        logger.debug(f"Generated {len(query_variations)} query variations: {query_variations}")

        # 2. Expanded search with improved query
        logger.debug("Expanding search query")
        expanded_query = self._expand_query(topic)
        logger.debug(f"Expanded query: '{expanded_query}'")

        logger.debug("Performing initial search with expanded query")
        raw_results = self._simple_search(expanded_query, limit=limit*2, use_expansion=True)
        logger.debug(f"Initial search returned {len(raw_results)} results")

        # 3. Add results from query variations
        all_results = raw_results.copy()

        for query in tqdm(query_variations, desc="Searching query variations"):
            logger.debug(f"Searching with variation: '{query}'")
            # Ensure limit is at least 1 to prevent Qdrant API error
            variation_limit = max(1, limit//2)
            additional_results = self._simple_search(query, limit=variation_limit, use_expansion=True)
            logger.debug(f"Variation '{query}' returned {len(additional_results)} results")
            all_results.extend(additional_results)

        # 4. Deduplicate results
        logger.debug(f"Deduplicating {len(all_results)} total results")
        unique_results = self._deduplicate_results(all_results)
        logger.debug(f"After deduplication: {len(unique_results)} unique results")

        # Check if we have any results at this point
        if not unique_results:
            logger.warning("No results found after initial search and deduplication")
            return {
                "topic": topic,
                "chunks": [],
                "total_retrieved": len(all_results),
                "unique_retrieved": 0,
                "consolidated_chunks": 0,
                "final_chunks": 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        # 5. Group results by source and arrange chunks in source order
        logger.debug("Grouping results by source")
        grouped_results = self._group_by_source(unique_results)
        logger.debug(f"Results grouped into {len(grouped_results)} sources")

        # 6. Consolidate small chunks into larger, meaningful units
        logger.debug("Consolidating chunks")
        consolidated_chunks = self._consolidate_chunks(grouped_results)
        logger.debug(f"Consolidated into {len(consolidated_chunks)} meaningful segments")

        # 7. Select most diverse and relevant chunks
        logger.debug("Selecting diverse chunks")
        final_chunks = self._select_diverse_chunks(consolidated_chunks, topic, limit)
        logger.debug(f"Selected {len(final_chunks)} diverse chunks")

        # 8. Arrange chunks in a logical order
        logger.debug("Ordering chunks logically")
        ordered_chunks = self._order_chunks_logically(final_chunks, topic)
        logger.debug(f"Final result: {len(ordered_chunks)} ordered chunks")

        # Ensure all chunks have proper scores before returning
        for chunk in ordered_chunks:
            # Make sure each chunk has at least one score field
            if not any(key in chunk for key in ['score', 'rerank_score', 'avg_score']):
                chunk['score'] = 0.75
                chunk['rerank_score'] = 0.75
            # If there's only one score field, populate the others
            elif 'avg_score' in chunk and not 'score' in chunk:
                chunk['score'] = chunk['avg_score']
                chunk['rerank_score'] = chunk['avg_score']
            elif 'score' in chunk and not 'rerank_score' in chunk:
                chunk['rerank_score'] = chunk['score']

        return {
            "topic": topic,
            "chunks": ordered_chunks,
            "total_retrieved": len(all_results),
            "unique_retrieved": len(unique_results),
            "consolidated_chunks": len(consolidated_chunks),
            "final_chunks": len(ordered_chunks),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def _generate_query_variations(self, topic: str) -> List[str]:
        """Generate variations of the query to capture different aspects"""
        # Prompt Gemini for query variations
        prompt = f"""
        I need to create comprehensive content about "{topic}".

        Please generate 5-8 specific queries that would help retrieve diverse,
        relevant content about different aspects of this topic.

        Format as a simple list with one query per line.
        For example, if the topic is "Machine Learning", your response might be:
        What is machine learning?
        History of machine learning algorithms
        Applications of machine learning in healthcare
        Machine learning vs deep learning
        Ethics of machine learning systems
        Future of machine learning technology
        """

        try:
            response = self.gemini_model.generate_content(prompt)

            # Extract queries from response
            lines = response.text.strip().split('\n')
            queries = [line.strip() for line in lines if line.strip()]

            # Filter out lines that aren't proper queries
            queries = [q for q in queries if len(q) > 10 and ('?' in q or any(w in q.lower() for w in ['how', 'what', 'why', 'when', 'where', 'which', 'who']))]

            return queries
        except Exception as e:
            logger.error(f"Error generating query variations: {str(e)}", exc_info=True)
            # Fallback - create basic variations
            return [
                f"What is {topic}?",
                f"History of {topic}",
                f"Examples of {topic}"
            ]

    def _expand_query(self, query: str, expansion_size: int = 100) -> str:
        """
        Expand query using Gemini model for improved retrieval

        Args:
            query: Original query to expand
            expansion_size: Target number of expansion terms

        Returns:
            Expanded query
        """
        try:
            # Prompt Gemini for query expansion
            prompt = f"""
            I need to expand this search query for a vector database lookup:

            QUERY: {query}

            Please generate around {expansion_size} related terms, synonyms, and concepts that would help improve retrieval.
            Format the result as a comma-separated list of terms.
            Focus on semantic relationships and domain-specific terminology.
            Return ONLY the comma-separated terms without explanations or other text.
            """

            response = self.gemini_model.generate_content(prompt)
            expansion_terms = response.text

            # Clean up the response
            if ',' in expansion_terms:
                terms = [term.strip() for term in expansion_terms.split(',')]
            else:
                terms = expansion_terms.split()

            # Limit to target size
            if len(terms) > expansion_size:
                terms = terms[:expansion_size]

            expanded_query = f"{query} {' '.join(terms)}"
            return expanded_query

        except Exception as e:
            logger.error(f"Query expansion error: {str(e)}", exc_info=True)
            return query  # Return original query on error

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate and near-duplicate results"""
        unique_results = []
        seen_hashes = set()

        for result in results:
            # Create a simplified representation for comparison
            text = result["text"]
            # Use simhash or other fuzzy hash methods for better near-dupe detection
            # Here we use a simple approach by hashing a sample of the text
            text_sample = text[:100] + text[len(text)//2:len(text)//2+100] + text[-100:]
            content_hash = hashlib.md5(text_sample.encode('utf-8')).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_results.append(result)

        return unique_results

    def _group_by_source(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group results by source and sort chunks within each source"""
        grouped = {}

        for result in results:
            source_id = result.get("source_id", "unknown")

            if source_id not in grouped:
                grouped[source_id] = []

            grouped[source_id].append(result)

        # Sort chunks within each source by chunk_index if available
        for source_id, chunks in grouped.items():
            if chunks and "chunk_index" in chunks[0]:
                grouped[source_id] = sorted(chunks, key=lambda x: x.get("chunk_index", 0))

        return grouped

    def _consolidate_chunks(self, grouped_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Consolidate small chunks into larger, meaningful units
        This addresses the paper's finding about embedding model performance deteriorating in longer contexts
        """
        consolidated = []

        for source_id, chunks in grouped_results.items():
            current_chunk = {
                "text": "",
                "source_id": source_id,
                "chunk_parts": [],
                "token_estimate": 0,
                "keywords": set(),
                "scores": []
            }

            for chunk in chunks:
                # Estimate tokens in this chunk
                token_estimate = len(chunk["text"].split()) * 1.3  # Rough approximation

                # If adding this chunk would exceed our desired size, finalize current chunk
                if current_chunk["token_estimate"] + token_estimate > self.max_chunk_tokens and current_chunk["text"]:
                    # Finalize current chunk
                    current_chunk["keywords"] = list(current_chunk["keywords"])

                    # Calculate and set scores
                    if current_chunk["scores"]:
                        avg_score = np.mean(current_chunk["scores"])
                        current_chunk["avg_score"] = avg_score
                        current_chunk["score"] = avg_score  # Set standard score field
                        current_chunk["rerank_score"] = avg_score  # Set rerank_score field
                    else:
                        current_chunk["avg_score"] = 0.5
                        current_chunk["score"] = 0.5
                        current_chunk["rerank_score"] = 0.5

                    consolidated.append(current_chunk)

                    # Start a new chunk
                    current_chunk = {
                        "text": "",
                        "source_id": source_id,
                        "chunk_parts": [],
                        "token_estimate": 0,
                        "keywords": set(),
                        "scores": []
                    }

                # Add this chunk's content
                if current_chunk["text"]:
                    current_chunk["text"] += "\n\n"
                current_chunk["text"] += chunk["text"]
                current_chunk["chunk_parts"].append(chunk)
                current_chunk["token_estimate"] += token_estimate

                # Add keywords and scores
                if "keywords" in chunk and chunk["keywords"]:
                    current_chunk["keywords"].update(chunk["keywords"])

                if "rerank_score" in chunk:
                    current_chunk["scores"].append(chunk["rerank_score"])
                elif "score" in chunk:
                    current_chunk["scores"].append(chunk["score"])

                # Add source path to consolidated chunk
                if "source_path" in chunk and not current_chunk.get("source_path"):
                    current_chunk["source_path"] = chunk["source_path"]

            # Add the last chunk if it has content
            if current_chunk["text"]:
                current_chunk["keywords"] = list(current_chunk["keywords"])

                # Calculate and set scores
                if current_chunk["scores"]:
                    avg_score = np.mean(current_chunk["scores"])
                    current_chunk["avg_score"] = avg_score
                    current_chunk["score"] = avg_score  # Set standard score field
                    current_chunk["rerank_score"] = avg_score  # Set rerank_score field
                else:
                    current_chunk["avg_score"] = 0.5
                    current_chunk["score"] = 0.5
                    current_chunk["rerank_score"] = 0.5

                consolidated.append(current_chunk)

        return consolidated

    @backoff.on_exception(backoff.expo,
                         (requests.exceptions.RequestException,
                          requests.exceptions.Timeout,
                          requests.exceptions.ConnectionError),
                         max_tries=3)
    def _select_diverse_chunks(self, chunks: List[Dict[str, Any]], topic: str, limit: int) -> List[Dict[str, Any]]:
        """
        Select a diverse set of chunks that provide good coverage of the topic

        Args:
            chunks: Consolidated chunks
            topic: Search topic
            limit: Maximum number of chunks to select

        Returns:
            List of selected chunks
        """
        if len(chunks) <= limit:
            return chunks

        # 1. Get the top 1/3 chunks by score
        top_third = max(1, int(limit / 3))
        by_score = sorted(chunks, key=lambda x: x.get("avg_score", 0), reverse=True)
        selected = by_score[:top_third]
        already_selected_ids = {chunk.get("source_id", "") for chunk in selected}

        # 2. Assess relevance to topic more deeply
        remaining = [c for c in chunks if c not in selected]

        # Create embeddings for deeper relevance assessment
        try:
            topic_embedding_resp = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.jina_api_key}",
                    "Accept": "application/json"
                },
                json={
                    "model": self.config.get('jina', 'embedding_model', default="jina-embeddings-v3"),
                    "input": [topic],
                    "task": "retrieval.query"
                },
                timeout=60
            )
            topic_embedding = topic_embedding_resp.json()["data"][0]["embedding"]

            # Get embeddings for remaining chunks (sample of text to stay within 1K tokens)
            chunk_texts = []
            for chunk in remaining:
                # Take first and last 300 words as representative sample
                words = chunk["text"].split()
                if len(words) > 600:
                    text = " ".join(words[:300] + words[-300:])
                else:
                    text = chunk["text"]
                chunk_texts.append(text)

            # Batch embed - respecting API limits
            batch_size = 20  # Adjust based on API limits
            chunk_embeddings = []

            for i in range(0, len(chunk_texts), batch_size):
                batch = chunk_texts[i:i+batch_size]
                response = requests.post(
                    "https://api.jina.ai/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.jina_api_key}",
                        "Accept": "application/json"
                    },
                    json={
                        "model": self.config.get('jina', 'embedding_model', default="jina-embeddings-v3"),
                        "input": batch,
                        "task": "retrieval.passage"
                    },
                    timeout=60
                )
                batch_embeddings = [item["embedding"] for item in response.json()["data"]]
                chunk_embeddings.extend(batch_embeddings)

            # Calculate similarity scores
            for chunk, embedding in zip(remaining, chunk_embeddings):
                # Cosine similarity
                similarity = self._cosine_similarity(topic_embedding, embedding)
                chunk["topic_similarity"] = similarity

            # Sort by similarity
            remaining.sort(key=lambda x: x.get("topic_similarity", 0), reverse=True)

        except Exception as e:
            logger.warning(f"Error during similarity calculation: {str(e)}", exc_info=True)
            # Fallback sorting
            remaining.sort(key=lambda x: x.get("avg_score", 0), reverse=True)

        # 3. Add diverse chunks from remaining pool, prioritizing different sources
        diverse_pool = []
        source_count = {}

        for chunk in remaining:
            source_id = chunk.get("source_id", "unknown")
            source_count[source_id] = source_count.get(source_id, 0) + 1

            # Prioritize chunks from sources not already well-represented
            chunk["source_redundancy"] = source_count[source_id]
            diverse_pool.append(chunk)

        # Sort by combination of relevance and source diversity
        diverse_pool.sort(key=lambda x: (
            x.get("topic_similarity", 0) * 0.7 -
            x.get("source_redundancy", 1) * 0.3
        ), reverse=True)

        # 4. Add diverse chunks up to the limit
        selected.extend(diverse_pool[:limit - len(selected)])

        return selected

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0

        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            if magnitude1 * magnitude2 == 0:
                return 0
            return dot_product / (magnitude1 * magnitude2)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}", exc_info=True)
            return 0

    def _order_chunks_logically(self, chunks: List[Dict[str, Any]], topic: str) -> List[Dict[str, Any]]:
        """Arrange chunks in a logical order"""
        try:
            # Extract text snippets from each chunk
            chunk_snippets = []
            for i, chunk in enumerate(chunks):
                # Get first 100 and last 50 words
                words = chunk["text"].split()
                snippet = " ".join(words[:100])
                if len(words) > 150:
                    snippet += "... " + " ".join(words[-50:])
                chunk_snippets.append((i, snippet))

            # Prompt Gemini to suggest ordering
            chunks_text = "\n\n".join([f"Chunk {i+1}:\n{snippet}" for i, snippet in chunk_snippets])

            prompt = f"""
            I have content about "{topic}" divided into {len(chunks)} chunks.
            I need to arrange these chunks in a logical order.

            Here are snippets from each chunk:

            {chunks_text}

            Please analyze these chunks and return a comma-separated list of chunk numbers in the optimal reading order.
            ONLY return the comma-separated numbers without any explanation.
            For example: 3,1,5,2,4
            """

            response = self.gemini_model.generate_content(prompt)
            order_text = response.text.strip()

            # Parse the ordering
            try:
                if ',' in order_text:
                    order = [int(x.strip()) - 1 for x in order_text.split(',')]
                else:
                    # Handle case where commas might be missing
                    order = [int(x.strip()) - 1 for x in order_text.split()]

                # Validate order (ensure all chunks are included)
                if set(order) != set(range(len(chunks))):
                    raise ValueError(f"Invalid ordering - missing chunks or duplicate indices: {order_text}")

                # Reorder chunks
                ordered_chunks = [chunks[i] for i in order]
                return ordered_chunks

            except Exception as e:
                logger.warning(f"Error parsing chunk order '{order_text}': {str(e)}", exc_info=True)
                return chunks

        except Exception as e:
            logger.warning(f"Error ordering chunks: {str(e)}", exc_info=True)
            return chunks  # Return original order if ordering fails

    def filter_search(self,
                      query: str,
                      filters: Dict[str, Any],
                      limit: int = 20,
                      use_optimized_retrieval: bool = True) -> List[Dict[str, Any]]:
        """
        Search with filtering criteria

        Args:
            query: Search query
            filters: Dictionary of filter criteria (e.g., {"source_type": ".md"})
            limit: Maximum number of results to return
            use_optimized_retrieval: Whether to use optimized retrieval strategy

        Returns:
            List of content chunks
        """
        logger.debug(f"Filter search: query='{query}', filters={filters}, limit={limit}, optimized={use_optimized_retrieval}")

        if not use_optimized_retrieval:
            # Use direct filtering through Qdrant with query vector
            try:
                # Expand the query and get embedding
                search_query = self._expand_query(query) if use_optimized_retrieval else query
                logger.debug(f"Using search query: '{search_query}'")

                response = requests.post(
                    "https://api.jina.ai/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.jina_api_key}",
                        "Accept": "application/json"
                    },
                    json={
                        "model": self.config.get('jina', 'embedding_model', default="jina-embeddings-v3"),
                        "input": [search_query],
                        "task": "retrieval.query"
                    },
                    timeout=60
                )

                query_embedding = response.json()["data"][0]["embedding"]
                logger.debug("Successfully generated query embedding")

                # Build filter conditions
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        # Handle list of values (any match)
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                any_of=[models.MatchValue(value=v) for v in value]
                            )
                        )
                    else:
                        # Handle single value
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )

                logger.debug(f"Filter conditions: {filter_conditions}")

                # Search with filter
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=limit * 2,
                    score_threshold=0.5,
                    query_filter=models.Filter(
                        must=filter_conditions
                    )
                )

                logger.debug(f"Direct search returned {len(search_results)} results")

                # Process results
                results = []
                for hit in search_results:
                    results.append({
                        "text": hit.payload.get("text", ""),
                        "score": hit.score,
                        "source_id": hit.payload.get("source_id", ""),
                        "source_path": hit.payload.get("source_path", ""),
                        "metadata": hit.payload.get("metadata", {})
                    })

                # Apply reranking
                if len(results) > 1:
                    texts = [result["text"] for result in results]

                    rerank_response = requests.post(
                        "https://api.jina.ai/v1/rerank",
                        headers={
                            "Authorization": f"Bearer {self.jina_api_key}",
                            "Accept": "application/json"
                        },
                        json={
                            "model": self.config.get('jina', 'reranker_model', default="jina-reranker-v2-base-multilingual"),
                            "query": query,
                            "documents": texts,
                            "top_n": min(limit, len(texts))
                        },
                        timeout=60
                    )

                    if rerank_response.status_code == 200:
                        # Create a mapping of text to new rank
                        new_scores = {}
                        for idx, item in enumerate(rerank_response.json()["results"]):
                            new_scores[item["document"]["text"]] = {
                                "new_score": item["relevance_score"],
                                "new_rank": idx
                            }

                        # Apply new scores and sort
                        for result in results:
                            if result["text"] in new_scores:
                                result["rerank_score"] = new_scores[result["text"]]["new_score"]
                                result["rerank_position"] = new_scores[result["text"]]["new_rank"]

                        results.sort(key=lambda x: x.get("rerank_position", 999))
                        logger.debug("Successfully reranked results")

                return results[:limit]

            except Exception as e:
                logger.error(f"Error during filtered search: {str(e)}", exc_info=True)
                return []

        else:
            # Use optimized retrieval first, then filter results
            logger.debug("Using optimized retrieval with post-filtering")
            full_results = self.retrieve_optimized_content(query, limit * 3)
            chunks = full_results["chunks"]

            # Apply filters
            filtered_chunks = []
            for chunk in chunks:
                matches_all = True
                for key, value in filters.items():
                    # Handle special case for metadata
                    if key.startswith("metadata."):
                        meta_key = key.split(".", 1)[1]
                        chunk_value = chunk.get("metadata", {}).get(meta_key)
                    else:
                        chunk_value = chunk.get(key)

                    # Check if value matches
                    if isinstance(value, list):
                        if chunk_value not in value:
                            matches_all = False
                            break
                    elif chunk_value != value:
                        matches_all = False
                        break

                if matches_all:
                    filtered_chunks.append(chunk)

            logger.debug(f"Post-filtering reduced {len(chunks)} chunks to {len(filtered_chunks)} chunks")
            return filtered_chunks[:limit]

    def get_source_content(self, source_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all content from a specific source

        Args:
            source_id: Source ID to retrieve

        Returns:
            List of chunks from the source
        """
        try:
            # Query Qdrant for all points with the given source_id
            scroll_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_id",
                            match=models.MatchValue(value=source_id)
                        )
                    ]
                ),
                limit=100,  # Batch size
                with_payload=True,
                with_vectors=False
            )

            chunks = []

            # Handle different scroll_results formats
            try:
                if isinstance(scroll_results, tuple):
                    # Direct tuple response (points, next_offset)
                    batch = scroll_results[0]
                    for point in batch:
                        chunks.append({
                            "text": point.payload.get("text", ""),
                            "chunk_id": point.payload.get("chunk_id", ""),
                            "chunk_index": point.payload.get("chunk_index", 0),
                            "source_id": source_id,
                            "source_path": point.payload.get("source_path", ""),
                            "metadata": point.payload.get("metadata", {})
                        })
                elif hasattr(scroll_results, '__iter__'):
                    # Iterator response - handle different iterator formats
                    try:
                        for item in scroll_results:
                            # Check if item is a tuple/list with 2 elements (batch, offset)
                            if isinstance(item, (tuple, list)) and len(item) == 2:
                                batch = item[0]  # First element is the batch
                            else:
                                # Item itself might be the batch
                                batch = item

                            # Process batch
                            if isinstance(batch, (list, tuple)) or hasattr(batch, '__iter__'):
                                for point in batch:
                                    if hasattr(point, 'payload'):
                                        chunks.append({
                                            "text": point.payload.get("text", ""),
                                            "chunk_id": point.payload.get("chunk_id", ""),
                                            "chunk_index": point.payload.get("chunk_index", 0),
                                            "source_id": source_id,
                                            "source_path": point.payload.get("source_path", ""),
                                            "metadata": point.payload.get("metadata", {})
                                        })
                    except Exception as iter_error:
                        logger.error(f"Error iterating scroll results: {str(iter_error)}")
                else:
                    logger.warning(f"Unknown scroll_results format: {type(scroll_results)}")
            except Exception as e:
                logger.error(f"Error processing scroll results: {str(e)}")

            # Sort by chunk index
            chunks.sort(key=lambda x: x.get("chunk_index", 0))

            return chunks

        except Exception as e:
            logger.error(f"Error retrieving source content: {str(e)}", exc_info=True)
            return []


# Example usage
if __name__ == "__main__":
    # Load config
    from config import get_config
    config = get_config()

    # Initialize the retriever
    retriever = AdvancedRetriever(
        jina_api_key=config.get('jina', 'api_key', default="your_jina_api_key_here"),
        gemini_api_key=config.get('gemini', 'api_key', default="your_gemini_api_key_here")
    )

    # Perform an optimized search
    result = retriever.search("artificial intelligence ethics", limit=10)

    # Print first result
    if result:
        print(f"Found {len(result)} chunks")
        print(f"First chunk preview: {result[0]['text'][:100]}...")