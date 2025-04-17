import uuid
import logging
import math
import os

logger = logging.getLogger("processing.processor")

class Processor:
    def __init__(self, chunker, embedding_provider, storage_manager, embedding_batch_size=None):
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.storage_manager = storage_manager
        # Allow batch size override, else from env/config
        self.embedding_batch_size = embedding_batch_size or int(os.getenv("EMBEDDING_BATCH_SIZE", 100))

    def process_document(self, document, metadata=None, progress_callback=None):
        collection_name = metadata.get("collection_name") if metadata else "content_library"
        filename = metadata.get("filename") if metadata else None
        base_meta = dict(metadata or {})
        # 1. Chunk the document
        chunks = self.chunker.chunk(document, metadata=base_meta)
        file_info = f"File '{filename}': " if filename else ""
        logger.info(f"{file_info}Document split into {len(chunks)} chunks for collection '{collection_name}'")
        total_chunks = len(chunks)
        points = []
        # 2. Batch embedding and upsert
        for start in range(0, total_chunks, self.embedding_batch_size):
            end = min(start + self.embedding_batch_size, total_chunks)
            batch_chunks = chunks[start:end]
            batch_texts = [chunk["text"] for chunk in batch_chunks]
            # Batch embedding (assume provider supports it, else fallback)
            if hasattr(self.embedding_provider, "get_query_embedding_batch"):
                batch_vectors = self.embedding_provider.get_query_embedding_batch(batch_texts)
            else:
                batch_vectors = [self.embedding_provider.get_query_embedding(text) for text in batch_texts]
            for i, (chunk, vector) in enumerate(zip(batch_chunks, batch_vectors)):
                chunk_meta = chunk["metadata"]
                point_id = str(uuid.uuid4())
                payload = {
                    "text": chunk["text"],
                    "metadata": chunk_meta
                }
                # Add filename at top-level for Qdrant filtering
                if "filename" in chunk_meta:
                    payload["filename"] = chunk_meta["filename"]
                points.append({
                    "id": point_id,
                    "vector": vector,
                    "payload": payload
                })
            # Upsert this batch
            logger.info(f"{file_info}Upserting batch {start+1}-{end} of {total_chunks} to collection '{collection_name}'...")
            self.storage_manager.upsert_vectors(collection_name, points[-len(batch_chunks):])
            # Progress callback
            if progress_callback:
                progress_callback({
                    "processed": end,
                    "total": total_chunks,
                    "percent": round(100*end/total_chunks, 1)
                })
        logger.info(f"{file_info}Ingestion complete: {len(points)} chunks upserted to collection '{collection_name}'")
        if progress_callback:
            progress_callback({"processed": total_chunks, "total": total_chunks, "percent": 100, "done": True})
        return {"chunks": len(points), "collection": collection_name, "point_ids": [p["id"] for p in points]}