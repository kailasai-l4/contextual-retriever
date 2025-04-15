import uuid

class Processor:
    def __init__(self, chunker, embedding_provider, storage_manager):
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.storage_manager = storage_manager

    def process_document(self, document, metadata=None):
        collection_name = metadata.get("collection_name") if metadata else "content_library"
        base_meta = dict(metadata or {})
        # 1. Chunk the document
        chunks = self.chunker.chunk(document, metadata=base_meta)
        # 2. Embed and upsert each chunk
        points = []
        for chunk in chunks:
            chunk_text = chunk["text"]
            chunk_meta = chunk["metadata"]
            vector = self.embedding_provider.get_query_embedding(chunk_text)
            point_id = str(uuid.uuid4())
            payload = {
                "text": chunk_text,
                "metadata": chunk_meta
            }
            points.append({
                "id": point_id,
                "vector": vector,
                "payload": payload
            })
        if points:
            self.storage_manager.upsert_vectors(collection_name, points)
        return {"chunks": len(points), "collection": collection_name, "point_ids": [p["id"] for p in points]}