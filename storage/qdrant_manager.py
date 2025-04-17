class QdrantManager:
    def __init__(self, client):
        self.client = client

    def create_collection(self, collection_name, vector_size=1024, distance="cosine"):
        from qdrant_client.models import VectorParams, Distance
        dist = getattr(Distance, distance.upper(), Distance.COSINE)
        return self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=dist)
        )

    def list_collections(self):
        return self.client.get_collections().collections

    def get_collection(self, collection_name):
        return self.client.get_collection(collection_name=collection_name)

    def delete_collection(self, collection_name):
        return self.client.delete_collection(collection_name=collection_name)

    def upsert_vectors(self, collection_name, points):
        # Upsert points into the specified collection using qdrant-client
        from qdrant_client.models import PointStruct
        qdrant_points = [
            PointStruct(
                id=point["id"],
                vector=point["vector"],
                payload=point["payload"]
            ) for point in points
        ]
        self.client.upsert(collection_name=collection_name, points=qdrant_points)

    def search(self, collection_name, query_vector, limit=10, score_threshold=0.5, filter=None):
        # Minimal implementation for end-to-end test
        # Uses the qdrant-client to search for similar vectors
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=filter  # Pass Qdrant filter as query_filter
        )
        # Flatten payload to match API response model
        results = []
        for hit in search_result:
            payload = hit.payload or {}
            results.append({
                "id": hit.id,
                "score": hit.score,
                "text": payload.get("text", ""),
                "source_id": payload.get("source_id", ""),
                "source_path": payload.get("source_path", ""),
                "metadata": payload.get("metadata", {}),
                "keywords": payload.get("keywords", []),
            })
        return results