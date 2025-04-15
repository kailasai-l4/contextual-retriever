import requests
from .provider import EmbeddingProvider

JINA_EMBEDDING_ENDPOINT = "https://api.jina.ai/v1/embeddings"

class JinaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = config.api_key
        self.model = config.model
        self.batch_size = getattr(config, 'batch_size', 100)  # Optional

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _embed_batch(self, texts):
        payload = {
            "input": texts,
            "model": self.model
        }
        response = requests.post(
            JINA_EMBEDDING_ENDPOINT,
            headers=self._headers(),
            json=payload
        )
        if response.status_code != 200:
            raise RuntimeError(f"Jina API error: {response.status_code} {response.text}")
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def get_embeddings(self, texts):
        # Batch if needed
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            results.extend(self._embed_batch(batch))
        return results

    def get_query_embedding(self, query):
        return self.get_embeddings([query])[0] 