import requests
from .provider import RerankerProvider

JINA_RERANK_ENDPOINT = "https://api.jina.ai/v1/rerank"

class JinaRerankerProvider(RerankerProvider):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = config.api_key
        self.model = config.model

    def rerank(self, query, documents, top_n=10):
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents))
        }
        response = requests.post(
            JINA_RERANK_ENDPOINT,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json"
            },
            json=payload
        )
        if response.status_code != 200:
            raise RuntimeError(f"Jina rerank API error: {response.status_code} {response.text}")
        data = response.json()
        results = []
        for idx, item in enumerate(data["results"]):
            results.append({
                "text": item["document"]["text"],
                "relevance_score": item["relevance_score"],
                "rank": idx
            })
        return results 