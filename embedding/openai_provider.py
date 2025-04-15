from .provider import EmbeddingProvider

class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config):
        super().__init__(config)

    def get_embeddings(self, texts):
        raise NotImplementedError("OpenAI embedding not implemented yet.")

    def get_query_embedding(self, query):
        raise NotImplementedError("OpenAI query embedding not implemented yet.") 