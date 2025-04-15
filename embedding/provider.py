from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_embeddings(self, texts):
        pass

    @abstractmethod
    def get_query_embedding(self, query):
        pass 