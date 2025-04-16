from abc import ABC, abstractmethod

class RerankerProvider(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def rerank(self, query, documents, top_n=10):
        """
        Rerank the documents for the given query and return a list of dicts with 'text', 'relevance_score', and 'rank'.
        """
        pass 