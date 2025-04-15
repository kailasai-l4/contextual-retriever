from abc import ABC, abstractmethod

class ExpansionProvider(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def expand_query(self, query, max_terms=100):
        pass 