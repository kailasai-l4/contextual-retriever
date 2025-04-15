from .provider import ExpansionProvider

class GeminiExpansionProvider(ExpansionProvider):
    def __init__(self, config):
        super().__init__(config)

    def expand_query(self, query, max_terms=100):
        raise NotImplementedError("Gemini expansion not implemented yet.") 