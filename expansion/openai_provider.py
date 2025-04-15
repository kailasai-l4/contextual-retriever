from .provider import ExpansionProvider
import requests

class OpenAIExpansionProvider(ExpansionProvider):
    def __init__(self, config):
        super().__init__(config)
        # Support both dict and pydantic config
        self.api_key = getattr(config, 'api_key', None) or getattr(config, 'API_KEY', None) or getattr(config, 'EXPANSION_OPENAI_API_KEY', None)
        self.model = getattr(config, 'model', None) or getattr(config, 'MODEL', None) or getattr(config, 'EXPANSION_OPENAI_MODEL', None)

    def expand_query(self, query, max_terms=100):
        api_key = self.api_key
        model = self.model
        if not api_key or api_key == 'None':
            raise RuntimeError('OpenAI API key for expansion is missing or invalid.')
        if not model:
            raise RuntimeError('OpenAI model for expansion is missing.')
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        prompt = f"Expand the following search query with synonyms, related terms, and rephrasings (comma separated, up to {max_terms} terms): {query}"
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful search assistant that expands user queries for better search results."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 64,
            "temperature": 0.3
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=10)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI API error: {resp.status_code} {resp.text}")
        content = resp.json()["choices"][0]["message"]["content"].strip()
        return content