from .jina_provider import JinaEmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider

def get_embedding_provider(config, provider_name=None):
    provider_name = provider_name or config.default_embedding_provider
    provider_cfg = config.embedding_providers[provider_name]
    if provider_name == 'jina':
        return JinaEmbeddingProvider(provider_cfg)
    elif provider_name == 'openai':
        return OpenAIEmbeddingProvider(provider_cfg)
    else:
        raise ValueError(f"Unknown embedding provider: {provider_name}") 