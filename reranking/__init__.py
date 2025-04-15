from .jina_provider import JinaRerankerProvider

def get_reranker_provider(config, provider_name=None):
    provider_name = provider_name or getattr(config, 'default_reranker_provider', 'jina')
    provider_cfg = config.embedding_providers.get(provider_name)  # Use embedding_providers for reranker config
    if provider_name == 'jina':
        return JinaRerankerProvider(provider_cfg)
    else:
        raise ValueError(f"Unknown reranker provider: {provider_name}") 