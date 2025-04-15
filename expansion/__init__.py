from .gemini_provider import GeminiExpansionProvider
from .openai_provider import OpenAIExpansionProvider

def get_expansion_provider(config, provider_name=None):
    provider_name = provider_name or config.default_expansion_provider
    provider_cfg = config.expansion_providers[provider_name]
    if provider_name == 'gemini':
        return GeminiExpansionProvider(provider_cfg)
    elif provider_name == 'openai':
        return OpenAIExpansionProvider(provider_cfg)
    else:
        raise ValueError(f"Unknown expansion provider: {provider_name}") 