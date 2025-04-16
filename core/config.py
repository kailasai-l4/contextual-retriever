import os
from typing import Dict, Optional
from pydantic import BaseModel, Field
import json

class ProviderConfig(BaseModel):
    api_key: str
    model: str
    # Add other provider-specific fields as needed

class QdrantConfig(BaseModel):
    url: str = Field(default="qdrant")
    port: int = Field(default=6333)

class Config(BaseModel):
    embedding_providers: Dict[str, ProviderConfig]
    expansion_providers: Dict[str, ProviderConfig]
    default_embedding_provider: str
    default_expansion_provider: str
    qdrant: QdrantConfig

    @classmethod
    def from_env(cls, config_path: Optional[str] = None):
        """
        Load config from environment variables and optionally from a config file (JSON or YAML).
        Environment variables take precedence over file config.
        """
        config_data = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    config_data = json.load(f)
                elif config_path.endswith(('.yaml', '.yml')):
                    import yaml
                    config_data = yaml.safe_load(f)
        # Override with env vars if present
        for provider in ['jina', 'openai']:
            key = os.getenv(f'EMBEDDING_{provider.upper()}_API_KEY')
            model = os.getenv(f'EMBEDDING_{provider.upper()}_MODEL')
            if key and model:
                config_data.setdefault('embedding_providers', {})[provider] = {
                    'api_key': key,
                    'model': model
                }
        for provider in ['gemini', 'openai']:
            key = os.getenv(f'EXPANSION_{provider.upper()}_API_KEY')
            model = os.getenv(f'EXPANSION_{provider.upper()}_MODEL')
            if key and model:
                config_data.setdefault('expansion_providers', {})[provider] = {
                    'api_key': key,
                    'model': model
                }
        default_embedding = os.getenv('DEFAULT_EMBEDDING_PROVIDER')
        if default_embedding:
            config_data['default_embedding_provider'] = default_embedding
        default_expansion = os.getenv('DEFAULT_EXPANSION_PROVIDER')
        if default_expansion:
            config_data['default_expansion_provider'] = default_expansion
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_port = os.getenv('QDRANT_PORT')
        if qdrant_url or qdrant_port:
            config_data.setdefault('qdrant', {})
            if qdrant_url:
                config_data['qdrant']['url'] = qdrant_url
            if qdrant_port:
                config_data['qdrant']['port'] = int(qdrant_port)

        # --- Error reporting for missing required config ---
        missing = []
        if 'embedding_providers' not in config_data or not config_data['embedding_providers']:
            missing.append('embedding_providers (set EMBEDDING_JINA_API_KEY, EMBEDDING_JINA_MODEL, etc. in .env)')
        if 'expansion_providers' not in config_data or not config_data['expansion_providers']:
            missing.append('expansion_providers (set EXPANSION_GEMINI_API_KEY, EXPANSION_GEMINI_MODEL, etc. in .env)')
        if 'default_embedding_provider' not in config_data:
            missing.append('default_embedding_provider (set DEFAULT_EMBEDDING_PROVIDER in .env)')
        if 'default_expansion_provider' not in config_data:
            missing.append('default_expansion_provider (set DEFAULT_EXPANSION_PROVIDER in .env)')
        if missing:
            raise ValueError(
                f"Missing required config values in .env or config file:\n- " + '\n- '.join(missing) +
                "\n\nPlease check your .env file and ensure all required provider keys are set."
            )
        return cls.parse_obj(config_data) 