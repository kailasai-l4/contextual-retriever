import os
import yaml
import json
import logging
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger("config")

class Config:
    """Configuration manager for RAG Content Retriever"""

    # Default values
    DEFAULTS = {
        "jina": {
            "api_key": "",
            "embedding_model": "jina-embeddings-v3",
            "reranker_model": "jina-reranker-v2-base-multilingual"
        },
        "gemini": {
            "api_key": "",
            "model": "gemini-1.5-flash-latest"  # Updated default to match what's used in code
        },
        "qdrant": {
            "url": "localhost",
            "port": 6333,
            "collection_name": "content_library"
        },
        "chunking": {
            "max_chunk_tokens": 1000,
            "overlap_tokens": 100
        },
        "embedding": {
            "batch_size": 10,
            "checkpoint_dir": "embedding_checkpoints"
        },
        "retrieval": {
            "max_consolidated_tokens": 4000,
            "default_result_limit": 20
        },
        "logging": {
            "level": "INFO",
            "file": "rag_retriever.log",
            "embedding_log": "embedding_process.log"
        },
        "api": {
            "api_key": "YOUR_API_KEY_HERE"
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager

        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        self.config = self.DEFAULTS.copy()

        # Try to load configuration from file
        if config_path:
            self._load_config(config_path)
        else:
            # Look for default config locations
            potential_paths = [
                "./config.yaml",
                "./config.yml",
                "./config.json",
                os.path.expanduser("~/.rag_retriever/config.yaml"),
                os.path.expanduser("~/.config/rag_retriever/config.yaml")
            ]

            for path in potential_paths:
                if os.path.exists(path):
                    self._load_config(path)
                    break

        # Override with environment variables
        self._override_from_env()

        # Validate critical configuration
        self._validate_config()

    def _validate_config(self):
        """Validate critical configuration settings"""
        # Check API keys (don't log actual keys)
        jina_key = self.get('jina', 'api_key')
        if not jina_key:
            logger.warning("No Jina API key provided - will need to be provided via environment variable or command line")

        gemini_key = self.get('gemini', 'api_key')
        if not gemini_key:
            logger.warning("No Gemini API key provided - will need to be provided via environment variable or command line")

        # Validate Qdrant URL and port
        qdrant_url = self.get('qdrant', 'url')
        if not qdrant_url:
            logger.warning("No Qdrant URL provided - using default: localhost")
            self.config['qdrant']['url'] = 'localhost'

    def _load_config(self, config_path: str):
        """Load configuration from file"""
        try:
            file_ext = os.path.splitext(config_path)[1].lower()
            with open(config_path, 'r') as f:
                if file_ext in ['.yaml', '.yml']:
                    file_config = yaml.safe_load(f)
                elif file_ext == '.json':
                    file_config = json.load(f)
                else:
                    logger.warning(f"Unsupported config file format: {file_ext}")
                    return

                # Update config with loaded values
                self._update_nested_dict(self.config, file_config)
                logger.info(f"Loaded configuration from {config_path}")

        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")

    def _update_nested_dict(self, base_dict: Dict, update_dict: Dict):
        """Update nested dictionary recursively"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._update_nested_dict(base_dict[key], value)
            else:
                base_dict[key] = value

    def _override_from_env(self):
        """Override config values from environment variables"""
        # Map of environment variables to config paths
        env_mappings = {
            "JINA_API_KEY": ["jina", "api_key"],
            "GEMINI_API_KEY": ["gemini", "api_key"],
            "QDRANT_URL": ["qdrant", "url"],
            "QDRANT_PORT": ["qdrant", "port"],
            "QDRANT_COLLECTION_NAME": ["qdrant", "collection_name"],
            "MAX_CHUNK_TOKENS": ["chunking", "max_chunk_tokens"],
            "EMBEDDING_BATCH_SIZE": ["embedding", "batch_size"],
            "LOG_LEVEL": ["logging", "level"]
        }

        for env_var, config_path in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value:
                # Navigate to the correct nested dict
                current = self.config
                for i, path_part in enumerate(config_path):
                    if i == len(config_path) - 1:
                        # Convert type if needed (handle integers, booleans)
                        try:
                            if isinstance(current[path_part], int):
                                current[path_part] = int(env_value)
                            elif isinstance(current[path_part], bool):
                                current[path_part] = env_value.lower() in ['true', 'yes', '1', 'y']
                            else:
                                current[path_part] = env_value
                        except (ValueError, TypeError):
                            # Fall back to string if conversion fails
                            current[path_part] = env_value
                    else:
                        current = current[path_part]

    def get(self, *keys, default=None):
        """
        Get a configuration value using dot notation

        Example:
            config.get("jina", "api_key")
            config.get("qdrant", "url")

        Args:
            *keys: Sequence of keys to navigate the config hierarchy
            default: Default value if key not found

        Returns:
            The configuration value or default
        """
        current = self.config
        for key in keys:
            if key not in current:
                return default
            current = current[key]
        return current

    def __getitem__(self, key):
        """Allow dictionary-like access to top-level config sections"""
        return self.config.get(key, {})

    def create_default_config_file(self, output_path: str = "config.yaml"):
        """
        Create a default configuration file with placeholders

        Args:
            output_path: Path to write the configuration file
        """
        # Create a sanitized copy with empty API keys
        sanitized = self.DEFAULTS.copy()
        sanitized["jina"]["api_key"] = "YOUR_JINA_API_KEY_HERE"
        sanitized["gemini"]["api_key"] = "YOUR_GEMINI_API_KEY_HERE"

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            # Write the config file
            with open(output_path, 'w') as f:
                if output_path.endswith('.json'):
                    json.dump(sanitized, f, indent=2)
                else:
                    yaml.dump(sanitized, f, default_flow_style=False, sort_keys=False)

            print(f"Created default configuration file: {output_path}")

        except Exception as e:
            logger.error(f"Error creating default config file: {str(e)}")
            print(f"Error creating default config file: {str(e)}")

# Global config instance
_config_instance = None

def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get or create the global config instance

    Args:
        config_path: Optional path to config file

    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance

# Example usage
if __name__ == "__main__":
    # Create a default config file
    config = Config()
    config.create_default_config_file()

    # Access config values
    print(f"Qdrant URL: {config.get('qdrant', 'url')}")
    print(f"Max chunk tokens: {config.get('chunking', 'max_chunk_tokens')}")