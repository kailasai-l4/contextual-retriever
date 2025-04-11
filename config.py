import os
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load .env file
load_dotenv()

# Configure logging
logger = logging.getLogger("config")

# Global QdrantClient instance for connection reuse
_qdrant_client_instance = None

def get_qdrant_client(url: str = None, port: int = None) -> QdrantClient:
    """
    Get or create the global QdrantClient instance with retry mechanisms

    Args:
        url: Qdrant server URL
        port: Qdrant server port

    Returns:
        QdrantClient instance
    """
    global _qdrant_client_instance
    
    if _qdrant_client_instance is None:
        config = get_config()
        
        # Use provided values or get from config
        qdrant_url = url or config.get('qdrant', 'url')
        qdrant_port = port or config.get('qdrant', 'port')
        
        logger.info(f"Initializing QdrantClient connection to {qdrant_url}:{qdrant_port}")
        
        # Configure client with timeout and retry settings
        _qdrant_client_instance = QdrantClient(
            url=qdrant_url,
            port=qdrant_port,
            timeout=30.0,  # Longer timeout for stability
            prefer_grpc=False  # Use HTTP API for better compatibility
        )
    
    return _qdrant_client_instance

def get_collection_name(collection: str = None) -> str:
    """
    Get the collection name, either the provided one or the default from config

    Args:
        collection: Optional collection name to use

    Returns:
        Collection name
    """
    config = get_config()
    return collection or config.get('qdrant', 'default_collection', default="content_library")

def get_available_collections() -> List[str]:
    """
    Get all available collections configured in the system

    Returns:
        List of collection names
    """
    config = get_config()
    return config.get('qdrant', 'collections', default=["content_library"])

class Config:
    """Configuration manager for RAG Content Retriever that uses only environment variables"""

    # Default values
    DEFAULTS = {
        "jina": {
            "api_key": "",
            "embedding_model": "jina-embeddings-v3",
            "reranker_model": "jina-reranker-v2-base-multilingual"
        },
        "gemini": {
            "api_key": "",
            "model": "gemini-2.0-flash-001"
        },
        "qdrant": {
            "url": "qdrant",
            "port": 6333,
            "default_collection": "content_library",
            "collections": ["content_library"],
            "vector_size": 1024
        },
        "chunking": {
            "max_chunk_tokens": 1000,
            "overlap_tokens": 100
        },
        "embedding": {
            "batch_size": 100,
            "checkpoint_dir": "embedding_checkpoints"
        },
        "retrieval": {
            "max_consolidated_tokens": 4000,
            "default_result_limit": 20
        },
        "logging": {
            "level": "INFO",
            "file": "logs/rag_retriever.log",
            "embedding_log": "logs/embedding_process.log"
        },
        "api": {
            "api_key": ""
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager using environment variables

        Args:
            config_path: Ignored parameter kept for compatibility
        """
        # Start with default configuration
        self.config = self.DEFAULTS.copy()
        
        # Load all configuration from environment variables
        self._load_from_env()
        
        # Validate critical configuration
        self._validate_config()
        
        # Log that we're using environment variables only
        logger.info("Configuration loaded from environment variables")

    def _validate_config(self):
        """Validate critical configuration settings"""
        # Check API keys (don't log actual keys)
        jina_key = self.get('jina', 'api_key')
        if not jina_key:
            logger.warning("No Jina API key provided in environment variables")

        gemini_key = self.get('gemini', 'api_key')
        if not gemini_key:
            logger.warning("No Gemini API key provided in environment variables")

        # Validate Qdrant URL and port
        qdrant_url = self.get('qdrant', 'url')
        if not qdrant_url:
            logger.warning("No Qdrant URL provided - using default: localhost")
            self.config['qdrant']['url'] = 'localhost'

    def _load_from_env(self):
        """Load all configuration from environment variables"""
        # Map of environment variables to config paths
        env_mappings = {
            # API Keys
            "JINA_API_KEY": ["jina", "api_key"],
            "GEMINI_API_KEY": ["gemini", "api_key"],
            "API_KEY": ["api", "api_key"],
            
            # Jina Configuration
            "JINA_EMBEDDING_MODEL": ["jina", "embedding_model"],
            "JINA_RERANKER_MODEL": ["jina", "reranker_model"],
            
            # Gemini Configuration
            "GEMINI_MODEL": ["gemini", "model"],
            
            # Qdrant Configuration
            "QDRANT_URL": ["qdrant", "url"],
            "QDRANT_PORT": ["qdrant", "port"],
            "DEFAULT_COLLECTION": ["qdrant", "default_collection"],
            "COLLECTIONS": ["qdrant", "collections"],  # Comma-separated list of collections
            "VECTOR_SIZE": ["qdrant", "vector_size"],
            
            # Chunking Configuration
            "MAX_CHUNK_TOKENS": ["chunking", "max_chunk_tokens"],
            "OVERLAP_TOKENS": ["chunking", "overlap_tokens"],
            
            # Embedding Configuration
            "EMBEDDING_BATCH_SIZE": ["embedding", "batch_size"],
            "CHECKPOINT_DIR": ["embedding", "checkpoint_dir"],
            
            # Retrieval Configuration
            "MAX_CONSOLIDATED_TOKENS": ["retrieval", "max_consolidated_tokens"],
            "DEFAULT_RESULT_LIMIT": ["retrieval", "default_result_limit"],
            
            # Logging Configuration
            "LOG_LEVEL": ["logging", "level"],
            "LOG_FILE": ["logging", "file"],
            "EMBEDDING_LOG_FILE": ["logging", "embedding_log"]
        }

        for env_var, config_path in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value:
                # Navigate to the correct nested dict
                current = self.config
                for i, path_part in enumerate(config_path):
                    if i == len(config_path) - 1:
                        # Convert type if needed (handle integers, booleans, lists)
                        try:
                            if path_part == "collections" and isinstance(env_value, str):
                                # Handle collections as a comma-separated list
                                current[path_part] = [c.strip() for c in env_value.split(',')]
                            elif isinstance(current[path_part], int):
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

        # For backward compatibility
        # If COLLECTION_NAME is set but COLLECTIONS or DEFAULT_COLLECTION is not
        collection_name = os.environ.get("COLLECTION_NAME")
        if collection_name:
            if "collections" not in self.config["qdrant"] or not self.config["qdrant"]["collections"]:
                self.config["qdrant"]["collections"] = [collection_name]
            if "default_collection" not in self.config["qdrant"] or not self.config["qdrant"]["default_collection"]:
                self.config["qdrant"]["default_collection"] = collection_name

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

# Global config instance
_config_instance = None

def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get or create the global config instance

    Args:
        config_path: Ignored parameter kept for compatibility

    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance