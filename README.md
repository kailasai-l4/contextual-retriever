# RAG Content Retriever

A comprehensive tool for processing documents into vector embeddings and retrieving relevant content using advanced semantic search techniques optimized for building contextual books and documents.

## Features

- Process multiple document types (Markdown, JSON, JSONL, TXT, CSV, YAML)
- Semantic chunking with optimal token boundaries
- Advanced vector search with specialized retrieval strategies
- Query expansion and reranking for improved retrieval
- Support for document filtering and diverse source retrieval
- Stateful processing with checkpoint support
- Enhanced statistics and dashboard visualization
- Docker support for easy deployment and scalability
- RESTful API with robust authentication
- **NEW**: Health monitoring endpoints for production deployment
- **NEW**: Improved error handling and stability
- **NEW**: Singleton connection management for better performance

## Recent Improvements

We've made several significant improvements to the codebase to enhance stability and performance:

- **Connection Management**: Implemented a singleton pattern for Qdrant client to prevent multiple connections
- **Error Handling**: Added robust error handling with detailed logs and backoff strategies
- **Health Monitoring**: Added `/health`, `/readiness`, and `/liveness` endpoints for Kubernetes compatibility
- **Qdrant Issue Detection**: Added capability to detect and clear Qdrant issues automatically
- **Testing**: Enhanced test scripts for better error reporting and API validation
- **Configuration**: Improved environment variable handling and validation

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for a complete list of recent enhancements.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/rag-content-retriever.git
cd rag-content-retriever
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys as environment variables:
```bash
export JINA_API_KEY=your_jina_key_here
export GEMINI_API_KEY=your_gemini_key_here
```

4. Run the system with the provided configuration or create your own:
```bash
python main.py config --output config.yaml
```

## Requirements

- Python 3.9+
- Jina AI API key ([Get one here](https://jina.ai/))
- Google Gemini API key ([Get one here](https://ai.google.dev/))
- Qdrant vector database (local or remote)

## Usage

### Creating Configuration

```bash
# Create a default configuration file
python main.py config --output config.yaml
```

### Processing Documents

```bash
# Process documents in a directory
python main.py process --path /path/to/documents --recursive

# Process only specific file types
python main.py process --path /path/to/documents --file-types .md .txt
```

### Searching Content

```bash
# Search for specific content
python main.py search --query "Your search query here"

# Use simple search (without optimizations)
python main.py search --query "Your search query" --simple

# Limit results
python main.py search --query "Your search query" --limit 10

# Filter results
python main.py search --query "Your search query" --filter "source_type=.md" "keyword=important"

# Save results to file
python main.py search --query "Your search query" --output results.json
```

### REST API

The system includes a FastAPI-based REST API for integration with other services:

```bash
# Start the API server
python api.py 

# Or with custom host/port
python api.py --host 0.0.0.0 --port 8080
```

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for complete API reference.

### System Statistics and Monitoring

```bash
# Show database statistics
python main.py stats

# Test API connections and check Qdrant issues
python main.py test --verbose

# Test local API endpoints
python tests/test_local_api.py

# Test search functionality
python specific_search.py "your search query"

# Show detailed stats with visualization
python test_enhanced_stats.py

# Run the comprehensive dashboard view
python simple_dashboard_test.py

# Clear database (requires confirmation)
python main.py clear --confirm

# Reset processing state to reprocess all files
python main.py reset --confirm
```

### Health Monitoring

The API now includes dedicated health check endpoints for production monitoring:

```bash
# Check API health (includes Qdrant status)
curl http://localhost:8000/health

# Kubernetes readiness probe
curl http://localhost:8000/readiness

# Kubernetes liveness probe
curl http://localhost:8000/liveness
```

### Docker Deployment

The system can be deployed using Docker for easier management:

```bash
# Build the Docker image
docker build -t rag-content-retriever .

# Run the container
docker run -p 8000:8000 -v /path/to/data:/app/data -v /path/to/.env:/app/.env rag-content-retriever

# Run with Docker Compose (recommended for production)
docker-compose up -d
```

## How It Works

This tool implements advanced retrieval techniques based on research on embedding model limitations, including:

1. Semantic chunking within optimal token boundaries (1000 tokens)
2. Overlap between chunks to preserve context
3. Query expansion for better recall
4. Hybrid search with multi-query and specialized token handling
5. Diverse source retrieval to ensure comprehensive coverage
6. Reranking for improved result relevance

## Advanced Options

You can override configuration options via command line:

```bash
python main.py search --query "Your search" --jina-key YOUR_KEY --qdrant-url custom-url --qdrant-port 6334
```

For detailed logging:
```bash
python main.py --log-details process --path /path/to/docs
```

## License

MIT