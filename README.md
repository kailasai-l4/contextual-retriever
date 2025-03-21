# RAG Content Retriever

A comprehensive tool for processing documents into vector embeddings and retrieving relevant content using advanced semantic search techniques optimized for building contextual books and documents.

## Features

- Process multiple document types (Markdown, JSON, JSONL, TXT, CSV, YAML)
- Semantic chunking with optimal token boundaries
- Advanced vector search with specialized retrieval strategies
- Query expansion and reranking for improved retrieval
- Support for document filtering and diverse source retrieval
- Stateful processing with checkpoint support

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

### Other Commands

```bash
# Show database statistics
python main.py stats

# Clear database (requires confirmation)
python main.py clear --confirm

# Reset processing state to reprocess all files
python main.py reset --confirm

# Test API connections
python main.py test
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