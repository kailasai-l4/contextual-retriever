# Qdrant-DB RAG Content Retriever API

A FastAPI-based system for document ingestion, search, and retrieval using Qdrant as a vector database. Supports batch ingestion, query expansion, and collection management.

## Features
- Batch document ingestion with progress tracking
- Embedding and reranking support
- Collection CRUD (create, list, get, delete)
- Query expansion
- Health and status endpoints

## Requirements
- Python 3.8+
- Qdrant running (local or remote)
- Required environment variables in `.env` file (see `.env.example`)
- Install dependencies: `pip install -r requirements.txt`

## API Endpoints & Example Usage

### 1. Health Check
**GET** `/health`

```bash
curl -X GET http://localhost:8001/health
```

---

### 2. Ingest Document
**POST** `/process/`

**Form-data parameters:**
- `file`: (required) Document file to upload (`.txt`, `.md`, `.json`, `.csv`)
- `collection_name`: (required) Name of the collection to ingest into
- `metadata`: (optional) JSON string with extra metadata
- `chunk_size`: (optional) Integer, chunk size in tokens (default: 1000)
- `overlap_size`: (optional) Integer, overlap size in tokens (default: 100)

**Example:**
```bash
curl -X POST http://localhost:8001/process/ \
  -F "file=@sample.txt" \
  -F "collection_name=my_collection" \
  -F "metadata={\"source\":\"test\"}" \
  -F "chunk_size=1000" \
  -F "overlap_size=100"
```

---

### 2b. Ingest All Files in a Directory
**POST** `/process/directory/`

**JSON body parameters:**
- `directory_path`: (required) Path to the directory to recursively ingest from (server-side path)
- `collection_name`: (required) Name of the collection to ingest into
- `metadata`: (optional) Dictionary with extra metadata to attach to each file
- `chunk_size`: (optional) Integer, chunk size in tokens (default: 1000)
- `overlap_size`: (optional) Integer, overlap size in tokens (default: 100)

**Example:**
```bash
curl -X POST http://localhost:8001/process/directory/ \
  -H "Content-Type: application/json" \
  -d '{
    "directory_path": "/absolute/path/to/my_folder",
    "collection_name": "my_collection",
    "metadata": {"source": "batch"},
    "chunk_size": 1000,
    "overlap_size": 100
  }'
```

Returns a summary of all processed files and any errors encountered.

---

### 3. Ingest Progress
**GET** `/process/ingest-progress/{task_id}`

**Example:**
```bash
curl -X GET http://localhost:8001/process/ingest-progress/<task_id>
```

---

### 4. Search
**POST** `/search/`

**JSON body parameters:**
- `query`: (required) Query string
- `limit`: (optional) Max results (default: 10)
- `use_expansion`: (optional) Boolean (default: true)
- `collection_name`: (optional) Collection to search (default: `content_library`)
- `expansion_model`: (optional) Which expansion model to use

**Example:**
```bash
curl -X POST http://localhost:8001/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Qdrant?",
    "limit": 5,
    "use_expansion": true,
    "collection_name": "my_collection",
    "expansion_model": "openai"
  }'
```

---

### 5. List Collections
**GET** `/collections/`

```bash
curl -X GET http://localhost:8001/collections/
```

---

### 6. Create Collection
**POST** `/collections/`

**JSON body parameters:**
- `collection_name`: (required) Name of the collection
- `vector_size`: (optional) Integer, vector dimension (default: 1024)
- `distance`: (optional) String, distance metric (`cosine`, `euclid`, etc.; default: `cosine`)

**Example:**
```bash
curl -X POST http://localhost:8001/collections/ \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_collection",
    "vector_size": 1024,
    "distance": "cosine"
  }'
```

---

### 7. Get Collection Info
**GET** `/collections/{collection_name}`

```bash
curl -X GET http://localhost:8001/collections/my_collection
```

---

### 8. Delete Collection
**DELETE** `/collections/{collection_name}`

```bash
curl -X DELETE http://localhost:8001/collections/my_collection
```

---

## Notes
- All endpoints return JSON responses.
- For authentication, set required API keys in your `.env` file as needed.
- For more advanced usage, see the code and expand as needed.

## License
MIT
