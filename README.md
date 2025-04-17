# Qdrant-DB RAG Content Retriever API

A FastAPI-based API for document ingestion, semantic search, and retrieval using Qdrant as a vector database.  
Supports batch ingestion, query expansion, collection management, and secure API key authentication.

---

## 🚀 Features

- Batch document ingestion with progress tracking
- Embedding and reranking (OpenAI, Jina, Gemini, etc.)
- Collection CRUD (create, list, get, delete)
- Query expansion
- Secure API key authentication
- Health and status endpoints

---

## ⚙️ Deployment

### 1. Local Setup

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd qdrant-db
   ```

2. **Create and configure your `.env` file**
   - Copy `.env.example` to `.env` and fill in required values, especially `API_KEY`.

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Qdrant (if not using Docker)**
   - [Qdrant Quickstart](https://qdrant.tech/documentation/quick-start/)

5. **Run the API**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

---

### 2. Docker & Docker Compose

1. **Configure `.env`**  
   Ensure `API_KEY` and all required variables are set in `.env`.

2. **Build and start services**
   ```bash
   docker-compose up --build
   ```
   - API will be available at `http://localhost:8333` (or as mapped in `docker-compose.yaml`)
   - Qdrant will be available at `http://localhost:6333`

3. **Environment Variables in Docker**
   - The API loads environment variables from `.env` automatically.
   - `API_KEY` is required for all endpoints except `/health`.

---

## 🔐 API Key Authentication

**All endpoints require API key authentication except `/health`.**  
Include your API key in the request headers:

- Header: `X-API-Key`
- Value: The API key set in your `.env` file

**Example:**
```bash
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_key" \
  -d '{ "query": "What is Qdrant?", "limit": 5, "collection_name": "my_collection" }'
```
If the API key is missing or incorrect, you'll receive a `401 Unauthorized` error.

**The `/health` endpoint is public and does not require authentication.**

---

## 🗂️ API Endpoints

### Health

- `GET /health`  
  Returns `{ "status": "ok" }` if the API is running.  
  **No authentication required.**

---

### Ingest Document

- `POST /process/`  
  Upload and ingest a document.

  **Form-data parameters:**
  - `file` (required): Document file (`.txt`, `.md`, `.json`, `.csv`)
  - `collection_name` (required): Target collection
  - `metadata` (optional): JSON string with extra metadata
  - `chunk_size` (optional): Chunk size in tokens (default: 1000)
  - `overlap_size` (optional): Overlap size in tokens (default: 100)

  **Example:**
  ```bash
  curl -X POST http://localhost:8000/process/ \
    -H "X-API-Key: your_secret_key" \
    -F "file=@sample.txt" \
    -F "collection_name=my_collection"
  ```

---

### Ingest Progress

- `GET /process/ingest-progress/{task_id}`  
  Check the progress of a document ingestion.

  ```bash
  curl -X GET http://localhost:8000/process/ingest-progress/<task_id> -H "X-API-Key: your_secret_key"
  ```

---

### Search

- `POST /search/`  
  Semantic search with optional filtering and query expansion.

  **JSON body:**
  - `query` (required): Query string
  - `limit` (optional): Max results (default: 10)
  - `use_expansion` (optional): Use query expansion (default: true)
  - `collection_name` (optional): Target collection
  - `expansion_model` (optional): Expansion model to use
  - `filter` (optional): Filter object (see below)

  **Example:**
  ```bash
  curl -X POST http://localhost:8000/search/ \
    -H "Content-Type: application/json" \
    -H "X-API-Key: your_secret_key" \
    -d '{
      "query": "What is Qdrant?",
      "limit": 5,
      "collection_name": "my_collection",
      "filter": {
        "must": [
          { "key": "filename", "match": { "value": "file1.md" } }
        ]
      }
    }'
  ```

---

### Collection Management

- `GET /collections/` — List all collections
- `POST /collections/` — Create a new collection
- `GET /collections/{collection_name}` — Get details of a collection
- `DELETE /collections/{collection_name}` — Delete a collection

**Example:**
```bash
curl -X GET http://localhost:8000/collections/ -H "X-API-Key: your_secret_key"
curl -X POST http://localhost:8000/collections/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_key" \
  -d '{ "collection_name": "my_collection", "vector_size": 1024, "distance": "cosine" }'
```

---

## 🔎 Filtering Search Results

You can filter search results by any payload field (e.g., `filename`, `source_path`).  
The filter object supports:

- `must`: All conditions must be true (AND)
- `should`: At least one should be true (OR)
- `must_not`: None should be true (NOT)

**Example:**
```json
"filter": {
  "must": [
    { "key": "filename", "match": { "value": "file1.md" } }
  ]
}
```

**Multiple conditions (AND):**
```json
"filter": {
  "must": [
    { "key": "filename", "match": { "value": "file1.md" } },
    { "key": "source", "match": { "value": "test" } }
  ]
}
```

**Any-of (OR):**
```json
"filter": {
  "must": [
    { "key": "filename", "match": { "any": ["file1.md", "file2.md"] } }
  ]
}
```

**Top-level vs Nested Filtering:**  
- Top-level (recommended): `{ "key": "filename", ... }`
- Nested: `{ "key": "metadata.filename", ... }`

See [Qdrant filter docs](https://qdrant.tech/documentation/concepts/filtering/) for advanced options.

---

## 📝 Environment Variables

See `.env.example` for all options.  
Key variables:
- `API_KEY`: API key for authentication (required)
- `QDRANT_URL`, `QDRANT_PORT`: Qdrant instance details
- Embedding/expansion/reranking provider keys

---

## 🐳 Docker & Compose Notes

- The API loads variables from `.env` (including `API_KEY`) via `env_file` in `docker-compose.yaml`.
- No changes needed to enable authentication in Docker—just set `API_KEY` in `.env` before running Compose.
- The `/health` endpoint is always public for container health checks.

---

## 💡 Notes

- All endpoints except `/health` require API key authentication.
- For batch ingestion, use the provided `upload_directory.py` script.
- For PowerShell, use `Invoke-WebRequest` with `-Headers @{"X-API-Key"="your_secret_key"}`.

---

If you need further customization or have questions, see the source code or contact the maintainer.
