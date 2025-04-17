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
curl -X GET http://localhost:8000/health
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
curl -X POST http://localhost:8000/process/ \
  -F "file=@sample.txt" \
  -F "collection_name=my_collection" \
  -F "metadata={\"source\":\"test\"}" \
  -F "chunk_size=1000" \
  -F "overlap_size=100"
```

**Note:** Only single file uploads are supported by the API. For bulk ingestion, you can loop over this endpoint in a script to ingest multiple files.

---

### 3. Ingest Progress
**GET** `/process/ingest-progress/{task_id}`

**Example:**
```bash
curl -X GET http://localhost:8000/process/ingest-progress/<task_id>
```

---

### 4. Search
**POST** `/search/`

**JSON body parameters:**
- `query`: (required) Query string
- `limit`: (optional) Max results (default: 10)
- `use_expansion`: (optional) Boolean (default: true)
- `collection_name`: (optional) Collection to search
- `expansion_model`: (optional) Which expansion model to use
- `filter`: (optional) Filter object (see below)

**Filter Object:**
- `must`: (optional) List of filter conditions (see below)

**Filter Condition:**
- `key`: (required) Field to filter on
- `match`: (required) Match object (see below)

**Match Object:**
- `value`: (optional) Value to match
- `any`: (optional) List of values to match any of

**Example:**
```bash
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Qdrant?",
    "limit": 5,
    "use_expansion": true,
    "collection_name": "my_collection",
    "expansion_model": "openai",
    "filter": {
      "must": [
        { "key": "filename", "match": { "value": "A Garland of Memories - Devotees' Reminiscences of Time Spent With Swamiji_English_processed (1).md" } }
      ]
    }
  }'
```

**Example with `source_path`:**
```bash
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Qdrant?",
    "limit": 5,
    "use_expansion": true,
    "collection_name": "my_collection",
    "expansion_model": "openai",
    "filter": {
      "must": [
        { "key": "source_path", "match": { "value": "uploads/uploads/uploads copy/A Garland of Memories - Devotees' Reminiscences of Time Spent With Swamiji_English_processed (1).md" } }
      ]
    }
  }'
```

**Example with `any` syntax:**
```bash
curl -X POST http://localhost:8001/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Qdrant?",
    "limit": 5,
    "use_expansion": true,
    "collection_name": "my_collection",
    "expansion_model": "openai",
    "filter": {
      "must": [
        { "key": "filename", "match": { "any": ["file1.md", "file2.md"] } }
      ]
    }
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
curl -X POST http://localhost:8000/collections/ \
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

## 🔎 Filtering Search Results by Document

You can filter search results to only include results from a specific file (by filename or source_path) or any payload field.

Depending on your ingestion logic, the `filename` may be present as a top-level field or nested inside `metadata`.

### 1. Filter by Top-Level `filename` (Recommended)
If your data was ingested after April 2025, or you have re-ingested with the latest code, `filename` is included as a top-level field in the payload. Use:

```json
"filter": {
  "must": [
    { "key": "filename", "match": { "value": "A Garland of Memories - Devotees' Reminiscences of Time Spent With Swamiji_English_processed (1).md" } }
  ]
}
```

### 2. Filter by Nested `metadata.filename`
If your data was ingested before this update, or you want to support both cases, use:

```json
"filter": {
  "must": [
    { "key": "metadata.filename", "match": { "value": "A Garland of Memories - Devotees' Reminiscences of Time Spent With Swamiji_English_processed (1).md" } }
  ]
}
```

**Note:**
- If you re-ingest documents after updating your code, both fields will be present and both filter methods will work.
- For best results and future compatibility, prefer the top-level `filename` approach.

#### Ingestion Logic for Top-Level `filename`
When ingesting documents, the code ensures `filename` is included at the top level of the Qdrant payload:

```python
payload = {
    "text": chunk["text"],
    "metadata": chunk_meta
}
if "filename" in chunk_meta:
    payload["filename"] = chunk_meta["filename"]
```

For more details, see [Qdrant’s filter documentation](https://qdrant.tech/documentation/concepts/filtering/).

---

### 🔍 Filter Syntax: What Does `must` Mean?

In the filter object, the `must` field specifies a list of conditions that **all must be satisfied** for a result to match (logical AND). This is based on Qdrant's boolean filtering logic.

- **`must`**: All conditions must be true (AND logic)
- **`should`**: At least one should be true (OR logic)
- **`must_not`**: None should be true (NOT logic)

#### Example: Multiple `must` Conditions

```json
"filter": {
  "must": [
    { "key": "filename", "match": { "value": "file1.md" } },
    { "key": "source", "match": { "value": "test" } }
  ]
}
```
This will only return results where **both** `filename` is `file1.md` **AND** `source` is `test`.

See [Qdrant filtering documentation](https://qdrant.tech/documentation/concepts/filtering/) for more advanced options.

## 🧪 Testing
- Add tests for the new filter functionality in your search endpoint to ensure correct results.

---

## Notes
- All endpoints return JSON responses.
- For authentication, set required API keys in your `.env` file as needed.
- For more advanced usage, see the code and expand as needed.

## License
MIT
