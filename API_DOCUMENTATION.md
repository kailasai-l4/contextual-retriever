# API Documentation

This document provides detailed information about the RESTful API endpoints available in the RAG Content Retriever system.

## Authentication

All API endpoints require authentication using an API key. The key should be provided in the `X-API-Key` header.

Example:
```
X-API-Key: your_api_key_here
```

The API key is defined in the `.env` file or as an environment variable `API_KEY`.

> **Note**: The health monitoring endpoints (`/health`, `/readiness`, `/liveness`) do not require authentication.

## API Endpoints

### Root Endpoint

```
GET /
```

Returns basic information about the API and available endpoints.

**Example Response:**
```json
{
  "success": true,
  "data": {
    "name": "RAG API",
    "version": "1.0.0",
    "description": "API for RAG Content Retriever",
    "endpoints": [
      {"path": "/", "method": "GET", "description": "Root endpoint"},
      {"path": "/search", "method": "POST", "description": "Search for content"},
      {"path": "/process", "method": "POST", "description": "Process documents into vector database"},
      {"path": "/stats", "method": "GET", "description": "Get statistics about the vector database"},
      {"path": "/collections", "method": "GET", "description": "List available collections"},
      {"path": "/collections/{name}", "method": "POST", "description": "Create a new collection"},
      {"path": "/collections/{name}", "method": "GET", "description": "Get collection details"},
      {"path": "/bulk/process", "method": "POST", "description": "Process bulk data"},
      {"path": "/bulk/embeddings", "method": "POST", "description": "Upload pre-computed embeddings"},
      {"path": "/sources/{source_id}", "method": "GET", "description": "Get all content from a specific source"},
      {"path": "/dashboard", "method": "GET", "description": "Get a comprehensive dashboard view of the database statistics"},
      {"path": "/health", "method": "GET", "description": "Health check endpoint for monitoring systems"},
      {"path": "/readiness", "method": "GET", "description": "Kubernetes readiness probe endpoint"},
      {"path": "/liveness", "method": "GET", "description": "Kubernetes liveness probe endpoint"}
    ]
  },
  "message": null,
  "duration_ms": 1.2
}
```

### Health Monitoring Endpoints

#### Health Check Endpoint

```
GET /health
```

Returns the health status of the API and its dependencies (e.g., Qdrant). This endpoint does not require authentication.

**Example Response:**
```json
{
  "status": "ok",
  "details": {
    "api": "ok",
    "qdrant": {
      "status": "ok",
      "details": "Connected: 1 collections available"
    }
  },
  "timestamp": "2025-03-30T20:25:44.111894"
}
```

Possible status values:
- `ok`: All systems are functioning properly
- `degraded`: Some components are experiencing issues but the system is still operational
- `error`: Critical system components are not functioning

#### Readiness Probe

```
GET /readiness
```

Kubernetes readiness probe endpoint. Returns 200 OK if the system is ready to accept requests, 503 Service Unavailable otherwise. This endpoint does not require authentication.

**Example Response:**
```json
{
  "status": "ready"
}
```

#### Liveness Probe

```
GET /liveness
```

Kubernetes liveness probe endpoint. Returns 200 OK if the API is running. This endpoint does not require authentication.

**Example Response:**
```json
{
  "status": "alive"
}
```

### Collection Management Endpoints

#### List Collections

```
GET /collections
```

Returns a list of all available collections in the system.

**Example Response:**
```json
{
  "success": true,
  "data": [
    {
      "name": "content_library",
      "vectors_count": 625,
      "points_count": 625,
      "vector_size": 1024,
      "distance_metric": "Cosine"
    },
    {
      "name": "research_data",
      "vectors_count": 318,
      "points_count": 318,
      "vector_size": 1024,
      "distance_metric": "Cosine"
    }
  ],
  "message": "Found 2 collections",
  "duration_ms": 35.6
}
```

#### Create Collection

```
POST /collections/{name}
```

Creates a new collection with the specified name and parameters.

**URL Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| name | string | Yes | Name of the collection to create |

**Request Body:**
```json
{
  "name": "my_collection",
  "description": "My collection for research data",
  "vector_size": 1024
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| name | string | Yes | Name of the collection (from URL) |
| description | string | No | Description of the collection |
| vector_size | integer | No | Vector dimension size for embeddings (defaults to 1024) |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "name": "my_collection",
    "vector_size": 1024,
    "description": "My collection for research data"
  },
  "message": "Collection 'my_collection' created or accessed successfully",
  "duration_ms": 120.4
}
```

#### Get Collection Details

```
GET /collections/{name}
```

Returns detailed information about a specific collection.

**URL Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| name | string | Yes | Name of the collection to retrieve |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "name": "content_library",
    "vectors_count": 625,
    "points_count": 625,
    "vector_size": 1024,
    "distance_metric": "Cosine",
    "segments_count": 10,
    "payload_schema": {
      "text": "string",
      "source_id": "string",
      "source_path": "string",
      "source_type": "string",
      "chunk_id": "string"
    },
    "status": "available"
  },
  "message": "Collection 'content_library' details retrieved",
  "duration_ms": 46.7
}
```

#### Delete Collection

```
DELETE /collections/{name}
```

Deletes a collection permanently.

**URL Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| name | string | Yes | Name of the collection to delete |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "name": "my_collection"
  },
  "message": "Collection 'my_collection' deleted successfully",
  "duration_ms": 78.3
}
```

### Search Endpoint

```
POST /search
```

Searches the vector database for content based on the provided query.

**Request Body:**
```json
{
  "query": "Your search query",
  "limit": 10,
  "use_optimized_retrieval": true,
  "filters": {
    "source_type": ".md",
    "metadata.custom_field": "value"
  },
  "collection": "content_library",
  "search_all_collections": false
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | string | Yes | The search query text |
| limit | integer | No | Maximum number of results to return (default: 20) |
| use_optimized_retrieval | boolean | No | Whether to use optimized search strategies (default: true) |
| filters | object | No | Filters to apply to search results |
| collection | string | No | Specific collection to search (if not provided, uses default) |
| search_all_collections | boolean | No | Whether to search across all available collections (default: false) |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "query": "Your search query",
    "count": 3,
    "results": [
      {
        "text": "This is the content of the first result",
        "source_id": "document1.md",
        "source_path": "/path/to/document1.md",
        "score": 0.92,
        "rerank_score": 0.95,
        "collection": "content_library",
        "metadata": {
          "source_type": ".md"
        }
      },
      // ... more results
    ],
    "collections_searched": ["content_library"]
  },
  "message": "Found 3 results",
  "duration_ms": 245.8
}
```

### Process Endpoint

```
POST /process
```

Processes documents into the vector database.

**Request Body:**
```json
{
  "directory_path": "/path/to/documents",
  "recursive": true,
  "file_types": [".md", ".txt"],
  "collection": "my_collection"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| directory_path | string | Yes | Path to the directory containing documents to process |
| recursive | boolean | No | Whether to process directories recursively (default: true) |
| file_types | array | No | File extensions to process (e.g., [".md", ".txt"]) |
| collection | string | No | Target collection for processed documents (defaults to current collection) |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "directory_path": "/path/to/documents",
    "recursive": true,
    "file_types": [".md", ".txt"],
    "collection": "my_collection"
  },
  "message": "Document processing started in the background",
  "duration_ms": 20.4
}
```

### Bulk Processing Endpoints

#### Process Bulk Data

```
POST /bulk/process
```

Processes structured data directly into the vector database, without needing to read from files.

**Request Body:**
```json
{
  "data": [
    {
      "text": "This is the content of the first item to process",
      "metadata": {
        "title": "First Item",
        "author": "John Doe",
        "category": "Documentation"
      },
      "source_id": "custom-id-1"
    },
    {
      "text": "This is the content of the second item to process",
      "metadata": {
        "title": "Second Item",
        "author": "Jane Smith"
      }
    }
  ],
  "collection": "my_collection"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| data | array | Yes | Array of data items to process |
| data[].text | string | Yes | Text content to embed |
| data[].metadata | object | No | Metadata to store with the vector |
| data[].source_id | string | No | Custom source ID (generated if not provided) |
| collection | string | No | Target collection for processed data (defaults to current collection) |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "items_count": 2,
    "collection": "my_collection",
    "task_id": "bulk_1679823456"
  },
  "message": "Bulk processing of 2 items started in the background",
  "duration_ms": 15.2
}
```

#### Upload Pre-computed Embeddings

```
POST /bulk/embeddings
```

Uploads pre-computed embeddings directly to the database, bypassing the embedding generation step.

**Request Body:**
```json
{
  "embeddings": [
    {
      "id": "doc-1",
      "vector": [0.1, 0.2, 0.3, ..., 0.9],
      "payload": {
        "text": "This is the text that corresponds to the vector",
        "source_id": "source-1",
        "metadata": {
          "title": "Document 1"
        }
      }
    },
    {
      "id": 42,
      "vector": [0.2, 0.3, 0.4, ..., 0.8],
      "payload": {
        "text": "Another document with a different vector",
        "source_id": "source-2"
      }
    }
  ],
  "collection": "my_collection"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| embeddings | array | Yes | Array of embedding records |
| embeddings[].id | string/integer | Yes | Unique identifier for the embedding |
| embeddings[].vector | array | Yes | Vector embedding (must match collection vector dimension) |
| embeddings[].payload | object | No | Metadata to store with the vector |
| collection | string | No | Target collection for embeddings (defaults to current collection) |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "embeddings_count": 2,
    "collection": "my_collection",
    "task_id": "emb_upload_1679823456"
  },
  "message": "Upload of 2 embeddings started in the background",
  "duration_ms": 18.7
}
```

### Statistics Endpoint

```
GET /stats
```

Returns statistics about the vector database. Can be filtered to a specific collection using the `collection` query parameter.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| collection | string | No | Collection to get statistics for (defaults to current collection) |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "summary": {
      "collection_name": "content_library",
      "status": "green",
      "total_chunks": 625,
      "unique_documents": 10,
      "avg_chunks_per_document": 62.5
    },
    "database_info": {
      "vectors_count": 625,
      "segments_count": 10,
      "vector_dimension": 1024
    },
    "content": {
      "file_type_distribution": {
        ".md": 625
      },
      "recently_processed": [
        {
          "source_id": "document1.md",
          "title": "Introduction to RAG Systems"
        },
        // ... more documents
      ]
    }
  },
  "message": "Database statistics retrieved successfully",
  "duration_ms": 350.4
}
```

### Dashboard Endpoint

```
GET /dashboard
```

Returns a comprehensive dashboard view of the database statistics with formatting hints for UI integration.

**Example Response:**
```json
{
  "success": true,
  "data": {
    "metrics": [
      {
        "name": "Total Documents",
        "value": 10,
        "format": "number",
        "icon": "document",
        "description": "Number of unique documents processed"
      },
      // ... more metrics
    ],
    "file_types": {
      "chart_type": "pie",
      "data": [
        {"name": ".md", "value": 625}
      ],
      "title": "Document Types Distribution"
    },
    "collection_info": {
      "name": "content_library",
      "segments": 10,
      "vector_dimension": 1024
    },
    "recent_activity": [
      {
        "title": "Introduction to RAG Systems",
        "id": "document1.md",
        "timestamp": 1679823456
      },
      // ... more documents
    ],
    "health": {
      "status": "healthy",
      "last_checked": "2023-03-26 12:30:56",
      "details": {
        "vectors_status": "available",
        "segments_status": "optimized"
      }
    }
  },
  "message": "Dashboard data retrieved successfully",
  "duration_ms": 163.2
}
```

### Source Content Endpoint

```
GET /sources/{source_id}
```

Returns all content from a specific source.

**URL Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| source_id | string | Yes | ID of the source to retrieve content from |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "source_id": "document1.md",
    "content": [
      {
        "content": "This is the content of the first chunk",
        "chunk_index": 0,
        "content_title": "Chapter 1: Introduction",
        "metadata": {
          "processed_at": 1679823456,
          "processed_time": "2023-03-26 12:30:56",
          "source_type": ".md"
        }
      },
      // ... more chunks
    ],
    "total_chunks": 15
  },
  "message": "Source content retrieved successfully",
  "duration_ms": 85.3
}
```

### Optimized Retrieval Endpoint

```
POST /optimized_retrieval
```

Performs an advanced semantic search with specialized retrieval strategies including query expansion, multi-query generation, and reranking for improved relevance.

**Request Body:**
```json
{
  "query": "Your search query",
  "limit": 10,
  "filters": {
    "source_type": ".md",
    "metadata.custom_field": "value"
  },
  "diversity_weight": 0.5,
  "rerank_results": true,
  "enable_query_expansion": true
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | string | Yes | The search query text |
| limit | integer | No | Maximum number of results to return (default: 20) |
| filters | object | No | Filters to apply to search results |
| diversity_weight | float | No | Weight for source diversity in results (0.0-1.0, default: 0.3) |
| rerank_results | boolean | No | Whether to apply reranking to search results (default: true) |
| enable_query_expansion | boolean | No | Whether to generate expanded query variants (default: true) |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "query": "Your search query",
    "expanded_queries": [
      "Your search query context",
      "search query examples",
      "search query explanation"
    ],
    "count": 5,
    "results": [
      {
        "text": "This is the content of the first result",
        "source_id": "document1.md",
        "source_path": "/path/to/document1.md",
        "score": 0.92,
        "rerank_score": 0.95,
        "metadata": {
          "source_type": ".md"
        }
      },
      // ... more results
    ],
    "metrics": {
      "original_query_time_ms": 120.2,
      "expansion_time_ms": 45.6,
      "reranking_time_ms": 80.0
    }
  },
  "message": "Found 5 results using optimized retrieval",
  "duration_ms": 245.8
}
```

### Upload Endpoint

```
POST /upload
```

Uploads a file and optionally processes it into the vector database.

**Form Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | file | Yes | The file to upload |
| process_now | boolean | No | Whether to process the file immediately (default: true) |
| collection | string | No | Target collection for the uploaded file (defaults to current collection) |

**Example cURL:**
```bash
curl -X POST \
  'http://localhost:8000/upload?process_now=true&collection=my_collection' \
  --header 'X-API-Key: your_api_key_here' \
  --form 'file=@/path/to/your/document.md'
```

**Example Response:**
```json
{
  "success": true,
  "data": {
    "filename": "document.md",
    "path": "uploads/document.md",
    "size": 1254,
    "content_type": "text/markdown",
    "processed": true,
    "collection": "my_collection"
  },
  "message": "File uploaded and processing started in the background",
  "duration_ms": 45.7
}
```

## Error Responses

All endpoints return standardized error responses with appropriate HTTP status codes.

**Example Error Response:**
```json
{
  "success": false,
  "error": "Collection 'unknown_collection' not found",
  "message": "The requested resource could not be found",
  "details": null
}
```

Common error status codes:
- 400: Bad Request - Invalid input parameters
- 403: Forbidden - Invalid API key
- 404: Not Found - Resource not found
- 500: Internal Server Error - Unexpected server error

## API Usage Examples

### Python Example

```python
import requests
import json

API_URL = "http://localhost:8000"
API_KEY = "your_api_key_here"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Search for content
search_data = {
    "query": "How does Qdrant handle vector similarity search?",
    "limit": 5,
    "collection": "documentation"
}

response = requests.post(f"{API_URL}/search", headers=headers, json=search_data)
results = response.json()

if results["success"]:
    print(f"Found {len(results['data']['results'])} results:")
    for idx, item in enumerate(results["data"]["results"], 1):
        print(f"{idx}. {item['text'][:100]}... (score: {item['score']})")
else:
    print(f"Error: {results.get('error')}")
```

### cURL Example

```bash
# Search across all collections
curl -X POST \
  'http://localhost:8000/search' \
  --header 'X-API-Key: your_api_key_here' \
  --header 'Content-Type: application/json' \
  --data-raw '{
    "query": "Qdrant vector database features",
    "limit": 10,
    "search_all_collections": true
  }'

# Create a new collection
curl -X POST \
  'http://localhost:8000/collections/new_collection' \
  --header 'X-API-Key: your_api_key_here' \
  --header 'Content-Type: application/json' \
  --data-raw '{
    "name": "new_collection",
    "description": "New collection for demonstration",
    "vector_size": 1024
  }'
```
