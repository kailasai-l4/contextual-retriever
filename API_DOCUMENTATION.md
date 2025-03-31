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
  }
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | string | Yes | The search query text |
| limit | integer | No | Maximum number of results to return (default: 20) |
| use_optimized_retrieval | boolean | No | Whether to use optimized search strategies (default: true) |
| filters | object | No | Filters to apply to search results |

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
        "metadata": {
          "source_type": ".md"
        }
      },
      // ... more results
    ]
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
  "file_types": [".md", ".txt"]
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| directory_path | string | Yes | Path to the directory containing documents to process |
| recursive | boolean | No | Whether to process directories recursively (default: true) |
| file_types | array | No | File extensions to process (e.g., [".md", ".txt"]) |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "status": "processing",
    "task_id": "task_12345",
    "directory": "/path/to/documents",
    "file_count": 10
  },
  "message": "Processing started in the background",
  "duration_ms": 20.4
}
```

### Statistics Endpoint

```
GET /stats
```

Returns statistics about the vector database.

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
    "optimizations_applied": {
      "query_expansion": true,
      "reranking": true,
      "source_diversity": true
    }
  },
  "message": "Found 5 results using optimized retrieval",
  "duration_ms": 355.2
}
```

## Response Structure

All API endpoints return responses with the following structure:

```json
{
  "success": true|false,
  "data": object|null,
  "message": string|null,
  "duration_ms": number
}
```

| Field | Type | Description |
|-------|------|-------------|
| success | boolean | Whether the request was successful |
| data | object | Response data (null if error) |
| message | string | Response or error message (null if not provided) |
| duration_ms | number | Request processing duration in milliseconds |

## Error Handling

All API endpoints now feature improved error handling with consistent error response formats.

### Error Response Format

```json
{
  "success": false,
  "error": "Error type or message",
  "message": "Detailed error message",
  "details": "Additional error details or stack trace (only in debug mode)"
}
```

### Common Error Codes

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - The request is malformed or contains invalid parameters |
| 401 | Unauthorized - API key is missing |
| 403 | Forbidden - API key is invalid |
| 404 | Not Found - The requested resource does not exist |
| 500 | Internal Server Error - An unexpected error occurred |
| 503 | Service Unavailable - The service is temporarily unavailable (e.g., during maintenance or when a dependent service is down) |

### Retry Strategy

For transient errors (status codes 429, 503), clients should implement an exponential backoff retry strategy.

## Docker Deployment

When deploying with Docker, ensure that:

1. The `.env` file is mounted or environment variables are properly set
2. The API port (default 8000) is mapped to the host
3. Data volumes are mounted for persistence

Example Docker run command:
```bash
docker run -p 8000:8000 \
  -v /path/to/data:/app/data \
  -v /path/to/.env:/app/.env \
  --name rag-api \
  rag-content-retriever
```

## Testing the API

You can use the included test scripts to verify API functionality:

```bash
# Test all API endpoints
python test_api.py

# Test only the stats endpoint with enhanced visualization
python test_enhanced_stats.py

# Test the dashboard endpoint with visualization
python simple_dashboard_test.py
```
