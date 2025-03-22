# API Documentation

This document provides detailed information about the RESTful API endpoints available in the RAG Content Retriever system.

## Authentication

All API endpoints require authentication using an API key. The key should be provided in the `X-API-Key` header.

Example:
```
X-API-Key: your_api_key_here
```

The API key is defined in the `.env` file or as an environment variable `API_KEY`.

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
      {"path": "/dashboard", "method": "GET", "description": "Get a comprehensive dashboard view of the database statistics"}
    ]
  },
  "message": null,
  "duration_ms": 1.2
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
  "optimize": true,
  "filter": {
    "source_type": ".md",
    "metadata.custom_field": "value"
  },
  "rerank": true,
  "diverse_sources": true
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | string | Yes | The search query text |
| limit | integer | No | Maximum number of results to return (default: 10) |
| optimize | boolean | No | Whether to use optimized search strategies (default: true) |
| filter | object | No | Filters to apply to search results |
| rerank | boolean | No | Whether to rerank results for improved relevance (default: true) |
| diverse_sources | boolean | No | Whether to ensure results come from different sources (default: true) |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "content": "This is the content of the first result",
        "source_id": "document1.md",
        "score": 0.92,
        "chunk_index": 3,
        "content_title": "Chapter 1: Introduction",
        "metadata": {
          "processed_at": 1679823456,
          "processed_time": "2023-03-26 12:30:56",
          "source_type": ".md"
        }
      },
      // ... more results
    ],
    "query": "Your search query",
    "total_results": 25,
    "returned_results": 10,
    "duration_ms": 245.8
  },
  "message": "Search completed successfully",
  "duration_ms": 250.1
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
  "file_path": "/path/to/document.md",
  "content": "Raw content to process if no file_path is provided",
  "source_id": "custom_source_id",
  "source_type": ".md",
  "metadata": {
    "author": "John Doe",
    "created_at": "2023-03-25",
    "custom_field": "custom_value"
  },
  "chunk_options": {
    "method": "token_based",
    "token_limit": 1000,
    "overlap": 100
  }
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file_path | string | No* | Path to the file to process (*required if content not provided) |
| content | string | No* | Raw content to process (*required if file_path not provided) |
| source_id | string | No | Custom ID for the source (auto-generated if not provided) |
| source_type | string | No | Type of the source document |
| metadata | object | No | Additional metadata for the document |
| chunk_options | object | No | Options for chunking the document |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "source_id": "document1.md",
    "chunks_processed": 15,
    "vectors_added": 15,
    "token_count": 12500
  },
  "message": "Document processed successfully",
  "duration_ms": 3520.4
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
    },
    "raw_statistics": {
      // Raw database statistics
    }
  },
  "message": "Database statistics retrieved successfully",
  "duration_ms": 156.8
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

In case of an error, the API returns an appropriate HTTP status code along with a JSON response:

```json
{
  "success": false,
  "data": null,
  "message": "Error message describing what went wrong",
  "duration_ms": 12.5
}
```

Common status codes:
- 400: Bad Request (invalid parameters)
- 401: Unauthorized (invalid API key)
- 404: Not Found (resource not found)
- 500: Internal Server Error (server-side error)

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
