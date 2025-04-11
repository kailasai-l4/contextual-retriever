import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8001"  # Change if your API runs on a different port
API_KEY = "581e2e5fc4ed201bca765731798f4834f8424a129b8a5a4722c292cf3a13cfe8"  # From your .env

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

def print_response(title, response):
    print(f"\n=== {title} ===")
    print(f"Status: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)

# Test 1: Health check (first test to make sure API is running)
response = requests.get(f"{BASE_URL}/health")
print_response("1. Health check", response)

# Test 2: List collections via our API
response = requests.get(f"{BASE_URL}/collections", headers=headers)
print_response("2. List collections", response)

# Test 3: Create test collection
collection_name = "test_collection"
create_data = {
    "name": collection_name,
    "description": "Test collection for API testing",
    "vector_size": 1024
}
response = requests.post(f"{BASE_URL}/collections/{collection_name}", 
                        headers=headers, 
                        json=create_data)
print_response("3. Create collection", response)

# Test 4: Get collection details
response = requests.get(f"{BASE_URL}/collections/{collection_name}", headers=headers)
print_response("4. Get collection details", response)

# Test 5: Add data using bulk upload
bulk_data = {
    "data": [
        {
            "text": "Qdrant is a vector database for similarity search",
            "metadata": {"source": "test"}
        },
        {
            "text": "Vector embeddings are useful for semantic search",
            "metadata": {"source": "test"}
        }
    ],
    "collection": collection_name
}
response = requests.post(f"{BASE_URL}/bulk/process", headers=headers, json=bulk_data)
print_response("5. Bulk process data", response)

# Wait for processing
print("\nWaiting 10 seconds for processing...")
time.sleep(10)

# Test 6: Get stats for collection
response = requests.get(f"{BASE_URL}/stats?collection={collection_name}", headers=headers)
print_response("6. Get collection stats", response)

# Test 7: Search the collection
search_data = {
    "query": "vector database",
    "collection": collection_name
}
response = requests.post(f"{BASE_URL}/search", headers=headers, json=search_data)
print_response("7. Search in collection", response)

# Test 8: Search across all collections
search_all = {
    "query": "vector database",
    "search_all_collections": True
}
response = requests.post(f"{BASE_URL}/search", headers=headers, json=search_all)
print_response("8. Search all collections", response)

# Test 9: Clean up - delete collection
response = requests.delete(f"{BASE_URL}/collections/{collection_name}", headers=headers)
print_response("9. Delete collection", response)