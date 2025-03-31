import requests
import json
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment or fall back to hardcoded value
API_KEY = os.environ.get("API_KEY") or "581e2e5fc4ed201bca765731798f4834f8424a129b8a5a4722c292cf3a13cfe8"
# Get base URL from environment or use default
BASE_URL = os.environ.get("API_BASE_URL") or "http://localhost:8000"
HEADERS = {"X-API-Key": API_KEY}

def print_json(json_data):
    """Pretty print JSON data"""
    print(json.dumps(json_data, indent=2))
    print("-" * 80)

def test_search_with_query(query, filters=None, limit=5, use_optimized=True):
    """
    Test the search endpoint with a specific query and optional filters
    
    Args:
        query: Search query string
        filters: Optional dict with filter criteria
        limit: Maximum number of results
        use_optimized: Whether to use optimized retrieval
        
    Returns:
        Response JSON or None on error
    """
    print(f"\n=== Testing /search endpoint with query: '{query}' ===\n")
    
    payload = {
        "query": query,
        "limit": limit,
        "use_optimized_retrieval": use_optimized
    }
    
    if filters:
        payload["filters"] = filters
        print(f"Using filters: {filters}")
    
    try:
        response = requests.post(f"{BASE_URL}/search", headers=HEADERS, json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print_json(result)
            return result
        else:
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Request failed: {str(e)}")
        return None

def main():
    """Main function to run the test search"""
    print(f"Using API at: {BASE_URL}")
    
    # Get query from command line if provided
    if len(sys.argv) > 1:
        query = sys.argv[1]
        result = test_search_with_query(query)
        if result and result.get('data', {}).get('count', 0) > 0:
            print(f"\n✅ Success! Found {result.get('data', {}).get('count', 0)} results with query: '{query}'")
        else:
            print(f"\n❌ No results found with query: '{query}'")
        return
    
    # Try different queries that are more likely to match document content
    queries = [
        "Swamiji",              # From the first document title
        "coffee table",         # From the second document title
        "devotees",             # From the first document title
        "usk",                  # From the second document title
        "reminiscences",        # From the first document title
        "memories devotees"     # Combined terms from first document
    ]

    # Try each query
    for query in queries:
        result = test_search_with_query(query)
        # If we get results, we don't need to try more queries
        if result and result.get('data', {}).get('count', 0) > 0:
            print(f"\n✅ Success! Found {result.get('data', {}).get('count', 0)} results with query: '{query}'")
            
            # Test filtering if we got results
            source_type = result.get('data', {}).get('results', [{}])[0].get('source_type')
            if source_type:
                print(f"\n=== Testing search with filter on source_type: '{source_type}' ===\n")
                filter_result = test_search_with_query(
                    query, 
                    filters={"source_type": source_type}
                )
                if filter_result and filter_result.get('data', {}).get('count', 0) > 0:
                    print(f"\n✅ Success! Filter search returned {filter_result.get('data', {}).get('count', 0)} results")
            
            break
    else:
        print("\n❌ No results found with any of the test queries.")
        print("Consider processing more documents or checking the Qdrant collection.")

if __name__ == "__main__":
    main()
