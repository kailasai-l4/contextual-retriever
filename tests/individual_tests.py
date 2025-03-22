import requests
import json
import time
import sys
from test_utils import API_KEY, BASE_URL, HEADERS, print_json

def print_json(json_data):
    """Pretty print JSON data"""
    print(json.dumps(json_data, indent=2))
    print("-" * 80)

def test_stats():
    """Test the stats endpoint"""
    print("\n=== Testing /stats endpoint ===\n")
    response = requests.get(f"{BASE_URL}/stats", headers=HEADERS)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print_json(response.json())
    else:
        print(f"Error: {response.text}")

def test_search():
    """Test the search endpoint"""
    print("\n=== Testing /search endpoint ===\n")
    payload = {
        "query": "What are memories?",
        "limit": 5,
        "use_optimized_retrieval": True
    }
    response = requests.post(f"{BASE_URL}/search", headers=HEADERS, json=payload)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print_json(response.json())
    else:
        print(f"Error: {response.text}")
    
    return response.json() if response.status_code == 200 else None

if __name__ == "__main__":
    # Choose which test to run (pass as command line argument: stats or search)
    if len(sys.argv) > 1:
        if sys.argv[1] == "stats":
            test_stats()
        elif sys.argv[1] == "search":
            test_search()
        else:
            print(f"Unknown test: {sys.argv[1]}")
    else:
        print("Please specify which test to run: stats or search")
