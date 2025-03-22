import requests
import json
import sys
from test_utils import API_KEY, BASE_URL, HEADERS, print_json

def print_json(json_data):
    """Pretty print JSON data"""
    print(json.dumps(json_data, indent=2))
    print("-" * 80)

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n🔍 Testing ROOT endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/", headers=HEADERS)
        response.raise_for_status()
        print("✅ Success!")
        print_json(response.json())
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_stats_endpoint():
    """Test the stats endpoint"""
    print("\n🔍 Testing STATS endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/stats", headers=HEADERS)
        response.raise_for_status()
        print("✅ Success!")
        print_json(response.json())
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_search_endpoint():
    """Test the search endpoint"""
    print("\n🔍 Testing SEARCH endpoint...")
    
    payload = {
        "query": "What is retrieval?",
        "limit": 3,
        "use_optimized_retrieval": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/search", headers=HEADERS, json=payload)
        response.raise_for_status()
        print("✅ Success!")
        print_json(response.json())
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    """Main test function"""
    print("🧪 API Endpoint Testing 🧪")
    print("=" * 80)
    
    # Test endpoints
    test_root_endpoint()
    test_stats_endpoint()
    test_search_endpoint()
    
    print("\n✨ API testing completed!")

if __name__ == "__main__":
    main()
