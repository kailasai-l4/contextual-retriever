import requests
import json
import sys
import os
import time
from pathlib import Path
from tests.test_utils import API_KEY, BASE_URL, HEADERS, print_json

UPLOADS_DIR = "uploads"

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
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_stats_endpoint():
    """Test the stats endpoint"""
    print("\n🔍 Testing STATS endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/stats", headers=HEADERS)
        response.raise_for_status()
        print("✅ Success!")
        print_json(response.json())
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_search_endpoint(query="knowledge retrieval", limit=3):
    """Test the search endpoint"""
    print(f"\n🔍 Testing SEARCH endpoint with query: '{query}'...")
    
    payload = {
        "query": query,
        "limit": limit,
        "use_optimized_retrieval": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/search", headers=HEADERS, json=payload)
        response.raise_for_status()
        print("✅ Success!")
        result = response.json()
        print_json(result)
        return result
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def test_optimized_retrieval_endpoint(topic="AI and machine learning", limit=3):
    """Test the optimized retrieval endpoint"""
    print(f"\n🔍 Testing OPTIMIZED RETRIEVAL endpoint with topic: '{topic}'...")
    
    payload = {
        "topic": topic,
        "limit": limit
    }
    
    try:
        response = requests.post(f"{BASE_URL}/optimized-retrieval", headers=HEADERS, json=payload)
        response.raise_for_status()
        print("✅ Success!")
        print_json(response.json())
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_process_directory_endpoint(directory_path="uploads"):
    """Test the process directory endpoint"""
    print(f"\n🔍 Testing PROCESS DIRECTORY endpoint with path: '{directory_path}'...")
    
    # Ensure the directory exists
    if not os.path.isdir(directory_path):
        print(f"❌ Error: Directory '{directory_path}' does not exist.")
        return False
    
    # Get absolute path
    abs_path = os.path.abspath(directory_path)
    print(f"Processing directory: {abs_path}")
    
    payload = {
        "directory_path": abs_path,
        "recursive": True,
        "file_types": [".md", ".json"]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/process", headers=HEADERS, json=payload)
        response.raise_for_status()
        print("✅ Success! Processing started in the background.")
        print_json(response.json())
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def upload_and_process_file(file_path):
    """Upload and process a file"""
    print(f"\n🔍 Testing FILE UPLOAD endpoint with file: '{file_path}'...")
    
    if not os.path.isfile(file_path):
        print(f"❌ Error: File '{file_path}' does not exist.")
        return False
    
    try:
        with open(file_path, 'rb') as file:
            files = {'file': (os.path.basename(file_path), file)}
            response = requests.post(
                f"{BASE_URL}/upload?process_now=true", 
                headers={"X-API-Key": API_KEY}, 
                files=files
            )
        response.raise_for_status()
        print("✅ Success! File uploaded and processing started.")
        print_json(response.json())
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_get_source_content(source_id):
    """Test getting content for a specific source"""
    print(f"\n🔍 Testing GET SOURCE CONTENT endpoint with ID: '{source_id}'...")
    
    try:
        response = requests.get(f"{BASE_URL}/sources/{source_id}", headers=HEADERS)
        response.raise_for_status()
        print("✅ Success!")
        print_json(response.json())
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🧪 Comprehensive API Endpoint Testing 🧪")
    print("=" * 80)
    
    # Test all endpoints
    test_root_endpoint()
    
    # Process the directory using the API
    test_process_directory_endpoint()
    
    # Wait for processing to complete
    print("\n⏳ Waiting for processing to complete (30 seconds)...")
    time.sleep(30)
    
    # Get stats after processing
    test_stats_endpoint()
    
    # Test search
    search_result = test_search_endpoint()
    
    # If search was successful and returned results, test getting a specific source
    if search_result and search_result.get('data', {}).get('count', 0) > 0:
        source_id = search_result['data']['results'][0].get('source_id')
        if source_id:
            test_get_source_content(source_id)
    
    # Test optimized retrieval
    test_optimized_retrieval_endpoint()
    
    # Try to upload and process individual files
    uploads_dir = Path("uploads")
    for file_path in uploads_dir.glob("*.md"):
        if file_path.name != ".gitkeep":
            upload_and_process_file(str(file_path))
    
    print("\n✨ Comprehensive API testing completed!")

if __name__ == "__main__":
    main()
