#!/usr/bin/env python3
"""
Local API Tester
---------------
Test script for validating the RAG Content Retriever API functionality on localhost.
This script doesn't require external API keys and focuses on the health checks and
Qdrant connection.
"""

import os
import sys
import time
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration - localhost only
BASE_URL = "http://localhost:8000"
API_KEY = os.environ.get("API_KEY")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

def print_result(endpoint, response):
    """Print test result"""
    print(f"\n===== {endpoint} =====")
    print(f"Status: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text[:200] + "..." if len(response.text) > 200 else response.text)
    print("-" * 50)

def test_endpoint(method, endpoint, payload=None, expected_status=200):
    """Test an API endpoint"""
    url = f"{BASE_URL}{endpoint}"
    
    print(f"Testing {method} {endpoint}...")
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=HEADERS)
        elif method.upper() == "POST":
            response = requests.post(url, headers=HEADERS, json=payload)
        else:
            print(f"Unsupported method: {method}")
            return False
            
        print_result(endpoint, response)
        
        if response.status_code == expected_status:
            print(f"✅ Success: {endpoint}")
            return response.json() if response.status_code == 200 else None
        else:
            print(f"❌ Failed: Expected status {expected_status}, got {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def main():
    """Run local API tests focusing on health endpoints and Qdrant connection"""
    print(f"Testing local API at {BASE_URL}")
    
    # Test health endpoints (these don't require API key)
    test_endpoint("GET", "/health")
    test_endpoint("GET", "/readiness")
    test_endpoint("GET", "/liveness")
    
    # If we have an API key, test the protected endpoints
    if API_KEY:
        # Test root endpoint
        test_endpoint("GET", "/")
        
        # Test stats endpoint (good test for Qdrant connection)
        test_endpoint("GET", "/stats")
        
        # Simple search test
        simple_query = {
            "query": "test",
            "limit": 3,
            "use_optimized_retrieval": False
        }
        test_endpoint("POST", "/search", simple_query)
    else:
        print("\n⚠️ No API key found in environment. Skipping protected endpoints.")
        print("Use the .env file to set your API_KEY to test protected endpoints.")

if __name__ == "__main__":
    main() 