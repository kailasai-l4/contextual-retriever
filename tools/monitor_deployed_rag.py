#!/usr/bin/env python
"""
RAG Deployment Monitor
---------------------
A simple, focused tool to monitor and manage your deployed RAG system.
"""

import requests
import json
import os
import time
from datetime import datetime
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default configuration
DEFAULT_API_KEY = "581e2e5fc4ed201bca765731798f4834f8424a129b8a5a4722c292cf3a13cfe8"
DEFAULT_BASE_URL = "https://vidyakosha.kailasa.ai"

def get_api_status(base_url, api_key):
    """Check if the API is responsive"""
    headers = {"X-API-Key": api_key}
    
    try:
        start_time = time.time()
        response = requests.get(f"{base_url}/", headers=headers)
        response.raise_for_status()
        latency = time.time() - start_time
        
        print(f"✅ API is online (latency: {latency:.2f}s)")
        print(f"API version: {response.json().get('version', 'unknown')}")
        return True
    except Exception as e:
        print(f"❌ API connection failed: {str(e)}")
        return False

def get_database_stats(base_url, api_key):
    """Get database statistics"""
    headers = {"X-API-Key": api_key}
    
    try:
        response = requests.get(f"{base_url}/stats", headers=headers)
        response.raise_for_status()
        data = response.json().get("data", {})
        
        # Extract summary data
        summary = data.get("summary", {})
        
        print("\n=== DATABASE STATISTICS ===")
        print(f"Collection: {summary.get('collection_name', 'unknown')}")
        print(f"Status: {summary.get('status', 'unknown')}")
        print(f"Total documents: {summary.get('unique_documents', 0)}")
        print(f"Total chunks: {summary.get('total_chunks', 0)}")
        print(f"Avg. chunks per document: {summary.get('avg_chunks_per_document', 0)}")
        
        # File type distribution
        file_types = data.get("content", {}).get("file_type_distribution", {})
        if file_types:
            print("\n=== FILE TYPE DISTRIBUTION ===")
            for file_type, count in file_types.items():
                print(f"{file_type}: {count}")
        
        return data
    except Exception as e:
        print(f"❌ Failed to get database statistics: {str(e)}")
        return None

def test_search(base_url, api_key, query="test query"):
    """Test search functionality"""
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    try:
        print(f"\n=== TESTING SEARCH WITH QUERY: '{query}' ===")
        start_time = time.time()
        response = requests.post(
            f"{base_url}/search", 
            headers=headers,
            json={"query": query, "limit": 5}
        )
        response.raise_for_status()
        data = response.json()
        
        results = data.get("data", {}).get("results", [])
        duration = data.get("duration_ms", 0) / 1000
        
        print(f"Search completed in {duration:.2f} seconds")
        print(f"Found {len(results)} results")
        
        if results:
            print("\nTop results:")
            for i, result in enumerate(results[:3], 1):
                print(f"{i}. {result.get('text', '')[:100]}... (score: {result.get('score', 0):.2f})")
        
        return len(results) > 0
    except Exception as e:
        print(f"❌ Search test failed: {str(e)}")
        return False

def upload_file(base_url, api_key, file_path):
    """Upload a file to the RAG system"""
    headers = {"X-API-Key": api_key}
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        print(f"\n=== UPLOADING FILE: {os.path.basename(file_path)} ===")
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            response = requests.post(f"{base_url}/upload", headers=headers, files=files)
        
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ File uploaded successfully")
        print(f"Message: {data.get('message', 'No message')}")
        print(f"Processing time: {data.get('duration_ms', 0) / 1000:.2f} seconds")
        
        return True
    except Exception as e:
        print(f"❌ File upload failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="RAG Deployment Monitor")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="Base URL of the RAG API")
    parser.add_argument("--key", default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--action", choices=["status", "stats", "search", "upload"], default="status", 
                        help="Action to perform")
    parser.add_argument("--query", default="What is RAG?", help="Query for search action")
    parser.add_argument("--file", help="File path for upload action")
    
    args = parser.parse_args()
    
    print(f"🔍 RAG Deployment Monitor")
    print(f"Target: {args.url}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    # Check API status first
    api_online = get_api_status(args.url, args.key)
    
    if not api_online and args.action != "status":
        print("❌ Cannot proceed with requested action because API is offline")
        return
    
    # Perform the requested action
    if args.action == "stats":
        get_database_stats(args.url, args.key)
    elif args.action == "search":
        test_search(args.url, args.key, args.query)
    elif args.action == "upload" and args.file:
        upload_file(args.url, args.key, args.file)
    
    print("\n✨ Monitoring complete")

if __name__ == "__main__":
    main()
