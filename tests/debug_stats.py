import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.environ.get("API_KEY", "581e2e5fc4ed201bca765731798f4834f8424a129b8a5a4722c292cf3a13cfe8")
BASE_URL = "http://localhost:8000"
HEADERS = {"X-API-Key": API_KEY}

def print_json(json_data):
    """Pretty print JSON data"""
    print(json.dumps(json_data, indent=2))

print("Fetching raw statistics data...")
response = requests.get(f"{BASE_URL}/stats", headers=HEADERS, timeout=30)
print(f"Status code: {response.status_code}")
print("\nRaw API response:")
print_json(response.json())
