"""
Utility functions for test scripts
"""
import os
import sys
import json
from dotenv import load_dotenv

# Add parent directory to path so we can import modules from the main application
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.environ.get("API_KEY", "581e2e5fc4ed201bca765731798f4834f8424a129b8a5a4722c292cf3a13cfe8")
BASE_URL = "http://localhost:8000"
HEADERS = {"X-API-Key": API_KEY}

def print_json(json_data):
    """Pretty print JSON data"""
    print(json.dumps(json_data, indent=2))
    print("-" * 80)

def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
