import requests
import json
from datetime import datetime
import os
import sys
import tabulate
from test_utils import API_KEY, BASE_URL, HEADERS, load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
HEADERS = {"X-API-Key": API_KEY}

def print_section_header(title):
    """Print a formatted section header"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

def format_file_types(file_types):
    """Format file type distribution as a table"""
    if not file_types:
        return "No file type information available"
    
    headers = ["File Type", "Count"]
    rows = [[file_type, count] for file_type, count in file_types.items()]
    return tabulate.tabulate(rows, headers=headers, tablefmt="grid")

def format_recent_documents(documents):
    """Format recently processed documents as a table"""
    if not documents:
        return "No recently processed documents available"
    
    headers = ["Document Title", "Source ID"]
    rows = [[doc.get("title", "Untitled"), doc.get("source_id", "Unknown")] for doc in documents]
    return tabulate.tabulate(rows, headers=headers, tablefmt="grid")

def format_summary(summary):
    """Format database summary as a table"""
    headers = ["Metric", "Value"]
    rows = [
        ["Collection Name", summary.get("collection_name", "Unknown")],
        ["Status", summary.get("status", "Unknown")],
        ["Total Chunks", summary.get("total_chunks", 0)],
        ["Unique Documents", summary.get("unique_documents", 0)],
        ["Avg. Chunks per Document", summary.get("avg_chunks_per_document", 0)]
    ]
    return tabulate.tabulate(rows, headers=headers, tablefmt="grid")

def format_db_info(db_info):
    """Format database technical information as a table"""
    headers = ["Metric", "Value"]
    rows = [
        ["Vectors Count", db_info.get("vectors_count", "Unknown")],
        ["Segments Count", db_info.get("segments_count", "Unknown")],
        ["Vector Dimension", db_info.get("vector_dimension", "Unknown")]
    ]
    return tabulate.tabulate(rows, headers=headers, tablefmt="grid")

def test_stats_endpoint():
    """Test the enhanced stats endpoint"""
    print_section_header("DATABASE STATISTICS")
    
    try:
        response = requests.get(f"{BASE_URL}/stats", headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            stats = data.get("data", {})
            duration = data.get("duration_ms", 0) / 1000.0
            
            # Print database summary
            print("\n📊 DATABASE SUMMARY")
            print(format_summary(stats.get("summary", {})))
            
            # Print database technical information
            print("\n🔧 TECHNICAL INFORMATION")
            print(format_db_info(stats.get("database_info", {})))
            
            # Print file type distribution
            print("\n📁 FILE TYPE DISTRIBUTION")
            print(format_file_types(stats.get("content", {}).get("file_type_distribution", {})))
            
            # Print recently processed documents
            print("\n🕒 RECENTLY PROCESSED DOCUMENTS")
            print(format_recent_documents(stats.get("content", {}).get("recently_processed", [])))
            
            print(f"\nStatistics retrieved in {duration:.2f} seconds")
            return True
        else:
            print(f"❌ Error: {data.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error fetching statistics: {str(e)}")
        return False

if __name__ == "__main__":
    # Check if tabulate is installed
    try:
        import tabulate
    except ImportError:
        print("Installing tabulate package...")
        os.system(f"{sys.executable} -m pip install tabulate")
        import tabulate
    
    test_stats_endpoint()
