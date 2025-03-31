import requests
import json
import os
import time
import sys
from tests.test_utils import API_KEY, BASE_URL, HEADERS, print_section, load_dotenv

# Load environment variables
load_dotenv()

def test_dashboard_endpoint():
    """Test the dashboard endpoint with a simpler output"""
    print("Fetching dashboard data...")
    
    try:
        response = requests.get(f"{BASE_URL}/dashboard", headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            dashboard = data.get("data", {})
            duration = data.get("duration_ms", 0) / 1000.0
            
            print(f" Data retrieved successfully in {duration:.2f} seconds\n")
            
            # Display collection info
            collection_info = dashboard.get("collection_info", {})
            print(f"Collection: {collection_info.get('name', 'Unknown')}")
            print(f"Vector Dimension: {collection_info.get('vector_dimension', 0)}")
            print(f"Segments: {collection_info.get('segments', 0)}")
            
            # Display metrics
            print_section("DATABASE METRICS")
            metrics = dashboard.get("metrics", [])
            for metric in metrics:
                name = metric.get("name", "Unknown")
                value = metric.get("value", "N/A")
                desc = metric.get("description", "")
                print(f"{name}: {value} - {desc}")
            
            # Display health
            print_section("SYSTEM HEALTH")
            health = dashboard.get("health", {})
            status = health.get("status", "unknown")
            print(f"Status: {' HEALTHY' if status == 'healthy' else ' WARNING'}")
            print(f"Last Check: {health.get('last_checked', 'Unknown')}")
            print("\nDetails:")
            for key, value in health.get("details", {}).items():
                print(f"• {key.replace('_', ' ').title()}: {value}")
            
            # Display file types
            print_section("FILE TYPES")
            file_types = dashboard.get("file_types", {}).get("data", [])
            if file_types:
                total = sum(item["value"] for item in file_types)
                for item in file_types:
                    file_type = item.get("name", "Unknown")
                    count = item.get("value", 0)
                    percentage = (count / total * 100) if total > 0 else 0
                    print(f"{file_type}: {count} ({percentage:.1f}%)")
            else:
                print("No file type data available")
            
            # Display recent activity
            print_section("RECENTLY PROCESSED DOCUMENTS")
            activities = dashboard.get("recent_activity", [])
            if activities:
                for i, activity in enumerate(activities, 1):
                    title = activity.get("title", "Unknown")
                    doc_id = activity.get("id", "")
                    timestamp = activity.get("timestamp", "Unknown")
                    
                    # Format timestamp if it's a number
                    if isinstance(timestamp, (int, float)):
                        from datetime import datetime
                        timestamp = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    
                    print(f"{i}. {title}")
                    print(f"   ID: {doc_id}")
                    print(f"   Processed: {timestamp}")
                    print()
            else:
                print("No recent activity data available")
            
            return True
        else:
            print(f" Error: {data.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f" Error fetching dashboard data: {str(e)}")
        return False

if __name__ == "__main__":
    test_dashboard_endpoint()
