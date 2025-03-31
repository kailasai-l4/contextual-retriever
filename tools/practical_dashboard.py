#!/usr/bin/env python
"""
Practical Dashboard for RAG Content Retriever
---------------------------------------------
This script provides actionable insights about your vector database,
focusing on practical metrics and system health monitoring.
"""

import requests
import json
import os
import time
import sys
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.text import Text
from rich import box
from rich.tree import Tree
from rich.progress import BarColumn, Progress
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.environ.get("API_KEY", "581e2e5fc4ed201bca765731798f4834f8424a129b8a5a4722c292cf3a13cfe8")
BASE_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")
HEADERS = {"X-API-Key": API_KEY}

# Initialize Rich console
console = Console()

def get_dashboard_data():
    """Fetch dashboard data from the API"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Fetching database metrics..."),
        transient=True,
    ) as progress:
        progress.add_task("fetch", total=None)
        start_time = time.time()
        try:
            response = requests.get(f"{BASE_URL}/dashboard", headers=HEADERS)
            response.raise_for_status()
            data = response.json()
            elapsed = time.time() - start_time
            console.print(f"✅ Data retrieved successfully in {elapsed:.2f} seconds\n")
            return data.get("data", {})
        except Exception as e:
            console.print(f"[bold red]Error fetching dashboard data: {str(e)}[/bold red]")
            return None

def get_stats_data():
    """Fetch detailed statistics from the API"""
    try:
        response = requests.get(f"{BASE_URL}/stats", headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        return data.get("data", {})
    except Exception as e:
        console.print(f"[bold red]Error fetching statistics: {str(e)}[/bold red]")
        return None

def check_system_health():
    """Check system health by making test API calls"""
    health_checks = []
    
    # Check 1: API responsiveness
    start_time = time.time()
    try:
        response = requests.get(f"{BASE_URL}/", headers=HEADERS)
        response.raise_for_status()
        api_latency = time.time() - start_time
        health_checks.append(("API Responsiveness", "Good" if api_latency < 1 else "Slow", f"{api_latency:.2f}s"))
    except Exception as e:
        health_checks.append(("API Responsiveness", "Failed", str(e)))
    
    # Check 2: Search functionality
    try:
        response = requests.post(
            f"{BASE_URL}/search", 
            headers=HEADERS,
            json={"query": "test query", "limit": 1}
        )
        response.raise_for_status()
        search_time = response.json().get("duration_ms", 0) / 1000
        health_checks.append(("Search Functionality", "Good" if search_time < 2 else "Slow", f"{search_time:.2f}s"))
    except Exception as e:
        health_checks.append(("Search Functionality", "Failed", str(e)))
    
    # Check 3: Database connectivity
    try:
        response = requests.get(f"{BASE_URL}/stats", headers=HEADERS)
        response.raise_for_status()
        db_status = response.json().get("data", {}).get("summary", {}).get("status", "unknown")
        health_checks.append(("Database Status", "Good" if db_status == "green" else "Warning", db_status))
    except Exception as e:
        health_checks.append(("Database Status", "Failed", str(e)))
    
    return health_checks

def display_system_health(health_checks):
    """Display system health information"""
    table = Table(title="System Health Check", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")
    
    for component, status, details in health_checks:
        status_style = "green" if status == "Good" else "yellow" if status == "Slow" or status == "Warning" else "red"
        table.add_row(component, f"[{status_style}]{status}[/{status_style}]", details)
    
    console.print(Panel(table, title="[bold]System Health[/bold]", border_style="blue"))

def display_database_metrics(dashboard_data, stats_data):
    """Display practical database metrics"""
    if not dashboard_data or not stats_data:
        console.print("[bold red]No data available to display[/bold red]")
        return
    
    # Extract metrics
    metrics = dashboard_data.get("metrics", [])
    metrics_dict = {m["name"]: m for m in metrics}
    
    # Create metrics table
    table = Table(title="Database Metrics", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Status", style="bold")
    
    # Total Documents
    total_docs = metrics_dict.get("Total Documents", {}).get("value", 0)
    doc_status = "Good" if total_docs > 0 else "Empty"
    doc_style = "green" if total_docs > 0 else "yellow"
    table.add_row("Total Documents", str(total_docs), f"[{doc_style}]{doc_status}[/{doc_style}]")
    
    # Total Chunks
    total_chunks = metrics_dict.get("Total Chunks", {}).get("value", 0)
    chunk_status = "Good" if total_chunks > 0 else "Empty"
    chunk_style = "green" if total_chunks > 0 else "yellow"
    table.add_row("Total Chunks", str(total_chunks), f"[{chunk_style}]{chunk_status}[/{chunk_style}]")
    
    # Avg Chunks per Document
    avg_chunks = metrics_dict.get("Avg. Chunks per Document", {}).get("value", 0)
    avg_status = "Good" if 10 <= avg_chunks <= 100 else "Review Needed"
    avg_style = "green" if 10 <= avg_chunks <= 100 else "yellow"
    table.add_row("Avg Chunks/Doc", str(avg_chunks), f"[{avg_style}]{avg_status}[/{avg_style}]")
    
    # Database Status
    db_status = metrics_dict.get("Database Status", {}).get("value", "unknown")
    status_text = "Healthy" if db_status == "green" else "Warning" if db_status == "yellow" else "Critical"
    status_style = "green" if db_status == "green" else "yellow" if db_status == "yellow" else "red"
    table.add_row("Database Status", db_status, f"[{status_style}]{status_text}[/{status_style}]")
    
    # Collection Info
    collection_info = dashboard_data.get("collection_info", {})
    collection_name = collection_info.get("name", "unknown")
    segments_count = collection_info.get("segments", 0)
    segment_status = "Optimized" if segments_count < 20 else "Needs Optimization"
    segment_style = "green" if segments_count < 20 else "yellow"
    table.add_row("Collection", collection_name, "Active")
    table.add_row("Segments", str(segments_count), f"[{segment_style}]{segment_status}[/{segment_style}]")
    
    console.print(Panel(table, title="[bold]Database Status[/bold]", border_style="blue"))

def display_content_insights(dashboard_data):
    """Display insights about the content in the database"""
    if not dashboard_data:
        return
    
    # File type distribution
    file_types_data = dashboard_data.get("file_types", {}).get("data", [])
    
    if file_types_data:
        table = Table(title="Content Distribution", box=box.ROUNDED)
        table.add_column("File Type", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")
        
        total_files = sum(item["value"] for item in file_types_data)
        
        for item in file_types_data:
            file_type = item["name"]
            count = item["value"]
            percentage = (count / total_files * 100) if total_files > 0 else 0
            table.add_row(file_type, str(count), f"{percentage:.1f}%")
        
        console.print(Panel(table, title="[bold]Content Distribution[/bold]", border_style="blue"))
    
    # Recent activity
    recent_activity = dashboard_data.get("recent_activity", [])
    
    if recent_activity:
        table = Table(title="Recent Activity", box=box.ROUNDED)
        table.add_column("Document", style="cyan")
        table.add_column("ID", style="dim")
        table.add_column("Timestamp", style="green")
        
        for item in recent_activity[:5]:  # Show only the 5 most recent
            table.add_row(item["title"], item["id"], item["timestamp"])
        
        console.print(Panel(table, title="[bold]Recent Activity[/bold]", border_style="blue"))
    else:
        console.print("[yellow]No recent activity data available[/yellow]")

def display_actionable_insights(dashboard_data, stats_data):
    """Display actionable insights and recommendations"""
    if not dashboard_data or not stats_data:
        return
    
    insights = []
    
    # Extract metrics
    metrics = dashboard_data.get("metrics", [])
    metrics_dict = {m["name"]: m for m in metrics}
    
    # Insight 1: Database size
    total_chunks = metrics_dict.get("Total Chunks", {}).get("value", 0)
    if total_chunks == 0:
        insights.append(("Database is empty", "Upload documents to start using the RAG system", "critical"))
    elif total_chunks < 100:
        insights.append(("Low document count", "Consider adding more documents for better retrieval results", "warning"))
    
    # Insight 2: Chunk size
    avg_chunks = metrics_dict.get("Avg. Chunks per Document", {}).get("value", 0)
    if avg_chunks > 100:
        insights.append(("High chunk count per document", "Consider adjusting chunk size for better performance", "warning"))
    elif avg_chunks < 5 and total_chunks > 0:
        insights.append(("Low chunk count per document", "Documents may be too small or chunk size too large", "warning"))
    
    # Insight 3: Segments optimization
    segments_count = dashboard_data.get("collection_info", {}).get("segments", 0)
    if segments_count > 20:
        insights.append(("Database needs optimization", "High segment count may impact performance", "warning"))
    
    # Insight 4: File type diversity
    file_types_data = dashboard_data.get("file_types", {}).get("data", [])
    if len(file_types_data) == 1:
        insights.append(("Limited content types", "Consider adding diverse document types for better coverage", "info"))
    
    # Insight 5: Database health
    db_status = metrics_dict.get("Database Status", {}).get("value", "unknown")
    if db_status != "green":
        insights.append(("Database health issue", f"Database status is {db_status}, check server logs", "critical"))
    
    # Display insights
    if insights:
        table = Table(title="Actionable Insights", box=box.ROUNDED)
        table.add_column("Insight", style="cyan")
        table.add_column("Recommendation", style="green")
        table.add_column("Priority", style="bold")
        
        for insight, recommendation, priority in insights:
            priority_style = "red" if priority == "critical" else "yellow" if priority == "warning" else "blue"
            table.add_row(insight, recommendation, f"[{priority_style}]{priority.upper()}[/{priority_style}]")
        
        console.print(Panel(table, title="[bold]Actionable Insights[/bold]", border_style="red"))
    else:
        console.print(Panel("[green]No issues detected. Your RAG system appears to be in good health.[/green]", 
                           title="[bold]System Status[/bold]", border_style="green"))

def main():
    """Main function to run the dashboard"""
    console.print("\n[bold blue]RAG Content Retriever - Practical Dashboard[/bold blue]")
    console.print("[dim]Fetching real-time metrics and actionable insights...[/dim]\n")
    
    # Get data
    dashboard_data = get_dashboard_data()
    if not dashboard_data:
        return
    
    stats_data = get_stats_data()
    health_checks = check_system_health()
    
    # Display sections
    display_system_health(health_checks)
    console.print()
    
    display_database_metrics(dashboard_data, stats_data)
    console.print()
    
    display_content_insights(dashboard_data)
    console.print()
    
    display_actionable_insights(dashboard_data, stats_data)
    console.print()
    
    # Final summary
    console.print("\n[bold green]Dashboard completed![/bold green]")
    console.print("[dim]Run this tool regularly to monitor your RAG system health and performance.[/dim]")

if __name__ == "__main__":
    main()
