import requests
import json
import os
import time
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.text import Text
from rich import box
import sys
from dotenv import load_dotenv

# Add parent directory to path so we can import modules from the main application
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.environ.get("API_KEY", "581e2e5fc4ed201bca765731798f4834f8424a129b8a5a4722c292cf3a13cfe8")
BASE_URL = "http://localhost:8000"
HEADERS = {"X-API-Key": API_KEY}

# Initialize Rich console
console = Console()

def create_metrics_panel(metrics):
    """Create a rich panel with metrics"""
    table = Table(show_header=False, expand=True)
    table.add_column("Name", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Description", style="dim")
    
    for metric in metrics:
        name = metric.get("name", "Unknown")
        value = metric.get("value", "N/A")
        desc = metric.get("description", "")
        
        # Format value based on type
        if metric.get("format") == "status":
            if value == "green":
                value_text = Text("✅ HEALTHY", style="bold green")
            elif value == "yellow":
                value_text = Text("⚠️ WARNING", style="bold yellow")
            else:
                value_text = Text("❌ ISSUE", style="bold red")
        else:
            value_text = Text(str(value))
            
        table.add_row(name, value_text, desc)
    
    return Panel(table, title="[bold]Database Metrics[/bold]", border_style="blue")

def create_file_types_panel(file_types):
    """Create a panel showing file type distribution"""
    if not file_types.get("data"):
        return Panel("No file type data available", title="[bold]File Types[/bold]", border_style="yellow")
    
    table = Table(show_header=True)
    table.add_column("File Type", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_column("Percentage", style="magenta", justify="right")
    
    total = sum(item["value"] for item in file_types.get("data", []))
    
    for item in file_types.get("data", []):
        file_type = item.get("name", "Unknown")
        count = item.get("value", 0)
        percentage = (count / total * 100) if total > 0 else 0
        
        # Create a simple bar chart
        bar_length = int(percentage / 5)  # 5% per character
        bar = "█" * bar_length
        
        table.add_row(file_type, str(count), f"{percentage:.1f}% {bar}")
    
    return Panel(table, title=f"[bold]{file_types.get('title', 'File Types')}[/bold]", border_style="green")

def create_health_panel(health):
    """Create a panel showing database health"""
    if not health:
        return Panel("No health data available", title="[bold]System Health[/bold]", border_style="red")
    
    status = health.get("status", "unknown")
    if status == "healthy":
        status_text = Text("✅ HEALTHY", style="bold green")
    else:
        status_text = Text("⚠️ WARNING", style="bold yellow")
    
    details = health.get("details", {})
    
    content = f"Status: {status_text}\n"
    content += f"Last Check: {health.get('last_checked', 'Unknown')}\n\n"
    content += "Details:\n"
    
    for key, value in details.items():
        content += f"• {key.replace('_', ' ').title()}: {value}\n"
    
    return Panel(content, title="[bold]System Health[/bold]", border_style="red" if status != "healthy" else "green")

def create_recent_activity_panel(activities):
    """Create a panel showing recent activity"""
    if not activities:
        return Panel("No recent activity data available", title="[bold]Recent Activity[/bold]", border_style="cyan")
    
    table = Table(show_header=True)
    table.add_column("Document", style="cyan")
    table.add_column("ID", style="dim")
    table.add_column("Timestamp", style="green")
    
    for activity in activities:
        title = activity.get("title", "Unknown")
        doc_id = activity.get("id", "")
        timestamp = activity.get("timestamp", "Unknown")
        
        # Format timestamp if it's a number
        if isinstance(timestamp, (int, float)):
            from datetime import datetime
            timestamp = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            
        table.add_row(title, doc_id, timestamp)
    
    return Panel(table, title="[bold]Recently Processed[/bold]", border_style="cyan")

def display_dashboard(dashboard_data):
    """Display the dashboard in a rich layout"""
    # Create layout
    layout = Layout()
    
    # Split into rows
    layout.split(
        Layout(name="header"),
        Layout(name="main", ratio=3),
        Layout(name="footer"),
    )
    
    # Split main area into columns
    layout["main"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    
    # Split left and right columns
    layout["left"].split(
        Layout(name="metrics"),
        Layout(name="health"),
    )
    
    layout["right"].split(
        Layout(name="file_types"),
        Layout(name="recent_activity"),
    )
    
    # Add content to each section
    collection_info = dashboard_data.get("collection_info", {})
    collection_name = collection_info.get("name", "Unknown")
    collection_dimension = collection_info.get("vector_dimension", 0)
    
    header_text = f"🔍 [bold blue]Qdrant Vector Database Dashboard[/bold blue] | [green]Collection: {collection_name}[/green] | [yellow]Vector Dim: {collection_dimension}[/yellow]"
    
    layout["header"].update(Panel(header_text, border_style="blue"))
    layout["metrics"].update(create_metrics_panel(dashboard_data.get("metrics", [])))
    layout["health"].update(create_health_panel(dashboard_data.get("health", {})))
    layout["file_types"].update(create_file_types_panel(dashboard_data.get("file_types", {})))
    layout["recent_activity"].update(create_recent_activity_panel(dashboard_data.get("recent_activity", [])))
    
    layout["footer"].update(Panel(f"Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}", border_style="dim"))
    
    # Render the layout
    console.print(layout)

def test_dashboard_endpoint():
    """Test the dashboard endpoint"""
    console.print("[bold]Fetching dashboard data...[/bold]")
    
    try:
        response = requests.get(f"{BASE_URL}/dashboard", headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            dashboard = data.get("data", {})
            duration = data.get("duration_ms", 0) / 1000.0
            
            console.print(f"[green]✅ Data retrieved successfully in {duration:.2f} seconds[/green]\n")
            
            # Display the dashboard
            display_dashboard(dashboard)
            return True
        else:
            console.print(f"[red]❌ Error: {data.get('message', 'Unknown error')}[/red]")
            return False
            
    except Exception as e:
        console.print(f"[red]❌ Error fetching dashboard data: {str(e)}[/red]")
        return False

if __name__ == "__main__":
    # Check if rich is installed
    try:
        import rich
    except ImportError:
        print("Installing rich package...")
        os.system(f"{sys.executable} -m pip install rich")
        import rich
    
    test_dashboard_endpoint()
