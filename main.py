#!/usr/bin/env python3
"""
RAG Content Retriever
-------------------
A comprehensive tool for processing documents into vector embeddings
and retrieving relevant content using advanced semantic search.
"""

import os
import sys
import argparse
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Import our configuration module
from config import get_config, Config

# Import our modules
from content_processor import ContentProcessor
from advanced_retriever import AdvancedRetriever

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RAG Content Retriever')

    # Config argument
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--log-details', action='store_true', help='Enable detailed logging')

    # Main commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Config command
    config_parser = subparsers.add_parser('config', help='Create default configuration file')
    config_parser.add_argument('--output', default='config.yaml', help='Output path for config file')

    # Process command
    process_parser = subparsers.add_parser('process', help='Process documents into vector database')
    process_parser.add_argument('--path', required=True, help='Path to directory containing documents')
    process_parser.add_argument('--recursive', action='store_true', help='Process directories recursively')
    process_parser.add_argument('--file-types', nargs='+', default=None, help='File extensions to process (e.g., .md .json)')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search for content')
    search_parser.add_argument('--query', required=True, help='Search query')
    search_parser.add_argument('--limit', type=int, default=None, help='Maximum number of results')
    search_parser.add_argument('--output', help='Output file for results (JSON format)')
    search_parser.add_argument('--simple', action='store_true', help='Use simple search instead of optimized retrieval')
    search_parser.add_argument('--filter', nargs='+', help='Filter results (format: key=value)')

    # Stats command
    subparsers.add_parser('stats', help='Show statistics about the vector database')

    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear the vector database')
    clear_parser.add_argument('--confirm', action='store_true', help='Confirm database clearing')

    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset processing state (allows reprocessing files)')
    reset_parser.add_argument('--confirm', action='store_true', help='Confirm state reset')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test API connections and Qdrant')
    test_parser.add_argument('--verbose', '-v', action='store_true', help='Show verbose test results')

    # Override specific config settings from command line
    parser.add_argument('--jina-key', help='Jina AI API key (overrides config)')
    parser.add_argument('--gemini-key', help='Gemini API key (overrides config)')
    parser.add_argument('--qdrant-url', help='Qdrant URL (overrides config)')
    parser.add_argument('--qdrant-port', type=int, help='Qdrant port (overrides config)')

    return parser.parse_args()

class ProgressFilter(logging.Filter):
    """Filter to select only INFO level progress messages for console"""
    def filter(self, record):
        # Show only progress messages (INFO level) and errors on console
        return record.levelno == logging.INFO and 'progress' in record.getMessage().lower()

def configure_logging(config, enable_detailed_logging=False):
    """Configure logging based on settings"""
    log_level = getattr(logging, config.get('logging', 'level', default='INFO'))
    log_file = config.get('logging', 'file', default='rag_retriever.log')

    # If detailed logging is enabled, use DEBUG level
    if enable_detailed_logging:
        log_level = logging.DEBUG

    # Reset root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(log_level)

    # Configure file handler to log everything to file with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8', errors='replace')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)

    # Configure error console handler (ERROR and CRITICAL)
    error_console = logging.StreamHandler(sys.stdout)
    error_console.setLevel(logging.ERROR)
    error_console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    root_logger.addHandler(error_console)

    # Configure progress console handler (only INFO messages with 'progress')
    progress_console = logging.StreamHandler(sys.stdout)
    progress_console.setLevel(logging.INFO)
    progress_console.addFilter(ProgressFilter())
    progress_console.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(progress_console)

    # Create logger for this module
    logger = logging.getLogger("rag_retriever")

    # If detailed logging is enabled, log a marker
    if enable_detailed_logging:
        logger.debug("*** DETAILED LOGGING ENABLED ***")

        # Set all loggers to DEBUG level
        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).setLevel(logging.DEBUG)

    return logger

def initialize_components(args, config):
    """Initialize components with configuration"""
    # Override config with command line arguments if provided
    jina_key = args.jina_key or config.get('jina', 'api_key')
    gemini_key = args.gemini_key or config.get('gemini', 'api_key')
    qdrant_url = args.qdrant_url or config.get('qdrant', 'url')
    qdrant_port = args.qdrant_port or config.get('qdrant', 'port')

    # Check for required API keys
    if not jina_key:
        print("Error: Jina API key not provided in config or command line.")
        print("Set JINA_API_KEY environment variable or add to config.yaml")
        sys.exit(1)

    if not gemini_key:
        print("Error: Gemini API key not provided in config or command line.")
        print("Set GEMINI_API_KEY environment variable or add to config.yaml")
        sys.exit(1)

    # Initialize processor with configuration
    processor = ContentProcessor(
        jina_api_key=jina_key,
        gemini_api_key=gemini_key,
        qdrant_url=qdrant_url,
        qdrant_port=qdrant_port,
        checkpoint_dir=config.get('embedding', 'checkpoint_dir'),
        config_path=args.config
    )

    # Initialize retriever with the same configuration
    retriever = AdvancedRetriever(
        jina_api_key=jina_key,
        gemini_api_key=gemini_key,
        qdrant_url=qdrant_url,
        qdrant_port=qdrant_port,
        config_path=args.config
    )

    return processor, retriever

def create_config_file(args):
    """Create a default configuration file"""
    config = Config()
    config.create_default_config_file(args.output)

def process_documents(processor, args, config):
    """Process documents into the vector database"""
    start_time = time.time()

    try:
        file_types = args.file_types
        if file_types:
            # Ensure all file types start with a dot
            file_types = [ft if ft.startswith('.') else f'.{ft}' for ft in file_types]

        processor.process_directory(
            directory_path=args.path,
            recursive=args.recursive,
            file_types=file_types
        )

        duration = time.time() - start_time
        print(f"Processing completed in {duration:.2f} seconds")

        # Show stats
        stats = processor.get_collection_stats()
        print("\nCollection Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)

def search_content(retriever, args, config):
    """Search for content"""
    start_time = time.time()

    try:
        # Get result limit from args or config
        limit = args.limit or config.get('retrieval', 'default_result_limit')
        logger.debug(f"Search limit: {limit}")

        # Parse filter arguments if provided
        filters = {}
        if args.filter:
            for filter_arg in args.filter:
                if '=' in filter_arg:
                    key, value = filter_arg.split('=', 1)
                    filters[key.strip()] = value.strip()

        logger.debug(f"Using filters: {filters}")
        logger.debug(f"Search query: '{args.query}'")
        logger.debug(f"Using optimized retrieval: {not args.simple}")

        # Perform search
        if filters:
            results = retriever.filter_search(
                query=args.query,
                filters=filters,
                limit=limit,
                use_optimized_retrieval=not args.simple
            )
        else:
            results = retriever.search(
                query=args.query,
                limit=limit,
                use_optimized_retrieval=not args.simple
            )

        duration = time.time() - start_time
        print(f"Search completed in {duration:.2f} seconds, found {len(results)} results")
        logger.debug(f"Search completed in {duration:.2f} seconds, found {len(results)} results")

        # Display results
        if results:
            print("\nSearch Results:")
            for i, result in enumerate(results):
                # Get score, try different possible fields with fallbacks
                score = result.get('rerank_score', result.get('score', result.get('avg_score', 0.0)))

                # Format score nicely if it's a number
                if isinstance(score, (int, float)):
                    formatted_score = f"{score:.4f}"
                else:
                    formatted_score = str(score)

                source = result.get('source_path', 'Unknown')

                print(f"\n[{i+1}] Score: {formatted_score}")
                print(f"Source: {source}")
                print(f"---")

                # Show a preview of the text
                text = result.get('text', '')
                preview_length = min(300, len(text))
                print(f"{text[:preview_length]}..." if len(text) > preview_length else text)

                logger.debug(f"Result {i+1} details: score={score}, source={source}, text_length={len(text)}")
        else:
            print("No results found")
            logger.debug("Search returned no results")

        # Save results to file if requested
        if args.output:
            output_file = args.output
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "query": args.query,
                    "filters": filters,
                    "limit": limit,
                    "count": len(results),
                    "duration_seconds": duration,
                    "results": results
                }, f, indent=2)
            print(f"\nResults saved to {output_file}")
            logger.debug(f"Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Error searching content: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)

def show_stats(processor):
    """Show statistics about the vector database"""
    try:
        stats = processor.get_collection_stats()
        print("\nCollection Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)

def clear_database(processor, args):
    """Clear the vector database"""
    if not args.confirm:
        print("Warning: This will permanently delete all vectors from the database.")
        print("To confirm, run again with --confirm")
        return

    try:
        result = processor.clear_collection(confirm=True)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("Database cleared successfully")

    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)

def reset_state(processor, args):
    """Reset processing state"""
    if not args.confirm:
        print("Warning: This will reset processing state and allow reprocessing all files.")
        print("To confirm, run again with --confirm")
        return

    try:
        result = processor.reset_processing_state(confirm=True)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("Processing state reset successfully")
            if result.get("backup"):
                print(f"Previous state backed up to: {result['backup']}")

    except Exception as e:
        logger.error(f"Error resetting state: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)

def test_connections(processor, retriever, args):
    """Test connection to APIs and Qdrant"""
    print("Testing API and database connections...\n")

    # Test Qdrant connection
    print("1. Testing Qdrant connection...")
    try:
        collection_info = processor.client.get_collection(processor.collection_name)
        print(f"✓ Qdrant connection successful")
        print(f"  Collection: {processor.collection_name}")

        # Handle different Qdrant client versions or response formats
        if hasattr(collection_info, 'points_count'):
            # Object-style response
            print(f"  Points: {collection_info.points_count}")
            print(f"  Vectors: {collection_info.vectors_count}")
        elif isinstance(collection_info, tuple) and len(collection_info) >= 2:
            # Tuple-style response
            print(f"  Points/Vectors: {collection_info[1]}")
    except Exception as e:
        print(f"✗ Qdrant connection failed: {str(e)}")
        if args.verbose:
            logger.error(f"Qdrant connection test failed", exc_info=True)

    # Test Jina AI connection
    print("\n2. Testing Jina AI API connection...")
    try:
        # Test embedding API
        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {processor.jina_api_key}",
                "Accept": "application/json"
            },
            json={
                "model": "jina-embeddings-v3",
                "input": ["Test connection to Jina AI API"],
                "task": "retrieval.passage"
            },
            timeout=10
        )

        if response.status_code == 200:
            print(f"✓ Jina AI embedding API connection successful")
            embedding = response.json()["data"][0]["embedding"]
            print(f"  Embedding dimension: {len(embedding)}")
        else:
            print(f"✗ Jina AI embedding API connection failed: {response.status_code} {response.text[:100]}")
    except Exception as e:
        print(f"✗ Jina AI API connection failed: {str(e)}")
        if args.verbose:
            logger.error(f"Jina AI API connection test failed", exc_info=True)

    # Test Gemini API connection
    print("\n3. Testing Gemini API connection...")
    try:
        response = processor.gemini_model.generate_content("Say hello!")
        print(f"✓ Gemini API connection successful")
        print(f"  Model: {processor.config.get('gemini', 'model')}")
        if args.verbose:
            print(f"  Response: {response.text[:50]}...")
    except Exception as e:
        print(f"✗ Gemini API connection failed: {str(e)}")
        if args.verbose:
            logger.error(f"Gemini API connection test failed", exc_info=True)

    print("\nConnection tests completed.")

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()

    # Load configuration
    config = get_config(args.config)

    # Configure logging
    global logger
    logger = configure_logging(config, enable_detailed_logging=args.log_details)

    # Handle config command
    if args.command == 'config':
        create_config_file(args)
        return

    # Initialize components
    processor, retriever = initialize_components(args, config)

    # Execute command
    if args.command == 'process':
        process_documents(processor, args, config)
    elif args.command == 'search':
        search_content(retriever, args, config)
    elif args.command == 'stats':
        show_stats(processor)
    elif args.command == 'clear':
        clear_database(processor, args)
    elif args.command == 'reset':
        reset_state(processor, args)
    elif args.command == 'test':
        test_connections(processor, retriever, args)
    else:
        print("Please specify a valid command. Use -h for help.")
        sys.exit(1)

if __name__ == "__main__":
    main()