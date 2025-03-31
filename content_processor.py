import os
import json
import glob
import hashlib
import time
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import re
import shutil
import uuid
import logging
import backoff

# External dependencies
import requests
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import markdown
import yaml
from bs4 import BeautifulSoup
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

# Import configuration
from config import get_config, get_qdrant_client

# Initialize logging
logger = logging.getLogger("content_processor")


class ContentProcessor:
    def __init__(self,
                 jina_api_key: str,
                 gemini_api_key: str,
                 qdrant_url: str = "localhost",
                 qdrant_port: int = 6333,
                 checkpoint_dir: str = None,
                 config_path: str = None):
        """
        Initialize the content processor with API keys and database connection

        Args:
            jina_api_key: API key for Jina AI services
            gemini_api_key: API key for Google's Gemini model
            qdrant_url: URL for Qdrant vector database
            qdrant_port: Port for Qdrant vector database
            checkpoint_dir: Directory to store checkpoint files
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = get_config(config_path)

        # Set API keys
        self.jina_api_key = jina_api_key
        os.environ["JINA_API_KEY"] = jina_api_key

        # Initialize Gemini
        self.gemini_api_key = gemini_api_key
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel(self.config.get('gemini', 'model', default="gemini-1.5-flash-latest"))

        # Initialize Qdrant client using the shared client
        self.client = get_qdrant_client(qdrant_url, qdrant_port)

        # Get collection name from config
        self.collection_name = self.config.get('qdrant', 'collection_name', default="content_library")

        # Get chunking parameters from config
        self.chunk_max_tokens = self.config.get('chunking', 'max_chunk_tokens', default=1000)
        self.chunk_overlap_tokens = self.config.get('chunking', 'overlap_tokens', default=100)

        # Get embedding parameters from config
        self.embedding_batch_size = self.config.get('embedding', 'batch_size', default=10)

        # Setup vector dimension based on Jina embedding model
        self.vector_size = 1024  # Jina embeddings v3 dimension

        # Set up logging
        log_file = self.config.get('logging', 'embedding_log', default="embedding_process.log")
        # Ensure the log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure file handler for this module's logger with UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding='utf-8', errors='replace')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # Add file handler to this module's logger
        if not logger.handlers:
            logger.addHandler(file_handler)

        # Set log level from config
        log_level = getattr(logging, self.config.get('logging', 'level', default="INFO"))
        logger.setLevel(log_level)

        # Checkpointing
        self.checkpoint_dir = checkpoint_dir or self.config.get('embedding', 'checkpoint_dir', default="embedding_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize state
        self.processed_files: Set[str] = set()
        self.embedded_chunks: Dict[str, bool] = {}  # chunk_id -> embedded status
        self.current_session_id = int(time.time())

        # Load state if exists
        self._load_state()

        # Ensure collection exists
        self._ensure_collection_exists()

        logger.info(f"ContentProcessor initialized with collection '{self.collection_name}'")

    def _ensure_collection_exists(self):
        """Create the vector collection if it doesn't exist"""
        try:
            if not self.client.collection_exists(self.collection_name):
                logger.info(f"Creating collection {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    ),
                    # HNSW configuration optimized for retrieval quality
                    hnsw_config=models.HnswConfigDiff(
                        m=16,
                        ef_construct=200,
                        full_scan_threshold=10000,
                        on_disk=True
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=10,
                        indexing_threshold=20000,
                        memmap_threshold=50000
                    ),
                    on_disk_payload=True
                )

                # Create payload indexes for efficient filtering
                self._create_indexes()
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Qdrant collection: {str(e)}") from e

    def _create_indexes(self):
        """Create necessary indexes for efficient filtering"""
        indexes = [
            ("source_id", models.PayloadSchemaType.KEYWORD),
            ("source_type", models.PayloadSchemaType.KEYWORD),
            ("chunk_id", models.PayloadSchemaType.KEYWORD),
            ("topic_keywords", models.PayloadSchemaType.KEYWORD),
            ("token_count", models.PayloadSchemaType.INTEGER),
        ]

        for field_name, field_type in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                logger.info(f"Created index on {field_name}")
            except Exception as e:
                logger.warning(f"Failed to create index on {field_name}: {str(e)}")

    def _load_state(self):
        """Load processing state from disk if it exists"""
        state_file = os.path.join(self.checkpoint_dir, "processor_state.pkl")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'rb') as f:
                    state = pickle.load(f)
                self.processed_files = state.get('processed_files', set())
                self.embedded_chunks = state.get('embedded_chunks', {})
                logger.info(f"Loaded state: {len(self.processed_files)} processed files, {len(self.embedded_chunks)} embedded chunks")
            except Exception as e:
                logger.error(f"Failed to load state: {str(e)}", exc_info=True)
                # Create backup of corrupted state file
                backup_file = f"{state_file}.corrupted.{int(time.time())}"
                try:
                    shutil.copy2(state_file, backup_file)
                    logger.info(f"Created backup of corrupted state file: {backup_file}")
                except Exception as backup_err:
                    logger.error(f"Failed to create backup of corrupted state file: {str(backup_err)}")

    def _save_state(self):
        """Save current processing state to disk"""
        state_file = os.path.join(self.checkpoint_dir, "processor_state.pkl")
        state = {
            'processed_files': self.processed_files,
            'embedded_chunks': self.embedded_chunks,
            'last_update': time.time()
        }
        try:
            # Write to temporary file first, then rename for atomicity
            temp_state_file = f"{state_file}.tmp"
            with open(temp_state_file, 'wb') as f:
                pickle.dump(state, f)
            # Replace original file with new one
            shutil.move(temp_state_file, state_file)
            logger.info(f"Saved state: {len(self.processed_files)} processed files, {len(self.embedded_chunks)} embedded chunks")
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}", exc_info=True)

    def process_directory(self, directory_path: str, recursive: bool = True, file_types: List[str] = None):
        """
        Process all supported files in a directory

        Args:
            directory_path: Path to directory containing files
            recursive: Whether to process subdirectories
            file_types: List of file extensions to process (e.g., ['.md', '.json'])
        """
        if file_types is None:
            file_types = ['.md', '.json', '.jsonl', '.txt', '.csv', '.yaml', '.yml']

        pattern = '**/*' if recursive else '*'
        all_files = []

        for file_type in file_types:
            all_files.extend(glob.glob(os.path.join(directory_path, pattern + file_type), recursive=recursive))

        logger.info(f"Progress: Found {len(all_files)} files to process")

        # Filter out already processed files
        files_to_process = [f for f in all_files if self._file_fingerprint(f) not in self.processed_files]
        logger.info(f"Progress: {len(files_to_process)} files need processing ({len(all_files) - len(files_to_process)} already processed)")

        # Process files with progress bar
        for file_path in tqdm(files_to_process, desc="Processing files"):
            try:
                self.process_file(file_path)
                self.processed_files.add(self._file_fingerprint(file_path))
                # Save state after each file to preserve progress
                self._save_state()
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
                continue

    def _file_fingerprint(self, file_path: str) -> str:
        """Generate a unique fingerprint for a file based on path and modification time"""
        try:
            stat = os.stat(file_path)
            return f"{file_path}:{stat.st_mtime}:{stat.st_size}"
        except Exception as e:
            logger.error(f"Error generating file fingerprint for {file_path}: {str(e)}")
            # Fallback to just the path if stat fails
            return file_path

    def process_file(self, file_path: str):
        """
        Process a single file based on its extension

        Args:
            file_path: Path to the file to process
        """
        extension = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)

        logger.info(f"Progress: Processing {file_path}")

        # Extract content based on file type
        if extension == '.md':
            chunks = self._process_markdown_file(file_path)
        elif extension == '.json':
            chunks = self._process_json_file(file_path)
        elif extension == '.jsonl':
            chunks = self._process_jsonl_file(file_path)
        elif extension == '.txt':
            chunks = self._process_text_file(file_path)
        elif extension == '.csv':
            chunks = self._process_csv_file(file_path)
        elif extension in ['.yaml', '.yml']:
            chunks = self._process_yaml_file(file_path)
        else:
            logger.warning(f"Unsupported file type: {extension}")
            return

        # Generate chunk IDs for deduplication and tracking
        source_id = self._generate_source_id(file_path)
        chunk_ids = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{source_id}_{i}"
            chunk['chunk_id'] = chunk_id
            chunk['source_id'] = source_id
            chunk['source_path'] = file_path
            chunk['source_type'] = extension
            chunk_ids.append(chunk_id)

        # Check which chunks need embedding (not already in DB)
        chunks_to_embed = []
        for chunk in chunks:
            if chunk['chunk_id'] not in self.embedded_chunks:
                chunks_to_embed.append(chunk)

        logger.info(f"Progress: Generated {len(chunks)} chunks from {file_path}, {len(chunks_to_embed)} need embedding")

        # Embed and store chunks in batches with progress tracking
        if chunks_to_embed:
            self._embed_and_store_chunks(chunks_to_embed)

    def _generate_source_id(self, file_path: str) -> str:
        """Generate a unique ID for a source file"""
        return hashlib.md5(file_path.encode('utf-8')).hexdigest()

    def _process_markdown_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a markdown file into chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Convert markdown to HTML, then to plain text
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text()

            # Extract metadata if present (YAML frontmatter)
            metadata = {}
            if content.startswith('---'):
                try:
                    end_idx = content.find('---', 3)
                    if end_idx != -1:
                        frontmatter = content[3:end_idx].strip()
                        metadata = yaml.safe_load(frontmatter)
                except Exception as e:
                    logger.warning(f"Failed to parse frontmatter in {file_path}: {str(e)}")

            # Create semantic chunks using Jina Segmenter
            chunks = self._create_semantic_chunks(text, metadata)
            return chunks
        except Exception as e:
            logger.error(f"Error processing markdown file {file_path}: {str(e)}", exc_info=True)
            return []

    def _process_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a JSON file into chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)

            chunks = []

            # Handle different JSON structures
            if isinstance(content, dict):
                # Extract text fields recursively
                texts = self._extract_text_fields(content)
                for field_name, text in texts:
                    # Create chunks with metadata
                    field_chunks = self._create_semantic_chunks(
                        text,
                        {"json_field": field_name, "source_structure": "object"}
                    )
                    chunks.extend(field_chunks)

            elif isinstance(content, list):
                # Process each list item
                for i, item in enumerate(content):
                    if isinstance(item, dict):
                        texts = self._extract_text_fields(item)
                        for field_name, text in texts:
                            field_chunks = self._create_semantic_chunks(
                                text,
                                {"json_index": i, "json_field": field_name, "source_structure": "array"}
                            )
                            chunks.extend(field_chunks)
                    elif isinstance(item, str) and len(item.strip()) > 0:
                        # Direct text in array
                        chunks.extend(self._create_semantic_chunks(
                            item,
                            {"json_index": i, "source_structure": "array"}
                        ))

            return chunks
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error processing JSON file {file_path}: {str(e)}", exc_info=True)
            return []

    def _extract_text_fields(self, obj: Dict[str, Any], parent_path: str = "") -> List[Tuple[str, str]]:
        """Recursively extract text fields from nested JSON objects"""
        texts = []

        for key, value in obj.items():
            current_path = f"{parent_path}.{key}" if parent_path else key

            if isinstance(value, str) and len(value.strip()) > 50:  # Only consider substantive text
                texts.append((current_path, value))
            elif isinstance(value, dict):
                texts.extend(self._extract_text_fields(value, current_path))
            elif isinstance(value, list):
                # Handle arrays of strings or objects
                for i, item in enumerate(value):
                    if isinstance(item, str) and len(item.strip()) > 50:
                        texts.append((f"{current_path}[{i}]", item))
                    elif isinstance(item, dict):
                        texts.extend(self._extract_text_fields(item, f"{current_path}[{i}]"))

        return texts

    def _process_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a JSONL file (one JSON object per line) into chunks"""
        chunks = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue

                    try:
                        obj = json.loads(line)

                        if isinstance(obj, dict):
                            texts = self._extract_text_fields(obj)
                            for field_name, text in texts:
                                field_chunks = self._create_semantic_chunks(
                                    text,
                                    {"jsonl_line": i, "json_field": field_name}
                                )
                                chunks.extend(field_chunks)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON on line {i+1} of {file_path}")
                        continue

            return chunks
        except Exception as e:
            logger.error(f"Error processing JSONL file {file_path}: {str(e)}", exc_info=True)
            return []

    def _process_text_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a plain text file into chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Create semantic chunks
            chunks = self._create_semantic_chunks(content, {})
            return chunks
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}", exc_info=True)
            return []

    def _process_csv_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a CSV file into chunks"""
        try:
            df = pd.read_csv(file_path)
            chunks = []

            # Process each row as a separate chunk
            for i, row in df.iterrows():
                # Convert row to string representation
                row_text = " ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])

                if row_text.strip():
                    chunks.extend(self._create_semantic_chunks(
                        row_text,
                        {"csv_row": i, "columns": list(df.columns)}
                    ))

            # Also process columns with substantial text
            for col in df.columns:
                if df[col].dtype == 'object':  # Text columns
                    # Combine text from column if values are substantial
                    texts = [str(val) for val in df[col] if pd.notna(val) and len(str(val).strip()) > 100]
                    if texts:
                        col_text = "\n".join(texts)
                        chunks.extend(self._create_semantic_chunks(
                            col_text,
                            {"csv_column": col}
                        ))

            return chunks
        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {str(e)}", exc_info=True)
            return []

    def _process_yaml_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a YAML file into chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)

            # Process similar to JSON
            chunks = []

            if isinstance(content, dict):
                texts = self._extract_text_fields(content)
                for field_name, text in texts:
                    field_chunks = self._create_semantic_chunks(
                        text,
                        {"yaml_field": field_name}
                    )
                    chunks.extend(field_chunks)

            return chunks
        except Exception as e:
            logger.error(f"Error processing YAML {file_path}: {str(e)}", exc_info=True)
            return []

    def _create_semantic_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create semantic chunks from text using Jina Segmenter API

        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk

        Returns:
            List of chunk objects with text and metadata
        """
        if not text or len(text.strip()) < 100:
            return []

        try:
            # Use Jina Segmenter for optimal chunking
            response = requests.post(
                "https://segment.jina.ai/",
                headers={
                    "Authorization": f"Bearer {self.jina_api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                json={
                    "content": text,
                    "tokenizer": "cl100k_base",
                    "return_chunks": True,
                    "max_chunk_length": self.chunk_max_tokens,
                    "return_tokens": False
                }
            )

            if response.status_code != 200:
                # Truncate response text to avoid encoding issues
                if len(response.text) > 200:
                    response_text = response.text[:200] + "... [text truncated]"
                else:
                    response_text = response.text

                logger.warning(f"Segmenter API error: {response.status_code} {response_text}")
                # Fall back to simple chunking
                logger.info("Progress: Falling back to simple chunking due to Segmenter API error")
                return self._fallback_chunking(text, metadata)

            result = response.json()
            chunks = result.get("chunks", [])

            # Create chunk objects with metadata
            chunk_objects = []

            for i, chunk_text in enumerate(chunks):
                # Skip very short chunks
                if len(chunk_text.split()) < 20:
                    continue

                # Create overlap for consecutive chunks
                if i > 0:
                    # Extract end of previous chunk and start of current chunk
                    overlap_start = chunks[i-1][-self.chunk_overlap_tokens:]
                    overlap_end = chunk_text[:self.chunk_overlap_tokens]
                    overlap_text = overlap_start + overlap_end

                    # Create an overlap chunk
                    chunk_objects.append({
                        "text": overlap_text,
                        "is_overlap": True,
                        "chunk_index": i - 0.5,
                        "metadata": metadata.copy(),
                        "estimated_tokens": len(overlap_text.split()) * 1.3,
                        "keywords": self._extract_keywords(overlap_text)
                    })

                # Add the main chunk
                chunk_objects.append({
                    "text": chunk_text,
                    "is_overlap": False,
                    "chunk_index": i,
                    "metadata": metadata.copy(),
                    "estimated_tokens": len(chunk_text.split()) * 1.3,
                    "keywords": self._extract_keywords(chunk_text)
                })

            return chunk_objects

        except Exception as e:
            logger.error(f"Error during semantic chunking: {str(e)}", exc_info=True)
            return self._fallback_chunking(text, metadata)

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=backoff.full_jitter)
    def _segment_content(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Segment content into chunks using Jina's API with improved error handling
        
        Args:
            content: Text content to segment
            file_path: Original file path for reference
            
        Returns:
            List of content chunks
        """
        # Check for oversized content and split if needed before sending to API
        content_size = len(content.encode('utf-8'))
        max_api_size = 60000  # 60KB, safely under Jina's 64KB limit
        
        if content_size > max_api_size:
            logger.info(f"Content too large for segmenter API ({content_size} bytes), splitting into smaller parts")
            return self._fallback_chunking(content, file_path)
        
        try:
            # Attempt to use Jina's segmenter API
            segments = self._call_segmenter_api(content)
            return segments
        except Exception as e:
            logger.warning(f"Segmenter API error: {str(e)}")
            logger.info("Progress: Falling back to simple chunking due to Segmenter API error")
            return self._fallback_chunking(content, file_path)
            
    def _fallback_chunking(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Fallback method to chunk content when the segmenter API fails
        
        Args:
            content: Text content to segment
            file_path: Original file path for reference
            
        Returns:
            List of content chunks
        """
        # Split text into paragraphs
        paragraphs = [p for p in re.split(r'\n\s*\n', content) if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_token_count = 0
        chunk_id = 1
        
        for para in paragraphs:
            # Estimate token count (rough approximation)
            para_token_count = len(para.split())
            
            # If adding this paragraph would exceed the chunk size, save the current chunk
            if current_token_count + para_token_count > self.chunk_max_tokens and current_chunk:
                chunks.append({
                    "chunk_id": f"{Path(file_path).stem}-{chunk_id}",
                    "text": current_chunk.strip(),
                    "token_count": current_token_count
                })
                chunk_id += 1
                current_chunk = para
                current_token_count = para_token_count
            else:
                current_chunk += "\n\n" + para if current_chunk else para
                current_token_count += para_token_count
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append({
                "chunk_id": f"{Path(file_path).stem}-{chunk_id}",
                "text": current_chunk.strip(),
                "token_count": current_token_count
            })
        
        return chunks

    def _embed_and_store_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Embed and store chunks in batches with progress tracking

        Args:
            chunks: List of chunk objects to embed and store
        """
        embedded_count = 0

        # Process in batches
        for i in tqdm(range(0, len(chunks), self.embedding_batch_size), desc="Embedding chunks"):
            batch = chunks[i:i + self.embedding_batch_size]
            batch_texts = [chunk["text"] for chunk in batch]

            # Retry loop for embedding generation
            max_retries = 3
            retry_delay = 2  # seconds
            success = False

            for retry in range(max_retries):
                try:
                    # Generate embeddings using Jina AI
                    logger.debug(f"Generating embeddings for batch of {len(batch_texts)} texts (retry {retry+1}/{max_retries})")
                    embeddings = self._generate_embeddings(batch_texts)
                    logger.debug(f"Successfully generated {len(embeddings)} embeddings")
                    success = True
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        logger.warning(f"Embedding generation failed (attempt {retry+1}/{max_retries}): {str(e)}")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"All retry attempts failed for embedding generation: {str(e)}", exc_info=True)

            if not success:
                logger.error(f"Skipping batch {i//self.embedding_batch_size + 1} due to embedding failures")
                continue

            # Prepare points for Qdrant with UUID format IDs
            points = []
            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                # Generate a proper UUID for each point
                point_id = str(uuid.uuid4())

                # Prepare metadata for storage
                metadata = chunk["metadata"].copy() if "metadata" in chunk else {}

                # Prepare payload
                payload = {
                    "text": chunk["text"],
                    "source_id": chunk.get("source_id", ""),
                    "source_path": chunk.get("source_path", ""),
                    "source_type": chunk.get("source_type", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "is_overlap": chunk.get("is_overlap", False),
                    "topic_keywords": chunk.get("keywords", []),
                    "token_count": int(chunk.get("estimated_tokens", 0)),
                    "metadata": metadata,
                    "embedding_time": time.time(),
                    "session_id": self.current_session_id
                }

                # Add source type if provided
                if "source_type" in chunk:
                    payload["source_type"] = chunk["source_type"]

                # Add chunking info if available
                if "token_count" in chunk:
                    payload["chunking"] = {
                        "token_count": chunk.get("token_count"),
                        "chunk_method": metadata.get("chunk_method", "token_based")
                    }

                # Create point with proper UUID format
                points.append(models.PointStruct(
                    id=point_id,  # UUID string
                    vector=embedding,
                    payload=payload
                ))

            # Log detailed information about the points
            logger.debug(f"Prepared {len(points)} points for upsert to Qdrant")

            # Check if points list is not empty
            if not points:
                logger.warning("No points to upsert - empty batch")
                continue

            # Upload to Qdrant with retry logic
            max_retries = 3
            retry_delay = 2  # seconds
            upsert_success = False

            for retry in range(max_retries):
                try:
                    logger.debug(f"Upserting {len(points)} points to Qdrant collection {self.collection_name} (retry {retry+1}/{max_retries})")
                    # Print first point ID for debugging
                    if points:
                        logger.debug(f"First point ID: {points[0].id}, Vector length: {len(points[0].vector)}")

                    operation_info = self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True
                    )
                    logger.debug(f"Upsert operation completed successfully: {operation_info}")

                    # Verify points were actually added
                    try:
                        collection_info = self.client.get_collection(self.collection_name)
                        if hasattr(collection_info, 'vectors_count') and hasattr(collection_info, 'points_count'):
                            logger.debug(f"Collection now has {collection_info.vectors_count} vectors and {collection_info.points_count} points")
                        else:
                            # Handle case where collection_info might be a tuple or different structure
                            logger.debug(f"Collection info retrieved successfully (details not available)")
                    except Exception as e:
                        logger.error(f"Error verifying point count: {str(e)}")

                    # Mark as embedded in tracking
                    for chunk in batch:
                        self.embedded_chunks[chunk.get("chunk_id", "")] = True

                    embedded_count += len(batch)
                    logger.info(f"Progress: Successfully embedded and stored {len(batch)} chunks")
                    upsert_success = True
                    break

                except Exception as e:
                    if retry < max_retries - 1:
                        logger.warning(f"Qdrant upsert failed (attempt {retry+1}/{max_retries}): {str(e)}")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"All retry attempts failed for Qdrant upsert: {str(e)}", exc_info=True)
                        # Try to get more details about the error
                        if points:
                            logger.error(f"First point ID: {points[0].id}")
                            logger.error(f"Vector dimension: {len(points[0].vector) if points[0].vector else 'Unknown'}")

            if not upsert_success:
                logger.error(f"Failed to upsert batch {i//self.embedding_batch_size + 1}")
                continue

            # Checkpoint periodically
            if embedded_count % (self.embedding_batch_size * 5) == 0 and embedded_count > 0:
                self._save_state()

        logger.info(f"Progress: Embedded and stored {embedded_count} chunks in total")
        self._save_state()

    @backoff.on_exception(backoff.expo,
                         (requests.exceptions.RequestException,
                          requests.exceptions.Timeout,
                          requests.exceptions.ConnectionError),
                         max_tries=5)
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Jina AI

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (vectors)
        """
        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.jina_api_key}",
                "Accept": "application/json"
            },
            json={
                "model": self.config.get('jina', 'embedding_model', default="jina-embeddings-v3"),
                "input": texts,
                "task": "retrieval.passage"  # Optimize for retrieval
            },
            timeout=60  # Set a timeout to avoid hanging
        )

        if response.status_code != 200:
            response_text = response.text[:1000] if len(response.text) > 1000 else response.text
            raise Exception(f"Embedding API error: {response.status_code} {response_text}")

        result = response.json()
        embeddings = [item["embedding"] for item in result["data"]]

        return embeddings

    def expand_query(self, query: str, expansion_size: int = 100) -> str:
        """
        Expand query using Gemini model for improved retrieval

        Args:
            query: Original query to expand
            expansion_size: Target number of expansion terms

        Returns:
            Expanded query
        """
        try:
            # Prompt Gemini for query expansion
            prompt = f"""
            I need to expand this search query for a vector database lookup:

            QUERY: {query}

            Please generate around {expansion_size} related terms, synonyms, and concepts that would help improve retrieval.
            Format the result as a comma-separated list of terms.
            Focus on semantic relationships and domain-specific terminology.
            Return ONLY the comma-separated terms without explanations or other text.
            """

            response = self.gemini_model.generate_content(prompt)
            expansion_terms = response.text

            # Clean up the response
            if ',' in expansion_terms:
                terms = [term.strip() for term in expansion_terms.split(',')]
            else:
                terms = expansion_terms.split()

            # Limit to target size
            if len(terms) > expansion_size:
                terms = terms[:expansion_size]

            expanded_query = f"{query} {' '.join(terms)}"
            return expanded_query

        except Exception as e:
            logger.error(f"Query expansion error: {str(e)}", exc_info=True)
            return query  # Return original query on error

    @backoff.on_exception(backoff.expo,
                         (requests.exceptions.RequestException,
                          requests.exceptions.Timeout,
                          requests.exceptions.ConnectionError),
                         max_tries=3)
    def search(self, query: str, limit: int = 10, use_expansion: bool = True):
        """
        Search for content using the vector database

        Args:
            query: Search query
            limit: Maximum number of results to return
            use_expansion: Whether to use query expansion

        Returns:
            Search results with metadata
        """
        # Ensure limit is at least 1 to prevent Qdrant API errors
        limit = max(1, limit)
        
        try:
            search_query = query

            # Expand query if requested
            if use_expansion:
                search_query = self.expand_query(query)
                logger.info(f"Expanded query: {len(search_query.split())} terms")

            # Generate embedding for the query
            response = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.jina_api_key}",
                    "Accept": "application/json"
                },
                json={
                    "model": self.config.get('jina', 'embedding_model', default="jina-embeddings-v3"),
                    "input": [search_query],
                    "task": "retrieval.query"  # Important: optimization for query embedding
                }
            )

            if response.status_code != 200:
                raise Exception(f"Embedding API error: {response.status_code} {response.text}")

            query_embedding = response.json()["data"][0]["embedding"]

            # Search in Qdrant with oversampling
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit * 2,  # Oversample for post-processing
                score_threshold=0.5  # Minimum similarity threshold
            )

            # Extract result information and deduplicate
            results = []
            seen_content = set()

            for hit in search_results:
                text = hit.payload.get("text", "")
                content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

                # Skip near-duplicates
                if content_hash in seen_content:
                    continue
                seen_content.add(content_hash)

                results.append({
                    "text": text,
                    "score": hit.score,
                    "source_id": hit.payload.get("source_id", ""),
                    "source_path": hit.payload.get("source_path", ""),
                    "metadata": hit.payload.get("metadata", {}),
                    "keywords": hit.payload.get("topic_keywords", [])
                })

            # Apply reranking for better relevance
            if len(results) > 1:
                texts = [result["text"] for result in results]

                rerank_response = requests.post(
                    "https://api.jina.ai/v1/rerank",
                    headers={
                        "Authorization": f"Bearer {self.jina_api_key}",
                        "Accept": "application/json"
                    },
                    json={
                        "model": self.config.get('jina', 'reranker_model', default="jina-reranker-v2-base-multilingual"),
                        "query": query,  # Use original query for reranking
                        "documents": texts,
                        "top_n": min(limit, len(texts))
                    }
                )

                if rerank_response.status_code == 200:
                    # Create a mapping of text to new rank
                    new_scores = {}
                    for idx, item in enumerate(rerank_response.json()["results"]):
                        new_scores[item["document"]["text"]] = {
                            "new_score": item["relevance_score"],
                            "new_rank": idx
                        }

                    # Apply new scores to our results
                    for result in results:
                        if result["text"] in new_scores:
                            result["rerank_score"] = new_scores[result["text"]]["new_score"]
                            result["rerank_position"] = new_scores[result["text"]]["new_rank"]

                    # Sort by reranker position
                    results.sort(key=lambda x: x.get("rerank_position", 999))

            # Limit to requested number
            return results[:limit]

        except Exception as e:
            logger.error(f"Search error: {str(e)}", exc_info=True)
            return []

    def get_collection_stats(self):
        """Get statistics about the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # Initialize stats dictionary with default values
            stats = {
                "vectors_count": None,
                "points_count": 0,
                "segments_count": None,
                "status": "unknown",
                "unique_sources": 0,
                "avg_chunks_per_source": 0,
                "collection_name": self.collection_name,
                "file_types": {},
                "recently_processed": [],
                "vector_dimension": 1024  # Default for Jina embeddings
            }
            
            # Handle different Qdrant client versions or response formats
            if hasattr(collection_info, 'vectors_count'):
                # Object-style response
                stats.update({
                    "vectors_count": collection_info.vectors_count,
                    "points_count": collection_info.points_count,
                    "segments_count": collection_info.segments_count,
                    "status": collection_info.status
                })
            elif isinstance(collection_info, tuple) and len(collection_info) >= 2:
                # Tuple-style response format
                stats.update({
                    "vectors_count": collection_info[0],
                    "points_count": collection_info[1],
                    "status": "active"
                })
            else:
                # Fallback for unknown format
                stats.update({
                    "status": "active",
                    "collection_exists": True
                })

            # Get unique source counts and additional metadata
            try:
                # Try to safely get scroll results and handle different formats
                scroll_results = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=10000,
                    with_payload=["source_id", "source_type", "metadata", "content_title", "chunking"],
                    with_vectors=False
                )

                unique_sources = set()
                file_types = {}
                recent_documents = []
                processed_timestamps = {}

                # Handle different scroll_results formats
                try:
                    # Check if it's a direct tuple response
                    if isinstance(scroll_results, tuple):
                        # If it's a tuple (points, next_page_offset)
                        points = scroll_results[0]
                        for point in points:
                            if hasattr(point, 'payload'):
                                payload = point.payload
                                
                                # Track unique sources
                                if "source_id" in payload:
                                    source_id = payload["source_id"]
                                    unique_sources.add(source_id)
                                    
                                    # Track file types
                                    if "source_type" in payload:
                                        source_type = payload["source_type"]
                                        file_types[source_type] = file_types.get(source_type, 0) + 1
                                    
                                    # Track processing timestamps for recency
                                    if "metadata" in payload and "processed_at" in payload["metadata"]:
                                        processed_at = payload["metadata"]["processed_at"]
                                        processed_timestamps[source_id] = (processed_at, payload.get("content_title", source_id))
                    
                    # Handle iterator differently to avoid unpacking errors
                    elif hasattr(scroll_results, '__iter__'):
                        try:
                            for item in scroll_results:
                                # Check if item is a tuple with two elements (batch, offset)
                                if isinstance(item, (tuple, list)) and len(item) == 2:
                                    batch = item[0]
                                else:
                                    batch = item

                                # Process the batch
                                if isinstance(batch, (list, tuple)) or hasattr(batch, '__iter__'):
                                    for point in batch:
                                        if hasattr(point, 'payload'):
                                            payload = point.payload
                                            
                                            # Track unique sources
                                            if "source_id" in payload:
                                                source_id = payload["source_id"]
                                                unique_sources.add(source_id)
                                                
                                                # Track file types
                                                if "source_type" in payload:
                                                    source_type = payload["source_type"]
                                                    file_types[source_type] = file_types.get(source_type, 0) + 1
                                                
                                                # Track processing timestamps for recency
                                                if "metadata" in payload and "processed_at" in payload["metadata"]:
                                                    processed_at = payload["metadata"]["processed_at"]
                                                    processed_timestamps[source_id] = (processed_at, payload.get("content_title", source_id))
                        except Exception as iter_error:
                            logger.error(f"Error iterating scroll results: {str(iter_error)}")
                except Exception as e:
                    logger.error(f"Error processing scroll results: {str(e)}")

                # Add unique sources count
                source_count = len(unique_sources)
                stats["unique_sources"] = source_count
                
                # Calculate average chunks per source
                if source_count > 0:
                    stats["avg_chunks_per_source"] = round(stats["points_count"] / source_count, 2)
                
                # Add file type distribution
                stats["file_types"] = file_types
                
                # Add recently processed documents (up to 5)
                recent_docs = sorted(processed_timestamps.items(), key=lambda x: x[1][0], reverse=True)[:5]
                stats["recently_processed"] = [{"source_id": src_id, "title": title} for src_id, (_, title) in recent_docs]

            except Exception as e:
                logger.warning(f"Failed to gather detailed stats: {str(e)}")
                stats["unique_sources"] = "Unknown"

            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}", exc_info=True)
            return {"error": str(e)}

    def clear_collection(self, confirm: bool = False):
        """Clear all vectors from the collection (dangerous!)"""
        if not confirm:
            return {"error": "Operation not confirmed. Set confirm=True to proceed."}

        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection_exists()  # Recreate empty collection

            # Reset state
            self.embedded_chunks = {}
            self._save_state()

            return {"status": "Collection cleared successfully"}

        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}", exc_info=True)
            return {"error": str(e)}

    def reset_processing_state(self, confirm: bool = False):
        """Reset the processing state (allows reprocessing all files)"""
        if not confirm:
            return {"error": "Operation not confirmed. Set confirm=True to proceed."}

        try:
            # Backup current state
            state_file = os.path.join(self.checkpoint_dir, "processor_state.pkl")
            backup_file = None

            if os.path.exists(state_file):
                backup_file = os.path.join(self.checkpoint_dir, f"processor_state_backup_{int(time.time())}.pkl")
                shutil.copy2(state_file, backup_file)

            # Reset state
            self.processed_files = set()
            self.embedded_chunks = {}
            self._save_state()

            return {
                "status": "Processing state reset successfully",
                "backup": backup_file
            }

        except Exception as e:
            logger.error(f"Failed to reset processing state: {str(e)}", exc_info=True)
            return {"error": str(e)}


# Example usage
if __name__ == "__main__":
    # Load config
    from config import get_config
    config = get_config()

    # Initialize the processor
    processor = ContentProcessor(
        jina_api_key=config.get('jina', 'api_key', default="your_jina_api_key_here"),
        gemini_api_key=config.get('gemini', 'api_key', default="your_gemini_api_key_here")
    )

    # Process a directory
    processor.process_directory("./test_docs")

    # Search for content
    results = processor.search("query here", limit=10)