FROM python:3.10-slim

# Set working directory before installing dependencies
WORKDIR /app

# Set environment variables for better performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/data /app/uploads /app/embedding_checkpoints /app/logs

# Copy application code
COPY . .

# Default environment variables (will be overridden by docker-compose)
ENV QDRANT_URL="qdrant" \
    QDRANT_PORT=6333 \
    EMBEDDING_JINA_MODEL=jina-embeddings-v3 \
    EMBEDDING_OPENAI_MODEL=text-embedding-3-small \
    # EXPANSION_GEMINI_MODEL=gemini-1.5-flash \
    EXPANSION_OPENAI_MODEL=gpt-4.1-nano \
    DEFAULT_EMBEDDING_PROVIDER=jina \
    DEFAULT_EXPANSION_PROVIDER=openai

# Health check
HEALTHCHECK --interval=86400s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Expose port 
EXPOSE 8000