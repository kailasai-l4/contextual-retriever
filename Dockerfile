FROM python:3.10-slim

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

# Set permissions
RUN chmod +x *.py

# Default environment variables (will be overridden by docker-compose)
ENV QDRANT_URL="qdrant" \
    QDRANT_PORT=6333

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

# Run the API server
CMD ["python", "api.py", "--host", "0.0.0.0", "--port", "8000"]

# Expose port
EXPOSE 8000