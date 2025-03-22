# Production Deployment Guide

This guide helps you deploy the RAG Content Retriever system for internal organizational use.

## Prerequisites

- Docker and Docker Compose installed on your deployment server
- Access to the repository containing the application code
- API keys for Jina AI, Gemini, and other required services
- Basic understanding of Docker and containerization concepts

## Environment Setup

### 1. Create Environment File

Create a `.env` file with all necessary API keys and configuration parameters:

```
# API Keys
JINA_API_KEY=your_jina_key_here
GEMINI_API_KEY=your_gemini_key_here
API_KEY=your_api_key_for_authentication
RAG_API_KEY=your_rag_system_api_key

# Qdrant Configuration
QDRANT_URL=http://qdrant:6333
QDRANT_HOST=qdrant
QDRANT_PORT=6333
COLLECTION_NAME=content_library

# Application Configuration
LOG_LEVEL=INFO
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
ENABLE_RERANKING=true
MAX_RESULTS=20
```

### 2. Create Docker Compose File

Create a `docker-compose.yml` file in your project root:

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - rag_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  rag_api:
    build:
      context: .
      dockerfile: Dockerfile
    image: rag-content-retriever:latest
    ports:
      - "8000:8000"
    volumes:
      - ./uploads:/app/uploads
      - ./data:/app/data
      - ./.env:/app/.env
    depends_on:
      qdrant:
        condition: service_healthy
    networks:
      - rag_network
    restart: unless-stopped
    environment:
      - QDRANT_URL=http://qdrant:6333
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

volumes:
  qdrant_data:
    driver: local

networks:
  rag_network:
    driver: bridge
```

### 3. Create or Update Dockerfile

Ensure your Dockerfile is properly configured for production:

```Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/data

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "api.py"]
```

## Deployment Steps

### 1. Build and Start Services

```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps
```

### 2. Verify Deployment

```bash
# Check API server logs
docker-compose logs -f rag_api

# Test API endpoint
curl -H "X-API-Key: your_api_key" http://localhost:8000/
```

### 3. Initialize Database (First-time Setup)

```bash
# Enter the container
docker-compose exec rag_api bash

# Initialize database or run tests
python main.py test
```

## Monitoring and Maintenance

### Health Monitoring

Set up monitoring for the following endpoints:
- `http://localhost:8000/` - API server health
- `http://localhost:6333/health` - Qdrant database health

### Logs

View application logs:
```bash
# View real-time logs
docker-compose logs -f

# View only API logs
docker-compose logs -f rag_api
```

### Backup and Restore

#### Database Backup

```bash
# Create a backup volume
docker volume create qdrant_backup

# Run temporary container to backup data
docker run --rm -v qdrant_data:/source -v qdrant_backup:/backup \
    alpine sh -c "cd /source && tar -czf /backup/qdrant_backup_$(date +%Y%m%d).tar.gz ."
```

#### Database Restore

```bash
# Restore from backup
docker run --rm -v qdrant_backup:/source -v qdrant_data:/destination \
    alpine sh -c "cd /destination && tar -xzf /source/qdrant_backup_20230326.tar.gz"
```

## Security Considerations

### API Key Management

- Regularly rotate the API keys used for authentication
- Store API keys securely and never commit them to version control
- Consider using a secrets management solution for larger deployments

### Network Security

- Use HTTPS for all API traffic in production
- Consider placing the API behind a reverse proxy like Nginx
- Implement IP address restrictions if appropriate for your organization

### Docker Security

- Keep Docker and container images updated
- Use specific version tags for images instead of `latest`
- Run containers with non-root users when possible
- Implement resource limits in docker-compose.yml

## Scaling Considerations

For higher load scenarios, consider:

1. **Vertical Scaling**: Allocate more resources to containers
   ```yaml
   rag_api:
     # ... other settings ...
     deploy:
       resources:
         limits:
           cpus: '2'
           memory: 4G
   ```

2. **Horizontal Scaling**: Run multiple API instances with a load balancer
   ```yaml
   rag_api:
     # ... other settings ...
     deploy:
       mode: replicated
       replicas: 3
   ```

3. **Qdrant Clustering**: For very large vector databases, consider setting up Qdrant in cluster mode.

## Troubleshooting

### Common Issues

1. **API Connection Issues**
   - Check if all containers are running: `docker-compose ps`
   - Verify network connectivity: `docker-compose exec rag_api ping qdrant`
   - Check API logs: `docker-compose logs rag_api`

2. **Database Issues**
   - Check Qdrant logs: `docker-compose logs qdrant`
   - Verify collection exists: Access the Qdrant dashboard at http://localhost:6333/dashboard

3. **Performance Problems**
   - Check container resources: `docker stats`
   - Monitor API response times through `/stats` endpoint
   - Consider scaling options mentioned above

## Production Checklist

Before going live, verify:

- [x] All API keys and credentials are set in the `.env` file
- [x] Docker containers start successfully and are healthy
- [x] API endpoints respond correctly with authentication
- [x] Data volumes are properly configured for persistence
- [x] Backup procedure is tested and working
- [x] Monitoring is set up for the application
- [x] Security measures are implemented

## Conclusion

This deployment guide should help you successfully deploy the RAG Content Retriever system for internal organizational use. For additional help or custom configurations, refer to the API documentation or contact your system administrator.
