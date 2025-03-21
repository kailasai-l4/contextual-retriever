FROM python:3.10-slim

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables (override these at runtime)
ENV JINA_API_KEY=""
ENV GEMINI_API_KEY=""
ENV RAG_API_KEY=""

# Run the API server
CMD ["python", "api.py", "--host", "0.0.0.0", "--port", "8000"]

# Expose port
EXPOSE 8000
