# Multi-stage build for optimized production image

# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies to /app/venv
COPY requirements.txt .
RUN python -m venv /app/venv && \
    /app/venv/bin/pip install --upgrade pip && \
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Stage 2: Production
FROM python:3.10-slim

WORKDIR /app

# Copy Python virtual environment from builder
COPY --from=builder /app/venv /app/venv

# Set PATH to use the virtual environment
ENV PATH=/app/venv/bin:$PATH

# Ensure Python is the venv python
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Create necessary directories
RUN mkdir -p logs logs/metrics models/registry data

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Create a non-root user for security
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app

# Switch to non-root user
USER apiuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Run the production API (use python -m to avoid permission issues)
CMD ["python", "-m", "uvicorn", "src.api_production:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
