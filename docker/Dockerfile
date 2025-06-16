# docker/Dockerfile
# Docker container for Furcate Nano development and testing
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY furcate_nano/ ./furcate_nano/
COPY configs/ ./configs/
COPY setup.py .
COPY README.md .

# Install Furcate Nano
RUN pip install -e .

# Create data directory
RUN mkdir -p /data/furcate-nano

# Expose ports for mesh networking
EXPOSE 8080 8081

# Set environment variables
ENV PYTHONPATH=/app
ENV FURCATE_NANO_CONFIG=/app/configs/default-config.yaml

# Create non-root user
RUN useradd -m -u 1000 furcate
RUN chown -R furcate:furcate /app /data
USER furcate

# Default command
CMD ["furcate-nano", "start", "--config", "/app/configs/default-config.yaml"]
