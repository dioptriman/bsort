# Dockerfile for bsort - Bottle Cap Color Detection
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY src/ src/
COPY settings.yaml .

# Create models directory
RUN mkdir -p models

# Install package
RUN pip install --no-cache-dir -e .

# Set entrypoint
ENTRYPOINT ["bsort"]

# Default command (show help)
CMD ["--help"]
