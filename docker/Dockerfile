FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY ../requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY .. /app/

# Expose ports for vLLM endpoints
EXPOSE 8000 8001 8002 8003

# Default command
CMD ["python3", "main.py"]
