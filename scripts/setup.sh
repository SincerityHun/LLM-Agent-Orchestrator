#!/bin/bash

# LLM-Agent-Orchestrator Setup Script
# Builds and starts all services

set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "======================================"
echo "LLM-Agent-Orchestrator Setup"
echo "======================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi
echo "✓ Docker found"

# Check Docker Compose
if ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose is not installed"
    exit 1
fi
echo "✓ Docker Compose found"

# Check NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found (NVIDIA GPU required)"
    exit 1
fi
echo "✓ NVIDIA GPU detected"

# Check model checkpoints
MODELS_PATH="/mnt/ssd1/shjung/embos"
if [ ! -d "$MODELS_PATH" ]; then
    echo "❌ Model checkpoints not found at $MODELS_PATH"
    exit 1
fi
echo "✓ Model checkpoints found"

# Check required checkpoint directories
REQUIRED_DIRS=(
    # "$MODELS_PATH/csqa-router/final_model" --- IGNORE ---
    "$MODELS_PATH/medqa-router/final_model"
    "$MODELS_PATH/casehold-router/final_model"
    "$MODELS_PATH/mathqa-router/final_model"
    "$MODELS_PATH/llama-1b-csqa"
    "$MODELS_PATH/llama-1b-medqa"
    "$MODELS_PATH/llama-1b-casehold"
    "$MODELS_PATH/llama-1b-mathqa"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "❌ Required checkpoint not found: $dir"
        exit 1
    fi
done
echo "✓ All required checkpoints present"

echo ""
echo "======================================"
echo "Building Docker Images"
echo "======================================"

cd "$PROJECT_ROOT/docker"

# Build router service image
echo ""
echo "Building router-service..."
docker compose build router-service

# vLLM images should pull automatically
# echo ""
# echo "Pulling vLLM images..."
# docker compose pull vllm-llama-1b vllm-llama-8b

echo ""
echo "======================================"
echo "Starting Services"
echo "======================================"

# Stop existing containers
echo ""
echo "Stopping existing containers..."
docker compose down

# Start all services
echo ""
echo "Starting all services..."
docker compose up -d

# Wait for services to be ready
echo ""
echo "Waiting for services to start..."
sleep 10

# Check service health
echo ""
echo "======================================"
echo "Checking Service Health"
echo "======================================"

check_service() {
    local name=$1
    local url=$2
    
    echo -n "Checking $name... "
    
    for i in {1..30}; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo "✓ Ready"
            return 0
        fi
        sleep 2
    done
    
    echo "✗ Not responding"
    return 1
}

check_service "vLLM 1B" "http://localhost:8000/health"
check_service "vLLM 8B" "http://localhost:8001/health"
check_service "Router Service" "http://localhost:8002/health"

echo ""
echo "======================================"
echo "Service Status"
echo "======================================"

docker compose ps

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Services running:"
echo "  - vLLM 1B:       http://localhost:8000"
echo "  - vLLM 8B:       http://localhost:8001"
echo "  - Router:        http://localhost:8002"
echo ""
echo "Next steps:"
echo "  1. Test router API:       cd $PROJECT_ROOT && python scripts/test_router_api.py"
echo "  2. Run full test suite:   cd $PROJECT_ROOT && python test/test_domain_architecture.py"
echo "  3. View logs:             cd $PROJECT_ROOT/docker && docker compose logs -f"
echo "  4. Stop services:         cd $PROJECT_ROOT/docker && docker compose down"
echo ""
