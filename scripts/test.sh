#!/bin/bash
set -e
# LLM-Agent-Orchestrator Test Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "======================================"
echo "LLM-Agent-Orchestrator Tests"
echo "======================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Navigate to project root
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Please set up the environment first."
    exit 1
fi

# Run Integration Tests
echo "Running integration tests..."
python -m test.test_integration