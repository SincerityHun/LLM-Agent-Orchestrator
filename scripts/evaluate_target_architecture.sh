#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# LLM-Agent-Orchestrator Target Architecture Evaluation Script
echo "======================================"
echo "LLM-Agent-Orchestrator Target Architecture Evaluation"
echo "======================================"
echo ""

# Activate virtual environment
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Please set up the environment first."
    exit 1
fi

# Parse command line arguments
INPUT_CSV="${1:-examples/eval_inputs_easy.csv}"
OUTPUT_CSV="${2:-}"
MAX_RETRY="${3:-3}"
MAX_TASKS="${4:-}"
MODE="${5:-orchestrator}"  # orchestrator or baseline

echo "Configuration:"
echo "  Mode: $MODE"
echo "  Input CSV: $INPUT_CSV"
echo "  Output CSV: ${OUTPUT_CSV:-auto-generated}"
echo "  Max Retry: $MAX_RETRY"
echo "  Max Tasks: ${MAX_TASKS:-all}"
echo ""

# Build Python command
PYTHON_CMD="python scripts/run_evaluation.py --input $INPUT_CSV --max-retry $MAX_RETRY --mode $MODE"

if [ -n "$OUTPUT_CSV" ]; then
    PYTHON_CMD="$PYTHON_CMD --output $OUTPUT_CSV"
fi

if [ -n "$MAX_TASKS" ]; then
    PYTHON_CMD="$PYTHON_CMD --max-tasks $MAX_TASKS"
fi

# Check if input file exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "❌ Error: Input file not found: $INPUT_CSV"
    exit 1
fi

echo "Starting evaluation..."
echo ""

# Run evaluation
eval $PYTHON_CMD

echo ""
echo "======================================"
echo "Evaluation Complete"
echo "======================================"
