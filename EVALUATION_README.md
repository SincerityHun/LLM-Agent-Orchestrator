# Evaluation System for LLM-Agent-Orchestrator

This directory contains scripts for evaluating the LLM-Agent-Orchestrator system performance.

## Overview

The evaluation system measures:
- **Final Answer**: The output generated for each task
- **End-to-End Latency**: Time taken to complete each task
- **Iterations**: Number of refinement iterations needed
- **Reason**: Why the task completed or failed
- **FLOPs**: Computational cost in TFLOPs (Tera FLOPs)

## Files

### 1. `run_evaluation.py`
Python script that processes tasks from CSV and collects metrics.

**Usage:**
```bash
python scripts/run_evaluation.py \
    --input examples/eval_inputs.csv \
    --output examples/eval_results.csv \
    --max-retry 3 \
    --max-tasks 10
```

**Arguments:**
- `--input, -i`: Input CSV file with tasks (default: `examples/eval_inputs.csv`)
- `--output, -o`: Output CSV file for results (default: auto-generated with timestamp)
- `--max-retry, -r`: Maximum retry count (default: 3)
- `--max-tasks, -n`: Maximum number of tasks to process (default: all)

### 2. `evaluate_target_architecture.sh`
Bash wrapper script for easier execution.

**Usage:**
```bash
./scripts/evaluate_target_architecture.sh [INPUT_CSV] [OUTPUT_CSV] [MAX_RETRY] [MAX_TASKS]
```

**Example:**
```bash
# Run all tasks with default settings
./scripts/evaluate_target_architecture.sh

# Run first 5 tasks
./scripts/evaluate_target_architecture.sh examples/eval_inputs.csv results.csv 3 5

# Run with specific output file
./scripts/evaluate_target_architecture.sh examples/eval_inputs.csv my_results.csv
```

### 3. `visualize_results.py`
Creates visualization charts and summary statistics.

**Usage:**
```bash
python scripts/visualize_results.py \
    --input examples/eval_results_20251215_120000.csv \
    --output-dir examples/visualizations
```

**Generated Outputs:**
- `flops_comparison.png` - Compares FLOPs usage vs baseline
- `latency_distribution.png` - Shows latency distribution
- `iterations_vs_flops.png` - Relationship between iterations and cost
- `agent_usage.png` - Component usage statistics
- `summary_stats.txt` - Detailed text summary

## Input CSV Format

The input CSV (`eval_inputs.csv`) should have these columns:
- `response_text`: Contains the task prefixed with "[Origin Task]:"
- `run_idx`: Run index (optional)
- Other metadata columns (optional)

**Example:**
```csv
,run_idx,response_id,model,response_text,input_tokens_api,output_tokens_api
2025-12-13T17:42:41,0,resp_xxx,gpt-5.2,"[Origin Task]:
Your task description here...",232,356
```

## Output CSV Format

The output CSV contains these columns:
- `task_id`: Task identifier
- `run_idx`: Run index
- `success`: Whether task completed successfully
- `final_answer`: Generated answer (truncated to 500 chars)
- `latency_seconds`: End-to-end latency
- `iterations`: Number of iterations used
- `reason`: Completion reason (completed/max_retry_reached/exception)
- `total_flops_tflops`: Total FLOPs in TFLOPs
- `router_flops_tflops`: Router FLOPs
- `agent_flops_tflops`: Agent FLOPs
- `handler_flops_tflops`: Result handler FLOPs
- `total_tokens`: Total tokens processed
- `router_calls`: Number of router calls
- `agent_calls`: Number of agent calls
- `handler_calls`: Number of handler calls
- `original_task_preview`: First 200 chars of original task

## FLOPs Calculation

FLOPs are calculated using the Kaplan Scaling Law:

```
FLOPs ≈ 2 × Parameters × Total_Tokens
```

**Model Sizes:**
- Global Router: 8B parameters (Llama-3.1-8B)
- Result Handler: 8B parameters (Llama-3.1-8B)
- Agent (small): 1B parameters (Llama-1B)
- Agent (large): 8B parameters (Llama-8B)

**Baseline Comparison:**
The visualization compares against a single 23B model baseline (similar to GPT-3.5 scale).

## Complete Workflow

1. **Prepare your input CSV:**
   ```bash
   # Your CSV should be in examples/eval_inputs.csv
   ls examples/eval_inputs.csv
   ```

2. **Run evaluation:**
   ```bash
   # Make script executable
   chmod +x scripts/evaluate_target_architecture.sh
   
   # Run evaluation (first 3 tasks for testing)
   ./scripts/evaluate_target_architecture.sh examples/eval_inputs.csv results.csv 3 3
   ```

3. **Visualize results:**
   ```bash
   python scripts/visualize_results.py \
       --input results.csv \
       --output-dir visualizations
   ```

4. **Review outputs:**
   ```bash
   # View summary
   cat visualizations/summary_stats.txt
   
   # View charts
   ls visualizations/*.png
   ```

## Example Output

### Summary Statistics
```
======================================================================
EVALUATION SUMMARY STATISTICS
======================================================================

Tasks Processed: 10
Success Rate: 90.0%
Failed Tasks: 1

LATENCY STATISTICS:
  Mean: 45.32s
  Median: 42.10s
  Min: 28.50s
  Max: 68.90s

FLOPS STATISTICS (TFLOPs):
  Mean: 12.3456
  Total: 123.456

ITERATION STATISTICS:
  Mean: 1.8
  Median: 2.0

COMPONENT USAGE (Average per task):
  Router Calls: 1.8
  Agent Calls: 3.2
  Handler Calls: 1.8
======================================================================
```

### FLOPs Comparison
The system generates a bar chart showing:
- Your orchestrator system (blue bars)
- Baseline 23B model (red bars)  
- Efficiency multiplier (green text, e.g., "×12.5")

## Troubleshooting

### Virtual Environment Issues
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
pip install matplotlib pandas numpy
```

### Empty Results
- Check that vLLM servers are running
- Verify Docker containers are up: `docker-compose ps`
- Check endpoint connectivity

### Memory Issues
- Process fewer tasks at once using `--max-tasks`
- Increase Docker container memory limits

## Dependencies

Required Python packages:
- pandas
- matplotlib
- numpy
- csv (built-in)
- json (built-in)

Install with:
```bash
pip install pandas matplotlib numpy
```

## Notes

- The evaluation system tracks actual token usage from vLLM responses
- FLOPs are calculated based on the Kaplan Scaling Law (2 × N × tokens)
- Baseline comparison uses a hypothetical 23B model for reference
- Results are automatically timestamped to avoid overwriting
