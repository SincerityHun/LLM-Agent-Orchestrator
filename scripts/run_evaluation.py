"""
Evaluation runner for LLM-Agent-Orchestrator
Processes eval_inputs.csv and generates metrics for each task
"""
import csv
import json
import sys
import os
from datetime import datetime
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import LLMOrchestrator


def extract_task_from_response_text(response_text: str) -> str:
    """
    Extract the actual task from response_text field
    The format is: "[Origin Task]:\n<actual task>"
    """
    if not response_text:
        return ""
    
    # Remove "[Origin Task]:" prefix
    if "[Origin Task]:" in response_text:
        task = response_text.split("[Origin Task]:", 1)[1].strip()
    else:
        task = response_text.strip()
    
    return task


def load_tasks_from_csv(csv_path: str) -> List[Dict]:
    """
    Load tasks from eval_inputs.csv
    
    Returns:
        List of dictionaries with task information
    """
    tasks = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            response_text = row.get('response_text', '')
            task = extract_task_from_response_text(response_text)
            
            if task:
                tasks.append({
                    'task_id': idx,
                    'run_idx': row.get('run_idx', idx),
                    'task': task,
                    'original_response_text': response_text,
                    'input_tokens_api': row.get('input_tokens_api', ''),
                    'output_tokens_api': row.get('output_tokens_api', ''),
                })
    
    return tasks


def get_csv_fieldnames():
    """Get CSV column names"""
    return [
        'task_id',
        'run_idx',
        'success',
        'final_answer',
        'answer_length',
        'latency_seconds',
        'iterations',
        'reason',
        'total_flops_tflops',
        'router_flops_tflops',
        'agent_flops_tflops',
        'handler_flops_tflops',
        'total_tokens',
        'router_calls',
        'agent_calls',
        'handler_calls',
        'original_task_preview'
    ]


def initialize_csv(output_path: str):
    """
    Initialize CSV file with headers
    """
    fieldnames = get_csv_fieldnames()
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    print(f"✓ Initialized CSV file: {output_path}\n")


def append_result_to_csv(result: Dict, output_path: str):
    """
    Append a single result to CSV file immediately
    """
    fieldnames = get_csv_fieldnames()
    
    # Prepare row data
    metrics = result.get('metrics', {})
    
    # Don't truncate final_answer - keep the full answer
    final_answer = result.get('final_answer', '')
    # If final_answer is empty, use a placeholder
    if not final_answer:
        final_answer = "[No answer generated]"
    
    # Truncate task preview
    original_task = result.get('original_task', '')
    task_preview = original_task[:200] + "..." if len(original_task) > 200 else original_task
    
    row = {
        'task_id': result.get('task_id', ''),
        'run_idx': result.get('run_idx', ''),
        'success': result.get('success', False),
        'final_answer': final_answer,
        'answer_length': len(final_answer),
        'latency_seconds': result.get('latency_seconds', 0),
        'iterations': result.get('iterations', 0),
        'reason': result.get('reason', ''),
        'total_flops_tflops': metrics.get('total_flops_tflops', 0),
        'router_flops_tflops': metrics.get('router_flops_tflops', 0),
        'agent_flops_tflops': metrics.get('agent_flops_tflops', 0),
        'handler_flops_tflops': metrics.get('handler_flops_tflops', 0),
        'total_tokens': metrics.get('total_tokens', 0),
        'router_calls': metrics.get('router_calls', 0),
        'agent_calls': metrics.get('agent_calls', 0),
        'handler_calls': metrics.get('handler_calls', 0),
        'original_task_preview': task_preview
    }
    
    # Append to CSV file
    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def run_evaluation(input_csv: str, output_csv: str, max_retry: int = 3, max_tasks: int = None, mode: str = "orchestrator"):
    """
    Run evaluation on tasks from input CSV
    
    Args:
        input_csv: Path to eval_inputs.csv
        output_csv: Path to output results CSV
        max_retry: Maximum retry count for orchestrator
        max_tasks: Maximum number of tasks to process (None = all)
        mode: Execution mode ("orchestrator" or "baseline")
    """
    print(f"\n{'='*80}")
    print("LLM-Agent-Orchestrator Evaluation")
    print(f"{'='*80}")
    print(f"Mode: {mode.upper()}")
    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Max Retry: {max_retry}")
    print(f"{'='*80}\n")
    
    # Load tasks
    print("Loading tasks from CSV...")
    tasks = load_tasks_from_csv(input_csv)
    
    if max_tasks:
        tasks = tasks[:max_tasks]
    
    print(f"Loaded {len(tasks)} tasks\n")
    
    # Initialize CSV file with headers
    initialize_csv(output_csv)
    
    # Initialize orchestrator
    orchestrator = LLMOrchestrator(max_retry=max_retry)
    
    # Track summary statistics
    successful_count = 0
    failed_count = 0
    total_latency = 0.0
    total_flops = 0.0
    
    # Process each task
    for i, task_info in enumerate(tasks):
        print(f"\n{'#'*80}")
        print(f"TASK {i+1}/{len(tasks)} (ID: {task_info['task_id']})")
        print(f"{'#'*80}")
        
        task = task_info['task']
        user_id = f"eval_task_{task_info['task_id']}"
        
        try:
            # Process task with specified mode
            result = orchestrator.process_task(task, user_id=user_id, mode=mode)
            
            # Add task metadata
            result['task_id'] = task_info['task_id']
            result['run_idx'] = task_info['run_idx']
            
            # Append to CSV immediately
            append_result_to_csv(result, output_csv)
            
            # Update summary stats
            if result.get('success'):
                successful_count += 1
            else:
                failed_count += 1
            total_latency += result.get('latency_seconds', 0)
            total_flops += result.get('metrics', {}).get('total_flops_tflops', 0)
            
            print(f"\n✓ Task {i+1} completed and saved to CSV:")
            print(f"  Success: {result.get('success')}")
            print(f"  Latency: {result.get('latency_seconds', 0):.2f}s")
            print(f"  Iterations: {result.get('iterations')}")
            print(f"  FLOPs: {result.get('metrics', {}).get('total_flops_tflops', 0):.4f} TFLOPs")
            print(f"  Answer Length: {len(result.get('final_answer', ''))} chars")
            
        except Exception as e:
            print(f"\n❌ Task {i+1} failed with error: {e}")
            import traceback
            traceback.print_exc()
            
            # Record failure and append to CSV immediately
            result = {
                'task_id': task_info['task_id'],
                'run_idx': task_info['run_idx'],
                'success': False,
                'final_answer': f"ERROR: {str(e)}",
                'latency_seconds': 0,
                'iterations': 0,
                'reason': 'exception',
                'metrics': {},
                'original_task': task
            }
            append_result_to_csv(result, output_csv)
            failed_count += 1
    
    # Print summary
    total_tasks = len(tasks)
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks: {total_tasks}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")
    
    if total_tasks > 0:
        avg_latency = total_latency / total_tasks
        avg_flops = total_flops / total_tasks
        print(f"Avg Latency: {avg_latency:.2f}s")
        print(f"Avg FLOPs: {avg_flops:.4f} TFLOPs")
    
    print(f"\n✓ All results saved to: {output_csv}")
    print(f"{'='*80}\n")


def main():
    """
    Main entry point
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run evaluation on LLM-Agent-Orchestrator')
    parser.add_argument('--input', '-i', 
                        default='examples/eval_inputs.csv',
                        help='Input CSV file path')
    parser.add_argument('--output', '-o',
                        default=None,
                        help='Output CSV file path (default: auto-generated with timestamp)')
    parser.add_argument('--max-retry', '-r',
                        type=int,
                        default=3,
                        help='Maximum retry count for orchestrator')
    parser.add_argument('--max-tasks', '-n',
                        type=int,
                        default=None,
                        help='Maximum number of tasks to process (default: all)')
    parser.add_argument('--mode', '-m',
                        choices=['orchestrator', 'baseline'],
                        default='orchestrator',
                        help='Execution mode: orchestrator (multi-agent) or baseline (single 8B model)')
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode_suffix = f"_{args.mode}" if args.mode == "baseline" else ""
        args.output = f'examples/eval_results{mode_suffix}_{timestamp}.csv'
    
    # Run evaluation
    run_evaluation(
        input_csv=args.input,
        output_csv=args.output,
        max_retry=args.max_retry,
        max_tasks=args.max_tasks,
        mode=args.mode
    )


if __name__ == "__main__":
    main()
