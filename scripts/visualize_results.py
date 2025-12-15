"""
Visualization script for evaluation results
Creates FLOPs comparison and performance analysis charts
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


def load_results(csv_path: str) -> pd.DataFrame:
    """Load evaluation results from CSV"""
    df = pd.DataFrame(pd.read_csv(csv_path))
    return df


def plot_flops_comparison(df: pd.DataFrame, output_dir: str):
    """
    Create FLOPs comparison chart similar to the provided example
    Compares the orchestrator system vs a baseline single large model
    """
    # Assumptions for baseline comparison
    # Baseline: Single 23B model (similar to GPT-3.5 scale)
    # Our system: Router (8B) + Agents (1B/8B)
    
    # Estimate baseline FLOPs (assuming similar token counts)
    baseline_model_size = 23.0  # 23B parameters
    
    # Calculate baseline FLOPs for each task
    baseline_flops = []
    for _, row in df.iterrows():
        total_tokens = row['total_tokens']
        # FLOPs = 2 * Parameters * Tokens
        flops = 2 * (baseline_model_size * 1e9) * total_tokens / 1e12  # TFLOPs
        baseline_flops.append(flops)
    
    df['baseline_flops_tflops'] = baseline_flops
    
    # Create figure
    plt.figure(figsize=(14, 7))
    
    x = np.arange(len(df))
    width = 0.35
    
    # Plot bars
    plt.bar(x - width/2, df['total_flops_tflops'], width, 
            label='LLM-Agent-Orchestrator (Router + Agents)', 
            color='#1f77b4', alpha=0.9)
    plt.bar(x + width/2, df['baseline_flops_tflops'], width, 
            label='Baseline (Single 23B Model)', 
            color='#d62728', alpha=0.9)
    
    # Styling
    plt.ylabel('Inference Cost (TFLOPs)', fontsize=12, fontweight='bold')
    plt.xlabel('Task ID', fontsize=12, fontweight='bold')
    plt.title('Inference Cost Comparison: Orchestrator vs Baseline', 
              fontsize=16, fontweight='bold')
    plt.xticks(x, [f"T{i}" for i in df['task_id']], rotation=45, ha='right')
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add value annotations and efficiency ratios
    for i in range(len(df)):
        our_val = df['total_flops_tflops'].iloc[i]
        base_val = df['baseline_flops_tflops'].iloc[i]
        
        # Our system value
        plt.text(i - width/2, our_val + max(df['total_flops_tflops'])*0.02, 
                f"{our_val:.1f}", ha='center', va='bottom', fontsize=8, color='blue')
        
        # Baseline value
        plt.text(i + width/2, base_val + max(df['baseline_flops_tflops'])*0.02, 
                f"{base_val:.1f}", ha='center', va='bottom', fontsize=8, color='red')
        
        # Efficiency multiplier
        if our_val > 0:
            efficiency = base_val / our_val
            plt.text(i, max(base_val, our_val) + max(df['baseline_flops_tflops'])*0.08, 
                    f"×{efficiency:.1f}", ha='center', fontweight='bold', 
                    color='green', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'flops_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved FLOPs comparison chart to {output_path}")
    plt.close()
    
    # Print efficiency report
    print("\n" + "="*60)
    print("EFFICIENCY REPORT")
    print("="*60)
    
    total_our_flops = df['total_flops_tflops'].sum()
    total_baseline_flops = df['baseline_flops_tflops'].sum()
    
    if total_our_flops > 0:
        overall_efficiency = total_baseline_flops / total_our_flops
        print(f"Overall Efficiency: {overall_efficiency:.2f}× more efficient than 23B baseline")
        print(f"Total FLOPs Saved: {total_baseline_flops - total_our_flops:.2f} TFLOPs")
        print(f"Cost Reduction: {(1 - 1/overall_efficiency) * 100:.1f}%")
    
    print("="*60 + "\n")


def plot_latency_distribution(df: pd.DataFrame, output_dir: str):
    """Plot latency distribution"""
    plt.figure(figsize=(10, 6))
    
    plt.hist(df['latency_seconds'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(df['latency_seconds'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {df["latency_seconds"].mean():.2f}s')
    plt.axvline(df['latency_seconds'].median(), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {df["latency_seconds"].median():.2f}s')
    
    plt.xlabel('Latency (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('End-to-End Latency Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'latency_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved latency distribution to {output_path}")
    plt.close()


def plot_iterations_vs_flops(df: pd.DataFrame, output_dir: str):
    """Plot relationship between iterations and FLOPs"""
    plt.figure(figsize=(10, 6))
    
    colors = ['green' if s else 'red' for s in df['success']]
    plt.scatter(df['iterations'], df['total_flops_tflops'], 
                c=colors, s=100, alpha=0.6, edgecolors='black')
    
    # Add trend line for successful tasks
    successful = df[df['success'] == True]
    if len(successful) > 1:
        z = np.polyfit(successful['iterations'], successful['total_flops_tflops'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(successful['iterations'].min(), 
                             successful['iterations'].max(), 100)
        plt.plot(x_trend, p(x_trend), "b--", alpha=0.5, linewidth=2)
    
    plt.xlabel('Number of Iterations', fontsize=12, fontweight='bold')
    plt.ylabel('Total FLOPs (TFLOPs)', fontsize=12, fontweight='bold')
    plt.title('Iterations vs Computational Cost', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='Success'),
        Patch(facecolor='red', alpha=0.6, label='Failed')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'iterations_vs_flops.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved iterations vs FLOPs chart to {output_path}")
    plt.close()


def plot_agent_usage(df: pd.DataFrame, output_dir: str):
    """Plot agent call distribution"""
    plt.figure(figsize=(10, 6))
    
    # Average calls per task
    avg_router = df['router_calls'].mean()
    avg_agent = df['agent_calls'].mean()
    avg_handler = df['handler_calls'].mean()
    
    categories = ['Router', 'Agents', 'Result Handler']
    values = [avg_router, avg_agent, avg_handler]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    plt.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for i, v in enumerate(values):
        plt.text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    plt.ylabel('Average Calls per Task', fontsize=12, fontweight='bold')
    plt.title('Component Usage Distribution', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'agent_usage.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved agent usage chart to {output_path}")
    plt.close()


def generate_summary_stats(df: pd.DataFrame, output_dir: str):
    """Generate summary statistics report"""
    output_path = Path(output_dir) / 'summary_stats.txt'
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EVALUATION SUMMARY STATISTICS\n")
        f.write("="*70 + "\n\n")
        
        # Success rate
        success_rate = (df['success'].sum() / len(df)) * 100
        f.write(f"Tasks Processed: {len(df)}\n")
        f.write(f"Success Rate: {success_rate:.1f}%\n")
        f.write(f"Failed Tasks: {len(df) - df['success'].sum()}\n\n")
        
        # Latency statistics
        f.write("LATENCY STATISTICS:\n")
        f.write(f"  Mean: {df['latency_seconds'].mean():.2f}s\n")
        f.write(f"  Median: {df['latency_seconds'].median():.2f}s\n")
        f.write(f"  Min: {df['latency_seconds'].min():.2f}s\n")
        f.write(f"  Max: {df['latency_seconds'].max():.2f}s\n")
        f.write(f"  Std Dev: {df['latency_seconds'].std():.2f}s\n\n")
        
        # FLOPs statistics
        f.write("FLOPS STATISTICS (TFLOPs):\n")
        f.write(f"  Mean: {df['total_flops_tflops'].mean():.4f}\n")
        f.write(f"  Median: {df['total_flops_tflops'].median():.4f}\n")
        f.write(f"  Min: {df['total_flops_tflops'].min():.4f}\n")
        f.write(f"  Max: {df['total_flops_tflops'].max():.4f}\n")
        f.write(f"  Total: {df['total_flops_tflops'].sum():.4f}\n\n")
        
        # Iteration statistics
        f.write("ITERATION STATISTICS:\n")
        f.write(f"  Mean: {df['iterations'].mean():.2f}\n")
        f.write(f"  Median: {df['iterations'].median():.0f}\n")
        f.write(f"  Mode: {df['iterations'].mode().values[0] if not df['iterations'].mode().empty else 'N/A'}\n\n")
        
        # Component usage
        f.write("COMPONENT USAGE (Average per task):\n")
        f.write(f"  Router Calls: {df['router_calls'].mean():.2f}\n")
        f.write(f"  Agent Calls: {df['agent_calls'].mean():.2f}\n")
        f.write(f"  Handler Calls: {df['handler_calls'].mean():.2f}\n\n")
        
        # Token usage
        f.write("TOKEN USAGE:\n")
        f.write(f"  Total Tokens: {df['total_tokens'].sum()}\n")
        f.write(f"  Avg per Task: {df['total_tokens'].mean():.0f}\n\n")
        
        f.write("="*70 + "\n")
    
    print(f"✓ Saved summary statistics to {output_path}")
    
    # Also print to console
    with open(output_path, 'r') as f:
        print("\n" + f.read())


def main():
    parser = argparse.ArgumentParser(description='Visualize evaluation results')
    parser.add_argument('--input', '-i', required=True,
                        help='Input CSV file with evaluation results')
    parser.add_argument('--output-dir', '-o', default='examples/visualizations',
                        help='Output directory for charts')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS VISUALIZATION")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Output Directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading results...")
    df = load_results(args.input)
    print(f"✓ Loaded {len(df)} task results\n")
    
    # Generate visualizations
    print("Generating visualizations...\n")
    
    plot_flops_comparison(df, args.output_dir)
    plot_latency_distribution(df, args.output_dir)
    plot_iterations_vs_flops(df, args.output_dir)
    plot_agent_usage(df, args.output_dir)
    generate_summary_stats(df, args.output_dir)
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"All charts saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
