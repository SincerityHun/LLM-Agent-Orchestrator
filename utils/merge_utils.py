"""
Utilities for merging subtask outputs
"""
from typing import List, Dict


def merge_outputs(outputs: List[Dict[str, str]]) -> str:
    """
    Merge multiple subtask outputs into a single coherent result
    
    Args:
        outputs: List of output dictionaries with 'domain' and 'result' keys
        
    Returns:
        Merged output string
    """
    if not outputs:
        return ""
    
    # Sort outputs by domain priority
    domain_priority = {"medical": 0, "law": 1, "math": 2, "commonsense": 3}
    sorted_outputs = sorted(
        outputs,
        key=lambda x: domain_priority.get(x.get("domain", ""), 999)
    )
    
    # Build merged output
    sections = []
    for output in sorted_outputs:
        domain = output.get("domain", "unknown").upper()
        result = output.get("result", "")
        
        if result:
            sections.append(f"[{domain}]\n{result}")
    
    return "\n\n".join(sections)


def format_graph_summary(graph: Dict) -> str:
    """
    Format dependency graph for logging and display
    
    Args:
        graph: Graph dictionary with nodes and edges
        
    Returns:
        Formatted string representation
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    summary_lines = ["Task Dependency Graph:"]
    
    # List nodes
    summary_lines.append("\nNodes:")
    for node in nodes:
        node_id = node.get("id", "unknown")
        domain = node.get("domain", "unknown")
        task_preview = node.get("task", "")[:60] + "..."
        summary_lines.append(f"  - {node_id} ({domain}): {task_preview}")
    
    # List edges
    summary_lines.append("\nDependencies:")
    if edges:
        for edge in edges:
            from_node = edge.get("from", "?")
            to_node = edge.get("to", "?")
            summary_lines.append(f"  - {from_node} -> {to_node}")
    else:
        summary_lines.append("  - No dependencies (parallel execution)")
    
    return "\n".join(summary_lines)


if __name__ == "__main__":
    # Unit test
    print("Testing merge_utils...")
    
    # Test merge_outputs
    test_outputs = [
        {"role": "execution", "result": "Analysis complete"},
        {"role": "planning", "result": "Plan created"},
        {"role": "review", "result": "All checks passed"}
    ]
    
    merged = merge_outputs(test_outputs)
    assert "[PLANNING]" in merged
    assert "[EXECUTION]" in merged
    assert "[REVIEW]" in merged
    
    # Verify order
    planning_idx = merged.index("[PLANNING]")
    execution_idx = merged.index("[EXECUTION]")
    review_idx = merged.index("[REVIEW]")
    assert planning_idx < execution_idx < review_idx
    
    # Test format_graph_summary
    test_graph = {
        "nodes": [
            {"id": "plan1", "role": "planning", "task": "Create plan"},
            {"id": "exec1", "role": "execution", "task": "Execute task"}
        ],
        "edges": [
            {"from": "plan1", "to": "exec1"}
        ]
    }
    
    summary = format_graph_summary(test_graph)
    assert "plan1" in summary
    assert "exec1" in summary
    assert "plan1 -> exec1" in summary
    
    print("merge_utils test passed")
