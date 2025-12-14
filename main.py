"""
Main orchestration entry point for LLM-Agent-Orchestrator system
"""
import json
import sys
from typing import Dict
from routers.global_router import GlobalRouter
from orchestrator.graph_builder import GraphBuilder
from orchestrator.result_handler import ResultHandler
from utils.merge_utils import merge_outputs, format_graph_summary


class LLMOrchestrator:
    """
    Main orchestrator coordinating the entire multi-agent system
    """
    
    def __init__(self, max_retry: int = 3):
        # Initialize components
        self.global_router = GlobalRouter()
        self.graph_builder = GraphBuilder()
        self.result_handler = ResultHandler(max_retry=max_retry)
        self.max_retry = max_retry
    
    def process_task(self, task: str, user_id: str = "default") -> Dict:
        """
        Process task through multi-agent orchestration
        
        Args:
            task: User task description
            user_id: User identifier
            
        Returns:
            Dictionary with final results and execution metadata
        """
        print(f"\n{'='*60}")
        print(f"Processing task for user: {user_id}")
        print(f"Task: {task}")
        print(f"{'='*60}\n")
        
        retry_count = 0
        feedback = None
        previous_results = None
        
        # Main refinement loop
        while retry_count < self.max_retry:
            print(f"\n--- Iteration {retry_count + 1} ---\n")
            
            # Step 1: Global Router generates DAG
            print("Step 1: Global Router - Task Decomposition")
            dag = self.global_router.decompose_task(
                task=task,
                feedback=feedback,
                previous_results=previous_results
            )
            
            print(format_graph_summary(dag))
            
            # Step 2: Build LangGraph from DAG
            print("Step 2: Building execution graph")
            graph = self.graph_builder.build_graph(dag)
            
            # Step 3: Execute gaphr
            print("Step 3: Executing agents")
            initial_state = {
                "original_task": task,
                "current_node": "",
                "results": {},
                "context": {"user_id": user_id}
            }
            
            final_state = self.graph_builder.execute_graph(graph, initial_state)
            
            # Step 4: Merge results
            print("\nStep 4: Merging agent outputs")
            print(f"{'='*80}")
            print("AGENT EXECUTION RESULTS")
            print(f"{'='*80}")
            
            agent_outputs = []
            for node_id, result_data in final_state.get("results", {}).items():
                domain = result_data.get("domain", "unknown")
                result = result_data.get("result", "")
                task_desc = result_data.get("task", "")
                
                print(f"\n{domain.upper()} AGENT ({node_id}):")
                print(f"   Task: {task_desc}")
                print(f"   Result: {result}")
                print(f"   Length: {len(result)} characters")
                
                agent_outputs.append({
                    "domain": domain,
                    "result": result
                })
            
            merged_results = merge_outputs(agent_outputs)
            
            print(f"\n{'-'*80}")
            print(f"MERGED RESULTS ({len(merged_results)} characters):")
            print(f"{'-'*80}")
            print(merged_results)
            print(f"{'='*80}\n")
            
            # Step 5: Results Handler evaluation
            print("Step 5: Results Handler - Evaluation")
            final_answer, should_stop, evaluation_feedback = self.result_handler.evaluate_results(
                original_task=task,
                merged_results=merged_results,
                retry_count=retry_count
            )
            
            print(f"Evaluation: {'COMPLETE' if should_stop else 'NEEDS REFINEMENT'}")
            print(f"Feedback: {evaluation_feedback}")
            
            # Check if we should stop
            if should_stop:
                print(f"\n{'='*60}")
                print("Task completed successfully")
                print(f"{'='*60}\n")
                
                return {
                    "success": True,
                    "final_answer": final_answer,
                    "iterations": retry_count + 1,
                    "user_id": user_id,
                    "original_task": task
                }
            
            # Prepare for next iteration
            retry_count += 1
            feedback = evaluation_feedback
            previous_results = merged_results
        
        # Max retry reached
        print(f"\n{'='*60}")
        print(f"Max retry limit ({self.max_retry}) reached")
        print(f"{'='*60}\n")
        
        return {
            "success": False,
            "final_answer": previous_results,
            "iterations": retry_count,
            "user_id": user_id,
            "original_task": task,
            "reason": "Max retry limit reached"
        }


def main():
    """
    Main entry point
    """
    # Load task from command line or use default
    if len(sys.argv) > 1:
        task_input = sys.argv[1]
        
        # Check if it's a JSON file
        if task_input.endswith('.json'):
            with open(task_input, 'r') as f:
                task_data = json.load(f)
            task = task_data.get("task", "")
            user_id = task_data.get("user_id", "default")
        else:
            task = task_input
            user_id = "cli_user"
    else:
        # Default test task
        task = "Explain whether a patient with sudden chest pain requires emergent CT angiography."
        user_id = "test_user"
    
    # Create orchestrator
    orchestrator = LLMOrchestrator(max_retry=3)
    
    # Process task
    result = orchestrator.process_task(task, user_id)
    
    # Output final result
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(json.dumps(result, indent=2))
    
    return result


if __name__ == "__main__":
    main()
