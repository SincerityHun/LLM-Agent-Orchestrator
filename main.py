"""
Main orchestration entry point for LLM-Agent-Orchestrator system
"""
import json
import sys
import time
from typing import Dict
from pydantic import BaseModel, Field
from routers.global_router import GlobalRouter
from orchestrator.graph_builder import GraphBuilder
from orchestrator.result_handler import ResultHandler
from utils.merge_utils import merge_outputs, format_graph_summary
from utils.metrics import MetricsCollector
from utils.fact_extractor import FactExtractor
from utils.contradiction_checker import ContradictionChecker


class BaselineResponse(BaseModel):
    """Schema for baseline response with guided JSON generation"""
    
    answer: str = Field(
        ...,
        description="The complete answer to the task. Must be detailed and comprehensive.",
        min_length=50
    )


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
        self.metrics = MetricsCollector()
        
        # NEW: Fact extraction and contradiction checking
        self.fact_extractor = FactExtractor()
        self.contradiction_checker = ContradictionChecker()
        
        # Import here to avoid circular dependency
        from utils.llm_loader import VLLMLoader
        self.llm_loader = VLLMLoader()
    
    def run_baseline(self, task: str, user_id: str = "default") -> Dict:
        """
        Run baseline evaluation using single 8B model without orchestration
        
        Args:
            task: User task description
            user_id: User identifier
            
        Returns:
            Dictionary with final results and execution metadata
        """
        time.sleep(5)
        # Start timing
        start_time = time.time()
        
        # Reset metrics for this task
        self.metrics.reset()
        
        print(f"\n{'='*60}")
        print(f"BASELINE MODE - Single 8B Model")
        print(f"Processing task for user: {user_id}")
        print(f"Task: {task}")
        print(f"{'='*60}\n")
        
        # Create structured baseline prompt
        baseline_system = "You are a comprehensive AI assistant that provides detailed, accurate, and well-structured answers. Analyze the task carefully and provide a complete response."

        baseline_prompt = f"{baseline_system}\n\nTask: {task}\n\nProvide your detailed answer:"
        
        print("Calling 8B model directly with guided JSON...")
        
        # Use guided JSON to ensure structured output
        guided_schema = BaselineResponse.model_json_schema()
        
        result = self.llm_loader.call_model(
            model_type="global-router",
            prompt=baseline_prompt,
            max_tokens=4096,
            temperature=0.1,
            guided_json=guided_schema,
            return_usage=True
        )
        
        # Extract text and usage from guided JSON response
        if isinstance(result, dict):
            text_content = result.get("text", "")
            usage = result.get("usage", {})
            
            # Parse guided JSON response
            try:
                import json
                parsed = json.loads(text_content)
                final_answer = parsed.get("answer", text_content)
            except (json.JSONDecodeError, AttributeError):
                # Fallback if JSON parsing fails
                final_answer = text_content
        else:
            final_answer = str(result)
            usage = {}
        
        # Track baseline metrics (single 8B model call)
        input_tokens = usage.get("prompt_tokens", len(baseline_prompt.split()) * 1.3)  # Rough estimate
        output_tokens = usage.get("completion_tokens", len(final_answer.split()) * 1.3)
        
        # Add to metrics as a special "baseline" agent
        self.metrics.add_agent_call(
            agent_domain="baseline",
            model_size="8b",
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens)
        )
        
        # Calculate end-to-end latency
        end_time = time.time()
        latency_seconds = end_time - start_time
        
        # Get metrics summary
        metrics_summary = self.metrics.get_summary()
        
        print(f"\n{'='*60}")
        print("Baseline completed")
        print(f"Latency: {latency_seconds:.2f} seconds")
        print(f"Total FLOPs: {metrics_summary['total_flops_tflops']:.4f} TFLOPs")
        print(f"Tokens: {int(input_tokens)} input + {int(output_tokens)} output")
        print(f"Answer length: {len(final_answer)} characters")
        print(f"{'='*60}\n")
        
        return {
            "success": True,
            "final_answer": final_answer,
            "iterations": 1,  # Baseline always uses 1 iteration
            "reason": "baseline_completed",
            "latency_seconds": latency_seconds,
            "metrics": metrics_summary,
            "user_id": user_id,
            "original_task": task,
            "mode": "baseline"
        }
    
    def process_task(self, task: str, user_id: str = "default", mode: str = "orchestrator") -> Dict:
        """
        Process task through multi-agent orchestration or baseline
        
        Args:
            task: User task description
            user_id: User identifier
            mode: "orchestrator" for multi-agent or "baseline" for single 8B model
            
        Returns:
            Dictionary with final results and execution metadata
        """
        # Route to baseline if requested
        if mode == "baseline":
            return self.run_baseline(task, user_id)
        # Start timing
        start_time = time.time()
        
        # Reset metrics for this task
        self.metrics.reset()
        
        print(f"\n{'='*60}")
        print(f"Processing task for user: {user_id}")
        print(f"Task: {task}")
        print(f"{'='*60}\n")
        
        # STEP 0: Extract immutable facts (GROUND TRUTH)
        # print("Step 0: Extracting Immutable Facts")
        # immutable_facts = self.fact_extractor.extract_immutable_facts(task)
        # facts_prompt = self.fact_extractor.format_facts_for_prompt(immutable_facts)
        
        # print(f"{'='*80}")
        # print(facts_prompt)
        # print(f"{'='*80}\n")
        
        retry_count = 0
        feedback = None
        previous_results = None
        dag = None  # DAG will be fixed after first iteration
        
        # Main refinement loop
        while retry_count < self.max_retry:
            print(f"\n--- Iteration {retry_count + 1} ---\n")
            
            # Step 1: Global Router generates DAG (only on first iteration)
            if retry_count == 0:
                print("Step 1: Global Router - Task Decomposition")
                dag = self.global_router.decompose_task(
                    task=task,
                    feedback=feedback,
                    previous_results=previous_results
                )
            else:
                print("Step 1: Using fixed DAG from first iteration (no re-decomposition)")
            
            # Track router usage (estimate based on typical calls)
            # Global router makes 1-3 calls, estimate ~500 input + 300 output tokens per call
            # This is a rough estimate - ideally should be tracked in GlobalRouter
            router_est_tokens = 800 * (retry_count + 1)  # Estimate increases with retries
            self.metrics.add_router_call(
                input_tokens=router_est_tokens // 2,
                output_tokens=router_est_tokens // 2
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
            
            # Step 4: Merge results and collect metrics
            print("\nStep 4: Merging agent outputs")
            print(f"{'='*80}")
            print("AGENT EXECUTION RESULTS")
            print(f"{'='*80}")
            
            agent_outputs = []
            total_agents = len(final_state.get("results", {}))
            print(f"Total agents executed: {total_agents}")
            
            for node_id, result_data in final_state.get("results", {}).items():
                domain = result_data.get("domain", "unknown")
                result = result_data.get("result", "")
                task_desc = result_data.get("task", "")
                usage = result_data.get("usage")
                model_size = result_data.get("model_size", "unknown")
                
                print(f"\n{domain.upper()} AGENT ({node_id}):")
                print(f"   Task: {task_desc}")
                print(f"   Result: {result}")
                print(f"   Length: {len(result)} characters")
                print(f"   Model: {model_size}")
                
                # Track metrics
                if usage:
                    # Track subrouter call (each agent uses subrouter to select model)
                    self.metrics.add_sub_router_call(
                        input_tokens=200,  # Subrouter: ~200 input tokens
                        output_tokens=1    # Subrouter: ~50 output tokens
                    )
                    # Track agent execution
                    self.metrics.add_agent_call(
                        agent_domain=domain,
                        model_size=model_size,
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0)
                    )
                    print(f"   Tokens: {usage.get('prompt_tokens', 0)} input + {usage.get('completion_tokens', 0)} output")
                else:
                    print(f"   WARNING: No usage data for {domain} agent")
                
                agent_outputs.append({
                    "domain": domain,
                    "result": result
                })
            
            # Step 4.5: Contradiction Checking
            # print(f"\nStep 4.5: Contradiction Checking")
            # print(f"{'='*80}")
            
            # contradiction_result = self.contradiction_checker.check_contradictions(
            #     final_state.get("results", {}),
            #     immutable_facts
            # )
            
            # print(self.contradiction_checker.generate_report(contradiction_result))
            # print(f"{'='*80}\n")
            
            # Use only safe (validated) results
            # if contradiction_result["has_contradictions"]:
            #     print(f"⚠️  Using only {len(contradiction_result['safe_results'])} validated results\n")
            #     safe_agent_outputs = []
            #     for node_id, result_data in contradiction_result["safe_results"].items():
            #         safe_agent_outputs.append({
            #             "domain": result_data.get("domain", "unknown"),
            #             "result": result_data.get("result", "")
            #         })
            #     merged_results = merge_outputs(safe_agent_outputs)
            # else:
            #     merged_results = merge_outputs(agent_outputs)
            merged_results = merge_outputs(agent_outputs)
            print(f"\n{'-'*80}")
            print(f"MERGED RESULTS ({len(merged_results)} characters):")
            print(f"{'-'*80}")
            print(merged_results)
            print(f"{'='*80}\n")
            
            # Step 5: Results Handler evaluation with structured context and immutable facts
            print("Step 5: Results Handler - Evaluation (Judge Mode)")
            final_answer, should_stop, evaluation_feedback = self.result_handler.evaluate_results(
                original_task=task,
                merged_results=merged_results,
                retry_count=retry_count,
                dag=dag,
                agent_results=final_state.get("results", {}),
            )
            
            # Track handler usage
            handler_usage = getattr(self.result_handler, 'last_usage', {})
            if handler_usage:
                self.metrics.add_handler_call(
                    input_tokens=handler_usage.get("prompt_tokens", 0),
                    output_tokens=handler_usage.get("completion_tokens", 0)
                )
                print(f"Handler tokens: {handler_usage.get('prompt_tokens', 0)} input + {handler_usage.get('completion_tokens', 0)} output")
            
            print(f"Evaluation: {'COMPLETE' if should_stop else 'NEEDS REFINEMENT'}")
            print(f"Feedback: {evaluation_feedback}")
            
            # Check if we should stop
            if should_stop:
                # Calculate end-to-end latency
                end_time = time.time()
                latency_seconds = end_time - start_time
                
                # Get metrics summary
                metrics_summary = self.metrics.get_summary()
                
                print(f"\n{'='*60}")
                print("Task completed successfully")
                print(f"Latency: {latency_seconds:.2f} seconds")
                print(f"Iterations: {retry_count + 1}")
                print(f"Router calls: {metrics_summary['router_calls']}")
                print(f"Agent calls: {metrics_summary['agent_calls']}")
                print(f"Handler calls: {metrics_summary['handler_calls']}")
                print(f"Total FLOPs: {metrics_summary['total_flops_tflops']:.4f} TFLOPs")
                print(f"{'='*60}\n")
                
                return {
                    "success": True,
                    "final_answer": final_answer,
                    "iterations": retry_count + 1,
                    "reason": "completed",
                    "latency_seconds": latency_seconds,
                    "metrics": metrics_summary,
                    "user_id": user_id,
                    "original_task": task
                }
            
            # Prepare for next iteration
            retry_count += 1
            feedback = evaluation_feedback
            previous_results = merged_results
        
        # Max retry reached
        # Calculate end-to-end latency
        end_time = time.time()
        latency_seconds = end_time - start_time
        
        # Get metrics summary
        metrics_summary = self.metrics.get_summary()
        
        print(f"\n{'='*60}")
        print(f"Max retry limit ({self.max_retry}) reached")
        print(f"Latency: {latency_seconds:.2f} seconds")
        print(f"Total FLOPs: {metrics_summary['total_flops_tflops']:.4f} TFLOPs")
        print(f"{'='*60}\n")
        
        return {
            "success": False,
            "final_answer": previous_results,
            "iterations": retry_count,
            "reason": "max_retry_reached",
            "latency_seconds": latency_seconds,
            "metrics": metrics_summary,
            "user_id": user_id,
            "original_task": task
        }


def main():
    """
    Main entry point
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM-Agent-Orchestrator')
    parser.add_argument('task', nargs='?', default=None,
                        help='Task description or JSON file path')
    parser.add_argument('--mode', '-m', 
                        choices=['orchestrator', 'baseline'],
                        default='orchestrator',
                        help='Execution mode: orchestrator (multi-agent) or baseline (single 8B model)')
    parser.add_argument('--max-retry', '-r',
                        type=int,
                        default=3,
                        help='Maximum retry count for orchestrator mode')
    
    args = parser.parse_args()
    
    # Load task from argument or use default
    if args.task:
        task_input = args.task
        
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
    orchestrator = LLMOrchestrator(max_retry=args.max_retry)
    
    # Process task with specified mode
    result = orchestrator.process_task(task, user_id, mode=args.mode)
    
    # Output final result
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(json.dumps(result, indent=2))
    
    return result


if __name__ == "__main__":
    main()
