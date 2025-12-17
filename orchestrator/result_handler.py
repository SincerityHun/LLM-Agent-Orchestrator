"""
Results Handler - Synthesizes final answers from agent results using RAG-style approach
"""
import json
from typing import Tuple, Dict, Any
from utils.llm_loader import VLLMLoader


class ResultHandler:
    """
    Synthesizes final answers from agent results, treating them as retrieved reference materials.
    """
    
    def __init__(self, max_retry: int = 3):
        self.llm_loader = VLLMLoader()
        self.max_retry = max_retry
        self.last_usage = {}  # Track last call's token usage
        
        # Define JSON Schema for answer synthesis
        self.answer_schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Final answer to the original task using agent results as supporting knowledge. Must be non-empty and directly address the task."
                },
                "used_agents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of agent domains or subtask IDs whose results were used."
                }
            },
            "required": ["answer"]
        }
    
    def evaluate_results(
        self,
        original_task: str,
        merged_results: str,
        retry_count: int,
        dag: Dict = None,
        agent_results: Dict = None,
        immutable_facts: Dict = None
    ) -> Tuple[str, bool, str]:
        """
        Synthesize final answer from agent results using RAG-style approach.
        
        Args:
            original_task: Original user task
            merged_results: Merged output from all agents (for backward compatibility)
            retry_count: Current retry iteration count
            dag: DAG structure with nodes and edges (optional, provides structure)
            agent_results: Dict of agent execution results with subtask info (optional)
            immutable_facts: Ground truth facts extracted from original task (optional, unused)
            
        Returns:
            Tuple of (final_answer, should_stop, feedback)
        """
        # 1. Check retry limit
        if retry_count >= self.max_retry:
            return (merged_results, True, "Max retry limit reached")
        
        # 2. Build synthesis prompt with structured context
        synthesis_prompt = self._build_synthesis_prompt(
            original_task,
            merged_results,
            dag,
            agent_results
        )
        
        # Debug: Print structured context info
        import sys
        print(f"\n{'='*80}", flush=True)
        print(f"RESULT HANDLER - ANSWER SYNTHESIS", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Has DAG: {dag is not None}", flush=True)
        print(f"Has Agent Results: {agent_results is not None}", flush=True)
        if dag:
            print(f"Number of subtasks: {len(dag.get('nodes', []))}", flush=True)
        if agent_results:
            print(f"Number of agent results: {len(agent_results)}", flush=True)
        print(f"\nPrompt length: {len(synthesis_prompt)} characters", flush=True)
        print(f"\n{'='*80}", flush=True)
        print(f"FULL SYNTHESIS PROMPT:", flush=True)
        print(f"{'='*80}", flush=True)
        print(synthesis_prompt, flush=True)
        print(f"{'='*80}\n", flush=True)
        sys.stdout.flush()
        
        # 3. Call LLM with guided_json and track usage
        # vLLM will force the output to match self.answer_schema
        try:
            result = self.llm_loader.call_model(
                model_type="global-router",
                prompt=synthesis_prompt,
                max_tokens=2048,
                temperature=0.5,  # Slightly higher for better synthesis
                guided_json=self.answer_schema,
                return_usage=True  # Track token usage
            )
        except (ValueError, Exception) as e:
            # vLLM call failed (e.g., empty response, connection error)
            print(f"❌ ResultHandler vLLM call failed: {e}")
            feedback = f"ResultHandler LLM call failed: {str(e)}. Unable to evaluate results. Please retry with different task decomposition."
            return (merged_results, False, feedback)
        
        # Extract text and usage
        if isinstance(result, dict):
            json_response_str = result.get("text", result)
            self.last_usage = result.get("usage", {})
        else:
            json_response_str = result
            self.last_usage = {}
        
        # Check if response is empty
        if not json_response_str or not json_response_str.strip():
            print(f"❌ ResultHandler received empty response from vLLM")
            feedback = "ResultHandler received empty response. Unable to evaluate results. Please retry."
            return (merged_results, False, feedback)
        
        # 4. Parse JSON
        return self._parse_json_response(json_response_str, merged_results)
    
    def _build_synthesis_prompt(
        self,
        original_task: str,
        merged_results: str,
        dag: Dict = None,
        agent_results: Dict = None
    ) -> str:
        """
        Build RAG-style synthesis prompt treating agent results as retrieved reference materials.
        """
        # Build structured context if dag and agent_results are provided
        structured_context = ""
        if dag and agent_results:
            structured_context = self._build_structured_context(dag, agent_results)
        
        # Use structured context if available, otherwise fall back to merged_results
        if structured_context:
            results_section = f"""Structured Task Decomposition and Results:
{structured_context}"""
        else:
            results_section = f"""Retrieved Agent Results:
{merged_results}"""
        
        prompt = f"""You are an answer synthesis component.

Below are results generated by multiple specialized agents.
Treat them as retrieved reference materials.
They may be incomplete or partially irrelevant.

Your task:
- Answer the Original Task directly and clearly.
- Use the agent results only as supporting knowledge.
- Do NOT judge, score, or reject the results.
- If the information is insufficient to answer, return an empty answer.

Original Task:(
{original_task})

({results_section})

Response (JSON):"""
        return prompt
    
    def _build_structured_context(self, dag: Dict, agent_results: Dict) -> str:
        """
        Build structured context showing subtask-agent-result relationships and dependencies.
        
        Args:
            dag: DAG structure with nodes and edges
            agent_results: Dict of agent execution results
            
        Returns:
            Formatted string with structured context
        """
        lines = []
        nodes = dag.get("nodes", [])
        edges = dag.get("edges", [])
        
        # Build dependency mapping
        dependencies_map = {}
        for edge in edges:
            target = edge.get("to")
            source = edge.get("from")
            if target not in dependencies_map:
                dependencies_map[target] = []
            dependencies_map[target].append(source)
        
        # Process each subtask
        subtask_counter = 0
        for node in nodes:
            node_id = node.get("id")
            subtask = node.get("task", "")
            domain = node.get("domain", "unknown")
            
            # Get agent result
            result_data = agent_results.get(node_id, {})
            result = result_data.get("result", "[No result available]")
            # model_size = result_data.get("model_size", "unknown")
            
            # Skip subtasks with no actual results - don't pass them to Result Handler at all
            if result.startswith("[MOCK RESPONSE") or result == "[No result available]":
                continue
            
            subtask_counter += 1
            
            # Get dependencies
            deps = dependencies_map.get(node_id, [])
            dep_str = f"Depends on: {', '.join(deps)}" if deps else "No dependencies (independent subtask)"
            
            lines.append(f"\nSubtask {subtask_counter} (ID: {node_id}):")
            lines.append(f"  Domain: {domain.upper()}")
            # lines.append(f"  Model: {model_size}")
            lines.append(f"  Dependencies: {dep_str}")
            lines.append(f"  Subtask Description: {subtask}")
            lines.append(f"  Agent Response: {result}")
            lines.append("-" * 80)
        
        return "\n".join(lines)
    
    def _parse_json_response(self, json_str: str, original_merged_results: str) -> Tuple[str, bool, str]:
        """
        Parse the JSON string returned by vLLM (answer schema).
        """
        try:
            # Try to clean up the response first
            cleaned_json = json_str.strip()
            
            # If response is too long and truncated, try to fix it
            if not cleaned_json.endswith('}'):
                print(f"⚠️ Warning: JSON response appears truncated")
                # Try to find the last complete field and close the JSON
                last_quote = cleaned_json.rfind('"')
                if last_quote > 0:
                    cleaned_json = cleaned_json[:last_quote+1] + '}'
            
            data = json.loads(cleaned_json)
            
            answer = data.get("answer", "").strip()
            used_agents = data.get("used_agents", [])
            
            # Log synthesized answer preview
            preview = answer[:200] + "..." if len(answer) > 200 else answer
            print(f"\n[Synthesized Answer Preview]: {preview}\n")
            if used_agents:
                print(f"[Used Agents]: {', '.join(used_agents)}\n")
            
            # Check if answer is empty or insufficient
            if self._is_empty_answer(answer):
                feedback = "Empty or insufficient answer generated. Agent results may be incomplete. Retrying with refinement."
                return (original_merged_results, False, feedback)
            else:
                return (answer, True, "Answer synthesized successfully")
                
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Raw Response (first 500 chars): {json_str[:500]}...")
            print(f"Raw Response (last 200 chars): ...{json_str[-200:]}")
            
            print(f"⚠️ JSON parsing failed - triggering retry")
            feedback = "ResultHandler failed to generate valid JSON response. Retrying with clearer synthesis instructions."
            return (original_merged_results, False, feedback)

    def _is_empty_answer(self, answer: str) -> bool:
        """
        Check if answer is empty or contains only placeholder text.
        """
        if not answer:
            return True
        
        # Check for common placeholder patterns
        answer_lower = answer.lower().strip()
        placeholders = [
            "[no result available]",
            "no answer",
            "insufficient information",
            "unable to answer",
            "cannot answer"
        ]
        
        # Answer is too short (less than 20 characters)
        if len(answer) < 20:
            return True
        
        # Check if answer is just a placeholder
        for placeholder in placeholders:
            if answer_lower == placeholder or answer_lower.startswith(placeholder):
                return True
        
        return False

if __name__ == "__main__":
    # Mock Test
    print("Testing ResultHandler with answer synthesis...")
    
    handler = ResultHandler(max_retry=3)
    
    # Simulate a valid JSON response from vLLM
    mock_json_response = """
    {
        "answer": "The weather is currently sunny with a temperature of 25 degrees Celsius. This is ideal weather for outdoor activities.",
        "used_agents": ["commonsense", "task-1"]
    }
    """
    
    final_answer, should_stop, feedback = handler._parse_json_response(
        mock_json_response, "Raw Merged Results"
    )
    
    print(f"Should Stop: {should_stop}")
    print(f"Final Answer: {final_answer}")
    print(f"Feedback: {feedback}")
    
    assert should_stop is True
    assert "sunny" in final_answer
    assert len(final_answer) > 20
    
    # Test empty answer
    mock_empty_response = '{"answer": ""}'
    final_answer_empty, should_stop_empty, feedback_empty = handler._parse_json_response(
        mock_empty_response, "Raw Merged Results"
    )
    
    assert should_stop_empty is False
    assert "Empty or insufficient" in feedback_empty
    
    print("\nAll tests passed!")