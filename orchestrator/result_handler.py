"""
Results Handler - Evaluates merged results and determines next steps using JSON schema
"""
import json
from typing import Tuple, Dict, Any
from utils.llm_loader import VLLMLoader


class ResultHandler:
    """
    Evaluates results and decides whether to continue or finalize based on the original task.
    """
    
    def __init__(self, max_retry: int = 3):
        self.llm_loader = VLLMLoader()
        self.max_retry = max_retry
        
        # Define JSON Schema for vLLM guided generation
        self.evaluation_schema = {
            "type": "object",
            "properties": {
                "thoughts": {
                    "type": "string",
                    "description": "Step-by-step analysis of the original task requirements and the provided merged results."
                },
                "status": {
                    "type": "string",
                    "enum": ["YES", "NO"],
                    "description": "YES if results are sufficient, NO if information is missing."
                },
                "content": {
                    "type": "string",
                    "description": "If status is YES, the final synthesized answer. If NO, feedback on what specific information is missing."
                }
            },
            "required": ["thoughts", "status", "content"]
        }
    
    def evaluate_results(
        self,
        original_task: str,
        merged_results: str,
        retry_count: int
    ) -> Tuple[str, bool, str]:
        """
        Evaluate merged results using vLLM's guided_json.
        
        Args:
            original_task: Original user task
            merged_results: Merged output from all agents
            retry_count: Current retry iteration count
            
        Returns:
            Tuple of (final_answer, should_stop, feedback)
        """
        # 1. Check retry limit
        if retry_count >= self.max_retry:
            return (merged_results, True, "Max retry limit reached")
        
        # 2. Build evaluation prompt (Simplified for JSON)
        evaluation_prompt = self._build_evaluation_prompt(
            original_task,
            merged_results
        )
        
        # 3. Call LLM with guided_json
        # vLLM will force the output to match self.evaluation_schema
        json_response_str = self.llm_loader.call_model(
            model_type="global-router",
            prompt=evaluation_prompt,
            max_tokens=2048,
            temperature=0.1, # Keep low for structural consistency
            guided_json=self.evaluation_schema
        )
        
        # 4. Parse JSON
        return self._parse_json_response(json_response_str, merged_results)
    
    def _build_evaluation_prompt(
        self,
        original_task: str,
        merged_results: str
    ) -> str:
        """
        Build prompt focused on logic, relying on schema for structure.
        """
        prompt = f"""You are a Result Evaluator. Your goal is to evaluate if the merged results are sufficient to answer the Original Task.

Original Task:
{original_task}

Merged Results from Agents:
{merged_results}

Instructions:
1. Analyze the 'Original Task' and 'Merged Results' in the 'thoughts' field.
2. Determine if the results are sufficient. Set 'status' to "YES" or "NO".
3. Provide the output in the 'content' field:
   - If "YES": Synthesize a comprehensive final answer.
   - If "NO": Explain exactly what information is missing.

Response (JSON):
"""
        return prompt
    
    def _parse_json_response(self, json_str: str, original_merged_results: str) -> Tuple[str, bool, str]:
        """
        Parse the JSON string returned by vLLM.
        """
        try:
            data = json.loads(json_str)
            
            thoughts = data.get("thoughts", "")
            status = data.get("status", "NO").upper()
            content = data.get("content", "")
            
            # Logging thoughts for debugging (Optional)
            print(f"\n[Evaluator Thoughts]: {thoughts}\n")
            
            if status == "YES":
                # Success: Content is Final Answer
                return (content, True, "Task completed successfully")
            else:
                # Failure: Content is Feedback
                return (original_merged_results, False, content)
                
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Raw Response: {json_str}")
            # Fallback: Treat as a retry with raw response as feedback if possible
            return (original_merged_results, False, "Error parsing evaluator response. Retrying.")

if __name__ == "__main__":
    # Mock Test
    print("Testing ResultHandler with guided_json Logic...")
    
    handler = ResultHandler(max_retry=3)
    
    # Simulate a valid JSON response from vLLM
    mock_json_response_yes = """
    {
        "thoughts": "The user asked for the weather. The results show it is sunny and 25 degrees. This is sufficient.",
        "status": "YES",
        "content": "The weather is currently sunny with a temperature of 25 degrees."
    }
    """
    
    final_answer, should_stop, feedback = handler._parse_json_response(
        mock_json_response_yes, "Raw Merged Results"
    )
    
    print(f"Status: {should_stop}")
    print(f"Final Answer: {final_answer}")
    
    assert should_stop is True
    assert "sunny" in final_answer