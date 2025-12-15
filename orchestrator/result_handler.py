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
        self.last_usage = {}  # Track last call's token usage
        
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
        
        # 3. Call LLM with guided_json and track usage
        # vLLM will force the output to match self.evaluation_schema
        # Increased max_tokens to handle long responses
        result = self.llm_loader.call_model(
            model_type="global-router",
            prompt=evaluation_prompt,
            max_tokens=4096,  # Increased from 2048 to handle longer responses
            temperature=0.1, # Keep low for structural consistency
            guided_json=self.evaluation_schema,
            return_usage=True  # Track token usage
        )
        
        # Extract text and usage
        if isinstance(result, dict):
            json_response_str = result.get("text", result)
            self.last_usage = result.get("usage", {})
        else:
            json_response_str = result
            self.last_usage = {}
        
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
        prompt = f"""You are a Result Synthesizer. Your goal is to create a comprehensive answer to the Original Task using the Merged Results from domain experts.

Original Task:
{original_task}

Merged Results from Agents:
{merged_results}

Instructions:
1. In 'thoughts': Briefly analyze if the merged results contain sufficient information to answer the original task.
2. Set 'status':
   - "YES" if merged results are sufficient to answer the original task
   - "NO" if critical information is missing
3. In 'content':
   - If "YES": Write a DIRECT, COMPREHENSIVE ANSWER to the Original Task. Synthesize all relevant information from the merged results. Answer the question that was asked, not just evaluate the results.
   - If "NO": Specify exactly what information is missing to answer the original task.

IMPORTANT: When status is YES, the 'content' field must be the final answer to "{original_task}", NOT an evaluation of whether results are sufficient.

Response (JSON):
"""
        return prompt
    
    def _parse_json_response(self, json_str: str, original_merged_results: str) -> Tuple[str, bool, str]:
        """
        Parse the JSON string returned by vLLM.
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
            
            thoughts = data.get("thoughts", "")
            status = data.get("status", "NO").upper()
            content = data.get("content", "")
            
            # Logging thoughts for debugging (Optional)
            print(f"\n[Evaluator Content]: {content}...\n")  # Truncate for readability
            
            if status == "YES":
                # Success: Content is Final Answer
                final_answer = content
                if not final_answer:
                    return (original_merged_results, False, "Empty final answer despite YES status")

                return (final_answer, True, "Task completed successfully")
            else:
                # Failure: Content is Feedback
                return (original_merged_results, False, content)
                
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Raw Response (first 500 chars): {json_str[:500]}...")
            print(f"Raw Response (last 200 chars): ...{json_str[-200:]}")
            
            # Fallback: Use merged results as final answer and treat as success
            # This ensures we always have a final_answer in the CSV
            print(f"⚠️ Using merged_results as final_answer due to parsing error")
            return (original_merged_results, True, "Completed with parsing error - using merged results")

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