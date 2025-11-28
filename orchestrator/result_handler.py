"""
Results Handler - Evaluates merged results and determines next steps
"""
from typing import Dict, Tuple
from utils.llm_loader import VLLMLoader


class ResultHandler:
    """
    Evaluates results and decides whether to continue or finalize
    """
    
    def __init__(self, max_retry: int = 3):
        self.llm_loader = VLLMLoader()
        self.max_retry = max_retry
    
    def evaluate_results(
        self,
        original_task: str,
        merged_results: str,
        retry_count: int
    ) -> Tuple[str, bool, str]:
        """
        Evaluate merged results and determine next action
        
        Args:
            original_task: Original user task
            merged_results: Merged output from all agents
            retry_count: Current retry iteration count
            
        Returns:
            Tuple of (final_answer, should_stop, feedback)
        """
        # Check retry limit
        if retry_count >= self.max_retry:
            return (merged_results, True, "Max retry limit reached")
        
        # Build evaluation prompt
        evaluation_prompt = self._build_evaluation_prompt(
            original_task,
            merged_results
        )
        
        # Call LLM for evaluation
        evaluation = self.llm_loader.call_model(
            model_type="commonsense",
            prompt=evaluation_prompt,
            max_tokens=256,
            temperature=0.3
        )
        
        # Parse evaluation
        should_stop, feedback = self._parse_evaluation(evaluation)
        
        if should_stop:
            return (merged_results, True, "Task completed successfully")
        else:
            return (merged_results, False, feedback)
    
    def _build_evaluation_prompt(
        self,
        original_task: str,
        merged_results: str
    ) -> str:
        """
        Build prompt for result evaluation
        """
        prompt = f"""Evaluate if the following results adequately address the original task.

Original Task:
{original_task}

Results:
{merged_results}

Evaluation:
- Does the result fully address the task? (YES/NO)
- If NO, what specific improvements are needed?

Output format:
STATUS: [YES/NO]
FEEDBACK: [Your feedback here]
"""
        return prompt
    
    def _parse_evaluation(self, evaluation: str) -> Tuple[bool, str]:
        """
        Parse evaluation response
        
        Returns:
            Tuple of (should_stop, feedback)
        """
        evaluation_lower = evaluation.lower()
        
        # Check for completion indicators
        if "status: yes" in evaluation_lower or "status:yes" in evaluation_lower:
            return (True, "")
        
        # Extract feedback
        if "feedback:" in evaluation_lower:
            feedback_start = evaluation_lower.index("feedback:") + len("feedback:")
            feedback = evaluation[feedback_start:].strip()
            return (False, feedback)
        
        # Default: assume needs refinement
        return (False, evaluation)


if __name__ == "__main__":
    # Unit test
    print("Testing ResultHandler...")
    
    handler = ResultHandler(max_retry=3)
    
    # Test successful evaluation
    test_task = "Explain chest pain diagnosis"
    test_results = "[PLANNING] Create diagnostic plan\n[EXECUTION] CT scan recommended\n[REVIEW] All steps verified"
    
    final_answer, should_stop, feedback = handler.evaluate_results(
        original_task=test_task,
        merged_results=test_results,
        retry_count=0
    )
    
    assert isinstance(final_answer, str)
    assert isinstance(should_stop, bool)
    assert isinstance(feedback, str)
    
    # Test max retry limit
    _, should_stop_limit, _ = handler.evaluate_results(
        original_task=test_task,
        merged_results=test_results,
        retry_count=3
    )
    assert should_stop_limit == True
    
    print("ResultHandler test passed")
