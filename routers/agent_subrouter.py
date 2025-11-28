"""
Agent SubRouter for model selection and execution
"""
from typing import Dict
from utils.agent_prompts import SUBROUTER_SELECTION_PROMPT
from utils.llm_loader import VLLMLoader


class AgentSubRouter:
    """
    SubRouter selects appropriate domain model and executes subtask
    """
    
    def __init__(self):
        self.llm_loader = VLLMLoader()
        
        # Model selection keywords
        self.domain_keywords = {
            "medical": ["medical", "patient", "diagnosis", "treatment", "symptom", "disease", "clinical"],
            "legal": ["legal", "law", "case", "regulation", "court", "statute", "attorney"],
            "math": ["math", "calculate", "equation", "number", "quantity", "formula", "compute"],
        }
    
    def select_model(self, task: str) -> str:
        """
        Select appropriate domain model based on task content
        
        Args:
            task: Task description
            
        Returns:
            Model type: 'medical', 'legal', 'math', or 'commonsense'
        """
        task_lower = task.lower()
        
        # Check for domain-specific keywords
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                return domain
        
        # Default to commonsense for general tasks
        return "commonsense"
    
    def execute_subtask(
        self,
        role: str,
        task: str,
        context: Dict = None
    ) -> str:
        """
        Execute subtask using selected domain model
        
        Args:
            role: Agent role (planning/execution/review)
            task: Task description
            context: Optional context from previous steps
            
        Returns:
            Execution result
        """
        # Select appropriate model
        model_type = self.select_model(task)
        
        # Build execution prompt
        prompt = self._build_execution_prompt(role, task, context)
        
        # Call selected model via vLLM
        result = self.llm_loader.call_model(
            model_type=model_type,
            prompt=prompt,
            max_tokens=512,
            temperature=0.5
        )
        
        return result
    
    def _build_execution_prompt(
        self,
        role: str,
        task: str,
        context: Dict = None
    ) -> str:
        """
        Build prompt for task execution
        """
        prompt_parts = []
        
        # Add role-specific instruction
        if role == "planning":
            prompt_parts.append("You are a planning specialist. Create a structured plan.")
        elif role == "execution":
            prompt_parts.append("You are an execution specialist. Perform the task accurately.")
        elif role == "review":
            prompt_parts.append("You are a review specialist. Verify and assess quality.")
        
        # Add context if available
        if context:
            prompt_parts.append(f"\nContext from previous steps:")
            for key, value in context.items():
                prompt_parts.append(f"- {key}: {value}")
        
        # Add task
        prompt_parts.append(f"\nTask: {task}")
        prompt_parts.append("\nProvide your response:")
        
        return "\n".join(prompt_parts)


if __name__ == "__main__":
    # Unit test
    print("Testing AgentSubRouter...")
    
    subrouter = AgentSubRouter()
    
    # Test model selection
    medical_task = "Diagnose patient with chest pain symptoms"
    assert subrouter.select_model(medical_task) == "medical"
    
    legal_task = "Analyze the legal implications of this contract"
    assert subrouter.select_model(legal_task) == "legal"
    
    math_task = "Calculate the derivative of x^2 + 3x"
    assert subrouter.select_model(math_task) == "math"
    
    general_task = "Explain why the sky is blue"
    assert subrouter.select_model(general_task) == "commonsense"
    
    # Test subtask execution
    result = subrouter.execute_subtask(
        role="planning",
        task=medical_task,
        context={"user": "test_user"}
    )
    assert isinstance(result, str)
    assert len(result) > 0
    
    print("AgentSubRouter test passed")
