"""
Execution Agent - Specialized in performing domain-specific tasks
"""
from typing import Dict
from routers.agent_subrouter import AgentSubRouter


class ExecutionAgent:
    """
    Agent specialized in executing domain-specific tasks
    """
    
    def __init__(self):
        self.subrouter = AgentSubRouter()
        self.role = "execution"
    
    def execute(self, task: str, context: Dict = None) -> Dict:
        """
        Execute domain-specific task
        
        Args:
            task: Execution task description
            context: Optional context from planning
            
        Returns:
            Dictionary with role and result
        """
        # Execute task via subrouter with domain model selection
        result = self.subrouter.execute_subtask(
            role=self.role,
            task=task,
            context=context
        )
        
        return {
            "role": self.role,
            "result": result,
            "task": task
        }


if __name__ == "__main__":
    # Unit test
    print("Testing ExecutionAgent...")
    
    agent = ExecutionAgent()
    
    test_task = "Perform differential diagnosis for acute chest pain"
    test_context = {"planning": "Follow emergency protocol"}
    
    result = agent.execute(test_task, context=test_context)
    
    assert result["role"] == "execution"
    assert "result" in result
    assert isinstance(result["result"], str)
    assert len(result["result"]) > 0
    
    print("ExecutionAgent test passed")
