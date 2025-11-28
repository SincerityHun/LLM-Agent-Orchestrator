"""
Review Agent - Specialized in quality assurance and validation
"""
from typing import Dict
from routers.agent_subrouter import AgentSubRouter


class ReviewAgent:
    """
    Agent specialized in reviewing and validating results
    """
    
    def __init__(self):
        self.subrouter = AgentSubRouter()
        self.role = "review"
    
    def execute(self, task: str, context: Dict = None) -> Dict:
        """
        Execute review task
        
        Args:
            task: Review task description
            context: Optional context including execution results
            
        Returns:
            Dictionary with role and result
        """
        # Execute review task via subrouter
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
    print("Testing ReviewAgent...")
    
    agent = ReviewAgent()
    
    test_task = "Review the diagnostic recommendations"
    test_context = {
        "planning": "Emergency protocol followed",
        "execution": "CT angiography recommended"
    }
    
    result = agent.execute(test_task, context=test_context)
    
    assert result["role"] == "review"
    assert "result" in result
    assert isinstance(result["result"], str)
    assert len(result["result"]) > 0
    
    print("ReviewAgent test passed")
