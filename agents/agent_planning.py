"""
Planning Agent - Specialized in task analysis and strategy formulation
"""
from typing import Dict
from routers.agent_subrouter import AgentSubRouter


class PlanningAgent:
    """
    Agent specialized in planning and task decomposition
    """
    
    def __init__(self):
        self.subrouter = AgentSubRouter()
        self.role = "planning"
    
    def execute(self, task: str, context: Dict = None) -> Dict:
        """
        Execute planning task
        
        Args:
            task: Planning task description
            context: Optional context information
            
        Returns:
            Dictionary with role and result
        """
        # Execute planning task via subrouter
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
    print("Testing PlanningAgent...")
    
    agent = PlanningAgent()
    
    test_task = "Create a plan for diagnosing chest pain in emergency setting"
    result = agent.execute(test_task)
    
    assert result["role"] == "planning"
    assert "result" in result
    assert isinstance(result["result"], str)
    assert len(result["result"]) > 0
    
    print("PlanningAgent test passed")
