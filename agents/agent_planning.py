"""
Planning Agent - Specialized in task analysis and strategy formulation
"""
from typing import Dict, Optional
from agents.base_agent import BaseAgent
from agents.agent_factory import AgentFactory


class PlanningAgent(BaseAgent):
    """
    Agent specialized in planning and task decomposition
    """
    
    def __init__(self):
        factory = AgentFactory()
        dynamic_agent = factory.create_agent("planning")
        super().__init__(role="planning", llm_loader=dynamic_agent.llm_loader)
        self.config = dynamic_agent.config
    
    def execute(self, task: str, context: Optional[Dict] = None) -> Dict:
        """
        Execute planning task
        
        Args:
            task: Planning task description
            context: Optional context information
            
        Returns:
            Dictionary with role and result
        """
        return super().execute(task, context=context)


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
