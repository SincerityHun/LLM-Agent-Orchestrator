"""
Review Agent - Specialized in quality assurance and validation
"""
from typing import Dict, Optional
from agents.base_agent import BaseAgent
from agents.agent_factory import AgentFactory


class ReviewAgent(BaseAgent):
    """
    Agent specialized in reviewing and validating results
    """
    
    def __init__(self):
        factory = AgentFactory()
        dynamic_agent = factory.create_agent("review")
        super().__init__(role="review", llm_loader=dynamic_agent.llm_loader)
        self.config = dynamic_agent.config
    
    def execute(self, task: str, context: Optional[Dict] = None) -> Dict:
        """
        Execute review task
        
        Args:
            task: Review task description
            context: Optional context including execution results
            
        Returns:
            Dictionary with role and result
        """
        return super().execute(task, context=context)


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
