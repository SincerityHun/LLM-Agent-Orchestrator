"""
Validation Agent - Specialized in data validation and verification
"""
from typing import Dict, Optional
from agents.base_agent import BaseAgent
from agents.agent_factory import AgentFactory


class ValidationAgent(BaseAgent):
    """Agent specialized in data validation and verification"""

    def __init__(self) -> None:
        factory = AgentFactory()
        dynamic_agent = factory.create_agent("validation")
        super().__init__(role="validation", llm_loader=dynamic_agent.llm_loader)
        self.config = dynamic_agent.config

    def execute(self, task: str, context: Optional[Dict] = None) -> Dict:
        """Execute validation task"""
        return super().execute(task, context=context)


if __name__ == "__main__":
    # Unit test
    print("Testing ValidationAgent...")
    
    agent = ValidationAgent()
    
    test_task = "Validate patient data completeness and accuracy"
    test_context = {"data": "Patient records with vital signs"}
    
    result = agent.execute(test_task, context=test_context)
    
    assert result["role"] == "validation"
    assert "result" in result
    assert isinstance(result["result"], str)
    assert len(result["result"]) > 0
    
    print("ValidationAgent test passed")