"""
Research Agent - Specialized in information gathering and analysis
"""
from typing import Dict, Optional
from agents.base_agent import BaseAgent
from agents.agent_factory import AgentFactory


class ResearchAgent(BaseAgent):
    """Agent specialized in information gathering and research"""

    def __init__(self) -> None:
        factory = AgentFactory()
        dynamic_agent = factory.create_agent("research")
        super().__init__(role="research", llm_loader=dynamic_agent.llm_loader)
        self.config = dynamic_agent.config

    def execute(self, task: str, context: Optional[Dict] = None) -> Dict:
        """Execute research task"""



if __name__ == "__main__":
    # Unit test
    print("Testing ResearchAgent...")
    
    agent = ResearchAgent()
    
    test_task = "Research latest treatment protocols for cardiac emergencies"
    result = agent.execute(test_task)
    
    assert result["role"] == "research"
    assert "result" in result
    assert isinstance(result["result"], str)
    assert len(result["result"]) > 0
    
    print("ResearchAgent test passed")