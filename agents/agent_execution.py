"""Execution Agent - Specialized in performing domain-specific tasks"""
from __future__ import annotations

from typing import Dict, Optional

from agents.base_agent import BaseAgent
from agents.agent_factory import AgentFactory


class ExecutionAgent(BaseAgent):
    """Agent specialized in executing domain-specific tasks"""

    def __init__(self) -> None:
        factory = AgentFactory()
        dynamic_agent = factory.create_agent("execution")
        super().__init__(role="execution", llm_loader=dynamic_agent.llm_loader)
        self.config = dynamic_agent.config

    def execute(self, task: str, context: Optional[Dict] = None) -> Dict:
        """Execute the execution task"""
        return super().execute(task, context=context)


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
