"""Shared base class for domain-specific agents"""
from __future__ import annotations

from typing import Dict, Optional

from routers.agent_subrouter import AgentSubRouter
from utils.llm_loader import VLLMLoader


class BaseAgent:
    """Provides common orchestration logic for domain-specific agents"""

    def __init__(self, domain: str, llm_loader: Optional[VLLMLoader] = None) -> None:
        self.domain = domain
        self.llm_loader = llm_loader or VLLMLoader()
        self.subrouter = AgentSubRouter(self.llm_loader)

    def execute(self, task: str, context: Optional[Dict] = None) -> Dict:
        """Execute agent workflow with domain-specific processing"""
        return self.process(task_content=task, context=context)

    def process(self, task_content: str, context: Optional[Dict] = None) -> Dict:
        """Execute task using domain-specific router and models"""
        context = context or {}

        # Use subrouter to select model and execute
        response = self.subrouter.execute_subtask(
            domain=self.domain,
            task=task_content,
            context=context,
        )

        return {
            "domain": self.domain,
            "result": response,
            "task": task_content,
        }

    def _format_context(self, context: Dict) -> str:
        if not context:
            return ""
        return "\n".join(f"{key}: {value}" for key, value in context.items())