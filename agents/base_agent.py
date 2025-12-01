"""Shared base class for domain-aware agents"""
from __future__ import annotations

from typing import Dict, Optional

from routers.agent_subrouter import AgentSubRouter
from utils.agent_prompts import get_agent_prompt
from utils.llm_loader import VLLMLoader


class BaseAgent:
    """Provides common orchestration logic for planning/execution/review agents"""

    def __init__(self, role: str, llm_loader: Optional[VLLMLoader] = None) -> None:
        self.role = role
        self.llm_loader = llm_loader or VLLMLoader()
        self.subrouter = AgentSubRouter(self.llm_loader)

    def execute(self, task: str, context: Optional[Dict] = None) -> Dict:
        """Execute agent workflow keeping legacy interface"""

        return self.process(task_content=task, context=context)

    def process(self, task_content: str, context: Optional[Dict] = None) -> Dict:
        """Detect domain, craft prompt, and call the appropriate LLM adapter"""

        context = context or {}
        context_text = self._format_context(context)

        domain = self.subrouter.detect_domain(task_content)
        endpoint_key, model_name = self.subrouter.select_model_adapter(domain)
        prompt = get_agent_prompt(self.role, domain, task_content, context_text)

        temperature = 0.5 if self.role == "execution" else 0.3

        response = self.llm_loader.generate(
            endpoint_key=endpoint_key,
            model_name=model_name,
            prompt=prompt,
            max_tokens=512,
            temperature=temperature,
            fallback_label=f"{self.role}:{domain}",
        )

        return {
            "role": self.role,
            "domain": domain,
            "result": response,
            "task": task_content,
        }

    def _format_context(self, context: Dict) -> str:
        if not context:
            return ""
        return "\n".join(f"{key}: {value}" for key, value in context.items())