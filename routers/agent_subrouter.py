"""Agent SubRouter for model selection and prompt orchestration"""

from typing import Dict, List, Optional, Tuple

from utils.agent_prompts import get_agent_prompt
from utils.llm_loader import VLLMLoader
from agents.agent_factory import AgentFactory


class AgentSubRouter:
    """SubRouter selects domain models and prepares prompts"""

    def __init__(self, llm_loader: Optional[VLLMLoader] = None) -> None:
        self.llm_loader = llm_loader or VLLMLoader()
        
        # Initialize with dynamic configuration
        factory = AgentFactory()
        self.domain_keywords = factory.get_domain_keywords()
        self.domain_to_model = factory.get_domain_model_mapping()

    def detect_domain(self, content: str) -> str:
        """Infer domain from task content using keyword heuristics"""
        # Use factory's domain detection
        domain_config = AgentFactory.detect_domain(content)
        return domain_config.name

    def select_model_adapter(self, domain: str) -> Tuple[str, str]:
        """Map domain to (endpoint_key, model_name) pair"""

        model_key = self.domain_to_model.get(domain, self.domain_to_model["general"])
        config = self.llm_loader.model_configs.get(model_key)
        if not config:
            raise ValueError(f"Model configuration missing for domain '{domain}'")

        return config["endpoint"], config["model"]

    def execute_subtask(
        self,
        role: str,
        task: str,
        context: Optional[Dict] = None,
    ) -> str:
        """Retained for backward compatibility with legacy agents"""

        context = context or {}
        domain = self.detect_domain(task)
        endpoint_key, model_name = self.select_model_adapter(domain)
        prompt = get_agent_prompt(role, domain, task, self._format_context(context))

        temperature = 0.5 if role == "execution" else 0.3

        return self.llm_loader.generate(
            endpoint_key=endpoint_key,
            model_name=model_name,
            prompt=prompt,
            max_tokens=512,
            temperature=temperature,
            fallback_label=f"{role}:{domain}",
        )

    def _format_context(self, context: Dict) -> str:
        if not context:
            return ""
        return "\n".join(f"{key}: {value}" for key, value in context.items())


if __name__ == "__main__":
    print("Testing AgentSubRouter...")

    subrouter = AgentSubRouter()

    medical_task = "Diagnose patient with chest pain symptoms"
    assert subrouter.detect_domain(medical_task) == "medical"

    legal_task = "Analyze the legal implications of this contract"
    assert subrouter.detect_domain(legal_task) == "law"

    math_task = "Calculate the derivative of x^2 + 3x"
    assert subrouter.detect_domain(math_task) == "math"

    general_task = "Explain why the sky is blue"
    assert subrouter.detect_domain(general_task) == "general"

    endpoint_key, model_name = subrouter.select_model_adapter("medical")
    assert endpoint_key == "llama"
    assert model_name

    result = subrouter.execute_subtask(
        role="planning",
        task=medical_task,
        context={"user": "test_user"},
    )
    assert isinstance(result, str)
    assert len(result) > 0

    print("AgentSubRouter test passed")
