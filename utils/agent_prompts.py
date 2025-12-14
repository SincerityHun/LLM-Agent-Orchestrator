"""Utilities for constructing agent prompts dynamically"""

from typing import Dict

# Import the dynamic agent factory
from agents.agent_factory import AgentFactory


def _get_domain_instruction(domain: str) -> str:
    """Get domain instruction from dynamic config"""
    factory = AgentFactory()
    domain_config = factory._domain_configs.get(domain)
    if domain_config:
        return domain_config.instruction
    return factory._domain_configs["commonsense"].instruction


def _get_domain_template(domain: str) -> str:
    """Get domain template from dynamic config"""
    factory = AgentFactory()
    agent_config = factory._agent_configs.get(domain)
    if agent_config:
        return agent_config.template
    return factory._agent_configs["commonsense"].template


def get_agent_prompt(
    domain: str,
    task_content: str,
    context: str = "",
) -> str:
    """Compose an agent prompt by combining domain template and task content"""

    # Use dynamic configuration only
    domain_template = _get_domain_template(domain)
    domain_instruction = _get_domain_instruction(domain)

    parts = [domain_template, "", f"Domain focus: {domain_instruction}"]

    parts.append("")
    parts.append("Task:")
    parts.append(task_content.strip())

    if context.strip():
        parts.append("")
        parts.append("Context:")
        parts.append(context.strip())

    parts.append("")
    parts.append("Provide a structured, concise, and domain-aware response.")

    return "\n".join(parts)


if __name__ == "__main__":
    print("Testing agent prompts...")

    sample_task = "Analyze chest pain symptoms"
    sample_context = "vitals: stable"

    medical_prompt = get_agent_prompt("medical", sample_task, sample_context)
    assert sample_task in medical_prompt
    assert "medical" in medical_prompt.lower()

    law_prompt = get_agent_prompt("law", "Analyze contract liability")
    assert "legal analysis" in law_prompt.lower()

    math_prompt = get_agent_prompt("math", "Calculate derivative of x^2")
    assert "mathematics" in math_prompt.lower() or "math" in math_prompt.lower()

    commonsense_prompt = get_agent_prompt("commonsense", "Explain why sky is blue")
    assert "commonsense" in commonsense_prompt.lower() or "reasoning" in commonsense_prompt.lower()

    print("Agent prompts test passed")
