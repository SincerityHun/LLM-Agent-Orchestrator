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
    return factory._domain_configs["general"].instruction


def _get_role_template(role: str) -> str:
    """Get role template from dynamic config"""
    factory = AgentFactory()
    agent_config = factory._agent_configs.get(role)
    if agent_config:
        return agent_config.template
    return factory._agent_configs["execution"].template


def get_agent_prompt(
    role: str,
    domain: str,
    task_content: str,
    context: str = "",
) -> str:
    """Compose an agent prompt by combining role guidance, domain focus, and task content"""

    # Use dynamic configuration only
    role_template = _get_role_template(role)
    domain_instruction = _get_domain_instruction(domain)

    parts = [role_template, "", f"Domain focus: {domain_instruction}"]

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

    planning_prompt = get_agent_prompt("planning", "medical", sample_task, sample_context)
    assert sample_task in planning_prompt
    assert "planning specialist" in planning_prompt

    execution_prompt = get_agent_prompt("execution", "law", sample_task)
    assert "legal analysis" in execution_prompt.lower()

    review_prompt = get_agent_prompt("review", "general", sample_task)
    assert "review specialist" in review_prompt.lower()

    print("Agent prompts test passed")
