"""
Agents package with domain-specific agent factory support
"""

from agents.base_agent import BaseAgent
from agents.agent_factory import AgentFactory, agent_factory


def create_agent(domain: str, **kwargs):
    """
    Factory function to create domain-specific agents dynamically
    
    Args:
        domain: Agent domain (commonsense, medical, law, math)
        **kwargs: Additional arguments passed to agent constructor
        
    Returns:
        Agent instance
    """
    return agent_factory.create_agent(domain, **kwargs)


def get_available_domains():
    """Get list of available domains"""
    return agent_factory.get_available_domains()


__all__ = [
    "BaseAgent",
    "AgentFactory",
    "agent_factory",
    "create_agent",
    "get_available_domains",
]
