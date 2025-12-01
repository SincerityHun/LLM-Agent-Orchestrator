"""
Agents package with dynamic agent factory support
"""

from agents.base_agent import BaseAgent
from agents.agent_planning import PlanningAgent
from agents.agent_execution import ExecutionAgent
from agents.agent_review import ReviewAgent
from agents.agent_research import ResearchAgent
from agents.agent_validation import ValidationAgent

# Import dynamic factory
from agents.agent_factory import AgentFactory, agent_factory
DYNAMIC_AGENTS_AVAILABLE = True


# Agent class registry for dynamic loading
AGENT_CLASSES = {
    "PlanningAgent": PlanningAgent,
    "ExecutionAgent": ExecutionAgent,
    "ReviewAgent": ReviewAgent,
    "ResearchAgent": ResearchAgent,
    "ValidationAgent": ValidationAgent,
}


def create_agent(role: str, **kwargs):
    """
    Factory function to create agents dynamically
    
    Args:
        role: Agent role (planning, execution, review, research, validation)
        **kwargs: Additional arguments passed to agent constructor
        
    Returns:
        Agent instance
    """
    return agent_factory.create_agent(role, **kwargs)


def get_available_roles():
    """Get list of available agent roles"""
    return agent_factory.get_available_roles()


def get_available_domains():
    """Get list of available domains"""
    return agent_factory.get_available_domains()


__all__ = [
    "BaseAgent",
    "PlanningAgent", 
    "ExecutionAgent",
    "ReviewAgent",
    "ResearchAgent",
    "ValidationAgent",
    "AgentFactory",
    "agent_factory",
    "create_agent",
    "get_available_roles",
    "get_available_domains",
    "AGENT_CLASSES",
    "DYNAMIC_AGENTS_AVAILABLE",
]
