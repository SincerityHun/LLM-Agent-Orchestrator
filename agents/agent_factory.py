"""
Dynamic Agent Factory for creating agents from configuration
"""
import json
import os
from typing import Dict, Type, Any, Optional, TYPE_CHECKING
from utils.llm_loader import VLLMLoader

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


class AgentConfig:
    """Configuration data class for agent parameters"""
    
    def __init__(self, config_data: Dict[str, Any]):
        self.domain = config_data["domain"]
        self.class_name = config_data["class"]
        self.description = config_data["description"]
        self.temperature = config_data.get("temperature", 0.3)
        self.max_tokens = config_data.get("max_tokens", 512)
        self.template = config_data["template"]


class DomainConfig:
    """Configuration data class for domain parameters"""
    
    def __init__(self, domain_name: str, config_data: Dict[str, Any]):
        self.name = domain_name
        self.keywords = config_data["keywords"]
        self.agent_key = config_data["agent_key"]
        self.instruction = config_data["instruction"]


class DynamicAgent:
    """
    Dynamic agent that loads configuration from config.json
    """
    
    def __init__(self, agent_config: AgentConfig, llm_loader: Optional[VLLMLoader] = None):
        # Import BaseAgent here to avoid circular import
        from agents.base_agent import BaseAgent
        
        # Create base agent internally with domain
        self._base_agent = BaseAgent(domain=agent_config.domain, llm_loader=llm_loader)
        self.config = agent_config
        
    def __getattr__(self, name):
        """Delegate attribute access to base agent"""
        return getattr(self._base_agent, name)


class AgentFactory:
    """
    Factory for creating agents dynamically from configuration
    """
    
    _instance = None
    _config_loaded = False
    _agent_configs: Dict[str, AgentConfig] = {}
    _domain_configs: Dict[str, DomainConfig] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config_loaded:
            self._load_config()
    
    def _load_config(self):
        """Load configuration from agents/config.json"""
        config_path = os.path.join(
            os.path.dirname(__file__), "config.json"
        )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            
        # Load agent configurations (now domain-based)
        for domain, agent_data in config_data["agents"].items():
            self._agent_configs[domain] = AgentConfig(agent_data)
            
        # Load domain configurations  
        for domain_name, domain_data in config_data["domains"].items():
            self._domain_configs[domain_name] = DomainConfig(domain_name, domain_data)
            
        self._config_loaded = True
        print(f"Loaded {len(self._agent_configs)} agent configs and {len(self._domain_configs)} domain configs")
    

    
    def create_agent(self, domain: str, llm_loader: Optional[VLLMLoader] = None) -> 'BaseAgent':
        """Create an agent instance for the given domain"""
        if domain not in self._agent_configs:
            raise ValueError(f"Unknown agent domain: {domain}. Available domains: {list(self._agent_configs.keys())}")
            
        agent_config = self._agent_configs[domain]
        return DynamicAgent(agent_config, llm_loader)
    
    def get_available_domains(self) -> list:
        """Get list of available agent domains"""
        return list(self._agent_configs.keys())
    
    def get_domain_agent_mapping(self) -> Dict[str, str]:
        """Get domain to agent key mapping"""
        return {
            domain_name: domain_config.agent_key 
            for domain_name, domain_config in self._domain_configs.items()
        }
    
    def get_domain_keywords(self) -> Dict[str, list]:
        """Get domain keywords mapping"""
        return {
            domain_name: domain_config.keywords
            for domain_name, domain_config in self._domain_configs.items()
        }


# Global factory instance
agent_factory = AgentFactory()


if __name__ == "__main__":
    print("Testing AgentFactory...")
    
    factory = AgentFactory()
    
    # Test available domains
    domains = factory.get_available_domains()
    print(f"Available domains: {domains}")
    
    # Test agent creation
    medical_agent = factory.create_agent("medical")
    law_agent = factory.create_agent("law")
    math_agent = factory.create_agent("math")
    commonsense_agent = factory.create_agent("commonsense")
    
    print(f"Created agents: {medical_agent.domain}, {law_agent.domain}, {math_agent.domain}, {commonsense_agent.domain}")
    
    # Test agent execution
    test_task = "Diagnose patient with chest pain"
    result = medical_agent.execute(test_task)
    print(f"Medical agent result: {result['domain']} - {len(result['result'])} chars")
    
    print("AgentFactory test completed!")