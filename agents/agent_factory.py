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
        self.role = config_data["role"]
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
        self.model_key = config_data["model_key"]
        self.instruction = config_data["instruction"]


class DynamicAgent:
    """
    Dynamic agent that loads configuration from config.json
    """
    
    def __init__(self, agent_config: AgentConfig, llm_loader: Optional[VLLMLoader] = None):
        # Import BaseAgent here to avoid circular import
        from agents.base_agent import BaseAgent
        
        # Create base agent internally
        self._base_agent = BaseAgent(role=agent_config.role, llm_loader=llm_loader)
        self.config = agent_config
        
    def __getattr__(self, name):
        """Delegate attribute access to base agent"""
        return getattr(self._base_agent, name)
        
    def process(self, task_content: str, context: Optional[Dict] = None) -> Dict:
        """Override process to use dynamic configuration"""
        context = context or {}
        context_text = self._format_context(context)

        # Use factory's domain detection
        domain_info = AgentFactory.detect_domain(task_content)
        endpoint_key, model_name = self.subrouter.select_model_adapter(domain_info.name)
        
        # Use dynamic template and instruction
        prompt = self._build_dynamic_prompt(
            domain_info, task_content, context_text
        )

        response = self.llm_loader.generate(
            endpoint_key=endpoint_key,
            model_name=model_name,
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            fallback_label=f"{self.role}:{domain_info.name}",
        )

        return {
            "role": self.role,
            "domain": domain_info.name,
            "result": response,
            "task": task_content,
        }
    
    def _build_dynamic_prompt(self, domain_info: DomainConfig, task_content: str, context: str) -> str:
        """Build prompt using dynamic templates"""
        parts = [self.config.template, "", f"Domain focus: {domain_info.instruction}"]
        
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
            
        # Load agent configurations
        for role, agent_data in config_data["agents"].items():
            self._agent_configs[role] = AgentConfig(agent_data)
            
        # Load domain configurations  
        for domain_name, domain_data in config_data["domains"].items():
            self._domain_configs[domain_name] = DomainConfig(domain_name, domain_data)
            
        self._config_loaded = True
        print(f"Loaded {len(self._agent_configs)} agent configs and {len(self._domain_configs)} domain configs")
    

    
    def create_agent(self, role: str, llm_loader: Optional[VLLMLoader] = None) -> 'BaseAgent':
        """Create an agent instance for the given role"""
        if role not in self._agent_configs:
            raise ValueError(f"Unknown agent role: {role}. Available roles: {list(self._agent_configs.keys())}")
            
        agent_config = self._agent_configs[role]
        return DynamicAgent(agent_config, llm_loader)
    
    def get_available_roles(self) -> list:
        """Get list of available agent roles"""
        return list(self._agent_configs.keys())
    
    def get_available_domains(self) -> list:
        """Get list of available domains"""
        return list(self._domain_configs.keys())
    
    @classmethod
    def detect_domain(cls, content: str) -> DomainConfig:
        """Detect domain from content using dynamic keyword matching"""
        factory = cls()
        
        if not content:
            return factory._domain_configs["general"]

        lowered = content.lower()
        
        # Check each domain's keywords
        for domain_name, domain_config in factory._domain_configs.items():
            if domain_name == "general":
                continue  # Skip general, use as fallback
                
            if any(keyword in lowered for keyword in domain_config.keywords):
                return domain_config
        
        # Fallback to general domain
        return factory._domain_configs["general"]
    
    def get_domain_model_mapping(self) -> Dict[str, str]:
        """Get domain to model key mapping for backward compatibility"""
        return {
            domain_name: domain_config.model_key 
            for domain_name, domain_config in self._domain_configs.items()
        }
    
    def get_domain_keywords(self) -> Dict[str, list]:
        """Get domain keywords mapping for backward compatibility"""
        return {
            domain_name: domain_config.keywords
            for domain_name, domain_config in self._domain_configs.items()
        }


# Global factory instance
agent_factory = AgentFactory()


if __name__ == "__main__":
    print("Testing AgentFactory...")
    
    factory = AgentFactory()
    
    # Test available roles
    roles = factory.get_available_roles()
    print(f"Available roles: {roles}")
    
    # Test agent creation
    planning_agent = factory.create_agent("planning")
    execution_agent = factory.create_agent("execution")
    review_agent = factory.create_agent("review")
    
    print(f"Created agents: {planning_agent.role}, {execution_agent.role}, {review_agent.role}")
    
    # Test domain detection
    medical_domain = factory.detect_domain("Patient has chest pain symptoms")
    legal_domain = factory.detect_domain("Analyze contract liability issues")
    math_domain = factory.detect_domain("Calculate the derivative of equation")
    general_domain = factory.detect_domain("Explain how this works")
    
    print(f"Domain detection: medical={medical_domain.name}, legal={legal_domain.name}, math={math_domain.name}, general={general_domain.name}")
    
    # Test agent execution
    test_task = "Diagnose patient with chest pain"
    result = planning_agent.execute(test_task)
    print(f"Planning result: {result['domain']} - {len(result['result'])} chars")
    
    print("AgentFactory test completed!")