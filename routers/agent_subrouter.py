"""Agent SubRouter for model selection using domain-specific routers"""

import os
from typing import Dict, List, Optional, Tuple
import requests

from pydantic import BaseModel, Field

from utils.agent_prompts import get_agent_prompt
from utils.llm_loader import VLLMLoader
from agents.agent_factory import AgentFactory


class AgentResponse(BaseModel):
    """Schema for agent response with guided JSON generation"""
    
    response: str = Field(
        ...,
        description="The agent's complete response to the task. Must be detailed and comprehensive.",
        min_length=10
    )


class AgentSubRouter:
    """SubRouter uses domain-specific router service to select between 1b and 8b models"""

    def __init__(self, llm_loader: Optional[VLLMLoader] = None, use_router_service: bool = True) -> None:
        self.llm_loader = llm_loader or VLLMLoader()
        
        # Initialize with dynamic configuration
        factory = AgentFactory()
        self.domain_keywords = factory.get_domain_keywords()
        self.agent_configs = factory._agent_configs
        
        # Router service configuration
        self.use_router_service = use_router_service
        self.router_service_url = os.getenv("ROUTER_SERVICE_URL", "http://localhost:8002")
        
        # Check if router service is available
        if self.use_router_service:
            try:
                response = requests.get(f"{self.router_service_url}/health", timeout=2)
                if response.status_code == 200:
                    print(f"✓ Router service available at {self.router_service_url}")
                else:
                    print(f"⚠ Router service returned status {response.status_code}, falling back to vLLM")
                    self.use_router_service = False
            except requests.exceptions.RequestException as e:
                print(f"⚠ Router service not available ({e}), falling back to vLLM")
                self.use_router_service = False

    def select_model_for_domain(self, domain: str, task: str, context: Optional[Dict] = None) -> Tuple[str, str, str]:
        """
        Use domain-specific router service to select between 1b and 8b models
        
        Args:
            domain: Domain name (commonsense, medical, law, math)
            task: Task description
            context: Optional context information
            
        Returns:
            Tuple of (endpoint_key, model_name, model_size)
        """
        # Use router service for model selection
        if self.use_router_service:
            try:
                model_size = self._call_router_service(domain, task, context)
            except Exception as e:
                # If router service fails, default to small model
                print(f"⚠️ Router service call failed for {domain}: {e}, defaulting to 1b")
                model_size = "1b"
        else:
            # If router service unavailable, default to small model
            print(f"⚠️ Router service not available, defaulting to 1b for {domain}")
            model_size = "1b"
        
        # Select the appropriate model based on router decision
        if model_size == "large" or model_size == "8b":
            model_key = f"{domain}-8b"
            actual_model_size = "8b"
        else:
            model_key = f"{domain}-1b"
            actual_model_size = "1b"
        
        config = self.llm_loader.model_configs.get(model_key)
        if not config:
            raise ValueError(f"Model configuration missing for {model_key}")

        return config["endpoint"], config["model"], actual_model_size
    
    def _call_router_service(self, domain: str, task: str, context: Optional[Dict] = None) -> str:
        """Call the router service API"""
        try:
            response = requests.post(
                f"{self.router_service_url}/route/{domain}",
                json={"task": task, "context": context},
                timeout=5
            )
            response.raise_for_status()
            
            result = response.json()
            prediction = result["prediction"]  # "1b" or "8b"
            probability = result["probability"]
            
            # Apply threshold: only use 8b if probability >= 0.68
            # THRESHOLD = 0.5
            # if prediction == "8b" and probability >= THRESHOLD:
            #     final_decision = "8b"
            # else:
            #     final_decision = "1b"
            final_decision = "8b" # ABlation Test
            
            # Log the decision
            if final_decision != prediction:
                print(f"[Router Service] {domain}: {prediction} (p={probability:.3f}) -> Override to {final_decision} (ABLATION TEST)")
            else:
                print(f"[Router Service] {domain}: {final_decision} (p={probability:.3f})")
            
            return final_decision
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Router service request failed: {e}")
    


    def execute_subtask(
        self,
        domain: str,
        task: str,
        context: Optional[Dict] = None,
    ) -> Dict:
        """
        Execute a domain-specific subtask with router-based model selection
        
        Args:
            domain: Domain name (commonsense, medical, law, math)
            task: Task description
            context: Optional context information
            
        Returns:
            Dict with 'text', 'usage', and 'model_size' keys
        """
        context = context or {}
        
        # Use domain router to select model size
        endpoint_key, model_name, model_size = self.select_model_for_domain(domain, task, context)
        
        # Get agent configuration for temperature
        agent_config = self.agent_configs.get(domain)
        temperature = agent_config.temperature if agent_config else 0.4
        max_tokens = agent_config.max_tokens if agent_config else 512
        
        # Build prompt using domain-specific template
        prompt = self._build_domain_prompt(domain, task, context)

        result = self.llm_loader.generate(
            endpoint_key=endpoint_key,
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            fallback_label=f"{domain}",
            return_usage=True,
        )
        
        # Add model size to result
        if isinstance(result, dict):
            result["model_size"] = model_size
            return result
        else:
            # Fallback for string responses (shouldn't happen with return_usage=True)
            return {
                "text": result,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model_size": model_size
            }
    
    def _build_domain_prompt(self, domain: str, task: str, context: Dict) -> str:
        """Build prompt using domain-specific template from config"""
        agent_config = self.agent_configs.get(domain)
        
        if not agent_config:
            # Fallback to simple prompt
            return f"Task: {task}\n\nResponse:"
        
        # Use compact format that works with LoRA models
        # Format: "{template}\n\nTask: {task}\n\n[Context: ...]\n\nResponse:"
        parts = [agent_config.template]
        
        # Add task first (most important)
        parts.append(f"\n\nTask: {task.strip()}")
        
        # Add context after task if available (less likely to interfere with generation)
        if context:
            context_str = self._format_context(context)
            # Filter out internal keys that shouldn't be in prompt
            filtered_context = {k: v for k, v in context.items() 
                              if k not in ['user_id'] and v and str(v).strip()}
            if filtered_context:
                context_str = "\n".join(f"{k}: {v}" for k, v in filtered_context.items())
                if context_str:
                    parts.append(f"\n\nContext: {context_str}")
        
        # End with generation prompt
        parts.append(f"\n\nResponse:")
        
        return "".join(parts)

    def _format_context(self, context: Dict) -> str:
        if not context:
            return ""
        return "\n".join(f"{key}: {value}" for key, value in context.items())


if __name__ == "__main__":
    print("Testing AgentSubRouter...")

    subrouter = AgentSubRouter()

    # Test model selection with router
    medical_task = "Diagnose patient with chest pain symptoms and recommend treatment plan"
    endpoint_key, model_name = subrouter.select_model_for_domain("medical", medical_task)
    assert endpoint_key == "llama"
    assert model_name
    print(f"Medical task routed to: {model_name}")

    math_task = "Calculate the derivative of x^2 + 3x"
    endpoint_key, model_name = subrouter.select_model_for_domain("math", math_task)
    print(f"Math task routed to: {model_name}")

    # Test subtask execution
    result = subrouter.execute_subtask(
        domain="medical",
        task=medical_task,
        context={"user": "test_user"},
    )
    assert isinstance(result, str)
    assert len(result) > 0
    print(f"Result length: {len(result)}")

    print("AgentSubRouter test passed")
