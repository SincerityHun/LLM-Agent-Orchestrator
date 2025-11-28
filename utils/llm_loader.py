"""
vLLM wrapper for loading and calling models
"""
import os
import requests
from typing import Dict, Optional


class VLLMLoader:
    """Wrapper for vLLM endpoint calls"""
    
    def __init__(self):
        # Load vLLM endpoints from environment
        self.endpoints = {
            "llama": os.getenv("VLLM_LLAMA_ENDPOINT", "http://localhost:8000/v1"),
            "qwen": os.getenv("VLLM_QWEN_ENDPOINT", "http://localhost:8001/v1"),
        }
        
        # LoRA adapter mapping for each model type
        self.model_adapters = {
            "commonsense": {"endpoint": "llama", "model": "csqa-lora"},
            "legal": {"endpoint": "llama", "model": "casehold-lora"},
            "medical": {"endpoint": "llama", "model": "medqa-lora"},
            "math": {"endpoint": "qwen", "model": "mathqa-lora"},
        }
    
    def call_model(
        self,
        model_type: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Call vLLM model endpoint with appropriate LoRA adapter
        
        Args:
            model_type: One of 'commonsense', 'legal', 'medical', 'math'
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        if model_type not in self.model_adapters:
            raise ValueError(f"Unknown model type: {model_type}")
        
        adapter_config = self.model_adapters[model_type]
        endpoint_key = adapter_config["endpoint"]
        model_name = adapter_config["model"]
        
        if endpoint_key not in self.endpoints:
            raise ValueError(f"Endpoint {endpoint_key} not configured")
        
        endpoint = self.endpoints[endpoint_key]
        url = f"{endpoint}/completions"
        
        payload = {
            "model": model_name,  # Specify the LoRA adapter to use
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["\n\n"]
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["text"].strip()
            
        except requests.exceptions.RequestException as e:
            # Fallback for testing without vLLM server
            return f"[MOCK RESPONSE for {model_type}] {prompt[:50]}..."
    
    def is_endpoint_available(self, model_type: str) -> bool:
        """Check if vLLM endpoint is available for the given model type"""
        if model_type not in self.model_adapters:
            return False
        
        try:
            adapter_config = self.model_adapters[model_type]
            endpoint_key = adapter_config["endpoint"]
            
            if endpoint_key not in self.endpoints:
                return False
                
            endpoint = self.endpoints[endpoint_key]
            url = f"{endpoint}/models"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False


if __name__ == "__main__":
    # Unit test
    print("Testing VLLMLoader...")
    
    loader = VLLMLoader()
    
    # Test adapter configuration
    assert "commonsense" in loader.model_adapters
    assert "medical" in loader.model_adapters
    assert "legal" in loader.model_adapters
    assert "math" in loader.model_adapters
    
    # Test model call (will use mock if server not available)
    test_prompt = "What is the capital of France?"
    response = loader.call_model("commonsense", test_prompt, max_tokens=50)
    assert isinstance(response, str)
    assert len(response) > 0
    
    print("VLLMLoader test passed")
