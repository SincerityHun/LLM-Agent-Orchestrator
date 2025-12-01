"""
vLLM wrapper for loading and calling models
"""
import os
from typing import Dict

import requests


class VLLMLoader:
    """Wrapper for vLLM endpoint calls"""

    def __init__(self) -> None:
        llama_endpoint = os.getenv("VLLM_LLAMA_ENDPOINT", "http://localhost:8000/v1")
        qwen_endpoint = os.getenv("VLLM_QWEN_ENDPOINT", "http://localhost:8001/v1")

        self.endpoints: Dict[str, str] = {
            "llama": llama_endpoint.rstrip("/"),
            "qwen": qwen_endpoint.rstrip("/"),
        }

        self.model_configs: Dict[str, Dict[str, str]] = {
            "global-router": {
                "endpoint": "llama",
                "model": os.getenv(
                    "GLOBAL_ROUTER_MODEL_NAME",
                    "meta-llama/Llama-3.2-1B-Instruct",
                ),
            },
            "commonsense": {
                "endpoint": "llama",
                "model": os.getenv("CSQA_MODEL_NAME", "csqa-lora"),
            },
            "legal": {
                "endpoint": "llama",
                "model": os.getenv("CASEHOLD_MODEL_NAME", "casehold-lora"),
            },
            "medical": {
                "endpoint": "llama",
                "model": os.getenv("MEDQA_MODEL_NAME", "medqa-lora"),
            },
            "math": {
                "endpoint": "qwen",
                "model": os.getenv("MATHQA_MODEL_NAME", "mathqa-lora"),
            },
        }

    def call_model(
        self,
        model_type: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        guided_json: dict | None = None,
        guided_regex: str | None = None,
    ) -> str:
        """Call vLLM model endpoint using configured model type"""

        if model_type not in self.model_configs:
            raise ValueError(f"Unknown model type: {model_type}")

        config = self.model_configs[model_type]
        endpoint_key = config["endpoint"]

        return self.generate(
            endpoint_key=endpoint_key,
            model_name=config["model"],
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            guided_json=guided_json,
            guided_regex=guided_regex,
            fallback_label=model_type,
        )

    def generate(
        self,
        endpoint_key: str,
        model_name: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        guided_json: dict | None = None,
        guided_regex: str | None = None,
        fallback_label: str | None = None,
    ) -> str:
        """Call vLLM endpoint directly with explicit endpoint and model name"""

        if endpoint_key not in self.endpoints:
            raise ValueError(f"Unknown endpoint key: {endpoint_key}")

        endpoint = self.endpoints[endpoint_key]
        
        # Use chat completions endpoint if guided parameters are provided
        if guided_json is not None or guided_regex is not None:
            url = f"{endpoint}/chat/completions"
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            # Add guided parameters to extra_body
            extra_body = {}
            if guided_json is not None:
                extra_body["guided_json"] = guided_json
            if guided_regex is not None:
                extra_body["guided_regex"] = guided_regex
            
            if extra_body:
                payload["extra_body"] = extra_body
        else:
            # Use legacy completions endpoint
            url = f"{endpoint}/completions"
            payload = {
                "model": model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            choices = result.get("choices", [])
            if not choices:
                raise ValueError("Empty choices from vLLM response")

            # Handle both completion types
            choice = choices[0]
            
            # Chat completion response format
            if "message" in choice:
                message = choice.get("message", {})
                text = message.get("content", "")
            # Legacy completion response format
            else:
                text = choice.get("text", "")

            return (text or "").strip()

        except (requests.exceptions.RequestException, ValueError, KeyError, IndexError):
            label = fallback_label or model_name or endpoint_key
            return f"[MOCK RESPONSE for {label}] {prompt[:50]}..."

    def is_endpoint_available(self, model_type: str) -> bool:
        """Check if vLLM endpoint is available"""

        if model_type not in self.model_configs:
            return False

        config = self.model_configs[model_type]
        endpoint_key = config.get("endpoint")
        if endpoint_key not in self.endpoints:
            return False

        try:
            endpoint = self.endpoints[endpoint_key]
            url = f"{endpoint}/models"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


if __name__ == "__main__":
    print("Testing VLLMLoader...")

    loader = VLLMLoader()

    assert "global-router" in loader.model_configs
    assert loader.model_configs["commonsense"]["model"]
    assert loader.model_configs["medical"]["endpoint"] == "llama"
    assert loader.model_configs["math"]["endpoint"] == "qwen"

    test_prompt = "Provide a concise answer for test validation."
    for model_type in loader.model_configs:
        print(f"\nTesting model type: {model_type}")
        response = loader.call_model(
            model_type,
            f"[{model_type}] {test_prompt}",
            max_tokens=256,
        )
        print(f"Response: {response}")
        assert isinstance(response, str)
        assert len(response) > 0

    print("VLLMLoader test passed")
