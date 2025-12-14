"""
vLLM wrapper for loading and calling models
"""
import os
from typing import Dict

import requests


class VLLMLoader:
    """Wrapper for vLLM endpoint calls"""

    def __init__(self) -> None:
        # vLLM server endpoints (docker-compose setup)
        llama_1b_endpoint = os.getenv("VLLM_LLAMA_1B_ENDPOINT", "http://localhost:8000/v1")
        llama_8b_endpoint = os.getenv("VLLM_LLAMA_8B_ENDPOINT", "http://localhost:8001/v1")

        self.endpoints: Dict[str, str] = {
            "llama-1b": llama_1b_endpoint.rstrip("/"),
            "llama-8b": llama_8b_endpoint.rstrip("/"),
        }

        # Model configurations using vLLM LoRA adapter names
        # These adapter names MUST exactly match the docker-compose.yml lora-modules configuration
        self.model_configs: Dict[str, Dict[str, str]] = {
            # Global router uses base Llama-3.1-8B (no LoRA)
            "global-router": {
                "endpoint": "llama-8b",
                "model": os.getenv("GLOBAL_ROUTER_MODEL_NAME", "meta-llama/Llama-3.1-8B"),
            },
            # Result handler uses base Llama-3.1-8B (no LoRA)
            "result-handler": {
                "endpoint": "llama-8b",
                "model": os.getenv("RESULT_HANDLER_MODEL", "meta-llama/Llama-3.1-8B"),
            },
            # Domain-specific routers are NOT in vLLM - they run in separate router-service
            # Commonsense domain models (LoRA adapter names from docker-compose.yml)
            "commonsense-1b": {
                "endpoint": "llama-1b",
                "model": "csqa-lora",  # Matches docker: csqa-lora=/csqa-adapter
            },
            "commonsense-8b": {
                "endpoint": "llama-8b",
                "model": "csqa-lora",  # Matches docker: csqa-lora=/csqa-adapter
            },
            # Medical domain models
            "medical-1b": {
                "endpoint": "llama-1b",
                "model": "medqa-lora",  # Matches docker: medqa-lora=/medqa-adapter
            },
            "medical-8b": {
                "endpoint": "llama-8b",
                "model": "medqa-lora",  # Matches docker: medqa-lora=/medqa-adapter
            },
            # Law domain models
            "law-1b": {
                "endpoint": "llama-1b",
                "model": "casehold-lora",  # Matches docker: casehold-lora=/casehold-adapter
            },
            "law-8b": {
                "endpoint": "llama-8b",
                "model": "casehold-lora",  # Matches docker: casehold-lora=/casehold-adapter
            },
            # Math domain models
            "math-1b": {
                "endpoint": "llama-1b",
                "model": "mathqa-lora",  # Matches docker: mathqa-lora=/mathqa-adapter
            },
            "math-8b": {
                "endpoint": "llama-8b",
                "model": "mathqa-lora",  # Matches docker: mathqa-lora=/mathqa-adapter
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
        
        # Use vLLM completions endpoint (endpoint already includes /v1)
        url = f"{endpoint}/completions"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Different settings for LoRA adapters vs base models
        if "lora" in model_name.lower() or any(d in model_name.lower() for d in ["medqa", "casehold", "mathqa", "csqa"]):
            # LoRA domain agents: stricter control
            payload["repetition_penalty"] = 1.1
            payload["stop"] = ["\n\n\n", "Task:", "Response:"]
        else:
            # Base models (router/evaluation): minimal constraints
            payload["repetition_penalty"] = 1.0
            payload["stop"] = ["\n\n\n"]  # Minimal stop sequences
        
        # Add guided parameters directly to payload (not in extra_body)
        if guided_json is not None:
            payload["guided_json"] = guided_json
        if guided_regex is not None:
            payload["guided_regex"] = guided_regex

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            choices = result.get("choices", [])
            if not choices:
                print(f"⚠️ [vLLM] Empty choices from {url}")
                print(f"   Response: {result}")
                raise ValueError("Empty choices from vLLM response")

            # Completions endpoint returns text directly
            choice = choices[0]
            text = choice.get("text", "")
            finish_reason = choice.get("finish_reason", "")
            completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
            
            # Warn if response is empty
            if not text or not text.strip():
                print(f"⚠️ [vLLM] Empty text from {url}")
                print(f"   Model: {model_name}")
                print(f"   Prompt ending: ...{prompt[-150:]}")
                print(f"   Finish reason: {finish_reason}")
                print(f"   Completion tokens: {completion_tokens}")
                print(f"   → Likely cause: Prompt ends with completed template or immediate stop token")

            return (text or "").strip()

        except (requests.exceptions.RequestException, ValueError, KeyError, IndexError) as e:
            print(f"❌ [vLLM] Error calling {url}: {type(e).__name__}: {e}")
            print(f"   Model: {model_name}")
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
