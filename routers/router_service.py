"""
Router Inference Service
Separate FastAPI service for router model inference with modules_to_save support
"""
import os
import sys
import re
import time
from typing import Dict, Optional

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.router_config import RouterModelConfig, ModelTypeEnum, PROMPT_FORMAT_CONFIGS
from utils.prompt_format import PromptFormat


class RouteRequest(BaseModel):
    """Request model for routing"""
    task: str
    context: Optional[Dict] = None


class RouteResponse(BaseModel):
    """Response model for routing"""
    prediction: str  # "1b" or "8b"
    probability: float  # Probability of 8B model
    label: str  # "[[1]]" or "[[2]]"
    softmax_scores: list  # [prob_1b, prob_8b]


class RouterInference:
    """Router inference using merged LoRA model"""
    
    def __init__(self, checkpoint_path: str, domain: str, base_model: str = "meta-llama/Llama-3.2-1B"):
        self.domain = domain
        self.checkpoint_path = checkpoint_path
        
        print(f"[{domain}] Loading router from {checkpoint_path}...")
        start_time = time.time()
        
        # Setup config
        config = RouterModelConfig(
            model_id=base_model,
            model_type=ModelTypeEnum.CAUSAL,
            num_outputs=2,  # Binary classification
            special_tokens=["[[1]]", "[[2]]"],
        )
        self.config = config
        
        # Load tokenizer from checkpoint (has special tokens)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                legacy=True,
                token=os.getenv("HF_TOKEN"),
            )
            print(f"[{domain}] Loaded tokenizer from checkpoint")
        except Exception as e:
            print(f"[{domain}] Warning: Could not load tokenizer from checkpoint: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.tokenizer.add_tokens(config.special_tokens, special_tokens=True)
        
        self.orig_vocab_size = len(self.tokenizer) - config.num_outputs
        
        # Load base model
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # Resize embeddings if needed
        if base.get_input_embeddings().weight.shape[0] != len(self.tokenizer):
            base.resize_token_embeddings(len(self.tokenizer))
        
        # Load and merge LoRA adapter
        model = PeftModel.from_pretrained(base, checkpoint_path)
        self.model = model.merge_and_unload()
        self.model.eval()
        
        # Setup prompt format
        prompt_format_config = PROMPT_FORMAT_CONFIGS.get(
            base_model,
            PROMPT_FORMAT_CONFIGS["meta-llama/Meta-Llama-3-8B"]
        )
        self.prompt_format = PromptFormat(**prompt_format_config, is_generation=True)
        
        print(f"[{domain}] Router loaded in {time.time() - start_time:.2f}s")
    
    def predict(self, task_text: str, context: Optional[Dict] = None) -> Dict:
        """Predict routing decision for a task"""
        # Router model is trained to generate [[1]] or [[2]] tokens directly
        # System message contains routing instructions (optional, model may work without it)
        # User message contains only the actual task
        
        # Build user message with task (and optional context)
        if context:
            context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            user_content = f"Context: {context_str}\n\nTask: {task_text}"
        else:
            user_content = task_text
        
        # Format as chat messages - router trained to generate [[1]] or [[2]] from task
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": ""}
        ]
        
        # Generate prompt using format
        formatted_prompt = self.prompt_format.generate_prompt(messages)
        
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        
        # Generate - router should produce [[1]] or [[2]] token immediately
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=3,  # Only need 1-2 tokens for [[1]] or [[2]]
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
        
        # Router classification: use logits from FIRST token generation step
        # Model is trained to encode routing decision in first token's logits
        # We don't need to actually generate [[1]] or [[2]] - just check their probabilities
        
        if len(outputs.scores) == 0:
            print(f"[{self.domain}] ⚠️ No scores available! Using fallback (1b)")
            return {
                "prediction": "1b",
                "probability": 0.0,
                "label": "[[1]]",
                "softmax_scores": [1.0, 0.0]
            }
        
        # Get logits from first generation step (index 0)
        first_token_logits = outputs.scores[0][0].cpu().numpy()
        
        # Extract logits for special tokens [[1]] and [[2]]
        # [[1]] is at vocab_size, [[2]] is at vocab_size + 1
        special_logits = first_token_logits[self.orig_vocab_size:self.orig_vocab_size + 2]
        
        # DEBUG: Show what was actually generated vs what we're using
        output_ids = outputs.sequences[0][input_ids.shape[1]:].cpu()
        generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        print(f"[{self.domain}] Generated text: '{generated_text}'")
        print(f"[{self.domain}] Special token logits: [[1]]={special_logits[0]:.3f}, [[2]]={special_logits[1]:.3f}")
        
        # Calculate softmax
        exp_scores = np.exp(special_logits - np.max(special_logits))
        softmax_scores = exp_scores / np.sum(exp_scores)
        
        # Binary probability (probability of [[2]] = 8B model)
        binary_prob = float(softmax_scores[1])
        
        # Decision
        if binary_prob > 0.5:
            prediction = "8b"
            label = "[[2]]"
        else:
            prediction = "1b"
            label = "[[1]]"
        
        return {
            "prediction": prediction,
            "probability": binary_prob,
            "label": label,
            "softmax_scores": [float(softmax_scores[0]), float(softmax_scores[1])]
        }


# Initialize FastAPI app
app = FastAPI(title="Router Inference Service", version="1.0.0")

# Global router instances
routers: Dict[str, RouterInference] = {}


@app.on_event("startup")
async def load_routers():
    """Load all domain routers on startup"""
    base_path = os.getenv("ROUTER_BASE_PATH", "/models")
    base_model = os.getenv("BASE_MODEL", "meta-llama/Llama-3.2-1B")
    
    domains = {
        "commonsense": "commonsense-router",
        "medical": "medqa-router",
        "law": "casehold-router", 
        "math": "mathqa-router",
    }
    
    for domain, router_name in domains.items():
        checkpoint_path = os.path.join(base_path, router_name, "final_model")
        
        if os.path.exists(checkpoint_path):
            try:
                routers[domain] = RouterInference(checkpoint_path, domain, base_model)
                print(f"✓ Loaded {domain} router")
            except Exception as e:
                print(f"✗ Failed to load {domain} router: {e}")
        else:
            print(f"⚠ Router checkpoint not found: {checkpoint_path}")
    
    print(f"\nLoaded {len(routers)} routers: {list(routers.keys())}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "loaded_routers": list(routers.keys()),
        "total_routers": len(routers)
    }


@app.post("/route/{domain}", response_model=RouteResponse)
async def route_query(domain: str, request: RouteRequest):
    """Route a query for a specific domain"""
    if domain not in routers:
        raise HTTPException(
            status_code=404,
            detail=f"Router for domain '{domain}' not found. Available: {list(routers.keys())}"
        )
    
    router = routers[domain]
    
    try:
        result = router.predict(request.task, request.context)
        return RouteResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Router inference failed: {str(e)}"
        )


@app.get("/routers")
async def list_routers():
    """List all available routers"""
    return {
        "routers": [
            {
                "domain": domain,
                "checkpoint": router.checkpoint_path,
                "vocab_size": len(router.tokenizer)
            }
            for domain, router in routers.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("ROUTER_SERVICE_PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)
