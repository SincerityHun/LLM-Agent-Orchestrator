"""
Metrics calculation utilities for LLM-Agent-Orchestrator evaluation
"""
from typing import Dict, List


def calculate_flops(param_billion: float, total_tokens: int) -> float:
    """
    Calculate FLOPs based on Kaplan Scaling Law
    Formula: FLOPs ≈ 2 * Parameters * Total_Tokens
    
    Args:
        param_billion: Model parameter count in billions (B)
        total_tokens: Total number of tokens (input + output)
        
    Returns:
        TFLOPs (Tera FLOPs, 10^12)
    """
    # 2 * (N * 10^9) * tokens
    flops = 2 * (param_billion * 1e9) * total_tokens
    return flops / 1e12  # Convert to TFLOPs


class MetricsCollector:
    """
    Collects and tracks metrics during orchestration execution
    """
    
    def __init__(self):
        self.reset()
        
        # Model sizes (in billions of parameters)
        self.model_sizes = {
            "global-router": 8.0,      # Llama-3.1-8B for global router
            "result-handler": 8.0,     # Llama-3.1-8B for result handler
            "agent-1b": 1.0,           # Llama-1B for lightweight agents
            "agent-8b": 8.0,           # Llama-8B for complex agents
        }
    
    def reset(self):
        """Reset all metrics for a new task"""
        self.router_calls = []
        self.agent_calls = []
        self.handler_calls = []
        self.total_flops = 0.0
    
    def add_router_call(self, input_tokens: int, output_tokens: int):
        """Track a global router call"""
        total_tokens = input_tokens + output_tokens
        flops = calculate_flops(self.model_sizes["global-router"], total_tokens)
        
        self.router_calls.append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "flops": flops
        })
        self.total_flops += flops
    def add_sub_router_call(self, input_tokens: int, output_tokens: int):
        """Track a sub router call"""
        total_tokens = input_tokens + output_tokens
        flops = calculate_flops(self.model_sizes["agent-1b"], total_tokens)
        
        self.router_calls.append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "flops": flops
        })
        self.total_flops += flops
        
    def add_agent_call(self, agent_domain: str, model_size: str, 
                       input_tokens: int, output_tokens: int):
        """
        Track an agent execution call
        
        Args:
            agent_domain: Domain of the agent (medical, law, math, commonsense)
            model_size: Model size used ('1b' or '8b')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        total_tokens = input_tokens + output_tokens
        model_key = f"agent-{model_size}"
        flops = calculate_flops(self.model_sizes[model_key], total_tokens)
        
        self.agent_calls.append({
            "domain": agent_domain,
            "model_size": model_size,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "flops": flops
        })
        self.total_flops += flops
    
    def add_handler_call(self, input_tokens: int, output_tokens: int):
        """Track a result handler call"""
        total_tokens = input_tokens + output_tokens
        flops = calculate_flops(self.model_sizes["result-handler"], total_tokens)
        
        self.handler_calls.append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "flops": flops
        })
        self.total_flops += flops
    
    def get_summary(self) -> Dict:
        """
        Get a summary of all collected metrics
        
        Returns:
            Dictionary containing aggregated metrics
        """
        total_router_tokens = sum(call["total_tokens"] for call in self.router_calls)
        total_agent_tokens = sum(call["total_tokens"] for call in self.agent_calls)
        total_handler_tokens = sum(call["total_tokens"] for call in self.handler_calls)
        
        router_flops = sum(call["flops"] for call in self.router_calls)
        agent_flops = sum(call["flops"] for call in self.agent_calls)
        handler_flops = sum(call["flops"] for call in self.handler_calls)
        
        return {
            "total_flops_tflops": self.total_flops,
            "router_flops_tflops": router_flops,
            "agent_flops_tflops": agent_flops,
            "handler_flops_tflops": handler_flops,
            "router_calls": len(self.router_calls),
            "agent_calls": len(self.agent_calls),
            "handler_calls": len(self.handler_calls),
            "total_router_tokens": total_router_tokens,
            "total_agent_tokens": total_agent_tokens,
            "total_handler_tokens": total_handler_tokens,
            "total_tokens": total_router_tokens + total_agent_tokens + total_handler_tokens
        }
    
    def get_agent_breakdown(self) -> List[Dict]:
        """Get per-agent breakdown of metrics"""
        return self.agent_calls.copy()
