# LLM-Agent-Orchestrator

**Domain-Specialized Multi-Agent LLM System with Intelligent Model Selection**

A sophisticated multi-agent orchestration framework that decomposes complex tasks into domain-specific subtasks and dynamically routes them to specialized language models, optimizing both accuracy and computational efficiency.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Development](#development)

---

## 🎯 Overview

LLM-Agent-Orchestrator addresses three critical limitations of single-model LLM systems:

1. **Domain Mismatch**: General LLMs lack domain-specific expertise (medical, legal, mathematical, etc.), leading to inaccurate or unsafe outputs
2. **Task Complexity**: Complex multi-faceted tasks require diverse specialized knowledge that single models cannot effectively provide
3. **Computational Inefficiency**: Using large models for simple tasks wastes compute resources

### Solution

Our system uses a **three-tier architecture**:

1. **Global Router**: Decomposes tasks into a Directed Acyclic Graph (DAG) of domain-specific subtasks
2. **Domain Agents**: Execute subtasks using specialized LoRA-adapted models (CommonsenseQA, MedQA, CaseHOLD, MathQA)
3. **Model Router**: Dynamically selects between 1B (simple) and 8B (complex) model variants for each subtask

This approach delivers:
- ✅ Higher accuracy through domain specialization
- ✅ Lower latency via intelligent model sizing
- ✅ Better resource utilization (60-80% compute savings on simple queries)

---

## 🌟 Key Features

### 🔀 Intelligent Task Decomposition
- Automatic task breakdown into domain-specific subtasks
- DAG-based dependency management for parallel execution
- LangGraph orchestration for complex workflows

### 🎓 Domain Specialization
- **4 specialized domains**: Commonsense, Medical, Law, Mathematics
- **LoRA-adapted models** trained on domain-specific datasets:
  - CommonsenseQA (CSQA)
  - MedQA-USMLE
  - CaseHOLD (legal)
  - MathQA

### ⚡ Adaptive Model Selection
- **Router Service**: Separate inference service (FastAPI + GPU) for model selection
- **Binary classification**: Routes simple queries → 1B model, complex queries → 8B model
- **Merged LoRA support**: Handles special tokens with `modules_to_save`

### 🚀 High-Performance Inference
- **vLLM backend**: Optimized inference with continuous batching
- **Multi-GPU deployment**: 
  - GPU 1: Router Service (port 8002)
  - GPU 2: Llama-3.2-1B + LoRA adapters (port 8000)
  - GPU 3: Llama-3.1-8B + LoRA adapters (port 8001)

### 🔄 Iterative Refinement
- Result quality evaluation and feedback loop
- Automatic retry with refined prompts (up to 3 iterations)
- Convergence monitoring

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                            │
│              "Diagnose chest pain and legal risks"              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                      GLOBAL ROUTER (LLM)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Task Decomposition & DAG Generation                     │   │
│  │  • Classify domains (medical, law, math, commonsense)    │   │
│  │  • Identify dependencies                                 │   │
│  │  • Generate execution graph                              │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓
              ┌──────────────────────────────┐
              │    DAG (Directed Graph)      │
              │  Node1: Medical Analysis     │
              │  Node2: Legal Implications   │
              │  Edge: Node1 → Node2         │
              └──────────────┬───────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                   GRAPH BUILDER (LangGraph)                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Parallel & Sequential Execution Engine                  │   │
│  │  • Create workflow nodes from DAG                        │   │
│  │  │  • Execute agents based on dependencies               │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────┬──────────────────────────────┬─────────────────────┘
             │                              │
             ↓                              ↓
    ┌────────────────┐            ┌────────────────┐
    │ DOMAIN AGENT 1 │            │ DOMAIN AGENT 2 │
    │   (Medical)    │            │     (Law)      │
    └────────┬───────┘            └────────┬───────┘
             │                              │
             ↓                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               AGENT SUBROUTER (Model Selection)                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  HTTP Client → Router Service                            │   │
│  │  POST /route/{domain}                                    │   │
│  │  {"task": "analyze symptoms", "context": {...}}          │   │
│  │  Fallback: Default to 1B if service unavailable          │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│            ROUTER SERVICE (Port 8002, GPU 1)                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Domain-Specific Router Models (LoRA-merged)             │   │
│  │  Classification: Logits-based (first token position)     │   │
│  │  • Medical Router → "1b" or "8b" (softmax over [[1]],[[2]]  )│
│  │  • Law Router → "1b" or "8b"                             │   │
│  │  • Math Router → "1b" or "8b"                            │   │
│  │  • Commonsense Router → "1b" or "8b"                     │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                   ┌─────────┴─────────┐
                   ↓                   ↓
      ┌────────────────────┐  ┌────────────────────┐
      │  vLLM Server 1B    │  │  vLLM Server 8B    │
      │  Port 8000, GPU 2  │  │  Port 8001, GPU 3  │
      │                    │  │                    │
      │  Llama-3.2-1B      │  │  Llama-3.1-8B      │
      │  + csqa-lora       │  │  + csqa-lora       │
      │  + medqa-lora      │  │  + medqa-lora      │
      │  + casehold-lora   │  │  + casehold-lora   │
      │  + mathqa-lora     │  │  + mathqa-lora     │
      └─────────┬──────────┘  └─────────┬──────────┘
                │                       │
                └───────┬───────────────┘
                        │
                        ↓
             ┌────────────────────┐
             │  Agent Responses   │
             └──────────┬─────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│                     RESULT HANDLER                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  • Merge agent outputs                                   │   │
│  │  • Evaluate completeness                                 │   │
│  │  • Generate feedback for refinement (if needed)          │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓
                   ┌───────────────────┐
                   │  FINAL RESPONSE   │
                   └───────────────────┘
```

### Component Details

#### 1. Global Router
- **Location**: `routers/global_router.py`
- **Model**: Llama-3.1-8B (base model, no LoRA)
- **Function**: 
  - Analyzes user query
  - Classifies into domains (commonsense, medical, law, math)
  - Generates DAG with task dependencies
  - Uses structured output (Pydantic schemas with guided_json)
- **Domain Validation**: 
  - Pydantic Literal type enforces valid domain names
  - Prevents invalid domains like "legal" (auto-corrects to "law")

#### 2. Domain Agents
- **Location**: `agents/base_agent.py`, `agents/agent_factory.py`
- **Configuration**: `agents/config.json`
- **Domains**:
  - **Commonsense**: General reasoning, everyday knowledge
  - **Medical**: Clinical diagnosis, treatment planning, medical analysis
  - **Law**: Legal reasoning, case analysis, regulatory compliance
  - **Math**: Quantitative analysis, calculations, proofs

#### 3. Agent SubRouter
- **Location**: `routers/agent_subrouter.py`
- **Function**:
  - Interfaces with Router Service via HTTP API (port 8002)
  - Defaults to 1B model if router service unavailable
  - Returns endpoint and model name for task execution

#### 4. Router Service
- **Location**: `routers/router_service.py`
- **Deployment**: Separate FastAPI container (GPU 1, port 8002)
- **Models**: 4 domain-specific LoRA-merged routers (commonsense, medical, law, math)
- **Classification Method**: 
  - Logits-based binary classification (not token generation)
  - Analyzes first token position logits for special tokens [[1]] and [[2]]
  - Returns probability distribution via softmax over special token logits
- **Endpoints**:
  - `GET /health` - Health check
  - `GET /routers` - List loaded routers
  - `POST /route/{domain}` - Get model selection for task

#### 5. vLLM Inference Servers
- **1B Server** (GPU 2, port 8000):
  - Base: `meta-llama/Llama-3.2-1B`
  - LoRA adapters: csqa, medqa, casehold, mathqa
  - Use: Fast inference for simple tasks
  
- **8B Server** (GPU 3, port 8001):
  - Base: `meta-llama/Llama-3.1-8B`
  - LoRA adapters: csqa, medqa, casehold, mathqa
  - Use: High-accuracy inference for complex tasks

#### 6. Graph Builder
- **Location**: `orchestrator/graph_builder.py`
- **Framework**: LangGraph
- **Function**:
  - Converts DAG to executable workflow
  - Manages state across nodes
  - Enables parallel execution where possible

#### 7. Result Handler
- **Location**: `orchestrator/result_handler.py`
- **Model**: Llama-3.1-8B (base model, no LoRA)
- **Function**:
  - Merges outputs from multiple agents using merge_utils
  - Evaluates result quality and completeness
  - Generates feedback for Global Router refinement
  - Implements iterative refinement loop (default: max 3 iterations)
- **Output Format**: Simple merged text concatenation with domain labels

---

## 📁 Project Structure

```
LLM-Agent-Orchestrator/
├── main.py                          # Main orchestration entry point
├── test_integration.py              # Comprehensive integration test suite
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── agents/                          # Domain-specific agents
│   ├── __init__.py
│   ├── base_agent.py               # Base agent class with common logic
│   ├── agent_factory.py            # Dynamic agent creation from config
│   └── config.json                 # Agent & domain configurations
│
├── routers/                         # Routing system
│   ├── __init__.py
│   ├── global_router.py            # Task decomposition & DAG generation
│   ├── agent_subrouter.py          # Model selection (1B vs 8B)
│   └── router_service.py           # FastAPI service for router inference
│
├── orchestrator/                    # Workflow orchestration
│   ├── __init__.py
│   ├── graph_builder.py            # LangGraph builder & executor
│   └── result_handler.py           # Result evaluation & refinement
│
├── utils/                           # Shared utilities
│   ├── __init__.py
│   ├── llm_loader.py               # vLLM endpoint wrapper
│   ├── router_prompts.py           # Global router prompts
│   ├── agent_prompts.py            # Domain agent prompts
│   ├── router_config.py            # Router model configurations
│   ├── prompt_format.py            # Prompt formatting utilities
│   └── merge_utils.py              # Output merging utilities
│
├── docker/                          # Docker deployment
│   ├── docker-compose.yml          # Multi-service deployment
│   ├── Dockerfile                  # Main orchestrator container
│   ├── Dockerfile.router           # Router service container
│   ├── requirements.router.txt     # Router service dependencies
│   └── .env                        # Environment variables (HF_TOKEN)
│
├── examples/                        # Example inputs & outputs
│   ├── sample_request.json
│   ├── sample_task_decomposition.json
│   ├── sample_task_decomposition_results.json
│   └── sample_results.txt
│
└── scripts/                         # Utility scripts
    ├── setup.sh                    # Setup script
    ├── test_router_api.py          # Router service API tests
    └── test_utils_integration.py   # Utils integration tests
```

---

## 🔧 Installation

### Prerequisites

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.10+
- **CUDA**: 11.8 or 12.8
- **GPU**: 4x GPUs (1 for router, 1 for 1B model, 1 for 8B model, 1 spare) or adjust configuration
- **Disk**: 50GB+ free space for models
- **Hugging Face Token**: For model downloads

### 1. Clone Repository

```bash
git clone https://github.com/SincerityHun/LLM-Agent-Orchestrator.git
cd LLM-Agent-Orchestrator
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- `langgraph`: Workflow orchestration
- `langchain`: Agent framework
- `pydantic`: Schema validation
- `requests`: HTTP client
- `transformers`, `torch`, `peft`: For router service

### 3. Set Up Environment Variables

Create `docker/.env` file:

```bash
# Hugging Face token for model downloads
HF_TOKEN=hf_your_token_here

# vLLM endpoints (if running outside Docker)
VLLM_LLAMA_1B_ENDPOINT=http://localhost:8000/v1
VLLM_LLAMA_8B_ENDPOINT=http://localhost:8001/v1

# Router service
ROUTER_SERVICE_URL=http://localhost:8002

# Model paths (adjust for your setup)
BASE_MODEL=meta-llama/Llama-3.2-1B

# LoRA adapter names (should match docker-compose.yml)
CSQA_1B=csqa-lora
MEDQA_1B=medqa-lora
CASEHOLD_1B=casehold-lora
MATHQA_1B=mathqa-lora

CSQA_8B=csqa-lora
MEDQA_8B=medqa-lora
CASEHOLD_8B=casehold-lora
MATHQA_8B=mathqa-lora
```

### 4. Prepare LoRA Adapters

Ensure your LoRA-adapted models are available at the paths specified in `docker-compose.yml`:

```bash
# Example structure:
/mnt/ssd1/shjung/embos/
├── llama-1b-csqa/          # CommonsenseQA adapter for 1B
├── llama-1b-medqa/         # MedQA adapter for 1B
├── llama-1b-casehold/      # CaseHOLD adapter for 1B
├── llama-1b-mathqa/        # MathQA adapter for 1B
├── llama-8b-csqa/          # CommonsenseQA adapter for 8B
├── llama-8b-medqa/         # MedQA adapter for 8B
├── llama-8b-casehold/      # CaseHOLD adapter for 8B
└── llama-8b-mathqa/        # MathQA adapter for 8B
```

**Note**: Update volume paths in `docker-compose.yml` to match your system.

---

## 🚀 Quick Start

### Option 1: Docker Deployment (Recommended)

```bash
cd docker
docker compose up -d
```

This starts:
1. **vLLM-Llama-1B** (GPU 2, port 8000)
2. **vLLM-Llama-8B** (GPU 3, port 8001)
3. **Router Service** (GPU 1, port 8002)

Wait 2-3 minutes for models to load, then verify:

```bash
# Check vLLM 1B server
curl http://localhost:8000/v1/models

# Check vLLM 8B server
curl http://localhost:8001/v1/models

# Check router service
curl http://localhost:8002/health
```

### Option 2: Local Development

If running vLLM servers separately:

```bash
# Set environment variables
export VLLM_LLAMA_1B_ENDPOINT=http://localhost:8000/v1
export VLLM_LLAMA_8B_ENDPOINT=http://localhost:8001/v1
export ROUTER_SERVICE_URL=http://localhost:8002

# Run orchestrator
python main.py
```

### Running the Orchestrator

#### 1. Default Test

```bash
python main.py
```

#### 2. Custom Task (CLI)

```bash
python main.py "Analyze whether a patient with chest pain needs emergency CT scan"
```

#### 3. From JSON File

```bash
python main.py examples/sample_request.json
```

**Example `sample_request.json`**:

```json
{
    "task": "Analyze the legal implications of a medical malpractice case involving chest pain diagnosis, calculate damages, and provide commonsense recommendations.",
    "user_id": "test_user_01"
}
```

#### 4. Programmatic Usage

```python
from main import LLMOrchestrator

orchestrator = LLMOrchestrator(max_retry=3)

result = orchestrator.process_task(
    task="Calculate damages in a medical malpractice case",
    user_id="attorney_456"
)

print(result["final_answer"])
```

### Expected Output
you can check `examples/sample_results.txt` for a sample output of the sample_request.json task.


---

## ⚙️ Configuration

### Agent Configuration

Edit `agents/config.json`:

```json
{
  "agents": {
    "medical": {
      "domain": "medical",
      "class": "MedicalAgent",
      "description": "Domain-specialized agent for medical diagnosis",
      "temperature": 0.3,
      "max_tokens": 768,
      "template": "You are a medical domain specialist..."
    }
  },
  "domains": {
    "medical": {
      "keywords": ["medical", "patient", "diagnosis", "treatment", ...],
      "agent_key": "medical",
      "instruction": "Focus on clinical reasoning..."
    }
  }
}
```

### Model Configuration

Edit `utils/llm_loader.py`:

```python
self.model_configs: Dict[str, Dict[str, str]] = {
    "medical-1b": {
        "endpoint": "llama-1b",
        "model": "medqa-lora",  # LoRA adapter name
    },
    "medical-8b": {
        "endpoint": "llama-8b",
        "model": "medqa-lora",
    },
    # Add more configurations...
}
```

### Docker GPU Assignment

Edit `docker/docker-compose.yml`:

```yaml
vllm-llama-1b:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['1']  # Change GPU ID here
            capabilities: [gpu]
```

---

## 🧪 Testing

### Comprehensive Integration Test

Run the full test suite:

```bash
./scripts/test.sh
```

**Test Coverage**:
- ✅ Agent Factory (domain configuration, creation, detection)
- ✅ Global Router (task decomposition, DAG generation)
- ✅ VLLM Loader (model configurations, endpoints)
- ✅ Agent SubRouter (model selection accuracy)
- ✅ Graph Builder (graph construction, execution)
- ✅ Result Handler (evaluation logic)
- ✅ End-to-End Orchestration (simple & complex tasks)

**Expected Output**:

```
======================================
LLM-Agent-Orchestrator Tests
======================================

Project root: /home/shjung/LLM-Agent-Orchestrator

✓ Virtual environment activated
Running integration tests...
Loaded 4 agent configs and 4 domain configs

================================================================================
LLM-AGENT-ORCHESTRATOR - COMPREHENSIVE INTEGRATION TEST
================================================================================

...

================================================================================
TEST SUMMARY
================================================================================
✅ PASS: Agent Factory - Domain Configuration
✅ PASS: Agent Factory - Agent Creation
✅ PASS: Global Router - Task Decomposition
✅ PASS: Global Router - Complex Tasks
✅ PASS: VLLM Loader - Model Configurations
✅ PASS: Agent SubRouter - Model Selection
✅ PASS: Graph Builder - Construction
✅ PASS: Graph Builder - Execution
✅ PASS: Result Handler - Evaluation
✅ PASS: End-to-End - Simple Task
✅ PASS: End-to-End - Complex Task

--------------------------------------------------------------------------------
TOTAL: 11/11 tests passed (100.0%)
================================================================================
```
---

## 📚 API Reference

### LLMOrchestrator

Main orchestration class.

```python
class LLMOrchestrator:
    def __init__(self, max_retry: int = 3):
        """
        Initialize orchestrator.
        
        Args:
            max_retry: Maximum refinement iterations
        """
    
    def process_task(self, task: str, user_id: str = "default") -> Dict:
        """
        Process user task through multi-agent system.
        
        Args:
            task: User task description
            user_id: User identifier
        
        Returns:
            {
                "success": bool,
                "final_answer": str,
                "iterations": int,
                "user_id": str,
                "original_task": str
            }
        """
```

### GlobalRouter

Task decomposition engine.

```python
class GlobalRouter:
    def decompose_task(
        self,
        task: str,
        feedback: Optional[str] = None,
        previous_results: Optional[str] = None
    ) -> Dict:
        """
        Decompose task into DAG structure.
        
        Returns:
            {
                "nodes": [
                    {"id": str, "domain": str, "task": str},
                    ...
                ],
                "edges": [
                    {"from": str, "to": str},
                    ...
                ]
            }
        """
```

### AgentFactory

Dynamic agent creation.

```python
class AgentFactory:
    def create_agent(self, domain: str) -> BaseAgent:
        """Create agent for specified domain."""
    
    def get_available_domains(self) -> List[str]:
        """Return list of available domains."""
    
    def detect_domain(self, task: str) -> str:
        """Detect domain from task description."""
```

### Router Service API

FastAPI endpoints for model selection.

**Technical Implementation**:
- Router models are trained with special tokens `[[1]]` (1B) and `[[2]]` (8B)
- Classification uses **logits-based inference** (not token generation):
  1. Model generates first token (may be any text like ": 1")
  2. Extract logits at first generation step for special tokens
  3. Apply softmax over `logits[vocab_size:vocab_size+2]`
  4. Decision: `probability([[2]]) > 0.5` → 8B, else → 1B
- This approach works even when model doesn't generate special tokens directly

#### Health Check

```bash
GET /health

Response:
{
  "status": "healthy",
  "loaded_routers": ["commonsense", "medical", "law", "math"],
  "total_routers": 4
}
```

#### List Routers

```bash
GET /routers

Response:
{
  "routers": [
    {
      "domain": "medical",
      "checkpoint": "/models/llama-1b-medqa-router",
      "vocab_size": 128258
    },
    ...
  ]
}
```

#### Route Task

```bash
POST /route/{domain}
Content-Type: application/json

{
  "task": "Analyze complex medical case",
  "context": {"patient_id": "12345"}
}

Response:
{
  "prediction": "8b",
  "probability": 0.892,
  "label": "[[2]]",
  "softmax_scores": [0.108, 0.892]
}
```

---

## 🛠️ Development

### Adding a New Domain

1. **Create LoRA Adapters**

Train LoRA adapters for 1B and 8B models on domain-specific datasets.

2. **Update Agent Configuration**

Edit `agents/config.json`:

```json
{
  "agents": {
    "finance": {
      "domain": "finance",
      "class": "FinanceAgent",
      "description": "Financial analysis and planning",
      "temperature": 0.3,
      "max_tokens": 512,
      "template": "You are a financial expert..."
    }
  },
  "domains": {
    "finance": {
      "keywords": ["finance", "investment", "stock", ...],
      "agent_key": "finance",
      "instruction": "Provide financial analysis..."
    }
  }
}
```

3. **Update Model Configuration**

Edit `utils/llm_loader.py`:

```python
"finance-1b": {
    "endpoint": "llama-1b",
    "model": "finance-lora",
},
"finance-8b": {
    "endpoint": "llama-8b",
    "model": "finance-lora",
}
```

4. **Train Router Model**

Train a router for the new domain to classify 1B vs 8B.

5. **Update Docker Compose**

Add LoRA adapter to `docker-compose.yml`:

```yaml
volumes:
  - /path/to/llama-1b-finance:/finance-adapter
command:
  - --lora-modules
  - finance-lora=/finance-adapter
```

6. **Update Router Service**

Add router checkpoint to router service volumes.

### Modifying Prompts

Edit prompt templates:

- **Global Router**: `utils/router_prompts.py`
- **Domain Agents**: `utils/agent_prompts.py`

### Adjusting Model Selection Threshold

Edit router service classification logic in `routers/router_service.py`:

```python
# Current: argmax selection
predicted_token = self.special_tokens[label_id]

# Alternative: threshold-based
if softmax_scores[1] > 0.75:  # High confidence for 8B
    predicted_token = self.special_tokens[1]
else:
    predicted_token = self.special_tokens[0]
```

### Logging and Debugging

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

1. **Batch Processing**: Modify `agent_subrouter.py` to batch requests
2. **Caching**: Implement result caching for repeated queries
3. **Load Balancing**: Add multiple vLLM instances with nginx
4. **Async Execution**: Convert synchronous calls to async

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{llm_agent_orchestrator,
  title={LLM-Agent-Orchestrator: Domain-Specialized Multi-Agent System},
  author={SincerityHun},
  year={2024},
  url={https://github.com/SincerityHun/LLM-Agent-Orchestrator}
}
```
---

## 📧 Contact

- **Author**: SincerityHun
- **GitHub**: [@SincerityHun](https://github.com/SincerityHun)
- **Issues**: [GitHub Issues](https://github.com/SincerityHun/LLM-Agent-Orchestrator/issues)

