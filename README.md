## Goal
Our multi-agent LLM system decomposes tasks into role-specific subtasks and assigns them to small, specialized models.
This structure reduces hallucination and improves accuracy through cross-verification between agents.
Because each subtask is handled by smaller LLMs, the system requires significantly less compute and memory compared to using a single large model, at the cost of only moderate additional latency.

## Motivation
Limitation of using a single general LLM model
1. Domain Mismatch
    - General LLMs lack domain alignment, often producing incorrect or unsafe outputs. (e.g. healthcare, finance, robotics)

2. Task Complexity
    - Complex tasks may require diverse skills and knowledge that a single model cannot provide effectively.

3. Computational Efficiency
    - Specialized models can be more efficient for specific tasks, reducing latency and resource consumption.

## Architecture Diagram
```markdown

                            ┌──────────────────────┐
                            │     Client Input     │
                            └──────────┬───────────┘
                                       │
                                       ↓
                            ┌──────────────────────┐
                            │  Global Router LLM   │
                            │ ─ split subtasks     │
                            │ ─ assign to agents   │
                            └──────────┬───────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ↓                        ↓                        ↓
    ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
    │  Agent 1 (Role R1)│    │  Agent 2 (Role R2)│    │  Agent 3 (Role R3)│
    │ e.g. Planning     │    │ e.g. Execution    │    │ e.g. Review       │
    └─────────┬─────────┘    └─────────┬─────────┘    └─────────┬─────────┘
              │                        │                        │
              ↓                        ↓                        ↓
    ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
    │ SubRouter LLM 1   │    │ SubRouter LLM 2   │    │ SubRouter LLM 3   │
    │ ─ choose model    │    │ ─ choose model    │    │ ─ choose model    │
    │ ─ role-specific   │    │ ─ role-specific   │    │ ─ role-specific   │
    └──────────┬────────┘    └──────────┬────────┘    └──────────┬────────┘
               │                        │                        │
               │ select model           │ select model           │ select model
               │                        │                        │
               │        ┌────────────────────────────────────────────────────────┐
               │        │   Global LLM Tool Pool (shared models)                 │
               │        │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
               └───────→│  │ Model A │  │ Model B │  │ Model C │  │ Model D │... │
                        │  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │
                        └───────┬────────────┬──────────────┬────────────────────┘
                                │            │              │            
        run model for Agent 1   │            │              │            
                                ↓            ↓              ↓            
                           ┌─────────┐   ┌─────────┐   ┌─────────┐
                           │Output 1 │   │Output 2 │   │Output 3 │
                           │From A1  │   │From A2  │   │From A3  │
                           └────┬────┘   └────┬────┘   └────┬────┘
                                │             │             │
                                └─────────────┼─────────────┘
                                              │
                                              ↓
                             ┌──────────────────────────────────┐
                             │      Merge Subtask Results       │
                             │ (Output 1 + Output 2 + Output 3) │
                             └────────────────┬─────────────────┘
                                              │
                                              ↓
                                 ┌─────────────────────────┐
                                 │   Results Handling LLM  │
                                 │ ─ evaluate merged result│
                                 │ ─ final answer OR       │
                                 │   generate new subtasks │
                                 └────────────┬────────────┘
                                              │
                        ┌─────────────────────┼─────────────────────┐
                        │                                           │
                        ↓                                           ↓
               ┌─────────────────────┐                   ┌──────────────────────┐
               │    Final Answer     │                   │ New / Refined Task   │
               └─────────────────────┘                   └──────────┬───────────┘
                                                                    │
                                                                    ↓
                                                         ┌─────────────────────────┐
                                                         │     Global Router LLM   │
                                                         │     (refinement loop)   │
                                                         └─────────────────────────┘

```

## Components Description
1. Global Router LLM

2. SubRouter LLMs for each Agent

3. Global LLM Tool Pool (shared models)
    - Current Set
        1. CommonsenseQA (상식추론)
        2. CaseHOLD (법률)
        3. MedQA-USMLE (의학)
        4. Mathqa (수학)


4. Results Handling LLM


## Evaluation Metrics
1. Quality Metrics
    a) Task Accuracy / Success Rate

    b) Hallucination Rate


2. Latency Metrics
    a) End-to-End Latency


3. Resource / Cost Metrics
    a) Peak GPU Memory per Request (GPU VRAM Usage)

    b) GPU Time per Request (Compute Time)

    c) Token Cost per Request (Monetary Cost)


4. Ratio Metrics (Efficiency Metrics)
    a) Accuracy per GPU-Second

    b) Accuracy per 1K Tokens