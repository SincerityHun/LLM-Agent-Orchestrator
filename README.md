## Architecture Diagram
```
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
 └──────────┬────────┘    └──────────┬────────┘    └──────────┬────────┘
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
                     │                     │                     │
                     ↓                     ↓                     ↓
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

4. Results Handling LLM