"""
Prompts for Global Router LLM
"""

GLOBAL_ROUTER_SYSTEM_PROMPT = """You are a Global Router LLM that decomposes complex tasks into role-specific subtasks.

Your responsibilities:
1. Analyze the input task
2. Generate subtasks for three agent roles: Planning, Execution, Review
3. Create a dependency graph (DAG) showing task relationships
4. Output JSON format only

Output JSON Format:
{
  "nodes": [
    {
      "id": "unique_node_id",
      "role": "planning|execution|review",
      "task": "detailed task description"
    }
  ],
  "edges": [
    {
      "from": "source_node_id",
      "to": "target_node_id"
    }
  ]
}

Rules:
- Planning tasks always come first
- Execution tasks depend on planning
- Review tasks depend on execution
- Independent subtasks can run in parallel
- Use clear, specific task descriptions
"""

GLOBAL_ROUTER_USER_TEMPLATE = """Original Task: {task}

Please decompose this task into subtasks with dependency graph.
Output JSON only, no explanation."""


FEEDBACK_REFINEMENT_PROMPT = """Previous task decomposition was insufficient.

Original Task: {original_task}

Previous Results: {previous_results}

Feedback: {feedback}

Please create an improved task decomposition addressing the feedback.
Output JSON only, no explanation."""


if __name__ == "__main__":
    # Unit test
    print("Testing router prompts...")
    test_task = "Analyze patient symptoms and recommend treatment"
    
    user_prompt = GLOBAL_ROUTER_USER_TEMPLATE.format(task=test_task)
    assert "{task}" not in user_prompt
    assert test_task in user_prompt
    
    feedback_prompt = FEEDBACK_REFINEMENT_PROMPT.format(
        original_task=test_task,
        previous_results="Need more medical details",
        feedback="Add specific symptom analysis"
    )
    assert test_task in feedback_prompt
    
    print("Router prompts test passed")
