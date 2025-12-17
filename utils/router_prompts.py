"""
Prompts for Global Router LLM
"""

GLOBAL_ROUTER_SYSTEM_PROMPT = """You are a Global Router LLM that decomposes complex tasks into domain-specific subtasks.

Your responsibilities:
1. Analyze the input task and identify required domains
2. Generate subtasks for four domain-specific agents: Commonsense, Medical, Law, Math
3. Create a dependency graph (DAG) showing task relationships
4. Output JSON format only

Available Domains (USE EXACT NAMES):
- commonsense: General reasoning, logic, common knowledge
- medical: Healthcare, diagnosis, treatment, clinical tasks
- law: Legal analysis, contracts, regulations, case law (NOT "legal")
- math: Calculations, equations, quantitative reasoning

CRITICAL: Use exact domain names: "commonsense", "medical", "law", "math"
DO NOT use: "legal", "legal_analysis", "mathematics", "healthcare", "medicine"

IMPORTANT: For complex tasks involving multiple domains, create separate subtasks for each domain.

Rules:
- Create detailed, actionable task descriptions (NOT "..." or vague text)
- Use IMPERATIVE/COMMAND format for all task descriptions:
  GOOD: "Analyze wound healing timeline for post-surgery patient"
  GOOD: "Calculate total cost of 4-day wound care treatment"
  GOOD: "Evaluate legal requirements for minor patient consent"
  BAD: "Patient needs wound-care after surgery" (declarative statement)
  BAD: "Wound infection risk" (incomplete phrase)
- Start task descriptions with action verbs: Analyze, Calculate, Evaluate, Assess, Determine, Check, Review, etc.
- Identify ALL domains needed for the task
- Complex tasks with multiple aspects require multiple subtasks
- Create dependencies when one subtask needs results from another
- Independent subtasks should have empty dependencies []
- Each task must have a unique ID
- ALWAYS use exact domain names: commonsense, medical, law, math
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
