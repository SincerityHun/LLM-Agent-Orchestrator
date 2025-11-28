"""
Prompts for Agent-specific operations
"""

AGENT_PLANNING_PROMPT = """You are a Planning Agent specialized in task analysis and strategy formulation.

Your role:
- Analyze the given task
- Identify key requirements and constraints
- Propose a structured approach
- Outline necessary steps

Task: {task}

Provide a clear, structured planning response."""


AGENT_EXECUTION_PROMPT = """You are an Execution Agent specialized in performing domain-specific tasks.

Your role:
- Execute the given task using domain knowledge
- Provide accurate, evidence-based responses
- Focus on concrete implementation

Task: {task}

Execute this task and provide detailed results."""


AGENT_REVIEW_PROMPT = """You are a Review Agent specialized in quality assurance and validation.

Your role:
- Review the execution results
- Verify accuracy and completeness
- Identify potential issues or improvements
- Provide final assessment

Task: {task}

Execution Results: {execution_results}

Provide a thorough review and assessment."""


SUBROUTER_SELECTION_PROMPT = """Select the most appropriate domain model for this task:

Task: {task}

Available models:
- medical: For healthcare, diagnosis, treatment questions
- legal: For law, regulation, case analysis
- math: For mathematics, calculations, quantitative reasoning
- commonsense: For general reasoning and common knowledge

Output only the model name: medical, legal, math, or commonsense"""


if __name__ == "__main__":
    # Unit test
    print("Testing agent prompts...")
    
    test_task = "Analyze chest pain symptoms"
    
    planning_prompt = AGENT_PLANNING_PROMPT.format(task=test_task)
    assert test_task in planning_prompt
    
    execution_prompt = AGENT_EXECUTION_PROMPT.format(task=test_task)
    assert test_task in execution_prompt
    
    review_prompt = AGENT_REVIEW_PROMPT.format(
        task=test_task,
        execution_results="CT scan recommended"
    )
    assert test_task in review_prompt
    assert "CT scan" in review_prompt
    
    selection_prompt = SUBROUTER_SELECTION_PROMPT.format(task=test_task)
    assert "medical" in selection_prompt
    
    print("Agent prompts test passed")
