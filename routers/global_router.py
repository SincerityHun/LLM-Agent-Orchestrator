"""
Global Router LLM for task decomposition and DAG generation
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

from utils.router_prompts import (
    GLOBAL_ROUTER_SYSTEM_PROMPT,
    GLOBAL_ROUTER_USER_TEMPLATE,
    FEEDBACK_REFINEMENT_PROMPT,
)
from utils.llm_loader import VLLMLoader


class SubTask(BaseModel):
    """Single unit of work produced by the global router"""

    id: str
    role: str
    content: str
    dependencies: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_role(cls, values: "SubTask") -> "SubTask":
        allowed_roles = {"planning", "execution", "review"}
        if values.role not in allowed_roles:
            raise ValueError(
                f"Invalid role '{values.role}'. Allowed roles: {sorted(allowed_roles)}"
            )
        if values.id in values.dependencies:
            raise ValueError(f"Task '{values.id}' cannot depend on itself")
        return values


class TaskDAG(BaseModel):
    """Schema representing the full DAG returned by the router"""

    tasks: List[SubTask]

    @model_validator(mode="after")
    def validate_graph(cls, values: "TaskDAG") -> "TaskDAG":
        task_ids = [task.id for task in values.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Task IDs must be unique")

        valid_ids = set(task_ids)
        for task in values.tasks:
            unknown = [dep for dep in task.dependencies if dep not in valid_ids]
            if unknown:
                raise ValueError(
                    f"Task '{task.id}' references unknown dependencies: {unknown}"
                )
        return values


class GlobalRouter:
    """Global Router decomposes tasks into subtasks with dependency graph"""

    MAX_RETRY = 3

    def __init__(self) -> None:
        self.llm_loader = VLLMLoader()

    def decompose_task(
        self,
        task: str,
        feedback: Optional[str] = None,
        previous_results: Optional[str] = None,
    ) -> Dict:
        """Decompose task into DAG structure consumed by downstream components"""

        print(f"\n{'=' * 80}")
        print("GLOBAL ROUTER - TASK DECOMPOSITION")
        print(f"{'=' * 80}")
        print(f"Original Task: {task}")

        if feedback:
            print(f"Feedback: {feedback}")
        if previous_results:
            print(f"Previous Results: {previous_results}")

        dag_model = self.create_dag(task, feedback, previous_results)
        if dag_model is None:
            print("Using default sequential graph due to validation failures")
            fallback = self._create_default_graph(task)
            self._log_graph(fallback)
            return fallback

        graph = self._taskdag_to_graph(dag_model)
        self._log_graph(graph)
        return graph

    def create_dag(
        self,
        task: str,
        feedback: Optional[str],
        previous_results: Optional[str],
    ) -> Optional[TaskDAG]:
        """Call the LLM with guided JSON to produce a valid TaskDAG"""

        # Get JSON schema for guided decoding
        json_schema = TaskDAG.model_json_schema()
        
        validation_errors: List[str] = []

        for attempt in range(1, self.MAX_RETRY + 1):
            prompt = self._build_structured_prompt(
                task=task,
                feedback=feedback,
                previous_results=previous_results,
                errors=validation_errors,
            )

            print(f"\nGlobal Router Prompt (attempt {attempt}):")
            print(f"{'-' * 80}")
            print(prompt)
            print(f"{'-' * 80}")
            print(f"Using guided JSON schema: {json_schema}")

            raw_response = self.llm_loader.call_model(
                model_type="global-router",
                prompt=prompt,
                max_tokens=1024,
                temperature=0.2,
                guided_json=json_schema,
            )

            print(f"\nGlobal Router Response (attempt {attempt}):")
            print(f"{'-' * 80}")
            print(raw_response)
            print(f"{'-' * 80}")

            try:
                # Clean the response even with guided JSON (LLM might still add markdown)
                cleaned_response = self._clean_json_string(raw_response)
                return TaskDAG.model_validate_json(cleaned_response)
            except ValidationError as err:
                error_message = (
                    f"Attempt {attempt} validation error: {err}"  # includes cause summary
                )
                print(error_message)
                validation_errors.append(error_message)

        return None

    def _build_structured_prompt(
        self,
        task: str,
        feedback: Optional[str],
        previous_results: Optional[str],
        errors: List[str],
    ) -> str:
        """Compose router prompt for guided JSON generation"""

        if feedback and previous_results:
            user_prompt = FEEDBACK_REFINEMENT_PROMPT.format(
                original_task=task,
                previous_results=previous_results,
                feedback=feedback,
            )
        else:
            user_prompt = GLOBAL_ROUTER_USER_TEMPLATE.format(task=task)

        # Clear instruction for TaskDAG schema format
        schema_instruction = (
            "Generate a task decomposition in JSON format with 'tasks' array. "
            "Each task should have: id (string), role (planning/execution/review), "
            "content (task description), and dependencies (array of task IDs). "
            "Example: {\"tasks\": [{\"id\": \"task1\", \"role\": \"planning\", \"content\": \"...\", \"dependencies\": []}]}"
        )

        error_block = ""
        if errors:
            formatted_errors = "\n".join(f"- {msg}" for msg in errors[-2:])  # Show fewer errors
            error_block = (
                "\nPrevious attempts had validation issues:\n"
                f"{formatted_errors}\n"
                "Please ensure task dependencies reference valid task IDs."
            )

        return (
            f"{GLOBAL_ROUTER_SYSTEM_PROMPT}\n\n"
            f"{user_prompt}\n\n"
            f"{schema_instruction}{error_block}"
        )
    
    def _build_prompt(
        self,
        task: str,
        feedback: Optional[str],
        previous_results: Optional[str],
        errors: List[str],
    ) -> str:
        """Legacy prompt method - kept for backward compatibility"""
        return self._build_structured_prompt(task, feedback, previous_results, errors)

    def _clean_json_string(self, response: str) -> str:
        """Strip markdown code fences and whitespace from LLM response"""
        import re
        
        cleaned = response.strip()
        
        # Remove markdown code blocks
        if "```" in cleaned:
            # Extract JSON content between ```json and ``` or between ``` and ```
            json_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
            match = re.search(json_pattern, cleaned)
            if match:
                cleaned = match.group(1).strip()
            else:
                # Fallback: remove all ``` lines
                lines = cleaned.splitlines()
                lines = [line for line in lines if not line.strip().startswith('```')]
                cleaned = '\n'.join(lines)
        
        return cleaned.strip()

    def _taskdag_to_graph(self, dag: TaskDAG) -> Dict:
        """Convert validated TaskDAG into legacy graph format"""

        nodes = []
        edges = []

        for task in dag.tasks:
            nodes.append({"id": task.id, "role": task.role, "task": task.content})
            for dependency in task.dependencies:
                edges.append({"from": dependency, "to": task.id})

        return {"nodes": nodes, "edges": edges}

    def _log_graph(self, graph: Dict) -> None:
        """Log resulting graph for observability"""

        print(f"\nGenerated DAG:")
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        print(f"Nodes ({len(nodes)}):")
        for node in nodes:
            print(f"   • {node['id']} ({node['role']}): {node['task']}")

        print(f"Edges ({len(edges)}):")
        for edge in edges:
            print(f"   • {edge['from']} → {edge['to']}")
        print(f"{'=' * 80}\n")

    def _create_default_graph(self, task: str) -> Dict:
        """Fallback to sequential workflow when parsing fails"""

        return {
            "nodes": [
                {
                    "id": "planning",
                    "role": "planning",
                    "task": f"Create a plan for: {task}",
                },
                {
                    "id": "execution",
                    "role": "execution",
                    "task": f"Execute: {task}",
                },
                {
                    "id": "review",
                    "role": "review",
                    "task": f"Review the execution of: {task}",
                },
            ],
            "edges": [
                {"from": "planning", "to": "execution"},
                {"from": "execution", "to": "review"},
            ],
        }


if __name__ == "__main__":
    # Unit test
    print("Testing GlobalRouter...")
    
    router = GlobalRouter()
    
    # Test basic task decomposition
    test_task = "Analyze patient with chest pain and recommend next steps"
    graph = router.decompose_task(test_task)
    
    assert "nodes" in graph
    assert "edges" in graph
    assert len(graph["nodes"]) >= 3  # At least planning, execution, review
    
    # Check node structure
    for node in graph["nodes"]:
        assert "id" in node
        assert "role" in node
        assert "task" in node
    
    # Test refinement with feedback
    feedback_graph = router.decompose_task(
        task=test_task,
        feedback="Need more specific medical analysis",
        previous_results="Basic plan created"
    )
    assert "nodes" in feedback_graph
    
    print("GlobalRouter test passed")
