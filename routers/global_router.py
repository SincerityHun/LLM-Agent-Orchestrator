"""
Global Router LLM for task decomposition and DAG generation
"""
import json
from typing import Dict, Optional
from utils.router_prompts import (
    GLOBAL_ROUTER_SYSTEM_PROMPT,
    GLOBAL_ROUTER_USER_TEMPLATE,
    FEEDBACK_REFINEMENT_PROMPT
)
from utils.llm_loader import VLLMLoader


class GlobalRouter:
    """
    Global Router decomposes tasks into subtasks with dependency graph
    """
    
    def __init__(self):
        self.llm_loader = VLLMLoader()
    
    def decompose_task(
        self,
        task: str,
        feedback: Optional[str] = None,
        previous_results: Optional[str] = None
    ) -> Dict:
        """
        Decompose task into subtasks with DAG
        
        Args:
            task: Original task description
            feedback: Optional feedback from previous iteration
            previous_results: Optional results from previous iteration
            
        Returns:
            Dictionary with 'nodes' and 'edges' representing DAG
        """
        # Build prompt based on whether this is refinement or initial decomposition
        if feedback and previous_results:
            user_prompt = FEEDBACK_REFINEMENT_PROMPT.format(
                original_task=task,
                previous_results=previous_results,
                feedback=feedback
            )
        else:
            user_prompt = GLOBAL_ROUTER_USER_TEMPLATE.format(task=task)
        
        # Combine system and user prompts
        full_prompt = f"{GLOBAL_ROUTER_SYSTEM_PROMPT}\n\n{user_prompt}"
        
        # Call LLM (using commonsense model for routing decisions)
        response = self.llm_loader.call_model(
            model_type="commonsense",
            prompt=full_prompt,
            max_tokens=1024,
            temperature=0.3
        )
        
        # Parse JSON response
        try:
            graph = json.loads(response)
            return self._validate_graph(graph)
        except json.JSONDecodeError:
            # Fallback: create basic sequential graph
            return self._create_default_graph(task)
    
    def _validate_graph(self, graph: Dict) -> Dict:
        """
        Validate and sanitize graph structure
        """
        if "nodes" not in graph or "edges" not in graph:
            raise ValueError("Graph must contain 'nodes' and 'edges'")
        
        # Ensure all nodes have required fields
        for node in graph["nodes"]:
            if "id" not in node or "role" not in node or "task" not in node:
                raise ValueError("Each node must have 'id', 'role', and 'task'")
        
        # Validate edges reference existing nodes
        node_ids = {node["id"] for node in graph["nodes"]}
        for edge in graph["edges"]:
            if edge["from"] not in node_ids or edge["to"] not in node_ids:
                raise ValueError(f"Edge references non-existent node: {edge}")
        
        return graph
    
    def _create_default_graph(self, task: str) -> Dict:
        """
        Create default sequential graph when parsing fails
        """
        return {
            "nodes": [
                {
                    "id": "planning",
                    "role": "planning",
                    "task": f"Create a plan for: {task}"
                },
                {
                    "id": "execution",
                    "role": "execution",
                    "task": f"Execute: {task}"
                },
                {
                    "id": "review",
                    "role": "review",
                    "task": f"Review the execution of: {task}"
                }
            ],
            "edges": [
                {"from": "planning", "to": "execution"},
                {"from": "execution", "to": "review"}
            ]
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
