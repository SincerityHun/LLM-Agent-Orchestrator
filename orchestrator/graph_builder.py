"""
Graph Builder - Converts DAG into LangGraph executable workflow
"""
from typing import Dict, List, Any
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from agents.agent_planning import PlanningAgent
from agents.agent_execution import ExecutionAgent
from agents.agent_review import ReviewAgent


class WorkflowState(TypedDict):
    """State maintained throughout workflow execution"""
    original_task: str
    current_node: str
    results: Dict[str, Any]
    context: Dict[str, Any]


class GraphBuilder:
    """
    Builds and executes LangGraph workflow from DAG specification
    """
    
    def __init__(self):
        self.agents = {
            "planning": PlanningAgent(),
            "execution": ExecutionAgent(),
            "review": ReviewAgent()
        }
    
    def build_graph(self, dag: Dict) -> StateGraph:
        """
        Build LangGraph from DAG specification
        
        Args:
            dag: Dictionary with 'nodes' and 'edges'
            
        Returns:
            Compiled LangGraph workflow
        """
        # Create state graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        nodes = dag.get("nodes", [])
        for node in nodes:
            node_id = node["id"]
            role = node["role"]
            task = node["task"]
            
            # Create node function
            node_func = self._create_node_function(node_id, role, task)
            workflow.add_node(node_id, node_func)
        
        # Add edges (dependencies)
        edges = dag.get("edges", [])
        for edge in edges:
            workflow.add_edge(edge["from"], edge["to"])
        
        # Set entry point (first node with no incoming edges)
        entry_node = self._find_entry_node(nodes, edges)
        workflow.set_entry_point(entry_node)
        
        # Set finish point (last node with no outgoing edges)
        finish_node = self._find_finish_node(nodes, edges)
        workflow.set_finish_point(finish_node)
        
        return workflow.compile()
    
    def execute_graph(self, graph: Any, initial_state: WorkflowState) -> Dict:
        """
        Execute compiled LangGraph workflow
        
        Args:
            graph: Compiled LangGraph
            initial_state: Initial workflow state
            
        Returns:
            Final state with all results
        """
        # Run graph
        final_state = graph.invoke(initial_state)
        return final_state
    
    def _create_node_function(self, node_id: str, role: str, task: str):
        """
        Create execution function for a graph node
        """
        def node_function(state: WorkflowState) -> WorkflowState:
            # Get appropriate agent
            agent = self.agents.get(role)
            if not agent:
                raise ValueError(f"Unknown agent role: {role}")
            
            # Build context from previous results
            context = {**state.get("context", {})}
            for prev_id, prev_result in state.get("results", {}).items():
                context[prev_id] = prev_result.get("result", "")
            
            # Execute agent task
            result = agent.execute(task, context=context)
            
            # Update state
            new_results = {**state.get("results", {})}
            new_results[node_id] = result
            
            return {
                **state,
                "current_node": node_id,
                "results": new_results
            }
        
        return node_function
    
    def _find_entry_node(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """
        Find entry node (no incoming edges)
        """
        node_ids = {node["id"] for node in nodes}
        target_nodes = {edge["to"] for edge in edges}
        entry_nodes = node_ids - target_nodes
        
        if not entry_nodes:
            return nodes[0]["id"]  # Fallback to first node
        
        return list(entry_nodes)[0]
    
    def _find_finish_node(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """
        Find finish node (no outgoing edges)
        """
        node_ids = {node["id"] for node in nodes}
        source_nodes = {edge["from"] for edge in edges}
        finish_nodes = node_ids - source_nodes
        
        if not finish_nodes:
            return nodes[-1]["id"]  # Fallback to last node
        
        return list(finish_nodes)[0]


if __name__ == "__main__":
    # Unit test
    print("Testing GraphBuilder...")
    
    builder = GraphBuilder()
    
    # Test DAG
    test_dag = {
        "nodes": [
            {"id": "plan1", "role": "planning", "task": "Create plan"},
            {"id": "exec1", "role": "execution", "task": "Execute task"},
            {"id": "review1", "role": "review", "task": "Review results"}
        ],
        "edges": [
            {"from": "plan1", "to": "exec1"},
            {"from": "exec1", "to": "review1"}
        ]
    }
    
    # Build graph
    graph = builder.build_graph(test_dag)
    assert graph is not None
    
    # Test execution
    initial_state = {
        "original_task": "Test task",
        "current_node": "",
        "results": {},
        "context": {}
    }
    
    final_state = builder.execute_graph(graph, initial_state)
    assert "results" in final_state
    assert len(final_state["results"]) == 3
    
    print("GraphBuilder test passed")
