"""
Graph Builder - Converts DAG into LangGraph executable workflow
"""
from typing import Dict, List, Any, Annotated
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from agents.agent_factory import AgentFactory


def merge_results(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two results dictionaries (for parallel execution)"""
    return {**left, **right}


class WorkflowState(TypedDict):
    """State maintained throughout workflow execution"""
    original_task: str
    current_node: str
    results: Annotated[Dict[str, Any], merge_results]  # Allow parallel updates
    context: Dict[str, Any]


class GraphBuilder:
    """
    Builds and executes LangGraph workflow from DAG specification
    """
    
    def __init__(self):
        # Use AgentFactory to create domain-specific agents dynamically
        factory = AgentFactory()
        self.agents = {}
        for domain in factory.get_available_domains():
            self.agents[domain] = factory.create_agent(domain)
    
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
            domain = node["domain"]
            task = node["task"]
            
            # Create node function
            node_func = self._create_node_function(node_id, domain, task)
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
    
    def _create_node_function(self, node_id: str, domain: str, task: str):
        """
        Create execution function for a graph node
        """
        def node_function(state: WorkflowState) -> Dict[str, Any]:
            # Get appropriate agent by domain
            agent = self.agents.get(domain)
            if not agent:
                raise ValueError(f"Unknown agent domain: {domain}")
            
            # Build context from previous results
            context = {**state.get("context", {})}
            for prev_id, prev_result in state.get("results", {}).items():
                context[prev_id] = prev_result.get("result", "")
            
            # Execute agent task
            result = agent.execute(task, context=context)
            
            # Update state - only return results (safe for parallel execution)
            new_results = {**state.get("results", {})}
            new_results[node_id] = result
            
            # Return only results dict - each node updates a different key
            # This allows parallel execution without conflicts
            return {
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
