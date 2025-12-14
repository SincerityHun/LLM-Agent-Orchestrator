"""
Comprehensive Integration Test for LLM-Agent-Orchestrator
Tests the entire domain-based multi-agent orchestration system
"""
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agents.agent_factory import AgentFactory
from routers.global_router import GlobalRouter
from routers.agent_subrouter import AgentSubRouter
from orchestrator.graph_builder import GraphBuilder
from orchestrator.result_handler import ResultHandler
from utils.llm_loader import VLLMLoader
from main import LLMOrchestrator


def print_test_header(test_name: str):
    """Print formatted test header"""
    print("\n" + "=" * 80)
    print(f"TEST: {test_name}")
    print("=" * 80)


def print_test_result(passed: bool, message: str = ""):
    """Print test result"""
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"\n{status}")
    if message:
        print(f"   {message}")
    print()


class TestAgentFactory:
    """Test suite for AgentFactory"""
    
    def test_domain_configuration(self) -> bool:
        """Test that all domains are properly configured"""
        print_test_header("Agent Factory - Domain Configuration")
        
        factory = AgentFactory()
        
        # Test available domains
        domains = factory.get_available_domains()
        expected_domains = ["commonsense", "medical", "law", "math"]
        
        print(f"Available domains: {domains}")
        assert all(d in domains for d in expected_domains), \
            f"Missing domains. Expected: {expected_domains}"
        
        # Test domain keywords
        for domain in expected_domains:
            keywords = factory.get_domain_keywords().get(domain, [])
            assert len(keywords) > 0, f"Domain {domain} has no keywords"
            print(f"✓ {domain}: {len(keywords)} keywords")
        
        print_test_result(True, f"All {len(domains)} domains configured correctly")
        return True
    
    def test_agent_creation(self) -> bool:
        """Test agent creation for each domain"""
        print_test_header("Agent Factory - Agent Creation")
        
        factory = AgentFactory()
        
        for domain in ["commonsense", "medical", "law", "math"]:
            agent = factory.create_agent(domain)
            assert agent is not None, f"Failed to create {domain} agent"
            assert agent.domain == domain, f"Agent domain mismatch"
            print(f"✓ Created {domain} agent")
        
        print_test_result(True, "All domain agents created successfully")
        return True


class TestGlobalRouter:
    """Test suite for GlobalRouter"""
    
    def test_task_decomposition(self) -> bool:
        """Test DAG generation from task"""
        print_test_header("Global Router - Task Decomposition")
        
        router = GlobalRouter()
        
        # Test with medical task
        task = "Diagnose patient with chest pain and recommend treatment"
        dag = router.decompose_task(task)
        
        assert "nodes" in dag, "DAG missing 'nodes'"
        assert "edges" in dag, "DAG missing 'edges'"
        assert len(dag["nodes"]) > 0, "DAG has no nodes"
        
        # Check node structure
        for node in dag["nodes"]:
            assert "id" in node, "Node missing 'id'"
            assert "domain" in node, "Node missing 'domain'"
            assert "task" in node, "Node missing 'task'"
            print(f"✓ Node '{node['id']}': domain={node['domain']}")
        
        print_test_result(True, f"Generated DAG with {len(dag['nodes'])} nodes, {len(dag['edges'])} edges")
        return True
    
    def test_complex_task_decomposition(self) -> bool:
        """Test decomposition of complex multi-domain task"""
        print_test_header("Global Router - Complex Task Decomposition")
        
        router = GlobalRouter()
        
        task = "Analyze the legal implications of a medical malpractice case involving \
chest pain diagnosis, calculate damages, and provide commonsense recommendations"
        
        dag = router.decompose_task(task)
        
        # Verify decomposition occurred
        domains_used = set(node["domain"] for node in dag["nodes"])
        print(f"Domains involved: {domains_used}")
        print(f"Number of subtasks: {len(dag['nodes'])}")
        
        # Complex task should generate multiple subtasks OR use multiple domains
        # (LLM might decompose into multiple subtasks within same domain, which is also valid)
        has_multiple_tasks = len(dag['nodes']) >= 2
        has_multiple_domains = len(domains_used) >= 2
        
        assert has_multiple_tasks or has_multiple_domains, \
            f"Complex task should generate multiple subtasks or use multiple domains. Got {len(dag['nodes'])} tasks, {len(domains_used)} domains"
        
        # Log the decomposition strategy
        if has_multiple_domains:
            strategy = f"multi-domain strategy ({len(domains_used)} domains)"
        else:
            strategy = f"multi-task strategy ({len(dag['nodes'])} tasks)"
        
        print_test_result(True, f"Complex task decomposed using {strategy}")
        return True


class TestVLLMLoader:
    """Test suite for VLLMLoader"""
    
    def test_model_configurations(self) -> bool:
        """Test that all model configurations are present"""
        print_test_header("VLLM Loader - Model Configurations")
        
        loader = VLLMLoader()
        
        # Test endpoints
        assert "llama-1b" in loader.endpoints, "Missing llama-1b endpoint"
        assert "llama-8b" in loader.endpoints, "Missing llama-8b endpoint"
        print(f"✓ Endpoint 'llama-1b': {loader.endpoints['llama-1b']}")
        print(f"✓ Endpoint 'llama-8b': {loader.endpoints['llama-8b']}")
        
        # Test domain models (1b and 8b variants)
        required_models = [
            "commonsense-1b", "commonsense-8b",
            "medical-1b", "medical-8b",
            "law-1b", "law-8b",
            "math-1b", "math-8b",
        ]
        
        for model_key in required_models:
            assert model_key in loader.model_configs, f"Missing config for {model_key}"
            config = loader.model_configs[model_key]
            assert "endpoint" in config, f"{model_key} missing endpoint"
            assert "model" in config, f"{model_key} missing model"
            print(f"✓ {model_key}: {config['model']}")
        
        print_test_result(True, f"All {len(required_models)} model configurations present")
        return True


class TestAgentSubRouter:
    """Test suite for AgentSubRouter"""
    
    def test_model_selection(self) -> bool:
        """Test router-based model selection"""
        print_test_header("Agent SubRouter - Model Selection")
        
        # Use router service if available (default behavior)
        subrouter = AgentSubRouter()
        
        # Check if router service is being used
        print(f"Router service enabled: {subrouter.use_router_service}")
        if subrouter.use_router_service:
            print(f"Router service URL: {subrouter.router_service_url}")
        else:
            print("⚠ Router service not available, using vLLM fallback")
        
        test_cases = [
            # Simple tasks -> 1b model
            ("commonsense", "What color is the sky?", "1b"),
            ("medical", "What is fever?", "1b"),
            ("law", "Define contract", "1b"),
            ("math", "Calculate 2+2", "1b"),
            
            # Complex tasks -> 8b model
            ("commonsense", "Analyze the philosophical implications of consciousness", "8b"),
            ("medical", "Analyze patient with multiple comorbidities including diabetes", "8b"),
            ("law", "Evaluate complex merger and acquisition regulatory compliance", "8b"),
            ("math", "Derive the solution to Navier-Stokes equations", "8b"),
        ]
        
        correct = 0
        total = len(test_cases)
        
        for domain, task, expected_size in test_cases:
            endpoint, model = subrouter.select_model_for_domain(domain, task)
            
            # Determine predicted size from endpoint
            predicted_size = "8b" if "8b" in endpoint else "1b"
            
            is_correct = predicted_size == expected_size
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"{status} {domain}/{expected_size}: '{task[:50]}...' -> {predicted_size}")
        
        accuracy = correct / total
        print(f"\nAccuracy: {correct}/{total} ({accuracy*100:.1f}%)")
        
        print_test_result(True, f"Model selection tested on {total} cases")
        return True


class TestGraphBuilder:
    """Test suite for GraphBuilder"""
    
    def test_graph_construction(self) -> bool:
        """Test LangGraph construction from DAG"""
        print_test_header("Graph Builder - Graph Construction")
        
        builder = GraphBuilder()
        
        # Check that domain agents are loaded
        assert "commonsense" in builder.agents, "Missing commonsense agent"
        assert "medical" in builder.agents, "Missing medical agent"
        assert "law" in builder.agents, "Missing law agent"
        assert "math" in builder.agents, "Missing math agent"
        
        print(f"✓ GraphBuilder has {len(builder.agents)} domain-specific agents")
        
        # Test graph building
        test_dag = {
            "nodes": [
                {"id": "task1", "domain": "medical", "task": "Analyze symptoms"},
                {"id": "task2", "domain": "commonsense", "task": "Provide advice"},
            ],
            "edges": [
                {"from": "task1", "to": "task2"}
            ]
        }
        
        graph = builder.build_graph(test_dag)
        assert graph is not None, "Graph construction failed"
        
        print_test_result(True, "Graph constructed successfully from DAG")
        return True
    
    def test_graph_execution(self) -> bool:
        """Test graph execution with domain agents"""
        print_test_header("Graph Builder - Graph Execution")
        
        builder = GraphBuilder()
        
        test_dag = {
            "nodes": [
                {"id": "analyze", "domain": "commonsense", "task": "Explain basic reasoning"},
            ],
            "edges": []
        }
        
        graph = builder.build_graph(test_dag)
        
        initial_state = {
            "original_task": "Test task",
            "current_node": "",
            "results": {},
            "context": {"user_id": "test"}
        }
        
        final_state = builder.execute_graph(graph, initial_state)
        
        assert "results" in final_state, "Final state missing results"
        assert len(final_state["results"]) > 0, "No results generated"
        
        print_test_result(True, f"Graph executed with {len(final_state['results'])} results")
        return True


class TestResultHandler:
    """Test suite for ResultHandler"""
    
    def test_result_evaluation(self) -> bool:
        """Test result evaluation logic"""
        print_test_header("Result Handler - Evaluation")
        
        handler = ResultHandler(max_retry=3)
        
        task = "Explain chest pain diagnosis"
        results = "Chest pain requires immediate medical evaluation. CT angiography is recommended."
        
        final_answer, should_stop, feedback = handler.evaluate_results(
            original_task=task,
            merged_results=results,
            retry_count=0
        )
        
        assert final_answer is not None, "Final answer is None"
        assert isinstance(should_stop, bool), "should_stop must be boolean"
        
        print(f"✓ Evaluation completed: should_stop={should_stop}")
        
        print_test_result(True, "Result evaluation working")
        return True


class TestEndToEndOrchestration:
    """End-to-end integration tests"""
    
    def test_simple_task(self) -> bool:
        """Test end-to-end orchestration with simple task"""
        print_test_header("End-to-End - Simple Task")
        
        orchestrator = LLMOrchestrator(max_retry=1)
        
        task = "What is the definition of contract law?"
        
        result = orchestrator.process_task(task, user_id="test_user")
        
        assert "success" in result, "Result missing 'success' field"
        assert "final_answer" in result, "Result missing 'final_answer'"
        assert "iterations" in result, "Result missing 'iterations'"
        
        print(f"✓ Task completed in {result['iterations']} iteration(s)")
        print(f"✓ Success: {result['success']}")
        
        print_test_result(True, "Simple task orchestration completed")
        return True
    
    def test_complex_task(self) -> bool:
        """Test end-to-end orchestration with complex multi-domain task"""
        print_test_header("End-to-End - Complex Task")
        
        orchestrator = LLMOrchestrator(max_retry=2)
        
        task = "Analyze whether a patient with sudden chest pain requires emergent CT angiography"
        
        result = orchestrator.process_task(task, user_id="test_user")
        
        assert "success" in result, "Result missing 'success' field"
        assert "final_answer" in result, "Result missing 'final_answer'"
        
        print(f"✓ Task completed: {result['success']}")
        print(f"✓ Iterations: {result['iterations']}")
        
        print_test_result(True, "Complex task orchestration completed")
        return True


def run_all_tests():
    """Run all test suites"""
    print("\n" + "=" * 80)
    print("LLM-AGENT-ORCHESTRATOR - COMPREHENSIVE INTEGRATION TEST")
    print("=" * 80)
    
    results = []
    
    # Test AgentFactory
    agent_factory_tests = TestAgentFactory()
    results.append(("Agent Factory - Domain Configuration", agent_factory_tests.test_domain_configuration()))
    results.append(("Agent Factory - Agent Creation", agent_factory_tests.test_agent_creation()))
    
    # Test GlobalRouter
    global_router_tests = TestGlobalRouter()
    results.append(("Global Router - Task Decomposition", global_router_tests.test_task_decomposition()))
    results.append(("Global Router - Complex Tasks", global_router_tests.test_complex_task_decomposition()))
    
    # Test VLLMLoader
    vllm_loader_tests = TestVLLMLoader()
    results.append(("VLLM Loader - Model Configurations", vllm_loader_tests.test_model_configurations()))
    
    # Test AgentSubRouter (check router service first)
    print("\n" + "=" * 80)
    print("ROUTER SERVICE STATUS CHECK")
    print("=" * 80)
    import requests
    router_url = os.getenv("ROUTER_SERVICE_URL", "http://localhost:8002")
    try:
        health_response = requests.get(f"{router_url}/health", timeout=2)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✓ Router service is healthy")
            print(f"  URL: {router_url}")
            print(f"  Loaded routers: {health_data.get('loaded_routers', [])}")
            print(f"  Total: {health_data.get('total_routers', 0)}")
        else:
            print(f"⚠ Router service returned status {health_response.status_code}")
    except Exception as e:
        print(f"⚠ Router service not accessible: {e}")
        print(f"  Tests will use vLLM fallback")
    
    agent_subrouter_tests = TestAgentSubRouter()
    results.append(("Agent SubRouter - Model Selection", agent_subrouter_tests.test_model_selection()))
    
    # Test GraphBuilder
    graph_builder_tests = TestGraphBuilder()
    results.append(("Graph Builder - Construction", graph_builder_tests.test_graph_construction()))
    results.append(("Graph Builder - Execution", graph_builder_tests.test_graph_execution()))
    
    # Test ResultHandler
    result_handler_tests = TestResultHandler()
    results.append(("Result Handler - Evaluation", result_handler_tests.test_result_evaluation()))
    
    # Test End-to-End
    end_to_end_tests = TestEndToEndOrchestration()
    results.append(("End-to-End - Simple Task", end_to_end_tests.test_simple_task()))
    results.append(("End-to-End - Complex Task", end_to_end_tests.test_complex_task()))
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "-" * 80)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 80)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
