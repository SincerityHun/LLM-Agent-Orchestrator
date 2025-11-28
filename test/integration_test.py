"""
Integration test for LLM-Agent-Orchestrator system
"""
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import LLMOrchestrator
from routers.global_router import GlobalRouter
from orchestrator.graph_builder import GraphBuilder
from orchestrator.result_handler import ResultHandler


def test_global_router():
    """Test Global Router task decomposition"""
    print("\n" + "="*60)
    print("TEST 1: Global Router Task Decomposition")
    print("="*60)
    
    router = GlobalRouter()
    task = "Analyze legal implications of medical malpractice in chest pain diagnosis"
    
    dag = router.decompose_task(task)
    
    # Validate DAG structure
    assert "nodes" in dag, "DAG must have nodes"
    assert "edges" in dag, "DAG must have edges"
    assert len(dag["nodes"]) >= 3, "DAG should have at least 3 nodes"
    
    # Check node structure
    for node in dag["nodes"]:
        assert "id" in node, "Node must have id"
        assert "role" in node, "Node must have role"
        assert "task" in node, "Node must have task"
    
    print("✓ Global Router test passed")
    print(f"  - Generated {len(dag['nodes'])} nodes")
    print(f"  - Generated {len(dag['edges'])} edges")
    
    return True


def test_graph_builder():
    """Test LangGraph building and execution"""
    print("\n" + "="*60)
    print("TEST 2: Graph Builder and Execution")
    print("="*60)
    
    builder = GraphBuilder()
    
    # Create test DAG
    test_dag = {
        "nodes": [
            {"id": "plan", "role": "planning", "task": "Create treatment plan"},
            {"id": "exec", "role": "execution", "task": "Execute diagnosis"},
            {"id": "rev", "role": "review", "task": "Review results"}
        ],
        "edges": [
            {"from": "plan", "to": "exec"},
            {"from": "exec", "to": "rev"}
        ]
    }
    
    # Build graph
    graph = builder.build_graph(test_dag)
    assert graph is not None, "Graph should be built successfully"
    
    # Execute graph
    initial_state = {
        "original_task": "Test medical diagnosis",
        "current_node": "",
        "results": {},
        "context": {}
    }
    
    final_state = builder.execute_graph(graph, initial_state)
    
    # Validate results
    assert "results" in final_state, "Final state must have results"
    assert len(final_state["results"]) == 3, "Should have 3 agent results"
    
    print("✓ Graph Builder test passed")
    print(f"  - Executed {len(final_state['results'])} agents")
    
    return True


def test_result_handler():
    """Test Results Handler evaluation"""
    print("\n" + "="*60)
    print("TEST 3: Results Handler Evaluation")
    print("="*60)
    
    handler = ResultHandler(max_retry=3)
    
    task = "Diagnose chest pain"
    results = "[PLANNING] Emergency protocol\n[EXECUTION] CT recommended\n[REVIEW] Verified"
    
    # Test evaluation
    final_answer, should_stop, feedback = handler.evaluate_results(
        original_task=task,
        merged_results=results,
        retry_count=0
    )
    
    assert isinstance(final_answer, str), "Final answer must be string"
    assert isinstance(should_stop, bool), "Should stop must be boolean"
    assert isinstance(feedback, str), "Feedback must be string"
    
    # Test max retry limit
    _, should_stop_max, _ = handler.evaluate_results(
        original_task=task,
        merged_results=results,
        retry_count=3
    )
    
    assert should_stop_max == True, "Should stop at max retry"
    
    print("✓ Results Handler test passed")
    
    return True


def test_end_to_end():
    """Test complete end-to-end orchestration"""
    print("\n" + "="*60)
    print("TEST 4: End-to-End Orchestration")
    print("="*60)
    
    orchestrator = LLMOrchestrator(max_retry=2)
    
    task = "Should a patient with acute chest pain get immediate CT angiography?"
    result = orchestrator.process_task(task, user_id="integration_test")
    
    # Validate result structure
    assert "success" in result, "Result must have success field"
    assert "final_answer" in result, "Result must have final answer"
    assert "iterations" in result, "Result must have iterations count"
    assert "original_task" in result, "Result must have original task"
    
    # Validate iterations
    assert result["iterations"] >= 1, "Should have at least 1 iteration"
    assert result["iterations"] <= 2, "Should not exceed max retry"
    
    print("✓ End-to-End test passed")
    print(f"  - Completed in {result['iterations']} iteration(s)")
    print(f"  - Success: {result['success']}")
    
    return True


def test_sample_request_file():
    """Test loading and processing sample request file"""
    print("\n" + "="*60)
    print("TEST 5: Sample Request File Processing")
    print("="*60)
    
    # Load sample request
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "examples",
        "sample_request.json"
    )
    
    with open(sample_path, 'r') as f:
        request_data = json.load(f)
    
    assert "task" in request_data, "Sample request must have task"
    assert "user_id" in request_data, "Sample request must have user_id"
    
    # Process task
    orchestrator = LLMOrchestrator(max_retry=1)
    result = orchestrator.process_task(
        task=request_data["task"],
        user_id=request_data["user_id"]
    )
    
    assert result is not None, "Should return result"
    
    print("✓ Sample request file test passed")
    
    return True


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "#"*60)
    print("# LLM-Agent-Orchestrator Integration Tests")
    print("#"*60)
    
    tests = [
        ("Global Router", test_global_router),
        ("Graph Builder", test_graph_builder),
        ("Result Handler", test_result_handler),
        ("End-to-End", test_end_to_end),
        ("Sample Request", test_sample_request_file)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} test failed: {str(e)}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
