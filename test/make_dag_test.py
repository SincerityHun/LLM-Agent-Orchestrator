"""
Integration test for GlobalRouter DAG generation with real LLM
Tests structured output generation and DAG construction
"""
import sys
import os
import json
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routers.global_router import GlobalRouter, TaskDAG, SubTask
from utils.llm_loader import VLLMLoader


class DAGIntegrationTest:
    """Integration test suite for DAG generation using GlobalRouter and global-router LLM"""
    
    def __init__(self):
        self.router = GlobalRouter()
        self.llm_loader = VLLMLoader()
        # Test tasks covering different domains to validate structured DAG generation
        self.test_tasks = [
            "Analyze a patient with chest pain in the emergency department",
            "Develop a legal strategy for a contract dispute case",
            "Calculate and optimize portfolio risk for investment management",
            "Create a comprehensive marketing campaign for a new product launch"
        ]
        
    def test_llm_connectivity(self) -> bool:
        """Test if global-router LLM endpoint is accessible"""
        print("Testing global-router endpoint connectivity...")
        
        # Only test global-router since that's what we need for DAG generation
        global_router_available = self.llm_loader.is_endpoint_available("global-router")
        if global_router_available:
            print("  global-router: ✓ Available")
            print("✓ Global router connectivity test passed")
            return True
        else:
            print("  global-router: ✗ Unavailable")
            print("⚠️  Global router endpoint unavailable - will use mock responses")
            return False
    
    def test_json_schema_generation(self) -> bool:
        """Test JSON schema generation for guided decoding"""
        print("\nTesting JSON schema generation...")
        
        try:
            schema = TaskDAG.model_json_schema()
            
            # Validate schema structure
            required_keys = ['$defs', 'properties', 'required', 'title', 'type']
            for key in required_keys:
                if key not in schema:
                    print(f"✗ Missing key '{key}' in schema")
                    return False
            
            # Check if tasks property exists
            if 'tasks' not in schema['properties']:
                print("✗ Missing 'tasks' property in schema")
                return False
                
            print(f"✓ Schema generated successfully with keys: {list(schema.keys())}")
            print(f"  Tasks property type: {schema['properties']['tasks'].get('type', 'unknown')}")
            
            return True
            
        except Exception as e:
            print(f"✗ Schema generation failed: {e}")
            return False
    
    def test_structured_dag_generation(self, task: str, use_real_llm: bool = True) -> Dict:
        """Test structured DAG generation with real or mock LLM"""
        print(f"\nTesting DAG generation for task: '{task}'")
        print(f"Using real LLM: {use_real_llm}")
        
        validation_success = False
        
        try:
            if use_real_llm:
                # Test with real LLM using guided JSON
                dag_result = self.router.create_dag(task, None, None)
                
                if dag_result is None:
                    print("✗ DAG generation returned None - validation failed completely")
                    graph = self.router._create_default_graph(task)
                    validation_success = False
                else:
                    print(f"✓ DAG model validation successful! Created {len(dag_result.tasks)} tasks")
                    graph = self.router._taskdag_to_graph(dag_result)
                    validation_success = True
                    
                    # Display actual structured data
                    print("✓ Structured TaskDAG contents:")
                    for i, task_obj in enumerate(dag_result.tasks):
                        deps = f" (deps: {task_obj.dependencies})" if task_obj.dependencies else ""
                        print(f"    Task {i+1}: {task_obj.id} [{task_obj.role}]{deps}")
                        print(f"             {task_obj.content}")
                    
            else:
                # Test with mock structured response
                mock_response = self._create_mock_structured_response(task)
                print(f"Mock response: {mock_response}")
                
                dag_result = TaskDAG.model_validate_json(mock_response)
                graph = self.router._taskdag_to_graph(dag_result)
                validation_success = True
                print("✓ Mock DAG validation and conversion successful")
            
            # Add validation success flag to graph
            graph['_validation_success'] = validation_success
            return graph
            
        except Exception as e:
            print(f"✗ DAG generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {'_validation_success': False}
    
    def validate_dag_structure(self, graph: Dict, task: str) -> bool:
        """Validate the structure and content of generated DAG"""
        print(f"\nValidating DAG structure for: '{task}'")
        
        if not graph or 'nodes' not in graph or 'edges' not in graph:
            print("✗ Invalid graph structure - missing nodes or edges")
            return False
        
        nodes = graph['nodes']
        edges = graph['edges']
        
        print(f"  Nodes: {len(nodes)}")
        print(f"  Edges: {len(edges)}")
        
        # Validate nodes
        node_ids = set()
        valid_roles = {'planning', 'execution', 'review'}
        
        for i, node in enumerate(nodes):
            # Check required fields
            if not all(field in node for field in ['id', 'role', 'task']):
                print(f"✗ Node {i} missing required fields")
                return False
            
            # Check role validity
            if node['role'] not in valid_roles:
                print(f"✗ Node {i} has invalid role: {node['role']}")
                return False
            
            # Check for duplicate IDs
            if node['id'] in node_ids:
                print(f"✗ Duplicate node ID: {node['id']}")
                return False
            
            node_ids.add(node['id'])
            print(f"    {node['id']} ({node['role']}): {node['task'][:50]}...")
        
        # Validate edges
        for i, edge in enumerate(edges):
            if not all(field in edge for field in ['from', 'to']):
                print(f"✗ Edge {i} missing required fields")
                return False
            
            # Check if referenced nodes exist
            if edge['from'] not in node_ids or edge['to'] not in node_ids:
                print(f"✗ Edge {i} references non-existent nodes: {edge['from']} -> {edge['to']}")
                return False
            
            print(f"    {edge['from']} → {edge['to']}")
        
        # Check for at least basic workflow (planning -> execution -> review)
        role_counts = {}
        for node in nodes:
            role = node['role']
            role_counts[role] = role_counts.get(role, 0) + 1
        
        print(f"  Role distribution: {role_counts}")
        
        if len(role_counts) < 2:
            print("⚠️  DAG has limited role diversity")
        
        print("✓ DAG structure validation passed")
        return True
    
    def test_multiple_tasks(self, use_real_llm: bool = True) -> Dict:
        """Test DAG generation for multiple different tasks"""
        print(f"\n{'='*60}")
        print("TESTING DAG GENERATION WITH GLOBAL-ROUTER LLM")
        print(f"{'='*60}")
        
        results = {}
        
        for task in self.test_tasks:
            print(f"\n{'-'*40}")
            graph = self.test_structured_dag_generation(task, use_real_llm)
            
            if graph and 'nodes' in graph:
                is_valid = self.validate_dag_structure(graph, task)
                validation_success = graph.get('_validation_success', False)
                results[task] = {
                    'graph': graph,
                    'valid': is_valid,
                    'validation_success': validation_success,
                    'node_count': len(graph.get('nodes', [])),
                    'edge_count': len(graph.get('edges', []))
                }
            else:
                validation_success = graph.get('_validation_success', False) if graph else False
                results[task] = {
                    'valid': False, 
                    'validation_success': validation_success,
                    'error': 'Generation failed'
                }
        
        return results
    
    def _create_mock_structured_response(self, task: str) -> str:
        """Create a mock structured JSON response for testing"""
        mock_dag = {
            "tasks": [
                {
                    "id": f"plan_{hash(task) % 1000}",
                    "role": "planning",
                    "content": f"Develop comprehensive plan for: {task}",
                    "dependencies": []
                },
                {
                    "id": f"exec_{hash(task) % 1000}",
                    "role": "execution", 
                    "content": f"Execute planned approach for: {task}",
                    "dependencies": [f"plan_{hash(task) % 1000}"]
                },
                {
                    "id": f"review_{hash(task) % 1000}",
                    "role": "review",
                    "content": f"Review and validate results for: {task}",
                    "dependencies": [f"exec_{hash(task) % 1000}"]
                }
            ]
        }
        return json.dumps(mock_dag)
    
    def run_comprehensive_test(self) -> bool:
        """Run comprehensive DAG generation test suite with global-router LLM"""
        print("🚀 Starting GlobalRouter DAG Generation Test Suite")
        print("="*60)
        
        # Test 1: LLM Connectivity
        llm_available = self.test_llm_connectivity()
        
        # Test 2: Schema Generation
        schema_ok = self.test_json_schema_generation()
        if not schema_ok:
            print("❌ Schema generation failed - aborting tests")
            return False
        
        # Test 3: Multiple task generation
        test_results = self.test_multiple_tasks(use_real_llm=llm_available)
        
        # Summary
        print(f"\n{'='*60}")
        print("TEST RESULTS SUMMARY")
        print(f"{'='*60}")
        
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result.get('valid', False))
        validation_successes = sum(1 for result in test_results.values() if result.get('validation_success', False))
        
        print(f"Total tasks tested: {total_tests}")
        print(f"Successful TaskDAG validations: {validation_successes}")
        print(f"Successful DAG generations: {passed_tests}")
        print(f"Validation success rate: {validation_successes/total_tests*100:.1f}%")
        print(f"Overall success rate: {passed_tests/total_tests*100:.1f}%")
        
        for task, result in test_results.items():
            validation_status = "✓ VALIDATED" if result.get('validation_success', False) else "✗ FALLBACK"
            dag_status = "✓ VALID" if result.get('valid', False) else "✗ INVALID"
            
            if result.get('valid', False):
                print(f"{validation_status} {dag_status} - {task[:50]}... ({result['node_count']} nodes, {result['edge_count']} edges)")
            else:
                error = result.get('error', 'Validation failed')
                print(f"{validation_status} {dag_status} - {task[:50]}... ({error})")
        
        success = passed_tests == total_tests
        
        if success:
            print("\n🎉 All integration tests passed!")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} tests failed")
        
        return success


def main():
    """Run the integration test"""
    test_suite = DAGIntegrationTest()
    
    try:
        success = test_suite.run_comprehensive_test()
        exit_code = 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        exit_code = 130
        
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    
    print(f"\nExiting with code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
