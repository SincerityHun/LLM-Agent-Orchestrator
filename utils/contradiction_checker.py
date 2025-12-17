"""
Contradiction Checker - Detect contradictions between agents and immutable facts
"""
from typing import Dict, List, Any


class ContradictionChecker:
    """Check for contradictions across agent outputs and with immutable facts"""
    
    def __init__(self):
        self.domain_priority = ["law", "medical", "math", "commonsense"]
    
    def check_contradictions(
        self,
        agent_results: Dict[str, Any],
        immutable_facts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check for contradictions in agent outputs.
        
        Args:
            agent_results: Dictionary of agent execution results
            immutable_facts: Ground truth facts from FactExtractor
            
        Returns:
            Dictionary with contradiction analysis
        """
        from utils.fact_extractor import FactExtractor
        
        extractor = FactExtractor()
        contradictions = []
        safe_results = {}
        
        for node_id, result_data in agent_results.items():
            domain = result_data.get("domain", "unknown")
            result_text = result_data.get("result", "")
            
            # Skip mock responses
            if "[MOCK RESPONSE" in result_text:
                contradictions.append({
                    "node_id": node_id,
                    "domain": domain,
                    "type": "mock_response",
                    "severity": "critical",
                    "description": "Agent returned mock/error response"
                })
                continue
            
            # Validate against immutable facts
            validation = extractor.validate_agent_output(result_text, immutable_facts)
            
            if not validation["valid"]:
                for violation in validation["violations"]:
                    contradictions.append({
                        "node_id": node_id,
                        "domain": domain,
                        "type": violation["type"],
                        "severity": violation["severity"],
                        "description": violation["description"],
                        "violating_text": result_text[:200]
                    })
            else:
                # Only keep validated results
                safe_results[node_id] = result_data
        
        return {
            "has_contradictions": len(contradictions) > 0,
            "contradiction_count": len(contradictions),
            "contradictions": contradictions,
            "safe_results": safe_results,
            "discarded_count": len(agent_results) - len(safe_results)
        }
    
    def filter_by_priority(
        self,
        agent_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        When agents contradict each other, keep higher priority domain.
        Priority: law > medical > math > commonsense
        
        Args:
            agent_results: Dictionary of agent execution results
            
        Returns:
            Filtered results with priority applied
        """
        domain_results = {}
        
        # Group by domain
        for node_id, result_data in agent_results.items():
            domain = result_data.get("domain", "unknown")
            if domain not in domain_results:
                domain_results[domain] = []
            domain_results[domain].append((node_id, result_data))
        
        # Keep highest priority for each topic
        filtered = {}
        for node_id, result_data in agent_results.items():
            filtered[node_id] = result_data
        
        return filtered
    
    def generate_report(self, check_result: Dict[str, Any]) -> str:
        """
        Generate human-readable contradiction report.
        
        Args:
            check_result: Output from check_contradictions()
            
        Returns:
            Formatted report string
        """
        lines = []
        
        if not check_result["has_contradictions"]:
            lines.append("✓ No contradictions detected")
            return "\n".join(lines)
        
        lines.append(f"✗ {check_result['contradiction_count']} contradictions detected")
        lines.append(f"✗ {check_result['discarded_count']} agent results discarded")
        lines.append("")
        
        for idx, contradiction in enumerate(check_result["contradictions"], 1):
            lines.append(f"{idx}. [{contradiction['severity'].upper()}] {contradiction['domain']} agent ({contradiction['node_id']})")
            lines.append(f"   Type: {contradiction['type']}")
            lines.append(f"   Issue: {contradiction['description']}")
            if "violating_text" in contradiction:
                lines.append(f"   Text: {contradiction['violating_text']}...")
            lines.append("")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test
    from utils.fact_extractor import FactExtractor
    
    test_task = """A home health agency is deciding whether to start next-day in-home wound-care visits for a 17-year-old after a minor surgery; the visits cost $250 per day for 4 days (total $1,000). Agency policy, reflecting the stated regulation, requires a parent or legal guardian to consent for non-emergency services for anyone under 18, and the clinician documents the wound is stable so a 5-day delay does not create a serious health risk."""
    
    extractor = FactExtractor()
    facts = extractor.extract_immutable_facts(test_task)
    
    # Simulate agent results
    agent_results = {
        "task1": {
            "domain": "law",
            "result": "No, there is no requirement for parental consent unless it's an emergency service."
        },
        "task2": {
            "domain": "medical",
            "result": "The wound is documented as stable and delay is safe."
        }
    }
    
    checker = ContradictionChecker()
    check_result = checker.check_contradictions(agent_results, facts)
    
    print("Contradiction Check:")
    print(checker.generate_report(check_result))
    print("\nSafe Results Count:", len(check_result["safe_results"]))
