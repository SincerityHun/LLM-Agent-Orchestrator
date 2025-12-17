"""
Fact Extractor - Extract immutable facts from Original Task
"""
import re
from typing import Dict, Any


class FactExtractor:
    """Extract and validate ground truth facts from tasks"""
    
    def extract_immutable_facts(self, task: str) -> Dict[str, Any]:
        """
        Extract immutable facts from the Original Task.
        These facts are GROUND TRUTH and cannot be contradicted by agents.
        
        Args:
            task: Original task description
            
        Returns:
            Dictionary of immutable facts with types and values
        """
        facts = {
            "original_task": task,
            "extracted_facts": {},
            "constraints": []
        }
        
        task_lower = task.lower()
        
        # Extract age-related facts
        age_match = re.search(r'(\d+)-year-old', task)
        if age_match:
            age = int(age_match.group(1))
            facts["extracted_facts"]["patient_age"] = age
            facts["extracted_facts"]["is_minor"] = age < 18
            if age < 18:
                facts["constraints"].append(f"Patient is a minor (age {age})")
        
        # Extract policy requirements
        if "requires" in task_lower and "consent" in task_lower:
            if "parent" in task_lower or "guardian" in task_lower:
                facts["extracted_facts"]["consent_required"] = True
                facts["constraints"].append("Parental/guardian consent is required by policy")
        
        # Extract medical status
        if "stable" in task_lower:
            facts["extracted_facts"]["medical_status_stable"] = True
            facts["constraints"].append("Medical condition is documented as stable")
        
        if "does not create" in task_lower and "risk" in task_lower:
            facts["extracted_facts"]["delay_is_safe"] = True
            facts["constraints"].append("Delay does not create serious health risk")
        
        # Extract emergency status
        if "non-emergency" in task_lower:
            facts["extracted_facts"]["is_emergency"] = False
            facts["constraints"].append("This is a non-emergency service")
        elif "emergency" in task_lower and "non-" not in task_lower:
            facts["extracted_facts"]["is_emergency"] = True
        
        # Extract financial information
        cost_matches = re.findall(r'\$(\d+(?:,\d+)?)', task)
        if cost_matches:
            costs = [int(c.replace(',', '')) for c in cost_matches]
            if len(costs) >= 2:
                facts["extracted_facts"]["total_cost"] = max(costs)
                facts["extracted_facts"]["offered_payment"] = min(costs)
                facts["constraints"].append(f"Total cost: ${max(costs)}, Offered: ${min(costs)}")
        
        # Extract timeline information
        day_match = re.search(r'(\d+)\s*day[s]?', task)
        if day_match:
            days = int(day_match.group(1))
            facts["extracted_facts"]["timeline_days"] = days
            facts["constraints"].append(f"Timeline involves {days} days")
        
        return facts
    
    def format_facts_for_prompt(self, facts: Dict[str, Any]) -> str:
        """
        Format extracted facts as a prompt section.
        
        Args:
            facts: Dictionary from extract_immutable_facts()
            
        Returns:
            Formatted string for prompt injection
        """
        if not facts.get("constraints"):
            return ""
        
        lines = [
            "IMMUTABLE FACTS FROM ORIGINAL TASK:",
            "These are GROUND TRUTH and CANNOT be contradicted:",
            ""
        ]
        
        for idx, constraint in enumerate(facts["constraints"], 1):
            lines.append(f"{idx}. {constraint}")
        
        lines.append("")
        lines.append("ANY OUTPUT CONTRADICTING THESE FACTS WILL BE REJECTED.")
        
        return "\n".join(lines)
    
    def validate_agent_output(self, agent_output: str, facts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate if agent output contradicts immutable facts.
        
        Args:
            agent_output: Agent's response text
            facts: Dictionary from extract_immutable_facts()
            
        Returns:
            Dictionary with validation result and details
        """
        output_lower = agent_output.lower()
        violations = []
        
        # Check consent requirement violation
        if facts["extracted_facts"].get("consent_required"):
            # Check for negation of consent requirement
            negation_patterns = [
                "does not require consent",
                "no consent required",
                "consent is not required",
                "don't need consent",
                "without consent"
            ]
            for pattern in negation_patterns:
                if pattern in output_lower and "unless" not in output_lower:
                    violations.append({
                        "type": "consent_requirement_violation",
                        "description": "Agent claims consent is not required, contradicting task policy",
                        "severity": "critical"
                    })
                    break
        
        # Check medical stability violation
        if facts["extracted_facts"].get("medical_status_stable"):
            if "not stable" in output_lower or "unstable" in output_lower:
                violations.append({
                    "type": "medical_status_violation",
                    "description": "Agent claims condition is unstable, contradicting documented status",
                    "severity": "critical"
                })
        
        # Check delay safety violation
        if facts["extracted_facts"].get("delay_is_safe"):
            danger_patterns = ["delay creates risk", "delay is dangerous", "delay poses risk"]
            for pattern in danger_patterns:
                if pattern in output_lower:
                    violations.append({
                        "type": "delay_safety_violation",
                        "description": "Agent claims delay is dangerous, contradicting clinical documentation",
                        "severity": "critical"
                    })
                    break
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "has_critical_violations": any(v["severity"] == "critical" for v in violations)
        }


if __name__ == "__main__":
    # Test
    extractor = FactExtractor()
    
    test_task = """A home health agency is deciding whether to start next-day in-home wound-care visits for a 17-year-old after a minor surgery; the visits cost $250 per day for 4 days (total $1,000). Agency policy, reflecting the stated regulation, requires a parent or legal guardian to consent for non-emergency services for anyone under 18, and the clinician documents the wound is stable so a 5-day delay does not create a serious health risk. The patient offers to pay $600 upfront and asks the agency not to contact their parents, but the guardian can sign consent in 5 days—what should the agency do?"""
    
    facts = extractor.extract_immutable_facts(test_task)
    print("Extracted Facts:")
    print(facts)
    print("\n" + "="*80)
    print(extractor.format_facts_for_prompt(facts))
    
    # Test violation detection
    bad_output = "No, there is no requirement for parental consent unless it's an emergency service."
    validation = extractor.validate_agent_output(bad_output, facts)
    print("\n" + "="*80)
    print("Validation Result:")
    print(validation)
