"""
Reasoning Chain Analysis MCP Skill
Analyzes and traces reasoning chains for transparency
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.a2a.sdk.mcpDecorators import mcp_tool

logger = logging.getLogger(__name__)


@mcp_tool(
    name="reasoning_chain_analysis",
    description="Analyze reasoning chains to trace logic flow and identify weaknesses"
)
async def reasoning_chain_analysis(
    reasoning_steps: List[Dict[str, Any]],
    validate_logic: bool = True,
    identify_gaps: bool = True,
    suggest_improvements: bool = True
) -> Dict[str, Any]:
    """
    Analyze a chain of reasoning steps
    
    Args:
        reasoning_steps: List of reasoning steps to analyze
        validate_logic: Validate logical consistency
        identify_gaps: Identify gaps in reasoning
        suggest_improvements: Suggest improvements
    """
    try:
        # Import chain of thought architecture
        from ..chainOfThoughtArchitecture import create_chain_of_thought_reasoner
        
        analysis_results = {
            "total_steps": len(reasoning_steps),
            "logic_validation": {},
            "identified_gaps": [],
            "improvements": [],
            "chain_strength": 0.0
        }
        
        # Validate logic flow
        if validate_logic:
            logic_issues = []
            for i, step in enumerate(reasoning_steps):
                # Check step structure
                if not isinstance(step, dict):
                    logic_issues.append(f"Step {i}: Invalid structure")
                    continue
                
                # Check dependencies
                deps = step.get("dependencies", [])
                for dep in deps:
                    if dep >= i:
                        logic_issues.append(f"Step {i}: Forward dependency on step {dep}")
                
                # Check confidence
                confidence = step.get("confidence", 0)
                if confidence < 0.5:
                    logic_issues.append(f"Step {i}: Low confidence ({confidence})")
            
            analysis_results["logic_validation"] = {
                "valid": len(logic_issues) == 0,
                "issues": logic_issues
            }
        
        # Identify gaps
        if identify_gaps:
            gaps = []
            
            # Check for missing evidence
            for i, step in enumerate(reasoning_steps):
                evidence = step.get("evidence", [])
                if not evidence and step.get("type") != "assumption":
                    gaps.append({
                        "step": i,
                        "type": "missing_evidence",
                        "description": f"Step {i} lacks supporting evidence"
                    })
            
            # Check for logical jumps
            for i in range(1, len(reasoning_steps)):
                prev_conclusion = reasoning_steps[i-1].get("conclusion", "")
                curr_premise = reasoning_steps[i].get("premise", "")
                
                if prev_conclusion and curr_premise and prev_conclusion not in curr_premise:
                    gaps.append({
                        "step": i,
                        "type": "logical_jump",
                        "description": f"Gap between step {i-1} and {i}"
                    })
            
            analysis_results["identified_gaps"] = gaps
        
        # Suggest improvements
        if suggest_improvements:
            improvements = []
            
            # Suggest evidence additions
            for gap in analysis_results["identified_gaps"]:
                if gap["type"] == "missing_evidence":
                    improvements.append({
                        "step": gap["step"],
                        "suggestion": "Add supporting evidence or references",
                        "priority": "high"
                    })
            
            # Suggest intermediate steps
            for gap in analysis_results["identified_gaps"]:
                if gap["type"] == "logical_jump":
                    improvements.append({
                        "step": gap["step"],
                        "suggestion": "Add intermediate reasoning step",
                        "priority": "medium"
                    })
            
            # Suggest confidence improvements
            low_confidence_steps = [i for i, s in enumerate(reasoning_steps) 
                                  if s.get("confidence", 0) < 0.6]
            for step_idx in low_confidence_steps:
                improvements.append({
                    "step": step_idx,
                    "suggestion": "Strengthen reasoning or gather more evidence",
                    "priority": "medium"
                })
            
            analysis_results["improvements"] = improvements
        
        # Calculate overall chain strength
        avg_confidence = sum(s.get("confidence", 0.5) for s in reasoning_steps) / len(reasoning_steps)
        gap_penalty = len(analysis_results["identified_gaps"]) * 0.1
        logic_penalty = 0.3 if not analysis_results["logic_validation"].get("valid", True) else 0
        
        chain_strength = max(0, min(1, avg_confidence - gap_penalty - logic_penalty))
        analysis_results["chain_strength"] = chain_strength
        
        return {
            "success": True,
            "analysis": analysis_results,
            "summary": {
                "strong_chain": chain_strength > 0.7,
                "main_issues": _summarize_issues(analysis_results),
                "recommended_actions": _prioritize_improvements(analysis_results["improvements"])
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "options": {
                    "validate_logic": validate_logic,
                    "identify_gaps": identify_gaps,
                    "suggest_improvements": suggest_improvements
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Reasoning chain analysis error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def _summarize_issues(analysis: Dict[str, Any]) -> List[str]:
    """Summarize main issues found"""
    issues = []
    
    if not analysis["logic_validation"].get("valid", True):
        issues.append("Logical inconsistencies detected")
    
    gap_types = {}
    for gap in analysis["identified_gaps"]:
        gap_type = gap["type"]
        gap_types[gap_type] = gap_types.get(gap_type, 0) + 1
    
    for gap_type, count in gap_types.items():
        issues.append(f"{count} {gap_type.replace('_', ' ')} gaps")
    
    return issues


def _prioritize_improvements(improvements: List[Dict[str, Any]]) -> List[str]:
    """Prioritize improvements by importance"""
    high_priority = [imp for imp in improvements if imp.get("priority") == "high"]
    medium_priority = [imp for imp in improvements if imp.get("priority") == "medium"]
    
    actions = []
    if high_priority:
        actions.append(f"Address {len(high_priority)} high-priority improvements")
    if medium_priority:
        actions.append(f"Consider {len(medium_priority)} medium-priority enhancements")
    
    return actions