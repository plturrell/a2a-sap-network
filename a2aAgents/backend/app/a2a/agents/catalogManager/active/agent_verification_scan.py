#!/usr/bin/env python3
"""
Comprehensive scan of all 16 A2A agents to verify 95/100 rating criteria
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Agent mapping
AGENT_MAPPING = {
    "dataProductAgent": ("agent0DataProduct", "Agent 0 - Data Product Agent"),
    "dataStandardizationAgent": ("agent1Standardization", "Agent 1 - Standardization Agent"),
    "aiPreparationAgent": ("agent2AiPreparation", "Agent 2 - AI Preparation Agent"),
    "vectorProcessingAgent": ("agent3VectorProcessing", "Agent 3 - Vector Processing Agent"),
    "calculationValidationAgent": ("agent4CalcValidation", "Agent 4 - Calc Validation Agent"),
    "qaValidationAgent": ("agent5QaValidation", "Agent 5 - QA Validation Agent"),
    "qualityControlManager": ("qualityControlManager", "Agent 6 - Quality Control Manager"),
    "agentBuilder": ("agentBuilder", "Agent 7 - Agent Builder"),
    "agentManager": ("agentManager", "Agent 8 - Agent Manager"),
    "reasoningAgent": ("reasoningAgent", "Agent 9 - Reasoning Agent"),
    "calculationAgent": ("calculationAgent", "Agent 10 - Calculation Agent"),
    "sqlAgent": ("sqlAgent", "Agent 11 - SQL Agent"),
    "catalogManager": ("catalogManager", "Agent 12 - Catalog Manager"),
    # Agent 13 is another instance of agentBuilder
    "embeddingFineTuner": ("embeddingFineTuner", "Agent 14 - Embedding Fine-Tuner"),
    "orchestratorAgent": ("orchestratorAgent", "Agent 15 - Orchestrator Agent")
}

def load_registry_capabilities(registry_file: str) -> List[str]:
    """Load capabilities from agent registry JSON file"""
    try:
        with open(registry_file, 'r') as f:
            data = json.load(f)
            return data.get("capabilities", [])
    except Exception as e:
        print(f"Error loading registry file {registry_file}: {e}")
        return []

def check_sdk_methods(sdk_file: str, capabilities: List[str]) -> Dict[str, bool]:
    """Check if SDK has @a2a_skill decorated methods for each capability"""
    if not os.path.exists(sdk_file):
        return {cap: False for cap in capabilities}
    
    with open(sdk_file, 'r') as f:
        content = f.read()
    
    results = {}
    for capability in capabilities:
        # Check for @a2a_skill decorator with the capability name
        import re
        # Match @a2a_skill(name="capability" with various spacing and quotes
        pattern = rf'@a2a_skill\s*\(\s*name\s*=\s*["\']?{re.escape(capability)}["\']?'
        
        has_decorator = bool(re.search(pattern, content, re.MULTILINE | re.DOTALL))
        
        # Also check if there's an async def method with the capability name
        method_pattern = rf'async\s+def\s+{re.escape(capability)}\s*\('
        has_method = bool(re.search(method_pattern, content, re.MULTILINE))
        
        results[capability] = has_decorator or has_method
    
    return results

def check_handler_operations(handler_file: str, capabilities: List[str]) -> Dict[str, bool]:
    """Check if A2A handler has capabilities in allowed_operations"""
    if not os.path.exists(handler_file):
        return {cap: False for cap in capabilities}
    
    with open(handler_file, 'r') as f:
        content = f.read()
    
    results = {}
    for capability in capabilities:
        # Check if capability is in allowed_operations
        in_allowed = f'"{capability}"' in content or f"'{capability}'" in content
        results[capability] = in_allowed
    
    return results

def verify_agent(agent_name: str, folder_name: str, base_path: Path) -> Dict[str, any]:
    """Verify a single agent meets the 95/100 criteria"""
    results = {
        "agent_name": agent_name,
        "folder_name": folder_name,
        "registry_found": False,
        "sdk_found": False,
        "handler_found": False,
        "capabilities": [],
        "sdk_implementations": {},
        "handler_operations": {},
        "issues": [],
        "score": 0
    }
    
    # Check registry file
    registry_path = f"/Users/apple/projects/a2a/a2aNetwork/data/agents/{agent_name}.json"
    if os.path.exists(registry_path):
        results["registry_found"] = True
        results["capabilities"] = load_registry_capabilities(registry_path)
    else:
        results["issues"].append(f"Registry file not found: {registry_path}")
    
    # Check SDK file
    sdk_patterns = [
        f"comprehensive{folder_name.replace('agent', '').capitalize()}Sdk.py",
        f"comprehensive{folder_name.replace('agent', '').capitalize()}AgentSdk.py",
        f"comprehensiveDataProductAgentSdk.py",  # Agent 0 specific
        f"enhancedDataStandardizationAgentMcp.py",  # Agent 1 specific
        f"enhancedAiPreparationAgentMcp.py",  # Agent 2 specific
        f"comprehensiveVectorProcessingSdk.py",  # Agent 3 specific
        f"comprehensiveCalcValidationAgentSdk.py",  # Agent 4 specific
        f"comprehensiveQaValidationAgentSdk.py",  # Agent 5 specific
        f"comprehensiveQualityControlManagerSdk.py",  # Agent 6 specific
        f"comprehensiveAgentBuilderSdk.py",  # Agent 7 specific
        f"comprehensiveAgentManagerSdk.py",  # Agent 8 specific
        f"comprehensiveReasoningAgentSdk.py",  # Agent 9 specific
        f"comprehensiveCalculationAgentSdk.py",  # Agent 10 specific
        f"comprehensiveSqlAgentSdk.py",  # Agent 11 specific
        f"comprehensiveCatalogManagerSdk.py",  # Agent 12 specific
        f"comprehensiveEmbeddingFineTunerSdk.py",  # Agent 14 specific
        f"comprehensiveOrchestratorAgentSdk.py",  # Agent 15 specific
        f"{folder_name}Sdk.py",
        f"{folder_name}AgentSdk.py"
    ]
    
    sdk_file = None
    for pattern in sdk_patterns:
        potential_sdk = base_path / folder_name / "active" / pattern
        if potential_sdk.exists():
            sdk_file = str(potential_sdk)
            results["sdk_found"] = True
            break
    
    if sdk_file:
        results["sdk_implementations"] = check_sdk_methods(sdk_file, results["capabilities"])
    else:
        results["issues"].append(f"SDK file not found in {folder_name}/active/")
    
    # Check A2A handler file
    handler_patterns = [
        f"{folder_name}A2AHandler.py",
        f"{folder_name.replace('agent', 'agent')}A2AHandler.py",
        f"{folder_name.replace('Agent', '_agent')}A2AHandler.py",
        f"agent7BuilderA2AHandler.py",  # Agent Builder specific
        f"agent9RouterA2AHandler.py",  # Reasoning Agent specific
        f"agent6QualityControlA2AHandler.py"  # Quality Control specific
    ]
    
    handler_file = None
    for pattern in handler_patterns:
        potential_handler = base_path / folder_name / "active" / pattern
        if potential_handler.exists():
            handler_file = str(potential_handler)
            results["handler_found"] = True
            break
    
    if handler_file:
        results["handler_operations"] = check_handler_operations(handler_file, results["capabilities"])
    else:
        results["issues"].append(f"A2A handler file not found in {folder_name}/active/")
    
    # Calculate score
    total_checks = 0
    passed_checks = 0
    
    # Basic file presence (30 points)
    if results["registry_found"]:
        passed_checks += 10
    total_checks += 10
    
    if results["sdk_found"]:
        passed_checks += 10
    total_checks += 10
    
    if results["handler_found"]:
        passed_checks += 10
    total_checks += 10
    
    # SDK implementations (35 points)
    if results["capabilities"]:
        for cap, implemented in results["sdk_implementations"].items():
            total_checks += 7
            if implemented:
                passed_checks += 7
            else:
                results["issues"].append(f"SDK missing @a2a_skill implementation for: {cap}")
    
    # Handler operations (35 points)
    if results["capabilities"]:
        for cap, included in results["handler_operations"].items():
            total_checks += 7
            if included:
                passed_checks += 7
            else:
                results["issues"].append(f"Handler missing capability in allowed_operations: {cap}")
    
    results["score"] = (passed_checks / total_checks * 100) if total_checks > 0 else 0
    
    return results

def main():
    """Run comprehensive scan of all agents"""
    base_path = Path("/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents")
    
    print("=== A2A Agent Verification Scan ===\n")
    
    all_results = []
    passing_agents = []
    failing_agents = []
    
    for registry_name, (folder_name, display_name) in AGENT_MAPPING.items():
        print(f"Scanning {display_name}...")
        results = verify_agent(registry_name, folder_name, base_path)
        all_results.append(results)
        
        if results["score"] >= 95:
            passing_agents.append(display_name)
        else:
            failing_agents.append((display_name, results["score"], results["issues"]))
    
    # Summary report
    print("\n=== SCAN SUMMARY ===\n")
    print(f"Total agents scanned: {len(all_results)}")
    print(f"Agents passing 95/100 criteria: {len(passing_agents)}")
    print(f"Agents needing fixes: {len(failing_agents)}")
    
    if passing_agents:
        print("\n✅ PASSING AGENTS:")
        for agent in passing_agents:
            print(f"  - {agent}")
    
    if failing_agents:
        print("\n❌ FAILING AGENTS:")
        for agent, score, issues in failing_agents:
            print(f"\n  {agent} (Score: {score:.1f}/100)")
            print("  Issues:")
            for issue in issues[:3]:  # Show first 3 issues
                print(f"    - {issue}")
            if len(issues) > 3:
                print(f"    ... and {len(issues) - 3} more issues")
    
    # Detailed report
    print("\n=== DETAILED RESULTS ===\n")
    for result in all_results:
        print(f"\n{result['agent_name']} ({result['folder_name']}):")
        print(f"  Score: {result['score']:.1f}/100")
        print(f"  Registry: {'✓' if result['registry_found'] else '✗'}")
        print(f"  SDK: {'✓' if result['sdk_found'] else '✗'}")
        print(f"  Handler: {'✓' if result['handler_found'] else '✗'}")
        print(f"  Capabilities: {', '.join(result['capabilities'])}")
        
        if result['sdk_implementations']:
            missing_sdk = [cap for cap, impl in result['sdk_implementations'].items() if not impl]
            if missing_sdk:
                print(f"  Missing SDK implementations: {', '.join(missing_sdk)}")
        
        if result['handler_operations']:
            missing_handler = [cap for cap, incl in result['handler_operations'].items() if not incl]
            if missing_handler:
                print(f"  Missing handler operations: {', '.join(missing_handler)}")

if __name__ == "__main__":
    main()