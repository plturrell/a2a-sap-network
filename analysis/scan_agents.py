#!/usr/bin/env python3
"""
Script to scan all 16 agents for service/adapter layer gaps, mocks, and simulations
"""

import os
import json
from pathlib import Path

# All 16 agents identified
AGENTS = [
    'agent0DataProduct', 'agent1Standardization', 'agent2AiPreparation',
    'agent3VectorProcessing', 'agent4CalcValidation', 'agent5QaValidation', 
    'agent6QualityControl', 'agentBuilder', 'agentManager', 'calculationAgent',
    'catalogManager', 'dataManager', 'embeddingFineTuner', 'gleanAgent',
    'orchestratorAgent', 'reasoningAgent', 'serviceDiscoveryAgent', 'sqlAgent'
]

BASE_PATH = Path('a2aAgents/backend/app/a2a/agents')

def scan_agent_structure(agent_name):
    """Scan individual agent for service/adapter layer structure"""
    agent_path = BASE_PATH / agent_name
    if not agent_path.exists():
        return {
            'name': agent_name,
            'exists': False,
            'error': 'Directory not found'
        }
    
    result = {
        'name': agent_name,
        'exists': True,
        'files': [],
        'directories': [],
        'service_layer': {},
        'adapter_layer': {},
        'mocks': {},
        'simulations': {},
        'gaps': []
    }
    
    # Scan all files and directories
    try:
        for item in agent_path.iterdir():
            if item.is_file():
                result['files'].append(item.name)
            elif item.is_dir():
                result['directories'].append(item.name)
    except Exception as e:
        result['error'] = str(e)
        return result
    
    # Look for service layer patterns
    service_patterns = ['service', 'services', 'adapter', 'adapters', 'interface', 'interfaces']
    for pattern in service_patterns:
        matching_files = [f for f in result['files'] if pattern.lower() in f.lower()]
        matching_dirs = [d for d in result['directories'] if pattern.lower() in d.lower()]
        if matching_files or matching_dirs:
            result['service_layer'][pattern] = {
                'files': matching_files,
                'directories': matching_dirs
            }
    
    # Look for mock patterns
    mock_patterns = ['mock', 'test', 'stub', 'fake', 'dummy']
    for pattern in mock_patterns:
        matching_files = [f for f in result['files'] if pattern.lower() in f.lower()]
        matching_dirs = [d for d in result['directories'] if pattern.lower() in d.lower()]
        if matching_files or matching_dirs:
            result['mocks'][pattern] = {
                'files': matching_files,
                'directories': matching_dirs
            }
    
    # Look for simulation patterns
    sim_patterns = ['simulation', 'sim', 'simulator', 'emulator', 'sandbox']
    for pattern in sim_patterns:
        matching_files = [f for f in result['files'] if pattern.lower() in f.lower()]
        matching_dirs = [d for d in result['directories'] if pattern.lower() in d.lower()]
        if matching_files or matching_dirs:
            result['simulations'][pattern] = {
                'files': matching_files,
                'directories': matching_dirs
            }
    
    # Identify potential gaps
    if not result['service_layer']:
        result['gaps'].append('No service layer detected')
    if not result['mocks']:
        result['gaps'].append('No mock implementations found')
    if not result['simulations']:
        result['gaps'].append('No simulation capabilities found')
    
    return result

def main():
    """Main function to scan all agents"""
    print("Scanning all 16 agents for service/adapter layer gaps...")
    print("=" * 60)
    
    all_results = []
    
    for agent in AGENTS:
        print(f"Scanning {agent}...")
        result = scan_agent_structure(agent)
        all_results.append(result)
        
        # Print immediate results
        if not result['exists']:
            print(f"  ❌ {agent}: {result.get('error', 'Not found')}")
        else:
            print(f"  ✅ {agent}: {len(result['files'])} files, {len(result['directories'])} dirs")
            if result['gaps']:
                for gap in result['gaps']:
                    print(f"    ⚠️  {gap}")
    
    # Save detailed results
    with open('agent_scan_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Scan complete. Results saved to agent_scan_results.json")
    
    # Summary statistics
    existing_agents = [r for r in all_results if r['exists']]
    print(f"\nSummary:")
    print(f"- Total agents: {len(AGENTS)}")
    print(f"- Existing agents: {len(existing_agents)}")
    print(f"- Missing agents: {len(AGENTS) - len(existing_agents)}")
    
    # Gap analysis
    agents_without_service_layer = len([r for r in existing_agents if 'No service layer detected' in r.get('gaps', [])])
    agents_without_mocks = len([r for r in existing_agents if 'No mock implementations found' in r.get('gaps', [])])
    agents_without_sims = len([r for r in existing_agents if 'No simulation capabilities found' in r.get('gaps', [])])
    
    print(f"- Agents without service layer: {agents_without_service_layer}")
    print(f"- Agents without mocks: {agents_without_mocks}")
    print(f"- Agents without simulations: {agents_without_sims}")

if __name__ == "__main__":
    main()
