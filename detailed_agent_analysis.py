#!/usr/bin/env python3
"""
Detailed analysis of agent implementations to identify service/adapter patterns,
mocks, simulations, and architectural gaps
"""

import os
import re
from pathlib import Path
from collections import defaultdict

# All 16 agents identified
AGENTS = [
    'agent0DataProduct', 'agent1Standardization', 'agent2AiPreparation',
    'agent3VectorProcessing', 'agent4CalcValidation', 'agent5QaValidation', 
    'agent6QualityControl', 'agentBuilder', 'agentManager', 'calculationAgent',
    'catalogManager', 'dataManager', 'embeddingFineTuner', 'gleanAgent',
    'orchestratorAgent', 'reasoningAgent', 'serviceDiscoveryAgent', 'sqlAgent'
]

BASE_PATH = Path('a2aAgents/backend/app/a2a/agents')

def analyze_python_file(file_path):
    """Analyze Python file for service/adapter patterns, mocks, simulations"""
    analysis = {
        'service_patterns': [],
        'adapter_patterns': [],
        'mock_patterns': [],
        'simulation_patterns': [],
        'imports': [],
        'classes': [],
        'functions': [],
        'line_count': 0,
        'architectural_indicators': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            analysis['line_count'] = len(lines)
            
            # Pattern matching
            service_keywords = ['service', 'adapter', 'interface', 'client', 'api', 'endpoint']
            mock_keywords = ['mock', 'stub', 'fake', 'dummy', 'test']
            simulation_keywords = ['simulate', 'emulate', 'sandbox', 'virtual']
            
            for line in lines:
                line_lower = line.lower()
                
                # Check for imports
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    analysis['imports'].append(line.strip())
                
                # Check for class definitions
                if line.strip().startswith('class '):
                    class_match = re.match(r'class\s+(\w+)', line.strip())
                    if class_match:
                        analysis['classes'].append(class_match.group(1))
                
                # Check for function definitions
                if line.strip().startswith('def '):
                    func_match = re.match(r'def\s+(\w+)', line.strip())
                    if func_match:
                        analysis['functions'].append(func_match.group(1))
                
                # Service patterns
                for keyword in service_keywords:
                    if keyword in line_lower:
                        analysis['service_patterns'].append(f"Line: {line.strip()}")
                        break
                
                # Mock patterns
                for keyword in mock_keywords:
                    if keyword in line_lower:
                        analysis['mock_patterns'].append(f"Line: {line.strip()}")
                        break
                
                # Simulation patterns
                for keyword in simulation_keywords:
                    if keyword in line_lower:
                        analysis['simulation_patterns'].append(f"Line: {line.strip()}")
                        break
            
            # Architectural indicators
            if any('abstract' in imp.lower() for imp in analysis['imports']):
                analysis['architectural_indicators'].append('Uses abstract base classes')
            
            if any('protocol' in imp.lower() for imp in analysis['imports']):
                analysis['architectural_indicators'].append('Uses protocols/interfaces')
            
            if any('mcp' in imp.lower() for imp in analysis['imports']):
                analysis['architectural_indicators'].append('Uses MCP framework')
            
            if any('sdk' in cls.lower() for cls in analysis['classes']):
                analysis['architectural_indicators'].append('Has SDK implementation')
            
    except Exception as e:
        analysis['error'] = str(e)
    
    return analysis

def analyze_agent_implementation(agent_name):
    """Analyze complete agent implementation"""
    agent_path = BASE_PATH / agent_name
    active_path = agent_path / 'active'
    
    result = {
        'name': agent_name,
        'has_active_dir': active_path.exists(),
        'python_files': [],
        'file_analyses': {},
        'summary': {
            'total_files': 0,
            'total_lines': 0,
            'has_service_layer': False,
            'has_adapter_layer': False,
            'has_mocks': False,
            'has_simulations': False,
            'architectural_patterns': set()
        }
    }
    
    # Analyze files in both main directory and active directory
    paths_to_check = [agent_path]
    if result['has_active_dir']:
        paths_to_check.append(active_path)
    
    for check_path in paths_to_check:
        if check_path.exists():
            for file_path in check_path.glob('*.py'):
                if file_path.name != '__init__.py':  # Skip __init__.py files
                    rel_path = str(file_path.relative_to(BASE_PATH))
                    result['python_files'].append(rel_path)
                    
                    analysis = analyze_python_file(file_path)
                    result['file_analyses'][rel_path] = analysis
                    
                    # Update summary
                    result['summary']['total_files'] += 1
                    result['summary']['total_lines'] += analysis['line_count']
                    
                    if analysis['service_patterns']:
                        result['summary']['has_service_layer'] = True
                    if analysis['adapter_patterns']:
                        result['summary']['has_adapter_layer'] = True
                    if analysis['mock_patterns']:
                        result['summary']['has_mocks'] = True
                    if analysis['simulation_patterns']:
                        result['summary']['has_simulations'] = True
                    
                    result['summary']['architectural_patterns'].update(analysis['architectural_indicators'])
    
    # Convert set to list for JSON serialization
    result['summary']['architectural_patterns'] = list(result['summary']['architectural_patterns'])
    
    return result

def generate_gap_analysis(agent_results):
    """Generate comprehensive gap analysis"""
    gaps = {
        'critical_gaps': [],
        'architectural_gaps': [],
        'implementation_gaps': [],
        'testing_gaps': [],
        'recommendations': []
    }
    
    # Analyze each agent for gaps
    agents_without_service = []
    agents_without_mocks = []
    agents_without_sims = []
    agents_minimal_impl = []
    
    for result in agent_results:
        agent_name = result['name']
        summary = result['summary']
        
        if not summary['has_service_layer']:
            agents_without_service.append(agent_name)
        
        if not summary['has_mocks']:
            agents_without_mocks.append(agent_name)
        
        if not summary['has_simulations']:
            agents_without_sims.append(agent_name)
        
        if summary['total_files'] <= 1 or summary['total_lines'] < 100:
            agents_minimal_impl.append(agent_name)
    
    # Critical gaps
    if agents_without_service:
        gaps['critical_gaps'].append(f"Service layer missing in {len(agents_without_service)} agents: {', '.join(agents_without_service)}")
    
    if agents_minimal_impl:
        gaps['critical_gaps'].append(f"Minimal implementation in {len(agents_minimal_impl)} agents: {', '.join(agents_minimal_impl)}")
    
    # Testing gaps
    if agents_without_mocks:
        gaps['testing_gaps'].append(f"Mock implementations missing in {len(agents_without_mocks)} agents")
    
    if agents_without_sims:
        gaps['testing_gaps'].append(f"Simulation capabilities missing in {len(agents_without_sims)} agents")
    
    # Recommendations
    gaps['recommendations'].extend([
        "Implement standardized service/adapter layer architecture across all agents",
        "Create comprehensive mock implementations for isolated testing",
        "Develop simulation capabilities for each agent's domain",
        "Establish consistent architectural patterns and interfaces",
        "Implement proper dependency injection for better testability"
    ])
    
    return gaps

def main():
    """Main analysis function"""
    print("Performing detailed analysis of all 16 agents...")
    print("=" * 60)
    
    all_results = []
    
    for agent in AGENTS:
        print(f"Analyzing {agent}...")
        result = analyze_agent_implementation(agent)
        all_results.append(result)
        
        summary = result['summary']
        print(f"  Files: {summary['total_files']}, Lines: {summary['total_lines']}")
        print(f"  Service: {'✅' if summary['has_service_layer'] else '❌'}, "
              f"Mocks: {'✅' if summary['has_mocks'] else '❌'}, "
              f"Sims: {'✅' if summary['has_simulations'] else '❌'}")
        if summary['architectural_patterns']:
            print(f"  Patterns: {', '.join(summary['architectural_patterns'])}")
    
    # Generate gap analysis
    print("\n" + "=" * 60)
    print("GAP ANALYSIS")
    print("=" * 60)
    
    gaps = generate_gap_analysis(all_results)
    
    print("\nCRITICAL GAPS:")
    for gap in gaps['critical_gaps']:
        print(f"❌ {gap}")
    
    print("\nTESTING GAPS:")
    for gap in gaps['testing_gaps']:
        print(f"⚠️  {gap}")
    
    print("\nRECOMMENDATIONS:")
    for i, rec in enumerate(gaps['recommendations'], 1):
        print(f"{i}. {rec}")
    
    # Save detailed results
    import json
    with open('detailed_agent_analysis.json', 'w') as f:
        json.dump({
            'agent_results': all_results,
            'gap_analysis': gaps
        }, f, indent=2, default=str)
    
    print(f"\nDetailed analysis saved to detailed_agent_analysis.json")

if __name__ == "__main__":
    main()
