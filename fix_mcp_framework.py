#!/usr/bin/env python3
"""
Script to fix agents that use local MCP definitions instead of framework imports
"""

import os
import re
from pathlib import Path

# All agents to check
AGENTS = [
    'agent0DataProduct', 'agent1Standardization', 'agent2AiPreparation',
    'agent3VectorProcessing', 'agent4CalcValidation', 'agent5QaValidation', 
    'agent6QualityControl', 'agentBuilder', 'agentManager', 'calculationAgent',
    'catalogManager', 'dataManager', 'embeddingFineTuner', 'gleanAgent',
    'orchestratorAgent', 'reasoningAgent', 'serviceDiscoveryAgent', 'sqlAgent'
]

BASE_PATH = Path('a2aAgents/backend/app/a2a/agents')

def analyze_mcp_usage(agent_name):
    """Analyze MCP usage patterns in an agent"""
    agent_path = BASE_PATH / agent_name
    active_path = agent_path / 'active'
    
    result = {
        'agent_name': agent_name,
        'has_framework_imports': False,
        'has_local_definitions': False,
        'files_with_framework': [],
        'files_with_local': [],
        'needs_fixing': False
    }
    
    if not active_path.exists():
        result['error'] = 'No active directory found'
        return result
    
    # Check for MCP framework imports
    framework_patterns = [
        r'from.*mcpDecorators.*import',
        r'from.*sdk.*mcpDecorators.*import',
        r'from.*a2a.*sdk.*mcpDecorators.*import'
    ]
    
    # Check for local MCP definitions
    local_patterns = [
        r'def mcp_tool\(',
        r'def mcp_resource\(',
        r'def mcp_prompt\('
    ]
    
    for py_file in active_path.glob('*.py'):
        if py_file.name == '__init__.py':
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for framework imports
            for pattern in framework_patterns:
                if re.search(pattern, content):
                    result['has_framework_imports'] = True
                    result['files_with_framework'].append(py_file.name)
                    break
            
            # Check for local definitions
            for pattern in local_patterns:
                if re.search(pattern, content):
                    result['has_local_definitions'] = True
                    result['files_with_local'].append(py_file.name)
                    break
                    
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
    
    # Determine if fixing is needed
    if result['has_local_definitions'] and not result['has_framework_imports']:
        result['needs_fixing'] = True
    
    return result

def fix_agent_mcp_usage(agent_name, dry_run=True):
    """Fix agent to use MCP framework imports instead of local definitions"""
    agent_path = BASE_PATH / agent_name / 'active'
    
    if not agent_path.exists():
        return {'error': 'No active directory found'}
    
    fixed_files = []
    
    for py_file in agent_path.glob('*.py'):
        if py_file.name == '__init__.py':
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                original_content = content
            
            # Check if file has local MCP definitions
            has_local_mcp = any(re.search(pattern, content) for pattern in [
                r'def mcp_tool\(',
                r'def mcp_resource\(',
                r'def mcp_prompt\('
            ])
            
            if not has_local_mcp:
                continue
            
            # Check if it already has framework imports
            has_framework_import = re.search(r'from.*mcpDecorators.*import', content)
            
            if has_framework_import:
                continue  # Already using framework, no need to fix
            
            print(f"Fixing {py_file.name} in {agent_name}...")
            
            # Add MCP framework import
            if 'from app.a2a.sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt' not in content:
                # Find the best place to add the import
                lines = content.split('\n')
                import_insert_index = 0
                
                # Find last import or first non-comment line
                for i, line in enumerate(lines):
                    if line.strip().startswith('from ') or line.strip().startswith('import '):
                        import_insert_index = i + 1
                    elif line.strip().startswith('#') or line.strip() == '':
                        continue
                    else:
                        break
                
                # Insert MCP import
                mcp_import = 'from app.a2a.sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt'
                lines.insert(import_insert_index, mcp_import)
                content = '\n'.join(lines)
            
            # Remove local MCP definitions
            local_definitions = [
                r'# MCP integration decorators\ndef mcp_tool\(name: str, description: str = "", \*\*kwargs\):\s*"""[^"]*"""\s*def decorator\(func\):[^}]*return func\s*return decorator',
                r'def mcp_tool\(name: str, description: str = "", \*\*kwargs\):\s*"""[^"]*"""\s*def decorator\(func\):[^}]*return func\s*return decorator',
                r'def mcp_resource\(name: str, uri: str, \*\*kwargs\):\s*"""[^"]*"""\s*def decorator\(func\):[^}]*return func\s*return decorator',
                r'def mcp_prompt\(name: str, description: str = "", \*\*kwargs\):\s*"""[^"]*"""\s*def decorator\(func\):[^}]*return func\s*return decorator'
            ]
            
            for pattern in local_definitions:
                content = re.sub(pattern, '', content, flags=re.DOTALL)
            
            # Clean up extra whitespace
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
            
            # Write the fixed content
            if content != original_content:
                if not dry_run:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                
                fixed_files.append({
                    'file': py_file.name,
                    'changes': 'Added MCP framework import, removed local definitions'
                })
        
        except Exception as e:
            print(f"Error fixing {py_file}: {e}")
    
    return {
        'agent_name': agent_name,
        'fixed_files': fixed_files,
        'dry_run': dry_run
    }

def main():
    """Main function to analyze and fix MCP usage"""
    print("Analyzing MCP framework usage across all agents...")
    print("=" * 60)
    
    agents_to_fix = []
    
    # Analyze all agents
    for agent in AGENTS:
        result = analyze_mcp_usage(agent)
        
        if result.get('error'):
            print(f"❌ {agent}: {result['error']}")
            continue
        
        status = "✅"
        if result['needs_fixing']:
            status = "🔧"
            agents_to_fix.append(agent)
        elif result['has_local_definitions'] and result['has_framework_imports']:
            status = "⚠️"
        
        print(f"{status} {agent}: Framework={result['has_framework_imports']}, Local={result['has_local_definitions']}")
        
        if result['files_with_local']:
            print(f"   Local definitions in: {', '.join(result['files_with_local'])}")
        if result['files_with_framework']:
            print(f"   Framework imports in: {', '.join(result['files_with_framework'])}")
    
    print("\n" + "=" * 60)
    print(f"ANALYSIS COMPLETE")
    print(f"Agents needing fixes: {len(agents_to_fix)}")
    
    if agents_to_fix:
        print(f"Agents to fix: {', '.join(agents_to_fix)}")
        
        # Ask for confirmation to fix
        response = input("\nFix agents to use MCP framework? (y/N): ")
        if response.lower() == 'y':
            print("\nFixing agents...")
            
            for agent in agents_to_fix:
                print(f"\nFixing {agent}...")
                result = fix_agent_mcp_usage(agent, dry_run=False)
                
                if result.get('error'):
                    print(f"❌ Error fixing {agent}: {result['error']}")
                elif result['fixed_files']:
                    print(f"✅ Fixed {len(result['fixed_files'])} files in {agent}")
                    for file_info in result['fixed_files']:
                        print(f"   - {file_info['file']}: {file_info['changes']}")
                else:
                    print(f"ℹ️  No changes needed for {agent}")
            
            print("\n" + "=" * 60)
            print("MCP framework fixes complete!")
        else:
            print("Dry run mode - showing what would be fixed:")
            
            for agent in agents_to_fix:
                result = fix_agent_mcp_usage(agent, dry_run=True)
                print(f"\n{agent}: Would fix {len(result.get('fixed_files', []))} files")
                for file_info in result.get('fixed_files', []):
                    print(f"   - {file_info['file']}")
    else:
        print("✅ All agents are already using MCP framework correctly!")

if __name__ == "__main__":
    main()
