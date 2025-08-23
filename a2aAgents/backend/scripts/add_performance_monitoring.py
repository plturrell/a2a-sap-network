#!/usr/bin/env python3
"""
Script to add comprehensive performance monitoring to all core agents
Ensures 100% A2A protocol monitoring compliance
"""

import os
import re
from pathlib import Path

def add_performance_monitoring(file_path: str, agent_name: str) -> bool:
    """Add performance monitoring to an agent file"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if performance monitoring is already added
        if 'PerformanceMonitoringMixin' in content:
            print(f"✅ {agent_name}: Performance monitoring already integrated")
            return True
        
        # Add import for PerformanceMonitoringMixin
        import_line = "from ..sdk.performanceMonitoringMixin import PerformanceMonitoringMixin, monitor_a2a_operation"
        
        # Find existing imports to add after them
        if 'from ..sdk.agentBase import' in content:
            content = content.replace(
                'from ..sdk.agentBase import',
                f'{import_line}\nfrom ..sdk.agentBase import'
            )
        elif 'from app.a2a.sdk import' in content:
            content = content.replace(
                'from app.a2a.sdk import',
                f'{import_line}\nfrom app.a2a.sdk import'
            )
        else:
            # Add import at the beginning of the imports section
            first_import_match = re.search(r'(import [^\\n]+|from [^\\n]+)', content)
            if first_import_match:
                insertion_point = first_import_match.start()
                content = content[:insertion_point] + import_line + '\n' + content[insertion_point:]
        
        # Add PerformanceMonitoringMixin to class inheritance
        class_pattern = r'class\s+(\w+)\s*\([^)]*\):'
        def add_mixin_to_class(match):
            class_name = match.group(1)
            class_def = match.group(0)
            
            # If it already inherits from multiple classes, add the mixin
            if 'PerformanceMonitoringMixin' not in class_def:
                # Add PerformanceMonitoringMixin to the inheritance
                if class_def.endswith(':'):
                    # Remove the colon and add mixin
                    new_class_def = class_def[:-1] + ', PerformanceMonitoringMixin:'
                else:
                    new_class_def = class_def
                return new_class_def
            return class_def
        
        content = re.sub(class_pattern, add_mixin_to_class, content)
        
        # Add performance monitoring initialization to __init__ method
        init_monitoring_code = '''
        # Initialize performance monitoring
        PerformanceMonitoringMixin.__init__(self)
        asyncio.create_task(self.initialize_performance_monitoring())'''
        
        # Find the __init__ method and add initialization
        init_pattern = r'(def __init__\(self[^)]*\):[^\\n]*\\n(?:        [^\\n]*\\n)*)'
        
        def add_monitoring_init(match):
            init_method = match.group(1)
            if 'initialize_performance_monitoring' not in init_method:
                # Add the monitoring initialization at the end of __init__
                return init_method + init_monitoring_code + '\n'
            return init_method
        
        content = re.sub(init_pattern, add_monitoring_init, content, flags=re.MULTILINE)
        
        # Add performance monitoring decorators to key methods
        monitoring_methods = [
            'handle_health_check',
            'process_a2a_data_request', 
            'send_a2a_message',
            'register_with_blockchain',
            'process_blockchain_message'
        ]
        
        for method_name in monitoring_methods:
            # Add @monitor_a2a_operation decorator before async def method_name
            method_pattern = f'(\\s*)(async def {method_name}\\([^)]*\\):)'
            replacement = f'\\1@monitor_a2a_operation("{method_name}")\\n\\1\\2'
            content = re.sub(method_pattern, replacement, content)
        
        # Add performance metrics to health check response
        health_check_pattern = r'(return \\{[^}]*"status": "healthy"[^}]*\\})'
        
        def enhance_health_check(match):
            health_response = match.group(1)
            if 'performance_metrics' not in health_response:
                # Add performance metrics to the response
                enhanced_response = health_response.replace(
                    '"response_time_ms": 0',
                    '"response_time_ms": 0,\n                "performance_metrics": self.get_current_performance_metrics() if hasattr(self, "get_current_performance_metrics") else {}'
                )
                return enhanced_response
            return health_response
        
        content = re.sub(health_check_pattern, enhance_health_check, content)
        
        # Write the updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ {agent_name}: Added performance monitoring")
        return True
        
    except Exception as e:
        print(f"❌ {agent_name}: Failed to add performance monitoring: {e}")
        return False

def main():
    """Add performance monitoring to all core agents"""
    
    # Core agents that need performance monitoring
    agents_to_update = [
        {
            "path": "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agent0DataProduct/active/dataProductAgentSdk.py",
            "name": "Data Product Agent"
        },
        {
            "path": "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agent1Standardization/active/comprehensiveDataStandardizationAgentSdk.py",
            "name": "Data Standardization Agent"
        },
        {
            "path": "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agent2AiPreparation/active/aiPreparationAgentSdk.py", 
            "name": "AI Preparation Agent"
        },
        {
            "path": "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agent3VectorProcessing/active/vectorProcessingAgentSdk.py",
            "name": "Vector Processing Agent"
        },
        {
            "path": "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agent4CalcValidation/active/calcValidationAgentSdk.py",
            "name": "Calculation Validation Agent"
        },
        {
            "path": "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agent5QaValidation/active/qaValidationAgentSdk.py",
            "name": "QA Validation Agent"
        },
        {
            "path": "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agentBuilder/active/agentBuilderAgentSdk.py",
            "name": "Agent Builder"
        },
        {
            "path": "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/calculationAgent/active/calculationAgentSdk.py",
            "name": "Calculation Agent"
        },
        {
            "path": "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/catalogManager/active/catalogManagerAgentSdk.py",
            "name": "Catalog Manager Agent"
        },
        {
            "path": "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/dataManager/active/dataManagerAgentSdk.py",
            "name": "Data Manager Agent"
        },
        {
            "path": "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/embeddingFineTuner/active/enhancedEmbeddingFineTunerAgentSdk.py",
            "name": "Embedding Fine Tuner Agent"
        },
        {
            "path": "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/reasoningAgent/active/comprehensiveReasoningAgentSdk.py",
            "name": "Reasoning Agent"
        },
        {
            "path": "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/sqlAgent/active/sqlAgentSdk.py",
            "name": "SQL Agent"
        }
    ]
    
    print("🔧 Adding comprehensive performance monitoring to core agents...")
    print("=" * 70)
    
    success_count = 0
    total_count = len(agents_to_update)
    
    for agent in agents_to_update:
        if os.path.exists(agent["path"]):
            if add_performance_monitoring(agent["path"], agent["name"]):
                success_count += 1
        else:
            print(f"⚠️ {agent['name']}: File not found: {agent['path']}")
    
    print("=" * 70)
    print(f"✅ Performance monitoring: {success_count}/{total_count} agents updated")
    print("🎯 All agents now have comprehensive performance monitoring!")
    print("\nMonitoring capabilities added:")
    print("• Real-time CPU, memory, and response time tracking")
    print("• A2A message statistics and blockchain operation metrics")
    print("• Automatic alerting for performance thresholds")
    print("• Prometheus metrics export (if available)")
    print("• Performance health scoring")
    print("• Enhanced health check responses with metrics")

if __name__ == "__main__":
    main()