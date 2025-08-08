#!/usr/bin/env python3
"""
Simple test for Agent 4 imports and basic functionality
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def test_imports():
    """Test that Agent 4 can be imported"""
    try:
        from app.a2a.agents.agent4_calc_validation.active.calc_validation_agent_sdk import (
            CalcValidationAgentSDK,
            ComputationType,
            TestMethodology
        )
        print("✅ Agent 4 imports successful")
        
        # Test basic instantiation
        agent = CalcValidationAgentSDK(
            base_url="http://localhost:8006",
            template_repository_url=None
        )
        
        print(f"✅ Agent 4 instantiation successful")
        print(f"   Agent ID: {agent.agent_id}")
        print(f"   Name: {agent.name}")
        print(f"   Version: {agent.version}")
        
        # Test template loading (basic)
        agent._load_builtin_templates()
        print(f"✅ Built-in templates loaded: {len(agent.test_templates)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent 4 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Agent 4 Simple Test")
    success = test_imports()
    sys.exit(0 if success else 1)