#!/usr/bin/env python3
"""
Simple test to verify a2a-agents installation
"""

import sys
import traceback

def test_imports():
    """Test basic imports"""
    print("Testing a2a-agents installation...")
    
    try:
        # Test app imports
        print("✓ Testing app imports...")
        import app
        print(f"  - app module found at: {app.__file__}")
        
        import app.a2a
        print("  - app.a2a package imported successfully")
        
        # Test SDK
        print("\n✓ Testing SDK imports...")
        from app.a2a.sdk import A2AAgentBase, A2AClient
        print("  - SDK base classes imported successfully")
        
        # Test core modules
        print("\n✓ Testing core modules...")
        from app.a2a.core import telemetry
        print("  - Core telemetry module imported")
        
        # Test agent imports
        print("\n✓ Testing agent imports...")
        from app.a2a.agents.agent0DataProduct.active.dataProductAgentSdk import DataProductRegistrationAgentSDK
        print("  - Agent0 (Data Product) imported successfully")
        
        # Test CLI
        print("\n✓ Testing CLI...")
        from app.a2a.cli import create_parser
        print("  - CLI module imported successfully")
        
        print("\n✅ All basic imports successful! The installation appears to be working.")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_agent_creation():
    """Test creating an agent instance"""
    print("\n\nTesting agent instantiation...")
    
    try:
        from app.a2a.agents.agent0DataProduct.active.dataProductAgentSdk import DataProductRegistrationAgentSDK
        
        # Create agent with required parameters
        agent = DataProductRegistrationAgentSDK(
            base_url="http://localhost:8001",
            ord_registry_url="http://localhost:8080"
        )
        
        print(f"✓ Agent created successfully:")
        print(f"  - Name: {agent.name}")
        print(f"  - ID: {agent.agent_id}")
        print(f"  - Version: {agent.version}")
        print(f"  - Skills: {len(agent.skills)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("A2A-Agents Installation Test")
    print("=" * 60)
    
    success = True
    
    # Run tests
    if not test_imports():
        success = False
        
    if not test_agent_creation():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed! Installation is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)