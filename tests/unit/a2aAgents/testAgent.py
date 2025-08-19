#!/usr/bin/env python3
"""
Simple test script to verify agent functionality
"""

import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

async def test_agent_creation():
    """Test creating an agent instance"""
    try:
        from app.a2a.agents.agent0DataProduct.active.dataProductAgentSdk import DataProductRegistrationAgentSDK
        
        print("🔄 Creating agent instance...")
        agent = DataProductRegistrationAgentSDK(
            base_url="http://localhost:8000",
            ord_registry_url="http://localhost:9000"
        )
        
        print("✅ Agent created successfully!")
        print(f"   Agent ID: {agent.agent_id}")
        print(f"   Name: {agent.name}")
        print(f"   Version: {agent.version}")
        print(f"   Skills: {len(agent.skills)}")
        print(f"   Handlers: {len(agent.handlers)}")
        
        # Test agent initialization
        print("🔄 Initializing agent...")
        await agent.initialize()
        print("✅ Agent initialized successfully!")
        
        # Test agent card generation
        print("🔄 Generating agent card...")
        agent_card = agent.get_agent_card()
        print("✅ Agent card generated successfully!")
        print(f"   Agent card URL: {agent_card.url}")
        print(f"   Protocol version: {agent_card.protocolVersion}")
        print(f"   Capabilities: {list(agent_card.capabilities.keys())}")
        
        # Test cleanup
        print("🔄 Shutting down agent...")
        await agent.shutdown()
        print("✅ Agent shutdown completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the test"""
    print("🚀 Starting A2A Agent Test Suite")
    print("=" * 50)
    
    success = await test_agent_creation()
    
    print("=" * 50)
    if success:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)