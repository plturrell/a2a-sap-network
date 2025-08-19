"""
Test the real Agent Manager MCP implementation
"""

import asyncio
import os
from agentManagerAgentMcp import AgentManagerAgentMCP

async def test_real_agent_manager():
    """Test the real Agent Manager with MCP"""
    
    # Set required environment variable for testing
    os.environ["AGENT_PRIVATE_KEY"] = "test_private_key_12345"
    
    try:
        # Create agent manager
        agent_manager = AgentManagerAgentMCP(base_url="http://localhost:8000")
        print("✅ Agent Manager created successfully")
        
        # Check MCP tools
        tools = agent_manager.list_mcp_tools()
        print(f"\n📋 MCP Tools ({len(tools)}):")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
        # Check MCP resources
        resources = agent_manager.list_mcp_resources()
        print(f"\n📊 MCP Resources ({len(resources)}):")
        for resource in resources:
            print(f"  - {resource.uri}: {resource.name}")
        
        # Initialize agent
        init_result = await agent_manager.initialize()
        print(f"\n🚀 Initialization: {init_result}")
        
        # Test registering an agent via MCP
        print("\n🧪 Testing agent registration via MCP...")
        reg_result = await agent_manager.register_agent_mcp(
            agent_id="test_agent_1",
            agent_name="Test Agent",
            base_url="http://localhost:8001",
            capabilities={"data_processing": True, "analysis": True},
            skills=[{"id": "skill1", "name": "Data Analysis"}]
        )
        print(f"Registration result: {reg_result}")
        
        # Test discovering agents via MCP
        print("\n🔍 Testing agent discovery via MCP...")
        disc_result = await agent_manager.discover_agents_mcp(
            required_capabilities=["data_processing"],
            strategy="least_loaded"
        )
        print(f"Discovery result: {disc_result}")
        
        # Test getting registry via MCP resource
        print("\n📖 Testing MCP resource access...")
        registry = await agent_manager.get_agent_registry()
        print(f"Registry has {registry['total_agents']} agents")
        
        # Test creating workflow via MCP
        print("\n🔄 Testing workflow creation via MCP...")
        wf_result = await agent_manager.create_workflow_mcp(
            workflow_name="Test Workflow",
            agents=["test_agent_1"],
            tasks=[{"task": "process_data", "params": {"data": "test"}}]
        )
        print(f"Workflow result: {wf_result}")
        
        # Check active workflows resource
        workflows = await agent_manager.get_active_workflows()
        print(f"Active workflows: {workflows['total_workflows']}")
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_real_agent_manager())