#!/usr/bin/env python3
"""
Test real MCP integration in Agent Manager
"""

import asyncio
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))  # backend
os.environ['AGENT_PRIVATE_KEY'] = 'test_key_12345'

async def test_real_mcp():
    """Test the real MCP integration"""
    
    try:
        # Import after paths are set
        from app.a2a.agents.agentManager.active.agentManagerAgentMcp import AgentManagerAgentMCP
        print("✅ Import successful!")
        
        # Create agent
        agent = AgentManagerAgentMCP(base_url="http://localhost:8000")
        print(f"✅ Agent created: {agent.name} (ID: {agent.agent_id})")
        
        # Check if MCP server was created
        if hasattr(agent, 'mcp_server'):
            print(f"✅ MCP Server initialized: {agent.mcp_server}")
            
            # Get tools from MCP server directly
            tools = agent.mcp_server.get_tool_definitions()
            print(f"\n📋 MCP Tools from server: {len(tools)}")
            for tool in tools:
                print(f"   - {tool['name']}: {tool['description']}")
            
            # Get resources from MCP server directly
            resources = agent.mcp_server.get_resource_definitions()
            print(f"\n📊 MCP Resources from server: {len(resources)}")
            for resource in resources:
                print(f"   - {resource['uri']}: {resource['name']}")
        else:
            print("❌ MCP Server not found in agent")
            
            # Check MCP decorated methods directly
            import inspect
            mcp_tools = []
            mcp_resources = []
            
            for name, method in inspect.getmembers(agent, predicate=inspect.ismethod):
                if hasattr(method, '_mcp_tool'):
                    mcp_tools.append((name, method._mcp_tool))
                if hasattr(method, '_mcp_resource'):
                    mcp_resources.append((name, method._mcp_resource))
            
            print(f"\n📋 MCP Tool methods found: {len(mcp_tools)}")
            for name, metadata in mcp_tools:
                print(f"   - {name}: {metadata}")
                
            print(f"\n📊 MCP Resource methods found: {len(mcp_resources)}")
            for name, metadata in mcp_resources:
                print(f"   - {name}: {metadata}")
        
        # Test initialization
        print("\n🚀 Testing initialization...")
        result = await agent.initialize()
        print(f"   Init result: {result}")
        
        # Test MCP tool directly
        print("\n🧪 Testing MCP tool execution...")
        discover_result = await agent.discover_agents_mcp(
            required_capabilities=["test"],
            strategy="round_robin"
        )
        print(f"   Discovery result: {discover_result}")
        
        # Test MCP resource directly
        print("\n📖 Testing MCP resource access...")
        registry = await agent.get_agent_registry()
        print(f"   Registry result: {registry}")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_real_mcp())
    sys.exit(0 if result else 1)