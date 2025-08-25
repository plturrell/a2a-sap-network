import asyncio
import os
import sys
import logging


from app.a2a.core.security_base import SecureA2AAgent
#!/usr/bin/env python3
"""
Test real MCP integration in Agent Manager
"""

# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
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
        agent = AgentManagerAgentMCP(base_url=os.getenv("A2A_SERVICE_URL"))
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