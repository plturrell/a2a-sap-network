#!/usr/bin/env python3
"""
Quick test to verify GleanAgent can start
"""
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(backend_dir))

print(f"Python path includes: {backend_dir}")

try:
    # Test imports
    print("Testing imports...")
    from app.a2a.agents.gleanAgent import GleanAgent
    print("✓ Successfully imported GleanAgent")

    # Test instantiation
    print("\nTesting instantiation...")
    agent = GleanAgent()
    print(f"✓ Successfully created GleanAgent instance")
    print(f"  Agent ID: {agent.agent_id}")
    print(f"  Agent Name: {agent.name}")
    print(f"  Agent Version: {agent.version}")
    print(f"  Base URL: {agent.base_url}")

    # Test agent card
    print("\nTesting agent card generation...")
    agent_card = agent.get_agent_card()
    print(f"✓ Successfully generated agent card")
    print(f"  Protocol Version: {agent_card.protocolVersion}")

    # Test skills listing
    print("\nTesting skills listing...")
    skills = agent.list_skills()
    print(f"✓ Found {len(skills)} skills:")
    for skill in skills:
        print(f"  - {skill['name']}: {skill['description']}")

    # Test MCP tools
    print("\nTesting MCP tools...")
    tools = agent.list_mcp_tools()
    print(f"✓ Found {len(tools)} MCP tools:")
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")

    print("\n✅ All basic tests passed!")

except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
