#!/usr/bin/env python3
"""
Test GleanAgent MCP tools specifically
"""
import asyncio
import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(backend_dir))

async def test_mcp_tools():
    """Test the GleanAgent MCP tools"""
    print("Testing GleanAgent MCP Tools")
    print("=" * 50)

    try:
        # Import the agent
        from app.a2a.agents.gleanAgent import GleanAgent
        print("✓ Successfully imported GleanAgent")

        # Create agent instance
        agent = GleanAgent()
        print(f"✓ Created agent: {agent.agent_id}")

        # Test MCP tools listing
        print("\n1. Testing MCP tools listing:")
        tools = agent.list_mcp_tools()
        print(f"   Found {len(tools)} MCP tools:")
        for tool in tools:
            print(f"     - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}")

        # Create a test directory with sample code
        test_dir = tempfile.mkdtemp(prefix="glean_mcp_test_")
        print(f"\n2. Created test directory: {test_dir}")

        # Create a sample Python file
        sample_py = Path(test_dir) / "sample.py"
        sample_py.write_text("""
# Sample Python file for MCP testing
def calculate_sum(a, b):
    '''Calculate sum of two numbers'''
    return a + b

def unused_function():
    pass

# Long line that violates PEP8
very_long_variable_name_that_exceeds_the_recommended_line_length_according_to_pep8_standards = "This is a very long line"
""")

        # Test individual MCP tools
        print("\n3. Testing MCP tool: glean_refactor_code")
        try:
            refactor_result = await agent.mcp_refactor_code(str(sample_py))
            print(f"   ✓ Refactor suggestions: {refactor_result.get('total_suggestions', 0)}")
        except Exception as e:
            print(f"   ⚠️  Refactor tool error: {e}")

        print("\n4. Testing MCP tool: glean_security_scan")
        try:
            security_result = await agent.mcp_security_scan(test_dir)
            print(f"   ✓ Security scan completed: {security_result.get('total_vulnerabilities', 0)} vulnerabilities found")
        except Exception as e:
            print(f"   ⚠️  Security scan error: {e}")

        print("\n5. Testing MCP tool: glean_test_coverage")
        try:
            coverage_result = await agent.mcp_test_coverage(test_dir)
            print(f"   ✓ Coverage analysis completed: {coverage_result.get('overall_coverage', 0):.1f}% coverage")
        except Exception as e:
            print(f"   ⚠️  Coverage tool error: {e}")

        print("\n6. Testing MCP tool: glean_run_linters")
        try:
            lint_result = await agent.mcp_run_linters(test_dir)
            print(f"   ✓ Linting completed: {lint_result.get('total_issues', 0)} issues found")
        except Exception as e:
            print(f"   ⚠️  Linting error: {e}")

        print("\n✅ MCP Tools testing completed!")

        # Cleanup
        shutil.rmtree(test_dir)
        print(f"Cleaned up test directory")

    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    # Run the async test
    success = asyncio.run(test_mcp_tools())
    sys.exit(0 if success else 1)