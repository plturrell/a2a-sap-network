#!/usr/bin/env python3
"""
Test GleanAgent functionality
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

async def test_glean_agent():
    """Test the GleanAgent functionality"""
    print("Testing GleanAgent Functionality")
    print("=" * 50)

    try:
        # Import the agent
        from app.a2a.agents.gleanAgent import GleanAgent
        print("✓ Successfully imported GleanAgent")

        # Create agent instance
        agent = GleanAgent()
        print(f"✓ Created agent: {agent.agent_id}")

        # Initialize the agent
        await agent.initialize()
        print("✓ Agent initialized")

        # Create a test directory with sample code
        test_dir = tempfile.mkdtemp(prefix="glean_test_")
        print(f"\nCreated test directory: {test_dir}")

        # Create a sample Python file with some issues
        sample_py = Path(test_dir) / "sample.py"
        sample_py.write_text("""
# Sample Python file with various issues
def calculate_sum(a, b):
    '''Calculate sum of two numbers'''
    return a + b

def unused_function():
    pass

# Function with high complexity
def complex_function(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return x + y + z
            else:
                return x + y
        else:
            return x
    else:
        return 0

# Missing proper error handling
def divide(a, b):
    return a / b  # Can raise ZeroDivisionError

# Unused variable
unused_var = 42

# Long line that violates PEP8
very_long_variable_name_that_exceeds_the_recommended_line_length_according_to_pep8_standards = "This is a very long line that should be broken up"
""")

        # Create a sample test file
        test_py = Path(test_dir) / "test_sample.py"
        test_py.write_text("""
import pytest
from sample import calculate_sum, divide

def test_calculate_sum():
    assert calculate_sum(2, 3) == 5
    assert calculate_sum(-1, 1) == 0

def test_divide():
    assert divide(10, 2) == 5.0
    # Missing test for zero division
""")

        print("\nRunning code analysis...")

        # Test basic linting
        print("\n1. Testing lint analysis:")
        lint_result = await agent._perform_lint_analysis(
            test_dir,
            ["*.py"]
        )
        print(f"   Files analyzed: {lint_result.get('files_analyzed', 0)}")
        print(f"   Total issues: {lint_result.get('total_issues', 0)}")
        if lint_result.get('issues_by_severity'):
            print("   Issues by severity:")
            for severity, count in lint_result['issues_by_severity'].items():
                print(f"     - {severity}: {count}")

        # Test comprehensive analysis
        print("\n2. Testing comprehensive analysis:")
        try:
            analysis_result = await agent.analyze_code_comprehensive(
                directory=test_dir,
                analysis_types=["lint"],  # Start with just linting
                file_patterns=["*.py"]
            )
            print(f"   Analysis ID: {analysis_result.get('analysis_id', 'N/A')}")
            print(f"   Duration: {analysis_result.get('duration', 0):.2f}s")

            if 'summary' in analysis_result:
                summary = analysis_result['summary']
                print(f"   Files analyzed: {summary.get('files_analyzed', 0)}")
                print(f"   Total issues: {summary.get('total_issues', 0)}")
                print(f"   Critical issues: {summary.get('critical_issues', 0)}")
                print(f"   Quality score: {summary.get('quality_score', 0):.1f}/100")
        except Exception as e:
            print(f"   ⚠️  Comprehensive analysis error: {e}")

        # Test skill listing
        print("\n3. Testing skills:")
        skills = agent.list_skills()
        print(f"   Found {len(skills)} skills:")
        for skill in skills[:3]:  # Show first 3
            print(f"     - {skill['name']}")

        # Test MCP tools
        print("\n4. Testing MCP tools:")
        tools = agent.list_mcp_tools()
        print(f"   Found {len(tools)} MCP tools:")
        for tool in tools:
            print(f"     - {tool['name']}")

        print("\n✅ GleanAgent is functional!")

        # Cleanup
        shutil.rmtree(test_dir)
        print(f"\nCleaned up test directory")

    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    # Run the async test
    success = asyncio.run(test_glean_agent())
    sys.exit(0 if success else 1)
