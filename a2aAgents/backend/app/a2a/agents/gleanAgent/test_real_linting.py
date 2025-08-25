#!/usr/bin/env python3
"""
Test real linting implementation
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

async def test_real_linting():
    """Test the real linting implementation"""
    print("Testing Real Linting Implementation")
    print("=" * 50)

    try:
        # Import the agent
        from app.a2a.agents.gleanAgent import GleanAgent
        print("✓ Successfully imported GleanAgent")

        # Create agent instance
        agent = GleanAgent()
        print(f"✓ Created agent: {agent.agent_id}")

        # Create a test directory with problematic Python code
        test_dir = tempfile.mkdtemp(prefix="glean_real_lint_test_")
        print(f"\n1. Created test directory: {test_dir}")

        # Create Python file with various linting issues
        sample_py = Path(test_dir) / "problematic.py"
        sample_py.write_text('''
# Problematic Python code for linting
import os
import sys
import unused_module

def badly_formatted_function(  a,b,c   ):
    """Missing type hints and bad formatting"""
    x=a+b  # No spaces around operators
    if x>10:
        if c>5:
            print("Too deeply nested")
    return x

def unused_function():
    pass

class MyClass:
    def __init__(self):
        pass

    def method_with_issues(self):
        password = "hardcoded_password"  # Security issue
        os.system("rm -rf /")  # Security issue
        eval("print('dangerous')")  # Security issue

# Missing newline at end of file''')

        print("\n2. Testing real linting execution:")

        # Test the real linting
        lint_result = await agent._perform_lint_analysis(test_dir, ["*.py"])

        print(f"   Files analyzed: {lint_result.get('files_analyzed', 0)}")
        print(f"   Total issues found: {lint_result.get('total_issues', 0)}")
        print(f"   Critical issues: {lint_result.get('critical_issues', 0)}")
        print(f"   Duration: {lint_result.get('duration', 0):.2f}s")

        if lint_result.get('issues_by_severity'):
            print("\n   Issues by severity:")
            for severity, count in lint_result['issues_by_severity'].items():
                print(f"     - {severity}: {count}")

        if lint_result.get('linter_results'):
            print(f"\n   Linters executed: {list(lint_result['linter_results'].keys())}")

        # Show sample issues
        issues = lint_result.get('issues', [])
        if issues:
            print(f"\n   Sample issues (showing first 3):")
            for i, issue in enumerate(issues[:3]):
                print(f"     {i+1}. {issue.get('tool', 'unknown')} - {issue.get('message', 'No message')}")
                print(f"        File: {issue.get('file_path', 'unknown')}:{issue.get('line', 0)}")
                print(f"        Severity: {issue.get('severity', 'unknown')}")

        print("\n✅ Real linting test completed!")

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
    success = asyncio.run(test_real_linting())
    sys.exit(0 if success else 1)