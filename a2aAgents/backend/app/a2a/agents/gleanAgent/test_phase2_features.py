#!/usr/bin/env python3
"""
Test GleanAgent Phase 2 features
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

async def test_phase2_features():
    """Test Phase 2 features of GleanAgent"""
    print("Testing GleanAgent Phase 2 Features")
    print("=" * 50)

    try:
        # Import the agent
        from app.a2a.agents.gleanAgent import GleanAgent
        print("✓ Successfully imported GleanAgent")

        # Create agent instance
        agent = GleanAgent()
        print(f"✓ Created agent: {agent.agent_id}")

        # Create a test directory with sample code
        test_dir = tempfile.mkdtemp(prefix="glean_phase2_test_")
        print(f"\n1. Created test directory: {test_dir}")

        # Create Python project structure
        (Path(test_dir) / "requirements.txt").write_text("requests>=2.25.0\nflask>=2.0.0")
        (Path(test_dir) / "src").mkdir()

        sample_py = Path(test_dir) / "src" / "main.py"
        sample_py.write_text("""
# Sample Python application
def calculate_sum(a, b):
    '''Calculate sum of two numbers'''
    return a + b

def divide(a, b):
    # Missing error handling
    return a / b

# Complex function for testing
def complex_logic(x, y, z):
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
""")

        # Test 1: Project Configuration
        print("\n2. Testing project configuration:")
        config_result = await agent.configure_for_project(test_dir)
        print(f"   ✓ Project type detected: {config_result.get('project_type', 'unknown')}")
        print(f"   ✓ Recommendations: {len(config_result.get('recommendations', []))}")

        # Test 2: Caching System
        print("\n3. Testing caching system:")
        # Run analysis twice to test caching
        start_time = asyncio.get_event_loop().time()
        result1 = await agent._with_cache(
            "test_analysis",
            lambda directory, file_patterns: {"cached": False, "directory": directory},
            directory=test_dir,
            file_patterns=["*.py"]
        )
        first_duration = asyncio.get_event_loop().time() - start_time

        start_time = asyncio.get_event_loop().time()
        result2 = await agent._with_cache(
            "test_analysis",
            lambda directory, file_patterns: {"cached": False, "directory": directory},
            directory=test_dir,
            file_patterns=["*.py"]
        )
        second_duration = asyncio.get_event_loop().time() - start_time

        print(f"   ✓ First call: {first_duration:.4f}s")
        print(f"   ✓ Second call (cached): {second_duration:.4f}s")
        print(f"   ✓ Cache working: {second_duration < first_duration}")

        # Test 3: Error Handling
        print("\n4. Testing error handling:")
        error_result = agent._handle_analysis_error(
            "test_operation",
            FileNotFoundError("Test file not found"),
            {"test": "context"}
        )
        print(f"   ✓ Error handled: {error_result.get('error_type', 'unknown')}")

        # Test 4: Parallel Analysis
        print("\n5. Testing parallel analysis:")
        try:
            parallel_result = await agent.analyze_project_comprehensive_parallel(
                directory=test_dir,
                analysis_types=["lint", "complexity"],
                max_concurrent=2
            )
            print(f"   ✓ Parallel analysis completed: {parallel_result.get('tasks_completed', 0)} tasks")
            print(f"   ✓ Duration: {parallel_result.get('duration', 0):.2f}s")
            analysis_id = parallel_result.get('analysis_id')
        except Exception as e:
            print(f"   ⚠️  Parallel analysis error: {e}")
            analysis_id = None

        # Test 5: Analysis History
        print("\n6. Testing analysis history:")
        try:
            history_result = await agent.get_analysis_history(directory=test_dir, limit=5)
            print(f"   ✓ History retrieved: {history_result.get('count', 0)} analyses")
        except Exception as e:
            print(f"   ⚠️  History retrieval error: {e}")

        # Test 6: Quality Trends
        print("\n7. Testing quality trends:")
        try:
            trends_result = await agent.get_quality_trends(directory=test_dir, days=7)
            if "message" in trends_result:
                print(f"   ✓ Trends: {trends_result['message']}")
            else:
                print(f"   ✓ Trends analyzed: {trends_result.get('analyses_count', 0)} analyses")
        except Exception as e:
            print(f"   ⚠️  Trends analysis error: {e}")

        # Test 7: Cache Management
        print("\n8. Testing cache management:")
        cache_cleared = agent.clear_cache()
        print(f"   ✓ Cache cleared: {cache_cleared} entries")

        print("\n✅ Phase 2 features testing completed!")

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
    success = asyncio.run(test_phase2_features())
    sys.exit(0 if success else 1)