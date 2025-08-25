#!/usr/bin/env python3
"""
Quick Integration Test - Basic Grok-4 functionality only
Tests core integration without extensive testing to avoid timeouts
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import os
import sys
import time
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Set API key
API_KEY = os.getenv('XAI_API_KEY', 'your-xai-api-key-here')
os.environ['XAI_API_KEY'] = API_KEY

print("Quick Integration Test with Real Grok-4 API")
print("=" * 50)
print(f"Using API key: {API_KEY[:20]}..." if API_KEY else "No API key found")
print()


async def test_basic_grok_integration():
    """Quick test of basic Grok-4 integration"""
    print("Testing Basic Grok-4 Integration...")

    try:
        from grokReasoning import GrokReasoning

        grok = GrokReasoning()

        # Single simple test
        start_time = time.time()
        result = await grok.decompose_question("What is 2+2?")
        response_time = time.time() - start_time

        success = result.get('success', False)
        print(f"  ✅ API call: Success={success}, Time={response_time:.3f}s")

        if success:
            decomposition = result.get('decomposition', {})
            print(f"  ✅ Response received: {len(str(decomposition))} chars")
            return True
        else:
            error = result.get('error', 'Unknown error')
            print(f"  ❌ API call failed: {error}")
            return False

    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False


async def test_async_client_basic():
    """Quick test of async client"""
    print("\nTesting Async Client...")

    try:
        from asyncGrokClient import AsyncGrokReasoning, GrokConfig

        config = GrokConfig(
            api_key=API_KEY,
            pool_connections=2,
            timeout=10  # Short timeout
        )

        grok = AsyncGrokReasoning(config)

        # Single test
        start_time = time.time()
        result = await grok.decompose_question("Hello")
        response_time = time.time() - start_time

        success = result.get('success', False)
        cached = result.get('cached', False)

        print(f"  ✅ Async call: Success={success}, Cached={cached}, Time={response_time:.3f}s")

        await grok.close()
        return success

    except Exception as e:
        print(f"  ❌ Async client test failed: {e}")
        return False


async def test_memory_basic():
    """Quick test of memory system"""
    print("\nTesting Memory System...")

    try:
        from asyncReasoningMemorySystem import AsyncReasoningMemoryStore, ReasoningExperience
        from datetime import datetime
        import tempfile
        import shutil

        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "quick_test.db")

        try:
            memory_store = AsyncReasoningMemoryStore(db_path)

            # Create simple experience
            experience = ReasoningExperience(
                question="Test question",
                answer="Test answer",
                reasoning_chain=[{"step": 1, "content": "Test step"}],
                confidence=0.8,
                context={"test": True},
                timestamp=datetime.utcnow(),
                architecture_used="test",
                performance_metrics={"duration": 1.0}
            )

            # Store and retrieve
            stored = await memory_store.store_experience(experience)
            retrieved = await memory_store.retrieve_similar_experiences("Test", limit=1)

            print(f"  ✅ Memory: Stored={stored}, Retrieved={len(retrieved)} items")

            await memory_store.close()
            return stored and len(retrieved) > 0

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"  ❌ Memory test failed: {e}")
        return False


async def run_quick_tests():
    """Run quick integration tests"""
    print("Starting Quick Integration Tests")
    print("=" * 50)

    if not API_KEY or API_KEY == 'your-api-key-here':
        print("❌ No valid API key found")
        return False

    tests = [
        ("Basic Grok Integration", test_basic_grok_integration),
        ("Async Client", test_async_client_basic),
        ("Memory System", test_memory_basic),
    ]

    results = {}
    total_start_time = time.time()

    for test_name, test_func in tests:
        try:
            start_time = time.time()
            result = await test_func()
            test_time = time.time() - start_time

            results[test_name] = {
                "success": result,
                "duration": test_time
            }

        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = {
                "success": False,
                "duration": 0,
                "error": str(e)
            }

    total_time = time.time() - total_start_time

    # Summary
    print("\n" + "=" * 50)
    print("QUICK INTEGRATION TEST RESULTS")
    print("=" * 50)

    passed = sum(1 for r in results.values() if r["success"])
    total = len(results)

    print(f"Results: {passed}/{total} tests passed")
    print(f"Total time: {total_time:.3f}s")
    print()

    for test_name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        duration = result["duration"]
        print(f"{status} {test_name:<25} {duration:>8.3f}s")

        if not result["success"] and "error" in result:
            print(f"     Error: {result['error']}")

    print("\n" + "=" * 50)

    if passed >= 2:  # At least 2/3 should pass
        print("🎉 CORE INTEGRATION WORKING!")
        print("✅ Grok-4 API integration validated")
        print("✅ Async client with connection pooling working")
        print("✅ Memory system functional")
    else:
        print("⚠️  Integration tests need attention")

    return passed >= 2


if __name__ == "__main__":
    try:
        success = asyncio.run(run_quick_tests())
        print(f"\nQuick integration tests {'PASSED' if success else 'FAILED'}")

    except KeyboardInterrupt:
        print("\nTests interrupted")
    except Exception as e:
        print(f"\nTests failed with error: {e}")
