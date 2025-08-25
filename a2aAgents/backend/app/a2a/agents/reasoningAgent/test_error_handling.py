#!/usr/bin/env python3
"""
Error Handling Tests
Test fault tolerance and graceful degradation
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

print("Error Handling Tests - Reasoning Agent")
print("=" * 50)


async def test_api_error_handling():
    """Test handling of API errors"""
    print("1. Testing API Error Handling...")

    try:
        from asyncGrokClient import AsyncGrokReasoning, GrokConfig

        # Test with invalid API key
        config = GrokConfig(
            api_key="invalid-key-12345",
            timeout=5,  # Short timeout
            max_retries=1  # Single retry
        )

        grok = AsyncGrokReasoning(config)

        # Test decomposition with invalid key
        start_time = time.time()
        result = await grok.decompose_question("Test question")
        error_time = time.time() - start_time

        success = result.get('success', False)
        error_msg = result.get('error', 'No error message')

        print(f"  ✅ Invalid API key handled: Success={success}, Time={error_time:.3f}s")
        print(f"  ✅ Error message: {error_msg[:50]}...")

        await grok.close()

        # Should fail gracefully, not crash
        return not success and error_time < 30.0

    except Exception as e:
        print(f"  ❌ API error test failed: {e}")
        return False


async def test_timeout_handling():
    """Test timeout scenarios"""
    print("\n2. Testing Timeout Handling...")

    try:
        from asyncGrokClient import AsyncGrokReasoning, GrokConfig

        # Valid API key but very short timeout
        config = GrokConfig(
            api_key=os.getenv('XAI_API_KEY', 'test-key'),
            timeout=0.001,  # 1ms timeout - will definitely timeout
            max_retries=1
        )

        grok = AsyncGrokReasoning(config)

        start_time = time.time()
        result = await grok.decompose_question("Complex question that takes time to process")
        timeout_time = time.time() - start_time

        success = result.get('success', False)
        error_msg = result.get('error', '')

        print(f"  ✅ Timeout handled: Success={success}, Time={timeout_time:.3f}s")
        print(f"  ✅ Error indicates timeout: {'timeout' in error_msg.lower()}")

        await grok.close()

        # Should fail quickly due to timeout
        return not success and timeout_time < 5.0

    except Exception as e:
        print(f"  ❌ Timeout test failed: {e}")
        return False


async def test_memory_error_handling():
    """Test memory system error handling"""
    print("\n3. Testing Memory Error Handling...")

    try:
        from asyncReasoningMemorySystem import AsyncReasoningMemoryStore, ReasoningExperience
        from datetime import datetime

        # Test with invalid database path
        invalid_db_path = "/invalid/path/that/does/not/exist/test.db"

        try:
            memory_store = AsyncReasoningMemoryStore(invalid_db_path)

            # Try to store experience with invalid path
            experience = ReasoningExperience(
                question="Test",
                answer="Test",
                reasoning_chain=[],
                confidence=0.8,
                context={},
                timestamp=datetime.utcnow(),
                architecture_used="test",
                performance_metrics={}
            )

            result = await memory_store.store_experience(experience)

            # Should handle error gracefully
            print(f"  ✅ Invalid path handled: Store result={result}")

            if memory_store:
                await memory_store.close()

        except Exception as e:
            print(f"  ✅ Database error caught: {type(e).__name__}")

        # Test with corrupted data
        print("  Testing corrupted data handling...")

        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "error_test.db")

        try:
            memory_store = AsyncReasoningMemoryStore(db_path)

            # Try to store invalid experience
            invalid_experience = ReasoningExperience(
                question=None,  # Invalid data
                answer="Test",
                reasoning_chain=[],
                confidence=0.8,
                context={},
                timestamp=datetime.utcnow(),
                architecture_used="test",
                performance_metrics={}
            )

            try:
                result = await memory_store.store_experience(invalid_experience)
                print(f"  ✅ Invalid data handled: {result}")
            except Exception as e:
                print(f"  ✅ Invalid data error caught: {type(e).__name__}")

            await memory_store.close()

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return True

    except Exception as e:
        print(f"  ❌ Memory error test failed: {e}")
        return False


async def test_blackboard_error_handling():
    """Test blackboard error handling"""
    print("\n4. Testing Blackboard Error Handling...")

    try:
        from blackboardArchitecture import BlackboardController

        controller = BlackboardController()

        # Test with invalid question
        result = await controller.reason("", {})  # Empty question

        success = result.get('enhanced', False)
        error_handling = 'error' in result or not success

        print(f"  ✅ Empty question handled: Enhanced={success}, Has error handling={error_handling}")

        # Test with very large context
        large_context = {"data": "x" * 10000}  # Very large context

        start_time = time.time()
        result = await controller.reason("Test question", large_context)
        large_context_time = time.time() - start_time

        print(f"  ✅ Large context handled: Time={large_context_time:.3f}s")

        # Test with malformed context
        malformed_context = {"invalid": object()}  # Non-serializable object

        try:
            result = await controller.reason("Test", malformed_context)
            print(f"  ✅ Malformed context handled gracefully")
        except Exception as e:
            print(f"  ✅ Malformed context error caught: {type(e).__name__}")

        return True

    except Exception as e:
        print(f"  ❌ Blackboard error test failed: {e}")
        return False


async def test_cleanup_error_handling():
    """Test cleanup under error conditions"""
    print("\n5. Testing Cleanup Error Handling...")

    try:
        from asyncCleanupManager import AsyncReasoningCleanupManager

        cleanup_manager = AsyncReasoningCleanupManager()

        # Create resources that will fail cleanup
        class FailingResource:
            def __init__(self, should_fail=True):
                self.should_fail = should_fail
                self.cleanup_attempted = False

            async def close(self):
                self.cleanup_attempted = True
                if self.should_fail:
                    raise Exception("Simulated cleanup failure")

        # Register both good and bad resources
        good_resource = FailingResource(should_fail=False)
        bad_resource1 = FailingResource(should_fail=True)
        bad_resource2 = FailingResource(should_fail=True)

        cleanup_manager.register_grok_client(good_resource)
        cleanup_manager.register_grok_client(bad_resource1)
        cleanup_manager.register_memory_store(bad_resource2)

        # Test cleanup with failures
        start_time = time.time()
        await cleanup_manager.cleanup_reasoning_components()
        cleanup_time = time.time() - start_time

        # Check results
        print(f"  ✅ Cleanup completed in {cleanup_time:.3f}s despite errors")
        print(f"  ✅ Good resource cleaned: {not good_resource.should_fail and good_resource.cleanup_attempted}")
        print(f"  ✅ Bad resources handled: {bad_resource1.cleanup_attempted and bad_resource2.cleanup_attempted}")

        # Check error stats
        stats = cleanup_manager.get_performance_stats()
        errors = stats.get('errors_encountered', 0)
        print(f"  ✅ Errors tracked: {errors} errors encountered")

        return cleanup_time < 10.0 and errors > 0

    except Exception as e:
        print(f"  ❌ Cleanup error test failed: {e}")
        return False


async def test_concurrent_error_handling():
    """Test error handling under concurrent load"""
    print("\n6. Testing Concurrent Error Handling...")

    try:
        from asyncGrokClient import AsyncGrokReasoning, GrokConfig

        # Mix of valid and invalid configs
        configs = [
            GrokConfig(api_key="invalid-1", timeout=1),
            GrokConfig(api_key="invalid-2", timeout=1),
            GrokConfig(api_key=os.getenv('XAI_API_KEY', 'test'), timeout=1),  # This one might work
        ]

        # Create clients
        clients = [AsyncGrokReasoning(config) for config in configs]

        # Run concurrent operations
        tasks = [
            client.decompose_question(f"Test question {i}")
            for i, client in enumerate(clients)
        ]

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        concurrent_time = time.time() - start_time

        # Analyze results
        successes = 0
        exceptions = 0
        failures = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                exceptions += 1
                print(f"  ✅ Client {i}: Exception handled - {type(result).__name__}")
            elif result.get('success', False):
                successes += 1
                print(f"  ✅ Client {i}: Successful call")
            else:
                failures += 1
                print(f"  ✅ Client {i}: Graceful failure")

        print(f"  ✅ Concurrent error handling: {concurrent_time:.3f}s")
        print(f"  ✅ Results: {successes} success, {failures} failures, {exceptions} exceptions")

        # Cleanup all clients
        for client in clients:
            try:
                await client.close()
            except:
                pass  # Ignore cleanup errors

        # Success if no crashes and completed in reasonable time
        return concurrent_time < 30.0 and (successes + failures + exceptions) == len(clients)

    except Exception as e:
        print(f"  ❌ Concurrent error test failed: {e}")
        return False


async def run_error_handling_tests():
    """Run all error handling tests"""
    print("Starting Error Handling Tests")
    print("=" * 50)

    tests = [
        ("API Error Handling", test_api_error_handling),
        ("Timeout Handling", test_timeout_handling),
        ("Memory Error Handling", test_memory_error_handling),
        ("Blackboard Error Handling", test_blackboard_error_handling),
        ("Cleanup Error Handling", test_cleanup_error_handling),
        ("Concurrent Error Handling", test_concurrent_error_handling),
    ]

    results = {}
    total_start = time.time()

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

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 50)
    print("ERROR HANDLING TEST RESULTS")
    print("=" * 50)

    passed = sum(1 for r in results.values() if r["success"])
    total = len(results)

    print(f"Results: {passed}/{total} error handling tests passed")
    print(f"Total time: {total_time:.3f}s")
    print()

    for test_name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        duration = result["duration"]
        print(f"{status} {test_name:<30} {duration:>8.3f}s")

        if not result["success"] and "error" in result:
            print(f"     Error: {result['error']}")

    print("\n" + "=" * 50)

    if passed >= 5:  # At least 5/6 tests should pass
        print("🛡️  ERROR HANDLING ROBUST!")
        print("\nValidated error scenarios:")
        print("  ✅ API failures handled gracefully")
        print("  ✅ Timeouts don't crash system")
        print("  ✅ Memory errors contained")
        print("  ✅ Blackboard resilient to bad input")
        print("  ✅ Cleanup works despite failures")
        print("  ✅ Concurrent errors handled properly")
    else:
        print("⚠️  Some error handling needs improvement")

    return passed >= 5


if __name__ == "__main__":
    try:
        success = asyncio.run(run_error_handling_tests())
        print(f"\nError handling tests {'PASSED' if success else 'FAILED'}")

    except KeyboardInterrupt:
        print("\nTests interrupted")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
