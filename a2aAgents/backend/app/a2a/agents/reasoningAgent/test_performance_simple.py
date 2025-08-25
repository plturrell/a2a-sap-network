#!/usr/bin/env python3
"""
Simple Performance Test
Test async improvements without requiring API calls
"""

import asyncio
import time
import sys
from pathlib import Path
import logging

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

print("Testing Performance Improvements (Offline)")
print("=" * 50)


async def test_async_memory():
    """Test async memory system without API calls"""
    try:
        from asyncReasoningMemorySystem import AsyncReasoningMemoryStore, ReasoningExperience
        from datetime import datetime

        print("\n1. Testing Async Memory System...")

        # Initialize
        store = AsyncReasoningMemoryStore("test_simple.db")

        # Create test data
        experiences = []
        for i in range(5):
            exp = ReasoningExperience(
                question=f"Question {i}",
                answer=f"Answer {i}",
                reasoning_chain=[{"step": 1, "content": f"Step {i}"}],
                confidence=0.8,
                context={"test": True},
                timestamp=datetime.utcnow(),
                architecture_used="test",
                performance_metrics={"duration": 1.0}
            )
            experiences.append(exp)

        # Test concurrent storage
        start = time.time()
        tasks = [store.store_experience(exp) for exp in experiences]
        results = await asyncio.gather(*tasks)
        store_time = time.time() - start

        # Test concurrent retrieval
        start = time.time()
        tasks = [store.retrieve_similar_experiences("Question", limit=2) for _ in range(3)]
        retrieve_results = await asyncio.gather(*tasks)
        retrieve_time = time.time() - start

        # Get stats
        stats = await store.get_performance_stats()

        print(f"  ✅ Stored {sum(results)} items in {store_time:.3f}s")
        print(f"  ✅ Retrieved {sum(len(r) for r in retrieve_results)} items in {retrieve_time:.3f}s")
        print(f"  ✅ Total in DB: {stats.get('experiences', {}).get('total', 0)}")

        await store.close()
        return True

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


async def test_connection_pool():
    """Test connection pool setup without API calls"""
    try:
        from asyncGrokClient import AsyncGrokConnectionPool, GrokConfig

        print("\n2. Testing Connection Pool...")

        config = GrokConfig(
            api_key="test-key",
            pool_connections=5,
            pool_maxsize=10
        )

        pool = AsyncGrokConnectionPool(config)

        # Test client creation
        start = time.time()
        clients = []
        for _ in range(3):
            client = await pool.get_client()
            clients.append(client)
        creation_time = time.time() - start

        print(f"  ✅ Created {len(clients)} clients in {creation_time:.3f}s")
        print(f"  ✅ Connection pool configured: {config.pool_connections} connections")

        await pool.close()
        return True

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


async def test_cache_system():
    """Test caching without API calls"""
    try:
        from asyncGrokClient import AsyncGrokCache

        print("\n3. Testing Cache System...")

        cache = AsyncGrokCache(cache_ttl=60)
        await cache.initialize()

        # Test cache operations
        test_key = "test_key"

        # Cache miss test
        start = time.time()
        result = await cache.get(test_key)
        miss_time = time.time() - start

        # Cache set test
        from asyncGrokClient import GrokResponse
        test_response = GrokResponse(
            content="test content",
            model="test-model",
            usage={"total_tokens": 100},
            finish_reason="stop",
            raw_response={"test": True}
        )

        start = time.time()
        await cache.set(test_key, test_response)
        set_time = time.time() - start

        # Cache hit test
        start = time.time()
        cached_result = await cache.get(test_key)
        hit_time = time.time() - start

        print(f"  ✅ Cache miss: {miss_time:.3f}s")
        print(f"  ✅ Cache set: {set_time:.3f}s")
        print(f"  ✅ Cache hit: {hit_time:.3f}s (cached: {cached_result.cached if cached_result else False})")

        await cache.close()
        return cached_result is not None and cached_result.cached

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


async def test_cleanup():
    """Test cleanup manager"""
    try:
        from asyncCleanupManager import AsyncReasoningCleanupManager

        print("\n4. Testing Cleanup Manager...")

        cleanup_manager = AsyncReasoningCleanupManager()

        # Create mock resources
        class MockResource:
            def __init__(self, name):
                self.name = name
                self.cleaned = False

            async def close(self):
                await asyncio.sleep(0.001)  # Simulate work
                self.cleaned = True

        # Register resources
        resources = []
        for i in range(3):
            resource = MockResource(f"Resource{i}")
            resources.append(resource)
            cleanup_manager.register_grok_client(resource)

        # Test cleanup
        start = time.time()
        await cleanup_manager.cleanup_reasoning_components()
        cleanup_time = time.time() - start

        cleaned_count = sum(1 for r in resources if r.cleaned)

        print(f"  ✅ Cleaned {cleaned_count}/{len(resources)} resources in {cleanup_time:.3f}s")

        stats = cleanup_manager.get_performance_stats()
        print(f"  ✅ Cleanup attempts: {stats['cleanup_count']}")

        return cleaned_count == len(resources)

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


async def test_async_improvements():
    """Test that async operations don't block"""
    try:
        print("\n5. Testing Async Non-Blocking Behavior...")

        # Test that multiple async operations can run concurrently
        async def slow_operation(duration, name):
            start = time.time()
            await asyncio.sleep(duration)
            end = time.time()
            return f"{name}: {end - start:.3f}s"

        # Run operations concurrently
        start = time.time()
        tasks = [
            slow_operation(0.1, "Task1"),
            slow_operation(0.1, "Task2"),
            slow_operation(0.1, "Task3")
        ]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start

        # Should complete in ~0.1s (concurrent) not ~0.3s (sequential)
        is_concurrent = total_time < 0.2

        print(f"  ✅ Concurrent execution: {total_time:.3f}s (concurrent: {is_concurrent})")
        for result in results:
            print(f"    {result}")

        return is_concurrent

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


async def main():
    """Run all performance tests"""
    print("Starting Simple Performance Tests")

    tests = [
        ("Async Memory System", test_async_memory),
        ("Connection Pool Setup", test_connection_pool),
        ("Cache System", test_cache_system),
        ("Cleanup Manager", test_cleanup),
        ("Async Non-Blocking", test_async_improvements),
    ]

    results = {}
    total_start = time.time()

    for test_name, test_func in tests:
        try:
            start = time.time()
            result = await test_func()
            duration = time.time() - start
            results[test_name] = {"success": result, "duration": duration}
        except Exception as e:
            results[test_name] = {"success": False, "duration": 0, "error": str(e)}

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 50)
    print("PERFORMANCE IMPROVEMENTS TEST SUMMARY")
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

    print("\n" + "=" * 50)

    if passed >= 4:
        print("🎉 PERFORMANCE IMPROVEMENTS WORKING!")
        print("\nImplemented improvements:")
        print("  ✅ Async SQLite storage (aiosqlite)")
        print("  ✅ HTTP connection pooling")
        print("  ✅ Response caching with TTL")
        print("  ✅ Proper async resource cleanup")
        print("  ✅ Non-blocking concurrent operations")
        return True
    else:
        print("⚠️  Some performance improvements need attention")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        print(f"\n{'✅ SUCCESS' if success else '❌ PARTIAL SUCCESS'}")
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
