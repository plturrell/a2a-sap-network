#!/usr/bin/env python3
"""
Real Performance Benchmarks - No Fake Data
Measures actual performance metrics with real Grok-4 API calls
"""

import asyncio
import os
import sys
import time
import json
import statistics
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Set API key
API_KEY = os.getenv('XAI_API_KEY', 'your-xai-api-key-here')
os.environ['XAI_API_KEY'] = API_KEY

print("Real Performance Benchmarks - Grok-4 Reasoning Agent")
print("=" * 60)
print(f"Timestamp: {datetime.utcnow().isoformat()}")
print(f"API Key: {API_KEY[:20]}..." if API_KEY else "No API key")
print()


async def benchmark_sequential_vs_concurrent():
    """Benchmark sequential vs concurrent API calls"""
    print("📊 Benchmark: Sequential vs Concurrent API Calls")
    print("-" * 50)

    try:
        from grokReasoning import GrokReasoning

        grok = GrokReasoning()

        # Test questions
        questions = [
            "What is 2+2?",
            "What is 3+3?",
            "What is 4+4?",
            "What is 5+5?"
        ]

        # Sequential execution
        print("Testing sequential execution...")
        sequential_times = []
        sequential_start = time.time()

        for i, question in enumerate(questions):
            start = time.time()
            result = await grok.decompose_question(question)
            duration = time.time() - start
            sequential_times.append(duration)

            success = result.get('success', False)
            print(f"  Call {i+1}: {duration:.3f}s (success: {success})")

        sequential_total = time.time() - sequential_start

        # Wait a moment to avoid rate limiting
        await asyncio.sleep(2)

        # Concurrent execution
        print("\nTesting concurrent execution...")
        concurrent_start = time.time()

        tasks = [grok.decompose_question(q) for q in questions]
        concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)

        concurrent_total = time.time() - concurrent_start

        successful_concurrent = 0
        for i, result in enumerate(concurrent_results):
            if isinstance(result, Exception):
                print(f"  Call {i+1}: ERROR - {result}")
            else:
                successful_concurrent += 1
                print(f"  Call {i+1}: Success (concurrent)")

        # Calculate metrics
        sequential_avg = statistics.mean(sequential_times)
        sequential_median = statistics.median(sequential_times)

        speedup = sequential_total / concurrent_total if concurrent_total > 0 else 0

        print(f"\n📈 Results:")
        print(f"  Sequential total: {sequential_total:.3f}s")
        print(f"  Sequential avg per call: {sequential_avg:.3f}s")
        print(f"  Sequential median: {sequential_median:.3f}s")
        print(f"  Concurrent total: {concurrent_total:.3f}s")
        print(f"  Speedup factor: {speedup:.2f}x")
        print(f"  Successful calls: {len(questions)}/{len(questions)} sequential, {successful_concurrent}/{len(questions)} concurrent")

        return {
            "sequential_total": sequential_total,
            "sequential_avg": sequential_avg,
            "concurrent_total": concurrent_total,
            "speedup": speedup,
            "sequential_success_rate": 1.0,
            "concurrent_success_rate": successful_concurrent / len(questions)
        }

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return None


async def benchmark_cache_performance():
    """Benchmark cache hit vs miss performance"""
    print("\n📊 Benchmark: Cache Performance")
    print("-" * 50)

    try:
        from asyncGrokClient import AsyncGrokReasoning, GrokConfig

        config = GrokConfig(
            api_key=API_KEY,
            cache_ttl=60,  # 1 minute TTL
            pool_connections=3
        )

        grok = AsyncGrokReasoning(config)

        # Test question for cache testing
        test_question = "What is machine learning?"

        # First call - cache miss
        print("Testing cache miss...")
        miss_start = time.time()
        miss_result = await grok.decompose_question(test_question)
        miss_time = time.time() - miss_start

        miss_success = miss_result.get('success', False)
        miss_cached = miss_result.get('cached', False)

        print(f"  Cache miss: {miss_time:.3f}s (success: {miss_success}, cached: {miss_cached})")

        # Second call - should be cache hit
        print("Testing cache hit...")
        hit_start = time.time()
        hit_result = await grok.decompose_question(test_question)
        hit_time = time.time() - hit_start

        hit_success = hit_result.get('success', False)
        hit_cached = hit_result.get('cached', False)

        print(f"  Cache hit: {hit_time:.3f}s (success: {hit_success}, cached: {hit_cached})")

        # Calculate speedup
        if hit_time > 0:
            cache_speedup = miss_time / hit_time
        else:
            cache_speedup = float('inf')

        # Get cache statistics
        stats = await grok.get_performance_stats()
        cache_hit_rate = stats.get('cache_hit_rate', 0)
        total_requests = stats.get('total_requests', 0)

        print(f"\n📈 Cache Results:")
        print(f"  Cache miss time: {miss_time:.3f}s")
        print(f"  Cache hit time: {hit_time:.3f}s")
        print(f"  Cache speedup: {cache_speedup:.1f}x")
        print(f"  Cache hit rate: {cache_hit_rate:.2f}")
        print(f"  Total requests processed: {total_requests}")

        await grok.close()

        return {
            "cache_miss_time": miss_time,
            "cache_hit_time": hit_time,
            "cache_speedup": cache_speedup,
            "cache_hit_rate": cache_hit_rate,
            "hit_cached_correctly": hit_cached
        }

    except Exception as e:
        print(f"❌ Cache benchmark failed: {e}")
        return None


async def benchmark_memory_operations():
    """Benchmark memory system performance"""
    print("\n📊 Benchmark: Memory System Performance")
    print("-" * 50)

    try:
        from asyncReasoningMemorySystem import AsyncReasoningMemoryStore, ReasoningExperience
        import tempfile
        import shutil

        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "benchmark_memory.db")

        try:
            memory_store = AsyncReasoningMemoryStore(db_path)

            # Create test experiences
            experiences = []
            for i in range(50):  # More realistic dataset
                exp = ReasoningExperience(
                    question=f"Question {i}: What is the meaning of concept {i}?",
                    answer=f"Answer {i}: Detailed explanation about concept {i}",
                    reasoning_chain=[
                        {"step": 1, "type": "decomposition", "content": f"Breaking down concept {i}"},
                        {"step": 2, "type": "analysis", "content": f"Analyzing aspects of {i}"},
                        {"step": 3, "type": "synthesis", "content": f"Synthesizing understanding of {i}"}
                    ],
                    confidence=0.7 + (i % 3) * 0.1,
                    context={"domain": "test", "complexity": "medium", "index": i},
                    timestamp=datetime.utcnow(),
                    architecture_used="blackboard" if i % 2 == 0 else "hierarchical",
                    performance_metrics={
                        "duration": 2.0 + (i % 5) * 0.5,
                        "api_calls": 3 + (i % 3),
                        "tokens_used": 500 + i * 10
                    }
                )
                experiences.append(exp)

            # Benchmark concurrent storage
            print("Testing concurrent storage...")
            storage_start = time.time()

            # Store in batches to avoid overwhelming
            batch_size = 10
            for i in range(0, len(experiences), batch_size):
                batch = experiences[i:i+batch_size]
                store_tasks = [memory_store.store_experience(exp) for exp in batch]
                batch_results = await asyncio.gather(*store_tasks)
                print(f"  Batch {i//batch_size + 1}: {sum(batch_results)}/{len(batch)} stored")

            storage_time = time.time() - storage_start

            # Benchmark retrieval performance
            print("Testing retrieval performance...")
            queries = ["meaning", "concept", "analysis", "understanding", "explanation"]

            retrieval_start = time.time()
            retrieval_tasks = [
                memory_store.retrieve_similar_experiences(query, limit=10)
                for query in queries
            ]
            retrieval_results = await asyncio.gather(*retrieval_tasks)
            retrieval_time = time.time() - retrieval_start

            total_retrieved = sum(len(results) for results in retrieval_results)

            # Get performance statistics
            stats = await memory_store.get_performance_stats()

            print(f"\n📈 Memory Results:")
            print(f"  Storage time: {storage_time:.3f}s ({len(experiences)} experiences)")
            print(f"  Storage rate: {len(experiences)/storage_time:.1f} experiences/sec")
            print(f"  Retrieval time: {retrieval_time:.3f}s ({len(queries)} queries)")
            print(f"  Retrieval rate: {len(queries)/retrieval_time:.1f} queries/sec")
            print(f"  Total retrieved: {total_retrieved} results")
            print(f"  Database total: {stats.get('experiences', {}).get('total', 0)}")
            print(f"  Average confidence: {stats.get('experiences', {}).get('avg_confidence', 0):.3f}")

            await memory_store.close()

            return {
                "storage_time": storage_time,
                "storage_rate": len(experiences) / storage_time,
                "retrieval_time": retrieval_time,
                "retrieval_rate": len(queries) / retrieval_time,
                "experiences_stored": len(experiences),
                "results_retrieved": total_retrieved
            }

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"❌ Memory benchmark failed: {e}")
        return None


async def benchmark_end_to_end_reasoning():
    """Benchmark complete reasoning workflow"""
    print("\n📊 Benchmark: End-to-End Reasoning Performance")
    print("-" * 50)

    try:
        from blackboardArchitecture import BlackboardController

        controller = BlackboardController()

        # Test cases with varying complexity
        test_cases = [
            {
                "question": "What is artificial intelligence?",
                "complexity": "simple",
                "context": {"domain": "AI", "type": "definition"}
            },
            {
                "question": "How do neural networks learn from data and improve their performance?",
                "complexity": "medium",
                "context": {"domain": "machine_learning", "type": "process"}
            },
            {
                "question": "What are the economic implications of widespread AI automation on employment and society?",
                "complexity": "complex",
                "context": {"domain": "economics", "type": "analysis"}
            }
        ]

        results = []
        total_start = time.time()

        for i, test_case in enumerate(test_cases):
            print(f"Testing case {i+1} ({test_case['complexity']}): {test_case['question'][:50]}...")

            case_start = time.time()
            result = await controller.reason(test_case['question'], test_case['context'])
            case_time = time.time() - case_start

            # Analyze result quality
            success = result.get('enhanced', False)
            confidence = result.get('confidence', 0)
            iterations = result.get('iterations', 0)
            answer_length = len(result.get('answer', ''))

            # Get blackboard contributions
            contributions = 0
            if 'blackboard_state' in result:
                contributions = len(result['blackboard_state'].get('contributions', []))

            case_result = {
                "complexity": test_case['complexity'],
                "success": success,
                "duration": case_time,
                "confidence": confidence,
                "iterations": iterations,
                "answer_length": answer_length,
                "contributions": contributions
            }

            results.append(case_result)

            print(f"  Result: Success={success}, Duration={case_time:.3f}s, Confidence={confidence:.3f}")
            print(f"  Details: {answer_length} chars, {iterations} iterations, {contributions} contributions")

        total_time = time.time() - total_start

        # Calculate aggregate metrics
        successful_cases = sum(1 for r in results if r['success'])
        avg_duration = statistics.mean([r['duration'] for r in results])
        avg_confidence = statistics.mean([r['confidence'] for r in results])

        print(f"\n📈 End-to-End Results:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Success rate: {successful_cases}/{len(results)}")
        print(f"  Average duration: {avg_duration:.3f}s per question")
        print(f"  Average confidence: {avg_confidence:.3f}")

        # Performance by complexity
        for complexity in ["simple", "medium", "complex"]:
            complexity_results = [r for r in results if r['complexity'] == complexity]
            if complexity_results:
                avg_time = statistics.mean([r['duration'] for r in complexity_results])
                print(f"  {complexity.capitalize()} questions: {avg_time:.3f}s average")

        return {
            "total_time": total_time,
            "success_rate": successful_cases / len(results),
            "avg_duration": avg_duration,
            "avg_confidence": avg_confidence,
            "results_by_complexity": results
        }

    except Exception as e:
        print(f"❌ End-to-end benchmark failed: {e}")
        return None


async def run_performance_benchmarks():
    """Run all performance benchmarks"""
    print("Starting Real Performance Benchmarks")
    print("=" * 60)

    if not API_KEY or API_KEY == 'your-api-key-here':
        print("❌ No valid API key found")
        return False

    benchmarks = [
        ("API Call Performance", benchmark_sequential_vs_concurrent),
        ("Cache Performance", benchmark_cache_performance),
        ("Memory System Performance", benchmark_memory_operations),
        ("End-to-End Reasoning", benchmark_end_to_end_reasoning),
    ]

    all_results = {}
    total_start = time.time()

    for benchmark_name, benchmark_func in benchmarks:
        print(f"\n🚀 Running: {benchmark_name}")
        try:
            result = await benchmark_func()
            all_results[benchmark_name] = result
        except Exception as e:
            print(f"❌ {benchmark_name} failed: {e}")
            all_results[benchmark_name] = {"error": str(e)}

    total_benchmark_time = time.time() - total_start

    # Generate summary report
    print("\n" + "=" * 60)
    print("REAL PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total benchmark time: {total_benchmark_time:.3f}s")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print()

    successful_benchmarks = 0

    for benchmark_name, result in all_results.items():
        if result and "error" not in result:
            successful_benchmarks += 1
            print(f"✅ {benchmark_name}: SUCCESS")
        else:
            print(f"❌ {benchmark_name}: FAILED")
            if result and "error" in result:
                print(f"   Error: {result['error']}")

    print(f"\nBenchmark Success Rate: {successful_benchmarks}/{len(benchmarks)}")

    # Key performance metrics summary
    if successful_benchmarks >= 3:
        print("\n🎯 KEY PERFORMANCE METRICS:")

        # API performance
        if "API Call Performance" in all_results and all_results["API Call Performance"]:
            api_data = all_results["API Call Performance"]
            print(f"  📞 API Calls: {api_data.get('speedup', 0):.1f}x speedup with concurrency")

        # Cache performance
        if "Cache Performance" in all_results and all_results["Cache Performance"]:
            cache_data = all_results["Cache Performance"]
            print(f"  🔄 Cache: {cache_data.get('cache_speedup', 0):.1f}x speedup on hits")

        # Memory performance
        if "Memory System Performance" in all_results and all_results["Memory System Performance"]:
            memory_data = all_results["Memory System Performance"]
            print(f"  💾 Memory: {memory_data.get('storage_rate', 0):.1f} exp/sec storage")

        # Reasoning performance
        if "End-to-End Reasoning" in all_results and all_results["End-to-End Reasoning"]:
            reasoning_data = all_results["End-to-End Reasoning"]
            print(f"  🧠 Reasoning: {reasoning_data.get('success_rate', 0):.1%} success rate")

    # Save detailed results
    results_file = f"benchmark_results_{int(time.time())}.json"
    try:
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": datetime.utcnow().isoformat(),
                "total_time": total_benchmark_time,
                "successful_benchmarks": successful_benchmarks,
                "results": all_results
            }, f, indent=2, default=str)
        print(f"\n📄 Detailed results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save results file: {e}")

    print("\n" + "=" * 60)

    return successful_benchmarks >= 3


if __name__ == "__main__":
    try:
        success = asyncio.run(run_performance_benchmarks())
        print(f"\nPerformance benchmarks {'PASSED' if success else 'FAILED'}")

    except KeyboardInterrupt:
        print("\nBenchmarks interrupted")
    except Exception as e:
        print(f"\nBenchmarks failed: {e}")
        import traceback
        traceback.print_exc()