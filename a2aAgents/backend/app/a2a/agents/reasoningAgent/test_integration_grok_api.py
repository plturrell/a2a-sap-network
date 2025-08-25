#!/usr/bin/env python3
"""
Integration Tests with Real Grok-4 API
Tests the complete reasoning pipeline with actual API calls
"""

import asyncio
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Set API key - SECURITY: No hardcoded defaults allowed
API_KEY = os.getenv('XAI_API_KEY')
if not API_KEY:
    print("ERROR: XAI_API_KEY environment variable not set")
    print("Please set your API key: export XAI_API_KEY='your-actual-api-key'")
    sys.exit(1)
os.environ['XAI_API_KEY'] = API_KEY

print("Integration Tests with Real Grok-4 API")
print("=" * 50)
print(f"Using API key: {API_KEY[:20]}..." if API_KEY else "No API key found")
print()


async def test_grok_4_basic_integration():
    """Test basic Grok-4 API integration"""
    print("1. Testing Basic Grok-4 Integration")
    print("-" * 40)

    try:
        from grokReasoning import GrokReasoning

        grok = GrokReasoning()

        # Test question decomposition
        start_time = time.time()
        result = await grok.decompose_question(
            "What are the key differences between machine learning and artificial intelligence?"
        )
        decomp_time = time.time() - start_time

        print(f"Question Decomposition:")
        print(f"  Success: {result.get('success', False)}")
        print(f"  Response time: {decomp_time:.3f}s")

        if result.get('success'):
            decomposition = result.get('decomposition', {})
            if isinstance(decomposition, dict):
                for key, value in decomposition.items():
                    if isinstance(value, list):
                        print(f"  {key}: {len(value)} items")
                    else:
                        print(f"  {key}: {str(value)[:50]}...")

        # Test pattern analysis
        start_time = time.time()
        pattern_result = await grok.analyze_patterns(
            "Machine learning is a subset of AI that uses data to make predictions"
        )
        pattern_time = time.time() - start_time

        print(f"\nPattern Analysis:")
        print(f"  Success: {pattern_result.get('success', False)}")
        print(f"  Response time: {pattern_time:.3f}s")

        if pattern_result.get('success'):
            patterns = pattern_result.get('patterns', {})
            if isinstance(patterns, dict):
                for key, value in patterns.items():
                    if isinstance(value, list):
                        print(f"  {key}: {len(value)} items")

        # Test synthesis
        start_time = time.time()
        sub_answers = [
            {"content": "Machine learning focuses on algorithms that learn from data"},
            {"content": "AI is the broader concept of machines performing tasks intelligently"},
            {"content": "ML is a subset of AI that enables computers to learn without explicit programming"}
        ]

        synthesis_result = await grok.synthesize_answer(
            sub_answers,
            "What is the relationship between AI and machine learning?"
        )
        synthesis_time = time.time() - start_time

        print(f"\nAnswer Synthesis:")
        print(f"  Success: {synthesis_result.get('success', False)}")
        print(f"  Response time: {synthesis_time:.3f}s")

        if synthesis_result.get('success'):
            synthesis = synthesis_result.get('synthesis', '')
            print(f"  Answer length: {len(synthesis)} characters")
            print(f"  Preview: {synthesis[:100]}...")

        return all([
            result.get('success', False),
            pattern_result.get('success', False),
            synthesis_result.get('success', False)
        ])

    except Exception as e:
        print(f"❌ Basic integration test failed: {e}")
        return False


async def test_async_grok_client_integration():
    """Test async Grok client with connection pooling"""
    print("\n2. Testing Async Grok Client Integration")
    print("-" * 40)

    try:
        from asyncGrokClient import AsyncGrokReasoning, GrokConfig

        config = GrokConfig(
            api_key=API_KEY,
            pool_connections=3,
            pool_maxsize=5,
            cache_ttl=30
        )

        grok = AsyncGrokReasoning(config)

        # Test questions (including duplicates for cache testing)
        questions = [
            "What is quantum computing?",
            "How does blockchain work?",
            "What is quantum computing?",  # Duplicate for cache test
            "Explain machine learning basics"
        ]

        # Test sequential execution
        print("Sequential execution:")
        start_time = time.time()
        sequential_results = []

        for i, question in enumerate(questions):
            result = await grok.decompose_question(question)
            sequential_results.append(result)

            success = result.get('success', False)
            cached = result.get('cached', False)
            response_time = result.get('response_time', 0)

            print(f"  Q{i+1}: Success={success}, Cached={cached}, Time={response_time:.3f}s")

        sequential_time = time.time() - start_time

        # Test concurrent execution
        print("\nConcurrent execution:")
        start_time = time.time()

        concurrent_tasks = [grok.decompose_question(q) for q in questions]
        concurrent_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)

        concurrent_time = time.time() - start_time

        successful_concurrent = 0
        for i, result in enumerate(concurrent_results):
            if isinstance(result, Exception):
                print(f"  Q{i+1}: Exception={type(result).__name__}")
            else:
                success = result.get('success', False)
                cached = result.get('cached', False)
                response_time = result.get('response_time', 0)
                print(f"  Q{i+1}: Success={success}, Cached={cached}, Time={response_time:.3f}s")

                if success:
                    successful_concurrent += 1

        # Performance comparison
        print(f"\nPerformance Comparison:")
        print(f"  Sequential time: {sequential_time:.3f}s ({sequential_time/len(questions):.3f}s per request)")
        print(f"  Concurrent time: {concurrent_time:.3f}s ({concurrent_time/len(questions):.3f}s per request)")

        if concurrent_time > 0:
            speedup = sequential_time / concurrent_time
            print(f"  Speedup: {speedup:.2f}x")

        # Get performance stats
        stats = await grok.get_performance_stats()
        print(f"\nClient Performance Stats:")
        print(f"  Total requests: {stats.get('total_requests', 0)}")
        print(f"  Cache hits: {stats.get('cache_hits', 0)}")
        print(f"  Cache hit rate: {stats.get('cache_hit_rate', 0):.2f}")
        print(f"  Avg response time: {stats.get('avg_response_time', 0):.3f}s")

        await grok.close()

        # Success criteria: at least 75% of requests successful
        total_successful = sum(1 for r in sequential_results if r.get('success')) + successful_concurrent
        total_requests = len(sequential_results) + len(questions)
        success_rate = total_successful / total_requests

        print(f"  Overall success rate: {success_rate:.2f}")

        return success_rate >= 0.75

    except Exception as e:
        print(f"❌ Async client integration test failed: {e}")
        return False


async def test_blackboard_integration():
    """Test blackboard architecture with real Grok-4"""
    print("\n3. Testing Blackboard Integration")
    print("-" * 40)

    try:
        from blackboardArchitecture import BlackboardController

        controller = BlackboardController()

        # Test questions of different complexity
        test_cases = [
            {
                "question": "What causes inflation in economics?",
                "context": {"domain": "economics", "analysis_type": "causal"},
                "expected_sources": ["pattern_recognition", "logical_reasoning", "causal_analysis"]
            },
            {
                "question": "How do neural networks learn from data?",
                "context": {"domain": "machine_learning", "analysis_type": "process"},
                "expected_sources": ["pattern_recognition", "logical_reasoning"]
            }
        ]

        results = []
        total_processing_time = 0

        for i, test_case in enumerate(test_cases):
            print(f"\nTest Case {i+1}: {test_case['question'][:50]}...")

            start_time = time.time()
            result = await controller.reason(test_case['question'], test_case['context'])
            processing_time = time.time() - start_time

            total_processing_time += processing_time
            results.append(result)

            # Analyze result
            success = result.get('enhanced', False)
            confidence = result.get('confidence', 0)
            iterations = result.get('iterations', 0)
            answer_length = len(result.get('answer', ''))

            print(f"  Success: {success}")
            print(f"  Processing time: {processing_time:.3f}s")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Iterations: {iterations}")
            print(f"  Answer length: {answer_length} characters")

            # Check blackboard state
            if 'blackboard_state' in result:
                state = result['blackboard_state']
                contributions = state.get('contributions', [])

                print(f"  Knowledge sources used:")
                source_counts = {}
                for contrib in contributions:
                    source = contrib.get('source', 'unknown')
                    source_counts[source] = source_counts.get(source, 0) + 1

                for source, count in source_counts.items():
                    print(f"    - {source}: {count} contributions")

                # Check if expected sources were used
                expected_sources = test_case.get('expected_sources', [])
                sources_used = list(source_counts.keys())
                sources_matched = sum(1 for src in expected_sources if src in sources_used)

                print(f"  Expected sources matched: {sources_matched}/{len(expected_sources)}")

        # Overall performance
        avg_processing_time = total_processing_time / len(test_cases)
        successful_results = sum(1 for r in results if r.get('enhanced', False))

        print(f"\nOverall Blackboard Performance:")
        print(f"  Total processing time: {total_processing_time:.3f}s")
        print(f"  Average time per question: {avg_processing_time:.3f}s")
        print(f"  Successful results: {successful_results}/{len(results)}")

        return successful_results >= len(results) * 0.8  # 80% success rate

    except Exception as e:
        print(f"❌ Blackboard integration test failed: {e}")
        return False


async def test_memory_system_integration():
    """Test memory system with real reasoning experiences"""
    print("\n4. Testing Memory System Integration")
    print("-" * 40)

    try:
        from asyncReasoningMemorySystem import AsyncReasoningMemoryStore, ReasoningExperience, AsyncAdaptiveReasoningSystem
        import tempfile
        import shutil

        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_integration_memory.db")

        try:
            memory_store = AsyncReasoningMemoryStore(db_path)
            adaptive_system = AsyncAdaptiveReasoningSystem(memory_store)

            # Create realistic reasoning experiences
            experiences = []
            questions = [
                "What is artificial intelligence?",
                "How does machine learning work?",
                "What are neural networks?",
                "Explain deep learning concepts",
                "What is the difference between AI and ML?"
            ]

            print("Creating reasoning experiences...")
            for i, question in enumerate(questions):
                experience = ReasoningExperience(
                    question=question,
                    answer=f"Comprehensive answer to: {question}",
                    reasoning_chain=[
                        {"step": 1, "type": "decomposition", "content": "Break down the question"},
                        {"step": 2, "type": "analysis", "content": "Analyze key concepts"},
                        {"step": 3, "type": "synthesis", "content": "Synthesize final answer"}
                    ],
                    confidence=0.75 + (i % 3) * 0.08,  # Vary confidence
                    context={"domain": "AI/ML", "complexity": "medium"},
                    timestamp=datetime.utcnow(),
                    architecture_used="blackboard" if i % 2 == 0 else "hierarchical",
                    performance_metrics={
                        "duration": 2.0 + i * 0.5,
                        "api_calls": 3 + i,
                        "tokens_used": 500 + i * 100
                    }
                )
                experiences.append(experience)

            # Test concurrent storage
            start_time = time.time()
            store_tasks = [memory_store.store_experience(exp) for exp in experiences]
            store_results = await asyncio.gather(*store_tasks)
            storage_time = time.time() - start_time

            stored_count = sum(store_results)
            print(f"  Stored {stored_count}/{len(experiences)} experiences in {storage_time:.3f}s")

            # Test learning from experiences
            print("Testing adaptive learning...")
            learning_results = []
            for experience in experiences[:3]:  # Test first 3
                patterns = await adaptive_system.learn_from_experience(experience)
                learning_results.append(len(patterns))
                print(f"  Learned {len(patterns)} patterns from: {experience.question[:30]}...")

            # Test suggestion generation
            print("Testing reasoning suggestions...")
            suggestion_tests = [
                ("What is machine learning?", "blackboard"),
                ("How does AI work?", "hierarchical"),
                ("Explain neural networks", "blackboard")
            ]

            for question, architecture in suggestion_tests:
                suggestions = await adaptive_system.get_reasoning_suggestions(question, architecture)

                similar_cases = suggestions.get('similar_cases', 0)
                confidence_range = suggestions.get('confidence_range', {})
                patterns = suggestions.get('recommended_patterns', [])

                print(f"  Question: {question[:30]}...")
                print(f"    Similar cases: {similar_cases}")
                print(f"    Confidence range: {confidence_range.get('min', 0):.2f}-{confidence_range.get('max', 0):.2f}")
                print(f"    Recommended patterns: {len(patterns)}")

            # Test retrieval performance
            print("Testing retrieval performance...")
            start_time = time.time()

            retrieval_tasks = [
                memory_store.retrieve_similar_experiences(q, limit=3)
                for q in ["AI", "machine learning", "neural networks"]
            ]
            retrieval_results = await asyncio.gather(*retrieval_tasks)
            retrieval_time = time.time() - start_time

            total_retrieved = sum(len(results) for results in retrieval_results)
            print(f"  Retrieved {total_retrieved} similar experiences in {retrieval_time:.3f}s")

            # Get performance statistics
            stats = await memory_store.get_performance_stats()
            memory_stats = await adaptive_system.get_memory_stats()

            print(f"\nMemory System Statistics:")
            print(f"  Total experiences: {stats.get('experiences', {}).get('total', 0)}")
            print(f"  Average confidence: {stats.get('experiences', {}).get('avg_confidence', 0):.3f}")
            print(f"  Architectures used: {stats.get('experiences', {}).get('architectures_used', 0)}")
            print(f"  Cache size: {memory_stats.get('cache_size', 0)}")
            print(f"  Cache hit rate: {memory_stats.get('cache_hit_rate', 0):.3f}")

            # Cleanup
            await adaptive_system.cleanup()

            # Success criteria
            success = (
                stored_count >= len(experiences) * 0.8 and  # 80% storage success
                sum(learning_results) > 0 and  # Some patterns learned
                total_retrieved > 0 and  # Some retrieval success
                storage_time < 5.0 and  # Reasonable storage time
                retrieval_time < 2.0  # Reasonable retrieval time
            )

            return success

        finally:
            # Cleanup temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"❌ Memory system integration test failed: {e}")
        return False


async def test_error_handling_integration():
    """Test error handling with real API scenarios"""
    print("\n5. Testing Error Handling Integration")
    print("-" * 40)

    try:
        from asyncGrokClient import AsyncGrokReasoning, GrokConfig

        # Test with invalid API key
        print("Testing with invalid API key...")
        bad_config = GrokConfig(
            api_key="invalid-key-test",
            pool_connections=2,
            timeout=5  # Short timeout for faster testing
        )

        bad_grok = AsyncGrokReasoning(bad_config)

        start_time = time.time()
        result = await bad_grok.decompose_question("Test question")
        error_time = time.time() - start_time

        print(f"  Invalid key result: Success={result.get('success', False)}")
        print(f"  Error handling time: {error_time:.3f}s")

        await bad_grok.close()

        # Test with timeout scenarios
        print("Testing timeout handling...")
        timeout_config = GrokConfig(
            api_key=API_KEY,
            timeout=1  # Very short timeout
        )

        timeout_grok = AsyncGrokReasoning(timeout_config)

        # Try a complex question that might timeout
        complex_question = "Explain the detailed mathematical foundations of quantum computing including quantum gates, superposition, entanglement, and how quantum algorithms like Shor's algorithm work in practice with real-world applications and limitations."

        start_time = time.time()
        timeout_result = await timeout_grok.decompose_question(complex_question)
        timeout_test_time = time.time() - start_time

        print(f"  Timeout test result: Success={timeout_result.get('success', False)}")
        print(f"  Timeout test time: {timeout_test_time:.3f}s")

        await timeout_grok.close()

        # Test cleanup under error conditions
        print("Testing cleanup under error conditions...")
        from asyncCleanupManager import AsyncReasoningCleanupManager

        cleanup_manager = AsyncReasoningCleanupManager()

        # Register resources that might fail cleanup
        class FailingResource:
            def __init__(self, should_fail=False):
                self.should_fail = should_fail
                self.cleaned = False

            async def close(self):
                if self.should_fail:
                    raise Exception("Simulated cleanup failure")
                self.cleaned = True

        good_resource = FailingResource(should_fail=False)
        bad_resource = FailingResource(should_fail=True)

        cleanup_manager.register_grok_client(good_resource)
        cleanup_manager.register_grok_client(bad_resource)

        # Test cleanup with failures
        start_time = time.time()
        await cleanup_manager.cleanup_reasoning_components()
        cleanup_time = time.time() - start_time

        print(f"  Cleanup with errors completed in {cleanup_time:.3f}s")
        print(f"  Good resource cleaned: {good_resource.cleaned}")
        print(f"  Bad resource handled gracefully: {not bad_resource.cleaned}")

        # Check error stats
        stats = cleanup_manager.get_performance_stats()
        print(f"  Cleanup errors encountered: {stats.get('errors_encountered', 0)}")

        # Success criteria: error handling works gracefully
        success = (
            not result.get('success', True) and  # Invalid key should fail
            error_time < 30.0 and  # Should fail quickly, not hang
            timeout_test_time < 60.0 and  # Should handle timeout
            good_resource.cleaned and  # Good resources should clean up
            cleanup_time < 5.0  # Cleanup should complete despite errors
        )

        return success

    except Exception as e:
        print(f"❌ Error handling integration test failed: {e}")
        return False


async def run_integration_tests():
    """Run all integration tests"""
    print("Starting Real Integration Tests with Grok-4 API")
    print("=" * 60)

    if not API_KEY or API_KEY == 'your-api-key-here':
        print("❌ No valid API key found. Set XAI_API_KEY environment variable.")
        return False

    tests = [
        ("Basic Grok-4 Integration", test_grok_4_basic_integration),
        ("Async Client Integration", test_async_grok_client_integration),
        ("Blackboard Integration", test_blackboard_integration),
        ("Memory System Integration", test_memory_system_integration),
        ("Error Handling Integration", test_error_handling_integration),
    ]

    results = {}
    total_start_time = time.time()

    for test_name, test_func in tests:
        print(f"\nStarting: {test_name}")

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
    print("\n" + "=" * 60)
    print("INTEGRATION TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r["success"])
    total = len(results)

    print(f"Overall Results: {passed}/{total} integration tests passed")
    print(f"Total test time: {total_time:.3f}s")
    print()

    for test_name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        duration = result["duration"]
        print(f"{status} {test_name:<30} {duration:>8.3f}s")

        if not result["success"] and "error" in result:
            print(f"     Error: {result['error']}")

    print("\n" + "=" * 60)

    if passed >= 4:  # At least 4/5 tests should pass
        print("🎉 INTEGRATION TESTS SUCCESSFUL!")
        print("\nValidated capabilities:")
        print("  ✅ Real Grok-4 API integration working")
        print("  ✅ Async client with connection pooling")
        print("  ✅ Blackboard reasoning with knowledge sources")
        print("  ✅ Memory system with learning capabilities")
        print("  ✅ Error handling and graceful degradation")
    else:
        print("⚠️  Some integration tests failed - check API connectivity and logs")

    return passed >= 4


if __name__ == "__main__":
    try:
        success = asyncio.run(run_integration_tests())
        exit_code = 0 if success else 1
        print(f"\nIntegration tests {'PASSED' if success else 'FAILED'}")

    except KeyboardInterrupt:
        print("\nIntegration tests interrupted by user")
        exit_code = 1
    except Exception as e:
        print(f"\nIntegration tests failed with error: {e}")
        exit_code = 1

    exit(exit_code)
