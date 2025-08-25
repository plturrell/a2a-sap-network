#!/usr/bin/env python3
"""
Real Integration Tests
Tests actual functionality without mocks
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


class RealIntegrationTests:
    """Integration tests that use real components"""

    def __init__(self):
        self.results = []
        self.test_mode = os.getenv("TEST_MODE", "true").lower() == "true"

    async def test_real_architectures(self):
        """Test all reasoning architectures with real implementations"""
        print("\n1. Testing Real Reasoning Architectures")
        print("=" * 50)

        test_question = "What are the benefits and risks of artificial intelligence?"

        architectures = [
            ("peer_to_peer", "peerToPeerArchitecture", "create_peer_to_peer_coordinator"),
            ("chain_of_thought", "chainOfThoughtArchitecture", "create_chain_of_thought_reasoner"),
            ("swarm", "swarmIntelligenceArchitecture", "create_swarm_intelligence_coordinator"),
            ("debate", "debateArchitecture", "create_debate_coordinator"),
            ("blackboard", "blackboardArchitecture", "create_blackboard_system"),
        ]

        for arch_name, module_name, factory_name in architectures:
            try:
                # Dynamically import architecture
                module = __import__(module_name, fromlist=[factory_name])
                factory = getattr(module, factory_name)

                # Create instance
                instance = factory()

                # Test reasoning
                print(f"\n  Testing {arch_name}...")
                start_time = datetime.utcnow()

                result = await instance.reason(test_question, {"test_mode": self.test_mode})

                execution_time = (datetime.utcnow() - start_time).total_seconds()

                # Validate result
                assert "answer" in result, f"{arch_name} missing answer"
                assert result["answer"], f"{arch_name} returned empty answer"
                assert "reasoning_type" in result, f"{arch_name} missing reasoning_type"

                print(f"    ✅ Success - {execution_time:.2f}s")
                print(f"    Answer preview: {result['answer'][:100]}...")

                self.results.append({
                    "test": f"architecture_{arch_name}",
                    "success": True,
                    "execution_time": execution_time
                })

            except Exception as e:
                print(f"    ❌ Failed: {e}")
                self.results.append({
                    "test": f"architecture_{arch_name}",
                    "success": False,
                    "error": str(e)
                })

    async def test_grok_integration(self):
        """Test real Grok-4 integration"""
        print("\n2. Testing Grok-4 Integration")
        print("=" * 50)

        try:
            from grokReasoning import GrokReasoning

            grok = GrokReasoning()

            # Test decomposition
            print("  Testing question decomposition...")
            result = await grok.decompose_question(
                "How do neural networks learn from data?",
                {"test_mode": self.test_mode}
            )

            assert result["success"], "Decomposition failed"
            assert "sub_questions" in result["decomposition"], "Missing sub_questions"

            print("    ✅ Decomposition successful")

            # Test pattern analysis
            print("  Testing pattern analysis...")
            result = await grok.analyze_patterns(
                "Machine learning involves training models on data"
            )

            assert result["success"], "Pattern analysis failed"
            assert "patterns" in result, "Missing patterns"

            print("    ✅ Pattern analysis successful")

            # Test synthesis
            print("  Testing answer synthesis...")
            sub_answers = [
                {"content": "Neural networks use backpropagation", "confidence": 0.8},
                {"content": "Weights are adjusted based on errors", "confidence": 0.85}
            ]

            result = await grok.synthesize_answer(
                sub_answers,
                "How do neural networks learn?"
            )

            assert result["success"], "Synthesis failed"
            assert "synthesis" in result, "Missing synthesis"

            print("    ✅ Synthesis successful")

            self.results.append({
                "test": "grok_integration",
                "success": True
            })

        except Exception as e:
            print(f"    ❌ Failed: {e}")
            self.results.append({
                "test": "grok_integration",
                "success": False,
                "error": str(e)
            })

    async def test_embedding_patterns(self):
        """Test embedding-based pattern matching"""
        print("\n3. Testing Embedding Pattern Matching")
        print("=" * 50)

        try:
            from embeddingPatternMatcher import EnhancedNLPPatternMatcher

            matcher = EnhancedNLPPatternMatcher()

            test_texts = [
                "Why does water boil at 100 degrees Celsius?",
                "Compare machine learning and deep learning",
                "How to implement a binary search algorithm",
                "What is quantum computing?"
            ]

            for text in test_texts:
                print(f"\n  Analyzing: '{text[:50]}...'")

                result = await matcher.analyze_patterns(text)

                assert "semantic_analysis" in result, "Missing semantic analysis"
                assert "combined_confidence" in result, "Missing confidence"
                assert result["combined_confidence"] > 0, "Zero confidence"

                print(f"    Domain: {result['semantic_analysis']['primary_domain']}")
                print(f"    Approach: {result['recommended_approach']}")
                print(f"    Confidence: {result['combined_confidence']:.2f}")

            # Test similarity
            print("\n  Testing semantic similarity...")
            sim1 = await matcher.embedding_matcher.compute_pattern_similarity(
                "How does machine learning work?",
                "What is the process of training ML models?"
            )

            sim2 = await matcher.embedding_matcher.compute_pattern_similarity(
                "How does machine learning work?",
                "What is the capital of France?"
            )

            assert sim1 > sim2, "Similar texts should have higher similarity"
            print(f"    Similar texts: {sim1:.3f}")
            print(f"    Different texts: {sim2:.3f}")
            print("    ✅ Similarity test passed")

            self.results.append({
                "test": "embedding_patterns",
                "success": True
            })

        except Exception as e:
            print(f"    ❌ Failed: {e}")
            self.results.append({
                "test": "embedding_patterns",
                "success": False,
                "error": str(e)
            })

    async def test_async_memory(self):
        """Test async memory system"""
        print("\n4. Testing Async Memory System")
        print("=" * 50)

        try:
            from asyncReasoningMemorySystem import AsyncReasoningMemorySystem

            memory = AsyncReasoningMemorySystem()
            await memory.initialize()

            # Store experience
            print("  Storing reasoning experience...")
            from asyncReasoningMemorySystem import ReasoningExperience

            experience = ReasoningExperience(
                question="Test question",
                answer="Test answer",
                reasoning_chain=[{"step": 1, "content": "Test reasoning"}],
                confidence=0.85,
                context={"test": True},
                timestamp=datetime.utcnow(),
                architecture_used="test",
                performance_metrics={"time": 1.5}
            )

            success = await memory.store_experience(experience)
            assert success, "Failed to store experience"
            print("    ✅ Experience stored")

            # Retrieve similar
            print("  Retrieving similar experiences...")
            similar = await memory.get_similar_experiences("Test question", limit=5)
            assert len(similar) > 0, "No similar experiences found"
            print(f"    ✅ Found {len(similar)} similar experiences")

            # Learn pattern
            print("  Learning from experience...")
            pattern = await memory.learn_from_experience(experience)
            assert pattern is not None, "Failed to learn pattern"
            print("    ✅ Pattern learned")

            self.results.append({
                "test": "async_memory",
                "success": True
            })

        except Exception as e:
            print(f"    ❌ Failed: {e}")
            self.results.append({
                "test": "async_memory",
                "success": False,
                "error": str(e)
            })

    async def test_end_to_end_reasoning(self):
        """Test complete reasoning flow"""
        print("\n5. Testing End-to-End Reasoning Flow")
        print("=" * 50)

        try:
            # This would test the full reasoning agent
            # For now, test the clean architecture components

            print("  Testing MCP skill separation...")
            from skills import MCP_SKILLS

            assert len(MCP_SKILLS) >= 4, "Missing MCP skills"
            print(f"    ✅ Found {len(MCP_SKILLS)} MCP skills")

            # Test skill has MCP decorator
            from skills import advanced_reasoning
            assert hasattr(advanced_reasoning, '_mcp_tool'), "Missing MCP decorator"
            print("    ✅ Skills properly decorated")

            # Test clean agent structure
            from reasoningAgentClean import ReasoningAgent
            agent = ReasoningAgent("TestAgent")

            # Verify no skills in agent
            agent_methods = dir(agent)
            skill_contamination = [m for m in ['advanced_reasoning', 'hypothesis_generation']
                                 if m in agent_methods]

            assert not skill_contamination, f"Agent contains skills: {skill_contamination}"
            print("    ✅ Agent properly separated from skills")

            self.results.append({
                "test": "end_to_end",
                "success": True
            })

        except Exception as e:
            print(f"    ❌ Failed: {e}")
            self.results.append({
                "test": "end_to_end",
                "success": False,
                "error": str(e)
            })

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)

        total = len(self.results)
        passed = sum(1 for r in self.results if r["success"])
        failed = total - passed

        print(f"\nTotal Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")

        if failed > 0:
            print("\nFailed Tests:")
            for result in self.results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result.get('error', 'Unknown error')}")

        print(f"\nSuccess Rate: {(passed/total)*100:.1f}%")

        # Save results
        with open("integration_test_results.json", "w") as f:
            json.dump({
                "timestamp": datetime.utcnow().isoformat(),
                "summary": {
                    "total": total,
                    "passed": passed,
                    "failed": failed,
                    "success_rate": passed/total
                },
                "results": self.results
            }, f, indent=2)

        return passed == total


async def main():
    """Run all integration tests"""
    print("🧪 Real Integration Tests (No Mocks)")
    print("=" * 50)

    if os.getenv("TEST_MODE", "true").lower() == "true":
        print("ℹ️  Running in TEST MODE (no real API calls)")
    else:
        print("⚠️  Running with REAL API calls")

    tester = RealIntegrationTests()

    # Run tests
    await tester.test_real_architectures()
    await tester.test_grok_integration()
    await tester.test_embedding_patterns()
    await tester.test_async_memory()
    await tester.test_end_to_end_reasoning()

    # Print summary
    success = tester.print_summary()

    if success:
        print("\n✅ All integration tests passed!")
    else:
        print("\n❌ Some tests failed - check results above")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())