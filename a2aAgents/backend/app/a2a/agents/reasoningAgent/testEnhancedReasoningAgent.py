"""
Test suite for Enhanced Reasoning Agent with MCP integration
"""

import asyncio
import json
from typing import Dict, Any
from datetime import datetime

from .enhancedReasoningAgent import EnhancedReasoningAgent, ReasoningArchitecture, ReasoningStrategy


class EnhancedReasoningAgentTest:
    """Test suite for Enhanced Reasoning Agent"""

    def __init__(self):
        self.reasoning_agent = EnhancedReasoningAgent()
        self.test_results = []

    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all enhanced reasoning agent tests"""

        # Initialize agent
        await self.reasoning_agent.initialize()

        test_methods = [
            ("Chain of Thought Reasoning", self.test_chain_of_thought_reasoning),
            ("Tree of Thought Reasoning", self.test_tree_of_thought_reasoning),
            ("Graph of Thought Reasoning", self.test_graph_of_thought_reasoning),
            ("Reasoning Pattern Analysis", self.test_reasoning_pattern_analysis),
            ("Multi-Perspective Debate", self.test_reasoning_debate),
            ("Counterfactual Reasoning", self.test_counterfactual_reasoning),
            ("Consistency Validation", self.test_consistency_validation),
            ("MCP Resource Access", self.test_mcp_resources),
            ("Performance Metrics", self.test_performance_metrics),
            ("Caching System", self.test_caching_system)
        ]

        results = {
            "overall_success": True,
            "tests_passed": 0,
            "tests_failed": 0,
            "total_tests": len(test_methods),
            "test_results": [],
            "mcp_features": {},
            "performance_metrics": {}
        }

        for test_name, test_method in test_methods:
            try:
                test_result = await test_method()
                test_result["test_name"] = test_name
                results["test_results"].append(test_result)

                if test_result["success"]:
                    results["tests_passed"] += 1
                else:
                    results["tests_failed"] += 1
                    results["overall_success"] = False

            except Exception as e:
                results["test_results"].append({
                    "test_name": test_name,
                    "success": False,
                    "error": f"Test execution failed: {str(e)}"
                })
                results["tests_failed"] += 1
                results["overall_success"] = False

        # Verify MCP integration
        results["mcp_features"] = await self._verify_mcp_features()

        await self.reasoning_agent.shutdown()
        return results

    async def test_chain_of_thought_reasoning(self) -> Dict[str, Any]:
        """Test 1: Chain of thought reasoning"""
        try:
            # Test basic chain of thought
            result = await self.reasoning_agent.call_mcp_tool("execute_reasoning_chain", {
                "question": "What are the implications of artificial general intelligence?",
                "architecture": "chain_of_thought",
                "strategy": "deductive",
                "max_depth": 5
            })

            if not result.get("success"):
                return {"success": False, "error": "Chain of thought failed", "result": result}

            # Verify chain structure
            if "chain_id" not in result:
                return {"success": False, "error": "No chain ID returned"}

            chain_id = result["chain_id"]
            chain = self.reasoning_agent.reasoning_chains.get(chain_id)

            if not chain:
                return {"success": False, "error": "Chain not found in registry"}

            # Verify chain properties
            if len(chain.nodes) < 3:
                return {"success": False, "error": "Chain too short"}

            if chain.architecture != ReasoningArchitecture.CHAIN_OF_THOUGHT:
                return {"success": False, "error": "Wrong architecture"}

            return {
                "success": True,
                "chain_created": True,
                "chain_length": len(chain.nodes),
                "confidence": result["confidence"],
                "architecture_correct": True
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_tree_of_thought_reasoning(self) -> Dict[str, Any]:
        """Test 2: Tree of thought reasoning"""
        try:
            # Test tree of thought with branching
            result = await self.reasoning_agent.call_mcp_tool("execute_reasoning_chain", {
                "question": "How can we solve climate change?",
                "architecture": "tree_of_thought",
                "strategy": "inductive",
                "max_depth": 4
            })

            if not result.get("success"):
                return {"success": False, "error": "Tree of thought failed"}

            chain = self.reasoning_agent.reasoning_chains.get(result["chain_id"])

            # Verify tree structure
            branches = [n for n in chain.nodes.values() if len(n.children) > 1]
            if not branches:
                return {"success": False, "error": "No branching in tree"}

            return {
                "success": True,
                "tree_created": True,
                "total_nodes": result.get("tree_nodes", 0),
                "branches": len(branches),
                "has_branching": True
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_graph_of_thought_reasoning(self) -> Dict[str, Any]:
        """Test 3: Graph of thought reasoning"""
        try:
            # Test graph-based reasoning
            result = await self.reasoning_agent.call_mcp_tool("execute_reasoning_chain", {
                "question": "What is the relationship between consciousness and intelligence?",
                "architecture": "graph_of_thought",
                "strategy": "analogical",
                "max_depth": 6
            })

            if not result.get("success"):
                return {"success": False, "error": "Graph of thought failed"}

            # Verify graph connections
            connections = result.get("connections", 0)
            if connections == 0:
                return {"success": False, "error": "No connections in graph"}

            return {
                "success": True,
                "graph_created": True,
                "graph_nodes": result.get("graph_nodes", 0),
                "connections": connections,
                "has_connections": True
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_reasoning_pattern_analysis(self) -> Dict[str, Any]:
        """Test 4: Reasoning pattern analysis"""
        try:
            # Create a reasoning session first
            chain_result = await self.reasoning_agent.call_mcp_tool("execute_reasoning_chain", {
                "question": "Why do humans dream?",
                "architecture": "hierarchical",
                "strategy": "causal"
            })

            if not chain_result.get("success"):
                return {"success": False, "error": "Failed to create reasoning chain"}

            # Get session ID from active sessions
            session_id = None
            for sid, session in self.reasoning_agent.active_sessions.items():
                if session.question == "Why do humans dream?":
                    session_id = sid
                    break

            if not session_id:
                return {"success": False, "error": "Session not found"}

            # Analyze patterns
            analysis_result = await self.reasoning_agent.call_mcp_tool("analyze_reasoning_patterns", {
                "session_id": session_id,
                "pattern_types": ["logical", "causal"],
                "depth_analysis": True
            })

            if not analysis_result.get("success"):
                return {"success": False, "error": "Pattern analysis failed"}

            analysis = analysis_result["analysis"]

            return {
                "success": True,
                "patterns_analyzed": True,
                "patterns_found": len(analysis.get("patterns_found", {})),
                "has_depth_metrics": "depth_metrics" in analysis,
                "has_recommendations": len(analysis.get("recommendations", [])) > 0
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_reasoning_debate(self) -> Dict[str, Any]:
        """Test 5: Multi-perspective reasoning debate"""
        try:
            # Create positions for debate
            positions = [
                {
                    "perspective": "optimist",
                    "argument": "AI will solve major world problems",
                    "confidence": 0.8
                },
                {
                    "perspective": "pessimist",
                    "argument": "AI poses existential risks",
                    "confidence": 0.7
                },
                {
                    "perspective": "realist",
                    "argument": "AI will have mixed impacts",
                    "confidence": 0.9
                }
            ]

            # Conduct debate
            debate_result = await self.reasoning_agent.call_mcp_tool("conduct_reasoning_debate", {
                "positions": positions,
                "debate_structure": "dialectical",
                "max_rounds": 3,
                "convergence_threshold": 0.8
            })

            if not debate_result.get("success"):
                return {"success": False, "error": "Debate failed"}

            return {
                "success": True,
                "debate_completed": True,
                "rounds_conducted": debate_result["rounds_conducted"],
                "consensus_achieved": debate_result.get("consensus_achieved", False),
                "has_synthesis": "synthesized_conclusion" in debate_result
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_counterfactual_reasoning(self) -> Dict[str, Any]:
        """Test 6: Counterfactual reasoning generation"""
        try:
            # Test counterfactual scenarios
            result = await self.reasoning_agent.call_mcp_tool("generate_counterfactual_reasoning", {
                "original_premise": "Technology advances exponentially",
                "conclusion": "AI will surpass human intelligence",
                "num_counterfactuals": 3,
                "variation_types": ["negation", "modification", "substitution"]
            })

            if not result.get("success"):
                return {"success": False, "error": "Counterfactual generation failed"}

            counterfactuals = result.get("counterfactuals", [])
            if len(counterfactuals) != 3:
                return {"success": False, "error": f"Expected 3 counterfactuals, got {len(counterfactuals)}"}

            # Check variation types
            variation_types = [cf["variation_type"] for cf in counterfactuals]
            expected_types = ["negation", "modification", "substitution"]

            return {
                "success": True,
                "counterfactuals_generated": len(counterfactuals),
                "all_variations_present": all(vt in variation_types for vt in expected_types),
                "has_insights": "insights" in result
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_consistency_validation(self) -> Dict[str, Any]:
        """Test 7: Cross-chain consistency validation"""
        try:
            # Create multiple reasoning chains
            chain_ids = []

            for i in range(3):
                result = await self.reasoning_agent.call_mcp_tool("execute_reasoning_chain", {
                    "question": "What is consciousness?",
                    "architecture": "chain_of_thought",
                    "strategy": "deductive" if i == 0 else "inductive",
                    "max_depth": 4
                })
                if result.get("success"):
                    chain_ids.append(result["chain_id"])

            if len(chain_ids) < 2:
                return {"success": False, "error": "Failed to create enough chains for validation"}

            # Validate consistency
            validation_result = await self.reasoning_agent.call_mcp_tool("validate_reasoning_consistency", {
                "chain_ids": chain_ids,
                "validation_criteria": ["logical_consistency", "conclusion_alignment"],
                "strict_mode": False
            })

            if not validation_result.get("success"):
                return {"success": False, "error": "Consistency validation failed"}

            validation = validation_result["validation_results"]

            return {
                "success": True,
                "chains_validated": validation_result["chains_validated"],
                "overall_consistency": validation["overall_consistency"],
                "criteria_tested": len(validation["criteria_results"]),
                "has_recommendations": len(validation.get("recommendations", [])) > 0
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_mcp_resources(self) -> Dict[str, Any]:
        """Test 8: MCP resource access"""
        try:
            # Test all MCP resources
            resources_to_test = [
                "reasoning://active-sessions",
                "reasoning://chain-library",
                "reasoning://performance-metrics",
                "reasoning://evidence-cache"
            ]

            resource_results = {}
            all_accessible = True

            for resource_uri in resources_to_test:
                try:
                    resource_data = await self.reasoning_agent.get_mcp_resource(resource_uri)
                    resource_results[resource_uri] = {
                        "accessible": True,
                        "has_data": bool(resource_data),
                        "has_timestamp": "last_updated" in resource_data
                    }
                except Exception as e:
                    resource_results[resource_uri] = {
                        "accessible": False,
                        "error": str(e)
                    }
                    all_accessible = False

            return {
                "success": all_accessible,
                "resources_tested": len(resources_to_test),
                "all_accessible": all_accessible,
                "resource_results": resource_results
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_performance_metrics(self) -> Dict[str, Any]:
        """Test 9: Performance metrics tracking"""
        try:
            # Get performance metrics
            metrics_data = await self.reasoning_agent.get_mcp_resource("reasoning://performance-metrics")

            required_sections = ["summary", "architecture_usage", "strategy_usage", "cache_performance"]
            missing_sections = [s for s in required_sections if s not in metrics_data]

            if missing_sections:
                return {"success": False, "error": f"Missing metric sections: {missing_sections}"}

            # Verify metrics structure
            summary = metrics_data["summary"]
            if "total_sessions" not in summary:
                return {"success": False, "error": "Missing total_sessions in summary"}

            return {
                "success": True,
                "metrics_available": True,
                "total_sessions": summary["total_sessions"],
                "has_architecture_breakdown": bool(metrics_data["architecture_usage"]),
                "has_cache_metrics": bool(metrics_data["cache_performance"])
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_caching_system(self) -> Dict[str, Any]:
        """Test 10: Caching system functionality"""
        try:
            # Execute same question twice
            question = "What is the meaning of life?"

            # First execution (cache miss)
            result1 = await self.reasoning_agent.call_mcp_tool("execute_reasoning_chain", {
                "question": question,
                "architecture": "chain_of_thought",
                "strategy": "deductive",
                "enable_caching": True
            })

            initial_cache_misses = self.reasoning_agent.metrics["cache_misses"]

            # Second execution (should be cache hit)
            result2 = await self.reasoning_agent.call_mcp_tool("execute_reasoning_chain", {
                "question": question,
                "architecture": "chain_of_thought",
                "strategy": "deductive",
                "enable_caching": True
            })

            cache_hits_after = self.reasoning_agent.metrics["cache_hits"]

            # Verify caching worked
            cache_hit_occurred = cache_hits_after > 0

            # Check cache resource
            cache_data = await self.reasoning_agent.get_mcp_resource("reasoning://evidence-cache")

            return {
                "success": cache_hit_occurred,
                "cache_working": cache_hit_occurred,
                "cache_size": cache_data["cache_size"],
                "cache_hit_on_second_call": cache_hit_occurred
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _verify_mcp_features(self) -> Dict[str, Any]:
        """Verify MCP features are properly integrated"""
        mcp_tools = self.reasoning_agent.list_mcp_tools()
        mcp_resources = self.reasoning_agent.list_mcp_resources()

        return {
            "tools_count": len(mcp_tools),
            "resources_count": len(mcp_resources),
            "tools": [tool["name"] for tool in mcp_tools],
            "resources": [resource["name"] for resource in mcp_resources],
            "mcp_integration_complete": len(mcp_tools) >= 5 and len(mcp_resources) >= 4
        }


async def run_enhanced_reasoning_agent_tests():
    """Run comprehensive tests for Enhanced Reasoning Agent"""
    test_suite = EnhancedReasoningAgentTest()
    results = await test_suite.run_comprehensive_tests()

    print("=" * 80)
    print("ENHANCED REASONING AGENT TEST RESULTS")
    print("=" * 80)

    print(f"Overall Success: {'✅ PASS' if results['overall_success'] else '❌ FAIL'}")
    print(f"Tests Passed: {results['tests_passed']}/{results['total_tests']}")
    print(f"Tests Failed: {results['tests_failed']}/{results['total_tests']}")

    print("\n🧠 MCP FEATURES:")
    print("-" * 40)
    mcp_features = results.get("mcp_features", {})
    print(f"MCP Tools: {mcp_features.get('tools_count', 0)}")
    print(f"MCP Resources: {mcp_features.get('resources_count', 0)}")
    print(f"Integration Complete: {'✅' if mcp_features.get('mcp_integration_complete') else '❌'}")

    print("\n📋 DETAILED TEST RESULTS:")
    print("-" * 40)

    for test_result in results["test_results"]:
        status = "✅" if test_result["success"] else "❌"
        print(f"{status} {test_result['test_name']}")
        if not test_result["success"]:
            print(f"   Error: {test_result.get('error', 'Unknown error')}")

    if results["overall_success"]:
        print(f"\n🎉 ENHANCED REASONING AGENT PASSES ALL TESTS! 🎉")
        print("✅ Chain/Tree/Graph of Thought: IMPLEMENTED")
        print("✅ Pattern Analysis: WORKING")
        print("✅ Multi-Perspective Debate: FUNCTIONAL")
        print("✅ Counterfactual Reasoning: OPERATIONAL")
        print("✅ Consistency Validation: ACTIVE")
        print("✅ MCP Integration: COMPLETE")
    else:
        print(f"\n⚠️  Some tests failed - review implementation")

    return results


if __name__ == "__main__":
    asyncio.run(run_enhanced_reasoning_agent_tests())
