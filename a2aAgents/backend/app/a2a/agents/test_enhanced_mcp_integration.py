"""
Comprehensive Test Suite for Enhanced MCP Tool Integration
Tests the MCP tool usage patterns implemented in high-priority agents
"""

import asyncio
import json
import logging
import pytest
from typing import Dict, List, Any
from datetime import datetime
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from reasoningAgent.enhancedMcpToolIntegration import EnhancedMCPReasoningAgent, create_enhanced_mcp_reasoning_agent
from agentManager.active.enhancedMcpAgentManager import EnhancedMCPAgentManager, create_enhanced_mcp_agent_manager


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
logger = logging.getLogger(__name__)


class TestEnhancedMCPIntegration:
    """Test suite for enhanced MCP tool integration"""

    @pytest.fixture
    async def reasoning_agent(self):
        """Create test reasoning agent"""
        agent = create_enhanced_mcp_reasoning_agent(os.getenv("A2A_SERVICE_URL"))
        await agent.initialize()
        return agent

    @pytest.fixture
    async def agent_manager(self):
        """Create test agent manager"""
        manager = create_enhanced_mcp_agent_manager(os.getenv("A2A_SERVICE_URL"))
        await manager.initialize()
        return manager

    @pytest.mark.asyncio
    async def test_mcp_reasoning_analysis(self, reasoning_agent):
        """Test enhanced reasoning analysis using MCP tools"""

        question = "How do machine learning algorithms improve through iterative training?"

        result = await reasoning_agent.enhanced_reasoning_analysis(
            question=question,
            analysis_depth="comprehensive",
            use_cross_agent_tools=True,
            performance_tracking=True
        )

        # Verify successful execution
        assert result["status"] == "success"
        assert result["question"] == question
        assert result["analysis_depth"] == "comprehensive"

        # Verify MCP tool usage
        assert "mcp_tools_used" in result
        expected_tools = [
            "validate_reasoning_input",
            "decompose_question",
            "analyze_patterns",
            "assess_reasoning_quality",
            "measure_performance_metrics"
        ]

        for tool in expected_tools:
            assert tool in result["mcp_tools_used"]

        # Verify decomposition results
        assert "decomposition" in result
        assert "sub_questions" in result["decomposition"]

        # Verify pattern analysis
        assert "pattern_analysis" in result

        # Verify reasoning chain
        assert "reasoning_chain" in result
        assert len(result["reasoning_chain"]) > 0

        # Verify quality assessment
        assert "quality_assessment" in result
        assert "overall_score" in result["quality_assessment"]

        # Verify performance metrics
        assert "performance_metrics" in result
        assert "duration_ms" in result["performance_metrics"]

        # Verify final answer
        assert "final_answer" in result
        assert "confidence" in result["final_answer"]

        logger.info(f"✅ MCP Reasoning Analysis test passed - Session: {result['session_id']}")

    @pytest.mark.asyncio
    async def test_mcp_cross_agent_collaboration(self, reasoning_agent):
        """Test cross-agent collaboration using MCP protocol"""

        task = "Analyze the relationship between economic indicators and market volatility"
        target_agents = ["agent_0_data_product", "calculation_agent", "reasoning_agent"]

        result = await reasoning_agent.cross_agent_collaboration(
            task=task,
            target_agents=target_agents,
            collaboration_mode="sequential",
            timeout_seconds=30
        )

        # Verify successful execution
        assert result["status"] == "success"
        assert result["task"] == task
        assert result["collaboration_mode"] == "sequential"
        assert result["target_agents"] == target_agents
        assert result["mcp_protocol_used"] is True

        # Verify individual results
        assert "individual_results" in result
        assert len(result["individual_results"]) == len(target_agents)

        for agent_id in target_agents:
            assert agent_id in result["individual_results"]
            agent_result = result["individual_results"][agent_id]
            assert "response" in agent_result

        # Verify synthesis
        assert "synthesis" in result
        assert "task" in result["synthesis"]
        assert "agents_participated" in result["synthesis"]
        assert "average_confidence" in result["synthesis"]

        # Verify performance tracking
        assert "performance_metrics" in result

        logger.info(f"✅ MCP Cross-Agent Collaboration test passed - Collaboration: {result['collaboration_id']}")

    @pytest.mark.asyncio
    async def test_mcp_reasoning_resources(self, reasoning_agent):
        """Test MCP resource access"""

        # Test session data resource
        session_data = await reasoning_agent.get_reasoning_session_data()

        assert "active_sessions" in session_data
        assert "total_sessions" in session_data
        assert "performance_metrics" in session_data
        assert "last_updated" in session_data

        logger.info("✅ MCP Reasoning Resources test passed")

    @pytest.mark.asyncio
    async def test_mcp_reasoning_prompts(self, reasoning_agent):
        """Test MCP prompt generation"""

        # Test reasoning prompt generation
        prompt = await reasoning_agent.generate_reasoning_prompt(
            question_type="analytical",
            domain="machine_learning",
            complexity="high"
        )

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "machine_learning" in prompt.lower()
        assert "analytical" in prompt.lower()

        logger.info("✅ MCP Reasoning Prompts test passed")

    @pytest.mark.asyncio
    async def test_mcp_agent_orchestration(self, agent_manager):
        """Test enhanced agent orchestration using MCP"""

        workflow_definition = {
            "type": "data_processing_pipeline",
            "steps": [
                {
                    "name": "data_validation",
                    "task": "Validate input data format and quality",
                    "required_capabilities": ["validation", "data_processing"],
                    "quality_criteria": {"completeness": 0.95, "accuracy": 0.98}
                },
                {
                    "name": "data_standardization",
                    "task": "Standardize data according to schema",
                    "required_capabilities": ["standardization", "schema_validation"],
                    "quality_criteria": {"consistency": 0.9}
                },
                {
                    "name": "calculation_processing",
                    "task": "Process calculations on standardized data",
                    "required_capabilities": ["calculation", "mathematical_computation"],
                    "quality_criteria": {"precision": 0.99}
                }
            ]
        }

        agents_involved = ["agent_0_data_product", "agent_1_standardization", "calculation_agent"]

        result = await agent_manager.enhanced_agent_orchestration(
            workflow_definition=workflow_definition,
            agents_involved=agents_involved,
            coordination_mode="sequential",
            timeout_minutes=5,
            quality_gates=True,
            performance_monitoring=True
        )

        # Verify successful execution
        assert result["status"] == "success"
        assert result["coordination_mode"] == "sequential"
        assert result["agents_involved"] == agents_involved

        # Verify MCP tool usage
        assert "mcp_tools_used" in result
        expected_tools = [
            "validate_workflow_definition",
            "verify_agent_availability",
            "execute_workflow_steps",
            "assess_orchestration_quality",
            "measure_performance_metrics"
        ]

        for tool in expected_tools:
            assert tool in result["mcp_tools_used"]

        # Verify execution results
        assert "execution_result" in result
        execution_result = result["execution_result"]
        assert "step_results" in execution_result
        assert execution_result["total_steps"] == len(workflow_definition["steps"])
        assert execution_result["completed_steps"] >= 0

        # Verify quality assessment
        if "quality_assessment" in execution_result:
            assert "overall_score" in execution_result["quality_assessment"]

        # Verify performance metrics
        assert "performance_metrics" in result

        logger.info(f"✅ MCP Agent Orchestration test passed - Orchestration: {result['orchestration_id']}")

    @pytest.mark.asyncio
    async def test_mcp_intelligent_agent_discovery(self, agent_manager):
        """Test intelligent agent discovery using MCP"""

        capabilities_required = ["data_processing", "calculation", "validation"]

        result = await agent_manager.intelligent_agent_discovery(
            capabilities_required=capabilities_required,
            discovery_criteria={"performance_threshold": 0.8},
            performance_requirements={"max_cpu_usage": 0.7, "max_memory_usage": 0.8},
            include_health_check=True,
            analyze_compatibility=True
        )

        # Verify successful execution
        assert result["status"] == "success"
        assert result["capabilities_required"] == capabilities_required

        # Verify discovery results
        assert "total_agents_scanned" in result
        assert "matching_agents_count" in result
        assert "ranked_agents" in result

        # Verify MCP tool usage
        assert "mcp_tools_used" in result
        expected_tools = [
            "scan_agents_via_mcp",
            "validate_agent_capabilities",
            "check_agent_health",
            "analyze_agent_compatibility",
            "rank_agents_by_suitability"
        ]

        for tool in expected_tools:
            assert tool in result["mcp_tools_used"]

        # Verify health results
        assert "health_results" in result

        # Verify ranking
        ranked_agents = result["ranked_agents"]
        assert "ranked_list" in ranked_agents

        logger.info(f"✅ MCP Intelligent Agent Discovery test passed - Discovery: {result['discovery_id']}")

    @pytest.mark.asyncio
    async def test_mcp_adaptive_load_balancing(self, agent_manager):
        """Test adaptive load balancing using MCP"""

        workload = {
            "type": "batch_processing",
            "tasks": [
                {"id": "task_1", "complexity": "medium", "estimated_duration": 30},
                {"id": "task_2", "complexity": "high", "estimated_duration": 60},
                {"id": "task_3", "complexity": "low", "estimated_duration": 15}
            ],
            "priority": "normal"
        }

        target_agents = ["agent_0_data_product", "calculation_agent", "reasoning_agent"]

        result = await agent_manager.adaptive_load_balancing(
            workload=workload,
            target_agents=target_agents,
            balancing_strategy="adaptive",
            monitor_performance=True,
            auto_rebalance=True
        )

        # Verify successful execution
        assert result["status"] == "success"
        assert result["strategy_used"] == "adaptive"
        assert result["target_agents"] == target_agents

        # Verify MCP tool usage
        assert "mcp_tools_used" in result
        expected_tools = [
            "get_agent_load_metrics",
            "analyze_workload_characteristics",
            "calculate_load_distribution",
            "distribute_work_to_agent",
            "measure_agent_performance"
        ]

        for tool in expected_tools:
            assert tool in result["mcp_tools_used"]

        # Verify load analysis
        assert "agent_loads_before" in result
        assert "workload_analysis" in result
        assert "distribution_plan" in result
        assert "distribution_results" in result

        # Verify monitoring
        if result.get("monitoring_results"):
            assert len(result["monitoring_results"]) <= len(target_agents)

        logger.info(f"✅ MCP Adaptive Load Balancing test passed - Balancing: {result['balancing_id']}")

    @pytest.mark.asyncio
    async def test_mcp_agent_manager_resources(self, agent_manager):
        """Test agent manager MCP resources"""

        # Test orchestration sessions resource
        sessions_data = await agent_manager.get_orchestration_sessions()

        assert "active_sessions" in sessions_data
        assert "total_sessions" in sessions_data
        assert "performance_tracking" in sessions_data
        assert "last_updated" in sessions_data

        # Test agent registry resource
        registry_data = await agent_manager.get_agent_registry()

        assert "registered_agents" in registry_data
        assert "total_agents" in registry_data
        assert "capability_summary" in registry_data
        assert "health_summary" in registry_data
        assert "last_updated" in registry_data

        logger.info("✅ MCP Agent Manager Resources test passed")

    @pytest.mark.asyncio
    async def test_mcp_agent_coordination_advisor(self, agent_manager):
        """Test agent coordination advisor prompt"""

        scenario = "Coordinate multiple agents for real-time financial risk analysis"
        agents_available = ["calculation_agent", "reasoning_agent", "agent_0_data_product"]
        requirements = {
            "real_time": True,
            "accuracy_threshold": 0.95,
            "max_latency_ms": 1000
        }

        advice = await agent_manager.agent_coordination_advisor_prompt(
            coordination_scenario=scenario,
            agents_available=agents_available,
            requirements=requirements
        )

        assert isinstance(advice, str)
        assert len(advice) > 0
        assert "financial risk analysis" in advice.lower()
        assert "coordination" in advice.lower()

        logger.info("✅ MCP Agent Coordination Advisor test passed")

    @pytest.mark.asyncio
    async def test_mcp_tool_error_handling(self, reasoning_agent):
        """Test MCP tool error handling"""

        # Test with invalid input
        result = await reasoning_agent.enhanced_reasoning_analysis(
            question="",  # Empty question should trigger validation error
            analysis_depth="comprehensive"
        )

        # Should handle error gracefully
        assert result["status"] == "error"
        assert "validation" in result["error"].lower()

        logger.info("✅ MCP Tool Error Handling test passed")

    @pytest.mark.asyncio
    async def test_mcp_performance_tracking(self, reasoning_agent, agent_manager):
        """Test MCP performance tracking across operations"""

        # Perform multiple operations to generate performance data
        question = "What are the key factors in distributed system design?"

        # Reasoning analysis
        reasoning_result = await reasoning_agent.enhanced_reasoning_analysis(
            question=question,
            performance_tracking=True
        )

        # Agent discovery
        discovery_result = await agent_manager.intelligent_agent_discovery(
            capabilities_required=["system_design", "distributed_computing"]
        )

        # Verify performance metrics are captured
        assert "performance_metrics" in reasoning_result
        assert "duration_ms" in reasoning_result["performance_metrics"]

        assert "discovery_duration" in discovery_result
        assert discovery_result["discovery_duration"] > 0

        logger.info("✅ MCP Performance Tracking test passed")


# Integration test runner
async def run_enhanced_mcp_integration_tests():
    """Run all enhanced MCP integration tests"""

    print("🚀 Starting Enhanced MCP Integration Tests")
    print("=" * 60)

    # Initialize test instances
    reasoning_agent = create_enhanced_mcp_reasoning_agent(os.getenv("A2A_SERVICE_URL"))
    agent_manager = create_enhanced_mcp_agent_manager(os.getenv("A2A_SERVICE_URL"))

    await reasoning_agent.initialize()
    await agent_manager.initialize()

    test_suite = TestEnhancedMCPIntegration()

    # Test results tracking
    test_results = {
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "test_details": []
    }

    # List of tests to run
    tests = [
        ("MCP Reasoning Analysis", test_suite.test_mcp_reasoning_analysis(reasoning_agent)),
        ("MCP Cross-Agent Collaboration", test_suite.test_mcp_cross_agent_collaboration(reasoning_agent)),
        ("MCP Reasoning Resources", test_suite.test_mcp_reasoning_resources(reasoning_agent)),
        ("MCP Reasoning Prompts", test_suite.test_mcp_reasoning_prompts(reasoning_agent)),
        ("MCP Agent Orchestration", test_suite.test_mcp_agent_orchestration(agent_manager)),
        ("MCP Intelligent Agent Discovery", test_suite.test_mcp_intelligent_agent_discovery(agent_manager)),
        ("MCP Adaptive Load Balancing", test_suite.test_mcp_adaptive_load_balancing(agent_manager)),
        ("MCP Agent Manager Resources", test_suite.test_mcp_agent_manager_resources(agent_manager)),
        ("MCP Agent Coordination Advisor", test_suite.test_mcp_agent_coordination_advisor(agent_manager)),
        ("MCP Tool Error Handling", test_suite.test_mcp_tool_error_handling(reasoning_agent)),
        ("MCP Performance Tracking", test_suite.test_mcp_performance_tracking(reasoning_agent, agent_manager))
    ]

    # Run each test
    for test_name, test_coro in tests:
        test_results["total_tests"] += 1
        start_time = datetime.now()

        try:
            print(f"\n🧪 Running: {test_name}")
            await test_coro
            test_results["passed_tests"] += 1
            duration = (datetime.now() - start_time).total_seconds()

            test_results["test_details"].append({
                "name": test_name,
                "status": "PASSED",
                "duration": duration
            })

            print(f"✅ {test_name} - PASSED ({duration:.2f}s)")

        except Exception as e:
            test_results["failed_tests"] += 1
            duration = (datetime.now() - start_time).total_seconds()

            test_results["test_details"].append({
                "name": test_name,
                "status": "FAILED",
                "duration": duration,
                "error": str(e)
            })

            print(f"❌ {test_name} - FAILED ({duration:.2f}s)")
            print(f"   Error: {str(e)}")

    # Print summary
    print("\n" + "=" * 60)
    print("🎯 Enhanced MCP Integration Test Results")
    print("=" * 60)
    print(f"Total Tests: {test_results['total_tests']}")
    print(f"Passed: {test_results['passed_tests']} ✅")
    print(f"Failed: {test_results['failed_tests']} ❌")
    print(f"Success Rate: {(test_results['passed_tests'] / test_results['total_tests'] * 100):.1f}%")

    # Detailed results
    print(f"\n📊 Test Details:")
    for test in test_results["test_details"]:
        status_icon = "✅" if test["status"] == "PASSED" else "❌"
        print(f"{status_icon} {test['name']}: {test['status']} ({test['duration']:.2f}s)")
        if test.get("error"):
            print(f"   └─ Error: {test['error']}")

    return test_results


# Command line runner
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        results = asyncio.run(run_enhanced_mcp_integration_tests())

        # Exit with error code if tests failed
        if results["failed_tests"] > 0:
            exit(1)
        else:
            print(f"\n🎉 All tests passed! Enhanced MCP integration is working correctly.")
            exit(0)

    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        exit(1)
