#!/usr/bin/env python3
"""
Test script for Quality Control Manager Agent
Demonstrates quality assessment and routing decisions
"""

import asyncio
import json
import logging
from typing import Dict, Any

from app.a2a.core.security_base import SecureA2AAgent
from qualityControlManagerAgent import (
    QualityControlManagerAgent,
    QualityAssessmentRequest,
    QualityDecision
)


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_calculation_result(scenario: str) -> Dict[str, Any]:
    """Create sample calculation agent results for different scenarios"""

    scenarios = {
        "high_quality": {
            "success": True,
            "total_tests": 100,
            "passed_tests": 95,
            "quality_scores": {
                "accuracy": 0.95,
                "precision": 0.93,
                "performance": 0.88
            },
            "confidence": 0.92,
            "execution_time": 2.3,
            "key_findings": ["High accuracy achieved", "Performance within limits"]
        },
        "medium_quality": {
            "success": True,
            "total_tests": 100,
            "passed_tests": 75,
            "quality_scores": {
                "accuracy": 0.75,
                "precision": 0.72,
                "performance": 0.68
            },
            "confidence": 0.73,
            "execution_time": 4.1,
            "key_findings": ["Moderate accuracy", "Some performance issues"]
        },
        "low_quality": {
            "success": False,
            "total_tests": 100,
            "passed_tests": 35,
            "quality_scores": {
                "accuracy": 0.35,
                "precision": 0.32,
                "performance": 0.28
            },
            "confidence": 0.31,
            "execution_time": 8.7,
            "key_findings": ["Low accuracy", "Performance problems", "Multiple failures"]
        }
    }

    return scenarios.get(scenario, scenarios["medium_quality"])


def create_sample_qa_result(scenario: str) -> Dict[str, Any]:
    """Create sample QA validation agent results for different scenarios"""

    scenarios = {
        "high_quality": {
            "success": True,
            "validation_count": 50,
            "validation_passed": 47,
            "quality_analysis": {
                "reliability": 0.94,
                "completeness": 0.92,
                "overall_score": 0.93
            },
            "confidence": 0.91,
            "coverage_metrics": {
                "dublin_core_coverage": 0.95,
                "technical_coverage": 0.88,
                "relationship_coverage": 0.82
            },
            "issues_found": ["Minor validation discrepancy"]
        },
        "medium_quality": {
            "success": True,
            "validation_count": 50,
            "validation_passed": 37,
            "quality_analysis": {
                "reliability": 0.74,
                "completeness": 0.71,
                "overall_score": 0.72
            },
            "confidence": 0.69,
            "coverage_metrics": {
                "dublin_core_coverage": 0.75,
                "technical_coverage": 0.68,
                "relationship_coverage": 0.65
            },
            "issues_found": ["Several validation failures", "Incomplete coverage"]
        },
        "low_quality": {
            "success": False,
            "validation_count": 50,
            "validation_passed": 18,
            "quality_analysis": {
                "reliability": 0.36,
                "completeness": 0.34,
                "overall_score": 0.35
            },
            "confidence": 0.33,
            "coverage_metrics": {
                "dublin_core_coverage": 0.42,
                "technical_coverage": 0.31,
                "relationship_coverage": 0.28
            },
            "issues_found": ["Multiple validation failures", "Poor coverage", "Data quality issues"]
        }
    }

    return scenarios.get(scenario, scenarios["medium_quality"])


async def test_quality_assessment(agent: QualityControlManagerAgent, scenario: str):
    """Test quality assessment for a specific scenario"""

    print(f"\n{'='*60}")
    print(f"Testing Quality Assessment - {scenario.upper()} QUALITY SCENARIO")
    print(f"{'='*60}")

    # Create sample inputs
    calc_result = create_sample_calculation_result(scenario)
    qa_result = create_sample_qa_result(scenario)

    print(f"\n📊 Input Data:")
    print(f"  Calculation Success Rate: {calc_result['passed_tests']}/{calc_result['total_tests']} ({calc_result['passed_tests']/calc_result['total_tests']:.1%})")
    print(f"  QA Validation Success Rate: {qa_result['validation_passed']}/{qa_result['validation_count']} ({qa_result['validation_passed']/qa_result['validation_count']:.1%})")

    # Create assessment request
    request = QualityAssessmentRequest(
        calculation_result=calc_result,
        qa_validation_result=qa_result,
        workflow_context={
            "workflow_type": "test_scenario",
            "is_critical": scenario == "high_quality"
        }
    )

    # Perform assessment
    try:
        result = await agent.quality_assessment_skill(request)

        print(f"\n🎯 Quality Assessment Results:")
        print(f"  Assessment ID: {result.assessment_id}")
        print(f"  Decision: {result.decision.value}")
        print(f"  Confidence Level: {result.confidence_level:.2%}")

        print(f"\n📈 Quality Scores:")
        for metric, score in result.quality_scores.items():
            print(f"  {metric.title()}: {score:.2f}")

        print(f"\n🧭 Routing Instructions:")
        if result.routing_instructions:
            print(f"  Route To: {result.routing_instructions.get('routing', 'N/A')}")
            print(f"  Reason: {result.routing_instructions.get('reason', 'N/A')}")

        print(f"\n💡 Improvement Recommendations:")
        for i, rec in enumerate(result.improvement_recommendations[:3], 1):
            print(f"  {i}. {rec}")

        # Test Lean Six Sigma if required
        if result.decision == QualityDecision.REQUIRE_LEAN_ANALYSIS:
            print(f"\n🏭 Lean Six Sigma Analysis:")
            if result.lean_sigma_parameters:
                lean_analysis = result.lean_sigma_parameters.get("analysis", {})
                print(f"  DMAIC Phase: {lean_analysis.get('dmaic_phase', 'N/A')}")
                print(f"  Sigma Level: {lean_analysis.get('sigma_level', 'N/A'):.1f}")
                print(f"  Defects per Million: {lean_analysis.get('defects_per_million', 'N/A'):,.0f}")
                print(f"  Process Capability: {lean_analysis.get('process_capability', 'N/A')}")

        # Test AI improvement if required
        if result.decision == QualityDecision.REQUIRE_AI_IMPROVEMENT:
            print(f"\n🤖 AI Improvement Parameters:")
            if result.ai_improvement_parameters:
                ai_params = result.ai_improvement_parameters
                print(f"  Improvement Type: {ai_params.get('improvement_type', 'N/A')}")
                print(f"  Use Machine Learning: {ai_params.get('optimization_approach', {}).get('use_machine_learning', False)}")
                print(f"  Target Metrics: {len(ai_params.get('target_metrics', {}))}")

        return result

    except Exception as e:
        print(f"❌ Assessment failed: {e}")
        return None


async def test_lean_six_sigma_analysis(agent: QualityControlManagerAgent):
    """Test detailed Lean Six Sigma analysis"""

    print(f"\n{'='*60}")
    print(f"Testing Detailed Lean Six Sigma Analysis")
    print(f"{'='*60}")

    # Sample quality and process data
    quality_data = {
        "current_sigma": 3.2,
        "target_sigma": 4.0,
        "defect_rate": 0.066,
        "success_rate": 0.934
    }

    process_data = {
        "total_opportunities": 1000,
        "total_defects": 66,
        "opportunities_per_unit": 1,
        "total_units": 1000,
        "process_mean": 0.85,
        "process_std_dev": 0.12,
        "upper_spec_limit": 1.0,
        "lower_spec_limit": 0.0,
        "target_value": 0.85,
        "sample_data": [0.82, 0.87, 0.83, 0.89, 0.85, 0.81, 0.88, 0.84, 0.86, 0.83],
        "defect_types": {
            "calculation_errors": 30,
            "validation_failures": 20,
            "timeout_issues": 10,
            "data_quality": 4,
            "other": 2
        }
    }

    try:
        result = await agent.lean_six_sigma_analysis_skill(quality_data, process_data)

        print(f"\n📊 Six Sigma Metrics:")
        sigma_metrics = result.get("sigma_metrics", {})
        print(f"  Current Sigma Level: {sigma_metrics.get('current_sigma', 'N/A'):.1f}")
        print(f"  Target Sigma Level: {sigma_metrics.get('target_sigma', 'N/A'):.1f}")
        print(f"  Current DPMO: {sigma_metrics.get('current_dpmo', 'N/A'):,.0f}")
        print(f"  Yield Rate: {sigma_metrics.get('yield_rate', 'N/A'):.2%}")

        print(f"\n🎯 Process Capability:")
        capability = result.get("capability_analysis", {})
        print(f"  Cp: {capability.get('cp', 'N/A')}")
        print(f"  Cpk: {capability.get('cpk', 'N/A')}")
        print(f"  Capability Rating: {capability.get('capability_rating', 'N/A')}")

        print(f"\n📈 Control Charts:")
        control_charts = result.get("control_charts", {})
        print(f"  Chart Type: {control_charts.get('chart_type', 'N/A')}")
        print(f"  Center Line: {control_charts.get('center_line', 'N/A'):.3f}")
        print(f"  UCL: {control_charts.get('upper_control_limit', 'N/A'):.3f}")
        print(f"  LCL: {control_charts.get('lower_control_limit', 'N/A'):.3f}")
        print(f"  Out of Control Points: {len(control_charts.get('out_of_control_points', []))}")

        print(f"\n🔍 Root Cause Analysis:")
        rca = result.get("root_cause_analysis", {})
        pareto = rca.get("pareto_analysis", {})
        vital_few = pareto.get("vital_few", [])
        print(f"  Vital Few Defects: {', '.join(vital_few[:3])}")
        print(f"  Recommendation: {pareto.get('recommendation', 'N/A')}")

        print(f"\n📋 Improvement Plan:")
        improvement = result.get("improvement_plan", {})
        short_term = improvement.get("short_term_actions", [])
        for i, action in enumerate(short_term[:2], 1):
            print(f"  {i}. {action.get('action', 'N/A')} ({action.get('timeline', 'N/A')})")

        print(f"\n💰 Estimated Benefits:")
        benefits = result.get("estimated_benefits", {})
        for key, value in benefits.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

        return result

    except Exception as e:
        print(f"❌ Lean Six Sigma analysis failed: {e}")
        return None


async def test_mcp_tools(agent: QualityControlManagerAgent):
    """Test MCP tools integration"""

    print(f"\n{'='*60}")
    print(f"Testing MCP Tools Integration")
    print(f"{'='*60}")

    try:
        # Test assess_quality MCP tool
        print(f"\n🔧 Testing MCP Tool: assess_quality")

        calc_result = create_sample_calculation_result("medium_quality")
        qa_result = create_sample_qa_result("medium_quality")

        mcp_result = await agent.assess_quality_mcp(
            calculation_result=calc_result,
            qa_validation_result=qa_result,
            quality_thresholds={"accuracy": 0.8, "reliability": 0.75}
        )

        print(f"  Decision: {mcp_result.get('decision', 'N/A')}")
        print(f"  Quality Scores: {len(mcp_result.get('quality_scores', {}))}")
        print(f"  Confidence: {mcp_result.get('confidence', 'N/A'):.2%}")
        print(f"  Recommendations: {len(mcp_result.get('recommendations', []))}")

        # Test quality metrics MCP resource
        print(f"\n📊 Testing MCP Resource: quality metrics")

        metrics = await agent.get_quality_metrics_mcp()
        print(f"  Processing Stats: {len(metrics.get('processing_stats', {}))}")
        print(f"  Default Thresholds: {len(metrics.get('default_thresholds', {}))}")
        print(f"  Recent Assessments: {metrics.get('recent_assessments', 'N/A')}")

        # Test quality improvement MCP prompt
        print(f"\n💭 Testing MCP Prompt: quality improvement")

        current_scores = {"accuracy": 0.65, "reliability": 0.70}
        target_scores = {"accuracy": 0.85, "reliability": 0.90}

        prompt = await agent.quality_improvement_prompt(current_scores, target_scores)
        print(f"  Prompt Length: {len(prompt)} characters")
        print(f"  Contains Scores: {'accuracy' in prompt and 'reliability' in prompt}")

        return True

    except Exception as e:
        print(f"❌ MCP tools test failed: {e}")
        return False


async def main():
    """Main test function"""

    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    Quality Control Manager Agent Test Suite                   ║
║                                    Agent 6                                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Create agent instance
    agent = QualityControlManagerAgent(
        base_url=os.getenv("A2A_SERVICE_URL"),
        data_manager_url=os.getenv("A2A_SERVICE_URL"),
        catalog_manager_url=os.getenv("A2A_SERVICE_URL"),
        enable_monitoring=False  # Disable for testing
    )

    try:
        # Initialize agent
        print("🔧 Initializing Quality Control Manager Agent...")
        await agent.initialize()
        print("✅ Agent initialized successfully")

        # Test different quality scenarios
        scenarios = ["high_quality", "medium_quality", "low_quality"]

        for scenario in scenarios:
            result = await test_quality_assessment(agent, scenario)

            if result:
                print(f"✅ {scenario} scenario test completed")
            else:
                print(f"❌ {scenario} scenario test failed")

        # Test Lean Six Sigma analysis
        lean_result = await test_lean_six_sigma_analysis(agent)
        if lean_result:
            print("✅ Lean Six Sigma analysis test completed")
        else:
            print("❌ Lean Six Sigma analysis test failed")

        # Test MCP tools
        mcp_success = await test_mcp_tools(agent)
        if mcp_success:
            print("✅ MCP tools test completed")
        else:
            print("❌ MCP tools test failed")

        # Show final statistics
        print(f"\n{'='*60}")
        print(f"Test Summary")
        print(f"{'='*60}")
        print(f"📊 Processing Statistics:")
        for key, value in agent.processing_stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

        print(f"\n🎯 Quality Thresholds:")
        for key, value in agent.default_thresholds.items():
            print(f"  {key.title()}: {value:.2f}")

    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        raise

    finally:
        # Cleanup
        try:
            await agent.shutdown()
            print("\n🛑 Agent shutdown completed")
        except:
            pass


if __name__ == "__main__":
    asyncio.run(main())