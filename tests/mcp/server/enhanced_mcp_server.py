#!/usr/bin/env python3
"""
Enhanced A2A Test Suite MCP Server with AI and Database Integration
Comprehensive MCP server with GROKClient AI analysis and database tracking
"""

import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import mcp.server.stdio
    import mcp.types as types
    from mcp.server import NotificationOptions, Server
except ImportError:
    print("MCP library not found. Install with: pip install mcp")
    sys.exit(1)

from pydantic import AnyUrl

# Import our enhanced components
from ..ai.grok_integration import AITestAnalyzer, AITestOptimizer, AIInsightsDashboard
from ..database.test_database_manager import TestDatabaseManager, TestAnalyticsService
from ..tools.test_executor import TestExecutor, TestSuiteBuilder, TestReporter
from ..agents.test_orchestrator import TestOrchestrator, TestPriority

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("a2a-enhanced-test-mcp-server")

# Test suite paths
TEST_ROOT = Path(__file__).parent.parent.parent
UNIT_TESTS = TEST_ROOT / "unit"
INTEGRATION_TESTS = TEST_ROOT / "integration"
E2E_TESTS = TEST_ROOT / "e2e"
PERFORMANCE_TESTS = TEST_ROOT / "performance"
SECURITY_TESTS = TEST_ROOT / "security"
CONTRACTS_TESTS = TEST_ROOT / "contracts"

# Initialize enhanced MCP server
server = Server("a2a-enhanced-test-suite")

# Initialize components
db_manager = TestDatabaseManager()
analytics_service = TestAnalyticsService(db_manager)
ai_analyzer = AITestAnalyzer()
ai_optimizer = AITestOptimizer(ai_analyzer)
ai_dashboard = AIInsightsDashboard(ai_analyzer)
test_executor = TestExecutor(TEST_ROOT)
suite_builder = TestSuiteBuilder(TEST_ROOT)
reporter = TestReporter(TEST_ROOT)
orchestrator = TestOrchestrator(TEST_ROOT)

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available test resources with enhanced AI and database capabilities."""
    return [
        types.Resource(
            uri=AnyUrl("test://unit"),
            name="Unit Tests",
            description="Isolated component tests with AI analysis and trend tracking",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("test://integration"),
            name="Integration Tests", 
            description="Component interaction tests with performance analytics",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("test://e2e"),
            name="End-to-End Tests",
            description="Complete user workflow tests with stability tracking",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("test://performance"),
            name="Performance Tests",
            description="Load, stress, and benchmark tests with trend analysis",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("test://security"),
            name="Security Tests", 
            description="Security validation with AI-powered vulnerability analysis",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("test://contracts"),
            name="Smart Contract Tests",
            description="Blockchain contract tests with gas optimization tracking",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("test://analytics"),
            name="Test Analytics",
            description="Comprehensive test analytics and AI insights",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("test://trends"),
            name="Test Trends",
            description="Historical test performance and stability trends",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("test://ai-insights"),
            name="AI Insights",
            description="AI-powered test analysis and optimization recommendations",
            mimeType="application/json",
        ),
    ]

@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """Read enhanced test resource information with database and AI integration."""
    
    if str(uri) == "test://analytics":
        # Generate comprehensive analytics report
        health_report = analytics_service.generate_test_health_report()
        optimization_opportunities = analytics_service.identify_optimization_opportunities()
        db_stats = db_manager.get_database_stats()
        
        return json.dumps({
            "test_health": health_report,
            "optimization_opportunities": optimization_opportunities,
            "database_stats": db_stats,
            "generated_at": health_report.get("generated_at")
        }, indent=2)
    
    elif str(uri) == "test://trends":
        # Get comprehensive trend analysis
        test_trends = db_manager.calculate_test_trends()
        coverage_trends = db_manager.get_coverage_trends()
        workflow_history = db_manager.get_workflow_history(limit=50)
        
        return json.dumps({
            "test_stability_trends": test_trends,
            "coverage_trends": coverage_trends,
            "workflow_history": workflow_history,
            "trend_period_days": 30
        }, indent=2)
    
    elif str(uri) == "test://ai-insights":
        # Get AI analysis history and insights
        ai_analyses = db_manager.get_ai_analysis_history(limit=20)
        
        # Generate executive summary
        recent_workflows = []  # Would get from orchestrator
        executive_summary = await ai_dashboard.generate_executive_summary(recent_workflows)
        
        return json.dumps({
            "executive_summary": executive_summary,
            "recent_ai_analyses": ai_analyses,
            "ai_capabilities": [
                "failure_analysis",
                "optimization_recommendations", 
                "stability_prediction",
                "performance_insights"
            ]
        }, indent=2)
    
    else:
        # Handle standard test resource requests with enhanced data
        path_mapping = {
            "test://unit": UNIT_TESTS,
            "test://integration": INTEGRATION_TESTS, 
            "test://e2e": E2E_TESTS,
            "test://performance": PERFORMANCE_TESTS,
            "test://security": SECURITY_TESTS,
            "test://contracts": CONTRACTS_TESTS,
        }
        
        if str(uri) in path_mapping:
            test_path = path_mapping[str(uri)]
            test_files = []
            
            if test_path.exists():
                for file_path in test_path.rglob("*test*"):
                    if file_path.is_file():
                        test_files.append({
                            "file": str(file_path.relative_to(TEST_ROOT)),
                            "size": file_path.stat().st_size,
                            "modified": file_path.stat().st_mtime
                        })
            
            # Get test execution history for this category
            test_type = str(uri).replace("test://", "")
            execution_history = db_manager.get_test_execution_history(limit=100)
            type_specific_history = [
                h for h in execution_history 
                if h.get("test_type") == test_type
            ]
            
            return json.dumps({
                "category": test_type,
                "path": str(test_path),
                "test_count": len(test_files),
                "files": test_files,
                "recent_executions": len(type_specific_history),
                "execution_history": type_specific_history[:10]  # Last 10 executions
            }, indent=2)
    
    raise ValueError(f"Unknown resource: {uri}")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available enhanced test management tools with AI and database integration."""
    return [
        # Enhanced existing tools
        types.Tool(
            name="run_tests_enhanced",
            description="Execute test suites with AI optimization and database tracking",
            inputSchema={
                "type": "object",
                "properties": {
                    "test_type": {
                        "type": "string",
                        "enum": ["unit", "integration", "e2e", "performance", "security", "contracts", "all"],
                        "description": "Type of tests to run"
                    },
                    "module": {
                        "type": "string",
                        "enum": ["a2aAgents", "a2aNetwork", "common", "all"],
                        "description": "Module to test"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Test name pattern to match"
                    },
                    "coverage": {
                        "type": "boolean",
                        "description": "Generate coverage report",
                        "default": False
                    },
                    "ai_optimize": {
                        "type": "boolean",
                        "description": "Apply AI optimization recommendations",
                        "default": True
                    },
                    "store_results": {
                        "type": "boolean",
                        "description": "Store results in database for analytics",
                        "default": True
                    },
                    "verbose": {
                        "type": "boolean", 
                        "description": "Verbose output",
                        "default": False
                    }
                },
                "required": ["test_type"]
            },
        ),
        
        # AI-powered tools
        types.Tool(
            name="analyze_failures_ai",
            description="AI-powered analysis of test failures with actionable insights",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "string",
                        "description": "Workflow ID to analyze failures for"
                    },
                    "test_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific test names to analyze"
                    },
                    "include_context": {
                        "type": "boolean",
                        "description": "Include environmental context in analysis",
                        "default": True
                    }
                }
            },
        ),
        
        types.Tool(
            name="optimize_execution_ai",
            description="AI-driven test execution optimization based on historical data",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "string",
                        "description": "Workflow ID to optimize"
                    },
                    "optimization_type": {
                        "type": "string",
                        "enum": ["performance", "reliability", "coverage", "all"],
                        "description": "Type of optimization to focus on",
                        "default": "all"
                    },
                    "apply_immediately": {
                        "type": "boolean",
                        "description": "Apply optimizations immediately",
                        "default": False
                    }
                }
            },
        ),
        
        types.Tool(
            name="predict_test_stability",
            description="AI prediction of test stability and identification of flaky tests",
            inputSchema={
                "type": "object",
                "properties": {
                    "days_back": {
                        "type": "integer",
                        "description": "Days of history to analyze",
                        "default": 30
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence for predictions",
                        "default": 0.7
                    }
                }
            },
        ),
        
        # Database and analytics tools
        types.Tool(
            name="get_test_analytics",
            description="Comprehensive test analytics and trends from database",
            inputSchema={
                "type": "object",
                "properties": {
                    "report_type": {
                        "type": "string",
                        "enum": ["health", "trends", "performance", "coverage", "all"],
                        "description": "Type of analytics report",
                        "default": "all"
                    },
                    "time_period": {
                        "type": "integer",
                        "description": "Analysis period in days",
                        "default": 30
                    },
                    "include_predictions": {
                        "type": "boolean",
                        "description": "Include AI predictions in report",
                        "default": True
                    }
                }
            },
        ),
        
        types.Tool(
            name="track_test_execution",
            description="Track and store detailed test execution information",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "string",
                        "description": "Workflow ID to track"
                    },
                    "include_performance": {
                        "type": "boolean",
                        "description": "Include performance metrics",
                        "default": True
                    },
                    "include_coverage": {
                        "type": "boolean",
                        "description": "Include coverage data",
                        "default": True
                    }
                },
                "required": ["workflow_id"]
            },
        ),
        
        types.Tool(
            name="generate_ai_insights",
            description="Generate comprehensive AI insights and recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "insight_type": {
                        "type": "string",
                        "enum": ["executive_summary", "optimization_opportunities", "risk_assessment", "trends_analysis"],
                        "description": "Type of insights to generate"
                    },
                    "time_frame": {
                        "type": "integer",
                        "description": "Time frame for analysis in days",
                        "default": 30
                    }
                },
                "required": ["insight_type"]
            },
        ),
        
        # Enhanced original tools
        types.Tool(
            name="discover_tests_enhanced",
            description="Enhanced test discovery with performance and stability metrics",
            inputSchema={
                "type": "object",
                "properties": {
                    "test_type": {
                        "type": "string",
                        "enum": ["unit", "integration", "e2e", "performance", "security", "contracts", "all"],
                        "description": "Type of tests to discover"
                    },
                    "include_metrics": {
                        "type": "boolean",
                        "description": "Include performance and stability metrics",
                        "default": True
                    },
                    "include_disabled": {
                        "type": "boolean",
                        "description": "Include disabled tests",
                        "default": False
                    }
                }
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
    """Handle enhanced tool execution requests with AI and database integration."""
    if arguments is None:
        arguments = {}
    
    try:
        if name == "run_tests_enhanced":
            return await _run_tests_enhanced(arguments)
        elif name == "analyze_failures_ai":
            return await _analyze_failures_ai(arguments)
        elif name == "optimize_execution_ai":
            return await _optimize_execution_ai(arguments)
        elif name == "predict_test_stability":
            return await _predict_test_stability(arguments)
        elif name == "get_test_analytics":
            return await _get_test_analytics(arguments)
        elif name == "track_test_execution":
            return await _track_test_execution(arguments)
        elif name == "generate_ai_insights":
            return await _generate_ai_insights(arguments)
        elif name == "discover_tests_enhanced":
            return await _discover_tests_enhanced(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

async def _run_tests_enhanced(args: Dict[str, Any]) -> List[types.TextContent]:
    """Execute tests with AI optimization and database tracking."""
    test_type = args["test_type"]
    module = args.get("module", "all")
    ai_optimize = args.get("ai_optimize", True)
    store_results = args.get("store_results", True)
    coverage = args.get("coverage", False)
    
    output = f"Enhanced Test Execution\n"
    output += f"Type: {test_type}, Module: {module}\n"
    output += f"AI Optimization: {ai_optimize}, Store Results: {store_results}\n\n"
    
    try:
        # Create workflow
        workflow_id = await orchestrator.create_workflow(
            name=f"Enhanced {test_type} Tests",
            test_type=test_type,
            module=module,
            coverage=coverage,
            parallel=True
        )
        
        # Apply AI optimization if requested
        if ai_optimize:
            execution_history = db_manager.get_workflow_history(limit=50)
            workflow = orchestrator.workflows[workflow_id]
            optimized_workflow = await ai_optimizer.optimize_workflow(workflow, execution_history)
            orchestrator.workflows[workflow_id] = optimized_workflow
            output += "✅ Applied AI optimizations\n"
        
        # Execute workflow
        result_workflow = await orchestrator.execute_workflow(workflow_id)
        
        # Store results in database if requested
        if store_results:
            db_manager.store_workflow(result_workflow)
            if result_workflow.results:
                db_manager.store_test_results(result_workflow.results, workflow_id)
            output += "✅ Results stored in database\n"
        
        # Generate summary
        passed = len([r for r in result_workflow.results if r.status.value == "passed"])
        failed = len([r for r in result_workflow.results if r.status.value == "failed"])
        total = len(result_workflow.results)
        
        output += f"\nExecution Summary:\n"
        output += f"Total Tests: {total}\n"
        output += f"Passed: {passed} ({passed/total*100:.1f}%)\n"
        output += f"Failed: {failed} ({failed/total*100:.1f}%)\n"
        output += f"Workflow Status: {result_workflow.status.value}\n"
        
        # AI failure analysis if there are failures
        if failed > 0:
            failed_tests = [r for r in result_workflow.results if r.status.value == "failed"]
            analysis = await ai_analyzer.analyze_test_failures(failed_tests)
            
            if store_results:
                db_manager.store_ai_analysis("failure_analysis", 
                                           {"failed_count": failed}, 
                                           analysis, workflow_id)
            
            output += f"\n🤖 AI Failure Analysis:\n"
            output += f"Analysis: {analysis.get('analysis', 'No analysis available')}\n"
            
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                output += "\nRecommendations:\n"
                for i, rec in enumerate(recommendations[:3], 1):
                    output += f"{i}. {rec}\n"
        
        return [types.TextContent(type="text", text=output)]
        
    except Exception as e:
        return [types.TextContent(type="text", text=f"Enhanced test execution failed: {str(e)}")]

async def _analyze_failures_ai(args: Dict[str, Any]) -> List[types.TextContent]:
    """AI-powered analysis of test failures."""
    workflow_id = args.get("workflow_id")
    test_names = args.get("test_names", [])
    include_context = args.get("include_context", True)
    
    output = "🤖 AI Failure Analysis\n"
    output += "=" * 50 + "\n\n"
    
    try:
        # Get failed tests
        if workflow_id:
            execution_history = db_manager.get_test_execution_history(limit=1000)
            failed_executions = [
                e for e in execution_history 
                if e.get("workflow_id") == workflow_id and e.get("status") == "failed"
            ]
        elif test_names:
            failed_executions = []
            for test_name in test_names:
                executions = db_manager.get_test_execution_history(test_name=test_name, limit=10)
                failed_executions.extend([e for e in executions if e.get("status") == "failed"])
        else:
            # Get recent failures
            execution_history = db_manager.get_test_execution_history(limit=100)
            failed_executions = [e for e in execution_history if e.get("status") == "failed"]
        
        if not failed_executions:
            return [types.TextContent(type="text", text="No failed tests found for analysis.")]
        
        # Convert to TestResult format for AI analysis
        from ..tools.test_executor import TestResult, TestStatus
        failed_tests = []
        for execution in failed_executions[:20]:  # Limit to 20 most recent
            test_result = TestResult(
                name=execution.get("test_name", "unknown"),
                status=TestStatus.FAILED,
                duration=execution.get("duration_seconds", 0),
                output=execution.get("output", ""),
                error=execution.get("error_message", "")
            )
            failed_tests.append(test_result)
        
        # Prepare context
        context = {}
        if include_context:
            context = {
                "total_failures": len(failed_executions),
                "analysis_scope": "recent failures" if not workflow_id else f"workflow {workflow_id}",
                "time_period": "last 30 days"
            }
        
        # Get AI analysis
        analysis = await ai_analyzer.analyze_test_failures(failed_tests, context)
        
        # Store analysis in database
        db_manager.store_ai_analysis(
            "failure_analysis",
            {"failed_tests": [t.name for t in failed_tests], "context": context},
            analysis,
            workflow_id
        )
        
        # Format output
        output += f"Analyzed {len(failed_tests)} failed tests\n\n"
        
        if "analysis" in analysis:
            output += f"📋 Analysis:\n{analysis['analysis']}\n\n"
        
        if "patterns" in analysis and analysis["patterns"]:
            output += "🔍 Patterns Identified:\n"
            for pattern in analysis["patterns"]:
                output += f"• {pattern}\n"
            output += "\n"
        
        if "recommendations" in analysis and analysis["recommendations"]:
            output += "💡 Recommendations:\n"
            for i, rec in enumerate(analysis["recommendations"], 1):
                priority = analysis.get("priorities", ["Medium"] * len(analysis["recommendations"]))[i-1] if i-1 < len(analysis.get("priorities", [])) else "Medium"
                effort = analysis.get("efforts", ["Medium"] * len(analysis["recommendations"]))[i-1] if i-1 < len(analysis.get("efforts", [])) else "Medium"
                output += f"{i}. {rec} (Priority: {priority}, Effort: {effort})\n"
        
        return [types.TextContent(type="text", text=output)]
        
    except Exception as e:
        return [types.TextContent(type="text", text=f"AI failure analysis failed: {str(e)}")]

async def _optimize_execution_ai(args: Dict[str, Any]) -> List[types.TextContent]:
    """AI-driven test execution optimization."""
    workflow_id = args.get("workflow_id")
    optimization_type = args.get("optimization_type", "all")
    apply_immediately = args.get("apply_immediately", False)
    
    output = "🔧 AI Execution Optimization\n"
    output += "=" * 50 + "\n\n"
    
    try:
        if not workflow_id:
            return [types.TextContent(type="text", text="Workflow ID is required for optimization.")]
        
        if workflow_id not in orchestrator.workflows:
            return [types.TextContent(type="text", text=f"Workflow {workflow_id} not found.")]
        
        workflow = orchestrator.workflows[workflow_id]
        execution_history = db_manager.get_workflow_history(limit=100)
        
        # Get AI optimization
        optimization = await ai_analyzer.optimize_test_execution(execution_history, workflow)
        
        # Store optimization analysis
        db_manager.store_ai_analysis(
            "optimization",
            {"workflow_id": workflow_id, "optimization_type": optimization_type},
            optimization,
            workflow_id
        )
        
        output += f"Optimizing workflow: {workflow.name}\n"
        output += f"Focus: {optimization_type}\n\n"
        
        if "strategy" in optimization:
            output += f"📈 Strategy Recommendations:\n{optimization['strategy']}\n\n"
        
        if "timeouts" in optimization and optimization["timeouts"]:
            output += "⏱️ Timeout Optimizations:\n"
            for key, value in optimization["timeouts"].items():
                output += f"• {key}: {value}\n"
            output += "\n"
        
        if "resources" in optimization and optimization["resources"]:
            output += "💾 Resource Optimizations:\n"
            for key, value in optimization["resources"].items():
                output += f"• {key}: {value}\n"
            output += "\n"
        
        if "bottlenecks" in optimization and optimization["bottlenecks"]:
            output += "🚧 Identified Bottlenecks:\n"
            for bottleneck in optimization["bottlenecks"]:
                output += f"• {bottleneck}\n"
            output += "\n"
        
        # Apply optimizations if requested
        if apply_immediately:
            optimized_workflow = await ai_optimizer.optimize_workflow(workflow, execution_history)
            orchestrator.workflows[workflow_id] = optimized_workflow
            db_manager.store_workflow(optimized_workflow)
            output += "✅ Optimizations applied to workflow\n"
        else:
            output += "💡 Optimizations available - use apply_immediately=true to apply\n"
        
        return [types.TextContent(type="text", text=output)]
        
    except Exception as e:
        return [types.TextContent(type="text", text=f"AI optimization failed: {str(e)}")]

async def _predict_test_stability(args: Dict[str, Any]) -> List[types.TextContent]:
    """AI prediction of test stability."""
    days_back = args.get("days_back", 30)
    confidence_threshold = args.get("confidence_threshold", 0.7)
    
    output = "🔮 AI Test Stability Prediction\n"
    output += "=" * 50 + "\n\n"
    
    try:
        # Get test execution history
        execution_history = db_manager.get_test_execution_history(limit=2000, days_back=days_back)
        
        # Get AI stability prediction
        prediction = await ai_analyzer.predict_test_stability(execution_history)
        
        # Store prediction analysis
        db_manager.store_ai_analysis(
            "stability_prediction",
            {"days_back": days_back, "confidence_threshold": confidence_threshold},
            prediction
        )
        
        output += f"Analysis Period: {days_back} days\n"
        output += f"Confidence Threshold: {confidence_threshold}\n\n"
        
        if "assessment" in prediction:
            output += f"📊 Overall Assessment:\n{prediction['assessment']}\n\n"
        
        if "flaky_tests" in prediction and prediction["flaky_tests"]:
            output += "⚠️ Potentially Flaky Tests:\n"
            for test in prediction["flaky_tests"]:
                if isinstance(test, dict):
                    confidence = test.get("confidence", "unknown")
                    output += f"• {test.get('name', 'unknown')} (Confidence: {confidence})\n"
                else:
                    output += f"• {test}\n"
            output += "\n"
        
        if "trends" in prediction and prediction["trends"]:
            output += "📈 Stability Trends:\n"
            for trend in prediction["trends"]:
                output += f"• {trend}\n"
            output += "\n"
        
        if "recommendations" in prediction and prediction["recommendations"]:
            output += "💡 Stability Recommendations:\n"
            for i, rec in enumerate(prediction["recommendations"], 1):
                output += f"{i}. {rec}\n"
        
        return [types.TextContent(type="text", text=output)]
        
    except Exception as e:
        return [types.TextContent(type="text", text=f"AI stability prediction failed: {str(e)}")]

async def _get_test_analytics(args: Dict[str, Any]) -> List[types.TextContent]:
    """Get comprehensive test analytics."""
    report_type = args.get("report_type", "all")
    time_period = args.get("time_period", 30)
    include_predictions = args.get("include_predictions", True)
    
    output = "📊 Test Analytics Report\n"
    output += "=" * 50 + "\n\n"
    
    try:
        output += f"Report Type: {report_type}\n"
        output += f"Time Period: {time_period} days\n\n"
        
        if report_type in ["health", "all"]:
            health_report = analytics_service.generate_test_health_report(time_period)
            output += "🏥 Test Health Summary:\n"
            output += f"Total Tests: {health_report['test_health']['total_tests']}\n"
            output += f"Stable Tests: {health_report['test_health']['stable_tests']}\n"
            output += f"Flaky Tests: {health_report['test_health']['flaky_tests']}\n"
            output += f"Stability: {health_report['test_health']['stability_percentage']:.1f}%\n"
            output += f"Workflow Success Rate: {health_report['workflow_health']['success_rate_percentage']:.1f}%\n"
            output += f"Average Coverage: {health_report['coverage_health']['average_coverage_percentage']:.1f}%\n\n"
        
        if report_type in ["trends", "all"]:
            trends = db_manager.calculate_test_trends(time_period)
            output += f"📈 Test Trends ({len(trends)} tests analyzed):\n"
            
            improving_tests = [name for name, data in trends.items() if data["trend_direction"] == "improving"]
            declining_tests = [name for name, data in trends.items() if data["trend_direction"] == "declining"]
            
            output += f"Improving: {len(improving_tests)} tests\n"
            output += f"Declining: {len(declining_tests)} tests\n"
            
            if declining_tests:
                output += "\n⚠️ Declining Tests (Top 5):\n"
                sorted_declining = sorted(
                    [(name, trends[name]["stability_score"]) for name in declining_tests],
                    key=lambda x: x[1]
                )[:5]
                for name, score in sorted_declining:
                    output += f"• {name} (Stability: {score:.2f})\n"
            output += "\n"
        
        if report_type in ["performance", "all"]:
            performance_metrics = db_manager.get_performance_metrics(days_back=time_period)
            output += f"⚡ Performance Metrics ({len(performance_metrics)} recorded):\n"
            
            if performance_metrics:
                avg_duration = sum(m.get("metric_value", 0) for m in performance_metrics if m.get("metric_name") == "execution_duration") / max(1, len([m for m in performance_metrics if m.get("metric_name") == "execution_duration"]))
                output += f"Average Execution Duration: {avg_duration:.2f}s\n"
            output += "\n"
        
        if report_type in ["coverage", "all"]:
            coverage_trends = db_manager.get_coverage_trends(days_back=time_period)
            output += f"📈 Coverage Trends ({len(coverage_trends)} modules):\n"
            
            for module, trend_data in coverage_trends.items():
                output += f"• {module}: {trend_data['current_coverage']:.1f}% ({trend_data['trend']})\n"
            output += "\n"
        
        # Include AI predictions if requested
        if include_predictions and report_type in ["all"]:
            optimization_opportunities = analytics_service.identify_optimization_opportunities()
            if optimization_opportunities:
                output += "🤖 AI Optimization Opportunities:\n"
                for opp in optimization_opportunities[:3]:
                    output += f"• {opp['title']}: {opp['description']} (Impact: {opp['impact']})\n"
        
        return [types.TextContent(type="text", text=output)]
        
    except Exception as e:
        return [types.TextContent(type="text", text=f"Analytics report generation failed: {str(e)}")]

async def _track_test_execution(args: Dict[str, Any]) -> List[types.TextContent]:
    """Track and store detailed test execution information."""
    workflow_id = args["workflow_id"]
    include_performance = args.get("include_performance", True)
    include_coverage = args.get("include_coverage", True)
    
    output = f"📊 Tracking Test Execution: {workflow_id}\n"
    output += "=" * 50 + "\n\n"
    
    try:
        if workflow_id not in orchestrator.workflows:
            return [types.TextContent(type="text", text=f"Workflow {workflow_id} not found.")]
        
        workflow = orchestrator.workflows[workflow_id]
        
        # Store workflow if not already stored
        db_manager.store_workflow(workflow)
        
        # Store test results
        if workflow.results:
            result_ids = db_manager.store_test_results(workflow.results, workflow_id)
            output += f"✅ Stored {len(result_ids)} test results\n"
        
        # Store performance metrics if requested
        if include_performance:
            total_duration = sum(r.duration for r in workflow.results)
            avg_duration = total_duration / len(workflow.results) if workflow.results else 0
            
            db_manager.store_performance_metric(
                "workflow_total_duration", total_duration, "seconds", "execution", workflow_id
            )
            db_manager.store_performance_metric(
                "workflow_avg_duration", avg_duration, "seconds", "execution", workflow_id
            )
            output += f"✅ Stored performance metrics\n"
        
        # Store coverage data if requested and available
        if include_coverage:
            coverage_stored = 0
            for result in workflow.results:
                if result.coverage:
                    # Extract module from test name (simplified)
                    module = "unknown"
                    if "a2aAgents" in result.name:
                        module = "a2aAgents"
                    elif "a2aNetwork" in result.name:
                        module = "a2aNetwork"
                    
                    db_manager.store_coverage_data(module, result.coverage, workflow_id)
                    coverage_stored += 1
            
            if coverage_stored > 0:
                output += f"✅ Stored coverage data for {coverage_stored} tests\n"
        
        # Generate execution summary
        passed = len([r for r in workflow.results if r.status.value == "passed"])
        failed = len([r for r in workflow.results if r.status.value == "failed"])
        
        output += f"\nExecution Summary:\n"
        output += f"Status: {workflow.status.value}\n"
        output += f"Tests: {len(workflow.results)} total, {passed} passed, {failed} failed\n"
        
        if workflow.start_time and workflow.end_time:
            duration = (workflow.end_time - workflow.start_time).total_seconds()
            output += f"Duration: {duration:.2f} seconds\n"
        
        output += f"Stored in database with ID: {workflow_id}\n"
        
        return [types.TextContent(type="text", text=output)]
        
    except Exception as e:
        return [types.TextContent(type="text", text=f"Test execution tracking failed: {str(e)}")]

async def _generate_ai_insights(args: Dict[str, Any]) -> List[types.TextContent]:
    """Generate comprehensive AI insights."""
    insight_type = args["insight_type"]
    time_frame = args.get("time_frame", 30)
    
    output = f"🤖 AI Insights: {insight_type}\n"
    output += "=" * 50 + "\n\n"
    
    try:
        if insight_type == "executive_summary":
            # Get recent workflows for analysis
            workflow_history = db_manager.get_workflow_history(limit=50, days_back=time_frame)
            
            # Convert to workflow objects (simplified)
            workflows = []  # Would convert from DB data to workflow objects
            
            summary = await ai_dashboard.generate_executive_summary(workflows, time_frame)
            
            output += f"📋 Executive Summary ({time_frame} days):\n\n"
            
            if "key_metrics" in summary:
                metrics = summary["key_metrics"]
                output += f"Key Metrics:\n"
                output += f"• Workflow Success Rate: {metrics.get('workflow_success_rate', 0):.1f}%\n"
                output += f"• Test Pass Rate: {metrics.get('test_pass_rate', 0):.1f}%\n"
                output += f"• Total Workflows: {metrics.get('total_workflows', 0)}\n"
                output += f"• Total Tests: {metrics.get('total_tests', 0)}\n\n"
            
            if "recommendations" in summary:
                recs = summary["recommendations"]
                output += "🎯 Priority Recommendations:\n"
                for i, rec in enumerate(recs[:5], 1):
                    priority = rec.get("priority", 1)
                    category = rec.get("category", "General")
                    output += f"{i}. [{category}] {rec.get('recommendation', 'No recommendation')} (Priority: {priority})\n"
        
        elif insight_type == "optimization_opportunities":
            opportunities = analytics_service.identify_optimization_opportunities()
            
            output += f"🔧 Optimization Opportunities:\n\n"
            
            for opp in opportunities:
                output += f"📌 {opp['title']}\n"
                output += f"   Type: {opp['type']}\n"
                output += f"   Impact: {opp['impact']}\n"
                output += f"   Description: {opp['description']}\n"
                
                if 'tests' in opp:
                    output += f"   Affected Tests: {len(opp['tests'])}\n"
                    for test in opp['tests'][:3]:
                        if isinstance(test, dict):
                            test_name = test.get('test_name', 'unknown')
                            output += f"     • {test_name}\n"
                output += "\n"
        
        elif insight_type == "risk_assessment":
            # Get test trends for risk assessment
            trends = db_manager.calculate_test_trends(time_frame)
            
            output += f"⚠️ Risk Assessment:\n\n"
            
            # Identify high-risk tests
            high_risk_tests = [
                (name, data) for name, data in trends.items()
                if data["stability_score"] < 0.5 or data["trend_direction"] == "declining"
            ]
            
            output += f"High Risk Tests: {len(high_risk_tests)}\n"
            
            if high_risk_tests:
                output += "\nTop Risk Tests:\n"
                sorted_risks = sorted(high_risk_tests, key=lambda x: x[1]["stability_score"])[:5]
                for name, data in sorted_risks:
                    output += f"• {name}: Stability {data['stability_score']:.2f}, Trend: {data['trend_direction']}\n"
            
            # Overall risk level
            total_tests = len(trends)
            risk_percentage = (len(high_risk_tests) / total_tests * 100) if total_tests > 0 else 0
            
            if risk_percentage > 20:
                risk_level = "HIGH"
            elif risk_percentage > 10:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            output += f"\nOverall Risk Level: {risk_level} ({risk_percentage:.1f}% of tests at risk)\n"
        
        elif insight_type == "trends_analysis":
            trends = db_manager.calculate_test_trends(time_frame)
            coverage_trends = db_manager.get_coverage_trends(days_back=time_frame)
            
            output += f"📈 Trends Analysis ({time_frame} days):\n\n"
            
            improving = len([t for t in trends.values() if t["trend_direction"] == "improving"])
            declining = len([t for t in trends.values() if t["trend_direction"] == "declining"])
            stable = len([t for t in trends.values() if t["trend_direction"] == "stable"])
            
            output += f"Test Stability Trends:\n"
            output += f"• Improving: {improving} tests\n"
            output += f"• Stable: {stable} tests\n"
            output += f"• Declining: {declining} tests\n\n"
            
            output += f"Coverage Trends:\n"
            for module, trend_data in coverage_trends.items():
                trend = trend_data.get("trend", "stable")
                coverage = trend_data.get("current_coverage", 0)
                output += f"• {module}: {coverage:.1f}% ({trend})\n"
        
        else:
            return [types.TextContent(type="text", text=f"Unknown insight type: {insight_type}")]
        
        # Store insights in database
        db_manager.store_ai_analysis(
            insight_type,
            {"time_frame": time_frame},
            {"insights": output, "generated_at": str(datetime.now())}
        )
        
        return [types.TextContent(type="text", text=output)]
        
    except Exception as e:
        return [types.TextContent(type="text", text=f"AI insights generation failed: {str(e)}")]

async def _discover_tests_enhanced(args: Dict[str, Any]) -> List[types.TextContent]:
    """Enhanced test discovery with performance and stability metrics."""
    test_type = args.get("test_type", "all")
    include_metrics = args.get("include_metrics", True)
    include_disabled = args.get("include_disabled", False)
    
    output = f"🔍 Enhanced Test Discovery\n"
    output += "=" * 50 + "\n\n"
    
    try:
        # Discover tests using existing functionality
        suites = suite_builder.build_suites(test_type, "all")
        
        output += f"Test Type: {test_type}\n"
        output += f"Include Metrics: {include_metrics}\n\n"
        
        total_tests = 0
        suite_details = []
        
        for suite in suites:
            suite_info = {
                "name": suite.name,
                "type": suite.type,
                "module": suite.module,
                "test_count": len(suite.tests)
            }
            
            # Add metrics if requested
            if include_metrics:
                # Get test execution history for this suite type
                execution_history = db_manager.get_test_execution_history(limit=1000)
                suite_executions = [
                    e for e in execution_history 
                    if any(test_name in e.get("test_name", "") for test_name in suite.tests)
                ]
                
                if suite_executions:
                    success_rate = len([e for e in suite_executions if e.get("status") == "passed"]) / len(suite_executions) * 100
                    avg_duration = sum(e.get("duration_seconds", 0) for e in suite_executions) / len(suite_executions)
                    
                    suite_info.update({
                        "success_rate": success_rate,
                        "avg_duration_seconds": avg_duration,
                        "execution_count": len(suite_executions)
                    })
            
            suite_details.append(suite_info)
            total_tests += len(suite.tests)
        
        output += f"Discovery Results:\n"
        output += f"Total Suites: {len(suites)}\n"
        output += f"Total Tests: {total_tests}\n\n"
        
        # Group by type
        type_summary = {}
        for suite_info in suite_details:
            suite_type = suite_info["type"]
            if suite_type not in type_summary:
                type_summary[suite_type] = {
                    "suites": 0,
                    "tests": 0,
                    "modules": set()
                }
            
            type_summary[suite_type]["suites"] += 1
            type_summary[suite_type]["tests"] += suite_info["test_count"]
            type_summary[suite_type]["modules"].add(suite_info["module"])
        
        output += "By Test Type:\n"
        for test_type, summary in type_summary.items():
            modules = ", ".join(summary["modules"])
            output += f"• {test_type}: {summary['suites']} suites, {summary['tests']} tests ({modules})\n"
        
        if include_metrics:
            output += "\nPerformance Metrics:\n"
            for suite_info in suite_details:
                if "success_rate" in suite_info:
                    output += f"• {suite_info['name']}: "
                    output += f"{suite_info['success_rate']:.1f}% success, "
                    output += f"{suite_info['avg_duration_seconds']:.2f}s avg duration, "
                    output += f"{suite_info['execution_count']} executions\n"
        
        return [types.TextContent(type="text", text=output)]
        
    except Exception as e:
        return [types.TextContent(type="text", text=f"Enhanced test discovery failed: {str(e)}")]

async def main():
    """Run the enhanced MCP server."""
    try:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                types.InitializeRequest(
                    params=types.InitializeRequestParams(
                        protocolVersion="2024-11-05",
                        capabilities=types.ClientCapabilities(),
                        clientInfo=types.Implementation(
                            name="a2a-enhanced-test-suite",
                            version="2.0.0"
                        )
                    )
                ),
            )
    except Exception as e:
        logger.error(f"Failed to start enhanced MCP server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())