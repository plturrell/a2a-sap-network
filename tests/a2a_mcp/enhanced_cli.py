#!/usr/bin/env python3
"""
Enhanced A2A Test MCP CLI Tool with AI and Database Integration
Advanced command-line interface with GROKClient AI analysis and comprehensive database tracking
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

# Add the tests directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import enhanced components with fallbacks
try:
    from a2a_mcp.ai.grok_integration import AITestAnalyzer, AITestOptimizer, AIInsightsDashboard
    from a2a_mcp.database.test_database_manager import TestDatabaseManager, TestAnalyticsService
    from a2a_mcp.tools.test_executor import TestExecutor, TestSuiteBuilder, TestReporter
    from a2a_mcp.agents.test_orchestrator import TestOrchestrator, TestPriority
    FULL_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Full imports not available ({e}), using fallback implementations...")
    FULL_IMPORTS_AVAILABLE = False
    
    # Fallback implementations
    import sqlite3
    import uuid
    from dataclasses import dataclass
    from enum import Enum
    
    class TestStatus(Enum):
        PASSED = "passed"
        FAILED = "failed"
        SKIPPED = "skipped"
        ERROR = "error"
    
    class TestPriority(Enum):
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
    
    @dataclass
    class TestResult:
        name: str
        status: TestStatus
        duration: float = 0.0
        output: str = ""
        error: str = ""
    
    class TestDatabaseManager:
        def __init__(self, db_path=None):
            self.db_path = db_path or Path(__file__).parent.parent / "test_results.db"
            self.db_path.parent.mkdir(exist_ok=True)
            self._init_database()
        
        def _init_database(self):
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS test_executions (
                        id TEXT PRIMARY KEY,
                        test_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        duration_seconds REAL,
                        error_message TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
        
        def get_database_stats(self):
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM test_executions")
                count = cursor.fetchone()[0]
                return {"test_executions_count": count, "database_size_bytes": 0}
    
    class TestAnalyticsService:
        def __init__(self, db_manager):
            self.db_manager = db_manager
    
    class AITestAnalyzer:
        async def analyze_test_failures(self, failed_tests):
            return {"analysis": "Fallback AI analysis", "recommendations": ["Use full installation for AI features"]}
    
    class AITestOptimizer:
        def __init__(self, analyzer):
            self.analyzer = analyzer
    
    class AIInsightsDashboard:
        def __init__(self, analyzer):
            self.analyzer = analyzer
    
    class TestExecutor:
        def __init__(self, test_root):
            self.test_root = test_root
    
    class TestSuiteBuilder:
        def __init__(self, test_root):
            self.test_root = test_root
    
    class TestReporter:
        def __init__(self, test_root):
            self.test_root = test_root
    
    class TestOrchestrator:
        def __init__(self, test_root):
            self.test_root = test_root
            self.workflows = {}
        
        async def create_workflow(self, name, **kwargs):
            workflow_id = str(uuid.uuid4())
            return workflow_id

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedA2ATestCLI:
    """Enhanced command-line interface with AI and database integration."""
    
    def __init__(self, test_root: Path):
        self.test_root = test_root
        
        # Initialize enhanced components
        self.db_manager = TestDatabaseManager()
        self.analytics_service = TestAnalyticsService(self.db_manager)
        self.ai_analyzer = AITestAnalyzer()
        self.ai_optimizer = AITestOptimizer(self.ai_analyzer)
        self.ai_dashboard = AIInsightsDashboard(self.ai_analyzer)
        
        # Initialize existing components
        self.executor = TestExecutor(test_root)
        self.suite_builder = TestSuiteBuilder(test_root)
        self.reporter = TestReporter(test_root)
        self.orchestrator = TestOrchestrator(test_root)
    
    async def run_tests_enhanced(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Execute tests with AI optimization and database tracking."""
        logger.info(f"Enhanced test execution - Type: {args.test_type}, Module: {args.module}")
        
        # Create workflow
        workflow_id = await self.orchestrator.create_workflow(
            name=f"Enhanced CLI {args.test_type} Tests",
            test_type=args.test_type,
            module=args.module,
            priority=TestPriority.HIGH if args.priority == "high" else TestPriority.MEDIUM,
            coverage=args.coverage,
            parallel=args.parallel
        )
        
        # Apply AI optimization if requested
        if args.ai_optimize:
            print("🤖 Applying AI optimizations...")
            execution_history = self.db_manager.get_workflow_history(limit=50)
            workflow = self.orchestrator.workflows[workflow_id]
            optimized_workflow = await self.ai_optimizer.optimize_workflow(workflow, execution_history)
            self.orchestrator.workflows[workflow_id] = optimized_workflow
        
        # Execute workflow
        print(f"🚀 Executing workflow: {workflow_id}")
        result_workflow = await self.orchestrator.execute_workflow(workflow_id)
        
        # Store results in database
        if args.store_results:
            print("💾 Storing results in database...")
            self.db_manager.store_workflow(result_workflow)
            if result_workflow.results:
                self.db_manager.store_test_results(result_workflow.results, workflow_id)
        
        # Generate summary
        passed = len([r for r in result_workflow.results if r.status.value == "passed"])
        failed = len([r for r in result_workflow.results if r.status.value == "failed"])
        total = len(result_workflow.results)
        
        print(f"\n📊 Execution Summary:")
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"Status: {result_workflow.status.value}")
        
        # AI failure analysis if there are failures
        if failed > 0 and args.ai_analyze:
            print("\n🤖 AI Failure Analysis:")
            failed_tests = [r for r in result_workflow.results if r.status.value == "failed"]
            analysis = await self.ai_analyzer.analyze_test_failures(failed_tests)
            
            if args.store_results:
                self.db_manager.store_ai_analysis("failure_analysis", 
                                                {"failed_count": failed}, 
                                                analysis, workflow_id)
            
            print(f"Analysis: {analysis.get('analysis', 'No analysis available')}")
            
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                print("\nRecommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"{i}. {rec}")
        
        # Generate report
        if args.format == "json":
            report = self.reporter.generate_json_report(result_workflow.results)
            print(f"\n{report}")
        
        return {
            "workflow_id": workflow_id,
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "status": result_workflow.status.value
        }
    
    async def analyze_failures(self, args: argparse.Namespace) -> Dict[str, Any]:
        """AI-powered failure analysis."""
        print("🤖 AI Failure Analysis")
        print("=" * 50)
        
        # Get failed tests based on criteria
        if args.workflow_id:
            execution_history = self.db_manager.get_test_execution_history(limit=1000)
            failed_executions = [
                e for e in execution_history 
                if e.get("workflow_id") == args.workflow_id and e.get("status") == "failed"
            ]
        elif args.test_name:
            executions = self.db_manager.get_test_execution_history(test_name=args.test_name, limit=20)
            failed_executions = [e for e in executions if e.get("status") == "failed"]
        else:
            execution_history = self.db_manager.get_test_execution_history(limit=100)
            failed_executions = [e for e in execution_history if e.get("status") == "failed"]
        
        if not failed_executions:
            print("No failed tests found for analysis.")
            return {"error": "No failures found"}
        
        print(f"Analyzing {len(failed_executions)} failed test executions...")
        
        # Convert to TestResult format for AI analysis
        from mcp.tools.test_executor import TestResult, TestStatus
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
        
        # Get AI analysis
        analysis = await self.ai_analyzer.analyze_test_failures(failed_tests)
        
        # Store analysis in database
        self.db_manager.store_ai_analysis(
            "failure_analysis",
            {"analyzed_count": len(failed_tests)},
            analysis,
            args.workflow_id
        )
        
        # Display results
        if args.format == "json":
            print(json.dumps(analysis, indent=2))
        else:
            if "analysis" in analysis:
                print(f"\n📋 Analysis:\n{analysis['analysis']}")
            
            if "patterns" in analysis and analysis["patterns"]:
                print(f"\n🔍 Patterns:")
                for pattern in analysis["patterns"]:
                    print(f"• {pattern}")
            
            if "recommendations" in analysis and analysis["recommendations"]:
                print(f"\n💡 Recommendations:")
                for i, rec in enumerate(analysis["recommendations"], 1):
                    print(f"{i}. {rec}")
        
        return analysis
    
    async def get_analytics(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Get comprehensive test analytics."""
        print(f"📊 Test Analytics - {args.report_type}")
        print("=" * 50)
        
        if args.report_type == "health":
            report = self.analytics_service.generate_test_health_report(args.days)
            
            print(f"\n🏥 Test Health Report ({args.days} days):")
            print(f"Total Tests: {report['test_health']['total_tests']}")
            print(f"Stable Tests: {report['test_health']['stable_tests']}")
            print(f"Flaky Tests: {report['test_health']['flaky_tests']}")
            print(f"Stability: {report['test_health']['stability_percentage']:.1f}%")
            print(f"Workflow Success Rate: {report['workflow_health']['success_rate_percentage']:.1f}%")
            print(f"Average Coverage: {report['coverage_health']['average_coverage_percentage']:.1f}%")
            
            if report.get('flaky_tests'):
                print(f"\n⚠️ Flaky Tests:")
                for test in report['flaky_tests'][:5]:
                    print(f"• {test['test_name']} (Stability: {test['stability_score']:.2f})")
            
        elif args.report_type == "trends":
            trends = self.db_manager.calculate_test_trends(args.days)
            
            print(f"\n📈 Test Trends ({args.days} days, {len(trends)} tests):")
            
            improving = [name for name, data in trends.items() if data["trend_direction"] == "improving"]
            declining = [name for name, data in trends.items() if data["trend_direction"] == "declining"]
            stable = [name for name, data in trends.items() if data["trend_direction"] == "stable"]
            
            print(f"Improving: {len(improving)} tests")
            print(f"Stable: {len(stable)} tests")
            print(f"Declining: {len(declining)} tests")
            
            if declining:
                print(f"\n⚠️ Declining Tests:")
                sorted_declining = sorted(
                    [(name, trends[name]["stability_score"]) for name in declining],
                    key=lambda x: x[1]
                )[:5]
                for name, score in sorted_declining:
                    print(f"• {name} (Stability: {score:.2f})")
        
        elif args.report_type == "optimization":
            opportunities = self.analytics_service.identify_optimization_opportunities()
            
            print(f"\n🔧 Optimization Opportunities:")
            for opp in opportunities:
                print(f"\n📌 {opp['title']}")
                print(f"   Type: {opp['type']}")
                print(f"   Impact: {opp['impact']}")
                print(f"   Description: {opp['description']}")
                
                if 'tests' in opp:
                    print(f"   Affected Tests: {len(opp['tests'])}")
        
        elif args.report_type == "coverage":
            coverage_trends = self.db_manager.get_coverage_trends(days_back=args.days)
            
            print(f"\n📈 Coverage Trends ({args.days} days):")
            for module, trend_data in coverage_trends.items():
                print(f"• {module}: {trend_data['current_coverage']:.1f}% ({trend_data['trend']})")
        
        return {"report_type": args.report_type, "success": True}
    
    async def ai_insights(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Generate AI insights and recommendations."""
        print(f"🤖 AI Insights: {args.insight_type}")
        print("=" * 50)
        
        if args.insight_type == "executive_summary":
            # Get recent workflows
            workflows = []  # Would convert from DB data to workflow objects
            summary = await self.ai_dashboard.generate_executive_summary(workflows, args.days)
            
            print(f"\n📋 Executive Summary ({args.days} days):")
            
            if "key_metrics" in summary:
                metrics = summary["key_metrics"]
                print(f"\nKey Metrics:")
                print(f"• Workflow Success Rate: {metrics.get('workflow_success_rate', 0):.1f}%")
                print(f"• Test Pass Rate: {metrics.get('test_pass_rate', 0):.1f}%")
                print(f"• Total Workflows: {metrics.get('total_workflows', 0)}")
                print(f"• Total Tests: {metrics.get('total_tests', 0)}")
            
            if "recommendations" in summary:
                recs = summary["recommendations"]
                print(f"\n🎯 Priority Recommendations:")
                for i, rec in enumerate(recs[:5], 1):
                    priority = rec.get("priority", 1)
                    category = rec.get("category", "General")
                    print(f"{i}. [{category}] {rec.get('recommendation', 'No recommendation')} (Priority: {priority})")
        
        elif args.insight_type == "stability_prediction":
            execution_history = self.db_manager.get_test_execution_history(limit=2000, days_back=args.days)
            prediction = await self.ai_analyzer.predict_test_stability(execution_history)
            
            print(f"\n🔮 Test Stability Prediction ({args.days} days):")
            
            if "assessment" in prediction:
                print(f"\n📊 Overall Assessment:\n{prediction['assessment']}")
            
            if "flaky_tests" in prediction and prediction["flaky_tests"]:
                print(f"\n⚠️ Potentially Flaky Tests:")
                for test in prediction["flaky_tests"]:
                    if isinstance(test, dict):
                        confidence = test.get("confidence", "unknown")
                        print(f"• {test.get('name', 'unknown')} (Confidence: {confidence})")
                    else:
                        print(f"• {test}")
            
            if "recommendations" in prediction and prediction["recommendations"]:
                print(f"\n💡 Stability Recommendations:")
                for i, rec in enumerate(prediction["recommendations"], 1):
                    print(f"{i}. {rec}")
        
        elif args.insight_type == "performance_analysis":
            # Get execution history for performance analysis
            execution_history = self.db_manager.get_workflow_history(limit=100, days_back=args.days)
            
            print(f"\n⚡ Performance Analysis:")
            
            if execution_history:
                # Calculate performance metrics
                total_workflows = len(execution_history)
                successful = len([w for w in execution_history if w["status"] == "completed"])
                
                print(f"• Total Workflows: {total_workflows}")
                print(f"• Success Rate: {successful/total_workflows*100:.1f}%")
                
                # Calculate average durations if available
                durations = []
                for workflow in execution_history:
                    if workflow.get("started_at") and workflow.get("completed_at"):
                        start = datetime.fromisoformat(workflow["started_at"].replace('Z', '+00:00'))
                        end = datetime.fromisoformat(workflow["completed_at"].replace('Z', '+00:00'))
                        duration = (end - start).total_seconds()
                        durations.append(duration)
                
                if durations:
                    avg_duration = sum(durations) / len(durations)
                    print(f"• Average Duration: {avg_duration:.2f} seconds")
                    print(f"• Fastest: {min(durations):.2f} seconds")
                    print(f"• Slowest: {max(durations):.2f} seconds")
        
        return {"insight_type": args.insight_type, "success": True}
    
    def database_status(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Get database status and statistics."""
        print("💾 Database Status")
        print("=" * 50)
        
        stats = self.db_manager.get_database_stats()
        
        print(f"\n📊 Database Statistics:")
        print(f"• Test Executions: {stats.get('test_executions_count', 0)}")
        print(f"• Workflows: {stats.get('workflows_count', 0)}")
        print(f"• AI Analyses: {stats.get('ai_analysis_count', 0)}")
        print(f"• Performance Metrics: {stats.get('performance_metrics_count', 0)}")
        print(f"• Coverage Records: {stats.get('coverage_history_count', 0)}")
        print(f"• Database Size: {stats.get('database_size_bytes', 0) / 1024 / 1024:.2f} MB")
        
        print(f"\n📈 Recent Activity (7 days):")
        print(f"• Test Executions: {stats.get('recent_test_executions', 0)}")
        print(f"• Workflows: {stats.get('recent_workflows', 0)}")
        
        if args.cleanup:
            print(f"\n🧹 Cleaning up old data...")
            cleanup_stats = self.db_manager.cleanup_old_data(args.retention_days)
            
            print(f"Cleanup Results:")
            for table, count in cleanup_stats.items():
                print(f"• {table}: {count} records removed")
        
        return stats

def main():
    """Enhanced CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced A2A Test MCP CLI Tool with AI and Database Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Examples:
  # Run tests with AI optimization and database tracking
  python enhanced_cli.py run --test-type unit --ai-optimize --store-results --coverage
  
  # AI failure analysis for specific workflow
  python enhanced_cli.py analyze-failures --workflow-id workflow_123
  
  # Generate comprehensive analytics
  python enhanced_cli.py analytics --report-type health --days 30
  
  # AI insights and predictions
  python enhanced_cli.py ai-insights --insight-type stability_prediction --days 30
  
  # Database management
  python enhanced_cli.py database --cleanup --retention-days 90
        """
    )
    
    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Test root directory (default: ../)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Enhanced run tests command
    run_parser = subparsers.add_parser("run", help="Execute tests with AI optimization")
    run_parser.add_argument("--test-type", choices=["unit", "integration", "e2e", "performance", "security", "contracts", "all"], default="unit")
    run_parser.add_argument("--module", choices=["a2aAgents", "a2aNetwork", "common", "all"], default="all")
    run_parser.add_argument("--priority", choices=["high", "medium", "low"], default="medium")
    run_parser.add_argument("--parallel", action="store_true", help="Enable parallel execution")
    run_parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    run_parser.add_argument("--ai-optimize", action="store_true", help="Apply AI optimizations")
    run_parser.add_argument("--ai-analyze", action="store_true", help="AI failure analysis")
    run_parser.add_argument("--store-results", action="store_true", default=True, help="Store results in database")
    run_parser.add_argument("--format", choices=["text", "json"], default="text")
    
    # AI failure analysis command
    analyze_parser = subparsers.add_parser("analyze-failures", help="AI-powered failure analysis")
    analyze_parser.add_argument("--workflow-id", help="Workflow ID to analyze")
    analyze_parser.add_argument("--test-name", help="Specific test name to analyze")
    analyze_parser.add_argument("--format", choices=["text", "json"], default="text")
    
    # Analytics command
    analytics_parser = subparsers.add_parser("analytics", help="Test analytics and trends")
    analytics_parser.add_argument("--report-type", choices=["health", "trends", "optimization", "coverage"], default="health")
    analytics_parser.add_argument("--days", type=int, default=30, help="Analysis period in days")
    analytics_parser.add_argument("--format", choices=["text", "json"], default="text")
    
    # AI insights command
    insights_parser = subparsers.add_parser("ai-insights", help="AI insights and recommendations")
    insights_parser.add_argument("--insight-type", choices=["executive_summary", "stability_prediction", "performance_analysis"], default="executive_summary")
    insights_parser.add_argument("--days", type=int, default=30, help="Analysis period in days")
    
    # Database management command
    db_parser = subparsers.add_parser("database", help="Database management and status")
    db_parser.add_argument("--cleanup", action="store_true", help="Clean up old data")
    db_parser.add_argument("--retention-days", type=int, default=90, help="Data retention period")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize enhanced CLI
    cli = EnhancedA2ATestCLI(args.test_root)
    
    # Execute command
    try:
        if args.command == "run":
            result = asyncio.run(cli.run_tests_enhanced(args))
            
        elif args.command == "analyze-failures":
            result = asyncio.run(cli.analyze_failures(args))
            
        elif args.command == "analytics":
            result = asyncio.run(cli.get_analytics(args))
            
        elif args.command == "ai-insights":
            result = asyncio.run(cli.ai_insights(args))
            
        elif args.command == "database":
            result = cli.database_status(args)
        
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()