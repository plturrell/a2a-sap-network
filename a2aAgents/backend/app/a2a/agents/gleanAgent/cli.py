#!/usr/bin/env python3
"""
GleanAgent CLI - Comprehensive Command Line Interface
A2A-compliant code analysis agent with full real implementations

Usage:
    python cli.py analyze <directory> [options]
    python cli.py lint <directory> [options]
    python cli.py security <directory> [options]
    python cli.py refactor <file> [options]
    python cli.py complexity <directory> [options]
    python cli.py coverage <directory> [options]
    python cli.py quality <directory> [options]
    python cli.py history <directory> [options]
    python cli.py server [options]
    python cli.py --help
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import argparse
import sys
import json
import os
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime
import time

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.a2a.agents.gleanAgent import GleanAgent
from app.a2a.agents.gleanAgent.intelligentScanManager import IntelligentScanCLI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GleanAgentCLI:
    """Comprehensive CLI for GleanAgent with all real implementations"""

    def __init__(self):
        self.agent = None
        self.start_time = None
        self.intelligent_scan_manager = None

    async def initialize_agent(self):
        """Initialize the GleanAgent"""
        print("🚀 Initializing GleanAgent with A2A Protocol...")
        self.start_time = time.time()

        try:
            self.agent = GleanAgent()
            print(f"✅ GleanAgent initialized: {self.agent.agent_id}")
            print(f"📋 Agent capabilities: {len(self.agent.list_mcp_tools())} MCP tools available")

            # Initialize intelligent scan manager with proper database path
            import tempfile
            scan_db_path = os.path.join(tempfile.gettempdir(), f"intelligent_scan_{os.getpid()}.db")
            self.intelligent_scan_manager = IntelligentScanCLI(scan_db_path)
            print(f"🧠 Intelligent Scan Manager initialized")

            return True
        except Exception as e:
            print(f"❌ Failed to initialize GleanAgent: {e}")
            return False

    async def comprehensive_analysis(self, directory: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive analysis with all real implementations"""
        print(f"\n🔍 COMPREHENSIVE CODE ANALYSIS")
        print(f"📁 Directory: {directory}")
        print("=" * 70)

        analysis_types = ["lint", "complexity", "glean", "security", "coverage"]
        if options.get("quick"):
            analysis_types = ["lint", "complexity", "glean"]
            print("⚡ Quick analysis mode - skipping security and coverage")

        # Run comprehensive parallel analysis
        result = await self.agent.analyze_project_comprehensive_parallel(
            directory=directory,
            analysis_types=analysis_types,
            max_concurrent=options.get("max_concurrent", 3)
        )

        # Display results
        self._display_comprehensive_results(result)

        # Save results if requested
        if options.get("output"):
            await self._save_results(result, options["output"], "comprehensive")

        return result

    async def lint_analysis(self, directory: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Run real linting analysis with severity and language filtering"""
        print(f"\n🔍 REAL LINTING ANALYSIS")
        print(f"📁 Directory: {directory}")

        # Show filters being applied
        if options.get("min_severity"):
            print(f"🎯 Minimum severity: {options['min_severity']}")
        if options.get("severity"):
            print(f"🎯 Exact severity: {options['severity']}")
        if options.get("languages"):
            print(f"🌐 Languages: {', '.join(options['languages'])}")
        print("-" * 50)

        file_patterns = options.get("patterns", ["*.py", "*.js", "*.ts"])

        # Pass filtering options to the analysis method
        result = await self.agent._perform_lint_analysis(directory, file_patterns, options)

        print(f"📊 Analysis Results:")
        print(f"   Files analyzed: {result.get('files_analyzed', 0)}")
        print(f"   Total issues: {result.get('total_issues', 0)}")
        print(f"   Critical issues: {result.get('critical_issues', 0)}")
        print(f"   Languages scanned: {', '.join(result.get('languages_scanned', []))}")
        print(f"   Linters used: {list(result.get('linter_results', {}).keys())}")
        print(f"   Duration: {result.get('duration', 0):.2f}s")

        # Show severity breakdown
        if result.get('issues_by_severity'):
            print(f"\n📈 Issues by Severity:")
            for severity, count in result['issues_by_severity'].items():
                print(f"   {severity.title()}: {count}")

        # Show filtered issues
        if options.get("show_issues", True) and result.get('issues'):
            filtered_issues = self._filter_issues_by_severity(result['issues'], options)
            self._display_lint_issues(filtered_issues, options.get("max_issues", 10))

        if options.get("output"):
            await self._save_results(result, options["output"], "lint")

        return result

    async def security_analysis(self, directory: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Run real security vulnerability analysis"""
        print(f"\n🛡️  REAL SECURITY VULNERABILITY ANALYSIS")
        print(f"📁 Directory: {directory}")
        print("-" * 50)

        scan_dev = options.get("include_dev", False)

        result = await self.agent.scan_dependency_vulnerabilities(directory, scan_dev)

        print(f"🔒 Security Analysis Results:")
        print(f"   Total vulnerabilities: {result.get('total_vulnerabilities', 0)}")
        print(f"   Scanned files: {result.get('scanned_files', 0)}")
        print(f"   Database version: {result.get('database_version', 'unknown')}")

        # Show risk metrics
        if 'risk_metrics' in result:
            metrics = result['risk_metrics']
            print(f"\n⚠️  Risk Assessment:")
            print(f"   Risk Score: {metrics.get('risk_score', 0)}/100")
            print(f"   Risk Level: {metrics.get('risk_level', 'unknown').title()}")
            print(f"   Critical: {metrics.get('critical_count', 0)}")
            print(f"   High: {metrics.get('high_count', 0)}")
            print(f"   Medium: {metrics.get('medium_count', 0)}")
            print(f"   Low: {metrics.get('low_count', 0)}")

        # Show vulnerabilities by source
        if options.get("show_vulnerabilities", True) and result.get('vulnerabilities'):
            self._display_security_vulnerabilities(result['vulnerabilities'], options.get("max_vulns", 15))

        if options.get("output"):
            await self._save_results(result, options["output"], "security")

        return result

    async def refactoring_analysis(self, file_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Run real AST-based refactoring analysis"""
        print(f"\n🔧 REAL AST-BASED REFACTORING ANALYSIS")
        print(f"📄 File: {file_path}")
        print("-" * 50)

        max_suggestions = options.get("max_suggestions", 15)

        result = await self.agent.analyze_code_refactoring(file_path, max_suggestions)

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return result

        print(f"🔧 Refactoring Analysis Results:")
        print(f"   Total suggestions: {result.get('total_suggestions', 0)}")

        summary = result.get('summary', {})
        print(f"   Critical priority: {summary.get('critical_priority', 0)}")
        print(f"   High priority: {summary.get('high_priority', 0)}")
        print(f"   Medium priority: {summary.get('medium_priority', 0)}")
        print(f"   Low priority: {summary.get('low_priority', 0)}")

        # Show metrics
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"\n📊 Refactoring Metrics:")
            print(f"   Priority Score: {metrics.get('refactoring_priority_score', 0)}")
            print(f"   Maintainability Index: {metrics.get('maintainability_index', 0)}/100")
            print(f"   Functions: {metrics.get('node_counts', {}).get('functions', 0)}")
            print(f"   Classes: {metrics.get('node_counts', {}).get('classes', 0)}")

        # Show suggestions
        if options.get("show_suggestions", True) and result.get('suggestions'):
            self._display_refactoring_suggestions(result['suggestions'])

        if options.get("output"):
            await self._save_results(result, options["output"], "refactoring")

        return result

    async def complexity_analysis(self, directory: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Run real AST-based complexity analysis"""
        print(f"\n📊 REAL AST-BASED COMPLEXITY ANALYSIS")
        print(f"📁 Directory: {directory}")
        print("-" * 50)

        file_patterns = options.get("patterns", ["*.py"])
        threshold = options.get("threshold", 10)

        result = await self.agent.analyze_code_complexity(directory, file_patterns, threshold)

        print(f"📊 Complexity Analysis Results:")
        print(f"   Files analyzed: {result.get('files_analyzed', 0)}")
        print(f"   Functions analyzed: {result.get('functions_analyzed', 0)}")
        print(f"   Classes analyzed: {result.get('classes_analyzed', 0)}")
        print(f"   Average complexity: {result.get('average_complexity', 0):.2f}")
        print(f"   Max complexity: {result.get('max_complexity', 0)}")
        print(f"   Duration: {result.get('duration', 0):.2f}s")

        # Show complexity distribution
        if result.get('complexity_distribution'):
            print(f"\n📈 Complexity Distribution:")
            for range_key, count in result['complexity_distribution'].items():
                print(f"   {range_key}: {count} functions")

        # Show high complexity functions
        high_complexity = result.get('high_complexity_functions', [])
        if high_complexity and options.get("show_functions", True):
            print(f"\n⚠️  High Complexity Functions ({len(high_complexity)}):")
            for func in high_complexity[:options.get("max_functions", 10)]:
                print(f"   • {func['name']}: complexity {func['complexity']} (line {func['line']})")

        # Show recommendations
        if result.get('recommendations') and options.get("show_recommendations", True):
            print(f"\n💡 Recommendations:")
            for rec in result['recommendations'][:5]:
                print(f"   • {rec}")

        if options.get("output"):
            await self._save_results(result, options["output"], "complexity")

        return result

    async def coverage_analysis(self, directory: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Run real test coverage analysis"""
        print(f"\n🧪 REAL TEST COVERAGE ANALYSIS")
        print(f"📁 Directory: {directory}")
        print("-" * 50)

        result = await self.agent.analyze_test_coverage(directory)

        print(f"🧪 Coverage Analysis Results:")
        print(f"   Overall coverage: {result.get('overall_coverage', 0):.1f}%")
        print(f"   Test files found: {result.get('test_files_count', 0)}")
        print(f"   Source files: {result.get('source_files_count', 0)}")

        # Show file coverage details
        if result.get('file_coverage') and options.get("show_files", True):
            print(f"\n📄 File Coverage Details:")
            for file_info in result['file_coverage'][:options.get("max_files", 10)]:
                filename = Path(file_info['file']).name
                coverage = file_info.get('coverage_percentage', 0)
                print(f"   • {filename}: {coverage:.1f}%")

        if options.get("output"):
            await self._save_results(result, options["output"], "coverage")

        return result

    async def quality_analysis(self, directory: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive quality score"""
        print(f"\n📈 COMPREHENSIVE QUALITY ANALYSIS")
        print(f"📁 Directory: {directory}")
        print("-" * 50)

        # Run basic analyses needed for quality score
        print("Running component analyses...")

        lint_result = await self.agent._perform_lint_analysis(directory, ["*.py", "*.js", "*.ts"])
        complexity_result = await self.agent.analyze_code_complexity(directory, ["*.py"])
        security_result = await self.agent.scan_dependency_vulnerabilities(directory)

        summary = {
            "files_analyzed": lint_result.get("files_analyzed", 0),
            "total_issues": lint_result.get("total_issues", 0),
            "critical_issues": lint_result.get("critical_issues", 0),
            "test_coverage": 0  # Would need coverage analysis
        }

        analyses = {
            "lint": lint_result,
            "complexity": complexity_result,
            "security": security_result
        }

        quality_score = self.agent._calculate_comprehensive_quality_score(summary, analyses)

        result = {
            "quality_score": quality_score,
            "component_scores": {
                "code_quality": f"Based on {summary['total_issues']} issues across {summary['files_analyzed']} files",
                "security": f"Risk level: {security_result.get('risk_metrics', {}).get('risk_level', 'unknown')}",
                "complexity": f"Average: {complexity_result.get('average_complexity', 0):.2f}",
                "test_coverage": "Not analyzed (run coverage command separately)"
            },
            "summary": summary,
            "timestamp": datetime.utcnow().isoformat()
        }

        print(f"📊 Quality Analysis Results:")
        print(f"   Overall Quality Score: {quality_score:.1f}/100")
        print(f"\n📋 Component Breakdown:")
        for component, description in result["component_scores"].items():
            print(f"   • {component.replace('_', ' ').title()}: {description}")

        # Quality recommendations
        print(f"\n💡 Quality Recommendations:")
        if quality_score < 60:
            print("   🔴 Poor quality - immediate attention required")
            print("   • Focus on reducing critical issues")
            print("   • Improve test coverage")
            print("   • Address security vulnerabilities")
        elif quality_score < 80:
            print("   🟡 Moderate quality - improvement needed")
            print("   • Reduce code complexity")
            print("   • Add more comprehensive tests")
            print("   • Fix high-priority issues")
        else:
            print("   🟢 Good quality - maintain current standards")
            print("   • Continue monitoring quality metrics")
            print("   • Address any new issues promptly")

        if options.get("output"):
            await self._save_results(result, options["output"], "quality")

        return result

    async def analysis_history(self, directory: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Show analysis history and trends"""
        print(f"\n📜 ANALYSIS HISTORY & TRENDS")
        print(f"📁 Directory: {directory}")
        print("-" * 50)

        limit = options.get("limit", 10)
        days = options.get("days", 7)

        # Get history
        history = await self.agent.get_analysis_history(directory, limit)
        trends = await self.agent.get_quality_trends(directory, days)

        print(f"📊 Analysis History (last {limit} analyses):")
        if isinstance(history, list) and history:
            for i, analysis in enumerate(history, 1):
                timestamp = analysis.get("timestamp", "unknown")
                analysis_type = analysis.get("analysis_type", "unknown")
                score = analysis.get("quality_score", "N/A")
                print(f"   {i}. {timestamp} - {analysis_type} (Quality: {score})")
        else:
            print("   No analysis history found")

        print(f"\n📈 Quality Trends (last {days} days):")
        if isinstance(trends, dict) and "message" not in trends:
            analyses_count = trends.get("analyses_count", 0)
            if analyses_count > 0:
                print(f"   Total analyses: {analyses_count}")
                if trends.get("trend"):
                    print(f"   Trend: {trends['trend']}")
                if trends.get("average_score"):
                    print(f"   Average score: {trends['average_score']:.1f}/100")
            else:
                print("   No trend data available")
        else:
            print(f"   {trends.get('message', 'No trend data available')}")

        result = {"history": history, "trends": trends}

        if options.get("output"):
            await self._save_results(result, options["output"], "history")

        return result

    async def smart_scan(self, directory: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Execute intelligent scan with change detection and optimization"""
        print(f"\n🧠 INTELLIGENT SMART SCAN")
        print(f"📁 Directory: {directory}")
        print("=" * 70)

        # First scan for changes
        print("🔍 Scanning for file changes...")
        await self.intelligent_scan_manager.scan_changes(directory)

        # Show recommendations
        print("\n🎯 Generating scan recommendations...")
        await self.intelligent_scan_manager.show_recommendations()

        # Execute smart scan
        max_files = options.get("max_files", 20)
        print(f"\n🚀 Executing smart scan (max {max_files} files)...")
        await self.intelligent_scan_manager.execute_smart_scan(max_files)

        return {"status": "completed", "type": "smart_scan"}

    async def scan_analytics(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Show comprehensive scan analytics and history"""
        print(f"\n📈 SCAN ANALYTICS & HISTORICAL TRACKING")
        print("=" * 70)

        # Show analytics dashboard
        await self.intelligent_scan_manager.show_analytics()

        # Show scan history
        days = options.get("days", 7)
        print(f"\n📜 Recent Scan History:")
        await self.intelligent_scan_manager.show_history(days)

        return {"status": "completed", "type": "analytics"}

    async def change_detection(self, directory: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Run change detection analysis"""
        print(f"\n🔄 CHANGE DETECTION ANALYSIS")
        print(f"📁 Directory: {directory}")
        print("=" * 70)

        await self.intelligent_scan_manager.scan_changes(directory)

        return {"status": "completed", "type": "change_detection"}

    async def start_server(self, options: Dict[str, Any]):
        """Start GleanAgent as A2A server"""
        print(f"\n🌐 STARTING GLEAN AGENT A2A SERVER")
        print("-" * 50)

        port = options.get("port", 8016)
        host = options.get("host", "0.0.0.0")

        print(f"🚀 Starting server on {host}:{port}")
        print(f"📋 Available MCP tools: {len(self.agent.list_mcp_tools())}")
        print(f"📊 Available MCP resources: {len(self.agent.list_mcp_resources())}")
        print(f"🔗 A2A Agent ID: {self.agent.agent_id}")

        # Create FastAPI app
        app = self.agent.create_fastapi_app()

        # Start server
        import uvicorn
        uvicorn.run(app, host=host, port=port)

    def _display_comprehensive_results(self, result: Dict[str, Any]):
        """Display comprehensive analysis results"""
        print(f"\n📊 COMPREHENSIVE ANALYSIS RESULTS")
        print(f"   Analysis ID: {result.get('analysis_id', 'N/A')}")
        print(f"   Duration: {result.get('duration', 0):.2f}s")
        print(f"   Tasks completed: {result.get('tasks_completed', 0)}")

        if 'summary' in result:
            summary = result['summary']
            print(f"\n📋 Summary:")
            print(f"   Files analyzed: {summary.get('files_analyzed', 0)}")
            print(f"   Total issues: {summary.get('total_issues', 0)}")
            print(f"   Critical issues: {summary.get('critical_issues', 0)}")
            print(f"   Quality Score: {summary.get('quality_score', 0)}/100")

        # Show individual analysis results
        analyses = result.get('analyses', {})
        for analysis_type, analysis_result in analyses.items():
            print(f"\n   {analysis_type.title()} Analysis:")
            if analysis_type == "lint":
                print(f"     Issues found: {analysis_result.get('total_issues', 0)}")
            elif analysis_type == "complexity":
                print(f"     Avg complexity: {analysis_result.get('average_complexity', 0):.2f}")
            elif analysis_type == "security":
                print(f"     Vulnerabilities: {analysis_result.get('total_vulnerabilities', 0)}")
            elif analysis_type == "glean":
                print(f"     Files analyzed: {analysis_result.get('files_analyzed', 0)}")

    def _filter_issues_by_severity(self, issues: List[Dict], options: Dict[str, Any]) -> List[Dict]:
        """Filter issues by severity level"""
        if not issues:
            return []

        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}

        # Filter by exact severity if specified
        if options.get("severity"):
            return [issue for issue in issues if issue.get('severity') == options["severity"]]

        # Filter by minimum severity
        min_severity = options.get("min_severity", "low")
        min_level = severity_order.get(min_severity, 3)

        filtered = []
        for issue in issues:
            issue_severity = issue.get('severity', 'low')
            issue_level = severity_order.get(issue_severity, 3)
            if issue_level <= min_level:
                filtered.append(issue)

        return filtered

    def _display_lint_issues(self, issues: list, max_issues: int):
        """Display linting issues"""
        print(f"\n🔍 Issues Found (showing first {min(len(issues), max_issues)}):")

        for i, issue in enumerate(issues[:max_issues], 1):
            file_path = Path(issue.get('file_path', 'unknown')).name
            line = issue.get('line', 0)
            tool = issue.get('tool', 'unknown')
            severity = issue.get('severity', 'unknown')
            message = issue.get('message', 'No message')

            severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(severity, "⚪")

            print(f"   {i}. {severity_emoji} [{tool.upper()}] {file_path}:{line}")
            print(f"      {message}")

    def _display_security_vulnerabilities(self, vulnerabilities: list, max_vulns: int):
        """Display security vulnerabilities"""
        print(f"\n🛡️  Vulnerabilities Found (showing first {min(len(vulnerabilities), max_vulns)}):")

        # Group by source
        by_source = {}
        for vuln in vulnerabilities:
            source = vuln.get('source', 'unknown')
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(vuln)

        for source, vulns in by_source.items():
            print(f"\n   📍 {source.replace('_', ' ').title()} ({len(vulns)} found):")
            for vuln in vulns[:5]:  # Show first 5 per source
                severity = vuln.get('severity', 'unknown')
                severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(severity, "⚪")

                print(f"     {severity_emoji} {vuln.get('vulnerability_id', 'N/A')}")
                print(f"       {vuln.get('description', 'No description')}")
                if vuln.get('package') != 'source_code':
                    print(f"       Package: {vuln.get('package', 'unknown')} v{vuln.get('version', 'unknown')}")
                if vuln.get('remediation'):
                    print(f"       Fix: {vuln.get('remediation', '')}")

    def _display_refactoring_suggestions(self, suggestions: list):
        """Display refactoring suggestions"""
        print(f"\n🔧 Refactoring Suggestions:")

        for i, suggestion in enumerate(suggestions, 1):
            severity = suggestion.get('severity', 'unknown')
            severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(severity, "⚪")

            print(f"\n   {i}. {severity_emoji} {suggestion.get('type', 'unknown').replace('_', ' ').title()}")
            print(f"      Line {suggestion.get('line', 0)}: {suggestion.get('message', 'No message')}")
            print(f"      Suggestion: {suggestion.get('suggestion', 'No suggestion')}")
            if suggestion.get('code_example'):
                print(f"      Example: {suggestion.get('code_example')}")

    async def _save_results(self, result: Dict[str, Any], output_path: str, analysis_type: str):
        """Save analysis results to file"""
        try:
            output_file = Path(output_path)

            # Add timestamp and analysis type to filename if it doesn't have extension
            if not output_file.suffix:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = output_file.with_name(f"{output_file.name}_{analysis_type}_{timestamp}.json")

            # Ensure directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Add metadata
            result["cli_metadata"] = {
                "analysis_type": analysis_type,
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent.agent_id if self.agent else "unknown",
                "version": "1.0.0"
            }

            # Save as JSON
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)

            print(f"💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

    def _show_completion_summary(self):
        """Show CLI completion summary"""
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"\n⏱️  Total execution time: {total_time:.2f}s")

        print(f"✅ GleanAgent CLI execution completed")
        print(f"🔗 A2A Agent ID: {self.agent.agent_id if self.agent else 'N/A'}")


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="GleanAgent CLI - A2A-compliant code analysis with real implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Traditional Analysis Commands
  python cli.py analyze /path/to/project --output results/
  python cli.py lint /path/to/project --show-issues --max-issues 20
  python cli.py security /path/to/project --include-dev --show-vulnerabilities
  python cli.py refactor /path/to/file.py --max-suggestions 10
  python cli.py complexity /path/to/project --threshold 15
  python cli.py coverage /path/to/project --show-files
  python cli.py quality /path/to/project --output quality_report.json
  python cli.py history /path/to/project --days 30
  python cli.py server --port 8016 --host 0.0.0.0

  # Intelligent Scan Management Commands
  python cli.py smart /path/to/project --max-files 30
  python cli.py analytics --days 14
  python cli.py changes /path/to/project
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Comprehensive analysis
    analyze_parser = subparsers.add_parser('analyze', help='Run comprehensive analysis')
    analyze_parser.add_argument('directory', help='Directory to analyze')
    analyze_parser.add_argument('--output', '-o', help='Output file/directory')
    analyze_parser.add_argument('--quick', action='store_true', help='Quick analysis (skip security and coverage)')
    analyze_parser.add_argument('--max-concurrent', type=int, default=3, help='Max concurrent analyses')

    # Lint analysis
    lint_parser = subparsers.add_parser('lint', help='Run linting analysis')
    lint_parser.add_argument('directory', help='Directory to analyze')
    lint_parser.add_argument('--output', '-o', help='Output file')
    lint_parser.add_argument('--patterns', nargs='+', default=['*.py', '*.js', '*.ts'], help='File patterns')
    lint_parser.add_argument('--show-issues', action='store_true', default=True, help='Show issues')
    lint_parser.add_argument('--max-issues', type=int, default=10, help='Max issues to show')
    lint_parser.add_argument('--severity', choices=['critical', 'high', 'medium', 'low'],
                           help='Filter by severity level (show only this level and above)')
    lint_parser.add_argument('--min-severity', choices=['critical', 'high', 'medium', 'low'], default='low',
                           help='Minimum severity to include (default: low - show all)')
    lint_parser.add_argument('--languages', nargs='+',
                           choices=['python', 'javascript', 'typescript', 'html', 'xml', 'yaml', 'json', 'shell', 'css', 'scss', 'cds', 'solidity'],
                           help='Scan only specific languages (default: all detected languages)')

    # Security analysis
    security_parser = subparsers.add_parser('security', help='Run security vulnerability analysis')
    security_parser.add_argument('directory', help='Directory to analyze')
    security_parser.add_argument('--output', '-o', help='Output file')
    security_parser.add_argument('--include-dev', action='store_true', help='Include dev dependencies')
    security_parser.add_argument('--show-vulnerabilities', action='store_true', default=True, help='Show vulnerabilities')
    security_parser.add_argument('--max-vulns', type=int, default=15, help='Max vulnerabilities to show')

    # Refactoring analysis
    refactor_parser = subparsers.add_parser('refactor', help='Run refactoring analysis')
    refactor_parser.add_argument('file', help='File to analyze')
    refactor_parser.add_argument('--output', '-o', help='Output file')
    refactor_parser.add_argument('--max-suggestions', type=int, default=15, help='Max suggestions')
    refactor_parser.add_argument('--show-suggestions', action='store_true', default=True, help='Show suggestions')

    # Complexity analysis
    complexity_parser = subparsers.add_parser('complexity', help='Run complexity analysis')
    complexity_parser.add_argument('directory', help='Directory to analyze')
    complexity_parser.add_argument('--output', '-o', help='Output file')
    complexity_parser.add_argument('--patterns', nargs='+', default=['*.py'], help='File patterns')
    complexity_parser.add_argument('--threshold', type=int, default=10, help='Complexity threshold')
    complexity_parser.add_argument('--show-functions', action='store_true', default=True, help='Show high complexity functions')
    complexity_parser.add_argument('--max-functions', type=int, default=10, help='Max functions to show')
    complexity_parser.add_argument('--show-recommendations', action='store_true', default=True, help='Show recommendations')

    # Coverage analysis
    coverage_parser = subparsers.add_parser('coverage', help='Run test coverage analysis')
    coverage_parser.add_argument('directory', help='Directory to analyze')
    coverage_parser.add_argument('--output', '-o', help='Output file')
    coverage_parser.add_argument('--show-files', action='store_true', default=True, help='Show file coverage')
    coverage_parser.add_argument('--max-files', type=int, default=10, help='Max files to show')

    # Quality analysis
    quality_parser = subparsers.add_parser('quality', help='Calculate comprehensive quality score')
    quality_parser.add_argument('directory', help='Directory to analyze')
    quality_parser.add_argument('--output', '-o', help='Output file')

    # History analysis
    history_parser = subparsers.add_parser('history', help='Show analysis history and trends')
    history_parser.add_argument('directory', help='Directory to check')
    history_parser.add_argument('--output', '-o', help='Output file')
    history_parser.add_argument('--limit', type=int, default=10, help='Max history entries')
    history_parser.add_argument('--days', type=int, default=7, help='Days for trend analysis')

    # Server mode
    server_parser = subparsers.add_parser('server', help='Start GleanAgent as A2A server')
    server_parser.add_argument('--port', type=int, default=8016, help='Server port')
    server_parser.add_argument('--host', default='0.0.0.0', help='Server host')

    # Intelligent Scan Management Commands
    smart_parser = subparsers.add_parser('smart', help='Run intelligent smart scan with change detection')
    smart_parser.add_argument('directory', help='Directory to analyze')
    smart_parser.add_argument('--max-files', type=int, default=20, help='Max files to scan')
    smart_parser.add_argument('--output', '-o', help='Output file')

    # Analytics and tracking
    analytics_parser = subparsers.add_parser('analytics', help='Show scan analytics and historical tracking')
    analytics_parser.add_argument('--days', type=int, default=7, help='Days for historical analysis')
    analytics_parser.add_argument('--output', '-o', help='Output file')

    # Change detection
    changes_parser = subparsers.add_parser('changes', help='Detect file changes since last scan')
    changes_parser.add_argument('directory', help='Directory to check for changes')
    changes_parser.add_argument('--output', '-o', help='Output file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize CLI
    cli = GleanAgentCLI()

    # Initialize agent
    if not await cli.initialize_agent():
        sys.exit(1)

    try:
        # Route to appropriate command
        if args.command == 'analyze':
            options = {
                'output': args.output,
                'quick': args.quick,
                'max_concurrent': args.max_concurrent
            }
            await cli.comprehensive_analysis(args.directory, options)

        elif args.command == 'lint':
            options = {
                'output': args.output,
                'patterns': args.patterns,
                'show_issues': args.show_issues,
                'max_issues': args.max_issues,
                'severity': getattr(args, 'severity', None),
                'min_severity': getattr(args, 'min_severity', 'low'),
                'languages': getattr(args, 'languages', None)
            }
            await cli.lint_analysis(args.directory, options)

        elif args.command == 'security':
            options = {
                'output': args.output,
                'include_dev': args.include_dev,
                'show_vulnerabilities': args.show_vulnerabilities,
                'max_vulns': args.max_vulns
            }
            await cli.security_analysis(args.directory, options)

        elif args.command == 'refactor':
            options = {
                'output': args.output,
                'max_suggestions': args.max_suggestions,
                'show_suggestions': args.show_suggestions
            }
            await cli.refactoring_analysis(args.file, options)

        elif args.command == 'complexity':
            options = {
                'output': args.output,
                'patterns': args.patterns,
                'threshold': args.threshold,
                'show_functions': args.show_functions,
                'max_functions': args.max_functions,
                'show_recommendations': args.show_recommendations
            }
            await cli.complexity_analysis(args.directory, options)

        elif args.command == 'coverage':
            options = {
                'output': args.output,
                'show_files': args.show_files,
                'max_files': args.max_files
            }
            await cli.coverage_analysis(args.directory, options)

        elif args.command == 'quality':
            options = {
                'output': args.output
            }
            await cli.quality_analysis(args.directory, options)

        elif args.command == 'history':
            options = {
                'output': args.output,
                'limit': args.limit,
                'days': args.days
            }
            await cli.analysis_history(args.directory, options)

        elif args.command == 'server':
            options = {
                'port': args.port,
                'host': args.host
            }
            await cli.start_server(options)

        elif args.command == 'smart':
            options = {
                'max_files': args.max_files,
                'output': args.output
            }
            await cli.smart_scan(args.directory, options)

        elif args.command == 'analytics':
            options = {
                'days': args.days,
                'output': args.output
            }
            await cli.scan_analytics(options)

        elif args.command == 'changes':
            options = {
                'output': args.output
            }
            await cli.change_detection(args.directory, options)

        # Show completion summary
        cli._show_completion_summary()

    except KeyboardInterrupt:
        print(f"\n🛑 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ CLI Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
