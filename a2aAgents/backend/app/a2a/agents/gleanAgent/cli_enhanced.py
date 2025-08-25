#!/usr/bin/env python3
"""
Enhanced CLI for Glean Agent with all capabilities
Usage: python cli_enhanced.py [command] [options]
"""

import argparse
import asyncio
import json
import sys
import os
from typing import Dict, Any, Optional

from .glean_agent_enhanced import create_enhanced_glean_agent, quick_analyze, GleanAgentEnhanced


class EnhancedGleanCLI:
    """Enhanced CLI for comprehensive code analysis"""

    def __init__(self):
        self.agent: Optional[GleanAgentEnhanced] = None

    async def analyze_comprehensive(self, args) -> int:
        """Run comprehensive analysis"""
        try:
            if not os.path.exists(args.directory):
                print(f"Error: Directory '{args.directory}' not found", file=sys.stderr)
                return 1

            print(f"Starting comprehensive analysis of: {args.directory}")
            print(f"Analysis types: {', '.join(args.analysis_types)}")

            # Prepare options
            options = {
                'include_semantic': 'semantic' in args.analysis_types,
                'include_security': 'security' in args.analysis_types,
                'include_performance': 'performance' in args.analysis_types,
                'include_quality': 'quality' in args.analysis_types,
                'file_patterns': args.file_patterns
            }

            # Create agent
            config = {
                'log_level': args.log_level,
                'enable_auto_monitoring': not args.no_monitoring,
                'enable_security_alerts': not args.no_alerts,
                'max_concurrent_scans': args.max_concurrent
            }

            self.agent = create_enhanced_glean_agent(config)

            # Run analysis
            with self.agent:
                result = await self.agent.analyze_codebase_comprehensive(
                    args.directory,
                    options
                )

            # Output results
            if args.output_format == 'json':
                print(json.dumps(result, indent=2, default=str))
            else:
                self._print_readable_results(result)

            # Save to file if requested
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"\nResults saved to: {args.output_file}")

            return 0

        except KeyboardInterrupt:
            print("\nAnalysis interrupted by user", file=sys.stderr)
            return 130
        except Exception as e:
            print(f"Error during analysis: {e}", file=sys.stderr)
            return 1

    async def security_scan(self, args) -> int:
        """Run security scan only"""
        try:
            if not os.path.exists(args.directory):
                print(f"Error: Directory '{args.directory}' not found", file=sys.stderr)
                return 1

            print(f"Starting security scan of: {args.directory}")

            # Create agent
            config = {'log_level': args.log_level}
            self.agent = create_enhanced_glean_agent(config)

            # Run security analysis only
            with self.agent:
                result = await self.agent._run_security_analysis(
                    args.directory,
                    str(asyncio.current_task())
                )

            # Output results
            if args.output_format == 'json':
                print(json.dumps(result, indent=2, default=str))
            else:
                self._print_security_results(result)

            return 0

        except Exception as e:
            print(f"Error during security scan: {e}", file=sys.stderr)
            return 1

    async def health_check(self, args) -> int:
        """Run health check"""
        try:
            print("Running health check...")

            # Create agent
            config = {'log_level': args.log_level}
            self.agent = create_enhanced_glean_agent(config)

            # Run health check
            with self.agent:
                result = await self.agent.run_health_check()

            # Output results
            if args.output_format == 'json':
                print(json.dumps(result, indent=2, default=str))
            else:
                self._print_health_results(result)

            # Return appropriate exit code
            if result['overall_status'] in ['healthy', 'degraded']:
                return 0
            else:
                return 1

        except Exception as e:
            print(f"Error during health check: {e}", file=sys.stderr)
            return 1

    async def status(self, args) -> int:
        """Get real-time status"""
        try:
            print("Getting real-time status...")

            # Create agent
            config = {'log_level': args.log_level}
            self.agent = create_enhanced_glean_agent(config)

            # Get status
            with self.agent:
                result = await self.agent.get_real_time_status()

            # Output results
            if args.output_format == 'json':
                print(json.dumps(result, indent=2, default=str))
            else:
                self._print_status_results(result)

            return 0

        except Exception as e:
            print(f"Error getting status: {e}", file=sys.stderr)
            return 1

    async def quick_analysis(self, args) -> int:
        """Run quick analysis (convenience function)"""
        try:
            if not os.path.exists(args.directory):
                print(f"Error: Directory '{args.directory}' not found", file=sys.stderr)
                return 1

            print(f"Running quick analysis of: {args.directory}")

            # Use quick analyze function
            result = await quick_analyze(args.directory)

            # Output summary
            print(f"\nQuick Analysis Summary:")
            print(f"Directory: {result['directory']}")
            print(f"Duration: {result.get('duration_ms', 0):.0f}ms")
            print(f"Status: {result['status']}")

            # Show component results
            for component, data in result.get('components', {}).items():
                print(f"\n{component.title()} Analysis:")
                if data['status'] == 'completed':
                    self._print_component_summary(component, data['data'])
                else:
                    print(f"  Status: {data['status']}")
                    if 'error' in data:
                        print(f"  Error: {data['error']}")

            return 0

        except Exception as e:
            print(f"Error during quick analysis: {e}", file=sys.stderr)
            return 1

    def _print_readable_results(self, result: Dict[str, Any]):
        """Print results in human-readable format"""
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE ANALYSIS RESULTS")
        print(f"{'='*60}")

        print(f"Directory: {result['directory']}")
        print(f"Correlation ID: {result['correlation_id']}")
        print(f"Duration: {result.get('duration_ms', 0):.0f}ms")
        print(f"Status: {result['status']}")
        print(f"Started: {result['started_at']}")
        print(f"Completed: {result.get('completed_at', 'N/A')}")

        # Show component results
        for component, data in result.get('components', {}).items():
            print(f"\n{'-'*40}")
            print(f"{component.upper()} ANALYSIS")
            print(f"{'-'*40}")

            if data['status'] == 'completed':
                self._print_component_summary(component, data['data'])
            else:
                print(f"Status: {data['status']}")
                if 'error' in data:
                    print(f"Error: {data['error']}")

    def _print_component_summary(self, component: str, data: Dict[str, Any]):
        """Print component-specific summary"""
        if component == 'semantic':
            print(f"  Files analyzed: {data.get('files_analyzed', 0)}")
            print(f"  Successful: {data.get('successful_analyses', 0)}")
            print(f"  Failed: {data.get('failed_analyses', 0)}")
            print(f"  Symbols found: {data.get('symbols_found', 0)}")

        elif component == 'security':
            print(f"  Files scanned: {data.get('files_scanned', 0)}")
            print(f"  Total vulnerabilities: {data.get('total_vulnerabilities', 0)}")
            print(f"  Critical: {data.get('critical_vulnerabilities', 0)}")
            print(f"  High: {data.get('high_vulnerabilities', 0)}")

        elif component == 'performance':
            print(f"  Total operations: {data.get('total_operations', 0)}")
            print(f"  Success rate: {data.get('overall_success_rate', 0):.1f}%")
            print(f"  Avg response time: {data.get('average_response_time', 0):.1f}ms")
            slow_ops = data.get('slow_operations', [])
            if slow_ops:
                print(f"  Slow operations: {len(slow_ops)}")

        elif component == 'quality':
            print(f"  Files analyzed: {data.get('files_analyzed', 0)}")
            print(f"  Total issues: {data.get('total_issues', 0)}")
            print(f"  Errors: {data.get('errors', 0)}")
            print(f"  Warnings: {data.get('warnings', 0)}")
            print(f"  Quality score: {data.get('quality_score', 0):.1f}/100")

    def _print_security_results(self, result: Dict[str, Any]):
        """Print security scan results"""
        print(f"\n{'='*50}")
        print(f"SECURITY SCAN RESULTS")
        print(f"{'='*50}")

        print(f"Files scanned: {result.get('files_scanned', 0)}")
        print(f"Total vulnerabilities: {result.get('total_vulnerabilities', 0)}")
        print(f"Critical vulnerabilities: {result.get('critical_vulnerabilities', 0)}")
        print(f"High vulnerabilities: {result.get('high_vulnerabilities', 0)}")

        # Show sample vulnerabilities
        scan_results = result.get('scan_results', [])
        if scan_results:
            print(f"\nSample vulnerabilities:")
            for scan_result in scan_results[:3]:  # Show first 3 files
                vulnerabilities = scan_result.get('vulnerabilities', [])
                if vulnerabilities:
                    print(f"\n  File: {scan_result['file_path']}")
                    for vuln in vulnerabilities[:2]:  # Show first 2 vulns per file
                        print(f"    - {vuln['severity'].upper()}: {vuln['title']}")
                        print(f"      Line {vuln['line_number']}: {vuln['code_snippet'][:50]}...")

    def _print_health_results(self, result: Dict[str, Any]):
        """Print health check results"""
        print(f"\n{'='*40}")
        print(f"HEALTH CHECK RESULTS")
        print(f"{'='*40}")

        status = result['overall_status']
        status_icon = "✓" if status == "healthy" else "⚠" if status == "degraded" else "✗"

        print(f"Overall Status: {status_icon} {status.upper()}")
        print(f"Timestamp: {result['timestamp']}")
        print(f"Active Operations: {result.get('active_operations_count', 0)}")

        # Show component health
        components = result.get('components', {})
        for component, health in components.items():
            component_status = health.get('status', 'unknown')
            component_icon = "✓" if component_status == "healthy" else "⚠" if component_status in ["warning", "degraded"] else "✗"
            print(f"  {component}: {component_icon} {component_status}")

    def _print_status_results(self, result: Dict[str, Any]):
        """Print real-time status results"""
        print(f"\n{'='*50}")
        print(f"REAL-TIME STATUS")
        print(f"{'='*50}")

        print(f"Timestamp: {result['timestamp']}")
        print(f"Overall Status: {result['overall_status'].upper()}")
        print(f"Active Operations: {result.get('active_operations', 0)}")

        # Show component status
        components = result.get('components', {})
        for component, status_info in components.items():
            print(f"\n{component.replace('_', ' ').title()}:")
            print(f"  Status: {status_info.get('status', 'unknown')}")

            if 'stats' in status_info:
                stats = status_info['stats']
                print(f"  Files scanned: {stats.get('total_files_scanned', 0)}")
                print(f"  Vulnerabilities found: {stats.get('vulnerabilities_found', 0)}")

        # Show active alerts
        active_alerts = result.get('active_alerts', [])
        if active_alerts:
            print(f"\nActive Alerts ({len(active_alerts)}):")
            for alert in active_alerts[:5]:  # Show first 5 alerts
                severity_icon = "🔴" if alert['severity'] == 'critical' else "🟡" if alert['severity'] == 'high' else "🟢"
                print(f"  {severity_icon} {alert['type']}: {alert['message']}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description='Enhanced Glean Agent - Comprehensive Code Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick analysis
  python cli_enhanced.py quick /path/to/code

  # Comprehensive analysis
  python cli_enhanced.py analyze /path/to/code --analysis-types semantic security quality

  # Security scan only
  python cli_enhanced.py security /path/to/code

  # Health check
  python cli_enhanced.py health

  # Real-time status
  python cli_enhanced.py status
        """
    )

    # Global options
    parser.add_argument('--log-level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO',
                        help='Set logging level')
    parser.add_argument('--output-format',
                        choices=['human', 'json'],
                        default='human',
                        help='Output format')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Quick analysis command
    quick_parser = subparsers.add_parser('quick', help='Run quick analysis')
    quick_parser.add_argument('directory', help='Directory to analyze')

    # Comprehensive analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Run comprehensive analysis')
    analyze_parser.add_argument('directory', help='Directory to analyze')
    analyze_parser.add_argument('--analysis-types',
                               choices=['semantic', 'security', 'performance', 'quality'],
                               nargs='+',
                               default=['semantic', 'security', 'quality'],
                               help='Types of analysis to run')
    analyze_parser.add_argument('--file-patterns',
                               nargs='+',
                               default=['*.py', '*.js', '*.ts'],
                               help='File patterns to analyze')
    analyze_parser.add_argument('--output-file',
                               help='Save results to file')
    analyze_parser.add_argument('--no-monitoring',
                               action='store_true',
                               help='Disable monitoring')
    analyze_parser.add_argument('--no-alerts',
                               action='store_true',
                               help='Disable security alerts')
    analyze_parser.add_argument('--max-concurrent',
                               type=int,
                               default=5,
                               help='Maximum concurrent scans')

    # Security scan command
    security_parser = subparsers.add_parser('security', help='Run security scan')
    security_parser.add_argument('directory', help='Directory to scan')

    # Health check command
    subparsers.add_parser('health', help='Run health check')

    # Status command
    subparsers.add_parser('status', help='Get real-time status')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Create CLI instance
    cli = EnhancedGleanCLI()

    # Run appropriate command
    try:
        if args.command == 'quick':
            return asyncio.run(cli.quick_analysis(args))
        elif args.command == 'analyze':
            return asyncio.run(cli.analyze_comprehensive(args))
        elif args.command == 'security':
            return asyncio.run(cli.security_scan(args))
        elif args.command == 'health':
            return asyncio.run(cli.health_check(args))
        elif args.command == 'status':
            return asyncio.run(cli.status(args))
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return 1

    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 130


if __name__ == '__main__':
    sys.exit(main())
