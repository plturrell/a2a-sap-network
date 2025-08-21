#!/usr/bin/env python3
"""
Code Quality CLI - Comprehensive code quality analysis and systematic fixing
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Add the tests directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from a2a_mcp.tools.code_quality_scanner import CodeQualityScanner, CodeQualityDatabase, IssueSeverity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeQualityCLI:
    """Command-line interface for code quality management."""
    
    def __init__(self):
        self.db = CodeQualityDatabase()
    
    async def scan_directory(self, args: argparse.Namespace):
        """Scan directory for code quality issues."""
        directory = Path(args.directory).resolve()
        
        if not directory.exists():
            print(f"❌ Directory not found: {directory}")
            return
        
        print(f"🔍 Scanning {directory}")
        print(f"Tools: {args.tools}")
        print("=" * 60)
        
        scanner = CodeQualityScanner(directory.parent)
        
        try:
            result = await scanner.scan_directory(
                directory=directory,
                tools=args.tools.split(',') if args.tools else None,
                file_extensions=args.extensions.split(',') if args.extensions else None
            )
            
            # Display results
            print(f"\n📊 Scan Results (ID: {result.scan_id})")
            print(f"Files scanned: {result.total_files}")
            print(f"Issues found: {result.issues_found}")
            print(f"Duration: {result.scan_duration:.2f}s")
            
            # Issues by severity
            print(f"\n⚠️ Issues by Severity:")
            for severity, count in sorted(result.issues_by_severity.items()):
                print(f"  {severity.upper()}: {count}")
            
            # Issues by type  
            print(f"\n📋 Issues by Type:")
            for issue_type, count in sorted(result.issues_by_type.items()):
                print(f"  {issue_type.replace('_', ' ').title()}: {count}")
            
            # Top issues preview
            if args.show_preview and result.issues:
                print(f"\n🔍 Preview of Issues:")
                for i, issue in enumerate(result.issues[:10], 1):
                    print(f"{i:2}. [{issue.severity.value.upper()}] {Path(issue.file_path).name}:{issue.line}")
                    print(f"     {issue.tool}: {issue.message}")
                    print()
                
                if result.issues_found > 10:
                    print(f"... and {result.issues_found - 10} more issues")
            
        except Exception as e:
            print(f"❌ Scan failed: {e}")
            logger.error(f"Scan error: {e}")
    
    def list_scans(self, args: argparse.Namespace):
        """List recent scans."""
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, directory, tools_used, issues_found, started_at, scan_duration
                FROM scan_results
                ORDER BY started_at DESC
                LIMIT ?
            """, (args.limit,))
            
            scans = cursor.fetchall()
        
        if not scans:
            print("No scans found.")
            return
        
        print(f"📋 Recent Code Quality Scans (last {args.limit})")
        print("=" * 80)
        
        for scan in scans:
            scan_id, directory, tools, issues, started_at, duration = scan
            tools_list = json.loads(tools) if tools else []
            
            print(f"ID: {scan_id[:8]}...")
            print(f"Directory: {directory}")
            print(f"Tools: {', '.join(tools_list)}")
            print(f"Issues: {issues}")
            print(f"Started: {started_at}")
            print(f"Duration: {duration:.2f}s")
            print("-" * 40)
    
    def show_summary(self, args: argparse.Namespace):
        """Show code quality summary."""
        summary = self.db.get_scan_summary(args.days)
        
        print(f"📊 Code Quality Summary ({args.days} days)")
        print("=" * 50)
        
        print(f"Total Scans: {summary['total_scans']}")
        print(f"Total Issues: {summary['total_issues']}")
        print(f"Avg Scan Duration: {summary['avg_scan_duration']:.2f}s")
        
        if summary['issues_by_severity']:
            print(f"\n⚠️ Issues by Severity:")
            total = sum(summary['issues_by_severity'].values())
            for severity, count in sorted(summary['issues_by_severity'].items(), 
                                        key=lambda x: ['critical', 'high', 'medium', 'low', 'info'].index(x[0])):
                percentage = (count / total) * 100 if total > 0 else 0
                print(f"  {severity.upper()}: {count} ({percentage:.1f}%)")
        
        if summary['issues_by_type']:
            print(f"\n📋 Issues by Type:")
            for issue_type, count in sorted(summary['issues_by_type'].items()):
                print(f"  {issue_type.replace('_', ' ').title()}: {count}")
    
    def show_top_issues(self, args: argparse.Namespace):
        """Show top issues by frequency/severity."""
        issues = self.db.get_top_issues(args.limit, args.severity)
        
        print(f"🔥 Top Issues (limit: {args.limit})")
        if args.severity:
            print(f"Severity filter: {args.severity.upper()}")
        print("=" * 80)
        
        for i, issue in enumerate(issues, 1):
            print(f"{i:2}. [{issue['severity'].upper()}] {Path(issue['file_path']).name}")
            print(f"    Type: {issue['issue_type'].replace('_', ' ').title()}")
            print(f"    Rule: {issue['rule'] or 'N/A'}")
            print(f"    Message: {issue['message']}")
            print(f"    Occurrences: {issue['count']}")
            print("-" * 40)
    
    def generate_ignore_flags(self, args: argparse.Namespace):
        """Generate ignore flags for systematic fixing."""
        issues = self.db.get_top_issues(limit=100, severity=args.min_severity)
        
        print(f"🛠️ Suggested Ignore Flags for Systematic Fixing")
        print(f"Minimum Severity: {args.min_severity.upper()}")
        print("=" * 60)
        
        # Group by tool
        tool_rules = {}
        for issue in issues:
            if issue['count'] >= args.min_occurrences:
                tool = self._extract_tool_from_issue(issue)
                if tool not in tool_rules:
                    tool_rules[tool] = set()
                if issue['rule']:
                    tool_rules[tool].add(issue['rule'])
        
        # Generate ignore flags
        for tool, rules in tool_rules.items():
            if tool == 'pylint':
                ignore_list = ','.join(sorted(rules))
                print(f"\n## Pylint")
                print(f"pylint --disable={ignore_list} {args.target or '/path/to/code'}")
                
            elif tool == 'flake8':
                ignore_list = ','.join(sorted(rules))
                print(f"\n## Flake8")  
                print(f"flake8 --ignore={ignore_list} {args.target or '/path/to/code'}")
                
            elif tool == 'eslint':
                rules_config = {rule: 'off' for rule in sorted(rules)}
                print(f"\n## ESLint (.eslintrc.json)")
                print(json.dumps({"rules": rules_config}, indent=2))
        
        print(f"\n💡 Tips:")
        print(f"1. Start by ignoring the most frequent, low-severity issues")
        print(f"2. Gradually fix issues and remove ignore flags")
        print(f"3. Focus on critical and high severity issues first")
        print(f"4. Use automated fixing tools where possible")
    
    def _extract_tool_from_issue(self, issue: Dict) -> str:
        """Extract tool name from issue data."""
        # This would be stored in the database if we had the full issue data
        # For now, infer from rule format
        rule = issue.get('rule', '')
        
        if rule and rule.startswith(('E', 'W', 'F', 'C')) and rule[1:].isdigit():
            return 'flake8'
        elif '/' in rule or rule in ['no-unused-vars', 'no-undef', 'semi']:
            return 'eslint'
        else:
            return 'pylint'
    
    def _get_connection(self):
        """Get database connection (helper method)."""
        import sqlite3
        return sqlite3.connect(self.db.db_path)

# Add the missing method to CodeQualityDatabase
def _get_connection_method(self):
    """Get database connection."""
    import sqlite3
    return sqlite3.connect(self.db_path)

# Monkey patch the method
CodeQualityDatabase._get_connection = _get_connection_method

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="A2A Code Quality CLI - Comprehensive analysis and systematic fixing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan a directory with default tools
  python code_quality_cli.py scan /Users/apple/projects/a2a/a2aAgents/backend
  
  # Scan with specific tools
  python code_quality_cli.py scan /path/to/code --tools pylint,flake8,eslint
  
  # Show recent scans
  python code_quality_cli.py list --limit 10
  
  # Show code quality summary
  python code_quality_cli.py summary --days 30
  
  # Show top issues
  python code_quality_cli.py top-issues --limit 20 --severity high
  
  # Generate ignore flags for systematic fixing
  python code_quality_cli.py ignore-flags --min-severity medium --min-occurrences 5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan directory for code quality issues')
    scan_parser.add_argument('directory', help='Directory to scan')
    scan_parser.add_argument('--tools', default='pylint,flake8,eslint', 
                           help='Comma-separated list of tools to run')
    scan_parser.add_argument('--extensions', default='.py,.js,.ts,.jsx,.tsx',
                           help='Comma-separated list of file extensions')
    scan_parser.add_argument('--show-preview', action='store_true',
                           help='Show preview of found issues')
    
    # List scans command
    list_parser = subparsers.add_parser('list', help='List recent scans')
    list_parser.add_argument('--limit', type=int, default=10, help='Number of scans to show')
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Show code quality summary')
    summary_parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    
    # Top issues command
    top_parser = subparsers.add_parser('top-issues', help='Show top issues by frequency/severity')
    top_parser.add_argument('--limit', type=int, default=20, help='Number of issues to show')
    top_parser.add_argument('--severity', choices=['critical', 'high', 'medium', 'low', 'info'],
                          help='Filter by severity')
    
    # Ignore flags command
    ignore_parser = subparsers.add_parser('ignore-flags', help='Generate ignore flags for systematic fixing')
    ignore_parser.add_argument('--min-severity', default='medium', 
                             choices=['critical', 'high', 'medium', 'low', 'info'],
                             help='Minimum severity level to include')
    ignore_parser.add_argument('--min-occurrences', type=int, default=3,
                             help='Minimum number of occurrences to include rule')
    ignore_parser.add_argument('--target', help='Target directory for generated commands')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = CodeQualityCLI()
    
    try:
        if args.command == 'scan':
            asyncio.run(cli.scan_directory(args))
        elif args.command == 'list':
            cli.list_scans(args)
        elif args.command == 'summary':
            cli.show_summary(args)
        elif args.command == 'top-issues':
            cli.show_top_issues(args)
        elif args.command == 'ignore-flags':
            cli.generate_ignore_flags(args)
    
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"CLI error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()