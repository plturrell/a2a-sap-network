#!/usr/bin/env python3
"""
GleanAgent JavaScript Analysis and Fix Script
Performs comprehensive JavaScript scanning and automated fixes
"""

import os
import sys

# Set required environment variables BEFORE any imports
if not os.getenv("A2A_SERVICE_URL"):
    os.environ["A2A_SERVICE_URL"] = "http://localhost:3000"
if not os.getenv("A2A_SERVICE_HOST"):
    os.environ["A2A_SERVICE_HOST"] = "localhost"
if not os.getenv("A2A_BASE_URL"):
    os.environ["A2A_BASE_URL"] = "http://localhost:3000"

import asyncio
import json
from datetime import datetime
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Import GleanAgent
try:
    from app.a2a.agents.gleanAgent.gleanAgentSdk import GleanAgent
    print("✓ Successfully imported GleanAgent")
except ImportError as e:
    print(f"✗ Failed to import GleanAgent: {e}")
    print("Make sure you're running from the backend directory and all dependencies are installed")
    sys.exit(1)

async def run_javascript_analysis():
    """Run comprehensive JavaScript analysis and fixes"""
    print("\n" + "="*80)
    print("GleanAgent JavaScript Analysis and Fix Operation")
    print("="*80 + "\n")
    
    # Initialize GleanAgent
    try:
        agent = GleanAgent()
        print("✓ GleanAgent initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize GleanAgent: {e}")
        return
    
    # Set analysis directory
    analysis_dir = backend_dir
    print(f"\nAnalysis directory: {analysis_dir}")
    
    # Step 1: Run initial JavaScript analysis
    print("\n" + "-"*60)
    print("STEP 1: Running initial JavaScript analysis...")
    print("-"*60)
    
    try:
        initial_analysis = await agent._perform_lint_analysis(
            directory=str(analysis_dir),
            file_patterns=["*.js", "*.jsx"]
        )
        
        print(f"\n✓ Initial analysis complete:")
        print(f"  • Files analyzed: {initial_analysis.get('files_analyzed', 0)}")
        print(f"  • Total issues found: {initial_analysis.get('total_issues', 0)}")
        print(f"  • Critical issues: {initial_analysis.get('critical_issues', 0)}")
        print(f"  • Analysis duration: {initial_analysis.get('duration', 0):.2f} seconds")
        
        # Show issues by severity
        if initial_analysis.get('issues_by_severity'):
            print("\n  Issues by severity:")
            for severity, count in initial_analysis['issues_by_severity'].items():
                print(f"    - {severity}: {count}")
        
        # Show issues by type
        if initial_analysis.get('issues_by_type'):
            print("\n  Issues by type:")
            for issue_type, count in initial_analysis['issues_by_type'].items():
                print(f"    - {issue_type}: {count}")
        
        # Save initial analysis report
        initial_report_path = backend_dir / "javascript_initial_analysis.json"
        with open(initial_report_path, 'w') as f:
            json.dump(initial_analysis, f, indent=2)
        print(f"\n✓ Initial analysis report saved to: {initial_report_path}")
        
    except Exception as e:
        print(f"\n✗ Error during initial analysis: {e}")
        return
    
    # Step 2: Apply automated fixes
    print("\n" + "-"*60)
    print("STEP 2: Applying automated JavaScript fixes...")
    print("-"*60)
    
    try:
        fix_results = await agent.fix_javascript_issues(
            directory=str(analysis_dir),
            auto_fix=True,
            dry_run=False
        )
        
        print(f"\n✓ Fix operation complete:")
        print(f"  • Files processed: {fix_results.get('files_processed', 0)}")
        print(f"  • Issues fixed: {fix_results.get('issues_fixed', 0)}")
        print(f"  • Files modified: {len(fix_results.get('files_modified', []))}")
        print(f"  • Fix duration: {fix_results.get('duration', 0):.2f} seconds")
        
        # Show fix summary
        if fix_results.get('fix_summary'):
            print("\n  Fix summary:")
            for fix_type, count in fix_results['fix_summary'].items():
                if count > 0:
                    print(f"    - {fix_type}: {count}")
        
        # Save fix report
        fix_report_path = backend_dir / "javascript_fix_report.json"
        with open(fix_report_path, 'w') as f:
            json.dump(fix_results, f, indent=2)
        print(f"\n✓ Fix report saved to: {fix_report_path}")
        
    except Exception as e:
        print(f"\n✗ Error during fix operation: {e}")
        return
    
    # Step 3: Run final analysis to show remaining issues
    print("\n" + "-"*60)
    print("STEP 3: Running final analysis to show remaining issues...")
    print("-"*60)
    
    try:
        final_analysis = await agent._perform_lint_analysis(
            directory=str(analysis_dir),
            file_patterns=["*.js", "*.jsx"]
        )
        
        print(f"\n✓ Final analysis complete:")
        print(f"  • Files analyzed: {final_analysis.get('files_analyzed', 0)}")
        print(f"  • Total remaining issues: {final_analysis.get('total_issues', 0)}")
        print(f"  • Critical issues remaining: {final_analysis.get('critical_issues', 0)}")
        
        # Show remaining issues by severity
        if final_analysis.get('issues_by_severity'):
            print("\n  Remaining issues by severity:")
            for severity, count in final_analysis['issues_by_severity'].items():
                print(f"    - {severity}: {count}")
        
        # Save final analysis report
        final_report_path = backend_dir / "javascript_final_analysis.json"
        with open(final_report_path, 'w') as f:
            json.dump(final_analysis, f, indent=2)
        print(f"\n✓ Final analysis report saved to: {final_report_path}")
        
    except Exception as e:
        print(f"\n✗ Error during final analysis: {e}")
        return
    
    # Step 4: Generate comprehensive report
    print("\n" + "-"*60)
    print("STEP 4: Generating comprehensive report...")
    print("-"*60)
    
    try:
        # Calculate improvements
        issues_fixed = initial_analysis.get('total_issues', 0) - final_analysis.get('total_issues', 0)
        fix_percentage = (issues_fixed / initial_analysis.get('total_issues', 1)) * 100
        
        comprehensive_report = {
            "timestamp": datetime.now().isoformat(),
            "analysis_directory": str(analysis_dir),
            "summary": {
                "files_analyzed": initial_analysis.get('files_analyzed', 0),
                "initial_issues": initial_analysis.get('total_issues', 0),
                "issues_fixed": issues_fixed,
                "remaining_issues": final_analysis.get('total_issues', 0),
                "fix_percentage": round(fix_percentage, 2),
                "files_modified": len(fix_results.get('files_modified', [])),
                "total_duration": sum([
                    initial_analysis.get('duration', 0),
                    fix_results.get('duration', 0),
                    final_analysis.get('duration', 0)
                ])
            },
            "initial_analysis": initial_analysis,
            "fix_results": fix_results,
            "final_analysis": final_analysis,
            "examples_of_fixes": []
        }
        
        # Add examples of fixes (first 5 issues that were fixed)
        initial_issues = initial_analysis.get('issues', [])
        final_issues = final_analysis.get('issues', [])
        
        # Create a set of remaining issue identifiers
        remaining_issue_ids = set()
        for issue in final_issues:
            if isinstance(issue, dict):
                issue_id = f"{issue.get('file_path', '')}:{issue.get('line', '')}:{issue.get('code', '')}"
                remaining_issue_ids.add(issue_id)
        
        # Find fixed issues
        for issue in initial_issues[:20]:  # Check first 20 issues
            if isinstance(issue, dict):
                issue_id = f"{issue.get('file_path', '')}:{issue.get('line', '')}:{issue.get('code', '')}"
                if issue_id not in remaining_issue_ids:
                    comprehensive_report["examples_of_fixes"].append({
                        "file": issue.get('file_path', ''),
                        "line": issue.get('line', ''),
                        "type": issue.get('issue_type', ''),
                        "severity": issue.get('severity', ''),
                        "message": issue.get('message', ''),
                        "rule": issue.get('code', '')
                    })
                    if len(comprehensive_report["examples_of_fixes"]) >= 5:
                        break
        
        # Save comprehensive report
        comprehensive_report_path = backend_dir / "javascript_comprehensive_report.json"
        with open(comprehensive_report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
        
        print(f"\n✓ Comprehensive report generated successfully")
        print(f"\nFINAL SUMMARY:")
        print(f"  • Initial issues: {initial_analysis.get('total_issues', 0)}")
        print(f"  • Issues fixed: {issues_fixed}")
        print(f"  • Remaining issues: {final_analysis.get('total_issues', 0)}")
        print(f"  • Fix rate: {fix_percentage:.1f}%")
        print(f"  • Total time: {comprehensive_report['summary']['total_duration']:.2f} seconds")
        
        print(f"\n✓ All reports saved:")
        print(f"  • Initial analysis: {initial_report_path}")
        print(f"  • Fix report: {fix_report_path}")
        print(f"  • Final analysis: {final_report_path}")
        print(f"  • Comprehensive report: {comprehensive_report_path}")
        
        # Show manual intervention needed
        if final_analysis.get('total_issues', 0) > 0:
            print(f"\n⚠ {final_analysis.get('total_issues', 0)} issues require manual intervention")
            print("  Review the final analysis report for details.")
        
    except Exception as e:
        print(f"\n✗ Error generating comprehensive report: {e}")
        return

if __name__ == "__main__":
    # Run the analysis
    asyncio.run(run_javascript_analysis())