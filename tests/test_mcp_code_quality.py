#!/usr/bin/env python3
"""
Test MCP Code Quality Integration
Demonstrates the integrated code quality tools in the MCP server
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the tests directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from a2a_mcp.tools.code_quality_scanner import CodeQualityScanner, CodeQualityDatabase, IssueSeverity
from a2a_mcp.server.enhanced_mcp_server import (
    handle_scan_code_quality,
    handle_analyze_code_quality,
    handle_review_code_fixes,
    handle_code_quality_trends
)

async def demonstrate_mcp_integration():
    """Demonstrate the MCP code quality integration."""
    
    print("🚀 A2A MCP Code Quality Integration Demo")
    print("=" * 60)
    
    # 1. Scan code quality through MCP tool
    print("\n1️⃣ Scanning code quality via MCP tool...")
    
    scan_args = {
        "directory": "a2aAgents/backend/app/a2a/core",
        "tools": ["pylint"],
        "preview": True
    }
    
    scan_result = await handle_scan_code_quality(**scan_args)
    print(f"✅ Scan completed: {scan_result['issues_found']} issues found")
    print(f"   Scan ID: {scan_result['scan_id']}")
    
    # 2. Analyze code quality
    print("\n2️⃣ Analyzing code quality...")
    
    analysis_args = {
        "scan_id": scan_result['scan_id'],
        "focus": "critical_issues"
    }
    
    analysis_result = await handle_analyze_code_quality(**analysis_args)
    print(f"✅ Analysis complete:")
    print(f"   Critical issues: {analysis_result['critical_issues']}")
    print(f"   Top recommendations:")
    for i, rec in enumerate(analysis_result['recommendations'][:3], 1):
        print(f"   {i}. {rec}")
    
    # 3. Review code fixes (safe approach)
    print("\n3️⃣ Reviewing potential fixes...")
    
    review_args = {
        "scan_id": scan_result['scan_id'],
        "issue_types": ["unused_imports"],
        "dry_run": True
    }
    
    review_result = await handle_review_code_fixes(**review_args)
    print(f"✅ Review complete:")
    print(f"   Fixable issues: {review_result['fixable_count']}")
    print(f"   Review required: {review_result['requires_review']}")
    
    # 4. Track quality trends
    print("\n4️⃣ Tracking quality trends...")
    
    trends_args = {
        "days": 7,
        "metric": "severity_distribution"
    }
    
    trends_result = await handle_code_quality_trends(**trends_args)
    print(f"✅ Trends analysis:")
    print(f"   Total scans: {trends_result['total_scans']}")
    print(f"   Trend: {trends_result['trend']}")
    
    print("\n" + "=" * 60)
    print("✅ MCP Code Quality Integration successfully demonstrated!")
    print("\n📝 Key Features:")
    print("   • Safe, review-based approach (no automatic bulk fixes)")
    print("   • Database tracking for historical analysis")
    print("   • Progressive improvement strategy")
    print("   • Integration with existing MCP toolset")

if __name__ == "__main__":
    asyncio.run(demonstrate_mcp_integration())