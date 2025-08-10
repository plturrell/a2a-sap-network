#!/usr/bin/env python3
"""
Security Testing Demo Script
Demonstrates the automated security testing capabilities
"""

import asyncio
import sys
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.securityTesting import get_test_runner, TestType
from app.core.loggingConfig import init_logging, get_logger, LogCategory

# Initialize logging
init_logging(level="INFO", format_type="simple", console=True)
logger = get_logger(__name__, LogCategory.SECURITY)


async def run_security_demo():
    """Run security testing demonstration"""
    logger.info("🔒 Starting Security Testing Demo")
    
    try:
        # Get test runner
        runner = get_test_runner()
        
        # Show available tests
        logger.info(f"\n📋 Available Security Tests: {len(runner.tests)}")
        for test_id, test in list(runner.tests.items())[:5]:  # Show first 5
            logger.info(f"  - {test.test_id}: {test.name} ({test.test_type.value})")
        
        # Run static analysis tests
        logger.info("\n🔍 Running Static Analysis Tests...")
        report = await runner.run_all_tests(
            test_types=[TestType.STATIC_ANALYSIS],
            tags=["code", "secrets"]
        )
        
        # Display results
        logger.info("\n📊 Test Results Summary:")
        logger.info(f"  Total Tests: {report['summary']['total_tests']}")
        logger.info(f"  Passed: {report['summary']['passed_tests']}")
        logger.info(f"  Failed: {report['summary']['failed_tests']}")
        logger.info(f"  Security Score: {report['summary']['overall_score']}/100")
        logger.info(f"  Security Posture: {report['summary']['security_posture']}")
        
        # Show vulnerabilities if any
        if report['vulnerabilities']['total'] > 0:
            logger.warning(f"\n⚠️  Found {report['vulnerabilities']['total']} vulnerabilities:")
            for severity, count in report['vulnerabilities']['by_severity'].items():
                if count > 0:
                    logger.warning(f"    {severity.upper()}: {count}")
        
        # Show recommendations
        if report['recommendations']:
            logger.info("\n💡 Recommendations:")
            for rec in report['recommendations'][:3]:  # Show first 3
                logger.info(f"  {rec}")
        
        # Show individual test results
        logger.info("\n📝 Individual Test Results:")
        for result in report['test_results'][:5]:  # Show first 5
            status_emoji = "✅" if result['status'] == "passed" else "❌"
            logger.info(f"  {status_emoji} {result['test_name']}: {result['status']} (Score: {result['score']})")
        
        logger.info("\n✅ Security Testing Demo Complete!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_security_demo())