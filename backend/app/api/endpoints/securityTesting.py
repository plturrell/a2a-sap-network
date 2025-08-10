"""
Security Testing API Endpoints
Provides administrative endpoints for running automated security tests
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import logging

from ...core.securityTesting import (
    get_test_runner,
    run_security_tests,
    TestType,
    TestStatus
)
from ...core.securityMonitoring import report_security_event, EventType, ThreatLevel
from ..deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/security/tests/run")
async def run_security_test_suite(
    background_tasks: BackgroundTasks,
    test_types: Optional[List[TestType]] = None,
    tags: Optional[List[str]] = None,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Run comprehensive security test suite
    
    - **test_types**: Optional list of test types to run (e.g., STATIC_ANALYSIS, DEPENDENCY_SCAN)
    - **tags**: Optional list of tags to filter tests (e.g., "sql", "injection", "authentication")
    
    Returns test execution summary and schedules full test suite in background.
    """
    try:
        logger.info(f"Security test suite initiated by user {current_user.id}")
        
        # Report security testing activity
        await report_security_event(
            event_type=EventType.SYSTEM_INTRUSION,
            threat_level=ThreatLevel.INFO,
            description=f"Security test suite initiated by admin",
            user_id=current_user.id,
            details={
                "test_types": test_types,
                "tags": tags
            }
        )
        
        # Schedule tests in background
        background_tasks.add_task(
            run_security_tests,
            test_types=test_types,
            tags=tags
        )
        
        return {
            "status": "scheduled",
            "message": "Security test suite has been scheduled for execution",
            "test_types": test_types or "all",
            "tags": tags or "all"
        }
        
    except Exception as e:
        logger.error(f"Failed to initiate security test suite: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/security/tests/status")
async def get_test_status(
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current status of security testing system
    
    Returns information about:
    - Number of tests available
    - Currently running tests
    - Recent test results summary
    """
    try:
        runner = get_test_runner()
        
        # Get test counts by type
        test_counts = {}
        for test in runner.tests.values():
            test_type = test.test_type.value
            test_counts[test_type] = test_counts.get(test_type, 0) + 1
        
        # Get recent results summary
        recent_results = {
            "total": len(runner.test_results),
            "passed": sum(1 for r in runner.test_results if r.status == TestStatus.PASSED),
            "failed": sum(1 for r in runner.test_results if r.status == TestStatus.FAILED),
            "error": sum(1 for r in runner.test_results if r.status == TestStatus.ERROR),
            "running": len(runner.running_tests)
        }
        
        return {
            "total_tests": len(runner.tests),
            "test_counts_by_type": test_counts,
            "currently_running": list(runner.running_tests),
            "recent_results": recent_results
        }
        
    except Exception as e:
        logger.error(f"Failed to get test status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/security/tests/results")
async def get_test_results(
    limit: int = 10,
    test_type: Optional[TestType] = None,
    status: Optional[TestStatus] = None,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get recent security test results
    
    - **limit**: Maximum number of results to return (default: 10, max: 100)
    - **test_type**: Filter by test type
    - **status**: Filter by test status (PASSED, FAILED, ERROR)
    """
    try:
        runner = get_test_runner()
        
        # Limit max results
        limit = min(limit, 100)
        
        # Filter results
        results = runner.test_results
        
        if test_type:
            results = [r for r in results if runner.tests.get(r.test_id, {}).test_type == test_type]
        
        if status:
            results = [r for r in results if r.status == status]
        
        # Get most recent results
        results = sorted(results, key=lambda r: r.started_at, reverse=True)[:limit]
        
        # Format results
        formatted_results = []
        for result in results:
            test = runner.tests.get(result.test_id)
            formatted_results.append({
                "test_id": result.test_id,
                "test_name": result.test_name,
                "test_type": test.test_type.value if test else "unknown",
                "status": result.status.value,
                "started_at": result.started_at.isoformat(),
                "completed_at": result.completed_at.isoformat() if result.completed_at else None,
                "duration_seconds": result.duration_seconds,
                "vulnerabilities_found": len(result.vulnerabilities_found),
                "score": result.score,
                "error_message": result.error_message
            })
        
        return {
            "count": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        logger.error(f"Failed to get test results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/security/tests/report/{test_id}")
async def get_test_report(
    test_id: str,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get detailed report for a specific test execution
    
    Returns comprehensive test results including:
    - Test metadata
    - Vulnerabilities found
    - Evidence collected
    - Remediation recommendations
    """
    try:
        runner = get_test_runner()
        
        # Find test result
        result = None
        for r in runner.test_results:
            if r.test_id == test_id:
                result = r
                break
        
        if not result:
            raise HTTPException(status_code=404, detail="Test result not found")
        
        # Get test definition
        test = runner.tests.get(result.test_id)
        
        return {
            "test_id": result.test_id,
            "test_name": result.test_name,
            "test_type": test.test_type.value if test else "unknown",
            "description": test.description if test else "N/A",
            "severity": test.severity.value if test else "unknown",
            "status": result.status.value,
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            "duration_seconds": result.duration_seconds,
            "score": result.score,
            "vulnerabilities": result.vulnerabilities_found,
            "remediation_steps": result.remediation_steps,
            "evidence": result.evidence,
            "error_message": result.error_message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get test report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/security/tests/catalog")
async def get_test_catalog(
    test_type: Optional[TestType] = None,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get catalog of available security tests
    
    - **test_type**: Filter by test type
    
    Returns list of all available tests with their metadata.
    """
    try:
        runner = get_test_runner()
        
        tests = []
        for test in runner.tests.values():
            if test_type and test.test_type != test_type:
                continue
                
            tests.append({
                "test_id": test.test_id,
                "name": test.name,
                "test_type": test.test_type.value,
                "description": test.description,
                "severity": test.severity.value,
                "automated": test.automated,
                "requires_auth": test.requires_auth,
                "timeout_seconds": test.timeout_seconds,
                "tags": test.tags
            })
        
        # Sort by test type and severity
        tests.sort(key=lambda t: (t["test_type"], t["severity"], t["name"]))
        
        return {
            "count": len(tests),
            "tests": tests
        }
        
    except Exception as e:
        logger.error(f"Failed to get test catalog: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/security/tests/schedule")
async def schedule_periodic_tests(
    interval_hours: int = 24,
    test_types: Optional[List[TestType]] = None,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Schedule periodic security tests (placeholder for future implementation)
    
    - **interval_hours**: How often to run tests (default: 24 hours)
    - **test_types**: Which test types to include in periodic runs
    """
    # This would integrate with a task scheduler like Celery or APScheduler
    return {
        "status": "not_implemented",
        "message": "Periodic test scheduling will be implemented in a future release",
        "requested_interval": interval_hours,
        "requested_types": test_types
    }


# Export router
__all__ = ["router"]