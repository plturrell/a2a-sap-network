"""
Health Check Endpoints for A2A Core Components
Provides standardized health check, capability, and status endpoints
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import psutil
import platform

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Individual health check result"""
    name: str
    status: HealthStatus
    response_time_ms: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_connections: int
    uptime_seconds: float
    load_average: List[float]


class HealthCheckRegistry:
    """
    Registry for health checks and capability endpoints
    """

    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.capabilities: List[str] = []
        self.custom_endpoints: Dict[str, Callable] = {}
        self.system_info = self._get_system_info()

        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.response_times: List[float] = []

        logger.info("Health check registry initialized")

    def _get_system_info(self) -> Dict[str, Any]:
        """Get static system information"""
        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
        }

    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function"""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")

    def register_capability(self, capability: str):
        """Register a capability"""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
            logger.info(f"Registered capability: {capability}")

    def register_custom_endpoint(self, name: str, handler_func: Callable):
        """Register a custom endpoint handler"""
        self.custom_endpoints[name] = handler_func
        logger.info(f"Registered custom endpoint: {name}")

    async def run_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks"""
        results = {}

        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()

                if asyncio.iscoroutinefunction(check_func):
                    check_result = await check_func()
                else:
                    check_result = check_func()

                response_time = (time.time() - start_time) * 1000

                if isinstance(check_result, HealthCheck):
                    check_result.response_time_ms = response_time
                    results[name] = check_result
                elif isinstance(check_result, bool):
                    results[name] = HealthCheck(
                        name=name,
                        status=HealthStatus.HEALTHY if check_result else HealthStatus.UNHEALTHY,
                        response_time_ms=response_time,
                        message="OK" if check_result else "Check failed"
                    )
                elif isinstance(check_result, dict):
                    results[name] = HealthCheck(
                        name=name,
                        status=HealthStatus(check_result.get('status', 'healthy')),
                        response_time_ms=response_time,
                        message=check_result.get('message', ''),
                        details=check_result.get('details', {})
                    )
                else:
                    results[name] = HealthCheck(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        response_time_ms=response_time,
                        message=f"Invalid check result: {type(check_result)}"
                    )

            except Exception as e:
                results[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    response_time_ms=0,
                    message=f"Health check error: {str(e)}"
                )
                logger.error(f"Health check failed for {name}: {e}")

        return results

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        try:
            return SystemMetrics(
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=psutil.virtual_memory().percent,
                disk_percent=psutil.disk_usage('/').percent,
                network_connections=len(psutil.net_connections()),
                uptime_seconds=time.time() - psutil.boot_time(),
                load_average=list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
            )
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return SystemMetrics(0, 0, 0, 0, 0, [0, 0, 0])

    def get_overall_health_status(self, health_checks: Dict[str, HealthCheck]) -> HealthStatus:
        """Determine overall health status from individual checks"""
        if not health_checks:
            return HealthStatus.HEALTHY

        statuses = [check.status for check in health_checks.values()]

        # If any check is critical, overall is critical
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL

        # If any check is unhealthy, overall is unhealthy
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY

        # If any check is degraded, overall is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED

        # All checks are healthy
        return HealthStatus.HEALTHY


# Global registry instance
_health_registry: Optional[HealthCheckRegistry] = None


def get_health_registry() -> HealthCheckRegistry:
    """Get or create the global health check registry"""
    global _health_registry
    if _health_registry is None:
        _health_registry = HealthCheckRegistry()
    return _health_registry


def create_health_router() -> APIRouter:
    """Create FastAPI router with health check endpoints"""
    router = APIRouter()
    registry = get_health_registry()

    @router.get("/health")
    async def health_check():
        """Main health check endpoint"""
        start_time = time.time()

        try:
            registry.request_count += 1

            # Run all health checks
            health_checks = await registry.run_health_checks()

            # Get system metrics
            system_metrics = registry.get_system_metrics()

            # Determine overall status
            overall_status = registry.get_overall_health_status(health_checks)

            # Calculate response time
            response_time = (time.time() - start_time) * 1000
            registry.response_times.append(response_time)
            if len(registry.response_times) > 100:
                registry.response_times = registry.response_times[-100:]

            # Build response
            response_data = {
                "status": overall_status.value,
                "timestamp": datetime.utcnow().isoformat(),
                "response_time_ms": response_time,
                "system": {
                    "cpu_percent": system_metrics.cpu_percent,
                    "memory_percent": system_metrics.memory_percent,
                    "disk_percent": system_metrics.disk_percent,
                    "network_connections": system_metrics.network_connections,
                    "uptime_seconds": system_metrics.uptime_seconds,
                    "load_average": system_metrics.load_average,
                },
                "checks": {
                    name: {
                        "status": check.status.value,
                        "response_time_ms": check.response_time_ms,
                        "message": check.message,
                        "details": check.details,
                        "last_updated": check.last_updated.isoformat()
                    }
                    for name, check in health_checks.items()
                }
            }

            # Set appropriate HTTP status code
            status_code = {
                HealthStatus.HEALTHY: 200,
                HealthStatus.DEGRADED: 200,  # Still serving traffic
                HealthStatus.UNHEALTHY: 503,
                HealthStatus.CRITICAL: 503
            }.get(overall_status, 503)

            return JSONResponse(content=response_data, status_code=status_code)

        except Exception as e:
            registry.error_count += 1
            logger.error(f"Health check endpoint error: {e}")

            return JSONResponse(
                content={
                    "status": "critical",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e)
                },
                status_code=503
            )

    @router.get("/ready")
    async def readiness_check():
        """Readiness check endpoint (can serve traffic)"""
        try:
            health_checks = await registry.run_health_checks()
            overall_status = registry.get_overall_health_status(health_checks)

            # Ready if healthy or degraded
            is_ready = overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

            response_data = {
                "ready": is_ready,
                "status": overall_status.value,
                "timestamp": datetime.utcnow().isoformat()
            }

            return JSONResponse(
                content=response_data,
                status_code=200 if is_ready else 503
            )

        except Exception as e:
            return JSONResponse(
                content={
                    "ready": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                },
                status_code=503
            )

    @router.get("/capabilities")
    async def capabilities_endpoint():
        """Capabilities endpoint"""
        return JSONResponse(content={
            "capabilities": registry.capabilities,
            "timestamp": datetime.utcnow().isoformat(),
            "count": len(registry.capabilities)
        })

    @router.get("/metrics")
    async def metrics_endpoint():
        """Metrics endpoint for monitoring"""
        system_metrics = registry.get_system_metrics()

        avg_response_time = (
            sum(registry.response_times) / len(registry.response_times)
            if registry.response_times else 0
        )

        return JSONResponse(content={
            "system_metrics": {
                "cpu_percent": system_metrics.cpu_percent,
                "memory_percent": system_metrics.memory_percent,
                "disk_percent": system_metrics.disk_percent,
                "network_connections": system_metrics.network_connections,
                "uptime_seconds": system_metrics.uptime_seconds,
                "load_average": system_metrics.load_average,
            },
            "endpoint_metrics": {
                "total_requests": registry.request_count,
                "total_errors": registry.error_count,
                "error_rate": registry.error_count / max(registry.request_count, 1),
                "avg_response_time_ms": avg_response_time,
            },
            "health_checks": {
                "registered_checks": len(registry.health_checks),
                "check_names": list(registry.health_checks.keys())
            },
            "timestamp": datetime.utcnow().isoformat()
        })

    @router.post("/messages")
    async def message_endpoint(request: Request):
        """Message processing endpoint"""
        try:
            # Get message data
            message_data = await request.json()

            # Validate required fields
            required_fields = ['id', 'agent_id', 'payload']
            for field in required_fields:
                if field not in message_data:
                    raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

            # Process message (this would integrate with actual message processing)
            response_data = {
                "message_id": message_data['id'],
                "status": "accepted",
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time_ms": 0.1  # Mock processing time
            }

            # Return appropriate status based on priority
            priority = request.headers.get('X-Message-Priority', 'normal')
            status_code = 200 if priority == 'normal' else 202  # 202 for high priority (async processing)

            return JSONResponse(content=response_data, status_code=status_code)

        except Exception as e:
            logger.error(f"Message processing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/replicate")
    async def replication_endpoint(request: Request):
        """Data replication endpoint"""
        try:
            replication_data = await request.json()

            # Validate replication data
            required_fields = ['key', 'data', 'version', 'source_node']
            for field in required_fields:
                if field not in replication_data:
                    raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

            # Process replication (mock implementation)
            response_data = {
                "key": replication_data['key'],
                "version": replication_data['version'],
                "status": "replicated",
                "timestamp": datetime.utcnow().isoformat()
            }

            return JSONResponse(content=response_data, status_code=201)

        except Exception as e:
            logger.error(f"Replication error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/data/{key}")
    async def data_fetch_endpoint(key: str, request: Request):
        """Data fetch endpoint"""
        try:
            consistency_level = request.headers.get('X-Consistency-Level', 'eventual')

            # Mock data fetch (would integrate with actual data store)
            response_data = {
                "key": key,
                "value": {"mock": "data"},
                "version": {
                    "version": 1,
                    "timestamp": datetime.utcnow().isoformat(),
                    "node_id": "current_node",
                    "vector_clock": {"current_node": 1},
                    "checksum": "abc123"
                },
                "consistency_level": consistency_level,
                "replicas": ["node1", "node2"],
                "timestamp": datetime.utcnow().isoformat()
            }

            return JSONResponse(content=response_data, status_code=200)

        except Exception as e:
            logger.error(f"Data fetch error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/version/{key}")
    async def version_endpoint(key: str):
        """Version check endpoint"""
        try:
            # Mock version check (would integrate with actual data store)
            response_data = {
                "key": key,
                "version": 1,  # Mock version
                "timestamp": datetime.utcnow().isoformat()
            }

            return JSONResponse(content=response_data, status_code=200)

        except Exception as e:
            logger.error(f"Version check error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/routing/update")
    async def routing_update_endpoint(request: Request):
        """Routing table update endpoint"""
        try:
            routing_data = await request.json()

            # Process routing update (mock implementation)
            response_data = {
                "status": "updated",
                "timestamp": datetime.utcnow().isoformat(),
                "routes_updated": routing_data.get("routes", {})
            }

            return JSONResponse(content=response_data, status_code=200)

        except Exception as e:
            logger.error(f"Routing update error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/recovery/execute")
    async def recovery_execute_endpoint(request: Request):
        """Recovery step execution endpoint"""
        try:
            recovery_data = await request.json()

            step = recovery_data.get('step')
            parameters = recovery_data.get('parameters', {})

            # Mock recovery step execution
            response_data = {
                "step": step,
                "status": "executed",
                "timestamp": datetime.utcnow().isoformat(),
                "execution_time_ms": 100,  # Mock execution time
                "result": "success"
            }

            return JSONResponse(content=response_data, status_code=200)

        except Exception as e:
            logger.error(f"Recovery execution error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return router


# Default health checks
def register_default_health_checks():
    """Register default health checks"""
    registry = get_health_registry()

    def database_check() -> HealthCheck:
        """Database connectivity check"""
        try:
            # Mock database check
            return HealthCheck(
                name="database",
                status=HealthStatus.HEALTHY,
                response_time_ms=5.0,
                message="Database connection OK"
            )
        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.CRITICAL,
                response_time_ms=0,
                message=f"Database connection failed: {str(e)}"
            )

    def memory_check() -> HealthCheck:
        """Memory usage check"""
        try:
            memory_percent = psutil.virtual_memory().percent

            if memory_percent < 80:
                status = HealthStatus.HEALTHY
                message = f"Memory usage OK ({memory_percent:.1f}%)"
            elif memory_percent < 90:
                status = HealthStatus.DEGRADED
                message = f"Memory usage high ({memory_percent:.1f}%)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage critical ({memory_percent:.1f}%)"

            return HealthCheck(
                name="memory",
                status=status,
                response_time_ms=1.0,
                message=message,
                details={"memory_percent": memory_percent}
            )
        except Exception as e:
            return HealthCheck(
                name="memory",
                status=HealthStatus.CRITICAL,
                response_time_ms=0,
                message=f"Memory check failed: {str(e)}"
            )

    def disk_check() -> HealthCheck:
        """Disk usage check"""
        try:
            disk_percent = psutil.disk_usage('/').percent

            if disk_percent < 85:
                status = HealthStatus.HEALTHY
                message = f"Disk usage OK ({disk_percent:.1f}%)"
            elif disk_percent < 95:
                status = HealthStatus.DEGRADED
                message = f"Disk usage high ({disk_percent:.1f}%)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Disk usage critical ({disk_percent:.1f}%)"

            return HealthCheck(
                name="disk",
                status=status,
                response_time_ms=1.0,
                message=message,
                details={"disk_percent": disk_percent}
            )
        except Exception as e:
            return HealthCheck(
                name="disk",
                status=HealthStatus.CRITICAL,
                response_time_ms=0,
                message=f"Disk check failed: {str(e)}"
            )

    # Register default checks
    registry.register_health_check("database", database_check)
    registry.register_health_check("memory", memory_check)
    registry.register_health_check("disk", disk_check)

    # Register default capabilities
    registry.register_capability("message_processing")
    registry.register_capability("data_replication")
    registry.register_capability("health_monitoring")
    registry.register_capability("routing_updates")
    registry.register_capability("recovery_execution")