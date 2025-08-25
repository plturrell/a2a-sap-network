"""
Real Agent Metrics Endpoint
Provides comprehensive performance and business metrics for A2A agents
NO FALLBACKS - Only real data from running agents
"""

import time
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import psutil
import platform
from collections import defaultdict, deque

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Comprehensive agent metrics - all real data"""

    # Performance Metrics
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    uptime_seconds: float

    # Task Performance
    total_tasks_completed: int
    active_tasks: int
    success_rate: float
    avg_response_time_ms: float

    # Business Metrics
    processed_today: int
    error_rate: float
    queue_depth: int

    # Resource Metrics
    skills_count: int
    handlers_count: int
    mcp_tools: int
    mcp_resources: int

    # Real-time Performance
    requests_per_minute: float
    peak_memory_mb: float
    network_io_mb: float

    timestamp: datetime = field(default_factory=datetime.utcnow)


class AgentMetricsCollector:
    """
    Collects real metrics from running agent operations
    NO MOCK DATA - Only actual performance data
    """

    def __init__(self, agent_instance=None):
        self.agent = agent_instance
        self.start_time = time.time()

        # Real performance tracking
        self.task_history = deque(maxlen=1000)  # Last 1000 tasks
        self.response_times = deque(maxlen=100)  # Last 100 response times
        self.error_log = deque(maxlen=50)  # Last 50 errors
        self.request_timestamps = deque(maxlen=60)  # Last 60 requests for RPM

        # Daily counters (reset at midnight)
        self.daily_processed = 0
        self.daily_errors = 0
        self.last_reset_date = datetime.utcnow().date()

        # Peak tracking
        self.peak_memory = 0.0
        self.peak_cpu = 0.0

        logger.info("Agent metrics collector initialized - real data only")

    def record_task_completion(self, task_id: str, success: bool, response_time_ms: float):
        """Record real task completion metrics"""
        self._reset_daily_if_needed()

        timestamp = datetime.utcnow()
        self.task_history.append({
            'task_id': task_id,
            'success': success,
            'response_time_ms': response_time_ms,
            'timestamp': timestamp
        })

        self.response_times.append(response_time_ms)
        self.daily_processed += 1

        if not success:
            self.daily_errors += 1
            self.error_log.append({
                'task_id': task_id,
                'timestamp': timestamp
            })

    def record_request(self):
        """Record incoming request for RPM calculation"""
        self.request_timestamps.append(time.time())

    def _reset_daily_if_needed(self):
        """Reset daily counters at midnight"""
        current_date = datetime.utcnow().date()
        if current_date != self.last_reset_date:
            self.daily_processed = 0
            self.daily_errors = 0
            self.last_reset_date = current_date

    def get_current_metrics(self) -> AgentMetrics:
        """Get real-time metrics - NO FALLBACKS"""
        try:
            # Real system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Track peaks
            self.peak_memory = max(self.peak_memory, memory.used / 1024 / 1024)  # MB
            self.peak_cpu = max(self.peak_cpu, cpu_percent)

            # Real task metrics
            total_tasks = len(self.task_history)
            successful_tasks = sum(1 for task in self.task_history if task['success'])
            success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0.0

            # Real response time metrics
            avg_response_time = (
                sum(self.response_times) / len(self.response_times)
                if self.response_times else 0.0
            )

            # Real error rate
            error_rate = (self.daily_errors / max(self.daily_processed, 1)) * 100

            # Real requests per minute
            now = time.time()
            recent_requests = [ts for ts in self.request_timestamps if now - ts <= 60]
            requests_per_minute = len(recent_requests)

            # Real agent capabilities (from actual agent instance)
            skills_count = len(self.agent.skills) if self.agent and hasattr(self.agent, 'skills') else 0
            handlers_count = len(self.agent.handlers) if self.agent and hasattr(self.agent, 'handlers') else 0
            mcp_tools = len(getattr(self.agent, 'mcp_server', {}).get('tools', [])) if self.agent else 0
            mcp_resources = len(getattr(self.agent, 'mcp_server', {}).get('resources', [])) if self.agent else 0

            # Real active tasks
            active_tasks = 0
            if self.agent and hasattr(self.agent, 'tasks'):
                from ..sdk.types import TaskStatus
                active_tasks = len([
                    t for t in self.agent.tasks.values()
                    if t.get("status") in [TaskStatus.PENDING, TaskStatus.RUNNING]
                ])

            # Real queue depth
            queue_depth = 0
            if self.agent and hasattr(self.agent, 'message_queue'):
                queue_depth = self.agent.message_queue.qsize()

            # Network I/O (real system metrics)
            net_io = psutil.net_io_counters()
            network_io_mb = (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024

            return AgentMetrics(
                # Real system performance
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                uptime_seconds=time.time() - self.start_time,

                # Real task performance
                total_tasks_completed=total_tasks,
                active_tasks=active_tasks,
                success_rate=success_rate,
                avg_response_time_ms=avg_response_time,

                # Real business metrics
                processed_today=self.daily_processed,
                error_rate=error_rate,
                queue_depth=queue_depth,

                # Real capabilities
                skills_count=skills_count,
                handlers_count=handlers_count,
                mcp_tools=mcp_tools,
                mcp_resources=mcp_resources,

                # Real-time performance
                requests_per_minute=float(requests_per_minute),
                peak_memory_mb=self.peak_memory,
                network_io_mb=network_io_mb
            )

        except Exception as e:
            logger.error(f"Error collecting agent metrics: {e}")
            raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")


# Global metrics collector instance
_metrics_collector: Optional[AgentMetricsCollector] = None


def initialize_metrics_collector(agent_instance=None) -> AgentMetricsCollector:
    """Initialize metrics collector with agent instance"""
    global _metrics_collector
    _metrics_collector = AgentMetricsCollector(agent_instance)
    return _metrics_collector


def get_metrics_collector() -> AgentMetricsCollector:
    """Get metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = AgentMetricsCollector()
    return _metrics_collector


def create_metrics_router() -> APIRouter:
    """Create FastAPI router for agent metrics - REAL DATA ONLY"""
    router = APIRouter()
    collector = get_metrics_collector()

    @router.get("/metrics")
    async def get_agent_metrics():
        """
        Comprehensive agent metrics endpoint
        Returns ONLY real data from running agent operations
        """
        try:
            # Record this request
            collector.record_request()

            # Get real metrics
            metrics = collector.get_current_metrics()

            return JSONResponse(content={
                # Core performance metrics (required)
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "disk_percent": metrics.disk_percent,
                "uptime_seconds": metrics.uptime_seconds,

                # Task performance metrics (required)
                "total_tasks_completed": metrics.total_tasks_completed,
                "active_tasks": metrics.active_tasks,
                "success_rate": metrics.success_rate,
                "avg_response_time_ms": metrics.avg_response_time_ms,

                # Business metrics (required)
                "processed_today": metrics.processed_today,
                "error_rate": metrics.error_rate,
                "queue_depth": metrics.queue_depth,

                # Capability metrics (required)
                "skills_count": metrics.skills_count,
                "handlers_count": metrics.handlers_count,
                "mcp_tools": metrics.mcp_tools,
                "mcp_resources": metrics.mcp_resources,

                # Real-time performance
                "requests_per_minute": metrics.requests_per_minute,
                "peak_memory_mb": metrics.peak_memory_mb,
                "network_io_mb": metrics.network_io_mb,

                # Metadata
                "timestamp": metrics.timestamp.isoformat(),
                "metrics_source": "real_agent_data",
                "collection_method": "live_monitoring"
            })

        except Exception as e:
            logger.error(f"Metrics endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/metrics/detailed")
    async def get_detailed_metrics():
        """Detailed metrics with historical data"""
        try:
            collector.record_request()
            metrics = collector.get_current_metrics()

            # Recent task history (real data)
            recent_tasks = list(collector.task_history)[-20:]  # Last 20 tasks

            # Response time distribution (real data)
            response_times = list(collector.response_times)
            response_time_stats = {
                "min_ms": min(response_times) if response_times else 0,
                "max_ms": max(response_times) if response_times else 0,
                "p95_ms": sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
                "p99_ms": sorted(response_times)[int(len(response_times) * 0.99)] if response_times else 0
            }

            # Error rate over time (real data)
            recent_errors = list(collector.error_log)[-10:]  # Last 10 errors

            return JSONResponse(content={
                **await get_agent_metrics().body.decode('utf-8'),  # Base metrics

                # Historical data (real)
                "task_history": [
                    {
                        "task_id": task["task_id"],
                        "success": task["success"],
                        "response_time_ms": task["response_time_ms"],
                        "timestamp": task["timestamp"].isoformat()
                    }
                    for task in recent_tasks
                ],

                # Performance distribution (real)
                "response_time_stats": response_time_stats,

                # Error tracking (real)
                "recent_errors": [
                    {
                        "task_id": error["task_id"],
                        "timestamp": error["timestamp"].isoformat()
                    }
                    for error in recent_errors
                ],

                # Peak performance (real)
                "peak_performance": {
                    "peak_memory_mb": collector.peak_memory,
                    "peak_cpu_percent": collector.peak_cpu
                }
            })

        except Exception as e:
            logger.error(f"Detailed metrics endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return router


# Metrics decorators for automatic tracking
def track_task_metrics(func):
    """Decorator to automatically track task completion metrics"""
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        collector = get_metrics_collector()
        task_id = kwargs.get('task_id', f"task_{int(start_time)}")

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Record successful completion
            response_time = (time.time() - start_time) * 1000  # ms
            collector.record_task_completion(task_id, True, response_time)

            return result

        except Exception as e:
            # Record failed completion
            response_time = (time.time() - start_time) * 1000  # ms
            collector.record_task_completion(task_id, False, response_time)
            raise

    return wrapper
