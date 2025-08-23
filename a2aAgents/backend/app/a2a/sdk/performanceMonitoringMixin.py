"""
Performance Monitoring Mixin for A2A Agents
Provides comprehensive performance monitoring capabilities for all agents
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps

# A2A Protocol Compliance: Import required monitoring components
try:
    from ..core.performanceMonitor import (
        PerformanceMetrics, 
        AlertThresholds, 
        PerformanceAlert,
        MetricsCollector,
        PrometheusMetrics,
        monitor_performance
    )
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    raise ImportError(
        "Performance monitoring components are required for A2A protocol compliance. "
        "Comprehensive monitoring is essential for network operations - no fallbacks allowed."
    )

logger = logging.getLogger(__name__)


class PerformanceMonitoringMixin:
    """
    Mixin class that adds comprehensive performance monitoring to any A2A agent
    Provides metrics collection, alerting, and performance optimization
    """
    
    def __init__(self):
        super().__init__()
        self.performance_monitoring_enabled = True
        self.metrics_collector = None
        self.prometheus_metrics = None
        self.alert_thresholds = AlertThresholds()
        self.performance_alerts = []
        
        # Performance tracking
        self.performance_metrics_history = []
        self.last_metrics_collection = None
        self.metrics_collection_interval = 30  # seconds
        
        # A2A specific metrics
        self.a2a_message_stats = {
            "sent": 0,
            "received": 0,
            "failed": 0,
            "avg_processing_time": 0.0,
            "blockchain_operations": 0,
            "ai_operations": 0
        }
        
        # Agent-specific performance data
        self.agent_performance_data = {
            "startup_time": datetime.utcnow(),
            "total_tasks_processed": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "avg_task_duration": 0.0,
            "peak_memory_usage": 0.0,
            "peak_cpu_usage": 0.0
        }
        
    async def initialize_performance_monitoring(self):
        """Initialize performance monitoring for the agent"""
        if not self.performance_monitoring_enabled:
            logger.info("Performance monitoring disabled")
            return
        
        try:
            agent_id = getattr(self, 'agent_id', 'unknown_agent')
            
            # Initialize metrics collector
            self.metrics_collector = MetricsCollector(agent_id)
            
            # Initialize Prometheus metrics if available
            try:
                self.prometheus_metrics = PrometheusMetrics(agent_id)
                logger.info(f"✅ Prometheus metrics initialized for {agent_id}")
            except Exception as e:
                logger.warning(f"Prometheus metrics initialization failed: {e}")
            
            # Start background metrics collection
            asyncio.create_task(self._metrics_collection_loop())
            
            logger.info(f"✅ Performance monitoring initialized for {agent_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize performance monitoring: {e}")
            self.performance_monitoring_enabled = False
    
    def performance_monitor(self, operation_name: str = None):
        """Decorator for monitoring function performance"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.performance_monitoring_enabled:
                    return await func(*args, **kwargs)
                
                start_time = time.time()
                success = True
                error = None
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    error = e
                    raise
                finally:
                    duration = time.time() - start_time
                    await self._record_operation_metrics(
                        operation_name or func.__name__,
                        duration,
                        success,
                        error
                    )
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.performance_monitoring_enabled:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                success = True
                error = None
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    error = e
                    raise
                finally:
                    duration = time.time() - start_time
                    asyncio.create_task(self._record_operation_metrics(
                        operation_name or func.__name__,
                        duration,
                        success,
                        error
                    ))
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    async def _record_operation_metrics(self, operation_name: str, duration: float, success: bool, error: Exception = None):
        """Record metrics for an operation"""
        try:
            if self.metrics_collector:
                self.metrics_collector.record_request(duration, success)
            
            # Update A2A specific metrics
            if 'a2a' in operation_name.lower() or 'blockchain' in operation_name.lower():
                self.a2a_message_stats["blockchain_operations"] += 1
            
            if 'ai' in operation_name.lower() or 'grok' in operation_name.lower():
                self.a2a_message_stats["ai_operations"] += 1
            
            # Update agent performance data
            self.agent_performance_data["total_tasks_processed"] += 1
            if success:
                self.agent_performance_data["successful_tasks"] += 1
            else:
                self.agent_performance_data["failed_tasks"] += 1
            
            # Update average task duration
            total_tasks = self.agent_performance_data["total_tasks_processed"]
            current_avg = self.agent_performance_data["avg_task_duration"]
            self.agent_performance_data["avg_task_duration"] = (
                (current_avg * (total_tasks - 1) + duration) / total_tasks
            )
            
            # Record with Prometheus if available
            if self.prometheus_metrics:
                self.prometheus_metrics.record_request(operation_name, duration, success)
            
            logger.debug(f"📊 Recorded metrics for {operation_name}: {duration:.3f}s, success: {success}")
            
        except Exception as e:
            logger.warning(f"Failed to record operation metrics: {e}")
    
    async def record_a2a_message(self, message_type: str, direction: str, success: bool, processing_time: float = 0.0):
        """Record A2A message statistics"""
        try:
            if direction == "sent":
                self.a2a_message_stats["sent"] += 1
            elif direction == "received":
                self.a2a_message_stats["received"] += 1
            
            if not success:
                self.a2a_message_stats["failed"] += 1
            
            # Update average processing time
            if processing_time > 0:
                current_avg = self.a2a_message_stats["avg_processing_time"]
                total_messages = self.a2a_message_stats["sent"] + self.a2a_message_stats["received"]
                self.a2a_message_stats["avg_processing_time"] = (
                    (current_avg * (total_messages - 1) + processing_time) / total_messages
                )
            
            logger.debug(f"📡 A2A Message: {message_type} {direction}, success: {success}")
            
        except Exception as e:
            logger.warning(f"Failed to record A2A message metrics: {e}")
    
    async def _metrics_collection_loop(self):
        """Background loop for metrics collection"""
        logger.info("🔄 Starting performance metrics collection loop")
        
        while self.performance_monitoring_enabled:
            try:
                await asyncio.sleep(self.metrics_collection_interval)
                await self._collect_and_store_metrics()
                
            except asyncio.CancelledError:
                logger.info("Metrics collection loop cancelled")
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    async def _collect_and_store_metrics(self):
        """Collect current metrics and store them"""
        try:
            if not self.metrics_collector:
                return
            
            # Get system metrics
            system_metrics = self.metrics_collector.get_system_metrics()
            performance_stats = self.metrics_collector.get_performance_stats()
            
            # Update peak values
            cpu_usage = system_metrics.get("cpu_usage", 0.0)
            memory_usage = system_metrics.get("memory_usage", 0.0)
            
            if cpu_usage > self.agent_performance_data["peak_cpu_usage"]:
                self.agent_performance_data["peak_cpu_usage"] = cpu_usage
            
            if memory_usage > self.agent_performance_data["peak_memory_usage"]:
                self.agent_performance_data["peak_memory_usage"] = memory_usage
            
            # Create comprehensive metrics
            metrics = PerformanceMetrics(
                timestamp=datetime.utcnow().isoformat(),
                agent_id=getattr(self, 'agent_id', 'unknown'),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                request_count=self.metrics_collector.request_count,
                response_time_avg=performance_stats.get("avg_response_time", 0.0),
                response_time_p95=performance_stats.get("p95_response_time", 0.0),
                error_rate=performance_stats.get("error_rate", 0.0),
                throughput=performance_stats.get("throughput", 0.0),
                active_connections=len(getattr(self, 'active_connections', [])),
                queue_size=len(getattr(self, 'message_queue', [])),
                cache_hit_rate=performance_stats.get("cache_hit_rate", 0.0)
            )
            
            # Store metrics
            self.performance_metrics_history.append(metrics)
            self.last_metrics_collection = datetime.utcnow()
            
            # Keep only last 1000 metrics (to prevent memory growth)
            if len(self.performance_metrics_history) > 1000:
                self.performance_metrics_history = self.performance_metrics_history[-1000:]
            
            # Check for alerts
            await self._check_performance_alerts(metrics)
            
            logger.debug(f"📊 Collected metrics: CPU {cpu_usage:.1f}%, Memory {memory_usage:.1f}%")
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
    
    async def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check if any performance thresholds are exceeded"""
        try:
            alerts = []
            
            # CPU usage alert
            if metrics.cpu_usage > self.alert_thresholds.cpu_threshold:
                alert = PerformanceAlert(
                    alert_type="high_cpu",
                    severity="high" if metrics.cpu_usage > 90 else "medium",
                    message=f"High CPU usage: {metrics.cpu_usage:.1f}%",
                    value=metrics.cpu_usage,
                    threshold=self.alert_thresholds.cpu_threshold,
                    agent_id=metrics.agent_id
                )
                alerts.append(alert)
            
            # Memory usage alert
            if metrics.memory_usage > self.alert_thresholds.memory_threshold:
                alert = PerformanceAlert(
                    alert_type="high_memory",
                    severity="high" if metrics.memory_usage > 95 else "medium",
                    message=f"High memory usage: {metrics.memory_usage:.1f}%",
                    value=metrics.memory_usage,
                    threshold=self.alert_thresholds.memory_threshold,
                    agent_id=metrics.agent_id
                )
                alerts.append(alert)
            
            # Response time alert
            if metrics.response_time_p95 > self.alert_thresholds.response_time_threshold:
                alert = PerformanceAlert(
                    alert_type="slow_response",
                    severity="medium",
                    message=f"Slow response time: {metrics.response_time_p95:.1f}ms",
                    value=metrics.response_time_p95,
                    threshold=self.alert_thresholds.response_time_threshold,
                    agent_id=metrics.agent_id
                )
                alerts.append(alert)
            
            # Error rate alert
            if metrics.error_rate > self.alert_thresholds.error_rate_threshold:
                alert = PerformanceAlert(
                    alert_type="high_error_rate",
                    severity="high",
                    message=f"High error rate: {metrics.error_rate:.1%}",
                    value=metrics.error_rate,
                    threshold=self.alert_thresholds.error_rate_threshold,
                    agent_id=metrics.agent_id
                )
                alerts.append(alert)
            
            # Store new alerts
            for alert in alerts:
                self.performance_alerts.append(alert)
                logger.warning(f"🚨 Performance Alert: {alert.message}")
            
            # Keep only recent alerts (last 100)
            if len(self.performance_alerts) > 100:
                self.performance_alerts = self.performance_alerts[-100:]
            
        except Exception as e:
            logger.error(f"Failed to check performance alerts: {e}")
    
    def get_current_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        try:
            if not self.metrics_collector:
                return {"error": "Performance monitoring not initialized"}
            
            system_metrics = self.metrics_collector.get_system_metrics()
            performance_stats = self.metrics_collector.get_performance_stats()
            
            return {
                "agent_id": getattr(self, 'agent_id', 'unknown'),
                "timestamp": datetime.utcnow().isoformat(),
                "system_metrics": system_metrics,
                "performance_stats": performance_stats,
                "a2a_message_stats": self.a2a_message_stats.copy(),
                "agent_performance_data": self.agent_performance_data.copy(),
                "recent_alerts": [alert.to_dict() for alert in self.performance_alerts[-5:]],
                "monitoring_enabled": self.performance_monitoring_enabled
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics"""
        try:
            if not self.performance_metrics_history:
                return {"status": "No metrics available"}
            
            recent_metrics = self.performance_metrics_history[-10:]  # Last 10 collections
            
            # Calculate averages
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            avg_response_time = sum(m.response_time_avg for m in recent_metrics) / len(recent_metrics)
            
            # Performance health assessment
            health_score = self._calculate_health_score(recent_metrics)
            
            uptime = (datetime.utcnow() - self.agent_performance_data["startup_time"]).total_seconds()
            
            return {
                "agent_id": getattr(self, 'agent_id', 'unknown'),
                "health_score": health_score,
                "uptime_seconds": uptime,
                "average_metrics": {
                    "cpu_usage": round(avg_cpu, 2),
                    "memory_usage": round(avg_memory, 2),
                    "response_time_ms": round(avg_response_time, 2)
                },
                "peak_metrics": {
                    "cpu_usage": self.agent_performance_data["peak_cpu_usage"],
                    "memory_usage": self.agent_performance_data["peak_memory_usage"]
                },
                "task_statistics": {
                    "total_processed": self.agent_performance_data["total_tasks_processed"],
                    "success_rate": self._calculate_success_rate(),
                    "avg_duration_ms": round(self.agent_performance_data["avg_task_duration"] * 1000, 2)
                },
                "a2a_statistics": self.a2a_message_stats.copy(),
                "active_alerts": len([a for a in self.performance_alerts if self._is_recent_alert(a)]),
                "last_updated": self.last_metrics_collection.isoformat() if self.last_metrics_collection else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}
    
    def _calculate_health_score(self, recent_metrics: List[PerformanceMetrics]) -> float:
        """Calculate overall health score (0-100)"""
        try:
            if not recent_metrics:
                return 0.0
            
            # Base score
            score = 100.0
            
            # Deduct points for high CPU usage
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            if avg_cpu > 80:
                score -= (avg_cpu - 80) * 2
            
            # Deduct points for high memory usage
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            if avg_memory > 80:
                score -= (avg_memory - 80) * 2
            
            # Deduct points for high error rate
            avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
            if avg_error_rate > 0.01:  # 1%
                score -= avg_error_rate * 1000  # Convert to percentage and deduct
            
            # Deduct points for slow response times
            avg_response_time = sum(m.response_time_avg for m in recent_metrics) / len(recent_metrics)
            if avg_response_time > 1000:  # 1 second
                score -= (avg_response_time - 1000) / 100
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.warning(f"Failed to calculate health score: {e}")
            return 50.0  # Default to neutral score
    
    def _calculate_success_rate(self) -> float:
        """Calculate task success rate"""
        total = self.agent_performance_data["total_tasks_processed"]
        if total == 0:
            return 1.0
        
        successful = self.agent_performance_data["successful_tasks"]
        return successful / total
    
    def _is_recent_alert(self, alert: PerformanceAlert) -> bool:
        """Check if alert is recent (within last hour)"""
        try:
            alert_time = datetime.fromisoformat(alert.timestamp)
            return (datetime.utcnow() - alert_time) < timedelta(hours=1)
        except:
            return False
    
    async def shutdown_performance_monitoring(self):
        """Gracefully shutdown performance monitoring"""
        try:
            self.performance_monitoring_enabled = False
            logger.info("🔄 Performance monitoring shutdown complete")
            
        except Exception as e:
            logger.warning(f"Performance monitoring shutdown error: {e}")


# Convenience decorator for monitoring A2A operations
def monitor_a2a_operation(operation_type: str = "a2a_operation"):
    """Decorator specifically for monitoring A2A operations"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = await func(self, *args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                if hasattr(self, 'record_a2a_message'):
                    processing_time = time.time() - start_time
                    await self.record_a2a_message(
                        operation_type, 
                        "processed", 
                        success, 
                        processing_time
                    )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else func
    return decorator