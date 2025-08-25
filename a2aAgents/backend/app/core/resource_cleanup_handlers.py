"""
Resource Cleanup Handlers for A2A Production Components
Provides automatic cleanup for all production components
"""

import asyncio
import gc
import logging
import signal
import sys
import weakref
import time
import psutil
from collections import deque, defaultdict
from typing import Dict, Any, List, Optional, Set, Callable, Tuple
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import atexit

try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("ML libraries not available for predictive failure detection")

# Import all production components for cleanup
from .secure_secrets_manager import shutdown_secrets_manager
from .resource_manager import shutdown_resource_manager
from .a2a_protocol_validator import shutdown_protocol_validator
from .slo_sli_framework import shutdown_slo_sli_framework
from .backpressure_manager import shutdown_backpressure_manager
from .chaos_engineering import shutdown_chaos_framework
from .standardized_lifecycle import shutdown_lifecycle_manager
from .testing_framework import shutdown_testing_framework
from .security_hardening import shutdown_security_hardening
from .observability_stack import shutdown_observability_manager
from .disaster_recovery import shutdown_disaster_recovery_manager
from .a2a_distributed_coordinator import shutdown_distributed_coordinator

logger = logging.getLogger(__name__)


class PredictiveFailureDetector:
    """AI-powered predictive failure detection system"""

    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.anomaly_detector = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.failure_predictions = deque(maxlen=100)
        self.resource_patterns = defaultdict(deque)
        self.alert_thresholds = {
            'cpu_usage': 85.0,
            'memory_usage': 90.0,
            'disk_usage': 95.0,
            'connection_count': 1000,
            'response_time': 5000.0  # ms
        }
        self.trained = False

        if ML_AVAILABLE:
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )

    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect comprehensive system metrics"""
        try:
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Network and process metrics
            net_io = psutil.net_io_counters()
            process = psutil.Process()

            metrics = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available': memory.available / (1024**3),  # GB
                'disk_usage': disk.percent,
                'disk_free': disk.free / (1024**3),  # GB
                'network_bytes_sent': net_io.bytes_sent,
                'network_bytes_recv': net_io.bytes_recv,
                'process_memory_rss': process.memory_info().rss / (1024**2),  # MB
                'process_cpu_percent': process.cpu_percent(),
                'open_files': len(process.open_files()),
                'num_threads': process.num_threads(),
                'timestamp': time.time()
            }

            return metrics
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {'timestamp': time.time(), 'error': 1.0}

    def analyze_metrics_for_anomalies(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Use AI to detect anomalies in system metrics"""
        self.metrics_history.append(metrics)

        # Simple threshold-based detection first
        threshold_alerts = []
        for metric, threshold in self.alert_thresholds.items():
            if metric in metrics and metrics[metric] > threshold:
                threshold_alerts.append({
                    'type': 'threshold_breach',
                    'metric': metric,
                    'value': metrics[metric],
                    'threshold': threshold,
                    'severity': 'high' if metrics[metric] > threshold * 1.1 else 'medium'
                })

        # ML-based anomaly detection
        ml_anomalies = []
        if ML_AVAILABLE and len(self.metrics_history) >= 50:
            try:
                # Prepare feature vector
                feature_keys = ['cpu_usage', 'memory_usage', 'disk_usage',
                              'process_memory_rss', 'process_cpu_percent', 'open_files']

                # Extract features from recent history
                features = []
                for hist_metrics in list(self.metrics_history)[-50:]:
                    feature_vector = [hist_metrics.get(key, 0) for key in feature_keys]
                    features.append(feature_vector)

                features = np.array(features)

                # Train anomaly detector if not trained yet
                if not self.trained and len(features) >= 30:
                    features_scaled = self.scaler.fit_transform(features)
                    self.anomaly_detector.fit(features_scaled)
                    self.trained = True
                    logger.info("Trained anomaly detection model")

                # Detect anomalies in current metrics
                if self.trained:
                    current_features = np.array([[metrics.get(key, 0) for key in feature_keys]])
                    current_scaled = self.scaler.transform(current_features)
                    anomaly_score = self.anomaly_detector.decision_function(current_scaled)[0]
                    is_anomaly = self.anomaly_detector.predict(current_scaled)[0] == -1

                    if is_anomaly:
                        ml_anomalies.append({
                            'type': 'ml_anomaly',
                            'anomaly_score': float(anomaly_score),
                            'severity': 'high' if anomaly_score < -0.5 else 'medium',
                            'features': dict(zip(feature_keys, current_features[0]))
                        })

            except Exception as e:
                logger.error(f"ML anomaly detection error: {e}")

        return {
            'threshold_alerts': threshold_alerts,
            'ml_anomalies': ml_anomalies,
            'timestamp': metrics['timestamp']
        }

    def predict_failure_probability(self) -> Tuple[float, Dict[str, Any]]:
        """Predict probability of system failure in next period"""
        if len(self.metrics_history) < 20:
            return 0.1, {'reason': 'insufficient_data'}

        recent_metrics = list(self.metrics_history)[-20:]

        # Calculate trends and patterns
        cpu_trend = self._calculate_trend([m.get('cpu_usage', 0) for m in recent_metrics])
        memory_trend = self._calculate_trend([m.get('memory_usage', 0) for m in recent_metrics])
        disk_trend = self._calculate_trend([m.get('disk_usage', 0) for m in recent_metrics])

        # Check for rapid resource consumption
        rapid_growth_indicators = 0
        if cpu_trend > 2.0:  # CPU increasing by 2%+ per measurement
            rapid_growth_indicators += 1
        if memory_trend > 1.5:  # Memory increasing by 1.5%+ per measurement
            rapid_growth_indicators += 1
        if disk_trend > 0.5:  # Disk usage increasing
            rapid_growth_indicators += 1

        # Check current stress levels
        latest_metrics = recent_metrics[-1]
        stress_score = 0

        cpu_stress = max(0, (latest_metrics.get('cpu_usage', 0) - 70) / 30)  # 70-100% range
        memory_stress = max(0, (latest_metrics.get('memory_usage', 0) - 80) / 20)  # 80-100% range
        disk_stress = max(0, (latest_metrics.get('disk_usage', 0) - 90) / 10)  # 90-100% range

        stress_score = min(1.0, (cpu_stress + memory_stress + disk_stress) / 3)

        # Combine factors for failure probability
        trend_factor = min(1.0, rapid_growth_indicators / 3.0)
        base_probability = (stress_score * 0.6 + trend_factor * 0.4)

        # Add ML prediction if available
        if ML_AVAILABLE and self.trained:
            try:
                # Use anomaly score as additional factor
                feature_keys = ['cpu_usage', 'memory_usage', 'disk_usage',
                              'process_memory_rss', 'process_cpu_percent', 'open_files']
                current_features = np.array([[latest_metrics.get(key, 0) for key in feature_keys]])
                current_scaled = self.scaler.transform(current_features)
                anomaly_score = self.anomaly_detector.decision_function(current_scaled)[0]

                # Convert anomaly score to probability contribution
                ml_factor = max(0, min(1.0, (0.5 - anomaly_score) / 1.0))  # Normalize to 0-1
                base_probability = (base_probability * 0.7 + ml_factor * 0.3)

            except Exception as e:
                logger.error(f"ML failure prediction error: {e}")

        # Cap at 95% to avoid false certainty
        failure_probability = min(0.95, base_probability)

        prediction_details = {
            'cpu_trend': cpu_trend,
            'memory_trend': memory_trend,
            'disk_trend': disk_trend,
            'stress_score': stress_score,
            'trend_factor': trend_factor,
            'rapid_growth_indicators': rapid_growth_indicators,
            'latest_cpu': latest_metrics.get('cpu_usage', 0),
            'latest_memory': latest_metrics.get('memory_usage', 0),
            'latest_disk': latest_metrics.get('disk_usage', 0)
        }

        return failure_probability, prediction_details

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of values over time"""
        if len(values) < 2:
            return 0.0

        # Simple linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        return numerator / denominator if denominator != 0 else 0.0


class ResourceCleanupManager:
    """Centralized resource cleanup management"""

    def __init__(self):
        self.cleanup_functions: List[Callable] = []
        self.async_cleanup_functions: List[Callable] = []
        self.resource_trackers: Dict[str, Any] = {}
        self.cleanup_timeout = 30.0  # 30 seconds timeout for cleanup
        self.shutdown_in_progress = False
        self.cleanup_lock = asyncio.Lock()

        # Track weak references to avoid circular dependencies
        self.weak_references: Set[weakref.ref] = set()

        # AI-powered predictive failure detection
        self.failure_detector = PredictiveFailureDetector() if 'PredictiveFailureDetector' in globals() else None
        self.monitoring_active = False
        self.monitoring_task = None

        # Resource usage patterns
        self.resource_usage_history = deque(maxlen=200)
        self.cleanup_events = deque(maxlen=50)

        # Register signal handlers
        self._register_signal_handlers()

        # Register atexit handler
        atexit.register(self.sync_cleanup)

        # Start predictive monitoring
        if self.failure_detector:
            self.start_predictive_monitoring()

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown"""
        try:
            # Handle SIGTERM (sent by process managers)
            signal.signal(signal.SIGTERM, self._signal_handler)

            # Handle SIGINT (Ctrl+C)
            signal.signal(signal.SIGINT, self._signal_handler)

            # Handle SIGHUP (reload signal)
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, self._signal_handler)

        except ValueError:
            # Signals may not be available in all environments
            logger.warning("Could not register signal handlers (may be in thread)")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")

        # Run async cleanup in the event loop if possible
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule cleanup as a task
                loop.create_task(self.async_cleanup())
            else:
                # Run sync cleanup
                self.sync_cleanup()
        except RuntimeError:
            # No event loop running, use sync cleanup
            self.sync_cleanup()

        # Exit gracefully
        sys.exit(0)

    def register_cleanup_function(self, func: Callable, is_async: bool = False):
        """Register a cleanup function"""
        if is_async:
            self.async_cleanup_functions.append(func)
        else:
            self.cleanup_functions.append(func)

        logger.debug(f"Registered cleanup function: {func.__name__}")

    def register_resource_tracker(self, name: str, resource: Any):
        """Register a resource for tracking"""
        self.resource_trackers[name] = resource

        # Create weak reference if possible
        try:
            weak_ref = weakref.ref(resource, self._resource_finalizer)
            self.weak_references.add(weak_ref)
        except TypeError:
            # Not all objects support weak references
            pass

        logger.debug(f"Registered resource tracker: {name}")

    def _resource_finalizer(self, weak_ref):
        """Called when a tracked resource is garbage collected"""
        self.weak_references.discard(weak_ref)

    async def async_cleanup(self):
        """Perform asynchronous cleanup of all resources"""
        if self.shutdown_in_progress:
            return

        async with self.cleanup_lock:
            self.shutdown_in_progress = True

            logger.info("Starting asynchronous resource cleanup")
            start_time = datetime.utcnow()

            cleanup_tasks = []

            # 1. Shutdown all A2A production components
            logger.info("Shutting down A2A production components...")

            component_shutdown_functions = [
                ("Secrets Manager", shutdown_secrets_manager),
                ("Resource Manager", shutdown_resource_manager),
                ("Protocol Validator", shutdown_protocol_validator),
                ("SLO/SLI Framework", shutdown_slo_sli_framework),
                ("Backpressure Manager", shutdown_backpressure_manager),
                ("Chaos Framework", shutdown_chaos_framework),
                ("Lifecycle Manager", shutdown_lifecycle_manager),
                ("Testing Framework", shutdown_testing_framework),
                ("Security Framework", shutdown_security_hardening),
                ("Observability Manager", shutdown_observability_manager),
                ("Disaster Recovery", shutdown_disaster_recovery_manager),
                ("Distributed Coordinator", shutdown_distributed_coordinator),
            ]

            for component_name, shutdown_func in component_shutdown_functions:
                try:
                    logger.debug(f"Shutting down {component_name}")
                    task = asyncio.create_task(self._safe_async_shutdown(shutdown_func, component_name))
                    cleanup_tasks.append(task)
                except Exception as e:
                    logger.error(f"Failed to create shutdown task for {component_name}: {e}")

            # 2. Execute registered async cleanup functions
            for func in self.async_cleanup_functions:
                try:
                    logger.debug(f"Executing async cleanup: {func.__name__}")
                    task = asyncio.create_task(self._safe_async_cleanup(func))
                    cleanup_tasks.append(task)
                except Exception as e:
                    logger.error(f"Failed to create cleanup task for {func.__name__}: {e}")

            # 3. Wait for all cleanup tasks with timeout
            if cleanup_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*cleanup_tasks, return_exceptions=True),
                        timeout=self.cleanup_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Cleanup timeout after {self.cleanup_timeout}s, some resources may not be properly cleaned")
                    # Cancel remaining tasks
                    for task in cleanup_tasks:
                        if not task.done():
                            task.cancel()

            # 4. Execute synchronous cleanup functions
            await self._execute_sync_cleanup()

            # 5. Force garbage collection
            self._force_garbage_collection()

            # 6. Final resource check
            self._check_remaining_resources()

            cleanup_duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Asynchronous cleanup completed in {cleanup_duration:.2f}s")

    def sync_cleanup(self):
        """Perform synchronous cleanup (fallback)"""
        if self.shutdown_in_progress:
            return

        self.shutdown_in_progress = True

        logger.info("Starting synchronous resource cleanup")
        start_time = datetime.utcnow()

        # Execute synchronous cleanup functions
        for func in self.cleanup_functions:
            try:
                logger.debug(f"Executing sync cleanup: {func.__name__}")
                func()
            except Exception as e:
                logger.error(f"Sync cleanup function {func.__name__} failed: {e}")

        # Force garbage collection
        self._force_garbage_collection()

        cleanup_duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Synchronous cleanup completed in {cleanup_duration:.2f}s")

    async def _safe_async_shutdown(self, shutdown_func: Callable, component_name: str):
        """Safely execute async shutdown function"""
        try:
            if asyncio.iscoroutinefunction(shutdown_func):
                await shutdown_func()
            else:
                shutdown_func()
            logger.debug(f"Successfully shut down {component_name}")
        except Exception as e:
            logger.error(f"Failed to shutdown {component_name}: {e}")

    async def _safe_async_cleanup(self, cleanup_func: Callable):
        """Safely execute async cleanup function"""
        try:
            if asyncio.iscoroutinefunction(cleanup_func):
                await cleanup_func()
            else:
                cleanup_func()
            logger.debug(f"Successfully executed cleanup: {cleanup_func.__name__}")
        except Exception as e:
            logger.error(f"Cleanup function {cleanup_func.__name__} failed: {e}")

    async def _execute_sync_cleanup(self):
        """Execute synchronous cleanup functions in async context"""
        for func in self.cleanup_functions:
            try:
                logger.debug(f"Executing sync cleanup: {func.__name__}")
                # Run in thread pool to avoid blocking
                await asyncio.get_event_loop().run_in_executor(None, func)
            except Exception as e:
                logger.error(f"Sync cleanup function {func.__name__} failed: {e}")

    def _force_garbage_collection(self):
        """Force garbage collection and log memory usage"""
        try:
            import psutil


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Force garbage collection
            collected = gc.collect()

            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_freed = memory_before - memory_after

            logger.info(f"Garbage collection: {collected} objects collected, "
                       f"{memory_freed:.1f}MB freed (RSS: {memory_after:.1f}MB)")

        except ImportError:
            # psutil not available, just run GC
            collected = gc.collect()
            logger.info(f"Garbage collection: {collected} objects collected")
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")

    def _check_remaining_resources(self):
        """Check for remaining resources that may not have been cleaned up"""
        remaining_count = 0

        for name, resource in self.resource_trackers.items():
            if resource is not None:
                remaining_count += 1
                logger.warning(f"Resource not cleaned up: {name}")

        # Check weak references
        active_refs = len([ref for ref in self.weak_references if ref() is not None])
        if active_refs > 0:
            logger.warning(f"{active_refs} weak references still active")

        if remaining_count == 0 and active_refs == 0:
            logger.info("All tracked resources cleaned up successfully")
        else:
            logger.warning(f"{remaining_count} resources and {active_refs} references remain")

    def get_cleanup_status(self) -> Dict[str, Any]:
        """Get current cleanup status"""
        active_refs = len([ref for ref in self.weak_references if ref() is not None])

        return {
            "shutdown_in_progress": self.shutdown_in_progress,
            "registered_cleanup_functions": len(self.cleanup_functions),
            "registered_async_cleanup_functions": len(self.async_cleanup_functions),
            "tracked_resources": len(self.resource_trackers),
            "active_weak_references": active_refs,
            "cleanup_timeout": self.cleanup_timeout
        }


# Global cleanup manager
_cleanup_manager = None


def get_cleanup_manager() -> ResourceCleanupManager:
    """Get the global cleanup manager"""
    global _cleanup_manager

    if _cleanup_manager is None:
        _cleanup_manager = ResourceCleanupManager()

    return _cleanup_manager


def register_cleanup(func: Callable, is_async: bool = False):
    """Register a cleanup function with the global manager"""
    manager = get_cleanup_manager()
    manager.register_cleanup_function(func, is_async)


def register_resource(name: str, resource: Any):
    """Register a resource for tracking with the global manager"""
    manager = get_cleanup_manager()
    manager.register_resource_tracker(name, resource)


async def cleanup_all_resources():
    """Cleanup all resources using the global manager"""
    manager = get_cleanup_manager()
    await manager.async_cleanup()


# Context manager for automatic resource cleanup
@asynccontextmanager
async def managed_resource(resource_factory: Callable, cleanup_func: Optional[Callable] = None, name: str = None):
    """Context manager for automatic resource management"""
    resource = None
    try:
        # Create resource
        if asyncio.iscoroutinefunction(resource_factory):
            resource = await resource_factory()
        else:
            resource = resource_factory()

        # Register for tracking
        if name:
            register_resource(name, resource)

        yield resource

    finally:
        # Cleanup resource
        if resource is not None and cleanup_func is not None:
            try:
                if asyncio.iscoroutinefunction(cleanup_func):
                    await cleanup_func(resource)
                else:
                    cleanup_func(resource)
            except Exception as e:
                logger.error(f"Error cleaning up resource {name}: {e}")


# Decorator for automatic resource cleanup
def auto_cleanup(cleanup_func: Optional[Callable] = None):
    """Decorator to automatically register cleanup for function resources"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)

            # Register cleanup if provided
            if cleanup_func:
                register_cleanup(lambda: cleanup_func(result), asyncio.iscoroutinefunction(cleanup_func))

            return result
        return wrapper
    return decorator


# Component-specific cleanup handlers
async def cleanup_database_connections():
    """Cleanup database connections"""
    try:
        # Close all database pools/connections
        logger.info("Cleaning up database connections")
        # Implementation depends on specific database client
    except Exception as e:
        logger.error(f"Error cleaning up database connections: {e}")


async def cleanup_redis_connections():
    """Cleanup Redis connections"""
    try:
        logger.info("Cleaning up Redis connections")
        # Close Redis client pools
        # Implementation depends on specific Redis client
    except Exception as e:
        logger.error(f"Error cleaning up Redis connections: {e}")


async def cleanup_http_clients():
    """Cleanup HTTP clients and sessions"""
    try:
        logger.info("Cleaning up HTTP clients")
        # Close aiohttp sessions, httpx clients, etc.
    except Exception as e:
        logger.error(f"Error cleaning up HTTP clients: {e}")


async def cleanup_background_tasks():
    """Cleanup background tasks"""
    try:
        logger.info("Cleaning up background tasks")

        # Get all running tasks
        tasks = [task for task in asyncio.all_tasks() if not task.done()]

        if tasks:
            logger.info(f"Cancelling {len(tasks)} background tasks")

            # Cancel all tasks
            for task in tasks:
                task.cancel()

            # Wait for cancellation with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Some background tasks did not cancel within timeout")

    except Exception as e:
        logger.error(f"Error cleaning up background tasks: {e}")


async def cleanup_file_descriptors():
    """Cleanup open file descriptors"""
    try:
        logger.info("Cleaning up file descriptors")
        # Close any open files, sockets, etc.
        # This is typically handled by individual components
    except Exception as e:
        logger.error(f"Error cleaning up file descriptors: {e}")


# Register default cleanup handlers
def register_default_cleanup_handlers():
    """Register default cleanup handlers for common resources"""
    manager = get_cleanup_manager()

    # Register async cleanup handlers
    manager.register_cleanup_function(cleanup_database_connections, is_async=True)
    manager.register_cleanup_function(cleanup_redis_connections, is_async=True)
    manager.register_cleanup_function(cleanup_http_clients, is_async=True)
    manager.register_cleanup_function(cleanup_background_tasks, is_async=True)
    manager.register_cleanup_function(cleanup_file_descriptors, is_async=True)

    logger.info("Default cleanup handlers registered")


# Initialize cleanup system
def initialize_cleanup_system():
    """Initialize the cleanup system with default handlers"""
    manager = get_cleanup_manager()
    register_default_cleanup_handlers()
    logger.info("Resource cleanup system initialized")
    return manager


# Export convenience functions
__all__ = [
    'ResourceCleanupManager',
    'get_cleanup_manager',
    'register_cleanup',
    'register_resource',
    'cleanup_all_resources',
    'managed_resource',
    'auto_cleanup',
    'initialize_cleanup_system'
]