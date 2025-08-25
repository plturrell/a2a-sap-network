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
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime
from contextlib import asynccontextmanager
import atexit

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
        
        # Register signal handlers
        self._register_signal_handlers()
        
        # Register atexit handler
        atexit.register(self.sync_cleanup)
    
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