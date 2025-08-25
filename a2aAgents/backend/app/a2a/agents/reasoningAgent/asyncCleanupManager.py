"""
Async Cleanup Manager
Handles proper cleanup of async resources and background tasks
"""

import asyncio
import logging
import weakref
import signal
import sys
from typing import Set, List, Dict, Any, Optional, Callable
from contextlib import asynccontextmanager
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)


class AsyncResourceManager:
    """Manages async resources and ensures proper cleanup"""

    def __init__(self):
        self._resources: Set[Any] = weakref.WeakSet()
        self._cleanup_tasks: List[asyncio.Task] = []
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_callbacks: List[Callable] = []
        self._is_shutting_down = False
        self._lock = asyncio.Lock()

    def register_resource(self, resource: Any):
        """Register a resource for cleanup"""
        self._resources.add(resource)
        logger.debug(f"Registered resource: {type(resource).__name__}")

    def register_background_task(self, task: asyncio.Task):
        """Register a background task for cleanup"""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        logger.debug(f"Registered background task: {task.get_name()}")

    def register_shutdown_callback(self, callback: Callable):
        """Register a shutdown callback"""
        self._shutdown_callbacks.append(callback)
        logger.debug(f"Registered shutdown callback: {callback.__name__}")

    async def cleanup_all(self, timeout: float = 30.0):
        """Cleanup all registered resources"""
        if self._is_shutting_down:
            return

        async with self._lock:
            if self._is_shutting_down:
                return
            self._is_shutting_down = True

        logger.info("Starting async resource cleanup...")

        # Execute shutdown callbacks
        for callback in self._shutdown_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
                logger.debug(f"Executed shutdown callback: {callback.__name__}")
            except Exception as e:
                logger.error(f"Error in shutdown callback {callback.__name__}: {e}")

        # Cancel background tasks
        if self._background_tasks:
            logger.info(f"Cancelling {len(self._background_tasks)} background tasks...")
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete or timeout
            if self._background_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self._background_tasks, return_exceptions=True),
                        timeout=timeout / 2
                    )
                except asyncio.TimeoutError:
                    logger.warning("Some background tasks did not complete within timeout")

        # Cleanup registered resources
        cleanup_errors = []
        for resource in list(self._resources):
            try:
                await self._cleanup_resource(resource)
            except Exception as e:
                cleanup_errors.append((resource, e))
                logger.error(f"Error cleaning up {type(resource).__name__}: {e}")

        if cleanup_errors:
            logger.warning(f"Cleanup completed with {len(cleanup_errors)} errors")
        else:
            logger.info("Async resource cleanup completed successfully")

    async def _cleanup_resource(self, resource: Any):
        """Cleanup a single resource"""
        try:
            # Try common async cleanup methods
            if hasattr(resource, 'close') and asyncio.iscoroutinefunction(resource.close):
                await resource.close()
            elif hasattr(resource, 'cleanup') and asyncio.iscoroutinefunction(resource.cleanup):
                await resource.cleanup()
            elif hasattr(resource, 'shutdown') and asyncio.iscoroutinefunction(resource.shutdown):
                await resource.shutdown()
            elif hasattr(resource, 'aclose') and asyncio.iscoroutinefunction(resource.aclose):
                await resource.aclose()
            # Try sync cleanup methods
            elif hasattr(resource, 'close') and not asyncio.iscoroutinefunction(resource.close):
                resource.close()
            elif hasattr(resource, 'cleanup') and not asyncio.iscoroutinefunction(resource.cleanup):
                resource.cleanup()

            logger.debug(f"Cleaned up resource: {type(resource).__name__}")

        except Exception as e:
            logger.error(f"Failed to cleanup {type(resource).__name__}: {e}")
            raise


class AsyncReasoningCleanupManager:
    """Specialized cleanup manager for reasoning agent components"""

    def __init__(self):
        self.resource_manager = AsyncResourceManager()
        self.grok_clients: List[Any] = []
        self.memory_stores: List[Any] = []
        self.blackboard_controllers: List[Any] = []
        self.connection_pools: List[Any] = []
        self.cache_systems: List[Any] = []
        self._performance_stats = {
            "cleanup_count": 0,
            "last_cleanup": None,
            "resources_cleaned": 0,
            "errors_encountered": 0
        }

    def register_grok_client(self, client):
        """Register a Grok client for cleanup"""
        self.grok_clients.append(client)
        self.resource_manager.register_resource(client)
        logger.debug("Registered Grok client for cleanup")

    def register_memory_store(self, store):
        """Register a memory store for cleanup"""
        self.memory_stores.append(store)
        self.resource_manager.register_resource(store)
        logger.debug("Registered memory store for cleanup")

    def register_blackboard_controller(self, controller):
        """Register a blackboard controller for cleanup"""
        self.blackboard_controllers.append(controller)
        self.resource_manager.register_resource(controller)
        logger.debug("Registered blackboard controller for cleanup")

    def register_connection_pool(self, pool):
        """Register a connection pool for cleanup"""
        self.connection_pools.append(pool)
        self.resource_manager.register_resource(pool)
        logger.debug("Registered connection pool for cleanup")

    def register_cache_system(self, cache):
        """Register a cache system for cleanup"""
        self.cache_systems.append(cache)
        self.resource_manager.register_resource(cache)
        logger.debug("Registered cache system for cleanup")

    async def cleanup_reasoning_components(self):
        """Cleanup all reasoning-related components"""
        start_time = datetime.utcnow()
        resources_cleaned = 0
        errors = 0

        logger.info("Starting reasoning components cleanup...")

        try:
            # Cleanup Grok clients first (they might have active connections)
            for client in self.grok_clients:
                try:
                    if hasattr(client, 'close'):
                        await client.close()
                    resources_cleaned += 1
                except Exception as e:
                    logger.error(f"Error cleaning up Grok client: {e}")
                    errors += 1

            # Cleanup memory stores
            for store in self.memory_stores:
                try:
                    if hasattr(store, 'close'):
                        await store.close()
                    elif hasattr(store, 'cleanup'):
                        await store.cleanup()
                    resources_cleaned += 1
                except Exception as e:
                    logger.error(f"Error cleaning up memory store: {e}")
                    errors += 1

            # Cleanup blackboard controllers
            for controller in self.blackboard_controllers:
                try:
                    if hasattr(controller, 'cleanup'):
                        await controller.cleanup()
                    resources_cleaned += 1
                except Exception as e:
                    logger.error(f"Error cleaning up blackboard controller: {e}")
                    errors += 1

            # Cleanup connection pools
            for pool in self.connection_pools:
                try:
                    if hasattr(pool, 'close'):
                        await pool.close()
                    resources_cleaned += 1
                except Exception as e:
                    logger.error(f"Error cleaning up connection pool: {e}")
                    errors += 1

            # Cleanup cache systems
            for cache in self.cache_systems:
                try:
                    if hasattr(cache, 'close'):
                        await cache.close()
                    resources_cleaned += 1
                except Exception as e:
                    logger.error(f"Error cleaning up cache system: {e}")
                    errors += 1

            # Cleanup all other resources
            await self.resource_manager.cleanup_all()

            # Update performance stats
            self._performance_stats.update({
                "cleanup_count": self._performance_stats["cleanup_count"] + 1,
                "last_cleanup": start_time.isoformat(),
                "resources_cleaned": resources_cleaned,
                "errors_encountered": errors
            })

            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Reasoning components cleanup completed in {duration:.2f}s "
                       f"({resources_cleaned} resources, {errors} errors)")

        except Exception as e:
            logger.error(f"Cleanup process failed: {e}")
            raise

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get cleanup performance statistics"""
        return self._performance_stats.copy()

    @asynccontextmanager
    async def managed_session(self):
        """Context manager for automatic cleanup"""
        try:
            yield self
        finally:
            await self.cleanup_reasoning_components()


class AsyncSignalHandler:
    """Handles async cleanup on system signals"""

    def __init__(self, cleanup_manager: AsyncReasoningCleanupManager):
        self.cleanup_manager = cleanup_manager
        self._cleanup_done = False

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        if sys.platform != 'win32':
            # Unix-like systems
            loop = asyncio.get_event_loop()

            for sig in [signal.SIGINT, signal.SIGTERM]:
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(self._handle_signal(s))
                )

            logger.info("Signal handlers setup for graceful shutdown")
        else:
            # Windows - limited signal support
            signal.signal(signal.SIGINT, self._sync_signal_handler)
            signal.signal(signal.SIGBREAK, self._sync_signal_handler)
            logger.info("Windows signal handlers setup")

    async def _handle_signal(self, sig):
        """Handle shutdown signal asynchronously"""
        if self._cleanup_done:
            return

        logger.info(f"Received signal {sig.name}, initiating graceful shutdown...")

        try:
            await self.cleanup_manager.cleanup_reasoning_components()
            self._cleanup_done = True
            logger.info("Graceful shutdown completed")
        except Exception as e:
            logger.error(f"Error during signal cleanup: {e}")
        finally:
            # Exit the event loop
            loop = asyncio.get_event_loop()
            loop.stop()

    def _sync_signal_handler(self, sig, frame):
        """Synchronous signal handler for Windows"""
        if self._cleanup_done:
            return

        logger.info(f"Received signal {sig}, initiating cleanup...")

        try:
            # Create new event loop for cleanup if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run cleanup
            loop.run_until_complete(self.cleanup_manager.cleanup_reasoning_components())
            self._cleanup_done = True
            logger.info("Signal cleanup completed")

        except Exception as e:
            logger.error(f"Error during signal cleanup: {e}")
        finally:
            sys.exit(0)


class AsyncPerformanceMonitor:
    """Monitors performance and triggers cleanup when needed"""

    def __init__(self, cleanup_manager: AsyncReasoningCleanupManager,
                 memory_threshold_mb: int = 500,
                 cleanup_interval: int = 3600):
        self.cleanup_manager = cleanup_manager
        self.memory_threshold = memory_threshold_mb * 1024 * 1024  # Convert to bytes
        self.cleanup_interval = cleanup_interval
        self.monitoring_task: Optional[asyncio.Task] = None
        self._stop_monitoring = False

    def start_monitoring(self):
        """Start performance monitoring"""
        if self.monitoring_task and not self.monitoring_task.done():
            return

        self.monitoring_task = asyncio.create_task(self._monitor_performance())
        logger.info("Started performance monitoring")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self._stop_monitoring = True
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
        logger.info("Stopped performance monitoring")

    async def _monitor_performance(self):
        """Monitor performance and trigger cleanup when needed"""
        last_cleanup = datetime.utcnow()

        while not self._stop_monitoring:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Check memory usage
                import psutil
                process = psutil.Process()
                memory_usage = process.memory_info().rss

                # Check if cleanup is needed
                now = datetime.utcnow()
                time_since_cleanup = (now - last_cleanup).total_seconds()

                should_cleanup = (
                    memory_usage > self.memory_threshold or
                    time_since_cleanup > self.cleanup_interval
                )

                if should_cleanup:
                    logger.info(f"Triggering cleanup - Memory: {memory_usage / 1024 / 1024:.1f}MB, "
                               f"Time since last: {time_since_cleanup:.0f}s")

                    # Trigger partial cleanup (not full shutdown)
                    await self._partial_cleanup()
                    last_cleanup = now

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying

    async def _partial_cleanup(self):
        """Perform partial cleanup without shutting down"""
        try:
            # Cleanup expired cache entries
            for cache in self.cleanup_manager.cache_systems:
                if hasattr(cache, 'cleanup_expired'):
                    await cache.cleanup_expired()

            # Cleanup old memory entries
            for store in self.cleanup_manager.memory_stores:
                if hasattr(store, 'cleanup_old_experiences'):
                    await store.cleanup_old_experiences()

            # Force garbage collection
            import gc
            gc.collect()

            logger.debug("Partial cleanup completed")

        except Exception as e:
            logger.error(f"Partial cleanup error: {e}")


# Global cleanup manager instance
_global_cleanup_manager: Optional[AsyncReasoningCleanupManager] = None


def get_cleanup_manager() -> AsyncReasoningCleanupManager:
    """Get the global cleanup manager instance"""
    global _global_cleanup_manager
    if _global_cleanup_manager is None:
        _global_cleanup_manager = AsyncReasoningCleanupManager()
    return _global_cleanup_manager


def setup_global_cleanup():
    """Setup global cleanup with signal handlers"""
    cleanup_manager = get_cleanup_manager()
    signal_handler = AsyncSignalHandler(cleanup_manager)
    signal_handler.setup_signal_handlers()

    # Optional: Start performance monitoring
    monitor = AsyncPerformanceMonitor(cleanup_manager)
    monitor.start_monitoring()

    return cleanup_manager


# Example usage
async def test_cleanup_manager():
    """Test the cleanup manager"""
    try:
        # Setup cleanup manager
        cleanup_manager = get_cleanup_manager()

        # Simulate registering some resources
        class MockResource:
            def __init__(self, name):
                self.name = name
                self.closed = False

            async def close(self):
                self.closed = True
                print(f"Closed {self.name}")

        # Register some mock resources
        resources = [MockResource(f"Resource{i}") for i in range(3)]
        for resource in resources:
            cleanup_manager.register_connection_pool(resource)

        print("Registered mock resources")

        # Test cleanup
        await cleanup_manager.cleanup_reasoning_components()

        # Check if resources were cleaned up
        for resource in resources:
            assert resource.closed, f"{resource.name} was not closed"

        print("✅ Cleanup manager test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_cleanup_manager())