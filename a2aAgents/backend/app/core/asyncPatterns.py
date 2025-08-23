"""
import time
A2A Platform Async/Await Standardization Framework
Provides consistent patterns, utilities, and decorators for async operations
"""

import asyncio
import inspect
import functools
import contextlib
import traceback
from typing import (
    Any, Awaitable, Callable, Coroutine, Dict, List, Optional, 
    TypeVar, Union, Generic, Protocol, runtime_checkable,
    AsyncContextManager, AsyncIterator
)

try:
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.core.loggingConfig import get_logger, LogCategory, log_operation
from app.core.exceptions import (


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
    A2ABaseException, A2ATimeoutError, A2AConcurrencyError, 
    A2AResourceExhaustionError, ErrorSeverity
)

T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')

logger = get_logger(__name__, LogCategory.SYSTEM)


class AsyncOperationType(str, Enum):
    """Types of async operations for standardization"""
    IO_BOUND = "io_bound"           # Network, file, database operations
    CPU_BOUND = "cpu_bound"         # Computational tasks
    BACKGROUND = "background"       # Fire-and-forget tasks
    STREAMING = "streaming"         # Long-running data streams
    BATCH = "batch"                # Batch processing operations
    AGENT_COMM = "agent_comm"      # Inter-agent communication
    EXTERNAL_API = "external_api"   # External service calls


@dataclass
class AsyncOperationConfig:
    """Configuration for async operation execution"""
    timeout_seconds: Optional[float] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    max_concurrent: Optional[int] = None
    circuit_breaker_threshold: int = 5
    enable_logging: bool = True
    operation_type: AsyncOperationType = AsyncOperationType.IO_BOUND


@dataclass
class AsyncOperationResult(Generic[T]):
    """Standardized result container for async operations"""
    success: bool
    result: Optional[T] = None
    error: Optional[Exception] = None
    duration: float = 0.0
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        return self.success and self.error is None
    
    @property
    def is_failure(self) -> bool:
        return not self.success or self.error is not None


@runtime_checkable
class AsyncCallable(Protocol[R]):
    """Protocol for async callable types"""
    async def __call__(self, *args: Any, **kwargs: Any) -> R:
        ...


class AsyncOperationManager:
    """Centralized manager for async operation execution and monitoring"""
    
    def __init__(self, max_workers: int = 10, thread_pool_size: int = 5):
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        self.active_operations: Dict[str, asyncio.Task] = {}
        self.operation_stats: Dict[str, Dict[str, Any]] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._shutdown_event = asyncio.Event()
        
    async def execute_async_operation(
        self,
        operation: AsyncCallable,
        config: AsyncOperationConfig,
        operation_id: Optional[str] = None,
        *args,
        **kwargs
    ) -> AsyncOperationResult[Any]:
        """Execute async operation with standardized error handling and monitoring"""
        
        operation_id = operation_id or f"{operation.__name__}_{id(operation)}"
        start_time = datetime.utcnow()
        
        # Initialize operation tracking
        if operation_id not in self.operation_stats:
            self.operation_stats[operation_id] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'avg_duration': 0.0,
                'last_success': None,
                'last_failure': None
            }
        
        self.operation_stats[operation_id]['total_calls'] += 1
        
        # Check circuit breaker
        if self._is_circuit_breaker_open(operation_id, config):
            error = A2AConcurrencyError(
                f"Circuit breaker open for operation {operation_id}",
                operation_id=operation_id,
                severity=ErrorSeverity.HIGH
            )
            return AsyncOperationResult(success=False, error=error)
        
        # Execute with retry logic
        last_exception = None
        retries = 0
        
        for attempt in range(config.max_retries + 1):
            try:
                # Apply timeout if specified
                if config.timeout_seconds:
                    result = await asyncio.wait_for(
                        operation(*args, **kwargs),
                        timeout=config.timeout_seconds
                    )
                else:
                    result = await operation(*args, **kwargs)
                
                # Record success
                duration = (datetime.utcnow() - start_time).total_seconds()
                self._record_operation_success(operation_id, duration)
                
                if config.enable_logging:
                    logger.info(
                        f"Async operation completed: {operation_id}",
                        operation_id=operation_id,
                        duration_seconds=duration,
                        retries=retries,
                        operation_type=config.operation_type.value
                    )
                
                return AsyncOperationResult(
                    success=True,
                    result=result,
                    duration=duration,
                    retries=retries,
                    metadata={'operation_type': config.operation_type.value}
                )
                
            except asyncio.TimeoutError as e:
                last_exception = A2ATimeoutError(
                    f"Operation {operation_id} timed out after {config.timeout_seconds}s",
                    timeout_seconds=config.timeout_seconds,
                    operation_id=operation_id
                )
                
            except Exception as e:
                last_exception = e
                
            # Record failure and implement retry delay
            retries = attempt
            self._record_operation_failure(operation_id, last_exception)
            
            if attempt < config.max_retries:
                delay = config.retry_delay
                if config.exponential_backoff:
                    delay *= (2 ** attempt)
                
                if config.enable_logging:
                    logger.warning(
                        f"Async operation failed, retrying: {operation_id}",
                        operation_id=operation_id,
                        attempt=attempt + 1,
                        max_retries=config.max_retries,
                        retry_delay=delay,
                        error=str(last_exception)
                    )
                
                await asyncio.sleep(delay)
        
        # All retries exhausted
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        if config.enable_logging:
            logger.error(
                f"Async operation failed after all retries: {operation_id}",
                operation_id=operation_id,
                total_duration=duration,
                retries=retries,
                final_error=str(last_exception)
            )
        
        return AsyncOperationResult(
            success=False,
            error=last_exception,
            duration=duration,
            retries=retries,
            metadata={'operation_type': config.operation_type.value}
        )
    
    async def execute_concurrent_operations(
        self,
        operations: List[tuple[AsyncCallable, tuple, dict]],  # (func, args, kwargs)
        config: AsyncOperationConfig,
        max_concurrent: Optional[int] = None
    ) -> List[AsyncOperationResult[Any]]:
        """Execute multiple async operations concurrently with proper resource management"""
        
        max_concurrent = max_concurrent or config.max_concurrent or len(operations)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(operation_data):
            async with semaphore:
                func, args, kwargs = operation_data
                return await self.execute_async_operation(func, config, None, *args, **kwargs)
        
        # Execute all operations concurrently
        tasks = [execute_with_semaphore(op) for op in operations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                final_results.append(AsyncOperationResult(success=False, error=result))
            else:
                final_results.append(result)
        
        return final_results
    
    async def create_background_task(
        self,
        operation: AsyncCallable,
        config: AsyncOperationConfig,
        task_name: str,
        *args,
        **kwargs
    ) -> str:
        """Create a background task with proper lifecycle management"""
        
        task_id = f"{task_name}_{datetime.utcnow().timestamp()}"
        
        async def background_wrapper():
            try:
                result = await self.execute_async_operation(
                    operation, config, task_id, *args, **kwargs
                )
                if result.is_failure and config.enable_logging:
                    logger.error(
                        f"Background task failed: {task_name}",
                        task_id=task_id,
                        error=str(result.error)
                    )
            except Exception as e:
                if config.enable_logging:
                    logger.error(
                        f"Background task exception: {task_name}",
                        task_id=task_id,
                        error=str(e),
                        exc_info=True
                    )
            finally:
                # Clean up task tracking
                self.active_operations.pop(task_id, None)
        
        # Create and track the task
        task = asyncio.create_task(background_wrapper())
        self.active_operations[task_id] = task
        
        if config.enable_logging:
            logger.info(
                f"Background task created: {task_name}",
                task_id=task_id,
                task_name=task_name
            )
        
        return task_id
    
    def _is_circuit_breaker_open(self, operation_id: str, config: AsyncOperationConfig) -> bool:
        """Check if circuit breaker is open for operation"""
        if operation_id not in self.circuit_breakers:
            self.circuit_breakers[operation_id] = {
                'failure_count': 0,
                'last_failure_time': None,
                'state': 'closed'  # closed, open, half-open
            }
        
        breaker = self.circuit_breakers[operation_id]
        
        if breaker['state'] == 'closed':
            return False
        elif breaker['state'] == 'open':
            # Check if we should transition to half-open
            if (breaker['last_failure_time'] and 
                datetime.utcnow() - breaker['last_failure_time'] > timedelta(minutes=5)):
                breaker['state'] = 'half-open'
                return False
            return True
        else:  # half-open
            return False
    
    def _record_operation_success(self, operation_id: str, duration: float):
        """Record successful operation execution"""
        stats = self.operation_stats[operation_id]
        stats['successful_calls'] += 1
        stats['last_success'] = datetime.utcnow()
        
        # Update average duration
        total_successful = stats['successful_calls']
        current_avg = stats['avg_duration']
        stats['avg_duration'] = ((current_avg * (total_successful - 1)) + duration) / total_successful
        
        # Reset circuit breaker on success
        if operation_id in self.circuit_breakers:
            self.circuit_breakers[operation_id]['failure_count'] = 0
            self.circuit_breakers[operation_id]['state'] = 'closed'
    
    def _record_operation_failure(self, operation_id: str, error: Exception):
        """Record failed operation execution"""
        stats = self.operation_stats[operation_id]
        stats['failed_calls'] += 1
        stats['last_failure'] = datetime.utcnow()
        
        # Update circuit breaker
        if operation_id not in self.circuit_breakers:
            self.circuit_breakers[operation_id] = {
                'failure_count': 0,
                'last_failure_time': None,
                'state': 'closed'
            }
        
        breaker = self.circuit_breakers[operation_id]
        breaker['failure_count'] += 1
        breaker['last_failure_time'] = datetime.utcnow()
        
        # Open circuit breaker if threshold exceeded
        if breaker['failure_count'] >= 5:  # Default threshold
            breaker['state'] = 'open'
    
    async def shutdown(self):
        """Gracefully shutdown all async operations"""
        self._shutdown_event.set()
        
        # Cancel all active background tasks
        for task_id, task in self.active_operations.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled background task: {task_id}")
        
        # Wait for tasks to complete or be cancelled
        if self.active_operations:
            await asyncio.gather(*self.active_operations.values(), return_exceptions=True)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("AsyncOperationManager shutdown complete")


# Global instance for application use
async_manager = AsyncOperationManager()


# Standardized Decorators for Common Patterns

def async_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    exponential_backoff: bool = True,
    timeout_seconds: Optional[float] = None,
    operation_type: AsyncOperationType = AsyncOperationType.IO_BOUND
):
    """Decorator for automatic retry logic on async functions"""
    
    def decorator(func: AsyncCallable) -> AsyncCallable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            config = AsyncOperationConfig(
                max_retries=max_retries,
                retry_delay=retry_delay,
                exponential_backoff=exponential_backoff,
                timeout_seconds=timeout_seconds,
                operation_type=operation_type
            )
            
            result = await async_manager.execute_async_operation(
                func, config, func.__name__, *args, **kwargs
            )
            
            if result.is_success:
                return result.result
            else:
                raise result.error
        
        return wrapper
    return decorator


def async_timeout(seconds: float):
    """Decorator for timeout enforcement on async functions"""
    
    def decorator(func: AsyncCallable) -> AsyncCallable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise A2ATimeoutError(
                    f"Function {func.__name__} timed out after {seconds} seconds",
                    timeout_seconds=seconds,
                    function_name=func.__name__
                )
        return wrapper
    return decorator


def async_background_task(
    task_name: Optional[str] = None,
    config: Optional[AsyncOperationConfig] = None
):
    """Decorator for background task execution"""
    
    def decorator(func: AsyncCallable) -> Callable[..., str]:
        task_name_final = task_name or func.__name__
        config_final = config or AsyncOperationConfig(
            operation_type=AsyncOperationType.BACKGROUND,
            max_retries=1,
            enable_logging=True
        )
        
        def wrapper(*args, **kwargs) -> str:
            # Return task ID for tracking
            loop = asyncio.get_event_loop()
            return loop.create_task(
                async_manager.create_background_task(
                    func, config_final, task_name_final, *args, **kwargs
                )
            )
        
        return wrapper
    return decorator


def async_concurrent_limit(max_concurrent: int):
    """Decorator for limiting concurrent executions of an async function"""
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    def decorator(func: AsyncCallable) -> AsyncCallable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# Context Managers for Resource Management

@contextlib.asynccontextmanager
async def async_transaction_context(
    connection,
    isolation_level: Optional[str] = None
) -> AsyncIterator[Any]:
    """Standardized async database transaction context"""
    
    transaction = None
    try:
        if hasattr(connection, 'begin'):
            transaction = await connection.begin()
        
        logger.debug(
            "Async transaction started",
            isolation_level=isolation_level,
            category=LogCategory.DATABASE
        )
        
        yield connection
        
        if transaction:
            await transaction.commit()
        
        logger.debug("Async transaction committed", category=LogCategory.DATABASE)
        
    except Exception as e:
        if transaction:
            await transaction.rollback()
            logger.error(
                "Async transaction rolled back",
                error=str(e),
                category=LogCategory.DATABASE,
                exc_info=True
            )
        raise


@contextlib.asynccontextmanager
async def async_resource_context(
    resource_factory: Callable[[], Awaitable[T]],
    resource_cleanup: Optional[Callable[[T], Awaitable[None]]] = None,
    resource_name: str = "async_resource"
) -> AsyncIterator[T]:
    """Generic async resource management context"""
    
    resource = None
    try:
        resource = await resource_factory()
        logger.debug(f"Async resource acquired: {resource_name}")
        
        yield resource
        
    except Exception as e:
        logger.error(
            f"Async resource error: {resource_name}",
            error=str(e),
            exc_info=True
        )
        raise
    finally:
        if resource and resource_cleanup:
            try:
                await resource_cleanup(resource)
                logger.debug(f"Async resource cleaned up: {resource_name}")
            except Exception as cleanup_error:
                logger.error(
                    f"Async resource cleanup failed: {resource_name}",
                    error=str(cleanup_error),
                    exc_info=True
                )


# Utility Functions

async def gather_with_error_handling(
    *coros: Awaitable[T],
    return_exceptions: bool = True,
    max_concurrent: Optional[int] = None
) -> List[Union[T, Exception]]:
    """Enhanced gather with proper error handling and concurrency control"""
    
    if max_concurrent:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_coro(coro):
            async with semaphore:
                return await coro
        
        limited_coros = [limited_coro(coro) for coro in coros]
        results = await asyncio.gather(*limited_coros, return_exceptions=return_exceptions)
    else:
        results = await asyncio.gather(*coros, return_exceptions=return_exceptions)
    
    # Log any exceptions if return_exceptions is True
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(
                f"Coroutine {i} failed in gather_with_error_handling",
                error=str(result),
                coroutine_index=i,
                exc_info=True
            )
    
    return results


async def async_sleep_with_cancellation(
    delay: float,
    check_interval: float = 0.1
) -> None:
    """Interruptible sleep that responds quickly to cancellation"""
    
    end_time = asyncio.get_event_loop().time() + delay
    
    while True:
        current_time = asyncio.get_event_loop().time()
        if current_time >= end_time:
            break
        
        sleep_time = min(check_interval, end_time - current_time)
        await asyncio.sleep(sleep_time)


def ensure_async(func: Union[Callable, AsyncCallable]) -> AsyncCallable:
    """Convert synchronous function to async if needed"""
    
    if inspect.iscoroutinefunction(func):
        return func
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Run sync function in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(async_manager.thread_pool, func, *args, **kwargs)
    
    return async_wrapper


# Type Checking Utilities

def validate_async_function_signature(func: Callable) -> bool:
    """Validate that function follows async best practices"""
    
    if not inspect.iscoroutinefunction(func):
        return False
    
    # Check for proper type hints
    signature = inspect.signature(func)
    
    # Should have return type annotation
    if signature.return_annotation == inspect.Signature.empty:
        logger.warning(
            f"Async function missing return type annotation: {func.__name__}",
            function_name=func.__name__
        )
        return False
    
    return True


# Export commonly used items
__all__ = [
    "AsyncOperationType",
    "AsyncOperationConfig", 
    "AsyncOperationResult",
    "AsyncOperationManager",
    "async_manager",
    "async_retry",
    "async_timeout", 
    "async_background_task",
    "async_concurrent_limit",
    "async_transaction_context",
    "async_resource_context",
    "gather_with_error_handling",
    "async_sleep_with_cancellation",
    "ensure_async",
    "validate_async_function_signature"
]