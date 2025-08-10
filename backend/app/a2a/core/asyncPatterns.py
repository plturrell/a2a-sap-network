"""
Async utility patterns for A2A agents
Provides decorators and utilities for async operations
"""

import asyncio
import functools
import logging
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AsyncOperationType(str, Enum):
    """Types of async operations"""
    IO_BOUND = "io_bound"
    CPU_BOUND = "cpu_bound"
    NETWORK = "network"


@dataclass
class AsyncOperationConfig:
    """Configuration for async operations"""
    max_retries: int = 3
    timeout: float = 30.0
    concurrent_limit: int = 10
    operation_type: AsyncOperationType = AsyncOperationType.IO_BOUND


class AsyncManager:
    """Manager for async operations"""
    
    def __init__(self):
        self.semaphores = {}
    
    def get_semaphore(self, name: str, limit: int) -> asyncio.Semaphore:
        """Get or create a named semaphore"""
        if name not in self.semaphores:
            self.semaphores[name] = asyncio.Semaphore(limit)
        return self.semaphores[name]


# Global async manager instance
async_manager = AsyncManager()


def async_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    operation_type: AsyncOperationType = AsyncOperationType.IO_BOUND
):
    """
    Decorator for retrying async functions
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
        operation_type: Type of operation for logging
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"({operation_type.value}): {str(e)}"
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All retries failed for {func.__name__}: {str(e)}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


def async_timeout(timeout_seconds: float):
    """
    Decorator for adding timeout to async functions
    
    Args:
        timeout_seconds: Timeout in seconds
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"Timeout ({timeout_seconds}s) exceeded for {func.__name__}"
                )
                raise
        
        return wrapper
    return decorator


def async_concurrent_limit(
    semaphore_name: str,
    limit: int = 10
):
    """
    Decorator for limiting concurrent executions of an async function
    
    Args:
        semaphore_name: Name of the semaphore to use
        limit: Maximum concurrent executions
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            semaphore = async_manager.get_semaphore(semaphore_name, limit)
            async with semaphore:
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


async def gather_with_concurrency(
    coros: list,
    limit: int = 10,
    return_exceptions: bool = False
) -> list:
    """
    Run multiple coroutines with a concurrency limit
    
    Args:
        coros: List of coroutines to run
        limit: Maximum concurrent coroutines
        return_exceptions: Whether to return exceptions or raise
    
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def run_with_semaphore(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(
        *[run_with_semaphore(coro) for coro in coros],
        return_exceptions=return_exceptions
    )


__all__ = [
    'AsyncOperationType',
    'AsyncOperationConfig',
    'async_manager',
    'async_retry',
    'async_timeout',
    'async_concurrent_limit',
    'gather_with_concurrency'
]