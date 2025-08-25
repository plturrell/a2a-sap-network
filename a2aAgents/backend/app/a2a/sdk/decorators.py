"""
Decorators for A2A Agent SDK
"""

from functools import wraps
from typing import Dict, Any, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def a2a_handler(method: str, description: str = ""):
    """
    Decorator to mark a method as an A2A message handler

    Args:
        method: The message method this handler processes
        description: Optional description of what the handler does

    Example:
        @a2a_handler("process_data", "Processes incoming data")
        async def handle_data_processing(self, message, context_id):
            # Handler implementation
    """
    def decorator(func: Callable) -> Callable:
        func._a2a_handler = {
            "method": method,
            "description": description,
            "function_name": func.__name__
        }

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        wrapper._a2a_handler = func._a2a_handler
        return wrapper

    return decorator


def a2a_task(
    task_type: str,
    description: str = "",
    timeout: int = 300,
    retry_attempts: int = 3
):
    """
    Decorator to mark a method as a task handler

    Args:
        task_type: Type of task this method handles
        description: Description of the task
        timeout: Task timeout in seconds
        retry_attempts: Number of retry attempts

    Example:
        @a2a_task("data_standardization", "Standardizes financial data", timeout=600)
        async def standardize_data(self, task_data):
            # Task implementation
    """
    def decorator(func: Callable) -> Callable:
        func._a2a_task = {
            "task_type": task_type,
            "description": description,
            "timeout": timeout,
            "retry_attempts": retry_attempts,
            "function_name": func.__name__
        }

        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Create task tracking
            task_id = await self.create_task(task_type, kwargs)

            try:
                # Update task status to running
                from .types import TaskStatus
                await self.update_task(task_id, TaskStatus.RUNNING)

                # Execute task
                result = await func(self, *args, **kwargs)

                # Mark as completed
                await self.update_task(task_id, TaskStatus.COMPLETED, result=result)
                return result

            except Exception as e:
                # Mark as failed
                await self.update_task(task_id, TaskStatus.FAILED, error=str(e))
                raise

        wrapper._a2a_task = func._a2a_task
        return wrapper

    return decorator


def a2a_skill(
    name: str,
    description: str = "",
    capabilities: List[str] = None,
    input_schema: Dict[str, Any] = None,
    output_schema: Dict[str, Any] = None,
    domain: str = "general"
):
    """
    Decorator to mark a method as an A2A skill

    Args:
        name: Name of the skill
        description: Description of what the skill does
        capabilities: List of capabilities this skill provides
        input_schema: JSON schema for input validation
        output_schema: JSON schema for output validation
        domain: Domain this skill belongs to (e.g., "financial", "data-processing")

    Example:
        @a2a_skill(
            name="account_standardization",
            description="Standardizes account data to common format",
            capabilities=["data-standardization", "financial-processing"],
            domain="financial"
        )
        async def standardize_accounts(self, input_data):
            # Skill implementation
            return standardized_data
    """
    def decorator(func: Callable) -> Callable:
        func._a2a_skill = {
            "name": name,
            "description": description,
            "capabilities": capabilities or [],
            "input_schema": input_schema,
            "output_schema": output_schema,
            "domain": domain,
            "function_name": func.__name__
        }

        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger.debug(f"Executing skill: {name}")
            return await func(*args, **kwargs)

        wrapper._a2a_skill = func._a2a_skill
        return wrapper

    return decorator


def requires_auth(roles: List[str] = None):
    """
    Decorator to require authentication for a handler

    Args:
        roles: List of required roles
    """
    def decorator(func: Callable) -> Callable:
        func._requires_auth = True
        func._required_roles = roles or []

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Authentication logic would go here
            # For now, just log the requirement
            logger.debug(f"Authentication required for {func.__name__}, roles: {roles}")
            return await func(*args, **kwargs)

        wrapper._requires_auth = func._requires_auth
        wrapper._required_roles = func._required_roles
        return wrapper

    return decorator


def rate_limit(requests_per_minute: int = 60, burst_size: int = 10):
    """
    Decorator to apply rate limiting to a handler

    Args:
        requests_per_minute: Number of requests allowed per minute
        burst_size: Size of burst bucket
    """
    def decorator(func: Callable) -> Callable:
        func._rate_limit = {
            "requests_per_minute": requests_per_minute,
            "burst_size": burst_size
        }

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Rate limiting logic would be implemented here
            logger.debug(f"Rate limiting applied to {func.__name__}: {requests_per_minute}/min")
            return await func(*args, **kwargs)

        wrapper._rate_limit = func._rate_limit
        return wrapper

    return decorator


def validate_input(schema: Dict[str, Any]):
    """
    Decorator to validate input against JSON schema

    Args:
        schema: JSON schema for validation
    """
    def decorator(func: Callable) -> Callable:
        func._input_schema = schema

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Input validation logic would go here
            # For now, just store the schema
            logger.debug(f"Input validation applied to {func.__name__}")
            return await func(*args, **kwargs)

        wrapper._input_schema = func._input_schema
        return wrapper

    return decorator


def cache_result(ttl_seconds: int = 300):
    """
    Decorator to cache method results

    Args:
        ttl_seconds: Time to live for cache entries
    """
    def decorator(func: Callable) -> Callable:
        func._cache_ttl = ttl_seconds

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Caching logic would go here
            logger.debug(f"Caching enabled for {func.__name__}, TTL: {ttl_seconds}s")
            return await func(*args, **kwargs)

        wrapper._cache_ttl = func._cache_ttl
        return wrapper

    return decorator


# Helper functions for extracting metadata
def get_handler_metadata(method: Callable) -> Optional[Dict[str, Any]]:
    """Extract handler metadata from decorated method"""
    if hasattr(method, '_a2a_handler'):
        return method._a2a_handler
    return None


def get_task_metadata(method: Callable) -> Optional[Dict[str, Any]]:
    """Extract task metadata from decorated method"""
    if hasattr(method, '_a2a_task'):
        return method._a2a_task
    return None


def get_skill_metadata(method: Callable) -> Optional[Dict[str, Any]]:
    """Extract skill metadata from decorated method"""
    if hasattr(method, '_a2a_skill'):
        return method._a2a_skill
    return None
