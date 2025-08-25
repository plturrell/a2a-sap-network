#!/usr/bin/env python3
"""
Error Handling Mixin for A2A Agents
Provides easy integration of comprehensive error handling capabilities
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta

from .errorHandling import (
    ErrorCategory,
    RecoveryStrategy,
    CircuitBreaker,
    error_handler,
    get_error_manager,
)

logger = logging.getLogger(__name__)


class ErrorHandlingMixin:
    """
    Mixin class that provides comprehensive error handling capabilities to A2A agents

    Usage:
        class MyAgent(A2AAgentBase, ErrorHandlingMixin):
            def __init__(self, ...):
                A2AAgentBase.__init__(self, ...)
                ErrorHandlingMixin.__init__(self)
    """

    def __init__(self):
        """Initialize error handling capabilities"""
        # Initialize error manager
        agent_id = getattr(self, "agent_id", "unknown_agent")
        self._error_manager = get_error_manager(agent_id)

        # Error handling configuration
        self._error_handling_enabled = True
        self._auto_circuit_breakers = True
        self._error_escalation_enabled = True

        # Error tracking
        self._operation_metrics = {}
        self._last_health_check = None

        logger.info(f"Error handling initialized for agent: {agent_id}")

    def enable_error_handling(
        self,
        auto_circuit_breakers: bool = True,
        error_escalation: bool = True,
        custom_strategies: Dict[str, RecoveryStrategy] = None,
    ):
        """Enable and configure error handling"""
        self._error_handling_enabled = True
        self._auto_circuit_breakers = auto_circuit_breakers
        self._error_escalation_enabled = error_escalation

        # Apply custom recovery strategies
        if custom_strategies:
            self._error_manager.recovery_strategies.update(custom_strategies)

        # Register default circuit breakers for common operations
        if auto_circuit_breakers:
            self._register_default_circuit_breakers()

        logger.info(f"Error handling enabled for {self._error_manager.agent_id}")

    def disable_error_handling(self):
        """Disable error handling (for testing or emergency scenarios)"""
        self._error_handling_enabled = False
        logger.warning(f"Error handling disabled for {self._error_manager.agent_id}")

    def _register_default_circuit_breakers(self):
        """Register circuit breakers for common operations"""
        default_breakers = {
            "http_requests": {"failure_threshold": 5, "recovery_timeout": 300},
            "database_operations": {"failure_threshold": 3, "recovery_timeout": 600},
            "external_apis": {"failure_threshold": 10, "recovery_timeout": 180},
            "file_operations": {"failure_threshold": 5, "recovery_timeout": 120},
            "message_processing": {"failure_threshold": 8, "recovery_timeout": 240},
        }

        for name, config in default_breakers.items():
            self._error_manager.register_circuit_breaker(name, **config)

    async def execute_with_recovery(
        self,
        operation_name: str,
        func: Callable,
        *args,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Dict[str, Any] = None,
        **kwargs,
    ) -> Any:
        """Execute operation with comprehensive error handling and recovery"""
        if not self._error_handling_enabled:
            # Execute without error handling
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        # Track operation metrics
        start_time = datetime.utcnow()

        try:
            result = await self._error_manager.execute_with_recovery(
                operation_name=operation_name,
                func=func,
                *args,
                category=category,
                context=context or {},
                **kwargs,
            )

            # Update success metrics
            self._update_operation_metrics(operation_name, True, start_time)
            return result

        except Exception:
            # Update failure metrics
            self._update_operation_metrics(operation_name, False, start_time)
            raise

    def _update_operation_metrics(self, operation: str, success: bool, start_time: datetime):
        """Update operation metrics for monitoring"""
        duration = (datetime.utcnow() - start_time).total_seconds()

        if operation not in self._operation_metrics:
            self._operation_metrics[operation] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_duration": 0.0,
                "avg_duration": 0.0,
                "last_call": None,
            }

        metrics = self._operation_metrics[operation]
        metrics["total_calls"] += 1
        metrics["total_duration"] += duration
        metrics["avg_duration"] = metrics["total_duration"] / metrics["total_calls"]
        metrics["last_call"] = datetime.utcnow().isoformat()

        if success:
            metrics["successful_calls"] += 1
        else:
            metrics["failed_calls"] += 1

    def register_circuit_breaker(self, name: str, **config) -> CircuitBreaker:
        """Register a custom circuit breaker"""
        return self._error_manager.register_circuit_breaker(name, **config)

    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self._error_manager.get_circuit_breaker(name)

    def add_recovery_strategy(self, name: str, strategy: RecoveryStrategy):
        """Add custom recovery strategy"""
        self._error_manager.recovery_strategies[name] = strategy
        logger.info(f"Added recovery strategy '{name}' for {self._error_manager.agent_id}")

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive error summary"""
        summary = self._error_manager.get_error_summary(hours)

        # Add operation metrics
        summary["operation_metrics"] = self._operation_metrics.copy()

        # Add agent-specific health indicators
        summary["agent_health"] = self._get_agent_health_indicators()

        return summary

    def _get_agent_health_indicators(self) -> Dict[str, Any]:
        """Get agent-specific health indicators"""
        total_errors = self._error_manager.global_error_count

        # Calculate error rates
        recent_summary = self._error_manager.get_error_summary(1)  # Last hour
        error_rate_1h = recent_summary["total_errors"]

        # Check circuit breaker status
        open_breakers = sum(
            1
            for breaker in self._error_manager.circuit_breakers.values()
            if breaker.state.value == "open"
        )

        # Calculate overall health score (0-100)
        health_score = 100
        health_score -= min(error_rate_1h * 2, 50)  # Reduce for recent errors
        health_score -= min(open_breakers * 10, 30)  # Reduce for open breakers
        health_score = max(health_score, 0)

        return {
            "health_score": health_score,
            "total_lifetime_errors": total_errors,
            "errors_last_hour": error_rate_1h,
            "open_circuit_breakers": open_breakers,
            "total_circuit_breakers": len(self._error_manager.circuit_breakers),
            "last_health_check": datetime.utcnow().isoformat(),
        }

    def get_recovery_recommendations(self) -> List[str]:
        """Get recovery recommendations"""
        return self._error_manager.get_recovery_recommendations()

    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check including error analysis"""
        self._last_health_check = datetime.utcnow()

        # Get base health info
        health = {
            "timestamp": self._last_health_check.isoformat(),
            "agent_id": self._error_manager.agent_id,
            "error_handling_enabled": self._error_handling_enabled,
        }

        # Add error summary
        health.update(self.get_error_summary(hours=1))  # Last hour summary

        # Add recommendations
        health["recommendations"] = self.get_recovery_recommendations()

        # Check if agent is in degraded state
        health["degraded"] = self._is_agent_degraded()

        return health

    def _is_agent_degraded(self) -> bool:
        """Check if agent is in degraded state"""
        summary = self._error_manager.get_error_summary(hours=1)

        # Agent is degraded if:
        # 1. More than 10 errors in last hour
        # 2. Any critical errors in last hour
        # 3. Multiple circuit breakers are open

        if summary["total_errors"] > 10:
            return True

        if summary["by_severity"].get("critical", 0) > 0:
            return True

        open_breakers = sum(
            1
            for breaker in self._error_manager.circuit_breakers.values()
            if breaker.state.value == "open"
        )
        if open_breakers > 2:
            return True

        return False

    def reset_circuit_breaker(self, name: str) -> bool:
        """Manually reset a circuit breaker"""
        breaker = self.get_circuit_breaker(name)
        if breaker:
            breaker.state = breaker.state.CLOSED
            breaker.failure_count = 0
            breaker.success_count = 0
            breaker.state_change_time = datetime.utcnow()
            logger.info(f"Manually reset circuit breaker '{name}'")
            return True
        return False

    def clear_error_history(self, hours: int = None):
        """Clear error history (optionally only older than specified hours)"""
        if hours is None:
            self._error_manager.error_history.clear()
            self._error_manager.global_error_count = 0
            self._error_manager.error_patterns.clear()
            logger.info("Cleared all error history")
        else:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            self._error_manager.error_history = [
                e for e in self._error_manager.error_history if e.timestamp >= cutoff_time
            ]
            logger.info(f"Cleared error history older than {hours} hours")

    # Convenience methods for common operations
    async def http_request_with_recovery(self, func: Callable, *args, **kwargs):
        """Execute HTTP request with error handling"""
        return await self.execute_with_recovery(
            "http_request", func, *args, category=ErrorCategory.NETWORK, **kwargs
        )

    async def database_operation_with_recovery(self, func: Callable, *args, **kwargs):
        """Execute database operation with error handling"""
        return await self.execute_with_recovery(
            "database_operation", func, *args, category=ErrorCategory.STORAGE, **kwargs
        )

    async def external_api_call_with_recovery(self, func: Callable, *args, **kwargs):
        """Execute external API call with error handling"""
        return await self.execute_with_recovery(
            "external_api_call", func, *args, category=ErrorCategory.EXTERNAL_SERVICE, **kwargs
        )

    async def processing_operation_with_recovery(self, func: Callable, *args, **kwargs):
        """Execute processing operation with error handling"""
        return await self.execute_with_recovery(
            "processing_operation", func, *args, category=ErrorCategory.PROCESSING, **kwargs
        )


# Convenience decorators for error handling
def with_error_recovery(category: ErrorCategory = ErrorCategory.UNKNOWN):
    """Decorator that automatically applies error recovery to methods"""

    def decorator(func):
        return error_handler(category=category)(func)

    return decorator


def with_circuit_breaker(breaker_name: str):
    """Decorator that applies circuit breaker protection"""

    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            if hasattr(self, "_error_manager"):
                breaker = self._error_manager.get_circuit_breaker(breaker_name)
                if breaker:
                    return await breaker.call(func, self, *args, **kwargs)

            # Fallback to direct execution
            if asyncio.iscoroutinefunction(func):
                return await func(self, *args, **kwargs)
            else:
                return func(self, *args, **kwargs)

        return wrapper

    return decorator
