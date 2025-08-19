"""
Self-healing calculation and validation capabilities for A2A agents.
Provides automatic recovery and correction mechanisms.
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Recovery action types."""
    RETRY = "retry"
    RECALCULATE = "recalculate"
    FALLBACK = "fallback"
    RESET = "reset"
    ESCALATE = "escalate"


@dataclass
class HealthMetric:
    """Health metric data."""
    name: str
    value: float
    threshold: float
    status: HealthStatus
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


@dataclass
class RecoveryPlan:
    """Recovery plan for self-healing."""
    issue: str
    severity: str
    actions: List[RecoveryAction]
    timeout: int
    fallback_strategy: Optional[str] = None


class SelfHealingCalculator:
    """
    Self-healing calculation system with automatic error recovery.
    """
    
    def __init__(
        self,
        name: str,
        health_check_interval: int = 60,
        recovery_timeout: int = 300,
        max_recovery_attempts: int = 3
    ):
        """
        Initialize self-healing calculator.
        
        Args:
            name: Calculator name
            health_check_interval: Seconds between health checks
            recovery_timeout: Max seconds for recovery attempt
            max_recovery_attempts: Max recovery attempts before escalation
        """
        self.name = name
        self.health_check_interval = health_check_interval
        self.recovery_timeout = recovery_timeout
        self.max_recovery_attempts = max_recovery_attempts
        
        # Health metrics
        self.health_metrics = {}
        self.health_history = deque(maxlen=100)
        self.recovery_history = deque(maxlen=50)
        
        # State
        self._health_status = HealthStatus.HEALTHY
        self._recovery_in_progress = False
        self._last_health_check = datetime.now()
        self._consecutive_failures = 0
        
        # Calculation cache for recovery
        self._calculation_cache = {}
        self._validation_rules = {}
    
    async def calculate_with_healing(
        self,
        calculation_func: Callable,
        *args,
        validation_func: Optional[Callable] = None,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute calculation with self-healing capabilities.
        
        Args:
            calculation_func: Function to execute
            validation_func: Optional validation function
            *args, **kwargs: Function arguments
            
        Returns:
            Tuple of (result, health_report)
        """
        start_time = time.time()
        attempt = 0
        last_error = None
        
        while attempt < self.max_recovery_attempts:
            try:
                # Perform calculation
                result = await self._execute_calculation(
                    calculation_func, *args, **kwargs
                )
                
                # Validate result if validation function provided
                if validation_func:
                    is_valid, validation_details = await self._validate_result(
                        result, validation_func
                    )
                    
                    if not is_valid:
                        raise ValueError(f"Validation failed: {validation_details}")
                
                # Update health metrics
                self._update_health_metrics({
                    'calculation_time': time.time() - start_time,
                    'attempts': attempt + 1,
                    'status': 'success'
                })
                
                # Reset failure counter
                self._consecutive_failures = 0
                self._health_status = HealthStatus.HEALTHY
                
                return result, self._generate_health_report()
                
            except Exception as e:
                last_error = e
                attempt += 1
                self._consecutive_failures += 1
                
                logger.warning(
                    f"Calculation failed (attempt {attempt}/{self.max_recovery_attempts}): {e}"
                )
                
                # Attempt recovery
                recovery_success = await self._attempt_recovery(
                    error=e,
                    attempt=attempt,
                    calculation_func=calculation_func,
                    args=args,
                    kwargs=kwargs
                )
                
                if not recovery_success and attempt < self.max_recovery_attempts:
                    # Wait before retry with exponential backoff
                    await asyncio.sleep(min(2 ** attempt, 30))
        
        # All attempts failed
        self._health_status = HealthStatus.CRITICAL
        self._update_health_metrics({
            'calculation_time': time.time() - start_time,
            'attempts': attempt,
            'status': 'failed',
            'error': str(last_error)
        })
        
        raise RuntimeError(
            f"Calculation failed after {attempt} attempts. Last error: {last_error}"
        )
    
    async def _execute_calculation(
        self,
        calculation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute the calculation with monitoring."""
        # Check if we have a cached result
        cache_key = self._generate_cache_key(calculation_func, args, kwargs)
        if cache_key in self._calculation_cache:
            cached_result, cached_time = self._calculation_cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=5):
                logger.debug(f"Using cached calculation result for {cache_key}")
                return cached_result
        
        # Execute calculation
        if asyncio.iscoroutinefunction(calculation_func):
            result = await calculation_func(*args, **kwargs)
        else:
            result = calculation_func(*args, **kwargs)
        
        # Cache result
        self._calculation_cache[cache_key] = (result, datetime.now())
        
        # Limit cache size
        if len(self._calculation_cache) > 100:
            oldest_key = min(
                self._calculation_cache.keys(),
                key=lambda k: self._calculation_cache[k][1]
            )
            del self._calculation_cache[oldest_key]
        
        return result
    
    async def _validate_result(
        self,
        result: Any,
        validation_func: Callable
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate calculation result."""
        try:
            if asyncio.iscoroutinefunction(validation_func):
                validation_result = await validation_func(result)
            else:
                validation_result = validation_func(result)
            
            if isinstance(validation_result, bool):
                return validation_result, {}
            elif isinstance(validation_result, tuple):
                return validation_result[0], validation_result[1]
            else:
                return bool(validation_result), {"details": str(validation_result)}
                
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, {"error": str(e)}
    
    async def _attempt_recovery(
        self,
        error: Exception,
        attempt: int,
        calculation_func: Callable,
        args: tuple,
        kwargs: dict
    ) -> bool:
        """
        Attempt to recover from calculation error.
        
        Returns:
            True if recovery successful
        """
        recovery_plan = self._create_recovery_plan(error, attempt)
        
        logger.info(f"Attempting recovery: {recovery_plan.issue}")
        
        for action in recovery_plan.actions:
            try:
                if action == RecoveryAction.RETRY:
                    # Simple retry with cleaned inputs
                    logger.info("Retrying with cleaned inputs")
                    cleaned_args = self._clean_inputs(args)
                    cleaned_kwargs = self._clean_inputs(kwargs)
                    return True  # Let main loop retry
                    
                elif action == RecoveryAction.RECALCULATE:
                    # Try alternative calculation method
                    logger.info("Attempting alternative calculation")
                    if hasattr(self, '_alternative_calculation'):
                        alt_result = await self._alternative_calculation(
                            *args, **kwargs
                        )
                        # Store for next attempt
                        self._calculation_cache[
                            self._generate_cache_key(calculation_func, args, kwargs)
                        ] = (alt_result, datetime.now())
                        return True
                        
                elif action == RecoveryAction.FALLBACK:
                    # Use fallback values
                    logger.info("Using fallback calculation")
                    if recovery_plan.fallback_strategy:
                        return await self._apply_fallback_strategy(
                            recovery_plan.fallback_strategy,
                            args,
                            kwargs
                        )
                        
                elif action == RecoveryAction.RESET:
                    # Reset internal state
                    logger.info("Resetting calculator state")
                    self._reset_state()
                    return True
                    
                elif action == RecoveryAction.ESCALATE:
                    # Escalate to external system
                    logger.warning("Escalating to external recovery")
                    await self._escalate_error(error, attempt)
                    return False
                    
            except Exception as recovery_error:
                logger.error(f"Recovery action {action} failed: {recovery_error}")
                continue
        
        return False
    
    def _create_recovery_plan(
        self,
        error: Exception,
        attempt: int
    ) -> RecoveryPlan:
        """Create recovery plan based on error type."""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # Determine severity
        if attempt >= self.max_recovery_attempts - 1:
            severity = "critical"
        elif self._consecutive_failures > 3:
            severity = "high"
        else:
            severity = "medium"
        
        # Create recovery actions based on error
        if "timeout" in error_msg or "connection" in error_msg:
            actions = [RecoveryAction.RETRY, RecoveryAction.RESET]
        elif "validation" in error_msg:
            actions = [RecoveryAction.RECALCULATE, RecoveryAction.FALLBACK]
        elif "memory" in error_msg or "resource" in error_msg:
            actions = [RecoveryAction.RESET, RecoveryAction.FALLBACK]
        else:
            actions = [RecoveryAction.RETRY, RecoveryAction.RECALCULATE]
        
        # Add escalation for critical issues
        if severity == "critical":
            actions.append(RecoveryAction.ESCALATE)
        
        return RecoveryPlan(
            issue=f"{error_type}: {error_msg[:100]}",
            severity=severity,
            actions=actions,
            timeout=self.recovery_timeout,
            fallback_strategy="statistical_approximation"
        )
    
    def _clean_inputs(self, inputs: Union[tuple, dict]) -> Union[tuple, dict]:
        """Clean and sanitize inputs."""
        if isinstance(inputs, dict):
            return {
                k: self._sanitize_value(v)
                for k, v in inputs.items()
            }
        elif isinstance(inputs, (list, tuple)):
            cleaned = [self._sanitize_value(v) for v in inputs]
            return tuple(cleaned) if isinstance(inputs, tuple) else cleaned
        else:
            return self._sanitize_value(inputs)
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize individual value."""
        if isinstance(value, (int, float)):
            # Handle infinity and NaN
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return value
        elif isinstance(value, str):
            # Remove problematic characters
            return value.strip()
        elif isinstance(value, (list, tuple, dict)):
            return self._clean_inputs(value)
        else:
            return value
    
    def _reset_state(self):
        """Reset internal state."""
        self._calculation_cache.clear()
        self._consecutive_failures = 0
        logger.info(f"Calculator {self.name} state reset")
    
    async def _escalate_error(
        self,
        error: Exception,
        attempt: int
    ):
        """Escalate error to external system."""
        escalation_data = {
            'calculator': self.name,
            'error': str(error),
            'error_type': type(error).__name__,
            'attempt': attempt,
            'consecutive_failures': self._consecutive_failures,
            'health_status': self._health_status.value,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.error(f"ESCALATION: {escalation_data}")
        
        # Record in recovery history
        self.recovery_history.append({
            'timestamp': datetime.now(),
            'action': 'escalation',
            'data': escalation_data
        })
    
    def _generate_cache_key(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict
    ) -> str:
        """Generate cache key for calculation."""
        import hashlib
        import json
        
        key_data = {
            'func': func.__name__,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        
        return hashlib.md5(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()
    
    def _update_health_metrics(self, metrics: Dict[str, Any]):
        """Update health metrics."""
        timestamp = datetime.now()
        
        for name, value in metrics.items():
            self.health_metrics[name] = HealthMetric(
                name=name,
                value=value,
                threshold=self._get_metric_threshold(name),
                status=self._evaluate_metric_status(name, value),
                timestamp=timestamp
            )
        
        # Record in history
        self.health_history.append({
            'timestamp': timestamp,
            'metrics': metrics,
            'status': self._health_status.value
        })
    
    def _get_metric_threshold(self, metric_name: str) -> float:
        """Get threshold for metric."""
        thresholds = {
            'calculation_time': 30.0,  # seconds
            'attempts': 2,
            'error_rate': 0.1,  # 10%
            'consecutive_failures': 3
        }
        return thresholds.get(metric_name, 0.0)
    
    def _evaluate_metric_status(
        self,
        name: str,
        value: Any
    ) -> HealthStatus:
        """Evaluate metric status."""
        if name == 'status':
            return HealthStatus.HEALTHY if value == 'success' else HealthStatus.UNHEALTHY
        
        threshold = self._get_metric_threshold(name)
        if isinstance(value, (int, float)):
            if value <= threshold:
                return HealthStatus.HEALTHY
            elif value <= threshold * 1.5:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.UNHEALTHY
        
        return HealthStatus.HEALTHY
    
    def _generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        recent_metrics = list(self.health_history)[-10:] if self.health_history else []
        
        # Calculate success rate
        total_calcs = len(recent_metrics)
        successful_calcs = sum(
            1 for m in recent_metrics
            if m['metrics'].get('status') == 'success'
        )
        success_rate = successful_calcs / total_calcs if total_calcs > 0 else 1.0
        
        return {
            'calculator': self.name,
            'status': self._health_status.value,
            'success_rate': success_rate,
            'consecutive_failures': self._consecutive_failures,
            'last_health_check': self._last_health_check.isoformat(),
            'metrics': {
                name: {
                    'value': metric.value,
                    'threshold': metric.threshold,
                    'status': metric.status.value
                }
                for name, metric in self.health_metrics.items()
            },
            'recovery_history': list(self.recovery_history)[-5:],
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate health recommendations."""
        recommendations = []
        
        if self._health_status == HealthStatus.CRITICAL:
            recommendations.append("Immediate intervention required")
            recommendations.append("Consider switching to backup calculator")
        elif self._health_status == HealthStatus.UNHEALTHY:
            recommendations.append("Monitor closely for further degradation")
            recommendations.append("Review recent error patterns")
        elif self._consecutive_failures > 1:
            recommendations.append("Investigate root cause of failures")
            recommendations.append("Consider adjusting calculation parameters")
        
        return recommendations
    
    async def _apply_fallback_strategy(
        self,
        strategy: str,
        args: tuple,
        kwargs: dict
    ) -> bool:
        """Apply fallback calculation strategy."""
        if strategy == "statistical_approximation":
            # Use statistical approximation based on historical data
            logger.info("Applying statistical approximation fallback")
            # Implementation would go here
            return True
        elif strategy == "last_known_good":
            # Use last known good result
            logger.info("Using last known good result")
            # Implementation would go here
            return True
        
        return False