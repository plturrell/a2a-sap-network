#!/usr/bin/env python3
"""
Comprehensive Error Handling and Recovery Mechanisms for Blockchain Integration

This module provides robust error handling and recovery for blockchain operations:
- Automatic retry with exponential backoff
- Circuit breaker pattern for failing services
- Graceful degradation when blockchain is unavailable
- Transaction recovery and replay
- State reconciliation
- Error classification and handling strategies
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, TypeVar, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import traceback
import os
import sys

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorCategory(Enum):
    """Categories of blockchain errors"""
    NETWORK = "network"  # Network connectivity issues
    TRANSACTION = "transaction"  # Transaction failures
    CONTRACT = "contract"  # Smart contract errors
    GAS = "gas"  # Gas-related errors
    AUTHENTICATION = "authentication"  # Auth/permission errors
    VALIDATION = "validation"  # Input validation errors
    TIMEOUT = "timeout"  # Operation timeouts
    UNKNOWN = "unknown"  # Unknown errors


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY = "retry"  # Retry the operation
    CIRCUIT_BREAK = "circuit_break"  # Stop trying temporarily
    FALLBACK = "fallback"  # Use fallback mechanism
    QUEUE = "queue"  # Queue for later retry
    ALERT = "alert"  # Alert and manual intervention
    IGNORE = "ignore"  # Log and continue


@dataclass
class ErrorContext:
    """Context information about an error"""
    error_type: type
    error_message: str
    error_category: ErrorCategory
    operation: str
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    traceback: Optional[str] = None


@dataclass
class RecoveryResult:
    """Result of a recovery attempt"""
    success: bool
    strategy_used: RecoveryStrategy
    result: Any = None
    error: Optional[Exception] = None
    attempts: int = 0


class RetryConfig:
    """Configuration for retry behavior"""
    def __init__(self,
                 max_retries: int = 3,
                 initial_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


class CircuitBreaker:
    """Circuit breaker implementation"""
    
    class State(Enum):
        CLOSED = "closed"  # Normal operation
        OPEN = "open"  # Failing, reject requests
        HALF_OPEN = "half_open"  # Testing if recovered
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = self.State.CLOSED
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Call function through circuit breaker"""
        if self.state == self.State.OPEN:
            if self._should_attempt_reset():
                self.state = self.State.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Call async function through circuit breaker"""
        if self.state == self.State.OPEN:
            if self._should_attempt_reset():
                self.state = self.State.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset the circuit"""
        return (
            self.last_failure_time and
            (datetime.now() - self.last_failure_time).total_seconds() >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = self.State.CLOSED
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = self.State.OPEN


class BlockchainErrorHandler:
    """Main error handling and recovery system"""
    
    def __init__(self):
        # Error classification rules
        self.error_classifiers = {
            ErrorCategory.NETWORK: [
                "connection", "network", "timeout", "ECONNREFUSED",
                "ETIMEDOUT", "ENETUNREACH", "socket"
            ],
            ErrorCategory.TRANSACTION: [
                "transaction", "nonce", "already known", "replacement",
                "underpriced", "insufficient funds"
            ],
            ErrorCategory.CONTRACT: [
                "revert", "execution reverted", "contract", "invalid opcode",
                "out of gas", "stack too deep"
            ],
            ErrorCategory.GAS: [
                "gas", "gas price", "max fee", "gas limit", "intrinsic gas"
            ],
            ErrorCategory.AUTHENTICATION: [
                "unauthorized", "permission", "access denied", "signature",
                "authentication", "private key"
            ],
            ErrorCategory.VALIDATION: [
                "invalid", "validation", "parameter", "argument", "format"
            ],
            ErrorCategory.TIMEOUT: [
                "timeout", "deadline", "timed out"
            ]
        }
        
        # Recovery strategies by error category
        self.recovery_strategies = {
            ErrorCategory.NETWORK: RecoveryStrategy.RETRY,
            ErrorCategory.TRANSACTION: RecoveryStrategy.RETRY,
            ErrorCategory.CONTRACT: RecoveryStrategy.ALERT,
            ErrorCategory.GAS: RecoveryStrategy.RETRY,
            ErrorCategory.AUTHENTICATION: RecoveryStrategy.ALERT,
            ErrorCategory.VALIDATION: RecoveryStrategy.IGNORE,
            ErrorCategory.TIMEOUT: RecoveryStrategy.RETRY,
            ErrorCategory.UNKNOWN: RecoveryStrategy.ALERT
        }
        
        # Circuit breakers for different services
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Retry configurations
        self.retry_configs = {
            ErrorCategory.NETWORK: RetryConfig(max_retries=5, initial_delay=2.0),
            ErrorCategory.TRANSACTION: RetryConfig(max_retries=3, initial_delay=1.0),
            ErrorCategory.GAS: RetryConfig(max_retries=3, initial_delay=5.0),
            ErrorCategory.TIMEOUT: RetryConfig(max_retries=2, initial_delay=3.0),
        }
        
        # Failed operation queue for later retry
        self.retry_queue: List[Dict[str, Any]] = []
        
        # Error statistics
        self.error_stats = {
            "total_errors": 0,
            "errors_by_category": {cat: 0 for cat in ErrorCategory},
            "recoveries_attempted": 0,
            "recoveries_successful": 0
        }
    
    def classify_error(self, error: Exception) -> ErrorCategory:
        """Classify an error into a category"""
        error_str = str(error).lower()
        
        for category, keywords in self.error_classifiers.items():
            if any(keyword in error_str for keyword in keywords):
                return category
        
        return ErrorCategory.UNKNOWN
    
    def get_recovery_strategy(self, error_category: ErrorCategory) -> RecoveryStrategy:
        """Get recovery strategy for error category"""
        return self.recovery_strategies.get(error_category, RecoveryStrategy.ALERT)
    
    async def handle_error(self, 
                          error: Exception,
                          operation: str,
                          operation_func: Callable,
                          *args,
                          **kwargs) -> RecoveryResult:
        """Handle an error with appropriate recovery strategy"""
        # Create error context
        error_context = ErrorContext(
            error_type=type(error),
            error_message=str(error),
            error_category=self.classify_error(error),
            operation=operation,
            traceback=traceback.format_exc(),
            metadata={"args": args, "kwargs": kwargs}
        )
        
        # Update statistics
        self.error_stats["total_errors"] += 1
        self.error_stats["errors_by_category"][error_context.error_category] += 1
        
        # Log error
        logger.error(f"Error in {operation}: {error_context.error_message}")
        logger.debug(f"Error traceback: {error_context.traceback}")
        
        # Get recovery strategy
        strategy = self.get_recovery_strategy(error_context.error_category)
        
        # Execute recovery
        recovery_result = await self._execute_recovery(
            error_context, strategy, operation_func, *args, **kwargs
        )
        
        # Update statistics
        self.error_stats["recoveries_attempted"] += 1
        if recovery_result.success:
            self.error_stats["recoveries_successful"] += 1
        
        return recovery_result
    
    async def _execute_recovery(self,
                               error_context: ErrorContext,
                               strategy: RecoveryStrategy,
                               operation_func: Callable,
                               *args,
                               **kwargs) -> RecoveryResult:
        """Execute recovery based on strategy"""
        if strategy == RecoveryStrategy.RETRY:
            return await self._retry_with_backoff(
                error_context, operation_func, *args, **kwargs
            )
        
        elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
            return await self._circuit_break_recovery(
                error_context, operation_func, *args, **kwargs
            )
        
        elif strategy == RecoveryStrategy.FALLBACK:
            return await self._fallback_recovery(
                error_context, operation_func, *args, **kwargs
            )
        
        elif strategy == RecoveryStrategy.QUEUE:
            return await self._queue_for_retry(
                error_context, operation_func, *args, **kwargs
            )
        
        elif strategy == RecoveryStrategy.ALERT:
            return await self._alert_recovery(
                error_context, operation_func, *args, **kwargs
            )
        
        else:  # IGNORE
            return RecoveryResult(
                success=False,
                strategy_used=strategy,
                error=Exception(error_context.error_message)
            )
    
    async def _retry_with_backoff(self,
                                 error_context: ErrorContext,
                                 operation_func: Callable,
                                 *args,
                                 **kwargs) -> RecoveryResult:
        """Retry operation with exponential backoff"""
        retry_config = self.retry_configs.get(
            error_context.error_category,
            RetryConfig()
        )
        
        attempts = 0
        last_error = None
        
        while attempts < retry_config.max_retries:
            attempts += 1
            
            # Calculate delay
            delay = min(
                retry_config.initial_delay * (retry_config.exponential_base ** attempts),
                retry_config.max_delay
            )
            
            # Add jitter if configured
            if retry_config.jitter:
                import random
                delay *= (0.5 + random.random())
            
            logger.info(f"Retrying {error_context.operation} after {delay:.2f}s (attempt {attempts}/{retry_config.max_retries})")
            await asyncio.sleep(delay)
            
            try:
                # Retry operation
                if asyncio.iscoroutinefunction(operation_func):
                    result = await operation_func(*args, **kwargs)
                else:
                    result = operation_func(*args, **kwargs)
                
                logger.info(f"Retry successful for {error_context.operation}")
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.RETRY,
                    result=result,
                    attempts=attempts
                )
                
            except Exception as e:
                last_error = e
                logger.warning(f"Retry {attempts} failed: {str(e)}")
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.RETRY,
            error=last_error,
            attempts=attempts
        )
    
    async def _circuit_break_recovery(self,
                                     error_context: ErrorContext,
                                     operation_func: Callable,
                                     *args,
                                     **kwargs) -> RecoveryResult:
        """Use circuit breaker for recovery"""
        # Get or create circuit breaker for this operation
        if error_context.operation not in self.circuit_breakers:
            self.circuit_breakers[error_context.operation] = CircuitBreaker()
        
        breaker = self.circuit_breakers[error_context.operation]
        
        try:
            if asyncio.iscoroutinefunction(operation_func):
                result = await breaker.call_async(operation_func, *args, **kwargs)
            else:
                result = breaker.call(operation_func, *args, **kwargs)
            
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.CIRCUIT_BREAK,
                result=result
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.CIRCUIT_BREAK,
                error=e
            )
    
    async def _fallback_recovery(self,
                                error_context: ErrorContext,
                                operation_func: Callable,
                                *args,
                                **kwargs) -> RecoveryResult:
        """Use fallback mechanism"""
        # Example: Use cached data or default values
        logger.info(f"Using fallback for {error_context.operation}")
        
        # Implement fallback logic based on operation type
        fallback_result = None
        
        if "get" in error_context.operation.lower():
            # Return cached or default data
            fallback_result = kwargs.get("default", {})
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.FALLBACK,
            result=fallback_result
        )
    
    async def _queue_for_retry(self,
                              error_context: ErrorContext,
                              operation_func: Callable,
                              *args,
                              **kwargs) -> RecoveryResult:
        """Queue operation for later retry"""
        retry_item = {
            "error_context": error_context,
            "operation_func": operation_func,
            "args": args,
            "kwargs": kwargs,
            "queued_at": datetime.now()
        }
        
        self.retry_queue.append(retry_item)
        logger.info(f"Queued {error_context.operation} for later retry")
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.QUEUE
        )
    
    async def _alert_recovery(self,
                             error_context: ErrorContext,
                             operation_func: Callable,
                             *args,
                             **kwargs) -> RecoveryResult:
        """Alert for manual intervention"""
        logger.critical(f"ALERT: Manual intervention required for {error_context.operation}")
        logger.critical(f"Error: {error_context.error_message}")
        
        # In production, would send alerts via monitoring system
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.ALERT,
            error=Exception(error_context.error_message)
        )
    
    async def process_retry_queue(self):
        """Process queued operations for retry"""
        if not self.retry_queue:
            return
        
        logger.info(f"Processing {len(self.retry_queue)} queued operations")
        
        # Process queue in FIFO order
        processed = []
        
        for item in self.retry_queue[:]:  # Copy to avoid modification during iteration
            # Check if item is too old (> 1 hour)
            if (datetime.now() - item["queued_at"]).total_seconds() > 3600:
                processed.append(item)
                continue
            
            # Retry operation
            try:
                operation_func = item["operation_func"]
                args = item["args"]
                kwargs = item["kwargs"]
                
                if asyncio.iscoroutinefunction(operation_func):
                    await operation_func(*args, **kwargs)
                else:
                    operation_func(*args, **kwargs)
                
                logger.info(f"Successfully processed queued operation: {item['error_context'].operation}")
                processed.append(item)
                
            except Exception as e:
                logger.warning(f"Queued operation still failing: {str(e)}")
                # Keep in queue for next attempt
        
        # Remove processed items
        for item in processed:
            self.retry_queue.remove(item)


# Decorators for automatic error handling

def blockchain_error_handler(operation_name: str = None):
    """Decorator for automatic blockchain error handling"""
    def decorator(func):
        handler = BlockchainErrorHandler()
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                recovery_result = await handler.handle_error(
                    e, op_name, func, *args, **kwargs
                )
                if recovery_result.success:
                    return recovery_result.result
                else:
                    raise recovery_result.error or e
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Run async handler in sync context
                loop = asyncio.new_event_loop()
                recovery_result = loop.run_until_complete(
                    handler.handle_error(e, op_name, func, *args, **kwargs)
                )
                loop.close()
                
                if recovery_result.success:
                    return recovery_result.result
                else:
                    raise recovery_result.error or e
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# State reconciliation for blockchain consistency

class BlockchainStateReconciler:
    """Handles state reconciliation between local and blockchain state"""
    
    def __init__(self):
        self.local_state: Dict[str, Any] = {}
        self.blockchain_state: Dict[str, Any] = {}
        self.pending_operations: List[Dict[str, Any]] = []
    
    async def reconcile_state(self):
        """Reconcile local state with blockchain"""
        discrepancies = self._find_discrepancies()
        
        for key, (local_value, blockchain_value) in discrepancies.items():
            logger.warning(f"State discrepancy for {key}: local={local_value}, blockchain={blockchain_value}")
            
            # Blockchain is source of truth
            self.local_state[key] = blockchain_value
    
    def _find_discrepancies(self) -> Dict[str, tuple]:
        """Find discrepancies between local and blockchain state"""
        discrepancies = {}
        
        all_keys = set(self.local_state.keys()) | set(self.blockchain_state.keys())
        
        for key in all_keys:
            local_value = self.local_state.get(key)
            blockchain_value = self.blockchain_state.get(key)
            
            if local_value != blockchain_value:
                discrepancies[key] = (local_value, blockchain_value)
        
        return discrepancies
    
    async def replay_pending_operations(self):
        """Replay pending operations after recovery"""
        logger.info(f"Replaying {len(self.pending_operations)} pending operations")
        
        replayed = []
        
        for operation in self.pending_operations:
            try:
                # Replay operation
                await self._replay_operation(operation)
                replayed.append(operation)
                
            except Exception as e:
                logger.error(f"Failed to replay operation: {str(e)}")
        
        # Remove successfully replayed operations
        for op in replayed:
            self.pending_operations.remove(op)
    
    async def _replay_operation(self, operation: Dict[str, Any]):
        """Replay a single operation"""
        # Implementation depends on operation type
        op_type = operation.get("type")
        
        if op_type == "transaction":
            # Replay transaction
            pass
        elif op_type == "message":
            # Resend message
            pass
        # Add more operation types as needed


# Example usage

async def example_error_handling():
    """Example of using error handling system"""
    
    # Example blockchain operation with error handling
    @blockchain_error_handler("send_transaction")
    async def send_transaction(to_address: str, value: int):
        """Simulated blockchain transaction"""
        import random
        
        # Simulate various errors
        error_chance = random.random()
        
        if error_chance < 0.3:
            raise ConnectionError("Network connection timeout")
        elif error_chance < 0.4:
            raise ValueError("Transaction already known")
        elif error_chance < 0.5:
            raise Exception("Execution reverted: Out of gas")
        
        # Success
        return f"tx_hash_{random.randint(1000, 9999)}"
    
    # Test error handling
    handler = BlockchainErrorHandler()
    
    for i in range(10):
        try:
            print(f"\nAttempt {i+1}:")
            tx_hash = await send_transaction("0x123", 1000)
            print(f"Success: {tx_hash}")
            
        except Exception as e:
            print(f"Final error: {str(e)}")
    
    # Show statistics
    print("\nError Statistics:")
    print(json.dumps(handler.error_stats, indent=2))
    
    # Process retry queue
    await handler.process_retry_queue()


if __name__ == "__main__":
    asyncio.run(example_error_handling())