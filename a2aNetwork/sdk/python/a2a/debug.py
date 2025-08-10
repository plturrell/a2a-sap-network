"""
Debug and profiling utilities for A2A SDK
"""
import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics

from .types import Agent, Message, MessageStatus, MessageType

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for profiling"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    errors: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class DebugMode:
    """Debug mode for A2A SDK with tracing and profiling"""
    
    def __init__(self, client):
        self.client = client
        self.enabled = False
        self.trace_enabled = False
        self.profile_enabled = False
        self.breakpoints: Dict[str, Callable] = {}
        self.traces: List[Dict[str, Any]] = []
        self.metrics = PerformanceMetrics()
        self.response_times: List[float] = []
        
    def enable(self, trace: bool = True, profile: bool = True):
        """Enable debug mode with optional tracing and profiling"""
        self.enabled = True
        self.trace_enabled = trace
        self.profile_enabled = profile
        logger.info("Debug mode enabled")
        
        # Wrap client methods with debug hooks
        self._wrap_client_methods()
        
    def disable(self):
        """Disable debug mode"""
        self.enabled = False
        self.trace_enabled = False
        self.profile_enabled = False
        logger.info("Debug mode disabled")
        
    def set_breakpoint(self, method_name: str, condition: Callable = None):
        """Set a breakpoint on a specific method"""
        self.breakpoints[method_name] = condition or (lambda *args, **kwargs: True)
        logger.info(f"Breakpoint set on {method_name}")
        
    def clear_breakpoint(self, method_name: str):
        """Clear a breakpoint"""
        if method_name in self.breakpoints:
            del self.breakpoints[method_name]
            logger.info(f"Breakpoint cleared on {method_name}")
            
    def get_trace(self) -> List[Dict[str, Any]]:
        """Get execution trace"""
        return self.traces
        
    def get_metrics(self) -> PerformanceMetrics:
        """Get performance metrics"""
        if self.response_times:
            self.metrics.avg_response_time = statistics.mean(self.response_times)
            self.metrics.min_response_time = min(self.response_times)
            self.metrics.max_response_time = max(self.response_times)
            
            if len(self.response_times) >= 20:
                sorted_times = sorted(self.response_times)
                self.metrics.p95_response_time = sorted_times[int(len(sorted_times) * 0.95)]
                self.metrics.p99_response_time = sorted_times[int(len(sorted_times) * 0.99)]
                
        return self.metrics
        
    def _wrap_client_methods(self):
        """Wrap client methods with debug hooks"""
        # Wrap agent manager methods
        if hasattr(self.client, 'agents'):
            self._wrap_method(self.client.agents, 'register')
            self._wrap_method(self.client.agents, 'get')
            self._wrap_method(self.client.agents, 'update')
            
        # Wrap message manager methods
        if hasattr(self.client, 'messages'):
            self._wrap_method(self.client.messages, 'send')
            self._wrap_method(self.client.messages, 'get')
            self._wrap_method(self.client.messages, 'list')
            
    def _wrap_method(self, obj: Any, method_name: str):
        """Wrap a single method with debug hooks"""
        original_method = getattr(obj, method_name)
        
        async def wrapped_method(*args, **kwargs):
            start_time = time.time()
            method_path = f"{obj.__class__.__name__}.{method_name}"
            
            # Check breakpoint
            if method_path in self.breakpoints:
                condition = self.breakpoints[method_path]
                if condition(*args, **kwargs):
                    await self._handle_breakpoint(method_path, args, kwargs)
                    
            # Trace entry
            if self.trace_enabled:
                self._add_trace({
                    'type': 'method_entry',
                    'method': method_path,
                    'args': str(args),
                    'kwargs': str(kwargs),
                    'timestamp': datetime.now().isoformat()
                })
                
            try:
                # Execute original method
                result = await original_method(*args, **kwargs)
                
                # Profile success
                if self.profile_enabled:
                    response_time = time.time() - start_time
                    self._update_metrics(True, response_time)
                    
                # Trace exit
                if self.trace_enabled:
                    self._add_trace({
                        'type': 'method_exit',
                        'method': method_path,
                        'result': str(result),
                        'duration': time.time() - start_time,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                return result
                
            except Exception as e:
                # Profile failure
                if self.profile_enabled:
                    response_time = time.time() - start_time
                    self._update_metrics(False, response_time, str(e))
                    
                # Trace exception
                if self.trace_enabled:
                    self._add_trace({
                        'type': 'method_exception',
                        'method': method_path,
                        'exception': str(e),
                        'duration': time.time() - start_time,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                raise
                
        setattr(obj, method_name, wrapped_method)
        
    def _add_trace(self, trace_entry: Dict[str, Any]):
        """Add entry to trace log"""
        self.traces.append(trace_entry)
        
        # Keep only last 1000 entries
        if len(self.traces) > 1000:
            self.traces.pop(0)
            
    def _update_metrics(self, success: bool, response_time: float, error: str = None):
        """Update performance metrics"""
        self.metrics.total_requests += 1
        
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
            if error:
                self.metrics.errors.append({
                    'error': error,
                    'timestamp': datetime.now().isoformat()
                })
                
        self.metrics.total_response_time += response_time
        self.response_times.append(response_time)
        
        # Keep only last 1000 response times
        if len(self.response_times) > 1000:
            self.response_times.pop(0)
            
    async def _handle_breakpoint(self, method_path: str, args: tuple, kwargs: dict):
        """Handle breakpoint interaction"""
        logger.info(f"🔴 Breakpoint hit: {method_path}")
        logger.info(f"Args: {args}")
        logger.info(f"Kwargs: {kwargs}")
        
        # In a real implementation, this could pause execution
        # For now, we just log
        await asyncio.sleep(0.1)


class Profiler:
    """Performance profiler for A2A operations"""
    
    def __init__(self, client):
        self.client = client
        self.debug = DebugMode(client)
        
    async def benchmark(
        self,
        operation: Callable,
        iterations: int = 100,
        concurrent: int = 1,
        warmup: int = 10
    ) -> Dict[str, Any]:
        """Benchmark an operation"""
        logger.info(f"Starting benchmark: {iterations} iterations, {concurrent} concurrent")
        
        # Enable profiling
        self.debug.enable(trace=False, profile=True)
        
        # Warmup
        logger.info(f"Warming up with {warmup} iterations...")
        for _ in range(warmup):
            try:
                await operation()
            except Exception:
                pass
                
        # Reset metrics after warmup
        self.debug.metrics = PerformanceMetrics()
        self.debug.response_times = []
        
        # Run benchmark
        start_time = time.time()
        
        if concurrent == 1:
            # Sequential execution
            for i in range(iterations):
                try:
                    await operation()
                except Exception as e:
                    logger.error(f"Iteration {i} failed: {e}")
        else:
            # Concurrent execution
            for batch in range(0, iterations, concurrent):
                tasks = []
                for _ in range(min(concurrent, iterations - batch)):
                    tasks.append(operation())
                    
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Concurrent task {batch + i} failed: {result}")
                        
        total_time = time.time() - start_time
        
        # Get final metrics
        metrics = self.debug.get_metrics()
        
        # Calculate additional statistics
        throughput = iterations / total_time if total_time > 0 else 0
        
        results = {
            'iterations': iterations,
            'concurrent': concurrent,
            'total_time': total_time,
            'throughput': throughput,
            'success_rate': metrics.successful_requests / metrics.total_requests if metrics.total_requests > 0 else 0,
            'avg_response_time': metrics.avg_response_time,
            'min_response_time': metrics.min_response_time,
            'max_response_time': metrics.max_response_time,
            'p95_response_time': metrics.p95_response_time,
            'p99_response_time': metrics.p99_response_time,
            'errors': len(metrics.errors)
        }
        
        # Disable profiling
        self.debug.disable()
        
        return results
        
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze benchmark results and provide recommendations"""
        analysis = {
            'performance_rating': 'unknown',
            'bottlenecks': [],
            'recommendations': []
        }
        
        # Rate performance
        if results['success_rate'] >= 0.99 and results['avg_response_time'] < 0.1:
            analysis['performance_rating'] = 'excellent'
        elif results['success_rate'] >= 0.95 and results['avg_response_time'] < 0.5:
            analysis['performance_rating'] = 'good'
        elif results['success_rate'] >= 0.90 and results['avg_response_time'] < 1.0:
            analysis['performance_rating'] = 'acceptable'
        else:
            analysis['performance_rating'] = 'needs improvement'
            
        # Identify bottlenecks
        if results['success_rate'] < 0.95:
            analysis['bottlenecks'].append('High error rate')
            analysis['recommendations'].append('Review error logs and improve error handling')
            
        if results['avg_response_time'] > 1.0:
            analysis['bottlenecks'].append('Slow response times')
            analysis['recommendations'].append('Optimize message processing logic')
            
        if results['p99_response_time'] > results['avg_response_time'] * 10:
            analysis['bottlenecks'].append('High latency variance')
            analysis['recommendations'].append('Investigate timeout issues or resource contention')
            
        if results['throughput'] < 10:
            analysis['bottlenecks'].append('Low throughput')
            analysis['recommendations'].append('Consider increasing concurrency or optimizing operations')
            
        return analysis


class MessageTracer:
    """Trace messages through the A2A network"""
    
    def __init__(self, client):
        self.client = client
        self.traces: Dict[str, List[Dict[str, Any]]] = {}
        
    async def trace_message(self, message_id: str) -> Dict[str, Any]:
        """Trace a message through its lifecycle"""
        trace_data = {
            'message_id': message_id,
            'hops': [],
            'total_time': 0,
            'status': 'unknown'
        }
        
        try:
            # Get message details
            message = await self.client.messages.get(message_id)
            
            trace_data['status'] = message.status
            trace_data['from_agent'] = message.from_agent
            trace_data['to_agent'] = message.to_agent
            trace_data['created_at'] = message.created_at
            
            # Trace routing path
            # This would integrate with blockchain events
            # For now, we simulate
            trace_data['hops'] = [
                {
                    'agent': message.from_agent,
                    'action': 'sent',
                    'timestamp': message.created_at
                },
                {
                    'agent': message.to_agent,
                    'action': 'received',
                    'timestamp': message.created_at + 1
                }
            ]
            
            if message.status == MessageStatus.DELIVERED:
                trace_data['hops'].append({
                    'agent': message.to_agent,
                    'action': 'processed',
                    'timestamp': message.created_at + 2
                })
                
        except Exception as e:
            trace_data['error'] = str(e)
            
        return trace_data
        
    async def trace_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Trace an entire workflow execution"""
        workflow_trace = {
            'workflow_id': workflow_id,
            'messages': [],
            'agents_involved': set(),
            'total_messages': 0,
            'successful_messages': 0,
            'failed_messages': 0
        }
        
        # Get all messages for workflow
        # This would query blockchain events
        # For now, we return a sample structure
        
        return {
            'workflow_id': workflow_id,
            'agents_involved': ['agent1', 'agent2', 'agent3'],
            'total_messages': 10,
            'successful_messages': 9,
            'failed_messages': 1,
            'avg_message_time': 0.5,
            'total_workflow_time': 5.0
        }