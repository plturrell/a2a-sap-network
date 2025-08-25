#!/usr/bin/env python3
"""
Performance Optimization Mixin for A2A Agents
Provides automatic performance optimization and tuning capabilities
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict, deque

from .performanceMonitor import PerformanceMetrics, AlertThresholds, monitor_performance

logger = logging.getLogger(__name__)


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""

    category: str  # "memory", "cpu", "io", "cache", "concurrency"
    priority: str  # "low", "medium", "high", "critical"
    description: str
    implementation: str  # How to implement the optimization
    expected_improvement: str  # Expected performance gain
    risk_level: str  # "low", "medium", "high"
    auto_applicable: bool = False  # Can be applied automatically


class AdaptiveThrottling:
    """Adaptive request throttling based on system performance"""

    def __init__(self, base_rate: int = 100, min_rate: int = 10, max_rate: int = 1000):
        self.base_rate = base_rate  # requests per second
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.current_rate = base_rate
        self.request_times = deque(maxlen=100)
        self.adjustment_factor = 0.1

    async def acquire(self) -> bool:
        """Acquire permission to make a request"""
        current_time = time.time()

        # Clean old request times
        while self.request_times and current_time - self.request_times[0] > 1.0:
            self.request_times.popleft()

        # Check if we're within rate limit
        if len(self.request_times) >= self.current_rate:
            return False

        self.request_times.append(current_time)
        return True

    def adjust_rate(self, cpu_usage: float, memory_usage: float, error_rate: float):
        """Adjust throttling rate based on system metrics"""
        adjustment = 0

        # Decrease rate if system is under stress
        if cpu_usage > 80 or memory_usage > 85:
            adjustment -= self.adjustment_factor
        elif error_rate > 0.05:
            adjustment -= self.adjustment_factor * 2

        # Increase rate if system is underutilized
        elif cpu_usage < 50 and memory_usage < 60 and error_rate < 0.01:
            adjustment += self.adjustment_factor

        if adjustment != 0:
            old_rate = self.current_rate
            self.current_rate = max(
                self.min_rate, min(self.max_rate, int(self.current_rate * (1 + adjustment)))
            )

            if self.current_rate != old_rate:
                logger.info(f"Adjusted throttling rate: {old_rate} -> {self.current_rate} req/s")


class CacheOptimizer:
    """Intelligent cache optimization"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.hit_rate_history = deque(maxlen=100)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
            self._record_hit(True)
            return self.cache[key]

        self._record_hit(False)
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with optional TTL"""
        # Evict old entries if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_least_valuable()

        self.cache[key] = value
        self.access_times[key] = time.time()
        self.access_counts[key] = 1

    def _record_hit(self, hit: bool):
        """Record cache hit/miss for analytics"""
        self.hit_rate_history.append(hit)

    def _evict_least_valuable(self):
        """Evict least valuable cache entry"""
        if not self.cache:
            return

        # Score based on recency and frequency
        current_time = time.time()
        scores = {}

        for key in self.cache:
            recency = current_time - self.access_times.get(key, current_time)
            frequency = self.access_counts.get(key, 1)
            # Lower score = less valuable
            scores[key] = frequency / (1 + recency)

        # Remove lowest scoring entry
        def get_score_value(x):
            return x[1]
        least_valuable = min(scores.items(), key=get_score_value)[0]
        del self.cache[least_valuable]
        del self.access_times[least_valuable]
        del self.access_counts[least_valuable]

    def get_hit_rate(self) -> float:
        """Get current cache hit rate"""
        if not self.hit_rate_history:
            return 0.0
        return sum(self.hit_rate_history) / len(self.hit_rate_history)

    def optimize_size(self) -> int:
        """Optimize cache size based on hit rate"""
        hit_rate = self.get_hit_rate()

        # If hit rate is low, consider reducing cache size
        if hit_rate < 0.5 and self.max_size > 100:
            self.max_size = max(100, int(self.max_size * 0.8))
            logger.info(
                f"Reduced cache size to {self.max_size} due to low hit rate ({hit_rate:.2%})"
            )

        # If hit rate is high and we're at capacity, consider increasing
        elif hit_rate > 0.8 and len(self.cache) >= self.max_size * 0.9:
            self.max_size = min(10000, int(self.max_size * 1.2))
            logger.info(
                f"Increased cache size to {self.max_size} due to high hit rate ({hit_rate:.2%})"
            )

        return self.max_size


class PerformanceOptimizationMixin:
    """Mixin class to add performance optimization to agents"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Performance monitoring
        self._performance_monitor = None
        self._optimization_enabled = True

        # Optimization components
        self._adaptive_throttling = AdaptiveThrottling()
        self._cache_optimizer = CacheOptimizer()
        self._optimization_recommendations = []

        # Performance state
        self._last_optimization_check = time.time()
        self._optimization_interval = 300  # 5 minutes
        self._performance_baseline = None

        logger.info(
            f"Performance optimization mixin initialized for {getattr(self, 'agent_id', 'unknown')}"
        )

    def enable_performance_monitoring(
        self, alert_thresholds: Optional[AlertThresholds] = None, metrics_port: Optional[int] = None
    ):
        """Enable performance monitoring for this agent"""
        if not hasattr(self, "agent_id"):
            logger.error("Agent must have agent_id attribute to enable monitoring")
            return

        from .performanceMonitor import create_performance_monitor

        self._performance_monitor = create_performance_monitor(
            agent_id=self.agent_id, alert_thresholds=alert_thresholds, metrics_port=metrics_port
        )

        # Add performance alert handler
        self._performance_monitor.add_alert_handler(self._handle_performance_alert)

        # Start monitoring
        self._performance_monitor.start_monitoring()

        logger.info(f"Performance monitoring enabled for {self.agent_id}")

    async def _handle_performance_alert(self, alert):
        """Handle performance alerts"""
        logger.warning(f"Performance alert [{alert.severity}]: {alert.message}")

        # Apply automatic optimizations based on alert type
        if alert.alert_type == "high_cpu_usage" and self._optimization_enabled:
            await self._optimize_cpu_usage()
        elif alert.alert_type == "high_memory_usage" and self._optimization_enabled:
            await self._optimize_memory_usage()
        elif alert.alert_type == "high_error_rate" and self._optimization_enabled:
            await self._optimize_error_handling()

    async def _optimize_cpu_usage(self):
        """Apply CPU usage optimizations"""
        logger.info("Applying CPU usage optimizations...")

        # Reduce throttling rate
        current_metrics = self._performance_monitor.get_current_metrics()
        self._adaptive_throttling.adjust_rate(
            current_metrics.cpu_usage, current_metrics.memory_usage, current_metrics.error_rate
        )

        # Add recommendation
        self._add_optimization_recommendation(
            OptimizationRecommendation(
                category="cpu",
                priority="high",
                description="High CPU usage detected - reduced request rate",
                implementation="Reduced throttling rate to decrease CPU load",
                expected_improvement="10-30% CPU usage reduction",
                risk_level="low",
                auto_applicable=True,
            )
        )

    async def _optimize_memory_usage(self):
        """Apply memory usage optimizations"""
        logger.info("Applying memory usage optimizations...")

        # Optimize cache size
        old_size = self._cache_optimizer.max_size
        new_size = self._cache_optimizer.optimize_size()

        if new_size < old_size:
            # Force cache cleanup
            while len(self._cache_optimizer.cache) > new_size:
                self._cache_optimizer._evict_least_valuable()

        self._add_optimization_recommendation(
            OptimizationRecommendation(
                category="memory",
                priority="high",
                description="High memory usage detected - optimized cache size",
                implementation=f"Adjusted cache size from {old_size} to {new_size}",
                expected_improvement="5-15% memory usage reduction",
                risk_level="low",
                auto_applicable=True,
            )
        )

    async def _optimize_error_handling(self):
        """Apply error handling optimizations"""
        logger.info("Applying error handling optimizations...")

        # Increase throttling backoff
        self._adaptive_throttling.current_rate = max(
            self._adaptive_throttling.min_rate, int(self._adaptive_throttling.current_rate * 0.7)
        )

        self._add_optimization_recommendation(
            OptimizationRecommendation(
                category="reliability",
                priority="high",
                description="High error rate detected - increased backoff",
                implementation="Reduced request rate to prevent error cascade",
                expected_improvement="30-50% error rate reduction",
                risk_level="low",
                auto_applicable=True,
            )
        )

    def _add_optimization_recommendation(self, recommendation: OptimizationRecommendation):
        """Add optimization recommendation"""
        recommendation.timestamp = datetime.utcnow().isoformat()
        self._optimization_recommendations.append(recommendation)

        # Keep only last 100 recommendations
        if len(self._optimization_recommendations) > 100:
            self._optimization_recommendations = self._optimization_recommendations[-100:]

    @monitor_performance("cached_operation")
    async def cached_get(self, key: str, factory_func: Optional[Callable] = None) -> Any:
        """Get value from cache or compute using factory function"""
        value = self._cache_optimizer.get(key)

        if value is not None:
            return value

        if factory_func:
            value = (
                await factory_func()
                if asyncio.iscoroutinefunction(factory_func)
                else factory_func()
            )
            self._cache_optimizer.set(key, value)
            return value

        return None

    def cached_set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in optimized cache"""
        self._cache_optimizer.set(key, value, ttl)

    async def throttled_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with adaptive throttling"""
        # Wait for throttling permission
        while not await self._adaptive_throttling.acquire():
            await asyncio.sleep(0.1)

        # Execute operation
        if asyncio.iscoroutinefunction(operation):
            return await operation(*args, **kwargs)
        else:
            return operation(*args, **kwargs)

    def get_optimization_recommendations(
        self, category: Optional[str] = None, priority: Optional[str] = None
    ) -> List[OptimizationRecommendation]:
        """Get optimization recommendations"""
        recommendations = self._optimization_recommendations

        if category:
            recommendations = [r for r in recommendations if r.category == category]

        if priority:
            recommendations = [r for r in recommendations if r.priority == priority]

        return recommendations

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            "agent_id": getattr(self, "agent_id", "unknown"),
            "monitoring_enabled": self._performance_monitor is not None,
            "optimization_enabled": self._optimization_enabled,
            "cache_stats": {
                "size": len(self._cache_optimizer.cache),
                "max_size": self._cache_optimizer.max_size,
                "hit_rate": self._cache_optimizer.get_hit_rate(),
            },
            "throttling_stats": {
                "current_rate": self._adaptive_throttling.current_rate,
                "base_rate": self._adaptive_throttling.base_rate,
                "recent_requests": len(self._adaptive_throttling.request_times),
            },
            "recommendations_count": len(self._optimization_recommendations),
        }

        if self._performance_monitor:
            current_metrics = self._performance_monitor.get_current_metrics()
            summary["current_metrics"] = current_metrics.to_dict()

        return summary

    async def run_performance_analysis(self) -> Dict[str, Any]:
        """Run comprehensive performance analysis"""
        logger.info(f"Running performance analysis for {getattr(self, 'agent_id', 'unknown')}")

        analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": getattr(self, "agent_id", "unknown"),
            "recommendations": [],
            "optimizations_applied": 0,
            "performance_score": 0.0,
        }

        if not self._performance_monitor:
            analysis["warning"] = "Performance monitoring not enabled"
            return analysis

        # Get recent metrics
        current_metrics = self._performance_monitor.get_current_metrics()
        recent_history = self._performance_monitor.get_metrics_history(hours=1)

        # Analyze performance patterns
        recommendations = await self._analyze_performance_patterns(current_metrics, recent_history)
        analysis["recommendations"] = [r.__dict__ for r in recommendations]

        # Calculate performance score (0-100)
        score = self._calculate_performance_score(current_metrics)
        analysis["performance_score"] = score

        # Apply automatic optimizations
        auto_recommendations = [r for r in recommendations if r.auto_applicable]
        for rec in auto_recommendations:
            try:
                await self._apply_recommendation(rec)
                analysis["optimizations_applied"] += 1
            except Exception as e:
                logger.error(f"Failed to apply optimization {rec.category}: {e}")

        return analysis

    async def _analyze_performance_patterns(
        self, current: PerformanceMetrics, history: List[PerformanceMetrics]
    ) -> List[OptimizationRecommendation]:
        """Analyze performance patterns and generate recommendations"""
        recommendations = []

        if not history:
            return recommendations

        # CPU usage trends
        cpu_values = [m.cpu_usage for m in history]
        avg_cpu = sum(cpu_values) / len(cpu_values)

        if avg_cpu > 70:
            recommendations.append(
                OptimizationRecommendation(
                    category="cpu",
                    priority="medium",
                    description=f"Average CPU usage ({avg_cpu:.1f}%) consistently high",
                    implementation="Consider implementing request queuing or load balancing",
                    expected_improvement="20-40% CPU usage reduction",
                    risk_level="medium",
                    auto_applicable=False,
                )
            )

        # Memory usage trends
        memory_values = [m.memory_usage for m in history]
        avg_memory = sum(memory_values) / len(memory_values)

        if avg_memory > 80:
            recommendations.append(
                OptimizationRecommendation(
                    category="memory",
                    priority="high",
                    description=f"Average memory usage ({avg_memory:.1f}%) consistently high",
                    implementation="Implement memory pooling or reduce cache sizes",
                    expected_improvement="15-30% memory usage reduction",
                    risk_level="medium",
                    auto_applicable=True,
                )
            )

        # Response time trends
        response_times = [m.response_time_avg for m in history]
        avg_response_time = sum(response_times) / len(response_times)

        if avg_response_time > 1000:  # 1 second
            recommendations.append(
                OptimizationRecommendation(
                    category="performance",
                    priority="high",
                    description=f"Average response time ({avg_response_time:.1f}ms) too high",
                    implementation="Implement caching, database connection pooling, or async processing",
                    expected_improvement="30-60% response time improvement",
                    risk_level="low",
                    auto_applicable=True,
                )
            )

        # Cache effectiveness
        if current.cache_hit_rate < 0.5:
            recommendations.append(
                OptimizationRecommendation(
                    category="cache",
                    priority="medium",
                    description=f"Cache hit rate ({current.cache_hit_rate:.1%}) is low",
                    implementation="Review caching strategy and key selection",
                    expected_improvement="10-25% performance improvement",
                    risk_level="low",
                    auto_applicable=True,
                )
            )

        return recommendations

    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall performance score (0-100)"""
        score = 100.0

        # CPU usage penalty
        if metrics.cpu_usage > 50:
            score -= min(50, (metrics.cpu_usage - 50) * 1.5)

        # Memory usage penalty
        if metrics.memory_usage > 60:
            score -= min(30, (metrics.memory_usage - 60) * 1.2)

        # Response time penalty
        if metrics.response_time_avg > 500:
            score -= min(25, (metrics.response_time_avg - 500) / 100 * 5)

        # Error rate penalty
        if metrics.error_rate > 0:
            score -= min(20, metrics.error_rate * 1000)  # 5% error = 50 point penalty

        # Cache hit rate bonus
        if metrics.cache_hit_rate > 0.8:
            score += 5

        return max(0.0, score)

    async def _apply_recommendation(self, recommendation: OptimizationRecommendation):
        """Apply optimization recommendation"""
        logger.info(f"Applying optimization: {recommendation.description}")

        if recommendation.category == "memory" and "cache" in recommendation.implementation:
            self._cache_optimizer.optimize_size()

        elif (
            recommendation.category == "performance" and "caching" in recommendation.implementation
        ):
            # Increase cache size for better hit rate
            self._cache_optimizer.max_size = min(10000, int(self._cache_optimizer.max_size * 1.5))

        # Record that we applied this optimization
        recommendation.applied_at = datetime.utcnow().isoformat()
        self._add_optimization_recommendation(recommendation)
