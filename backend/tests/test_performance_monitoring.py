#!/usr/bin/env python3
"""
Test suite for performance monitoring enhancements
"""

import asyncio
import pytest
import time
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.a2a.core.performanceMonitor import (
    PerformanceMonitor, PerformanceMetrics, AlertThresholds,
    PrometheusMetrics, MetricsCollector, PerformanceAlert,
    create_performance_monitor
)
from app.a2a.core.performanceOptimizer import (
    PerformanceOptimizationMixin, AdaptiveThrottling, CacheOptimizer,
    OptimizationRecommendation
)


class TestMetricsCollector:
    """Test MetricsCollector functionality"""
    
    def test_collector_initialization(self):
        """Test collector initializes correctly"""
        collector = MetricsCollector("test_agent")
        assert collector.agent_id == "test_agent"
        assert collector.request_count == 0
        assert collector.error_count == 0
    
    def test_record_request(self):
        """Test request recording"""
        collector = MetricsCollector("test_agent")
        
        # Record successful request
        collector.record_request(0.5, success=True)
        assert collector.request_count == 1
        assert collector.error_count == 0
        assert len(collector.request_times) == 1
        
        # Record failed request
        collector.record_request(1.0, success=False)
        assert collector.request_count == 2
        assert collector.error_count == 1
        
    def test_performance_stats(self):
        """Test performance statistics calculation"""
        collector = MetricsCollector("test_agent")
        
        # Record some requests
        request_times = [0.1, 0.2, 0.3, 0.5, 1.0]  # seconds
        for duration in request_times:
            collector.record_request(duration, success=True)
        
        # Record one failure
        collector.record_request(0.2, success=False)
        
        stats = collector.get_performance_stats()
        
        assert stats["avg_response_time"] > 0
        assert stats["p95_response_time"] > 0
        assert stats["error_rate"] == 1/6  # 1 error out of 6 requests
        
    def test_cache_stats(self):
        """Test cache hit/miss tracking"""
        collector = MetricsCollector("test_agent")
        
        # Record cache hits and misses
        collector.record_cache_hit(True)
        collector.record_cache_hit(True)
        collector.record_cache_hit(False)
        
        stats = collector.get_performance_stats()
        assert stats["cache_hit_rate"] == 2/3  # 2 hits out of 3 operations


class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality"""
    
    def test_monitor_creation(self):
        """Test monitor creation and initialization"""
        monitor = PerformanceMonitor(
            agent_id="test_agent",
            metrics_port=8001
        )
        assert monitor.agent_id == "test_agent"
        assert not monitor.is_monitoring
        
    @pytest.mark.asyncio
    async def test_monitor_lifecycle(self):
        """Test monitor start and stop"""
        monitor = PerformanceMonitor(
            agent_id="test_agent",
            collection_interval=1  # 1 second for testing
        )
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor.is_monitoring
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.is_monitoring
        
    def test_alert_thresholds(self):
        """Test alert threshold checking"""
        thresholds = AlertThresholds(
            cpu_threshold=80.0,
            memory_threshold=85.0,
            response_time_threshold=1000.0,
            error_rate_threshold=0.05
        )
        
        monitor = PerformanceMonitor(
            agent_id="test_agent",
            alert_thresholds=thresholds
        )
        
        # Create metrics that should trigger alerts
        metrics = PerformanceMetrics(
            timestamp="2025-01-01T00:00:00Z",
            agent_id="test_agent",
            cpu_usage=90.0,  # Above threshold
            memory_usage=90.0,  # Above threshold
            request_count=100,
            response_time_avg=500.0,
            response_time_p95=1500.0,  # Above threshold
            error_rate=0.1,  # Above threshold
            throughput=10.0,
            active_connections=5,
            queue_size=10
        )
        
        alerts = monitor._check_alerts(metrics)
        
        # Should have alerts for CPU, memory, response time, and error rate
        assert len(alerts) >= 3
        alert_types = [alert.alert_type for alert in alerts]
        assert "high_cpu_usage" in alert_types
        assert "high_memory_usage" in alert_types
        assert "high_error_rate" in alert_types


class TestAdaptiveThrottling:
    """Test adaptive throttling functionality"""
    
    @pytest.mark.asyncio
    async def test_throttling_acquire(self):
        """Test throttling permission acquisition"""
        throttling = AdaptiveThrottling(base_rate=5)  # 5 requests per second
        
        # Should be able to acquire up to rate limit
        acquisitions = 0
        for _ in range(10):
            if await throttling.acquire():
                acquisitions += 1
        
        # Should not exceed rate limit
        assert acquisitions <= 5
        
    def test_rate_adjustment(self):
        """Test rate adjustment based on metrics"""
        throttling = AdaptiveThrottling(base_rate=100)
        initial_rate = throttling.current_rate
        
        # High CPU should decrease rate
        throttling.adjust_rate(cpu_usage=90.0, memory_usage=50.0, error_rate=0.01)
        assert throttling.current_rate < initial_rate
        
        # Low utilization should increase rate
        throttling.current_rate = initial_rate
        throttling.adjust_rate(cpu_usage=30.0, memory_usage=40.0, error_rate=0.001)
        assert throttling.current_rate >= initial_rate


class TestCacheOptimizer:
    """Test cache optimization functionality"""
    
    def test_cache_operations(self):
        """Test basic cache operations"""
        cache = CacheOptimizer(max_size=3)
        
        # Set values
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Get values
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("nonexistent") is None
        
    def test_cache_eviction(self):
        """Test cache eviction when at capacity"""
        cache = CacheOptimizer(max_size=2)
        
        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Add third item - should evict least valuable
        cache.set("key3", "value3")
        
        # Cache should only have 2 items
        assert len(cache.cache) == 2
        
    def test_hit_rate_calculation(self):
        """Test hit rate calculation"""
        cache = CacheOptimizer(max_size=5)
        
        # Set some values
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Mix of hits and misses
        cache.get("key1")  # Hit
        cache.get("key2")  # Hit
        cache.get("key3")  # Miss
        cache.get("key4")  # Miss
        
        hit_rate = cache.get_hit_rate()
        assert hit_rate == 0.5  # 2 hits out of 4 operations
        
    def test_size_optimization(self):
        """Test automatic size optimization"""
        cache = CacheOptimizer(max_size=1000)
        initial_size = cache.max_size
        
        # Simulate low hit rate
        for _ in range(100):
            cache._record_hit(False)  # All misses
        
        new_size = cache.optimize_size()
        assert new_size < initial_size  # Should reduce size


class MockAgent(PerformanceOptimizationMixin):
    """Mock agent for testing optimization mixin"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        super().__init__()


class TestPerformanceOptimizationMixin:
    """Test performance optimization mixin"""
    
    def test_mixin_initialization(self):
        """Test mixin initializes correctly"""
        agent = MockAgent("test_agent")
        assert agent._optimization_enabled is True
        assert agent._cache_optimizer is not None
        assert agent._adaptive_throttling is not None
        
    @pytest.mark.asyncio
    async def test_cached_operations(self):
        """Test cached operations"""
        agent = MockAgent("test_agent")
        
        # Test factory function
        async def factory():
            return "computed_value"
        
        # First call should compute
        result1 = await agent.cached_get("test_key", factory)
        assert result1 == "computed_value"
        
        # Second call should return cached value
        result2 = await agent.cached_get("test_key")
        assert result2 == "computed_value"
        
    @pytest.mark.asyncio
    async def test_throttled_operations(self):
        """Test throttled operations"""
        agent = MockAgent("test_agent")
        
        # Set low throttle rate for testing
        agent._adaptive_throttling.current_rate = 1  # 1 per second
        
        async def test_operation():
            return "operation_result"
        
        # First operation should succeed immediately
        start_time = time.time()
        result = await agent.throttled_operation(test_operation)
        duration = time.time() - start_time
        
        assert result == "operation_result"
        assert duration < 0.1  # Should be immediate
        
    def test_performance_summary(self):
        """Test performance summary generation"""
        agent = MockAgent("test_agent")
        
        summary = agent.get_performance_summary()
        
        assert "agent_id" in summary
        assert "cache_stats" in summary
        assert "throttling_stats" in summary
        assert summary["agent_id"] == "test_agent"
        
    @pytest.mark.asyncio
    async def test_performance_analysis(self):
        """Test performance analysis"""
        agent = MockAgent("test_agent")
        
        # Mock performance monitor
        mock_monitor = Mock()
        mock_metrics = PerformanceMetrics(
            timestamp="2025-01-01T00:00:00Z",
            agent_id="test_agent",
            cpu_usage=75.0,
            memory_usage=70.0,
            request_count=100,
            response_time_avg=500.0,
            response_time_p95=800.0,
            error_rate=0.02,
            throughput=10.0,
            active_connections=5,
            queue_size=2
        )
        mock_monitor.get_current_metrics.return_value = mock_metrics
        mock_monitor.get_metrics_history.return_value = [mock_metrics]
        
        agent._performance_monitor = mock_monitor
        
        analysis = await agent.run_performance_analysis()
        
        assert "timestamp" in analysis
        assert "agent_id" in analysis
        assert "performance_score" in analysis
        assert analysis["agent_id"] == "test_agent"
        assert 0 <= analysis["performance_score"] <= 100


@pytest.mark.asyncio
async def test_integration():
    """Integration test of performance monitoring system"""
    # Create performance monitor
    monitor = create_performance_monitor(
        agent_id="integration_test_agent",
        metrics_port=8002
    )
    
    # Create agent with optimization
    agent = MockAgent("integration_test_agent")
    agent.enable_performance_monitoring()
    
    # Simulate some activity
    for i in range(10):
        monitor.record_request("test_method", f"20{i % 3}", 0.1 + (i % 5) * 0.1)
        await asyncio.sleep(0.01)
    
    # Get metrics
    current_metrics = monitor.get_current_metrics()
    assert current_metrics.agent_id == "integration_test_agent"
    assert current_metrics.request_count >= 10
    
    # Run performance analysis
    analysis = await agent.run_performance_analysis()
    assert analysis["agent_id"] == "integration_test_agent"
    
    # Cleanup
    monitor.stop_monitoring()


def test_create_performance_monitor_registry():
    """Test performance monitor registry"""
    # Create multiple monitors
    monitor1 = create_performance_monitor("agent1", metrics_port=8003)
    monitor2 = create_performance_monitor("agent2", metrics_port=8004)
    
    # Should be registered
    from app.a2a.core.performanceMonitor import get_performance_monitor
    
    assert get_performance_monitor("agent1") == monitor1
    assert get_performance_monitor("agent2") == monitor2
    assert get_performance_monitor("nonexistent") is None


if __name__ == "__main__":
    # Run tests
    print("Running performance monitoring tests...")
    
    # Run sync tests
    test_instance = TestMetricsCollector()
    test_instance.test_collector_initialization()
    test_instance.test_record_request()
    test_instance.test_performance_stats()
    test_instance.test_cache_stats()
    print("✅ MetricsCollector tests passed")
    
    test_instance = TestAdaptiveThrottling()
    test_instance.test_rate_adjustment()
    print("✅ AdaptiveThrottling tests passed")
    
    test_instance = TestCacheOptimizer()
    test_instance.test_cache_operations()
    test_instance.test_cache_eviction()
    test_instance.test_hit_rate_calculation()
    test_instance.test_size_optimization()
    print("✅ CacheOptimizer tests passed")
    
    test_instance = TestPerformanceOptimizationMixin()
    test_instance.test_mixin_initialization()
    test_instance.test_performance_summary()
    print("✅ PerformanceOptimizationMixin tests passed")
    
    test_create_performance_monitor_registry()
    print("✅ Performance monitor registry tests passed")
    
    # Run async tests
    async def run_async_tests():
        test_instance = TestPerformanceMonitor()
        await test_instance.test_monitor_lifecycle()
        print("✅ PerformanceMonitor tests passed")
        
        test_instance = TestAdaptiveThrottling()
        await test_instance.test_throttling_acquire()
        print("✅ AdaptiveThrottling async tests passed")
        
        test_instance = TestPerformanceOptimizationMixin()
        await test_instance.test_cached_operations()
        await test_instance.test_throttled_operations()
        await test_instance.test_performance_analysis()
        print("✅ PerformanceOptimizationMixin async tests passed")
        
        await test_integration()
        print("✅ Integration tests passed")
    
    asyncio.run(run_async_tests())
    
    print("\n🎉 All performance monitoring tests passed!")
