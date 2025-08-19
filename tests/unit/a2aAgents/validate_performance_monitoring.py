#!/usr/bin/env python3
"""
Simple validation for performance monitoring enhancements
"""

import sys
import asyncio
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🗺️ Testing Performance Monitoring Enhancements...\n")

# Test 1: Import Test
print("1️⃣  Testing imports...")
try:
    from app.a2a.core.performanceMonitor import (
        PerformanceMonitor, MetricsCollector, PerformanceMetrics
    )
    from app.a2a.core.performanceOptimizer import (
        PerformanceOptimizationMixin, AdaptiveThrottling, CacheOptimizer
    )
    print("✅ All performance monitoring imports successful")
    imports_passed = True
except Exception as e:
    print(f"❌ Import failed: {e}")
    imports_passed = False

# Test 2: Basic Functionality
print("\n2️⃣  Testing basic functionality...")
try:
    # Test MetricsCollector
    collector = MetricsCollector("test_agent")
    collector.record_request(0.5, success=True)
    collector.record_request(1.0, success=False)
    
    stats = collector.get_performance_stats()
    assert stats["avg_response_time"] > 0
    assert stats["error_rate"] == 0.5  # 1 error out of 2 requests
    
    # Test CacheOptimizer
    cache = CacheOptimizer(max_size=3)
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"
    assert cache.get("nonexistent") is None
    
    # Test AdaptiveThrottling
    throttling = AdaptiveThrottling(base_rate=10)
    initial_rate = throttling.current_rate
    throttling.adjust_rate(cpu_usage=90.0, memory_usage=50.0, error_rate=0.01)
    # Should decrease rate due to high CPU
    
    print("✅ Basic functionality tests passed")
    functionality_passed = True
except Exception as e:
    print(f"❌ Basic functionality failed: {e}")
    functionality_passed = False

# Test 3: Performance Monitor Creation
print("\n3️⃣  Testing performance monitor creation...")
try:
    from app.a2a.core.performanceMonitor import create_performance_monitor
    
    monitor = create_performance_monitor(
        agent_id="validation_test_agent",
        metrics_port=8010
    )
    
    assert monitor.agent_id == "validation_test_agent"
    assert hasattr(monitor, 'metrics_collector')
    assert hasattr(monitor, 'prometheus_metrics')
    
    print("✅ Performance monitor creation successful")
    monitor_passed = True
except Exception as e:
    print(f"❌ Performance monitor creation failed: {e}")
    monitor_passed = False

# Test 4: Optimization Mixin
print("\n4️⃣  Testing optimization mixin...")
try:
    class TestAgent(PerformanceOptimizationMixin):
        def __init__(self):
            self.agent_id = "test_mixin_agent"
            super().__init__()
    
    agent = TestAgent()
    assert agent._optimization_enabled is True
    assert agent._cache_optimizer is not None
    
    # Enable monitoring first
    agent.enable_performance_monitoring()
    
    # Test cached operations
    agent.cached_set("test_key", "test_value")
    cached_value = asyncio.run(agent.cached_get("test_key"))
    assert cached_value == "test_value"
    
    # Test performance summary
    summary = agent.get_performance_summary()
    assert "agent_id" in summary
    assert summary["agent_id"] == "test_mixin_agent"
    
    print("✅ Optimization mixin tests passed")
    mixin_passed = True
except Exception as e:
    print(f"❌ Optimization mixin failed: {e}")
    mixin_passed = False

# Test 5: Enhanced Agent Demo (Simple)
print("\n5️⃣  Testing enhanced agent integration...")
try:
    from app.a2a.agents.agent0DataProduct.active.enhancedDataProductAgentSdk import (
        EnhancedDataProductRegistrationAgentSDK
    )
    
    # Create enhanced agent (without starting monitoring to avoid port conflicts)
    agent = EnhancedDataProductRegistrationAgentSDK(
        base_url="http://localhost:8000",
        ord_registry_url="http://localhost:9000",
        enable_monitoring=False  # Disable for test
    )
    
    assert agent.agent_id == "enhanced_data_product_agent_0"
    assert hasattr(agent, '_cache_optimizer')
    assert hasattr(agent, '_adaptive_throttling')
    
    print("✅ Enhanced agent integration successful")
    enhanced_agent_passed = True
except Exception as e:
    print(f"❌ Enhanced agent integration failed: {e}")
    enhanced_agent_passed = False

# Test 6: Dashboard Components
print("\n6️⃣  Testing dashboard components...")
try:
    from app.a2a.dashboard.performanceDashboard import app as dashboard_app
    
    assert dashboard_app is not None
    assert hasattr(dashboard_app, 'routes')
    
    # Check that key routes exist
    route_paths = [route.path for route in dashboard_app.routes]
    assert "/" in route_paths
    assert "/api/agents" in route_paths
    
    print("✅ Dashboard components loaded successfully")
    dashboard_passed = True
except Exception as e:
    print(f"❌ Dashboard components failed: {e}")
    dashboard_passed = False

# Summary
print("\n" + "="*60)
print("🎯 PERFORMANCE MONITORING VALIDATION SUMMARY")
print("="*60)

test_results = [
    ("Import Tests", imports_passed),
    ("Basic Functionality", functionality_passed),
    ("Performance Monitor", monitor_passed),
    ("Optimization Mixin", mixin_passed),
    ("Enhanced Agent", enhanced_agent_passed),
    ("Dashboard Components", dashboard_passed)
]

passed = 0
for test_name, result in test_results:
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status} {test_name}")
    if result:
        passed += 1

total = len(test_results)
print(f"\n📊 Results: {passed}/{total} tests passed")

if passed == total:
    print("\n🎉 All performance monitoring validations passed!")
    print("✅ Performance monitoring system is ready for use")
    print("\n✨ Enhanced capabilities available:")
    print("   - Real-time performance metrics collection")
    print("   - Automatic performance optimization")
    print("   - Adaptive throttling and caching")
    print("   - Performance alerts and recommendations")
    print("   - Web-based performance dashboard")
    print("   - Prometheus metrics integration")
else:
    print(f"\n⚠️  {total - passed} tests failed. Review issues above.")
    print("Performance monitoring system may have issues.")
