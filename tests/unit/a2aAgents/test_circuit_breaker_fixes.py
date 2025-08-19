#!/usr/bin/env python3
"""
Simple test script for circuit breaker and trust system fixes
"""

import asyncio
import sys
import os

# Add paths for imports
sys.path.append('/Users/apple/projects/a2a/a2aAgents/backend')
sys.path.append('/Users/apple/projects/a2a/a2aAgents/backend/app')

# Test imports
try:
    from app.a2a.core.trustManager import TrustManager, sign_a2a_message
    from app.a2a.core.circuitBreaker import CircuitBreakerManager
    print("✅ Successfully imported trust and circuit breaker modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

async def test_trust_manager():
    """Test trust manager with various message types"""
    print("\n🔒 Testing Trust Manager...")
    
    trust_manager = TrustManager()
    
    # Test with simple dict
    simple_message = {"test": "message", "data": "value"}
    result = trust_manager.sign_message(simple_message, "test_agent")
    print(f"   Simple message signing: {result.get('signature', 'failed')}")
    
    # Test with complex dict containing potential unhashable types
    complex_message = {
        "data": {"nested": "value"},
        "timestamp": "2023-01-01T00:00:00",
        "metadata": {
            "tags": ["tag1", "tag2"],
            "config": {"setting": "value"}
        }
    }
    result = trust_manager.sign_message(complex_message, "test_agent")
    print(f"   Complex message signing: {result.get('signature', 'failed')}")
    
    # Test with message that might cause unhashable type error
    problematic_message = {
        "data": {"key": {"nested_dict": "value"}},
        "list_data": [1, 2, 3, {"inner": "dict"}],
        "metadata": {"set_data": {"a", "b", "c"}}  # This could cause issues
    }
    result = trust_manager.sign_message(problematic_message, "test_agent")
    print(f"   Problematic message signing: {result.get('signature', 'failed')}")

async def test_circuit_breaker():
    """Test circuit breaker functionality"""
    print("\n⚡ Testing Circuit Breaker...")
    
    manager = CircuitBreakerManager()
    breaker = manager.get_breaker("test_breaker", failure_threshold=2, success_threshold=1, timeout=1.0)
    
    # Test successful calls
    async def success_func():
        return "success"
    
    result = await breaker.call(success_func)
    print(f"   Successful call: {result}")
    print(f"   Breaker state: {breaker.state.value}")
    
    # Test failing calls
    failure_count = 0
    async def fail_func():
        nonlocal failure_count
        failure_count += 1
        if failure_count <= 2:
            raise Exception("Test failure")
        return "recovered"
    
    try:
        await breaker.call(fail_func)
    except Exception as e:
        print(f"   First failure handled: {type(e).__name__}")
    
    try:
        await breaker.call(fail_func)
    except Exception as e:
        print(f"   Second failure handled: {type(e).__name__}")
    
    print(f"   Breaker state after failures: {breaker.state.value}")
    
    # Test circuit open behavior
    try:
        await breaker.call(success_func)
    except Exception as e:
        print(f"   Circuit open error: {type(e).__name__}")
    
    # Wait for timeout and recovery
    await asyncio.sleep(1.1)
    
    # Test recovery
    try:
        result = await breaker.call(fail_func)  # Should succeed on 3rd call
        print(f"   Recovery successful: {result}")
        print(f"   Breaker state after recovery: {breaker.state.value}")
    except Exception as e:
        print(f"   Recovery failed: {e}")

async def main():
    """Run all tests"""
    print("🧪 Testing Circuit Breaker and Trust System Fixes")
    print("=" * 50)
    
    await test_trust_manager()
    await test_circuit_breaker()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())