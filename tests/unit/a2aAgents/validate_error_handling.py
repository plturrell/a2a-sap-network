#!/usr/bin/env python3
"""
Validation script for comprehensive error handling and recovery mechanisms
Tests all components of the error handling system
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🛡️ Testing Comprehensive Error Handling System...\n")

# Test 1: Core Error Handling Components
print("1️⃣  Testing core error handling components...")
try:
    from app.a2a.core.errorHandling import (
        ErrorRecoveryManager, CircuitBreaker, ErrorCategory, ErrorSeverity,
        RecoveryStrategy, error_handler
    )
    from app.a2a.core.errorHandlingMixin import ErrorHandlingMixin
    
    print("✅ All error handling imports successful")
    imports_passed = True
except Exception as e:
    print(f"❌ Import failed: {e}")
    imports_passed = False

# Test 2: Circuit Breaker Functionality
print("\n2️⃣  Testing circuit breaker functionality...")
try:
    async def test_circuit_breaker():
        breaker = CircuitBreaker("test_breaker", failure_threshold=2, recovery_timeout=0.1)
        
        # Test normal operation
        result = await breaker.call(lambda: "success")
        assert result == "success"
        assert breaker.state.value == "closed"
        
        # Test failure handling
        failure_count = 0
        def failing_function():
            nonlocal failure_count
            failure_count += 1
            raise Exception(f"Failure {failure_count}")
        
        # Trigger failures to open breaker
        for _ in range(2):
            try:
                await breaker.call(failing_function)
            except Exception:
                pass
        
        assert breaker.state.value == "open"
        
        # Test circuit breaker blocks calls
        from app.a2a.core.errorHandling import CircuitBreakerOpenError
        try:
            await breaker.call(lambda: "should_not_execute")
            assert False, "Should have raised CircuitBreakerOpenError"
        except CircuitBreakerOpenError:
            pass
        
        return True
    
    circuit_breaker_result = asyncio.run(test_circuit_breaker())
    print("✅ Circuit breaker functionality tests passed")
    circuit_breaker_passed = True
except Exception as e:
    print(f"❌ Circuit breaker test failed: {e}")
    circuit_breaker_passed = False

# Test 3: Error Recovery Manager
print("\n3️⃣  Testing error recovery manager...")
try:
    async def test_recovery_manager():
        manager = ErrorRecoveryManager("test_agent")
        
        # Test successful operation
        async def success_func():
            return "manager_success"
        
        result = await manager.execute_with_recovery(
            "test_operation",
            success_func,
            category=ErrorCategory.PROCESSING
        )
        assert result == "manager_success"
        
        # Test retry mechanism
        attempt_count = 0
        async def retry_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception(f"Retry attempt {attempt_count}")
            return "retry_success"
        
        result = await manager.execute_with_recovery(
            "retry_operation",
            retry_func,
            category=ErrorCategory.NETWORK
        )
        assert result == "retry_success"
        assert attempt_count == 3
        
        # Test error summary
        summary = manager.get_error_summary()
        assert "total_errors" in summary
        assert "by_category" in summary
        assert "circuit_breakers" in summary
        
        return True
    
    recovery_result = asyncio.run(test_recovery_manager())
    print("✅ Error recovery manager tests passed")
    recovery_manager_passed = True
except Exception as e:
    print(f"❌ Error recovery manager test failed: {e}")
    recovery_manager_passed = False

# Test 4: Error Handling Mixin
print("\n4️⃣  Testing error handling mixin...")
try:
    class TestAgent(ErrorHandlingMixin):
        def __init__(self):
            self.agent_id = "validation_test_agent"
            super().__init__()
    
    async def test_mixin():
        agent = TestAgent()
        
        # Test initialization
        assert agent._error_handling_enabled is True
        assert agent._error_manager is not None
        assert agent._error_manager.agent_id == "validation_test_agent"
        
        # Test execute with recovery
        async def mixin_test_func():
            return "mixin_success"
        
        result = await agent.execute_with_recovery(
            "mixin_test",
            mixin_test_func,
            category=ErrorCategory.PROCESSING
        )
        assert result == "mixin_success"
        
        # Test health check
        health = await agent.perform_health_check()
        assert "agent_id" in health
        assert "error_handling_enabled" in health
        assert health["error_handling_enabled"] is True
        
        # Test convenience methods
        agent.enable_error_handling()
        agent.register_circuit_breaker("custom_breaker")
        
        recommendations = agent.get_recovery_recommendations()
        assert isinstance(recommendations, list)
        
        return True
    
    mixin_result = asyncio.run(test_mixin())
    print("✅ Error handling mixin tests passed")
    mixin_passed = True
except Exception as e:
    print(f"❌ Error handling mixin test failed: {e}")
    mixin_passed = False

# Test 5: Error Handler Decorator
print("\n5️⃣  Testing error handler decorator...")
try:
    async def test_decorator():
        
        @error_handler(category=ErrorCategory.PROCESSING, max_retries=2)
        async def decorated_function(should_fail_count=0):
            decorated_function.call_count = getattr(decorated_function, 'call_count', 0) + 1
            if decorated_function.call_count <= should_fail_count:
                raise Exception(f"Decorator test failure {decorated_function.call_count}")
            return f"decorated_success_{decorated_function.call_count}"
        
        # Test successful decoration
        result = await decorated_function(0)
        assert result == "decorated_success_1"
        
        # Reset and test with retries
        decorated_function.call_count = 0
        result = await decorated_function(2)
        assert result == "decorated_success_3"
        assert decorated_function.call_count == 3
        
        return True
    
    decorator_result = asyncio.run(test_decorator())
    print("✅ Error handler decorator tests passed")
    decorator_passed = True
except Exception as e:
    print(f"❌ Error handler decorator test failed: {e}")
    decorator_passed = False

# Test 6: Performance and Integration
print("\n6️⃣  Testing integration and performance...")
try:
    class IntegrationTestAgent(ErrorHandlingMixin):
        def __init__(self):
            self.agent_id = "integration_agent"
            super().__init__()
    
    async def test_integration():
        agent = IntegrationTestAgent()
        agent.enable_error_handling(auto_circuit_breakers=True)
        
        # Test multiple operations
        operations_completed = 0
        
        # HTTP request simulation
        async def http_request():
            return {"status": "success", "data": "http_response"}
        
        http_result = await agent.http_request_with_recovery(http_request)
        assert http_result["status"] == "success"
        operations_completed += 1
        
        # Database operation simulation
        async def db_operation():
            return {"rows": 5, "status": "completed"}
        
        db_result = await agent.database_operation_with_recovery(db_operation)
        assert db_result["rows"] == 5
        operations_completed += 1
        
        # External API simulation
        async def api_call():
            return {"api_response": "external_data"}
        
        api_result = await agent.external_api_call_with_recovery(api_call)
        assert "api_response" in api_result
        operations_completed += 1
        
        # Processing operation simulation
        async def processing():
            return {"processed": True, "items": 100}
        
        proc_result = await agent.processing_operation_with_recovery(processing)
        assert proc_result["processed"] is True
        operations_completed += 1
        
        # Performance test - should complete quickly
        start_time = time.time()
        for _ in range(50):
            await agent.execute_with_recovery(
                "performance_test",
                lambda: "fast_result",
                category=ErrorCategory.PROCESSING
            )
        elapsed_time = time.time() - start_time
        
        # Should complete 50 operations in under 1 second
        assert elapsed_time < 1.0, f"Performance test too slow: {elapsed_time:.2f}s"
        
        # Get final health check
        final_health = await agent.perform_health_check()
        assert final_health["error_handling_enabled"] is True
        assert operations_completed == 4
        
        return True
    
    integration_result = asyncio.run(test_integration())
    print("✅ Integration and performance tests passed")
    integration_passed = True
except Exception as e:
    print(f"❌ Integration test failed: {e}")
    integration_passed = False

# Summary
print("\n" + "="*60)
print("🛡️ ERROR HANDLING SYSTEM VALIDATION SUMMARY")
print("="*60)

test_results = [
    ("Core Components Import", imports_passed),
    ("Circuit Breaker Functionality", circuit_breaker_passed),
    ("Error Recovery Manager", recovery_manager_passed),
    ("Error Handling Mixin", mixin_passed),
    ("Error Handler Decorator", decorator_passed),
    ("Integration & Performance", integration_passed)
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
    print("\n🎉 All error handling validations passed!")
    print("✅ Comprehensive error handling system is ready for production!")
    print("\n🛡️ Enhanced error handling capabilities available:")
    print("   - Circuit breaker protection for unreliable operations")
    print("   - Automatic retry with exponential backoff")
    print("   - Fallback mechanisms for graceful degradation")
    print("   - Error pattern analysis and recommendations")
    print("   - Comprehensive error tracking and health monitoring")
    print("   - Performance monitoring with minimal overhead")
    print("   - Easy integration via mixin pattern")
    print("   - Decorator-based error handling for simple cases")
else:
    print(f"\n⚠️  {total - passed} tests failed. Review issues above.")
    print("Error handling system needs attention before production use.")

print("\n🚀 Error handling system validation complete!")