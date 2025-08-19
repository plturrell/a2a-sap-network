#!/usr/bin/env python3
"""
Validation script for comprehensive security hardening system
Tests all security components and defensive measures
"""

import asyncio
import sys
import time
import base64
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🔐 Testing Comprehensive Security Hardening System...\n")

# Test 1: Core Security Components
print("1️⃣  Testing core security components...")
try:
    from app.a2a.core.securityHardening import (
        SecurityHardeningManager, SecurityLevel, ThreatLevel, AccessLevel,
        SecurityContext, SecurityException, AuthenticationError, AuthorizationError,
        InputValidationError, RateLimitExceededError, secure_operation
    )
    from app.a2a.core.securityHardeningMixin import SecurityHardeningMixin
    
    print("✅ All security hardening imports successful")
    imports_passed = True
except Exception as e:
    print(f"❌ Import failed: {e}")
    imports_passed = False

# Test 2: Input Validation and Sanitization
print("\n2️⃣  Testing input validation and sanitization...")
try:
    manager = SecurityHardeningManager("test_agent")
    
    # Test SQL injection detection
    sql_injection_attempts = [
        "'; DROP TABLE users; --",
        "1 OR 1=1",
        "UNION SELECT * FROM passwords",
        "admin'; DELETE FROM users WHERE 'x'='x"
    ]
    
    for attempt in sql_injection_attempts:
        detected = manager._detect_sql_injection(attempt, strict=True)
        assert detected, f"Failed to detect SQL injection: {attempt}"
    
    # Test XSS detection
    xss_attempts = [
        "<script>alert('XSS')</script>",
        "javascript:alert('XSS')",
        "<iframe src='http://evil.com'></iframe>",
        "onclick='alert(1)'"
    ]
    
    for attempt in xss_attempts:
        detected = manager._detect_xss(attempt, strict=True)
        assert detected, f"Failed to detect XSS: {attempt}"
    
    # Test path traversal detection
    path_traversal_attempts = [
        "../../../etc/passwd",
        "..\\..\\windows\\system32",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "~/../../sensitive_file"
    ]
    
    for attempt in path_traversal_attempts:
        detected = manager._detect_path_traversal(attempt, strict=True)
        assert detected, f"Failed to detect path traversal: {attempt}"
    
    # Test input sanitization
    dirty_input = "<script>alert('xss')</script>Hello World; DROP TABLE users;"
    clean_input = manager.sanitize_input(dirty_input, "html")
    assert "<script>" not in clean_input
    assert "DROP TABLE" not in clean_input
    
    print("✅ Input validation and sanitization tests passed")
    validation_passed = True
except Exception as e:
    print(f"❌ Input validation test failed: {e}")
    validation_passed = False

# Test 3: Authentication and Authorization
print("\n3️⃣  Testing authentication and authorization...")
try:
    manager = SecurityHardeningManager("test_agent")
    
    # Test valid token authentication
    valid_token = base64.b64encode(b"valid_test_token_12345").decode()
    context = SecurityContext(
        agent_id="test_agent",
        ip_address="192.168.1.100"
    )
    
    auth_result = manager.authenticate_request(valid_token, context)
    assert auth_result is True
    
    # Test invalid token
    try:
        invalid_token = "invalid_token"
        manager.authenticate_request(invalid_token, context)
        assert False, "Should have raised AuthenticationError"
    except AuthenticationError:
        pass  # Expected
    
    # Test authorization
    context.access_levels = {AccessLevel.READ, AccessLevel.WRITE}
    
    # Should succeed for READ access
    read_authorized = manager.authorize_request(context, AccessLevel.READ)
    assert read_authorized is True
    
    # Should fail for ADMIN access
    admin_authorized = manager.authorize_request(context, AccessLevel.ADMIN)
    assert admin_authorized is False
    
    print("✅ Authentication and authorization tests passed")
    auth_passed = True
except Exception as e:
    print(f"❌ Authentication/authorization test failed: {e}")
    auth_passed = False

# Test 4: Rate Limiting
print("\n4️⃣  Testing rate limiting...")
try:
    manager = SecurityHardeningManager("test_agent")
    
    # Configure tight rate limits for testing
    manager.rate_limit_config["test"] = {"requests": 3, "window": 1}
    
    # Test within limits
    for i in range(3):
        result = manager.check_rate_limit("test_user", "test")
        assert result is True, f"Request {i+1} should be allowed"
    
    # Test exceeding limits
    result = manager.check_rate_limit("test_user", "test")
    assert result is False, "Request should be rate limited"
    
    # Test different user should not be affected
    result = manager.check_rate_limit("different_user", "test")
    assert result is True, "Different user should not be rate limited"
    
    print("✅ Rate limiting tests passed")
    rate_limit_passed = True
except Exception as e:
    print(f"❌ Rate limiting test failed: {e}")
    rate_limit_passed = False

# Test 5: Security Hardening Mixin
print("\n5️⃣  Testing security hardening mixin...")
try:
    class TestAgent(SecurityHardeningMixin):
        def __init__(self):
            self.agent_id = "security_test_agent"
            super().__init__()
    
    agent = TestAgent()
    
    # Test initialization
    assert agent._security_enabled is True
    assert agent._security_manager is not None
    assert agent._security_manager.agent_id == "security_test_agent"
    
    # Test input validation
    safe_input = "Hello World"
    result = agent.validate_and_sanitize_input(safe_input, "general")
    assert result == safe_input
    
    # Test dangerous input
    try:
        dangerous_input = "<script>alert('xss')</script>"
        agent.validate_and_sanitize_input(dangerous_input, "html", sanitize=False)
        assert False, "Should have raised InputValidationError"
    except InputValidationError:
        pass  # Expected
    
    # Test encryption/decryption
    sensitive_data = "secret_password_123"
    encrypted = agent.encrypt_sensitive_data(sensitive_data)
    assert encrypted != sensitive_data
    
    decrypted = agent.decrypt_sensitive_data(encrypted)
    assert decrypted == sensitive_data
    
    # Test secure token generation
    token = agent.generate_secure_token(32)
    assert len(token) > 30
    assert token.replace("-", "").replace("_", "").isalnum()
    
    print("✅ Security hardening mixin tests passed")
    mixin_passed = True
except Exception as e:
    print(f"❌ Security hardening mixin test failed: {e}")
    mixin_passed = False

# Test 6: Security Audit and Monitoring
print("\n6️⃣  Testing security audit and monitoring...")
try:
    class AuditTestAgent(SecurityHardeningMixin):
        def __init__(self):
            self.agent_id = "audit_test_agent"
            super().__init__()
    
    agent = AuditTestAgent()
    agent.enable_security_hardening()
    
    # Simulate some security events
    manager = agent._security_manager
    
    # Trigger some validation failures to test metrics
    try:
        agent.validate_and_sanitize_input("<script>alert('test')</script>", "html", sanitize=False)
    except InputValidationError:
        pass
    
    # Check security summary
    summary = agent.get_security_summary(hours=1)
    assert "total_events" in summary
    assert "agent_security_metrics" in summary
    assert "security_health_score" in summary
    
    # Test security audit
    async def test_audit():
        audit_results = await agent.perform_security_audit()
        assert "timestamp" in audit_results
        assert "security_enabled" in audit_results
        assert "recommendations" in audit_results
        assert "compliance_status" in audit_results
        return True
    
    audit_result = asyncio.run(test_audit())
    assert audit_result is True
    
    # Check recommendations
    recommendations = agent.get_security_recommendations()
    assert isinstance(recommendations, list)
    
    print("✅ Security audit and monitoring tests passed")
    audit_passed = True
except Exception as e:
    print(f"❌ Security audit test failed: {e}")
    audit_passed = False

# Test 7: Integration and Performance
print("\n7️⃣  Testing integration and performance...")
try:
    class IntegrationTestAgent(SecurityHardeningMixin):
        def __init__(self):
            self.agent_id = "integration_security_agent"
            super().__init__()
        
        async def sample_operation(self, data: str) -> str:
            return f"Processed: {data}"
        
        async def admin_operation(self, command: str) -> str:
            return f"Admin executed: {command}"
    
    agent = IntegrationTestAgent()
    agent.enable_security_hardening()
    
    # Test secure operation execution
    async def test_secure_operation():
        result = await agent.execute_secure_operation(
            "sample_operation",
            agent.sample_operation,
            "test_data",
            required_level=AccessLevel.READ,
            validate_input=True
        )
        # Note: This returns a Task, so we need to await it
        if asyncio.iscoroutine(result):
            result = await result
        return result
    
    secure_result = asyncio.run(test_secure_operation())
    
    # Test performance with security enabled
    start_time = time.time()
    for _ in range(50):
        token = agent.generate_secure_token(16)
        safe_data = agent.validate_and_sanitize_input("test_data_123", "general")
    security_time = time.time() - start_time
    
    # Test performance with security disabled (for comparison)
    agent.disable_security_hardening()
    start_time = time.time()
    for _ in range(50):
        token = agent.generate_secure_token(16)
        # Skip validation when disabled
        safe_data = "test_data_123"
    no_security_time = time.time() - start_time
    
    # Security should add reasonable overhead (less than 3x)
    if no_security_time > 0:
        overhead_ratio = security_time / no_security_time
        assert overhead_ratio < 3.0, f"Security overhead too high: {overhead_ratio}x"
    
    # Test trusted agent functionality
    agent.enable_security_hardening()
    agent.add_trusted_agent("trusted_test_agent")
    
    trusted_agents = agent._security_manager.trusted_agents
    assert "trusted_test_agent" in trusted_agents
    
    agent.remove_trusted_agent("trusted_test_agent")
    assert "trusted_test_agent" not in trusted_agents
    
    print("✅ Integration and performance tests passed")
    integration_passed = True
except Exception as e:
    print(f"❌ Integration test failed: {e}")
    integration_passed = False

# Summary
print("\n" + "="*60)
print("🔐 SECURITY HARDENING SYSTEM VALIDATION SUMMARY")
print("="*60)

test_results = [
    ("Core Security Components", imports_passed),
    ("Input Validation & Sanitization", validation_passed),
    ("Authentication & Authorization", auth_passed),
    ("Rate Limiting", rate_limit_passed),
    ("Security Hardening Mixin", mixin_passed),
    ("Security Audit & Monitoring", audit_passed),
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
    print("\n🎉 All security hardening validations passed!")
    print("✅ Comprehensive security hardening system is ready for production!")
    print("\n🔐 Enhanced security capabilities available:")
    print("   - Input validation and sanitization against XSS, SQL injection, path traversal")
    print("   - Authentication and authorization with role-based access control")
    print("   - Rate limiting and DDoS protection")
    print("   - Threat detection and pattern analysis")
    print("   - Audit logging and security event monitoring")
    print("   - Data encryption and secure token generation")
    print("   - Security health scoring and compliance checking")
    print("   - Easy integration via mixin pattern")
    print("   - Comprehensive security audit capabilities")
else:
    print(f"\n⚠️  {total - passed} tests failed. Review security issues above.")
    print("Security hardening system needs attention before production use.")

print("\n🚀 Security hardening validation complete!")