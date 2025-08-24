# A2A Platform Security Hardening Guide

## Executive Summary

This guide provides comprehensive security hardening instructions for the A2A platform, addressing critical vulnerabilities identified in the security audit. All agents must implement these security measures before production deployment.

## Critical Security Fixes Implemented

### 1. ✅ Hardcoded API Keys (FIXED)
**File**: `reasoningAgent/test_integration_grok_api.py`
- Removed hardcoded fallback API keys
- Now requires environment variables with no defaults
- Exits gracefully if API keys not configured

### 2. ✅ Dangerous eval() Usage (FIXED)
**File**: `calculationAgent/comprehensiveCalculationAgentSdk.py`
- Replaced `eval()` with safe mathematical expression parser
- Created `SafeMathParser` class with AST-based evaluation
- Prevents code injection while maintaining functionality

### 3. ✅ Command Injection (FIXED)
**File**: `gleanAgent/gleanAgentSdk.py`
- Implemented whitelist of allowed tools
- Added input validation for tool names
- Prevents arbitrary command execution

### 4. ✅ Security Middleware Created
**Files**: 
- `core/security_middleware.py` - Comprehensive security framework
- `core/secure_agent_base.py` - Secure base class for all agents
- `core/safe_math_parser.py` - Safe mathematical expression evaluation

## Security Components

### 1. Authentication & Authorization

```python
from app.a2a.core.security_middleware import require_auth, SecurityMiddleware

# Protect endpoints with authentication
@require_auth(permissions=['agent.execute'])
async def protected_endpoint(request: Request):
    # Authenticated requests only
    user = request.state.auth
    return {"user_id": user['user_id']}
```

### 2. Rate Limiting

```python
# Automatic rate limiting per user/IP
rate_limiter = RateLimiter(
    requests_per_minute=100,
    burst=10  # Allow burst of 10 requests
)
```

### 3. Input Validation

```python
from app.a2a.core.security_middleware import InputValidator

validator = InputValidator()

# Validate strings
safe_string = validator.validate_string(user_input, "field_name", max_length=1000)

# Validate paths (prevents traversal)
safe_path = validator.validate_path(file_path)

# Check for SQL injection
safe_query = validator.validate_sql_safe(query, "query")

# Check for script injection
safe_html = validator.validate_no_scripts(content, "content")
```

### 4. Secure Logging

```python
from app.a2a.core.security_middleware import get_secure_logger

# Automatically sanitizes sensitive data
logger = get_secure_logger(__name__)
logger.info("User login", {"user_id": "123", "password": "secret"})
# Output: User login {'user_id': '123', 'password': '***MASKED***'}
```

## Implementation Guide

### Step 1: Convert Existing Agents to Secure Base

```python
from app.a2a.core.secure_agent_base import SecureA2AAgent, SecureAgentConfig

class MySecureAgent(SecureA2AAgent):
    def __init__(self):
        config = SecureAgentConfig(
            agent_id="my_agent",
            agent_name="My Secure Agent",
            allowed_operations={"calculate", "analyze", "process"},
            enable_authentication=True,
            enable_rate_limiting=True,
            enable_input_validation=True
        )
        super().__init__(config)
    
    @self.secure_handler("calculate")
    async def handle_calculate(self, message, context_id, data):
        # Automatically protected with auth, rate limiting, and validation
        expression = data.get("expression")  # Already validated
        result = self.safe_calculate(expression)
        return self.create_secure_response({"result": result})
```

### Step 2: Environment Configuration

Create `.env` file with required security settings:

```bash
# Authentication
JWT_SECRET=your-256-bit-secret-here
JWT_ALGORITHM=HS256
JWT_EXPIRY_MINUTES=60

# API Keys (no defaults allowed)
GROK_API_KEY=your-actual-api-key
OPENAI_API_KEY=your-actual-api-key
PERPLEXITY_API_KEY=your-actual-api-key

# Security Settings
ENABLE_AUTH=true
ENABLE_RATE_LIMITING=true
MAX_REQUEST_SIZE=1048576
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

### Step 3: Apply Security Decorators

```python
from app.a2a.core.security_middleware import require_auth, validate_input

class AgentRouter:
    @router.post("/execute")
    @require_auth(permissions=["agent.execute"])
    @validate_input(schema={
        "operation": {"type": "string", "required": True},
        "data": {"type": "dict", "required": True}
    })
    async def execute_operation(request: Request):
        # Request is authenticated and validated
        data = await request.json()  # Safe, validated data
        return await process_operation(data)
```

### Step 4: Safe External API Calls

```python
# Instead of direct HTTP calls:
# ❌ response = requests.post(url, data=payload)

# Use A2A-compliant external requests:
# ✅ Through secure agent base
response = await self.make_external_request(
    service="perplexity",
    endpoint="/chat/completions",
    data=payload
)
```

## Security Best Practices

### 1. API Key Management
- ❌ **Never** hardcode API keys in source code
- ❌ **Never** use default/fallback API keys
- ✅ **Always** load from environment variables
- ✅ **Always** validate API keys exist before use
- ✅ Use secret management services in production

### 2. Input Validation
- ✅ Validate all user inputs
- ✅ Use whitelist approach for allowed values
- ✅ Sanitize file paths and prevent traversal
- ✅ Check for injection attacks (SQL, script, command)
- ✅ Limit input sizes to prevent DoS

### 3. Authentication & Authorization
- ✅ Require authentication for all agent operations
- ✅ Implement role-based access control (RBAC)
- ✅ Use JWT tokens with expiration
- ✅ Validate permissions for each operation
- ✅ Log all authentication attempts

### 4. Rate Limiting
- ✅ Implement per-user rate limits
- ✅ Use token bucket algorithm for burst handling
- ✅ Return proper 429 status with Retry-After header
- ✅ Different limits for different operations
- ✅ Monitor and alert on rate limit violations

### 5. Secure Logging
- ✅ Never log sensitive information (passwords, keys, tokens)
- ✅ Use structured logging with sanitization
- ✅ Log security events (auth failures, rate limits, validation errors)
- ✅ Implement log rotation and retention policies
- ✅ Monitor logs for security anomalies

### 6. Error Handling
- ✅ Never expose internal errors to users
- ✅ Log detailed errors internally
- ✅ Return generic error messages to clients
- ✅ Include request IDs for troubleshooting
- ✅ Implement proper error recovery

## Testing Security

### Unit Tests
```python
def test_no_eval_usage():
    """Ensure no eval() in codebase"""
    assert not has_eval_usage("calculationAgent.py")

def test_api_key_validation():
    """Test API key requirements"""
    with pytest.raises(SystemExit):
        load_api_key_with_default()  # Should fail

def test_input_validation():
    """Test input sanitization"""
    assert validate_path("../../../etc/passwd") raises ValueError
    assert validate_sql_safe("'; DROP TABLE users; --") raises ValueError
```

### Integration Tests
```python
async def test_rate_limiting():
    """Test rate limiter"""
    client = TestClient(app)
    
    # Make 100 requests (limit)
    for _ in range(100):
        response = client.post("/api/execute")
        assert response.status_code == 200
    
    # 101st request should be rate limited
    response = client.post("/api/execute")
    assert response.status_code == 429
    assert "Retry-After" in response.headers
```

### Security Scanning
```bash
# Run security scanners
bandit -r app/a2a/agents/  # Python security linter
safety check  # Check dependencies for vulnerabilities
```

## Monitoring & Alerting

### Security Events to Monitor
1. **Authentication Failures** - Multiple failed login attempts
2. **Rate Limit Violations** - Excessive requests from single source
3. **Input Validation Failures** - Potential attack attempts
4. **External API Errors** - Service availability issues
5. **Permission Denials** - Unauthorized access attempts

### Alert Thresholds
```yaml
alerts:
  - name: "High Authentication Failures"
    condition: "auth_failures > 10 per minute"
    severity: "HIGH"
    
  - name: "Rate Limit Abuse"
    condition: "rate_limit_violations > 50 per hour from single IP"
    severity: "MEDIUM"
    
  - name: "Potential SQL Injection"
    condition: "sql_validation_errors > 5 per minute"
    severity: "CRITICAL"
```

## Compliance Checklist

Before deploying any agent to production:

- [ ] No hardcoded secrets or API keys
- [ ] All eval() usage replaced with safe alternatives
- [ ] Input validation on all user inputs
- [ ] Authentication required for all operations
- [ ] Rate limiting implemented
- [ ] Secure logging with sanitization
- [ ] Error messages don't expose sensitive info
- [ ] External API calls go through A2A protocol
- [ ] Security tests passing
- [ ] Security scan completed

## Incident Response

### If Security Breach Detected:
1. **Immediate Actions**
   - Revoke compromised API keys
   - Block suspicious IPs
   - Enable emergency rate limiting
   - Alert security team

2. **Investigation**
   - Review security logs
   - Identify attack vector
   - Assess data exposure
   - Document timeline

3. **Remediation**
   - Patch vulnerabilities
   - Update security rules
   - Rotate all secrets
   - Notify affected users

4. **Post-Incident**
   - Security audit
   - Update procedures
   - Staff training
   - Improve monitoring

## Future Enhancements

### Phase 1 (Immediate)
- ✅ Remove hardcoded secrets
- ✅ Fix eval() usage
- ✅ Fix command injection
- ✅ Implement basic auth

### Phase 2 (Short-term)
- 🔄 Convert all agents to secure base
- 🔄 Implement RBAC
- 🔄 Add security monitoring
- 🔄 Automated security testing

### Phase 3 (Long-term)
- 📋 Web Application Firewall (WAF)
- 📋 DDoS protection
- 📋 Encrypted data storage
- 📋 Security audit certification

---

**Security is not a feature, it's a requirement.** Every developer is responsible for maintaining the security posture of the A2A platform.