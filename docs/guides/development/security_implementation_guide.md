# Security Implementation Guide

## Production Security Implementation for A2A Platform

This guide documents the security implementations required for SAP production deployment.

---

## Security Architecture

### Defense in Depth Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                        Edge Security                             │
│                   (WAF, DDoS Protection)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                     Network Security                             │
│              (Firewalls, IDS/IPS, VPN)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                   Application Security                           │
│        (Authentication, Authorization, Encryption)               │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                      Data Security                               │
│           (Encryption at Rest, Access Controls)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Requirements

### 1. Authentication Implementation

```python
# backend/app/core/security/authentication.py

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import pyotp
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class AuthenticationService:
    """SAP-compliant authentication service"""
    
    def __init__(self):
        self.pwd_context = CryptContext(
            schemes=["argon2", "bcrypt"],
            deprecated="auto",
            argon2__memory_cost=65536,
            argon2__time_cost=3,
            argon2__parallelism=4
        )
        self.security = HTTPBearer()
        
    def create_access_token(self, 
                          subject: str,
                          expires_delta: Optional[timedelta] = None,
                          additional_claims: Dict[str, Any] = None) -> str:
        """Create JWT access token with SAP security standards"""
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=1)
            
        to_encode = {
            "sub": subject,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": self.generate_jti(),
            "iss": "a2a-platform",
            "aud": ["a2a-api"],
            **(additional_claims or {})
        }
        
        # Use RS256 for production
        encoded_jwt = jwt.encode(
            to_encode,
            self.get_private_key(),
            algorithm="RS256"
        )
        
        return encoded_jwt
    
    def verify_mfa_token(self, user_id: str, token: str) -> bool:
        """Verify TOTP MFA token"""
        
        user_secret = self.get_user_mfa_secret(user_id)
        if not user_secret:
            return False
            
        totp = pyotp.TOTP(user_secret)
        
        # Allow for time drift
        return totp.verify(token, valid_window=1)
```

### 2. Authorization Implementation

```python
# backend/app/core/security/authorization.py

from typing import List, Dict, Any
from functools import wraps
import casbin
from fastapi import HTTPException

class AuthorizationService:
    """Attribute-based access control (ABAC) implementation"""
    
    def __init__(self):
        self.enforcer = casbin.Enforcer(
            "config/rbac_model.conf",
            "config/rbac_policy.csv"
        )
        
    def require_permissions(self, *required_permissions: str):
        """Decorator for permission-based access control"""
        
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract user context
                user = kwargs.get('current_user')
                if not user:
                    raise HTTPException(
                        status_code=401,
                        detail="Authentication required"
                    )
                
                # Check permissions
                for permission in required_permissions:
                    resource, action = permission.split(':')
                    
                    if not self.check_permission(
                        user.id,
                        resource,
                        action,
                        kwargs.get('resource_id')
                    ):
                        raise HTTPException(
                            status_code=403,
                            detail=f"Permission denied: {permission}"
                        )
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def check_permission(self,
                        user_id: str,
                        resource: str,
                        action: str,
                        resource_id: Optional[str] = None) -> bool:
        """Check if user has permission to perform action on resource"""
        
        # Build permission request
        request = [user_id, resource, action]
        
        if resource_id:
            request.append(resource_id)
            
        # Check with policy engine
        return self.enforcer.enforce(*request)
```

### 3. Encryption Implementation

```python
# backend/app/core/security/encryption.py

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os
import base64

class EncryptionService:
    """SAP-compliant encryption service"""
    
    def __init__(self):
        self.key_rotation_days = 90
        self.initialize_keys()
        
    def encrypt_sensitive_data(self, 
                             data: bytes,
                             associated_data: bytes = None) -> Dict[str, str]:
        """Encrypt data using AES-256-GCM"""
        
        # Generate nonce
        nonce = os.urandom(12)
        
        # Get current encryption key
        key = self.get_current_key()
        aesgcm = AESGCM(key)
        
        # Encrypt
        ciphertext = aesgcm.encrypt(nonce, data, associated_data)
        
        return {
            "ciphertext": base64.b64encode(ciphertext).decode('utf-8'),
            "nonce": base64.b64encode(nonce).decode('utf-8'),
            "key_version": self.get_key_version(),
            "algorithm": "AES-256-GCM"
        }
    
    def decrypt_sensitive_data(self,
                             encrypted_data: Dict[str, str],
                             associated_data: bytes = None) -> bytes:
        """Decrypt data encrypted with AES-256-GCM"""
        
        # Decode components
        ciphertext = base64.b64decode(encrypted_data["ciphertext"])
        nonce = base64.b64decode(encrypted_data["nonce"])
        
        # Get appropriate key version
        key = self.get_key_by_version(encrypted_data["key_version"])
        aesgcm = AESGCM(key)
        
        # Decrypt
        return aesgcm.decrypt(nonce, ciphertext, associated_data)
```

### 4. API Security Implementation

```python
# backend/app/core/security/api_security.py

from typing import Dict, List, Optional
import time
import hmac
import hashlib
from fastapi import Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

class APISecurityService:
    """API security implementation with rate limiting and request signing"""
    
    def __init__(self):
        self.limiter = Limiter(key_func=get_remote_address)
        self.request_cache = {}  # For replay attack prevention
        
    def verify_request_signature(self, 
                               request: Request,
                               body: bytes,
                               signature: str) -> bool:
        """Verify HMAC-SHA256 request signature"""
        
        # Extract components
        timestamp = request.headers.get("X-Request-Timestamp")
        nonce = request.headers.get("X-Request-Nonce")
        
        if not timestamp or not nonce:
            raise HTTPException(
                status_code=400,
                detail="Missing required headers"
            )
        
        # Check timestamp freshness (5 minute window)
        request_time = int(timestamp)
        current_time = int(time.time())
        
        if abs(current_time - request_time) > 300:
            raise HTTPException(
                status_code=401,
                detail="Request timestamp too old"
            )
        
        # Check for replay attacks
        cache_key = f"{nonce}:{timestamp}"
        if cache_key in self.request_cache:
            raise HTTPException(
                status_code=401,
                detail="Duplicate request detected"
            )
        
        # Store in cache
        self.request_cache[cache_key] = current_time
        
        # Build signature base string
        signature_base = f"{request.method}\\n{request.url.path}\\n{timestamp}\\n{nonce}\\n{body.decode('utf-8')}"
        
        # Get API key secret
        api_key = request.headers.get("X-API-Key")
        secret = self.get_api_key_secret(api_key)
        
        # Calculate expected signature
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            signature_base.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Constant time comparison
        return hmac.compare_digest(expected_signature, signature)
```

### 5. Audit Logging Implementation

```python
# backend/app/core/security/audit_logging.py

import json
from datetime import datetime
from typing import Dict, Any, Optional
import asyncio
from dataclasses import dataclass, asdict

@dataclass
class AuditEvent:
    """Audit event structure"""
    event_id: str
    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    resource: str
    action: str
    result: str
    details: Dict[str, Any]
    risk_score: float = 0.0

class AuditLogger:
    """Comprehensive audit logging for security compliance"""
    
    def __init__(self):
        self.audit_queue = asyncio.Queue()
        self.start_background_tasks()
        
    async def log_security_event(self,
                               event_type: str,
                               user_id: Optional[str],
                               request: Request,
                               resource: str,
                               action: str,
                               result: str,
                               details: Dict[str, Any] = None):
        """Log security-relevant events"""
        
        audit_event = AuditEvent(
            event_id=self.generate_event_id(),
            timestamp=datetime.utcnow(),
            event_type=event_type,
            user_id=user_id,
            ip_address=self.get_client_ip(request),
            user_agent=request.headers.get("User-Agent", ""),
            resource=resource,
            action=action,
            result=result,
            details=details or {},
            risk_score=self.calculate_risk_score(event_type, result)
        )
        
        # Queue for async processing
        await self.audit_queue.put(audit_event)
        
        # Real-time alerting for high-risk events
        if audit_event.risk_score > 0.8:
            await self.send_security_alert(audit_event)
    
    def calculate_risk_score(self, event_type: str, result: str) -> float:
        """Calculate risk score for security events"""
        
        risk_scores = {
            "authentication_failure": 0.6,
            "authorization_failure": 0.7,
            "multiple_failed_attempts": 0.9,
            "privilege_escalation": 0.95,
            "data_exfiltration": 0.95,
            "suspicious_api_usage": 0.8
        }
        
        base_score = risk_scores.get(event_type, 0.3)
        
        # Increase score for failures
        if result == "failure":
            base_score = min(base_score * 1.5, 1.0)
            
        return base_score
```

---

## Security Testing Requirements

### Unit Tests for Security Components

```python
# backend/tests/test_security.py

import pytest
from app.core.security import AuthenticationService, AuthorizationService
import jwt
import time

class TestAuthentication:
    """Test authentication implementation"""
    
    @pytest.fixture
    def auth_service(self):
        return AuthenticationService()
    
    def test_jwt_token_creation(self, auth_service):
        """Test JWT token creation with proper claims"""
        
        token = auth_service.create_access_token(
            subject="user123",
            additional_claims={"roles": ["admin"]}
        )
        
        # Decode and verify
        decoded = jwt.decode(
            token,
            auth_service.get_public_key(),
            algorithms=["RS256"],
            audience=["a2a-api"]
        )
        
        assert decoded["sub"] == "user123"
        assert "admin" in decoded["roles"]
        assert decoded["iss"] == "a2a-platform"
        assert decoded["exp"] > time.time()
    
    def test_mfa_verification(self, auth_service):
        """Test MFA token verification"""
        
        # Generate test secret
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)
        
        # Test valid token
        valid_token = totp.now()
        assert auth_service.verify_mfa_token("user123", valid_token) is True
        
        # Test invalid token
        assert auth_service.verify_mfa_token("user123", "000000") is False

class TestAuthorization:
    """Test authorization implementation"""
    
    @pytest.fixture
    def authz_service(self):
        return AuthorizationService()
    
    def test_permission_check(self, authz_service):
        """Test permission checking"""
        
        # Add test policy
        authz_service.enforcer.add_policy("user123", "data", "read")
        
        # Test allowed action
        assert authz_service.check_permission(
            "user123", "data", "read"
        ) is True
        
        # Test denied action
        assert authz_service.check_permission(
            "user123", "data", "delete"
        ) is False
```

---

## Security Monitoring Dashboard

### Required Security Metrics

```python
# backend/app/core/security/metrics.py

from prometheus_client import Counter, Histogram, Gauge

# Authentication metrics
auth_attempts = Counter(
    'security_auth_attempts_total',
    'Total authentication attempts',
    ['method', 'result']
)

auth_duration = Histogram(
    'security_auth_duration_seconds',
    'Authentication duration',
    ['method']
)

active_sessions = Gauge(
    'security_active_sessions',
    'Number of active sessions',
    ['user_type']
)

# Authorization metrics
authz_checks = Counter(
    'security_authz_checks_total',
    'Total authorization checks',
    ['resource', 'action', 'result']
)

# Security events
security_events = Counter(
    'security_events_total',
    'Total security events',
    ['event_type', 'severity']
)

# API security
api_violations = Counter(
    'security_api_violations_total',
    'API security violations',
    ['violation_type']
)

rate_limit_hits = Counter(
    'security_rate_limit_hits_total',
    'Rate limit hits',
    ['endpoint']
)
```

---

## Incident Response Procedures

### Security Incident Response Plan

```yaml
incident_response:
  classification:
    P1_Critical:
      - Data breach
      - System compromise
      - Active attack
      response_time: 15 minutes
      
    P2_High:
      - Suspected breach
      - Multiple auth failures
      - Privilege escalation attempts
      response_time: 1 hour
      
    P3_Medium:
      - Vulnerability discovered
      - Policy violations
      - Suspicious activity
      response_time: 4 hours
  
  response_team:
    - role: Incident Commander
      contact: security-lead@company.com
      
    - role: Technical Lead
      contact: tech-lead@company.com
      
    - role: Communications Lead
      contact: comms@company.com
  
  procedures:
    1_Detection:
      - Automated alerting
      - Manual reporting
      - Threat intelligence
      
    2_Analysis:
      - Severity assessment
      - Impact analysis
      - Root cause analysis
      
    3_Containment:
      - Isolate affected systems
      - Revoke compromised credentials
      - Block malicious IPs
      
    4_Eradication:
      - Remove malware
      - Patch vulnerabilities
      - Reset credentials
      
    5_Recovery:
      - Restore services
      - Verify integrity
      - Monitor for recurrence
      
    6_Lessons_Learned:
      - Post-incident review
      - Update procedures
      - Security improvements
```

---

## Compliance Documentation

### Required Compliance Artifacts

1. **Security Policies**
   - Information Security Policy
   - Access Control Policy
   - Data Protection Policy
   - Incident Response Policy
   - Vulnerability Management Policy

2. **Procedures**
   - User Access Management Procedure
   - Security Monitoring Procedure
   - Patch Management Procedure
   - Backup and Recovery Procedure
   - Security Awareness Training

3. **Evidence Collection**
   - Access reviews (quarterly)
   - Security training records
   - Vulnerability scan reports
   - Patch compliance reports
   - Incident response tests

---

## Next Steps for Production

1. **Schedule Security Assessments**
   - Contact SAP-approved penetration testing vendor
   - Schedule vulnerability assessment
   - Plan compliance audit

2. **Complete Security Implementations**
   - Implement remaining security controls
   - Configure security monitoring
   - Set up incident response tools

3. **Security Training**
   - Developer security training
   - Security awareness for all staff
   - Incident response drills

4. **Documentation Review**
   - Complete all security policies
   - Update incident response procedures
   - Prepare audit evidence

---

*Last Updated: December 2024*
*Version: 1.0.0*