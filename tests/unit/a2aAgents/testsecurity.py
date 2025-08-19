"""
Comprehensive test suite for A2A security module
Tests authentication, authorization, encryption, and security utilities
"""
import email

import pytest
import time
import jwt
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import os

from app.core.security import (
    SecurityLevel,
    Permission,
    Role,
    SecurityConfig,
    PasswordHasher,
    TokenManager,
    DataEncryption,
    SecurityHeaders,
    RateLimiter,
    SecurityValidator,
    SecurityAuditor,
    ROLE_PERMISSIONS,
    generate_api_key,
    generate_csrf_token,
    verify_csrf_token,
    mask_sensitive_data,
    generate_secure_filename,
    require_auth,
    require_permissions,
    hash_password,
    verify_password,
    generate_secure_password,
    encrypt_data,
    decrypt_data,
    validate_rate_limit,
    validate_user_input,
    validate_email,
    validate_password_strength
)
from app.core.exceptions import (
    A2AAuthenticationError,
    A2AAuthorizationError,
    A2ATokenExpiredError,
    A2AInvalidTokenError,
    A2AConfigurationError
)


class TestSecurityEnums:
    """Test security enumeration classes"""
    
    def test_security_level_enum(self):
        """Test SecurityLevel enum values"""
        assert SecurityLevel.PUBLIC == "public"
        assert SecurityLevel.AUTHENTICATED == "authenticated"
        assert SecurityLevel.AUTHORIZED == "authorized"
        assert SecurityLevel.ADMIN == "admin"
        assert SecurityLevel.SYSTEM == "system"
    
    def test_permission_enum(self):
        """Test Permission enum values"""
        expected_permissions = [
            "read_data", "write_data", "delete_data", "manage_users",
            "manage_agents", "admin_access", "system_config",
            "view_metrics", "execute_operations"
        ]
        
        for perm in expected_permissions:
            assert hasattr(Permission, perm.upper())
    
    def test_role_enum(self):
        """Test Role enum values"""
        expected_roles = ["viewer", "operator", "developer", "admin", "system"]
        
        for role in expected_roles:
            assert hasattr(Role, role.upper())
    
    def test_role_permissions_mapping(self):
        """Test role-permission mappings"""
        # Test viewer permissions
        viewer_perms = ROLE_PERMISSIONS[Role.VIEWER]
        assert Permission.READ_DATA in viewer_perms
        assert Permission.VIEW_METRICS in viewer_perms
        assert Permission.DELETE_DATA not in viewer_perms
        
        # Test system role has all permissions
        system_perms = ROLE_PERMISSIONS[Role.SYSTEM]
        all_permissions = list(Permission)
        assert set(system_perms) == set(all_permissions)
        
        # Test admin has most permissions
        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
        assert Permission.ADMIN_ACCESS in admin_perms
        assert Permission.MANAGE_USERS in admin_perms
        assert len(admin_perms) > len(ROLE_PERMISSIONS[Role.OPERATOR])


class TestPasswordHasher:
    """Test password hashing functionality"""
    
    def test_hash_password_success(self):
        """Test successful password hashing"""
        password = "TestPassword123!"
        hashed = PasswordHasher.hash_password(password)
        
        assert hashed != password
        assert isinstance(hashed, str)
        assert len(hashed) > 50  # bcrypt hashes are long
        assert hashed.startswith("$2b$")  # bcrypt prefix
    
    def test_hash_password_too_short(self):
        """Test password too short error"""
        short_password = "short"
        
        with pytest.raises(A2AAuthenticationError) as exc_info:
            PasswordHasher.hash_password(short_password)
        
        assert "at least" in str(exc_info.value).lower()
    
    def test_verify_password_success(self):
        """Test successful password verification"""
        password = "TestPassword123!"
        hashed = PasswordHasher.hash_password(password)
        
        assert PasswordHasher.verify_password(password, hashed)
    
    def test_verify_password_failure(self):
        """Test password verification failure"""
        password = "TestPassword123!"
        wrong_password = "WrongPassword123!"
        hashed = PasswordHasher.hash_password(password)
        
        assert not PasswordHasher.verify_password(wrong_password, hashed)
    
    def test_verify_password_invalid_hash(self):
        """Test password verification with invalid hash"""
        password = "TestPassword123!"
        invalid_hash = "invalid_hash"
        
        assert not PasswordHasher.verify_password(password, invalid_hash)
    
    def test_generate_secure_password(self):
        """Test secure password generation"""
        password = PasswordHasher.generate_secure_password()
        
        assert len(password) == 16  # default length
        assert isinstance(password, str)
        
        # Test custom length
        long_password = PasswordHasher.generate_secure_password(32)
        assert len(long_password) == 32
        
        # Test passwords are different
        password2 = PasswordHasher.generate_secure_password()
        assert password != password2


class TestTokenManager:
    """Test JWT token management"""
    
    @pytest.fixture
    def token_manager(self):
        """Create token manager with test secret"""
        secret = "test_secret_key_that_is_long_enough_for_testing_purposes_12345"
        return TokenManager(secret)
    
    def test_token_manager_invalid_secret(self):
        """Test token manager with invalid secret"""
        with pytest.raises(A2AConfigurationError):
            TokenManager("short")
    
    def test_create_access_token(self, token_manager):
        """Test access token creation"""
        user_id = "test_user_123"
        role = Role.DEVELOPER
        
        token = token_manager.create_access_token(user_id, role)
        
        assert isinstance(token, str)
        assert len(token) > 100  # JWT tokens are long
        
        # Verify token contents
        payload = token_manager.verify_token(token)
        assert payload["sub"] == user_id
        assert payload["role"] == role.value
        assert payload["type"] == "access"
        assert "permissions" in payload
        assert "exp" in payload
        assert "iat" in payload
    
    def test_create_access_token_with_custom_permissions(self, token_manager):
        """Test access token with custom permissions"""
        user_id = "test_user"
        role = Role.VIEWER
        custom_permissions = [Permission.READ_DATA, Permission.WRITE_DATA]
        
        token = token_manager.create_access_token(
            user_id, role, permissions=custom_permissions
        )
        
        payload = token_manager.verify_token(token)
        assert set(payload["permissions"]) == {p.value for p in custom_permissions}
    
    def test_create_refresh_token(self, token_manager):
        """Test refresh token creation"""
        user_id = "test_user_456"
        
        token = token_manager.create_refresh_token(user_id)
        
        assert isinstance(token, str)
        
        # Verify token contents
        payload = token_manager.verify_token(token)
        assert payload["sub"] == user_id
        assert payload["type"] == "refresh"
        assert "permissions" not in payload  # Refresh tokens don't have permissions
    
    def test_verify_token_expired(self, token_manager):
        """Test token verification with expired token"""
        user_id = "test_user"
        role = Role.VIEWER
        
        # Create token that expires immediately
        token = token_manager.create_access_token(
            user_id, role, expires_delta=timedelta(seconds=-1)
        )
        
        with pytest.raises(A2ATokenExpiredError):
            token_manager.verify_token(token)
    
    def test_verify_token_invalid(self, token_manager):
        """Test token verification with invalid token"""
        invalid_token = "invalid.token.here"
        
        with pytest.raises(A2AInvalidTokenError):
            token_manager.verify_token(invalid_token)
    
    def test_refresh_access_token(self, token_manager):
        """Test refreshing access token"""
        user_id = "test_user"
        role = Role.OPERATOR
        
        # Create refresh token
        refresh_token = token_manager.create_refresh_token(user_id)
        
        # Refresh access token
        new_access_token = token_manager.refresh_access_token(refresh_token, role)
        
        # Verify new token
        payload = token_manager.verify_token(new_access_token)
        assert payload["sub"] == user_id
        assert payload["role"] == role.value
        assert payload["type"] == "access"
    
    def test_refresh_access_token_invalid_refresh(self, token_manager):
        """Test refresh with invalid refresh token"""
        # Create access token (not refresh)
        access_token = token_manager.create_access_token("user", Role.VIEWER)
        
        with pytest.raises(A2AInvalidTokenError):
            token_manager.refresh_access_token(access_token, Role.VIEWER)


class TestDataEncryption:
    """Test data encryption functionality"""
    
    @pytest.fixture
    def encryption(self):
        """Create encryption instance"""
        return DataEncryption()
    
    def test_encrypt_decrypt_string(self, encryption):
        """Test string encryption/decryption"""
        original_data = "sensitive_information_123"
        
        encrypted = encryption.encrypt(original_data)
        decrypted = encryption.decrypt(encrypted)
        
        assert encrypted != original_data
        assert decrypted == original_data
        assert isinstance(encrypted, str)
    
    def test_encrypt_decrypt_dict(self, encryption):
        """Test dictionary encryption/decryption"""
        original_dict = {
            "user_id": "123",
            "email": "test@example.com",
            "permissions": ["read", "write"]
        }
        
        encrypted = encryption.encrypt_dict(original_dict)
        decrypted = encryption.decrypt_dict(encrypted)
        
        assert encrypted != str(original_dict)
        assert decrypted == original_dict
    
    def test_decrypt_invalid_data(self, encryption):
        """Test decryption with invalid data"""
        invalid_data = "invalid_encrypted_data"
        
        with pytest.raises(A2AAuthenticationError):
            encryption.decrypt(invalid_data)
    
    def test_encryption_consistency(self):
        """Test that same data encrypts to different values"""
        data = "test_data"
        enc1 = DataEncryption()
        enc2 = DataEncryption()
        
        encrypted1 = enc1.encrypt(data)
        encrypted2 = enc2.encrypt(data)
        
        # Different keys should produce different encrypted values
        assert encrypted1 != encrypted2
        
        # But each should decrypt correctly with their own key
        assert enc1.decrypt(encrypted1) == data
        assert enc2.decrypt(encrypted2) == data


class TestSecurityHeaders:
    """Test security headers functionality"""
    
    def test_get_security_headers(self):
        """Test security headers generation"""
        headers = SecurityHeaders.get_security_headers()
        
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
            "Referrer-Policy",
            "Permissions-Policy"
        ]
        
        for header in expected_headers:
            assert header in headers
        
        # Test specific values
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert "max-age=31536000" in headers["Strict-Transport-Security"]
        assert "default-src 'self'" in headers["Content-Security-Policy"]


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter instance"""
        return RateLimiter()
    
    def test_rate_limit_under_threshold(self, rate_limiter):
        """Test requests under rate limit threshold"""
        client_id = "test_client_1"
        
        # Make requests under the limit
        for i in range(5):
            allowed, retry_after = rate_limiter.is_allowed(
                client_id, max_requests=10, window_seconds=60
            )
            assert allowed is True
            assert retry_after is None
    
    def test_rate_limit_exceeded(self, rate_limiter):
        """Test rate limit exceeded"""
        client_id = "test_client_2"
        max_requests = 3
        
        # Make requests up to the limit
        for i in range(max_requests):
            allowed, _ = rate_limiter.is_allowed(
                client_id, max_requests=max_requests, window_seconds=60
            )
            assert allowed is True
        
        # Next request should be blocked
        allowed, retry_after = rate_limiter.is_allowed(
            client_id, max_requests=max_requests, window_seconds=60
        )
        assert allowed is False
        assert retry_after == 60
    
    def test_rate_limit_window_expiry(self, rate_limiter):
        """Test rate limit window expiry"""
        client_id = "test_client_3"
        
        with patch('time.time') as mock_time:
            # Start at time 0
            mock_time.return_value = 0
            
            # Make requests to hit limit
            for i in range(3):
                rate_limiter.is_allowed(client_id, max_requests=3, window_seconds=10)
            
            # Should be blocked
            allowed, _ = rate_limiter.is_allowed(client_id, max_requests=3, window_seconds=10)
            assert allowed is False
            
            # Advance time past window
            mock_time.return_value = 15
            
            # Should be allowed again
            allowed, _ = rate_limiter.is_allowed(client_id, max_requests=3, window_seconds=10)
            assert allowed is True
    
    def test_reset_client(self, rate_limiter):
        """Test resetting client rate limit"""
        client_id = "test_client_4"
        
        # Hit rate limit
        for i in range(5):
            rate_limiter.is_allowed(client_id, max_requests=3)
        
        # Should be blocked
        allowed, _ = rate_limiter.is_allowed(client_id, max_requests=3)
        assert allowed is False
        
        # Reset client
        rate_limiter.reset_client(client_id)
        
        # Should be allowed again
        allowed, _ = rate_limiter.is_allowed(client_id, max_requests=3)
        assert allowed is True


class TestSecurityValidator:
    """Test security validation functionality"""
    
    def test_validate_user_input_success(self):
        """Test successful input validation"""
        valid_input = "Hello World 123"
        result = SecurityValidator.validate_user_input(valid_input)
        assert result == valid_input.strip()
    
    def test_validate_user_input_too_long(self):
        """Test input too long error"""
        long_input = "x" * 1001  # Default max is 1000
        
        with pytest.raises(A2AAuthenticationError) as exc_info:
            SecurityValidator.validate_user_input(long_input)
        
        assert "too long" in str(exc_info.value).lower()
    
    def test_validate_user_input_not_string(self):
        """Test non-string input error"""
        with pytest.raises(A2AAuthenticationError) as exc_info:
            SecurityValidator.validate_user_input(123)
        
        assert "must be string" in str(exc_info.value).lower()
    
    def test_validate_user_input_dangerous_content(self):
        """Test dangerous content detection"""
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<div onload='alert()'></div>",
            "<img onerror='alert()' src='x'>"
        ]
        
        for dangerous_input in dangerous_inputs:
            with pytest.raises(A2AAuthenticationError) as exc_info:
                SecurityValidator.validate_user_input(dangerous_input)
            
            assert "dangerous content" in str(exc_info.value).lower()
    
    def test_validate_email_success(self):
        """Test successful email validation"""
        valid_emails = [
            "test@example.com",
            "user.name+tag@domain.co.uk",
            "firstname-lastname@domain.org"
        ]
        
        for email in valid_emails:
            result = SecurityValidator.validate_email(email)
            assert result == email.lower().strip()
    
    def test_validate_email_invalid(self):
        """Test invalid email formats"""
        invalid_emails = [
            "not-an-email",
            "@domain.com",
            "user@",
            "user@domain",
            "user name@domain.com"  # space
        ]
        
        for email in invalid_emails:
            with pytest.raises(A2AAuthenticationError) as exc_info:
                SecurityValidator.validate_email(email)
            
            assert "invalid email" in str(exc_info.value).lower()
    
    def test_validate_password_strength_success(self):
        """Test successful password strength validation"""
        strong_passwords = [
            "StrongPassword123!",
            "MySecure@Pass1",
            "Complex!Pass2023"
        ]
        
        for password in strong_passwords:
            assert SecurityValidator.validate_password_strength(password) is True
    
    def test_validate_password_strength_too_short(self):
        """Test password too short"""
        short_password = "Short1!"
        
        with pytest.raises(A2AAuthenticationError) as exc_info:
            SecurityValidator.validate_password_strength(short_password)
        
        assert "at least" in str(exc_info.value)
    
    def test_validate_password_strength_missing_requirements(self):
        """Test password missing character requirements"""
        weak_passwords = [
            "onlylowercase123!",  # No uppercase
            "ONLYUPPERCASE123!",  # No lowercase
            "OnlyLetters!",       # No numbers
            "OnlyAlphanumeric123" # No special chars
        ]
        
        for password in weak_passwords:
            with pytest.raises(A2AAuthenticationError) as exc_info:
                SecurityValidator.validate_password_strength(password)
            
            assert "must contain" in str(exc_info.value)


class TestSecurityAuditor:
    """Test security auditing functionality"""
    
    @pytest.fixture
    def auditor(self):
        """Create security auditor instance"""
        return SecurityAuditor()
    
    def test_log_security_event(self, auditor):
        """Test logging security events"""
        auditor.log_security_event(
            event_type="login_attempt",
            user_id="user123",
            ip_address="192.168.1.1",
            details={"success": True}
        )
        
        assert len(auditor.audit_log) == 1
        event = auditor.audit_log[0]
        
        assert event["event_type"] == "login_attempt"
        assert event["user_id"] == "user123"
        assert event["ip_address"] == "192.168.1.1"
        assert event["details"]["success"] is True
        assert "timestamp" in event
    
    def test_get_audit_summary(self, auditor):
        """Test getting audit summary"""
        # Log multiple events
        events = [
            ("login_attempt", "user1"),
            ("login_attempt", "user2"),
            ("password_change", "user1"),
            ("logout", "user1")
        ]
        
        for event_type, user_id in events:
            auditor.log_security_event(event_type, user_id=user_id)
        
        summary = auditor.get_audit_summary()
        
        assert summary["total_events"] == 4
        assert summary["event_types"]["login_attempt"] == 2
        assert summary["event_types"]["password_change"] == 1
        assert summary["event_types"]["logout"] == 1
        assert len(summary["recent_events"]) <= 10


class TestUtilityFunctions:
    """Test security utility functions"""
    
    def test_generate_api_key(self):
        """Test API key generation"""
        api_key = generate_api_key()
        
        assert isinstance(api_key, str)
        assert len(api_key) > 30  # Should be long
        
        # Test custom length
        custom_key = generate_api_key(16)
        assert len(custom_key) > 16  # URL-safe encoding increases length
        
        # Test uniqueness
        key1 = generate_api_key()
        key2 = generate_api_key()
        assert key1 != key2
    
    def test_generate_csrf_token(self):
        """Test CSRF token generation"""
        token1 = generate_csrf_token()
        token2 = generate_csrf_token()
        
        assert isinstance(token1, str)
        assert isinstance(token2, str)
        assert len(token1) > 30
        assert token1 != token2
    
    def test_verify_csrf_token(self):
        """Test CSRF token verification"""
        token = generate_csrf_token()
        
        # Valid token
        assert verify_csrf_token(token, token) is True
        
        # Invalid token
        assert verify_csrf_token(token, "different_token") is False
        
        # Empty tokens
        assert verify_csrf_token("", "") is True
        assert verify_csrf_token("token", "") is False
    
    def test_mask_sensitive_data(self):
        """Test sensitive data masking"""
        # Credit card number
        card_number = "1234567890123456"
        masked = mask_sensitive_data(card_number)
        assert masked == "1234********3456"
        
        # Short data
        short_data = "123"
        masked_short = mask_sensitive_data(short_data)
        assert masked_short == "***"
        
        # Custom parameters
        email = "test@example.com"
        masked_email = mask_sensitive_data(email, mask_char="#", visible_chars=2)
        assert masked_email == "te###########om"
    
    def test_generate_secure_filename(self):
        """Test secure filename generation"""
        # Normal filename
        filename = generate_secure_filename("document.pdf")
        assert filename.endswith(".pdf")
        assert "_" in filename  # Should have timestamp and random suffix
        
        # Filename with dangerous characters
        dangerous_name = "../../../etc/passwd"
        safe_name = generate_secure_filename(dangerous_name)
        assert "../" not in safe_name
        assert "etc" in safe_name  # Content preserved but safe
        
        # Long filename
        long_name = "a" * 300 + ".txt"
        safe_long = generate_secure_filename(long_name)
        assert len(safe_long) <= 255
        assert safe_long.endswith(".txt")


class TestConvenienceFunctions:
    """Test module-level convenience functions"""
    
    def test_hash_password_convenience(self):
        """Test convenience hash_password function"""
        password = "TestPassword123!"
        hashed = hash_password(password)
        assert verify_password(password, hashed)
    
    def test_generate_secure_password_convenience(self):
        """Test convenience generate_secure_password function"""
        password = generate_secure_password()
        assert len(password) == 16
        assert isinstance(password, str)
    
    def test_encryption_convenience_functions(self):
        """Test convenience encryption functions"""
        data = "sensitive_data"
        encrypted = encrypt_data(data)
        decrypted = decrypt_data(encrypted)
        assert decrypted == data
    
    def test_validation_convenience_functions(self):
        """Test convenience validation functions"""
        # Valid input
        result = validate_user_input("valid input")
        assert result == "valid input"
        
        # Valid email
        email = validate_email("test@example.com")
        assert email == "test@example.com"
        
        # Valid password
        assert validate_password_strength("StrongPass123!") is True
    
    def test_rate_limit_convenience(self):
        """Test rate limit convenience functions"""
        client_id = "test_client"
        
        # Should be allowed initially
        allowed, retry_after = validate_rate_limit(client_id)
        assert allowed is True
        assert retry_after is None