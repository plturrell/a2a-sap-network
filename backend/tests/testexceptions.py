"""
Comprehensive test suite for A2A exception handling
Tests custom exception hierarchy and error handling patterns
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from app.core.exceptions import (
    A2ABaseException,
    A2AConfigurationError,
    A2AValidationError,
    A2AAuthenticationError,
    A2AAuthorizationError,
    A2ATokenExpiredError,
    A2AInvalidTokenError,
    A2ADatabaseError,
    A2AConnectionError,
    A2AConstraintViolationError,
    A2ANetworkError,
    A2AExternalServiceError,
    A2ATimeoutError,
    A2ARateLimitError,
    A2AAgentError,
    A2AAgentUnavailableError,
    A2AAgentTimeoutError,
    A2AAgentCommunicationError,
    A2ABusinessLogicError,
    A2AResourceNotFoundError,
    A2AResourceConflictError,
    A2ASAPIntegrationError,
    A2AHANAError,
    A2ABTPError,
    ErrorCategory,
    ErrorSeverity,
    create_error_response,
    handle_exception_chain,
    EXCEPTION_REGISTRY
)


class TestA2ABaseException:
    """Test base exception functionality"""
    
    def test_basic_exception_creation(self):
        """Test creating basic exception"""
        exc = A2ABaseException("Test error message")
        
        assert str(exc) == "Test error message"
        assert exc.message == "Test error message"
        assert exc.error_code == "A2ABaseException"
        assert exc.category == ErrorCategory.SYSTEM
        assert exc.severity == ErrorSeverity.MEDIUM
        assert isinstance(exc.timestamp, datetime)
        assert exc.context == {}
        assert exc.original_error is None
    
    def test_exception_with_custom_parameters(self):
        """Test exception with all custom parameters"""
        original_error = ValueError("Original error")
        context = {"user_id": "123", "action": "test"}
        
        exc = A2ABaseException(
            message="Custom error",
            error_code="CUSTOM_001",
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            original_error=original_error
        )
        
        assert exc.message == "Custom error"
        assert exc.error_code == "CUSTOM_001"
        assert exc.category == ErrorCategory.AUTHENTICATION
        assert exc.severity == ErrorSeverity.HIGH
        assert exc.context == context
        assert exc.original_error == original_error
    
    def test_exception_to_dict(self):
        """Test converting exception to dictionary"""
        exc = A2ABaseException(
            "Test message",
            error_code="TEST_001",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            context={"field": "username"}
        )
        
        result = exc.to_dict()
        
        expected_keys = {"error_code", "message", "category", "severity", "timestamp", "context"}
        assert set(result.keys()) == expected_keys
        assert result["error_code"] == "TEST_001"
        assert result["message"] == "Test message"
        assert result["category"] == "validation"
        assert result["severity"] == "low"
        assert result["context"] == {"field": "username"}
    
    @patch('app.core.exceptions.logger')
    def test_exception_logging(self, mock_logger):
        """Test that exceptions are logged appropriately"""
        exc = A2ABaseException(
            "Test error",
            severity=ErrorSeverity.CRITICAL
        )
        
        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == 50  # CRITICAL log level
        assert "Test error" in call_args[0][1]


class TestSpecificExceptions:
    """Test specific exception types"""
    
    def test_configuration_error(self):
        """Test configuration error"""
        exc = A2AConfigurationError("Missing config key", config_key="database_url")
        
        assert exc.category == ErrorCategory.CONFIGURATION
        assert exc.severity == ErrorSeverity.HIGH
        assert exc.context["config_key"] == "database_url"
    
    def test_validation_error(self):
        """Test validation error with field errors"""
        field_errors = {
            "email": ["Invalid format", "Too long"],
            "password": ["Too weak"]
        }
        
        exc = A2AValidationError("Validation failed", field_errors=field_errors)
        
        assert exc.category == ErrorCategory.VALIDATION
        assert exc.context["field_errors"] == field_errors
    
    def test_authentication_errors(self):
        """Test authentication error types"""
        # Basic auth error
        auth_exc = A2AAuthenticationError("Login failed")
        assert auth_exc.category == ErrorCategory.AUTHENTICATION
        assert auth_exc.severity == ErrorSeverity.HIGH
        
        # Token expired
        token_exc = A2ATokenExpiredError()
        assert token_exc.error_code == "TOKEN_EXPIRED"
        assert "expired" in token_exc.message.lower()
        
        # Invalid token
        invalid_exc = A2AInvalidTokenError()
        assert invalid_exc.error_code == "INVALID_TOKEN"
    
    def test_authorization_error(self):
        """Test authorization error with permissions"""
        required_perms = ["read_data", "write_data"]
        exc = A2AAuthorizationError(
            "Insufficient permissions",
            required_permissions=required_perms
        )
        
        assert exc.category == ErrorCategory.AUTHORIZATION
        assert exc.context["required_permissions"] == required_perms
    
    def test_database_errors(self):
        """Test database error types"""
        # Generic database error
        db_exc = A2ADatabaseError("Query failed", operation="SELECT", table="users")
        assert db_exc.category == ErrorCategory.DATABASE
        assert db_exc.context["operation"] == "SELECT"
        assert db_exc.context["table"] == "users"
        
        # Connection error
        conn_exc = A2AConnectionError()
        assert conn_exc.error_code == "DB_CONNECTION_FAILED"
        
        # Constraint violation
        constraint_exc = A2AConstraintViolationError(
            "Unique constraint failed",
            constraint_name="unique_email"
        )
        assert constraint_exc.error_code == "CONSTRAINT_VIOLATION"
        assert constraint_exc.context["constraint_name"] == "unique_email"
    
    def test_network_errors(self):
        """Test network error types"""
        # Basic network error
        net_exc = A2ANetworkError("Connection failed", endpoint="https://api.example.com")
        assert net_exc.category == ErrorCategory.NETWORK
        assert net_exc.context["endpoint"] == "https://api.example.com"
        
        # Timeout error
        timeout_exc = A2ATimeoutError("Request timeout", timeout_duration=30.0)
        assert timeout_exc.error_code == "REQUEST_TIMEOUT"
        assert timeout_exc.context["timeout_duration"] == 30.0
        
        # Rate limit error
        rate_exc = A2ARateLimitError("Too many requests", retry_after=60)
        assert rate_exc.error_code == "RATE_LIMIT_EXCEEDED"
        assert rate_exc.context["retry_after"] == 60
    
    def test_agent_errors(self):
        """Test agent-specific errors"""
        # Generic agent error
        agent_exc = A2AAgentError("Agent error", agent_id="agent_001")
        assert agent_exc.category == ErrorCategory.AGENT_COMMUNICATION
        assert agent_exc.context["agent_id"] == "agent_001"
        
        # Agent unavailable
        unavail_exc = A2AAgentUnavailableError(agent_id="agent_002")
        assert unavail_exc.error_code == "AGENT_UNAVAILABLE"
        assert unavail_exc.severity == ErrorSeverity.HIGH
        
        # Agent timeout
        timeout_exc = A2AAgentTimeoutError(
            agent_id="agent_003",
            timeout_duration=45.0
        )
        assert timeout_exc.error_code == "AGENT_TIMEOUT"
        assert timeout_exc.context["timeout_duration"] == 45.0
        
        # Agent communication error
        comm_exc = A2AAgentCommunicationError(
            "Failed to communicate",
            source_agent="agent_001",
            target_agent="agent_002"
        )
        assert comm_exc.error_code == "AGENT_COMMUNICATION_FAILED"
        assert comm_exc.context["source_agent"] == "agent_001"
        assert comm_exc.context["target_agent"] == "agent_002"
    
    def test_business_logic_errors(self):
        """Test business logic errors"""
        # Generic business error
        biz_exc = A2ABusinessLogicError("Rule violation", rule_name="max_daily_limit")
        assert biz_exc.category == ErrorCategory.BUSINESS_LOGIC
        assert biz_exc.context["rule_name"] == "max_daily_limit"
        
        # Resource not found
        not_found_exc = A2AResourceNotFoundError(
            "User not found",
            resource_type="user",
            resource_id="123"
        )
        assert not_found_exc.error_code == "RESOURCE_NOT_FOUND"
        assert not_found_exc.context["resource_type"] == "user"
        assert not_found_exc.context["resource_id"] == "123"
        
        # Resource conflict
        conflict_exc = A2AResourceConflictError(
            "Duplicate resource",
            conflict_type="duplicate_email"
        )
        assert conflict_exc.error_code == "RESOURCE_CONFLICT"
        assert conflict_exc.context["conflict_type"] == "duplicate_email"
    
    def test_sap_integration_errors(self):
        """Test SAP integration errors"""
        # Generic SAP error
        sap_exc = A2ASAPIntegrationError(
            "SAP call failed",
            sap_system="S4HANA",
            sap_error_code="RFC_001"
        )
        assert sap_exc.service_name == "SAP"
        assert sap_exc.error_code == "SAP_INTEGRATION_ERROR"
        assert sap_exc.context["sap_system"] == "S4HANA"
        assert sap_exc.context["sap_error_code"] == "RFC_001"
        
        # HANA error
        hana_exc = A2AHANAError("HANA query failed", hana_error_code="SQL_001")
        assert hana_exc.error_code == "HANA_ERROR"
        assert hana_exc.context["sap_system"] == "HANA"
        assert hana_exc.context["sap_error_code"] == "SQL_001"
        
        # BTP error
        btp_exc = A2ABTPError("BTP service failed", btp_service="destination")
        assert btp_exc.error_code == "BTP_ERROR"
        assert btp_exc.context["btp_service"] == "destination"


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_create_error_response_basic(self):
        """Test creating basic error response"""
        exc = A2AValidationError(
            "Validation error",
            field_errors={"email": ["Invalid"]}
        )
        
        response = create_error_response(exc)
        
        expected_keys = {"error_code", "message", "category", "severity", "timestamp", "context"}
        assert set(response.keys()) == expected_keys
        assert response["message"] == "Validation error"
        assert response["category"] == "validation"
    
    def test_create_error_response_no_context(self):
        """Test creating error response without context"""
        exc = A2AValidationError("Error", field_errors={"field": ["error"]})
        
        response = create_error_response(exc, include_context=False)
        
        assert "context" not in response
        assert response["message"] == "Error"
    
    def test_create_error_response_with_stack_trace(self):
        """Test creating error response with stack trace"""
        original_error = ValueError("Original")
        exc = A2ABaseException("Wrapped error", original_error=original_error)
        
        response = create_error_response(
            exc,
            include_context=True,
            include_stack_trace=True
        )
        
        assert "stack_trace" in response
        assert isinstance(response["stack_trace"], list)
    
    def test_handle_exception_chain_a2a_exception(self):
        """Test handling A2A exceptions (pass-through)"""
        original_exc = A2AValidationError("Validation failed")
        
        result = handle_exception_chain(original_exc)
        
        assert result is original_exc
    
    def test_handle_exception_chain_mapped_exceptions(self):
        """Test mapping common Python exceptions"""
        # Connection error
        conn_err = ConnectionError("Network failed")
        result = handle_exception_chain(conn_err)
        assert isinstance(result, A2ANetworkError)
        assert result.original_error is conn_err
        
        # Timeout error
        timeout_err = TimeoutError("Request timeout")
        result = handle_exception_chain(timeout_err)
        assert isinstance(result, A2ATimeoutError)
        
        # Value error
        value_err = ValueError("Invalid value")
        result = handle_exception_chain(value_err)
        assert isinstance(result, A2AValidationError)
        
        # Key error
        key_err = KeyError("missing_key")
        result = handle_exception_chain(key_err)
        assert isinstance(result, A2AConfigurationError)
        
        # File not found
        file_err = FileNotFoundError("File missing")
        result = handle_exception_chain(file_err)
        assert isinstance(result, A2AResourceNotFoundError)
        
        # Permission error
        perm_err = PermissionError("Access denied")
        result = handle_exception_chain(perm_err)
        assert isinstance(result, A2AAuthorizationError)
    
    def test_handle_exception_chain_generic_exception(self):
        """Test handling unmapped exceptions"""
        generic_err = RuntimeError("Runtime error")
        result = handle_exception_chain(generic_err)
        
        assert isinstance(result, A2ABaseException)
        assert result.original_error is generic_err
        assert result.error_code == "WRAPPED_RUNTIMEERROR"
        assert str(result) == "Runtime error"


class TestExceptionRegistry:
    """Test exception registry"""
    
    def test_exception_registry_completeness(self):
        """Test that exception registry contains expected entries"""
        expected_codes = [
            "AUTH_001", "AUTH_002", "AUTH_003", "AUTH_004",
            "DB_001", "DB_002", "DB_003",
            "NET_001", "NET_002", "EXT_001", "EXT_002",
            "AGT_001", "AGT_002", "AGT_003", "AGT_004",
            "BIZ_001", "BIZ_002", "BIZ_003",
            "SAP_001", "SAP_002", "SAP_003",
            "SYS_001", "SYS_002"
        ]
        
        for code in expected_codes:
            assert code in EXCEPTION_REGISTRY
            assert issubclass(EXCEPTION_REGISTRY[code], A2ABaseException)
    
    def test_exception_registry_auth_codes(self):
        """Test authentication exception codes"""
        assert EXCEPTION_REGISTRY["AUTH_001"] == A2AAuthenticationError
        assert EXCEPTION_REGISTRY["AUTH_002"] == A2AAuthorizationError
        assert EXCEPTION_REGISTRY["AUTH_003"] == A2ATokenExpiredError
        assert EXCEPTION_REGISTRY["AUTH_004"] == A2AInvalidTokenError
    
    def test_exception_registry_database_codes(self):
        """Test database exception codes"""
        assert EXCEPTION_REGISTRY["DB_001"] == A2ADatabaseError
        assert EXCEPTION_REGISTRY["DB_002"] == A2AConnectionError
        assert EXCEPTION_REGISTRY["DB_003"] == A2AConstraintViolationError


class TestExceptionIntegration:
    """Test exception integration with other components"""
    
    @pytest.mark.asyncio
    async def test_exception_in_async_context(self):
        """Test exceptions work in async contexts"""
        async def failing_async_function():
            raise A2AAgentTimeoutError("Async timeout", agent_id="async_agent")
        
        with pytest.raises(A2AAgentTimeoutError) as exc_info:
            await failing_async_function()
        
        assert exc_info.value.agent_id == "async_agent"
        assert exc_info.value.error_code == "AGENT_TIMEOUT"
    
    def test_exception_serialization(self):
        """Test exception can be serialized/deserialized"""
        import json
        
        exc = A2AValidationError(
            "Validation failed",
            field_errors={"email": ["Invalid format"]}
        )
        
        # Convert to dict and serialize
        exc_dict = exc.to_dict()
        json_str = json.dumps(exc_dict, default=str)
        
        # Deserialize
        restored_dict = json.loads(json_str)
        
        assert restored_dict["message"] == "Validation failed"
        assert restored_dict["category"] == "validation"
        assert restored_dict["context"]["field_errors"]["email"] == ["Invalid format"]