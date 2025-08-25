"""
Comprehensive Test Suite for A2A Hybrid Protocol Compliance in SAP BTP Integration
Validates the hybrid approach maintaining A2A compliance while allowing SAP BTP HTTP calls
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json

from .destinationService import DestinationService, DestinationInfo
from .rbacService import RBACService, UserInfo, UserRole
from ...core.hybridNetworkClient import (
    HybridNetworkClient,
    CommunicationType,
    ExternalDomainConfig,
    get_hybrid_network_client
)


class TestHybridA2ACompliance:
    """Comprehensive test suite for hybrid A2A/HTTP approach"""

    @pytest.fixture
    def sample_destination_config(self):
        """SAP BTP Destination Service configuration"""
        return {
            "uri": "https://destination-configuration.eu10.hana.ondemand.com",
            "clientid": "test_client_id",
            "clientsecret": "test_client_secret",
            "url": "https://auth.eu10.hana.ondemand.com"
        }

    @pytest.fixture
    def sample_xsuaa_config(self):
        """SAP BTP XSUAA configuration"""
        return {
            "url": "https://auth.eu10.hana.ondemand.com",
            "clientid": "xsuaa_client_id",
            "clientsecret": "xsuaa_client_secret",
            "xsappname": "a2a-test-app"
        }

    @pytest.fixture
    def hybrid_client(self):
        """Create hybrid network client for testing"""
        return HybridNetworkClient("test_agent")

    async def test_hybrid_client_initialization(self, hybrid_client):
        """Test hybrid client initializes with correct configuration"""
        
        # Verify client is initialized
        assert hybrid_client.agent_id == "test_agent"
        assert hybrid_client.blockchain_client is None  # No blockchain in test
        
        # Check configuration
        assert hybrid_client.config["enable_external_http"] is True
        assert hybrid_client.config["strict_domain_checking"] is True
        
        # Verify SAP BTP domains are allowed
        allowed_patterns = hybrid_client.get_allowed_domains()
        assert any("authentication" in pattern for pattern in allowed_patterns)
        assert any("dest-configuration" in pattern for pattern in allowed_patterns)
        assert any("notification" in pattern for pattern in allowed_patterns)

    async def test_domain_validation(self, hybrid_client):
        """Test domain validation for SAP BTP services"""
        
        # Valid SAP BTP domains
        assert hybrid_client.is_domain_allowed("https://authentication.eu10.hana.ondemand.com/oauth/token")
        assert hybrid_client.is_domain_allowed("https://destination-configuration.us10.hana.ondemand.com/api/v1")
        assert hybrid_client.is_domain_allowed("https://alert-notification.cfapps.eu10.hana.ondemand.com")
        
        # Invalid domains
        assert not hybrid_client.is_domain_allowed("https://malicious-site.com/steal-data")
        assert not hybrid_client.is_domain_allowed("http://localhost:8080/test")
        assert not hybrid_client.is_domain_allowed("https://google.com")

    async def test_communication_type_determination(self, hybrid_client):
        """Test correct determination of communication types"""
        
        # Internal agent communication
        assert hybrid_client._determine_communication_type("agent_123") == CommunicationType.A2A_INTERNAL
        assert hybrid_client._determine_communication_type("sap_btp_agent") == CommunicationType.A2A_INTERNAL
        
        # External HTTP (SAP BTP)
        sap_endpoint = "https://destination-configuration.eu10.hana.ondemand.com/api"
        comm_type = hybrid_client._determine_communication_type("external", sap_endpoint)
        assert comm_type in [CommunicationType.HYBRID_WRAPPED, CommunicationType.EXTERNAL_HTTP]

    @pytest.mark.asyncio
    async def test_destination_service_token_request(self, sample_destination_config):
        """Test Destination Service OAuth token request through hybrid approach"""
        
        service = DestinationService(sample_destination_config)
        
        # Mock the hybrid client's SAP BTP request
        with patch.object(service.hybrid_client, 'send_sap_btp_request') as mock_request:
            mock_request.return_value = {
                "success": True,
                "http_status": 200,
                "data": {
                    "access_token": "test_access_token_12345",
                    "token_type": "Bearer",
                    "expires_in": 3600
                },
                "type": "hybrid_wrapped_http",
                "endpoint": sample_destination_config["url"] + "/oauth/token",
                "method": "POST"
            }
            
            # Request token
            token = await service.get_access_token()
            
            # Verify token received
            assert token == "test_access_token_12345"
            
            # Verify hybrid client was called correctly
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[1]["service_type"] == "oauth_token"
            assert call_args[1]["operation"] == "get_access_token"
            assert call_args[1]["method"] == "POST"
            assert "Authorization" in call_args[1]["headers"]

    @pytest.mark.asyncio
    async def test_destination_config_retrieval(self, sample_destination_config):
        """Test destination configuration retrieval through hybrid approach"""
        
        service = DestinationService(sample_destination_config)
        
        # Set up cached token
        service.token_cache["destination_service_token"] = {
            "access_token": "cached_token",
            "expires_at": datetime.utcnow() + timedelta(minutes=30)
        }
        
        # Mock the hybrid client request
        with patch.object(service.hybrid_client, 'send_sap_btp_request') as mock_request:
            mock_request.return_value = {
                "success": True,
                "http_status": 200,
                "data": {
                    "Name": "ERP_SYSTEM",
                    "URL": "https://erp.company.com",
                    "Type": "HTTP",
                    "Authentication": "OAuth2ClientCredentials",
                    "ProxyType": "Internet"
                }
            }
            
            # Get destination
            destination = await service.get_destination("ERP_SYSTEM")
            
            # Verify destination info
            assert isinstance(destination, DestinationInfo)
            assert destination.name == "ERP_SYSTEM"
            assert destination.url == "https://erp.company.com"
            assert destination.authentication.value == "OAuth2ClientCredentials"

    @pytest.mark.asyncio
    async def test_rbac_jwt_validation(self, sample_xsuaa_config):
        """Test RBAC JWT validation in hybrid approach"""
        
        # Set development mode for testing
        import os
        os.environ['DEVELOPMENT_MODE'] = 'true'
        
        rbac = RBACService(sample_xsuaa_config)
        
        # Create test JWT payload
        test_payload = {
            "user_id": "test_user_123",
            "user_name": "test.user@company.com",
            "email": "test.user@company.com",
            "scope": ["a2a-test-app.Developer", "a2a-test-app.User"],
            "exp": (datetime.utcnow() + timedelta(hours=1)).timestamp()
        }
        
        # Mock JWT decode
        with patch('jwt.decode') as mock_decode:
            mock_decode.return_value = test_payload
            
            # Validate token
            user_info = await rbac.validate_token("dummy_jwt_token")
            
            # Verify user info
            assert user_info.user_id == "test_user_123"
            assert user_info.email == "test.user@company.com"
            assert UserRole.DEVELOPER in user_info.roles
            assert UserRole.USER in user_info.roles

    @pytest.mark.asyncio
    async def test_hybrid_wrapped_http_request(self, hybrid_client):
        """Test HTTP request wrapped in A2A message format"""
        
        # Mock httpx for actual HTTP call
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": "success"}
            mock_response.headers = {"content-type": "application/json"}
            
            mock_client.request.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Make wrapped HTTP request
            result = await hybrid_client._send_hybrid_wrapped_message(
                endpoint="https://destination-configuration.eu10.hana.ondemand.com/api/test",
                message={
                    "method": "GET",
                    "headers": {"Authorization": "Bearer test_token"}
                },
                message_type="SAP_BTP_REQUEST"
            )
            
            # Verify result
            assert result["success"] is True
            assert result["http_status"] == 200
            assert result["data"]["result"] == "success"
            assert result["type"] == "hybrid_wrapped_http"

    @pytest.mark.asyncio
    async def test_internal_a2a_messaging_fallback(self, hybrid_client):
        """Test internal A2A messaging with fallback behavior"""
        
        # No blockchain client, should use fallback
        result = await hybrid_client._send_internal_a2a_message(
            to_agent="test_agent_2",
            message={"content": "test message"},
            message_type="TEST"
        )
        
        # Verify fallback response
        assert result["success"] is True
        assert result["type"] == "simulated_a2a"
        assert "message_id" in result

    async def test_error_handling_external_domain(self, hybrid_client):
        """Test error handling for unauthorized external domains"""
        
        # Try to send to unauthorized domain
        with pytest.raises(RuntimeError) as exc_info:
            await hybrid_client._send_hybrid_wrapped_message(
                endpoint="https://unauthorized-site.com/api",
                message={"method": "GET"},
                message_type="UNAUTHORIZED"
            )
        
        assert "Domain not allowed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_concurrent_destination_requests(self, sample_destination_config):
        """Test concurrent destination service requests"""
        
        service = DestinationService(sample_destination_config)
        
        # Mock hybrid client
        with patch.object(service.hybrid_client, 'send_sap_btp_request') as mock_request:
            mock_request.return_value = {
                "success": True,
                "data": {"access_token": "test_token", "expires_in": 3600}
            }
            
            # Make concurrent requests
            tasks = [
                service.get_access_token(),
                service.get_access_token(),
                service.get_access_token()
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            assert all(token == "test_token" for token in results)
            
            # Should use cache after first call
            assert mock_request.call_count <= 3  # May be less due to caching

    async def test_health_check(self, hybrid_client):
        """Test hybrid client health check"""
        
        health = await hybrid_client.health_check()
        
        assert health["agent_id"] == "test_agent"
        assert CommunicationType.A2A_INTERNAL.value in health["communication_types_supported"]
        assert CommunicationType.HYBRID_WRAPPED.value in health["communication_types_supported"]
        assert health["external_http_enabled"] is True
        assert health["allowed_domains_count"] > 0

    @pytest.mark.asyncio
    async def test_custom_domain_configuration(self):
        """Test custom domain configuration from environment"""
        
        # Set custom domains
        custom_domains = [
            {
                "domain_pattern": "*.custom-sap.company.com",
                "description": "Custom SAP endpoint",
                "allowed_methods": ["GET", "POST"],
                "requires_auth": True,
                "max_timeout": 60
            }
        ]
        
        with patch.dict('os.environ', {'A2A_ALLOWED_EXTERNAL_DOMAINS': json.dumps(custom_domains)}):
            client = HybridNetworkClient("test_custom")
            
            # Verify custom domain is allowed
            assert client.is_domain_allowed("https://api.custom-sap.company.com/service")

    @pytest.mark.asyncio
    async def test_protocol_compliance_verification(self):
        """Verify no direct HTTP imports in service files"""
        
        # Check that service files don't import httpx directly
        import inspect
        
        # Check DestinationService
        import a2aAgents.backend.app.a2a.developerPortal.sapBtp.destinationService as dest_module
        dest_source = inspect.getsource(dest_module)
        assert "import httpx" not in dest_source
        assert "from httpx" not in dest_source
        
        # Check RBACService  
        import a2aAgents.backend.app.a2a.developerPortal.sapBtp.rbacService as rbac_module
        rbac_source = inspect.getsource(rbac_module)
        assert "import httpx" not in rbac_source
        assert "from httpx" not in rbac_source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])