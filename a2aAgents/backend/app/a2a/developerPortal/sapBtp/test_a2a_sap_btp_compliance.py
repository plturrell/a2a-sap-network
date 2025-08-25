"""
Test A2A Protocol Compliance for SAP BTP Integration
Verifies that SAP BTP services work through A2A blockchain messaging
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from .a2aSapBtpAgent import (
    A2ASAPBTPAgent,
    A2ASAPBTPRequest, 
    A2ASAPBTPResponse,
    SAPBTPServiceType,
    send_sap_btp_request
)
from .destinationService import DestinationService
from .rbacService import RBACService
from ...core.externalServiceBridge import ExternalServiceBridgeAgent
from ...sdk.types import A2AMessage, MessageType


class TestA2ASAPBTPCompliance:
    """Test suite for A2A SAP BTP compliance"""

    @pytest.fixture
    async def sap_btp_agent(self):
        """Create SAP BTP agent for testing"""
        agent = A2ASAPBTPAgent("test_sap_btp_agent")
        await agent.initialize()
        return agent

    @pytest.fixture
    def mock_network_client(self):
        """Mock A2A network client"""
        mock_client = Mock()
        mock_client.send_a2a_message = AsyncMock()
        return mock_client

    @pytest.fixture
    def sample_destination_config(self):
        """Sample destination service configuration"""
        return {
            "uri": "https://destination-configuration.sap.hana.ondemand.com",
            "clientid": "test_client_id",
            "clientsecret": "test_client_secret",
            "url": "https://auth.sap.hana.ondemand.com"
        }

    @pytest.fixture
    def sample_xsuaa_config(self):
        """Sample XSUAA configuration"""
        return {
            "url": "https://auth.sap.hana.ondemand.com",
            "clientid": "xsuaa_client_id",
            "clientsecret": "xsuaa_client_secret",
            "xsappname": "a2a-test-app"
        }

    async def test_no_direct_http_imports(self):
        """Test that no direct HTTP client imports exist in fixed files"""
        
        # Check destinationService.py
        with open("/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/developerPortal/sapBtp/destinationService.py", "r") as f:
            destination_content = f.read()
        
        # Should not contain httpx imports
        assert "import httpx" not in destination_content
        assert "from httpx" not in destination_content
        assert "httpx.AsyncClient()" not in destination_content
        
        # Should contain A2A imports
        assert "from .a2aSapBtpAgent import" in destination_content
        assert "send_sap_btp_request" in destination_content

    async def test_destination_service_a2a_compliance(self, sap_btp_agent, mock_network_client):
        """Test that Destination Service uses A2A protocol"""
        
        # Mock the network client
        sap_btp_agent.network_client = mock_network_client
        
        # Mock successful token response
        mock_network_client.send_a2a_message.return_value = {
            "success": True,
            "data": {
                "access_token": "test_token_123",
                "expires_in": 3600
            }
        }
        
        # Test token request
        request = A2ASAPBTPRequest(
            service_type=SAPBTPServiceType.DESTINATION_SERVICE,
            operation="get_access_token",
            parameters={"grant_type": "client_credentials"}
        )
        
        response = await sap_btp_agent._handle_sap_btp_request(request)
        
        # Verify response
        assert response.success is True
        assert "access_token" in response.data
        
        # Verify A2A message was sent
        mock_network_client.send_a2a_message.assert_called_once()
        call_args = mock_network_client.send_a2a_message.call_args
        assert call_args[1]["to_agent"] == "external_service_bridge"
        assert call_args[1]["message_type"] == "SERVICE_REQUEST"

    async def test_destination_config_retrieval_a2a(self, sap_btp_agent, mock_network_client):
        """Test destination configuration retrieval through A2A"""
        
        sap_btp_agent.network_client = mock_network_client
        
        # Mock responses for token and destination
        mock_network_client.send_a2a_message.side_effect = [
            # Token response
            {
                "success": True,
                "data": {"access_token": "test_token"}
            },
            # Destination response
            {
                "success": True,
                "data": {
                    "Name": "test_destination",
                    "URL": "https://api.test.com",
                    "Authentication": "OAuth2ClientCredentials"
                }
            }
        ]
        
        request = A2ASAPBTPRequest(
            service_type=SAPBTPServiceType.DESTINATION_SERVICE,
            operation="get_destination",
            parameters={
                "destination_name": "test_destination",
                "service_url": "https://destination-service.com"
            }
        )
        
        response = await sap_btp_agent._handle_sap_btp_request(request)
        
        assert response.success is True
        assert "Name" in response.data
        assert response.data["Name"] == "test_destination"

    async def test_xsuaa_jwt_validation_a2a(self, sap_btp_agent, mock_network_client):
        """Test XSUAA JWT validation through A2A protocol"""
        
        sap_btp_agent.network_client = mock_network_client
        
        # Mock JWT validation response
        mock_network_client.send_a2a_message.return_value = {
            "success": True,
            "data": {
                "decoded_token": {
                    "user_id": "test_user",
                    "user_name": "testuser@example.com",
                    "scope": ["a2a-test-app.User"],
                    "exp": (datetime.utcnow() + timedelta(hours=1)).timestamp()
                }
            }
        }
        
        request = A2ASAPBTPRequest(
            service_type=SAPBTPServiceType.XSUAA_SERVICE,
            operation="validate_token",
            parameters={
                "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "xsuaa_config": {"clientid": "test_client"}
            }
        )
        
        response = await sap_btp_agent._handle_sap_btp_request(request)
        
        assert response.success is True
        assert "decoded_token" in response.data
        assert response.data["decoded_token"]["user_id"] == "test_user"

    async def test_destination_service_integration(self, sample_destination_config):
        """Test DestinationService class uses A2A protocol"""
        
        service = DestinationService(sample_destination_config)
        
        # Mock the send_sap_btp_request function
        with patch('a2aAgents.backend.app.a2a.developerPortal.sapBtp.destinationService.send_sap_btp_request') as mock_send:
            mock_send.return_value = A2ASAPBTPResponse(
                success=True,
                data={"access_token": "test_token"},
                service_type=SAPBTPServiceType.DESTINATION_SERVICE,
                operation="get_access_token"
            )
            
            # Test token retrieval
            token = await service.get_access_token()
            
            # Verify A2A request was made
            mock_send.assert_called_once()
            call_args = mock_send.call_args[1]
            assert call_args["service_type"] == SAPBTPServiceType.DESTINATION_SERVICE
            assert call_args["operation"] == "get_access_token"
            assert call_args["from_agent"] == "destination_service"

    async def test_rbac_service_integration(self, sample_xsuaa_config):
        """Test RBACService class uses A2A protocol for JWT validation"""
        
        rbac = RBACService(sample_xsuaa_config)
        
        # Mock the send_sap_btp_request function
        with patch('a2aAgents.backend.app.a2a.developerPortal.sapBtp.rbacService.send_sap_btp_request') as mock_send:
            mock_send.return_value = A2ASAPBTPResponse(
                success=True,
                data={
                    "decoded_token": {
                        "user_id": "test_user",
                        "user_name": "testuser@example.com",
                        "email": "testuser@example.com",
                        "scope": ["a2a-test-app.User"]
                    }
                },
                service_type=SAPBTPServiceType.XSUAA_SERVICE,
                operation="validate_token"
            )
            
            # Test token validation
            user_info = await rbac.validate_token("dummy_jwt_token")
            
            # Verify A2A request was made
            mock_send.assert_called_once()
            call_args = mock_send.call_args[1]
            assert call_args["service_type"] == SAPBTPServiceType.XSUAA_SERVICE
            assert call_args["operation"] == "validate_token"
            assert call_args["from_agent"] == "rbac_service"
            
            # Verify user info
            assert user_info.user_id == "test_user"
            assert user_info.user_name == "testuser@example.com"

    async def test_external_service_bridge_agent(self):
        """Test External Service Bridge Agent functionality"""
        
        bridge = ExternalServiceBridgeAgent()
        await bridge.initialize()
        
        # Test domain validation
        assert bridge._is_domain_allowed("authentication.sap.hana.ondemand.com")
        assert not bridge._is_domain_allowed("malicious-site.com")
        
        # Test rate limiting
        assert await bridge._check_rate_limit("oauth2_token", "test_agent")
        
        # Add many requests to test rate limiting
        for i in range(60):  # Default limit is 60 for oauth2_token
            await bridge._check_rate_limit("oauth2_token", "test_agent")
        
        # Should now be rate limited
        assert not await bridge._check_rate_limit("oauth2_token", "test_agent")

    async def test_a2a_message_handling(self, sap_btp_agent):
        """Test A2A message handling in SAP BTP agent"""
        
        # Create test A2A message
        test_message = A2AMessage(
            id="test_msg_123",
            type=MessageType.REQUEST,
            sender="test_sender",
            receiver=sap_btp_agent.agent_id,
            payload={
                "request_type": "sap_btp_service",
                "service_request": {
                    "service_type": "destination_service",
                    "operation": "get_access_token",
                    "parameters": {"grant_type": "client_credentials"}
                }
            },
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Mock network client
        mock_client = Mock()
        mock_client.send_a2a_message = AsyncMock(return_value={
            "success": True,
            "data": {"access_token": "test_token"}
        })
        sap_btp_agent.network_client = mock_client
        
        # Handle message
        response_message = await sap_btp_agent.handle_message(test_message)
        
        # Verify response
        assert response_message is not None
        assert response_message.type == MessageType.RESPONSE
        assert response_message.sender == sap_btp_agent.agent_id
        assert response_message.receiver == "test_sender"
        assert response_message.in_reply_to == "test_msg_123"
        assert "service_response" in response_message.payload

    async def test_error_handling(self, sap_btp_agent):
        """Test error handling in A2A SAP BTP integration"""
        
        # Mock network client to return error
        mock_client = Mock()
        mock_client.send_a2a_message = AsyncMock(return_value={
            "success": False,
            "error": "Network error"
        })
        sap_btp_agent.network_client = mock_client
        
        request = A2ASAPBTPRequest(
            service_type=SAPBTPServiceType.DESTINATION_SERVICE,
            operation="get_access_token",
            parameters={}
        )
        
        response = await sap_btp_agent._handle_sap_btp_request(request)
        
        # Verify error response
        assert response.success is False
        assert "Network error" in response.error

    def test_protocol_compliance_documentation(self):
        """Test that files contain A2A protocol compliance notices"""
        
        files_to_check = [
            "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/developerPortal/sapBtp/destinationService.py",
            "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/developerPortal/sapBtp/rbacService.py"
        ]
        
        for file_path in files_to_check:
            with open(file_path, "r") as f:
                content = f.read()
            
            # Should contain A2A protocol compliance notices
            assert "A2A Protocol" in content
            # Should not contain WARNING comments about violations
            assert "WARNING: httpx AsyncClient usage violates A2A protocol" not in content


@pytest.mark.asyncio
class TestIntegrationFlow:
    """Integration tests for complete SAP BTP flows"""
    
    async def test_complete_destination_flow(self):
        """Test complete destination retrieval flow through A2A protocol"""
        
        # This would be a comprehensive integration test
        # that verifies the entire flow from API request
        # through A2A messaging to external service bridge
        
        # Initialize agents
        sap_btp_agent = A2ASAPBTPAgent()
        bridge_agent = ExternalServiceBridgeAgent()
        
        await sap_btp_agent.initialize()
        await bridge_agent.initialize()
        
        # Mock the actual HTTP calls in bridge agent
        with patch.object(bridge_agent, '_execute_http_request') as mock_http:
            mock_http.return_value = {
                "status_code": 200,
                "data": {"access_token": "test_token", "expires_in": 3600},
                "headers": {"content-type": "application/json"}
            }
            
            # Test the flow
            result = await send_sap_btp_request(
                service_type=SAPBTPServiceType.DESTINATION_SERVICE,
                operation="get_access_token",
                parameters={"grant_type": "client_credentials"},
                from_agent="test_client"
            )
            
            # In a real test, this would verify the complete message flow
            assert isinstance(result, A2ASAPBTPResponse)


if __name__ == "__main__":
    """Run tests"""
    pytest.main([__file__, "-v"])