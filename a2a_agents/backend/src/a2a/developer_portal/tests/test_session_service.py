"""
Real tests for session service with Redis storage
These tests verify actual session storage and retrieval functionality
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import sys
import redis

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sap_btp.session_service import (
    SessionService,
    SessionInfo,
    SessionStatus,
    SessionActivity,
    get_session_service,
    initialize_session_service
)
from sap_btp.rbac_service import UserInfo, UserRole


class TestSessionService:
    """Test session service functionality"""
    
    @pytest.fixture
    def session_config(self):
        """Test session configuration"""
        return {
            "session_timeout_minutes": 30,
            "max_sessions_per_user": 3,
            "cleanup_interval_minutes": 1,
            "enable_cleanup_task": False,  # Disable background task for tests
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 1,  # Use test database
                "password": None
            }
        }
    
    @pytest.fixture
    def user_info(self):
        """Test user info"""
        return UserInfo(
            user_id="test-user-123",
            user_name="Test User",
            email="test@example.com",
            given_name="Test",
            family_name="User",
            roles=[UserRole.DEVELOPER, UserRole.USER],
            scopes=["read", "write"],
            tenant_id="test-tenant"
        )
    
    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request"""
        request = Mock()
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.headers = {"user-agent": "Test Agent"}
        return request
    
    @pytest.fixture
    async def session_service(self, session_config):
        """Create session service instance"""
        service = SessionService(session_config)
        yield service
        
        # Cleanup test data
        if service.use_redis:
            try:
                keys = service.redis_client.keys("session:*")
                if keys:
                    service.redis_client.delete(*keys)
                keys = service.redis_client.keys("user_sessions:*")
                if keys:
                    service.redis_client.delete(*keys)
            except Exception:
                pass
    
    def test_initialization_with_redis(self, session_config):
        """Test that session service initializes with Redis"""
        service = SessionService(session_config)
        
        # Check Redis connection attempt
        assert service.redis_client is not None
        assert service.session_timeout.total_seconds() == 30 * 60
        assert service.max_sessions_per_user == 3
    
    def test_initialization_without_redis(self, session_config):
        """Test fallback to in-memory storage"""
        # Mock Redis to fail connection
        with patch('redis.Redis') as mock_redis:
            mock_redis.return_value.ping.side_effect = Exception("Redis unavailable")
            
            service = SessionService(session_config)
            
            assert service.use_redis is False
            assert isinstance(service.memory_sessions, dict)
    
    @pytest.mark.asyncio
    async def test_create_session_redis(self, session_service, user_info, mock_request):
        """Test creating session with Redis storage"""
        if not session_service.use_redis:
            pytest.skip("Redis not available for testing")
        
        access_token = "test-access-token"
        refresh_token = "test-refresh-token"
        
        session_info = await session_service.create_session(
            user_info, mock_request, access_token, refresh_token
        )
        
        assert session_info is not None
        assert session_info.user_info.user_id == user_info.user_id
        assert session_info.access_token == access_token
        assert session_info.refresh_token == refresh_token
        assert session_info.status == SessionStatus.ACTIVE
        assert session_info.ip_address == "127.0.0.1"
        
        # Verify stored in Redis
        key = f"session:{session_info.session_id}"
        stored_data = session_service.redis_client.get(key)
        assert stored_data is not None
    
    @pytest.mark.asyncio
    async def test_create_session_memory(self, session_config, user_info, mock_request):
        """Test creating session with in-memory storage"""
        # Force in-memory mode
        with patch('redis.Redis') as mock_redis:
            mock_redis.return_value.ping.side_effect = Exception("Redis unavailable")
            
            service = SessionService(session_config)
            access_token = "test-access-token"
            
            session_info = await service.create_session(
                user_info, mock_request, access_token
            )
            
            assert session_info is not None
            assert session_info.session_id in service.memory_sessions
            assert service.memory_sessions[session_info.session_id].user_info.user_id == user_info.user_id
    
    @pytest.mark.asyncio
    async def test_get_session_redis(self, session_service, user_info, mock_request):
        """Test retrieving session from Redis"""
        if not session_service.use_redis:
            pytest.skip("Redis not available for testing")
        
        # Create session first
        session_info = await session_service.create_session(
            user_info, mock_request, "access-token"
        )
        
        # Retrieve session
        retrieved_session = await session_service.get_session(session_info.session_id)
        
        assert retrieved_session is not None
        assert retrieved_session.session_id == session_info.session_id
        assert retrieved_session.user_info.user_id == user_info.user_id
        assert retrieved_session.activity_count == 1  # Should increment on access
    
    @pytest.mark.asyncio
    async def test_session_expiration(self, session_service, user_info, mock_request):
        """Test that expired sessions are handled correctly"""
        if not session_service.use_redis:
            pytest.skip("Redis not available for testing")
        
        # Create session with short timeout
        original_timeout = session_service.session_timeout
        session_service.session_timeout = timedelta(seconds=1)
        
        session_info = await session_service.create_session(
            user_info, mock_request, "access-token"
        )
        
        # Wait for session to expire
        await asyncio.sleep(2)
        
        # Try to retrieve expired session
        retrieved_session = await session_service.get_session(session_info.session_id)
        
        # Should return None for expired session
        assert retrieved_session is None
        
        # Restore original timeout
        session_service.session_timeout = original_timeout
    
    @pytest.mark.asyncio
    async def test_refresh_session(self, session_service, user_info, mock_request):
        """Test session refresh functionality"""
        if not session_service.use_redis:
            pytest.skip("Redis not available for testing")
        
        # Create session
        session_info = await session_service.create_session(
            user_info, mock_request, "access-token"
        )
        
        original_expires_at = session_info.expires_at
        
        # Wait a moment
        await asyncio.sleep(0.1)
        
        # Refresh session
        refreshed_session = await session_service.refresh_session(session_info.session_id)
        
        assert refreshed_session is not None
        assert refreshed_session.expires_at > original_expires_at
    
    @pytest.mark.asyncio
    async def test_terminate_session(self, session_service, user_info, mock_request):
        """Test session termination"""
        if not session_service.use_redis:
            pytest.skip("Redis not available for testing")
        
        # Create session
        session_info = await session_service.create_session(
            user_info, mock_request, "access-token"
        )
        
        # Terminate session
        result = await session_service.terminate_session(session_info.session_id)
        
        assert result is True
        
        # Try to retrieve terminated session
        retrieved_session = await session_service.get_session(session_info.session_id)
        
        # Should return None as terminated sessions are not accessible
        assert retrieved_session is None
    
    @pytest.mark.asyncio
    async def test_session_limits(self, session_service, user_info, mock_request):
        """Test session limits per user"""
        if not session_service.use_redis:
            pytest.skip("Redis not available for testing")
        
        # Set low session limit for testing
        original_limit = session_service.max_sessions_per_user
        session_service.max_sessions_per_user = 2
        
        sessions = []
        
        # Create sessions up to limit
        for i in range(3):  # Try to create more than limit
            session_info = await session_service.create_session(
                user_info, mock_request, f"access-token-{i}"
            )
            sessions.append(session_info)
        
        # Check that only max allowed sessions are active
        user_sessions = await session_service.get_user_sessions(user_info.user_id)
        assert len(user_sessions) <= session_service.max_sessions_per_user
        
        # Restore original limit
        session_service.max_sessions_per_user = original_limit
    
    @pytest.mark.asyncio
    async def test_session_activity_tracking(self, session_service, user_info, mock_request):
        """Test session activity tracking"""
        if not session_service.use_redis:
            pytest.skip("Redis not available for testing")
        
        # Create session
        session_info = await session_service.create_session(
            user_info, mock_request, "access-token"
        )
        
        # Update session activity
        await session_service.update_session_activity(
            session_info.session_id,
            "api_call",
            {"endpoint": "/api/test", "method": "GET"}
        )
        
        # Retrieve session and check activity
        retrieved_session = await session_service.get_session(session_info.session_id)
        
        assert retrieved_session is not None
        assert retrieved_session.last_activity["type"] == "api_call"
        assert retrieved_session.last_activity["details"]["endpoint"] == "/api/test"
    
    @pytest.mark.asyncio
    async def test_get_session_statistics(self, session_service, user_info, mock_request):
        """Test session statistics"""
        if not session_service.use_redis:
            pytest.skip("Redis not available for testing")
        
        # Create multiple sessions
        for i in range(3):
            await session_service.create_session(
                user_info, mock_request, f"access-token-{i}"
            )
        
        # Get statistics
        stats = await session_service.get_session_statistics()
        
        assert stats["total_sessions"] >= 3
        assert stats["active_sessions"] >= 3
        assert stats["unique_users"] >= 1
        assert "average_session_duration" in stats
        assert "most_active_users" in stats
    
    @pytest.mark.asyncio
    async def test_redis_connection_failure_during_operation(self, session_config, user_info, mock_request):
        """Test handling Redis failure during operation"""
        service = SessionService(session_config)
        
        if not service.use_redis:
            pytest.skip("Redis not available for testing")
        
        # Create session successfully
        session_info = await service.create_session(
            user_info, mock_request, "access-token"
        )
        
        # Mock Redis failure
        with patch.object(service.redis_client, 'get', side_effect=Exception("Redis connection lost")):
            # Should handle gracefully and return None
            retrieved_session = await service.get_session(session_info.session_id)
            assert retrieved_session is None
    
    def test_global_session_service_initialization(self, session_config):
        """Test global session service initialization"""
        # Initialize global service
        initialize_session_service(session_config)
        
        # Get global service
        global_service = get_session_service()
        
        assert global_service is not None
        assert isinstance(global_service, SessionService)
    
    @pytest.mark.asyncio
    async def test_concurrent_session_operations(self, session_service, user_info, mock_request):
        """Test concurrent session operations"""
        if not session_service.use_redis:
            pytest.skip("Redis not available for testing")
        
        # Create multiple sessions concurrently
        tasks = []
        for i in range(5):
            task = session_service.create_session(
                user_info, mock_request, f"access-token-{i}"
            )
            tasks.append(task)
        
        sessions = await asyncio.gather(*tasks)
        
        # Verify all sessions were created
        assert len(sessions) == 5
        assert all(session.session_id for session in sessions)
        assert len(set(session.session_id for session in sessions)) == 5  # All unique


class TestSessionServiceIntegration:
    """Integration tests for session service"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_redis_operations(self):
        """Test with real Redis instance (requires Redis to be running)"""
        config = {
            "session_timeout_minutes": 30,
            "max_sessions_per_user": 5,
            "cleanup_interval_minutes": 5,
            "enable_cleanup_task": False,  # Disable for tests
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 1,
                "password": None
            }
        }
        
        service = SessionService(config)
        
        if not service.use_redis:
            pytest.skip("Redis not available for integration testing")
        
        user_info = UserInfo(
            user_id="integration-test-user",
            user_name="Integration Test User",
            email="test@integration.com",
            given_name="Test",
            family_name="User",
            roles=[UserRole.DEVELOPER],
            scopes=["test"],
            tenant_id="integration-tenant"
        )
        
        mock_request = Mock()
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {"user-agent": "Integration Test"}
        
        try:
            # Create session
            session_info = await service.create_session(
                user_info, mock_request, "integration-access-token"
            )
            
            assert session_info is not None
            
            # Verify Redis storage
            key = f"session:{session_info.session_id}"
            stored_data = service.redis_client.get(key)
            assert stored_data is not None
            
            # Test session retrieval
            retrieved_session = await service.get_session(session_info.session_id)
            assert retrieved_session is not None
            assert retrieved_session.user_info.user_id == user_info.user_id
            
            # Test session termination
            terminated = await service.terminate_session(session_info.session_id)
            assert terminated is True
            
        finally:
            # Cleanup
            try:
                keys = service.redis_client.keys("session:*")
                if keys:
                    service.redis_client.delete(*keys)
                keys = service.redis_client.keys("user_sessions:*")
                if keys:
                    service.redis_client.delete(*keys)
            except Exception:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])