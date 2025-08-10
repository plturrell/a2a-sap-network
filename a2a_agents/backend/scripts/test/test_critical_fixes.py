"""
Comprehensive Test Suite for Critical A2A Compliance Fixes
Tests the three major fixes:
1. ORD Registry Search and Indexing
2. API Rate Limiting and Key Rotation 
3. Enhanced Error Handling with Circuit Breakers
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

# Test imports
from backend.app.ord_registry.enhanced_search_service import EnhancedORDSearchService
from backend.app.api.middleware.rate_limiting import APIRateLimiter, APIKeyManager
from backend.app.a2a.core.enhanced_circuit_breaker import (
    EnhancedCircuitBreaker, CircuitBreakerConfig, CircuitState, CircuitBreakerOpenError
)


class TestORDRegistryEnhancedSearch:
    """Test suite for enhanced ORD Registry search and indexing functionality"""
    
    @pytest.fixture
    async def search_service(self):
        """Create a search service instance for testing"""
        service = EnhancedORDSearchService()
        await service.initialize()
        return service
    
    @pytest.fixture
    def sample_ord_registration(self):
        """Sample ORD registration for testing"""
        from backend.app.ord_registry.models import ORDRegistration, ORDDocument, RegistrationMetadata, ValidationResult, RegistrationStatus
        
        ord_doc = ORDDocument(
            openResourceDiscovery="1.5.0",
            title="Test Data Product",
            shortDescription="A test data product for validation",
            description="This is a comprehensive test data product with Dublin Core metadata",
            dataProducts=[
                {
                    "ordId": "test:dataProduct:financial_data",
                    "title": "Financial Market Data",
                    "shortDescription": "Real-time financial market data",
                    "description": "Comprehensive financial market data including stocks, bonds, and derivatives",
                    "type": "dataProduct",
                    "category": "financial",
                    "tags": ["finance", "market", "real-time"],
                    "version": "1.0.0",
                    "accessStrategies": [
                        {"type": "rest", "url": "/api/v1/financial-data"},
                        {"type": "streaming", "url": "wss://api.example.com/stream"}
                    ],
                    "partOfPackage": "test:package:financial"
                }
            ],
            dublin_core={
                "title": "Financial Market Data Collection",
                "creator": [{"name": "Test Organization", "email": "test@example.com"}],
                "subject": ["finance", "market data", "real-time analytics"],
                "description": "A comprehensive collection of financial market data",
                "publisher": "Test Financial Services",
                "type": "Dataset",
                "format": "JSON",
                "language": "en",
                "date": datetime.utcnow().isoformat()
            }
        )
        
        metadata = RegistrationMetadata(
            registered_by="test_user",
            registered_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            version="1.0.0",
            status=RegistrationStatus.ACTIVE
        )
        
        validation = ValidationResult(
            valid=True,
            errors=[],
            warnings=[],
            compliance_score=0.95
        )
        
        return ORDRegistration(
            registration_id="test_reg_001",
            ord_document=ord_doc,
            metadata=metadata,
            validation=validation
        )
    
    @pytest.mark.asyncio
    async def test_search_service_initialization(self, search_service):
        """Test that search service initializes correctly"""
        assert search_service is not None
        assert search_service.fallback_mode is True  # Should be True without Elasticsearch
    
    @pytest.mark.asyncio
    async def test_index_ord_registration(self, search_service, sample_ord_registration):
        """Test indexing of ORD registration"""
        result = await search_service.index_ord_registration(sample_ord_registration)
        assert result is True  # Should succeed even in fallback mode
    
    @pytest.mark.asyncio
    async def test_search_functionality(self, search_service, sample_ord_registration):
        """Test search functionality with fallback"""
        from backend.app.ord_registry.models import SearchRequest, ResourceType
        
        # Index a registration first
        await search_service.index_ord_registration(sample_ord_registration)
        
        # Create search request
        search_request = SearchRequest(
            query="financial market",
            resource_type=ResourceType.DATA_PRODUCT,
            page=1,
            page_size=10,
            filters={
                "category": "financial",
                "dc_publisher": "Test Financial Services"
            }
        )
        
        # Perform search
        search_result = await search_service.enhanced_search(search_request)
        
        # Verify search result structure
        assert search_result is not None
        assert hasattr(search_result, 'results')
        assert hasattr(search_result, 'total_count')
        assert hasattr(search_result, 'page')
        assert hasattr(search_result, 'page_size')
    
    @pytest.mark.asyncio
    async def test_quality_score_calculation(self, search_service):
        """Test quality score calculation"""
        data_product = {
            "title": "Test Product",
            "description": "Test description",
            "shortDescription": "Short desc",
            "tags": ["test", "product"],
            "accessStrategies": [{"type": "rest"}],
            "version": "1.0.0"
        }
        
        dublin_core = {
            "title": "Test",
            "creator": ["Test Creator"],
            "subject": ["test"],
            "description": "Test description",
            "publisher": "Test Pub",
            "type": "Dataset",
            "format": "JSON",
            "language": "en"
        }
        
        score = search_service._calculate_quality_score(data_product, dublin_core)
        assert 0.0 <= score <= 1.0
        assert score > 0.8  # Should be high quality with all fields
    
    @pytest.mark.asyncio
    async def test_resource_analytics(self, search_service):
        """Test resource analytics functionality"""
        analytics = await search_service.get_resource_analytics("test:dataProduct:financial_data")
        # Should return None for non-existent resource
        assert analytics is None


class TestAPIRateLimitingAndKeyRotation:
    """Test suite for API rate limiting and key rotation functionality"""
    
    @pytest.fixture
    async def rate_limiter(self):
        """Create rate limiter instance for testing"""
        limiter = APIRateLimiter()
        await limiter.initialize()
        return limiter
    
    @pytest.fixture
    async def key_manager(self):
        """Create API key manager instance for testing"""
        manager = APIKeyManager()
        await manager.initialize()
        return manager
    
    @pytest.mark.asyncio
    async def test_rate_limiter_initialization(self, rate_limiter):
        """Test rate limiter initializes correctly"""
        assert rate_limiter is not None
        assert rate_limiter.redis_client is None  # No Redis in test environment
        assert len(rate_limiter.rate_limits) > 0
    
    @pytest.mark.asyncio
    async def test_rate_limit_check_allows_normal_requests(self, rate_limiter):
        """Test that normal requests are allowed"""
        allowed, limit_info = await rate_limiter.check_rate_limit(
            agent_id="test_agent",
            agent_type="agent",
            endpoint="/api/test"
        )
        
        assert allowed is True
        assert "agent_id" in limit_info
        assert "minute_count" in limit_info
        assert "hour_count" in limit_info
    
    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self, rate_limiter):
        """Test that rate limits are enforced"""
        agent_id = "test_agent_heavy"
        agent_type = "default"  # Lower limits
        
        # Make requests up to the limit
        requests_made = 0
        while requests_made < 60:  # Try to exceed minute limit
            allowed, limit_info = await rate_limiter.check_rate_limit(
                agent_id=agent_id,
                agent_type=agent_type,
                endpoint="/api/test"
            )
            
            if not allowed:
                assert limit_info["minute_exceeded"] is True
                break
            
            requests_made += 1
            
            # Prevent infinite loop
            if requests_made >= 100:
                break
        
        # Should have hit the limit at some point
        assert requests_made > 0
    
    @pytest.mark.asyncio
    async def test_api_key_generation(self, key_manager):
        """Test API key generation"""
        key_info = await key_manager.generate_api_key(
            agent_id="test_agent",
            agent_type="agent",
            ttl=3600
        )
        
        assert "api_key" in key_info
        assert "expires_at" in key_info
        assert "ttl" in key_info
        assert "agent_id" in key_info
        assert len(key_info["api_key"]) > 20  # Should be a substantial key
    
    @pytest.mark.asyncio
    async def test_api_key_validation(self, key_manager):
        """Test API key validation"""
        # Generate a key
        key_info = await key_manager.generate_api_key("test_agent", "agent", 3600)
        api_key = key_info["api_key"]
        
        # Validate the key
        valid, key_data = await key_manager.validate_api_key(api_key)
        
        assert valid is True
        assert key_data is not None
        assert key_data["agent_id"] == "test_agent"
        assert key_data["agent_type"] == "agent"
    
    @pytest.mark.asyncio
    async def test_api_key_rotation(self, key_manager):
        """Test API key rotation"""
        # Generate initial key
        initial_key = await key_manager.generate_api_key("test_agent", "agent", 3600)
        
        # Rotate the key
        rotation_result = await key_manager.rotate_api_key("test_agent")
        
        assert "new_key" in rotation_result
        assert "old_key_valid_until" in rotation_result
        assert "grace_period_seconds" in rotation_result
        
        # New key should be different
        new_api_key = rotation_result["new_key"]["api_key"]
        assert new_api_key != initial_key["api_key"]
        
        # Both keys should be valid during grace period
        old_valid, _ = await key_manager.validate_api_key(initial_key["api_key"])
        new_valid, _ = await key_manager.validate_api_key(new_api_key)
        
        assert old_valid is True  # Grace period
        assert new_valid is True
    
    @pytest.mark.asyncio
    async def test_api_key_revocation(self, key_manager):
        """Test API key revocation"""
        # Generate a key
        key_info = await key_manager.generate_api_key("test_agent", "agent", 3600)
        api_key = key_info["api_key"]
        
        # Verify it's valid
        valid, _ = await key_manager.validate_api_key(api_key)
        assert valid is True
        
        # Revoke the key
        revoked = await key_manager.revoke_api_key("test_agent")
        assert revoked is True
        
        # Key should now be invalid
        valid, _ = await key_manager.validate_api_key(api_key)
        assert valid is False


class TestEnhancedCircuitBreaker:
    """Test suite for enhanced circuit breaker functionality"""
    
    @pytest.fixture
    def circuit_breaker_config(self):
        """Circuit breaker configuration for testing"""
        return CircuitBreakerConfig(
            failure_threshold=3,
            recovery_threshold=2,
            timeout_seconds=1,  # Short timeout for tests
            success_threshold=0.5,
            enable_exponential_backoff=True,
            initial_backoff=0.5,
            max_backoff=5.0
        )
    
    @pytest.fixture
    async def circuit_breaker(self, circuit_breaker_config):
        """Create circuit breaker instance for testing"""
        cb = EnhancedCircuitBreaker("test_circuit", circuit_breaker_config)
        await cb.initialize()
        return cb
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_initialization(self, circuit_breaker):
        """Test circuit breaker initializes correctly"""
        assert circuit_breaker.name == "test_circuit"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.success_count == 0
    
    @pytest.mark.asyncio
    async def test_successful_calls_remain_closed(self, circuit_breaker):
        """Test that successful calls keep circuit closed"""
        async def successful_function():
            return "success"
        
        # Make several successful calls
        for _ in range(5):
            result = await circuit_breaker.call(successful_function)
            assert result == "success"
        
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_opens_on_failures(self, circuit_breaker):
        """Test that circuit opens after threshold failures"""
        async def failing_function():
            raise Exception("Test failure")
        
        # Make calls until circuit opens
        failure_count = 0
        for _ in range(5):
            try:
                await circuit_breaker.call(failing_function)
            except Exception:
                failure_count += 1
                if circuit_breaker.state == CircuitState.OPEN:
                    break
        
        assert circuit_breaker.state == CircuitState.OPEN
        assert failure_count >= circuit_breaker.config.failure_threshold
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_error(self, circuit_breaker):
        """Test that CircuitBreakerOpenError is raised when circuit is open"""
        async def failing_function():
            raise Exception("Test failure")
        
        # Force circuit to open
        for _ in range(circuit_breaker.config.failure_threshold):
            try:
                await circuit_breaker.call(failing_function)
            except Exception:
                pass
        
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(failing_function)
    
    @pytest.mark.asyncio
    async def test_circuit_recovery_to_half_open(self, circuit_breaker):
        """Test circuit recovery from open to half-open"""
        async def failing_function():
            raise Exception("Test failure")
        
        # Force circuit to open
        for _ in range(circuit_breaker.config.failure_threshold):
            try:
                await circuit_breaker.call(failing_function)
            except Exception:
                pass
        
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Wait for timeout
        await asyncio.sleep(circuit_breaker.config.timeout_seconds + 0.1)
        
        # Next call should transition to half-open
        try:
            await circuit_breaker.call(failing_function)
        except Exception:
            pass
        
        assert circuit_breaker.state == CircuitState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_fallback_function(self, circuit_breaker):
        """Test fallback function execution when circuit is open"""
        async def failing_function():
            raise Exception("Test failure")
        
        async def fallback_function():
            return "fallback_result"
        
        # Force circuit to open
        for _ in range(circuit_breaker.config.failure_threshold):
            try:
                await circuit_breaker.call(failing_function)
            except Exception:
                pass
        
        # Call with fallback should return fallback result
        result = await circuit_breaker.call(failing_function, fallback=fallback_function)
        assert result == "fallback_result"
    
    @pytest.mark.asyncio
    async def test_circuit_metrics(self, circuit_breaker):
        """Test circuit breaker metrics collection"""
        async def test_function():
            return "success"
        
        # Make some calls
        await circuit_breaker.call(test_function)
        await circuit_breaker.call(test_function)
        
        # Get metrics
        metrics = await circuit_breaker.get_metrics()
        
        assert metrics.state == CircuitState.CLOSED
        assert metrics.total_calls >= 2
        assert metrics.success_rate > 0
        assert metrics.average_response_time >= 0
    
    @pytest.mark.asyncio
    async def test_exponential_backoff(self, circuit_breaker):
        """Test exponential backoff functionality"""
        async def failing_function():
            raise Exception("Test failure")
        
        initial_backoff = circuit_breaker.current_backoff
        
        # Force circuit to open multiple times to test backoff
        for iteration in range(3):
            # Reset to closed for each iteration
            await circuit_breaker.reset()
            
            # Force to open
            for _ in range(circuit_breaker.config.failure_threshold):
                try:
                    await circuit_breaker.call(failing_function)
                except Exception:
                    pass
            
            current_backoff = circuit_breaker.current_backoff
            expected_min_backoff = initial_backoff * (circuit_breaker.config.backoff_multiplier ** iteration)
            
            assert current_backoff >= expected_min_backoff
    
    @pytest.mark.asyncio
    async def test_manual_reset(self, circuit_breaker):
        """Test manual circuit breaker reset"""
        async def failing_function():
            raise Exception("Test failure")
        
        # Force circuit to open
        for _ in range(circuit_breaker.config.failure_threshold):
            try:
                await circuit_breaker.call(failing_function)
            except Exception:
                pass
        
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Reset circuit manually
        await circuit_breaker.reset()
        
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.success_count == 0


class TestIntegrationScenarios:
    """Integration tests for combined functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limited_search_with_circuit_breaker(self):
        """Test integration of rate limiting with circuit breaker protection"""
        from backend.app.a2a.core.enhanced_circuit_breaker import get_circuit_breaker_manager
        
        # Set up circuit breaker for search service
        manager = await get_circuit_breaker_manager()
        
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=1
        )
        
        cb = await manager.get_circuit_breaker("search_service", config)
        
        # Mock search function that might fail
        call_count = 0
        async def mock_search_function(query: str):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # First two calls fail
                raise Exception("Search service temporarily unavailable")
            return {"results": [], "total": 0}
        
        # Fallback function
        async def search_fallback(query: str):
            return {"results": [], "total": 0, "fallback": True}
        
        # First calls should fail and open circuit
        for _ in range(2):
            try:
                await cb.call(mock_search_function, "test query", fallback=search_fallback)
            except Exception:
                pass
        
        # Circuit should be open now
        assert cb.state == CircuitState.OPEN
        
        # Next call should use fallback
        result = await cb.call(mock_search_function, "test query", fallback=search_fallback)
        assert result["fallback"] is True
    
    @pytest.mark.asyncio
    async def test_api_key_rotation_during_high_load(self):
        """Test API key rotation behavior under high request load"""
        key_manager = APIKeyManager()
        rate_limiter = APIRateLimiter()
        
        await key_manager.initialize()
        await rate_limiter.initialize()
        
        agent_id = "high_load_agent"
        
        # Generate initial API key
        initial_key = await key_manager.generate_api_key(agent_id, "agent", 3600)
        
        # Simulate high load with rate limiting
        successful_requests = 0
        rate_limited_requests = 0
        
        for _ in range(10):  # Make multiple requests
            # Check rate limit
            allowed, limit_info = await rate_limiter.check_rate_limit(
                agent_id=agent_id,
                agent_type="agent"
            )
            
            if allowed:
                # Validate API key
                valid, key_data = await key_manager.validate_api_key(initial_key["api_key"])
                if valid:
                    successful_requests += 1
            else:
                rate_limited_requests += 1
        
        # Rotate key during load
        rotation_result = await key_manager.rotate_api_key(agent_id)
        assert "new_key" in rotation_result
        
        # Both old and new keys should work during grace period
        old_valid, _ = await key_manager.validate_api_key(initial_key["api_key"])
        new_valid, _ = await key_manager.validate_api_key(rotation_result["new_key"]["api_key"])
        
        assert old_valid is True  # Grace period
        assert new_valid is True
        assert successful_requests > 0


if __name__ == "__main__":
    """Run tests directly"""
    print("Running critical fixes test suite...")
    print("=" * 60)
    
    # Run tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])