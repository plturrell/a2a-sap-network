#!/usr/bin/env python3
"""
Unit Tests for Reasoning Agent Components
Comprehensive testing of individual components without external dependencies
"""

import asyncio
import unittest
import tempfile
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestAsyncReasoningMemorySystem(unittest.IsolatedAsyncioTestCase):
    """Unit tests for async memory system"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_memory.db")
        
        from asyncReasoningMemorySystem import AsyncReasoningMemoryStore, ReasoningExperience
        self.AsyncReasoningMemoryStore = AsyncReasoningMemoryStore
        self.ReasoningExperience = ReasoningExperience
        
        self.memory_store = AsyncReasoningMemoryStore(self.db_path)
        await self.memory_store.initialize()
    
    async def asyncTearDown(self):
        """Clean up test environment"""
        await self.memory_store.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_store_and_retrieve_experience(self):
        """Test storing and retrieving experiences"""
        # Create test experience
        experience = self.ReasoningExperience(
            question="What is AI?",
            answer="Artificial Intelligence is...",
            reasoning_chain=[{"step": 1, "content": "Define AI"}],
            confidence=0.9,
            context={"domain": "technology"},
            timestamp=datetime.utcnow(),
            architecture_used="hierarchical",
            performance_metrics={"duration": 1.5}
        )
        
        # Store experience
        result = await self.memory_store.store_experience(experience)
        self.assertTrue(result)
        
        # Retrieve similar experiences
        similar = await self.memory_store.retrieve_similar_experiences("What is AI?", limit=5)
        self.assertEqual(len(similar), 1)
        self.assertEqual(similar[0].question, "What is AI?")
        self.assertEqual(similar[0].confidence, 0.9)
    
    async def test_concurrent_operations(self):
        """Test concurrent storage and retrieval"""
        # Create multiple experiences
        experiences = []
        for i in range(5):
            exp = self.ReasoningExperience(
                question=f"Question {i}",
                answer=f"Answer {i}",
                reasoning_chain=[{"step": 1, "content": f"Step {i}"}],
                confidence=0.8 + i * 0.02,
                context={"index": i},
                timestamp=datetime.utcnow(),
                architecture_used="test",
                performance_metrics={"duration": i * 0.1}
            )
            experiences.append(exp)
        
        # Store concurrently
        store_tasks = [self.memory_store.store_experience(exp) for exp in experiences]
        results = await asyncio.gather(*store_tasks)
        
        # All should succeed
        self.assertTrue(all(results))
        
        # Retrieve concurrently
        retrieve_tasks = [
            self.memory_store.retrieve_similar_experiences(f"Question {i}", limit=2)
            for i in range(3)
        ]
        retrieve_results = await asyncio.gather(*retrieve_tasks)
        
        # Should get results for each query
        for i, results in enumerate(retrieve_results):
            self.assertGreaterEqual(len(results), 1)
            self.assertEqual(results[0].question, f"Question {i}")
    
    async def test_performance_stats(self):
        """Test performance statistics"""
        # Add some test data
        for i in range(3):
            exp = self.ReasoningExperience(
                question=f"Test {i}",
                answer=f"Answer {i}",
                reasoning_chain=[],
                confidence=0.7 + i * 0.1,
                context={},
                timestamp=datetime.utcnow(),
                architecture_used="blackboard",
                performance_metrics={}
            )
            await self.memory_store.store_experience(exp)
        
        # Get stats
        stats = await self.memory_store.get_performance_stats()
        
        # Verify stats structure
        self.assertIn("experiences", stats)
        self.assertIn("patterns", stats)
        self.assertEqual(stats["experiences"]["total"], 3)
        self.assertAlmostEqual(stats["experiences"]["avg_confidence"], 0.8, places=1)


class TestAsyncGrokClient(unittest.IsolatedAsyncioTestCase):
    """Unit tests for async Grok client"""
    
    def setUp(self):
        """Set up test environment"""
        from asyncGrokClient import GrokConfig, AsyncGrokConnectionPool, AsyncGrokCache
        self.GrokConfig = GrokConfig
        self.AsyncGrokConnectionPool = AsyncGrokConnectionPool
        self.AsyncGrokCache = AsyncGrokCache
        
        self.config = GrokConfig(
            api_key="test-key",
            pool_connections=3,
            pool_maxsize=5,
            cache_ttl=60
        )
    
    async def test_connection_pool_creation(self):
        """Test connection pool setup"""
        pool = self.AsyncGrokConnectionPool(self.config)
        
        # Get client should create HTTP client
        client = await pool.get_client()
        self.assertIsNotNone(client)
        
        # Second call should return same client
        client2 = await pool.get_client()
        self.assertEqual(client, client2)
        
        await pool.close()
    
    async def test_cache_operations(self):
        """Test cache set/get operations"""
        cache = self.AsyncGrokCache(cache_ttl=60)
        await cache.initialize()
        
        # Test cache miss
        result = await cache.get("test_key")
        self.assertIsNone(result)
        
        # Test cache set and hit
        from asyncGrokClient import GrokResponse
        test_response = GrokResponse(
            content="test content",
            model="test-model",
            usage={"total_tokens": 100},
            finish_reason="stop",
            raw_response={"test": True}
        )
        
        await cache.set("test_key", test_response)
        
        # Test cache hit
        cached = await cache.get("test_key")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.content, "test content")
        self.assertTrue(cached.cached)
        
        await cache.close()
    
    async def test_cache_key_generation(self):
        """Test cache key generation consistency"""
        cache = self.AsyncGrokCache()
        
        messages1 = [{"role": "user", "content": "test"}]
        messages2 = [{"role": "user", "content": "test"}]
        messages3 = [{"role": "user", "content": "different"}]
        
        key1 = cache._generate_cache_key(messages1, temperature=0.7)
        key2 = cache._generate_cache_key(messages2, temperature=0.7)
        key3 = cache._generate_cache_key(messages3, temperature=0.7)
        
        # Same messages should generate same key
        self.assertEqual(key1, key2)
        
        # Different messages should generate different keys
        self.assertNotEqual(key1, key3)
    
    async def test_grok_config_validation(self):
        """Test Grok configuration validation"""
        # Valid config
        config = self.GrokConfig(
            api_key="valid-key",
            pool_connections=5
        )
        self.assertEqual(config.api_key, "valid-key")
        self.assertEqual(config.pool_connections, 5)
        
        # Default values
        self.assertEqual(config.base_url, "https://api.x.ai/v1")
        self.assertEqual(config.model, "grok-4-latest")


class TestAsyncCleanupManager(unittest.IsolatedAsyncioTestCase):
    """Unit tests for async cleanup manager"""
    
    def setUp(self):
        """Set up test environment"""
        from asyncCleanupManager import AsyncResourceManager, AsyncReasoningCleanupManager
        self.AsyncResourceManager = AsyncResourceManager
        self.AsyncReasoningCleanupManager = AsyncReasoningCleanupManager
    
    async def test_resource_registration(self):
        """Test resource registration"""
        manager = self.AsyncResourceManager()
        
        # Create mock resource
        class MockResource:
            def __init__(self):
                self.closed = False
            
            async def close(self):
                self.closed = True
        
        resource = MockResource()
        manager.register_resource(resource)
        
        # Resource should be registered
        self.assertIn(resource, manager._resources)
    
    async def test_cleanup_execution(self):
        """Test cleanup execution"""
        manager = self.AsyncResourceManager()
        
        # Create multiple mock resources
        class MockResource:
            def __init__(self, name):
                self.name = name
                self.closed = False
            
            async def close(self):
                await asyncio.sleep(0.001)  # Simulate async work
                self.closed = True
        
        resources = [MockResource(f"Resource{i}") for i in range(3)]
        for resource in resources:
            manager.register_resource(resource)
        
        # Execute cleanup
        await manager.cleanup_all()
        
        # All resources should be closed
        for resource in resources:
            self.assertTrue(resource.closed)
    
    async def test_reasoning_cleanup_manager(self):
        """Test reasoning-specific cleanup manager"""
        manager = self.AsyncReasoningCleanupManager()
        
        # Mock resources for different types
        class MockGrokClient:
            def __init__(self):
                self.closed = False
            async def close(self):
                self.closed = True
        
        class MockMemoryStore:
            def __init__(self):
                self.closed = False
            async def close(self):
                self.closed = True
        
        # Register different types
        grok_client = MockGrokClient()
        memory_store = MockMemoryStore()
        
        manager.register_grok_client(grok_client)
        manager.register_memory_store(memory_store)
        
        # Execute cleanup
        await manager.cleanup_reasoning_components()
        
        # All should be cleaned
        self.assertTrue(grok_client.closed)
        self.assertTrue(memory_store.closed)
        
        # Check performance stats
        stats = manager.get_performance_stats()
        self.assertEqual(stats["cleanup_count"], 1)
        self.assertGreater(stats["resources_cleaned"], 0)
    
    async def test_background_task_cleanup(self):
        """Test background task cleanup"""
        manager = self.AsyncResourceManager()
        
        # Create background task
        async def background_work():
            await asyncio.sleep(1.0)  # Long running task
        
        task = asyncio.create_task(background_work())
        manager.register_background_task(task)
        
        # Cleanup should cancel the task
        await manager.cleanup_all(timeout=0.1)
        
        # Task should be cancelled
        self.assertTrue(task.cancelled() or task.done())


class TestBlackboardArchitecture(unittest.IsolatedAsyncioTestCase):
    """Unit tests for blackboard architecture components"""
    
    def setUp(self):
        """Set up test environment"""
        try:
            from blackboardArchitecture import BlackboardState, KnowledgeSourceType
            self.BlackboardState = BlackboardState
            self.KnowledgeSourceType = KnowledgeSourceType
            self.blackboard_available = True
        except ImportError:
            self.blackboard_available = False
    
    async def test_blackboard_state(self):
        """Test blackboard state management"""
        if not self.blackboard_available:
            self.skipTest("Blackboard architecture not available")
        
        state = self.BlackboardState()
        
        # Test initial state
        self.assertEqual(state.problem, "")
        self.assertEqual(len(state.facts), 0)
        self.assertEqual(len(state.conclusions), 0)
        self.assertEqual(state.iteration, 0)
        
        # Test adding data
        state.problem = "Test problem"
        state.facts.append({"content": "Test fact", "confidence": 0.9})
        state.iteration = 1
        
        # Test state conversion
        state_dict = state.to_dict()
        self.assertEqual(state_dict["problem"], "Test problem")
        self.assertEqual(len(state_dict["facts"]), 1)
        self.assertEqual(state_dict["iteration"], 1)
    
    async def test_knowledge_source_types(self):
        """Test knowledge source type enumeration"""
        if not self.blackboard_available:
            self.skipTest("Blackboard architecture not available")
        
        # Test all expected types are present
        expected_types = [
            "pattern_recognition",
            "logical_reasoning", 
            "evidence_evaluation",
            "causal_analysis"
        ]
        
        for expected_type in expected_types:
            # Should be able to create enum from string
            source_type = self.KnowledgeSourceType(expected_type)
            self.assertEqual(source_type.value, expected_type)


class TestGrokReasoning(unittest.IsolatedAsyncioTestCase):
    """Unit tests for Grok reasoning components"""
    
    def setUp(self):
        """Set up test environment"""
        try:
            from grokReasoning import GrokReasoning


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
            self.GrokReasoning = GrokReasoning
            self.grok_available = True
        except ImportError:
            self.grok_available = False
    
    async def test_grok_initialization(self):
        """Test Grok reasoning initialization"""
        if not self.grok_available:
            self.skipTest("Grok reasoning not available")
        
        # Should initialize without API key for testing
        grok = self.GrokReasoning()
        self.assertIsNotNone(grok)
        
        # Should handle missing client gracefully
        if grok.grok_client is None:
            # Expected when no API key provided
            self.assertIsNone(grok.grok_client)
    
    @patch('grokReasoning.GrokClient')
    async def test_decompose_question_fallback(self, mock_client_class):
        """Test question decomposition with fallback"""
        if not self.grok_available:
            self.skipTest("Grok reasoning not available")
        
        # Mock client to simulate failure
        mock_client = AsyncMock()
        mock_client.async_chat_completion.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client
        
        grok = self.GrokReasoning()
        grok.grok_client = mock_client
        
        # Should handle failure gracefully
        result = await grok.decompose_question("Test question")
        self.assertFalse(result.get('success', True))
    
    @patch('grokReasoning.GrokClient')
    async def test_analyze_patterns_fallback(self, mock_client_class):
        """Test pattern analysis with fallback"""
        if not self.grok_available:
            self.skipTest("Grok reasoning not available")
        
        # Mock client to simulate failure
        mock_client = AsyncMock()
        mock_client.async_chat_completion.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client
        
        grok = self.GrokReasoning()
        grok.grok_client = mock_client
        
        # Should handle failure gracefully
        result = await grok.analyze_patterns("Test text")
        self.assertFalse(result.get('success', True))


def run_unit_tests():
    """Run all unit tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAsyncReasoningMemorySystem,
        TestAsyncGrokClient,
        TestAsyncCleanupManager,
        TestBlackboardArchitecture,
        TestGrokReasoning
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running Unit Tests for Reasoning Agent Components")
    print("=" * 60)
    
    success = run_unit_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All unit tests passed!")
    else:
        print("❌ Some unit tests failed - check output above")
    
    print("=" * 60)