#!/usr/bin/env python3
"""
A2A Architecture Integration Test Suite
Tests all integration points in the A2A architecture
"""

import asyncio
import sys
import os
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import tempfile
from datetime import datetime
from uuid import uuid4

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('integration_test.log')
    ]
)
logger = logging.getLogger(__name__)

class IntegrationTestResult:
    """Track integration test results"""
    
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.overall_success = True
        
    def add_result(self, test_name: str, success: bool, details: str, error: Optional[str] = None):
        """Add test result"""
        self.results[test_name] = {
            "success": success,
            "details": details,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }
        if not success:
            self.overall_success = False
        
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {test_name} - {details}")
        if error:
            logger.error(f"   Error: {error}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary"""
        passed = sum(1 for r in self.results.values() if r["success"])
        total = len(self.results)
        
        return {
            "overall_success": self.overall_success,
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "results": self.results
        }


class A2AIntegrationTester:
    """Comprehensive integration tester for A2A architecture"""
    
    def __init__(self):
        self.results = IntegrationTestResult()
        self.temp_files: List[str] = []
        
    async def run_all_tests(self) -> IntegrationTestResult:
        """Run all integration tests"""
        logger.info("🚀 Starting A2A Architecture Integration Tests")
        
        # Test 1: Import Resolution
        await self.test_import_resolution()
        
        # Test 2: Configuration Integration  
        await self.test_configuration_integration()
        
        # Test 3: Distributed Storage Backends
        await self.test_distributed_storage_backends()
        
        # Test 4: Request Signing Integration
        await self.test_request_signing_integration()
        
        # Test 5: Agent Base Integration
        await self.test_agent_base_integration()
        
        # Test 6: Network Connector Integration
        await self.test_network_connector_integration()
        
        # Test 7: Circular Dependencies
        await self.test_circular_dependencies()
        
        # Test 8: End-to-End Integration
        await self.test_end_to_end_integration()
        
        # Cleanup
        await self.cleanup()
        
        logger.info("🏁 Integration Tests Complete")
        return self.results
    
    async def test_import_resolution(self):
        """Test 1: Verify all imports resolve correctly"""
        logger.info("🔍 Testing import resolution...")
        
        import_tests = [
            ("app.core.config", "Settings configuration"),
            ("app.a2a.storage.distributedStorage", "Distributed storage"),
            ("app.a2a.security.requestSigning", "Request signing"),
            ("app.a2a.sdk.agentBase", "Agent base class"),
            ("app.a2a.network.networkConnector", "Network connector"),
            ("app.a2a.sdk.types", "A2A types"),
            ("app.a2a.sdk.mcpServer", "MCP server"),
        ]
        
        failed_imports = []
        
        for module_name, description in import_tests:
            try:
                __import__(module_name)
                self.results.add_result(
                    f"import_{module_name.replace('.', '_')}",
                    True,
                    f"Successfully imported {description}"
                )
            except ImportError as e:
                failed_imports.append((module_name, str(e)))
                self.results.add_result(
                    f"import_{module_name.replace('.', '_')}",
                    False,
                    f"Failed to import {description}",
                    str(e)
                )
            except Exception as e:
                failed_imports.append((module_name, str(e)))
                self.results.add_result(
                    f"import_{module_name.replace('.', '_')}",
                    False,
                    f"Error importing {description}",
                    str(e)
                )
        
        if failed_imports:
            logger.error(f"Failed imports: {failed_imports}")
        else:
            logger.info("✅ All imports successful")
    
    async def test_configuration_integration(self):
        """Test 2: Verify configuration loading and usage"""
        logger.info("🔧 Testing configuration integration...")
        
        try:
            from app.core.config import settings
            
            # Test basic configuration access
            test_configs = [
                ("APP_NAME", "Application name"),
                ("API_V1_STR", "API version string"),
                ("A2A_NETWORK_PATH", "A2A network path"),
                ("A2A_REGISTRY_STORAGE", "Registry storage type"),
                ("REDIS_URL", "Redis URL"),
                ("DEFAULT_TIMEOUT", "Default timeout"),
            ]
            
            for config_name, description in test_configs:
                try:
                    value = getattr(settings, config_name)
                    self.results.add_result(
                        f"config_{config_name.lower()}",
                        True,
                        f"{description}: {value}"
                    )
                except Exception as e:
                    self.results.add_result(
                        f"config_{config_name.lower()}",
                        False,
                        f"Failed to access {description}",
                        str(e)
                    )
            
            # Test secrets manager integration
            try:
                from app.core.secrets import get_secrets_manager
                secrets_manager = get_secrets_manager()
                self.results.add_result(
                    "config_secrets_manager",
                    True,
                    f"Secrets manager initialized: {type(secrets_manager).__name__}"
                )
            except Exception as e:
                self.results.add_result(
                    "config_secrets_manager",
                    False,
                    "Failed to initialize secrets manager",
                    str(e)
                )
                
        except Exception as e:
            self.results.add_result(
                "config_overall",
                False,
                "Failed to test configuration",
                str(e)
            )
    
    async def test_distributed_storage_backends(self):
        """Test 3: Test distributed storage backend initialization"""
        logger.info("💾 Testing distributed storage backends...")
        
        try:
            from app.a2a.storage.distributedStorage import (
                DistributedStorage, 
                RedisBackend, 
                EtcdBackend, 
                LocalFileBackend
            )
            
            # Test local file backend (should always work)
            try:
                local_backend = LocalFileBackend()
                await local_backend.connect()
                
                # Test basic operations
                test_data = {"test": "data", "timestamp": datetime.utcnow().isoformat()}
                
                # Test set/get
                set_result = await local_backend.set("test_key", test_data, ttl=60)
                get_result = await local_backend.get("test_key")
                exists_result = await local_backend.exists("test_key")
                
                if set_result and get_result and exists_result:
                    self.results.add_result(
                        "storage_local_backend",
                        True,
                        "Local storage backend operations successful"
                    )
                else:
                    self.results.add_result(
                        "storage_local_backend",
                        False,
                        "Local storage operations failed"
                    )
                
                await local_backend.disconnect()
                
            except Exception as e:
                self.results.add_result(
                    "storage_local_backend",
                    False,
                    "Local storage backend failed",
                    str(e)
                )
            
            # Test distributed storage factory
            try:
                storage = DistributedStorage()
                await storage.connect()
                
                # Test agent operations
                test_agent_data = {
                    "agent_id": "test_agent",
                    "name": "Test Agent",
                    "capabilities": ["test"]
                }
                
                register_result = await storage.register_agent("test_agent", test_agent_data)
                get_result = await storage.get_agent("test_agent")
                list_result = await storage.list_agents()
                
                if register_result and get_result and isinstance(list_result, list):
                    self.results.add_result(
                        "storage_distributed_operations",
                        True,
                        f"Distributed storage operations successful, backend: {type(storage.backend).__name__}"
                    )
                else:
                    self.results.add_result(
                        "storage_distributed_operations",
                        False,
                        "Distributed storage operations failed"
                    )
                
                await storage.disconnect()
                
            except Exception as e:
                self.results.add_result(
                    "storage_distributed_operations",
                    False,
                    "Distributed storage failed",
                    str(e)
                )
                
        except Exception as e:
            self.results.add_result(
                "storage_overall",
                False,
                "Failed to test distributed storage",
                str(e)
            )
    
    async def test_request_signing_integration(self):
        """Test 4: Test request signing and verification"""
        logger.info("🔐 Testing request signing integration...")
        
        try:
            from app.a2a.security.requestSigning import A2ARequestSigner, JWTRequestSigner
            
            # Test RSA-based signing
            try:
                signer = A2ARequestSigner()
                
                # Generate key pair
                private_pem, public_pem = signer.generate_key_pair()
                
                # Create new signer with keys
                key_signer = A2ARequestSigner(private_pem, public_pem)
                
                # Test signing
                signature_headers = key_signer.sign_request(
                    agent_id="test_agent_1",
                    target_agent_id="test_agent_2",
                    method="POST",
                    path="/api/v1/test",
                    body={"test": "data"}
                )
                
                # Test verification
                is_valid, error = key_signer.verify_request(
                    headers=signature_headers,
                    method="POST",
                    path="/api/v1/test",
                    body={"test": "data"}
                )
                
                if is_valid and not error:
                    self.results.add_result(
                        "signing_rsa_integration",
                        True,
                        "RSA request signing and verification successful"
                    )
                else:
                    self.results.add_result(
                        "signing_rsa_integration",
                        False,
                        "RSA verification failed",
                        error
                    )
                    
            except Exception as e:
                self.results.add_result(
                    "signing_rsa_integration",
                    False,
                    "RSA request signing failed",
                    str(e)
                )
            
            # Test JWT-based signing
            try:
                jwt_signer = JWTRequestSigner("test_secret_key")
                
                # Test signing
                jwt_headers = jwt_signer.sign_request(
                    agent_id="test_agent_1",
                    target_agent_id="test_agent_2",
                    method="GET",
                    path="/api/v1/status"
                )
                
                # Test verification
                is_valid, error = jwt_signer.verify_request(
                    headers=jwt_headers,
                    method="GET",
                    path="/api/v1/status"
                )
                
                if is_valid and not error:
                    self.results.add_result(
                        "signing_jwt_integration",
                        True,
                        "JWT request signing and verification successful"
                    )
                else:
                    self.results.add_result(
                        "signing_jwt_integration",
                        False,
                        "JWT verification failed",
                        error
                    )
                    
            except Exception as e:
                self.results.add_result(
                    "signing_jwt_integration",
                    False,
                    "JWT request signing failed",
                    str(e)
                )
                
        except Exception as e:
            self.results.add_result(
                "signing_overall",
                False,
                "Failed to test request signing",
                str(e)
            )
    
    async def test_agent_base_integration(self):
        """Test 5: Test agent base class integration"""
        logger.info("🤖 Testing agent base integration...")
        
        try:
            from app.a2a.sdk.agentBase import A2AAgentBase
            from app.a2a.sdk.types import A2AMessage, MessagePart, MessageRole
            from app.a2a.sdk.decorators import a2a_handler, a2a_skill
            
            # Create test agent
            class TestAgent(A2AAgentBase):
                def __init__(self):
                    super().__init__(
                        agent_id="test_integration_agent",
                        name="Test Integration Agent",
                        description="Agent for integration testing",
                        version="1.0.0",
                        enable_telemetry=False,  # Disable for testing
                        enable_request_signing=True
                    )
                
                async def initialize(self):
                    pass
                
                async def shutdown(self):
                    pass
                
                @a2a_handler("test_handler")
                async def handle_test(self, message: A2AMessage, context_id: str):
                    return {"status": "handled", "message_id": message.messageId}
                
                @a2a_skill("test_skill", "A test skill")
                async def test_skill(self, input_data: dict):
                    return {"result": "processed", "input": input_data}
            
            # Test agent creation
            try:
                agent = TestAgent()
                
                # Test agent card generation
                agent_card = agent.get_agent_card()
                
                if agent_card and hasattr(agent_card, 'name'):
                    self.results.add_result(
                        "agent_creation",
                        True,
                        f"Agent created successfully: {agent.name}"
                    )
                else:
                    self.results.add_result(
                        "agent_creation",
                        False,
                        "Agent card generation failed"
                    )
                
                # Test handler discovery
                if "test_handler" in agent.handlers:
                    self.results.add_result(
                        "agent_handler_discovery",
                        True,
                        "Handler discovery successful"
                    )
                else:
                    self.results.add_result(
                        "agent_handler_discovery",
                        False,
                        "Handler discovery failed"
                    )
                
                # Test skill discovery
                if "test_skill" in agent.skills:
                    self.results.add_result(
                        "agent_skill_discovery",
                        True,
                        "Skill discovery successful"
                    )
                else:
                    self.results.add_result(
                        "agent_skill_discovery",
                        False,
                        "Skill discovery failed"
                    )
                
                # Test message processing
                test_message = A2AMessage(
                    messageId=str(uuid4()),
                    role=MessageRole.USER,
                    parts=[MessagePart(kind="data", data={"method": "test_handler", "test": True})]
                )
                
                result = await agent.process_message(test_message, "test_context")
                
                if result and result.get("success"):
                    self.results.add_result(
                        "agent_message_processing",
                        True,
                        "Message processing successful"
                    )
                else:
                    self.results.add_result(
                        "agent_message_processing",
                        False,
                        "Message processing failed",
                        result.get("error") if result else "No result"
                    )
                
                # Test skill execution
                skill_result = await agent.execute_skill("test_skill", {"test_input": "value"})
                
                if skill_result and skill_result.get("success"):
                    self.results.add_result(
                        "agent_skill_execution",
                        True,
                        "Skill execution successful"
                    )
                else:
                    self.results.add_result(
                        "agent_skill_execution",
                        False,
                        "Skill execution failed",
                        skill_result.get("error") if skill_result else "No result"
                    )
                
                # Test request signing integration
                if agent.enable_request_signing and agent.request_signer:
                    signed_request = await agent.send_signed_request(
                        target_agent_id="target_agent",
                        method="POST",
                        path="/test",
                        body={"test": True}
                    )
                    
                    if 'X-A2A-Signature' in signed_request.get('headers', {}):
                        self.results.add_result(
                            "agent_request_signing",
                            True,
                            "Agent request signing integration successful"
                        )
                    else:
                        self.results.add_result(
                            "agent_request_signing",
                            False,
                            "Agent request signing failed"
                        )
                else:
                    self.results.add_result(
                        "agent_request_signing",
                        False,
                        "Request signing not enabled"
                    )
                
                # Test public key retrieval
                public_key = agent.get_public_key()
                if public_key and "-----BEGIN PUBLIC KEY-----" in public_key:
                    self.results.add_result(
                        "agent_public_key",
                        True,
                        "Public key retrieval successful"
                    )
                else:
                    self.results.add_result(
                        "agent_public_key",
                        False,
                        "Public key retrieval failed"
                    )
                
            except Exception as e:
                self.results.add_result(
                    "agent_integration",
                    False,
                    "Agent integration test failed",
                    str(e)
                )
                
        except Exception as e:
            self.results.add_result(
                "agent_overall",
                False,
                "Failed to test agent base",
                str(e)
            )
    
    async def test_network_connector_integration(self):
        """Test 6: Test network connector integration"""
        logger.info("🌐 Testing network connector integration...")
        
        try:
            from app.a2a.network.networkConnector import NetworkConnector, get_network_connector
            
            # Test connector creation
            try:
                connector = NetworkConnector()
                
                # Test initialization (will likely fail without network, but should handle gracefully)
                await connector.initialize()
                
                # Test status retrieval
                status = await connector.get_network_status()
                
                if status and "initialized" in status:
                    self.results.add_result(
                        "network_connector_status",
                        True,
                        f"Network connector status: {status['network_available']}"
                    )
                else:
                    self.results.add_result(
                        "network_connector_status",
                        False,
                        "Network connector status failed"
                    )
                
                # Test global instance
                global_connector = get_network_connector()
                if global_connector:
                    self.results.add_result(
                        "network_connector_singleton",
                        True,
                        "Global network connector instance successful"
                    )
                else:
                    self.results.add_result(
                        "network_connector_singleton",
                        False,
                        "Global network connector failed"
                    )
                
                # Test local agent registration (should work even without network)
                fake_agent = type('FakeAgent', (), {
                    'agent_id': 'test_network_agent',
                    'name': 'Test Network Agent',
                    'description': 'Test agent for network',
                    'version': '1.0.0',
                    'base_url': 'http://localhost:8000',
                    'skills': {'test_skill': 'test'}
                })()
                
                registration_result = await connector.register_agent(fake_agent)
                
                if registration_result and registration_result.get("success"):
                    self.results.add_result(
                        "network_agent_registration",
                        True,
                        f"Agent registration: {registration_result.get('registration_type', 'unknown')}"
                    )
                else:
                    self.results.add_result(
                        "network_agent_registration",
                        False,
                        "Agent registration failed",
                        registration_result.get("error") if registration_result else "No result"
                    )
                
                await connector.shutdown()
                
            except Exception as e:
                self.results.add_result(
                    "network_connector_integration",
                    False,
                    "Network connector integration failed",
                    str(e)
                )
                
        except Exception as e:
            self.results.add_result(
                "network_overall",
                False,
                "Failed to test network connector",
                str(e)
            )
    
    async def test_circular_dependencies(self):
        """Test 7: Check for circular dependencies"""
        logger.info("🔄 Testing for circular dependencies...")
        
        try:
            import ast
            import importlib.util
            
            # Key modules to check
            modules_to_check = [
                "app/core/config.py",
                "app/a2a/storage/distributedStorage.py", 
                "app/a2a/security/requestSigning.py",
                "app/a2a/sdk/agentBase.py",
                "app/a2a/network/networkConnector.py"
            ]
            
            dependency_graph = {}
            
            for module_path in modules_to_check:
                full_path = backend_dir / module_path
                if full_path.exists():
                    try:
                        with open(full_path, 'r') as f:
                            tree = ast.parse(f.read())
                        
                        imports = []
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    imports.append(alias.name)
                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    imports.append(node.module)
                        
                        # Filter to only A2A modules
                        a2a_imports = [imp for imp in imports if imp and ('app.' in imp or imp.startswith('.'))]
                        dependency_graph[module_path] = a2a_imports
                        
                    except Exception as e:
                        logger.warning(f"Could not analyze {module_path}: {e}")
            
            # Simple circular dependency detection
            circular_deps = []
            for module, deps in dependency_graph.items():
                for dep in deps:
                    if dep in dependency_graph:
                        if module.replace('/', '.').replace('.py', '') in dependency_graph[dep]:
                            circular_deps.append((module, dep))
            
            if circular_deps:
                self.results.add_result(
                    "circular_dependencies",
                    False,
                    "Circular dependencies found",
                    str(circular_deps)
                )
            else:
                self.results.add_result(
                    "circular_dependencies",
                    True,
                    "No obvious circular dependencies detected"
                )
                
        except Exception as e:
            self.results.add_result(
                "circular_dependencies",
                False,
                "Failed to check circular dependencies",
                str(e)
            )
    
    async def test_end_to_end_integration(self):
        """Test 8: End-to-end integration test"""
        logger.info("🎯 Testing end-to-end integration...")
        
        try:
            # Import all components
            from app.a2a.sdk.agentBase import A2AAgentBase
            from app.a2a.storage.distributedStorage import get_distributed_storage
            from app.a2a.network.networkConnector import get_network_connector
            from app.a2a.sdk.types import A2AMessage, MessagePart, MessageRole
            from app.a2a.sdk.decorators import a2a_handler, a2a_skill
            
            # Create test agent with all integrations
            class E2ETestAgent(A2AAgentBase):
                def __init__(self):
                    super().__init__(
                        agent_id="e2e_test_agent",
                        name="End-to-End Test Agent",
                        description="Agent for full integration testing",
                        version="1.0.0",
                        enable_telemetry=False,
                        enable_request_signing=True
                    )
                
                async def initialize(self):
                    # Initialize agent with distributed storage
                    self.storage = await get_distributed_storage()
                    self.network = get_network_connector()
                    await self.network.initialize()
                
                async def shutdown(self):
                    if hasattr(self, 'storage'):
                        await self.storage.disconnect()
                    if hasattr(self, 'network'):
                        await self.network.shutdown()
                
                @a2a_handler("e2e_test")
                async def handle_e2e_test(self, message: A2AMessage, context_id: str):
                    # Test full message processing pipeline
                    return {
                        "status": "success",
                        "message_id": message.messageId,
                        "context_id": context_id,
                        "agent_id": self.agent_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                @a2a_skill("e2e_skill", "End-to-end integration skill")
                async def e2e_skill(self, input_data: dict):
                    # Test skill with storage integration
                    if hasattr(self, 'storage'):
                        # Try to store and retrieve test data
                        test_key = f"e2e_test_{uuid4().hex[:8]}"
                        await self.storage.backend.set(test_key, input_data, ttl=60)
                        retrieved = await self.storage.backend.get(test_key)
                        
                        return {
                            "result": "e2e_integration_successful",
                            "storage_test": retrieved == input_data,
                            "input": input_data
                        }
                    else:
                        return {
                            "result": "e2e_integration_partial",
                            "input": input_data
                        }
            
            # Run end-to-end test
            try:
                agent = E2ETestAgent()
                await agent.initialize()
                
                # Test 1: Message processing with all integrations
                test_message = A2AMessage(
                    messageId=str(uuid4()),
                    role=MessageRole.USER,
                    parts=[MessagePart(kind="data", data={"method": "e2e_test", "test_data": "e2e_value"})]
                )
                
                message_result = await agent.process_message(test_message, "e2e_context")
                
                if message_result and message_result.get("success"):
                    self.results.add_result(
                        "e2e_message_processing",
                        True,
                        "End-to-end message processing successful"
                    )
                else:
                    self.results.add_result(
                        "e2e_message_processing",
                        False,
                        "End-to-end message processing failed",
                        message_result.get("error") if message_result else "No result"
                    )
                
                # Test 2: Skill execution with storage
                skill_result = await agent.execute_skill("e2e_skill", {"test_key": "test_value"})
                
                if skill_result and skill_result.get("success"):
                    skill_data = skill_result.get("result", {})
                    storage_test = skill_data.get("storage_test", False)
                    
                    self.results.add_result(
                        "e2e_skill_storage",
                        storage_test,
                        f"End-to-end skill with storage: {storage_test}"
                    )
                else:
                    self.results.add_result(
                        "e2e_skill_storage",
                        False,
                        "End-to-end skill execution failed",
                        skill_result.get("error") if skill_result else "No result"
                    )
                
                # Test 3: Agent registration with network
                if hasattr(agent, 'network'):
                    registration_result = await agent.network.register_agent(agent)
                    
                    if registration_result and registration_result.get("success"):
                        self.results.add_result(
                            "e2e_agent_registration",
                            True,
                            f"End-to-end agent registration: {registration_result.get('registration_type')}"
                        )
                    else:
                        self.results.add_result(
                            "e2e_agent_registration",
                            False,
                            "End-to-end agent registration failed",
                            registration_result.get("error") if registration_result else "No result"
                        )
                
                # Test 4: Request signing end-to-end
                if agent.enable_request_signing:
                    signed_request = await agent.send_signed_request(
                        target_agent_id="e2e_target",
                        method="POST",
                        path="/e2e/test",
                        body={"e2e": True}
                    )
                    
                    # Verify our own request
                    headers = signed_request.get("headers", {})
                    is_valid, error = await agent.verify_incoming_request(
                        headers=headers,
                        method="POST",
                        path="/e2e/test",
                        body={"e2e": True}
                    )
                    
                    if is_valid and not error:
                        self.results.add_result(
                            "e2e_request_signing",
                            True,
                            "End-to-end request signing successful"
                        )
                    else:
                        self.results.add_result(
                            "e2e_request_signing",
                            False,
                            "End-to-end request signing verification failed",
                            error
                        )
                
                await agent.shutdown()
                
            except Exception as e:
                self.results.add_result(
                    "e2e_integration",
                    False,
                    "End-to-end integration test failed",
                    str(e)
                )
                
        except Exception as e:
            self.results.add_result(
                "e2e_overall",
                False,
                "Failed to run end-to-end test",
                str(e)
            )
    
    async def cleanup(self):
        """Clean up test artifacts"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_file}: {e}")
        
        # Clean up temporary registry files
        temp_files = [
            "/tmp/a2a_local_registry.json",
            "/tmp/a2a_local_data_products.json"
        ]
        
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_file}: {e}")


async def main():
    """Run integration tests"""
    tester = A2AIntegrationTester()
    
    try:
        results = await tester.run_all_tests()
        summary = results.get_summary()
        
        # Print summary
        print("\n" + "="*60)
        print("🏁 A2A ARCHITECTURE INTEGRATION TEST SUMMARY")
        print("="*60)
        print(f"Overall Status: {'✅ PASS' if summary['overall_success'] else '❌ FAIL'}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print()
        
        # Print failed tests
        if summary['failed'] > 0:
            print("❌ FAILED TESTS:")
            for test_name, result in summary['results'].items():
                if not result['success']:
                    print(f"  - {test_name}: {result['details']}")
                    if result['error']:
                        print(f"    Error: {result['error']}")
            print()
        
        # Print passed tests  
        print("✅ PASSED TESTS:")
        for test_name, result in summary['results'].items():
            if result['success']:
                print(f"  - {test_name}: {result['details']}")
        print()
        
        # Save detailed report
        with open("integration_test_report.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Detailed report saved to: integration_test_report.json")
        print(f"📄 Log file: integration_test.log")
        
        # Exit with appropriate code
        sys.exit(0 if summary['overall_success'] else 1)
        
    except Exception as e:
        logger.error(f"Integration test suite failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())