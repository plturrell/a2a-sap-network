"""
Comprehensive Testing Framework for A2A Agents
Provides unit, integration, and E2E testing capabilities with agent-specific patterns
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Type, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import inspect
import tempfile
import shutil
import os

# Testing imports
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient
import fakeredis
import sqlite3

# A2A imports
from ..a2a.sdk.agentBase import A2AAgentBase
from ..a2a.sdk.types import A2AMessage, MessagePart, MessageRole, TaskStatus
from ..a2a.core.telemetry import trace_async, add_span_attributes
from .standardized_lifecycle import StandardizedLifecycleManager
from ..clients.redisClient import RedisClient, RedisConfig

logger = logging.getLogger(__name__)


class TestLevel(str, Enum):
    """Test execution levels"""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    CHAOS = "chaos"


class TestResult(str, Enum):
    """Test result status"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """Individual test case definition"""
    test_id: str
    name: str
    description: str
    test_level: TestLevel
    test_function: Callable
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    timeout_seconds: int = 60
    retry_count: int = 0
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    fixtures: List[str] = field(default_factory=list)


@dataclass
class TestExecution:
    """Test execution result"""
    test_case: TestCase
    result: TestResult
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    error_message: Optional[str] = None
    output: Optional[str] = None
    coverage_data: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class TestFixture(ABC):
    """Abstract base class for test fixtures"""
    
    @abstractmethod
    async def setup(self) -> Any:
        """Setup the fixture"""
        pass
    
    @abstractmethod
    async def teardown(self):
        """Teardown the fixture"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get fixture name"""
        pass


class MockA2AAgent(A2AAgentBase):
    """Mock A2A agent for testing"""
    
    def __init__(self, agent_id: str = "test_agent"):
        super().__init__(
            agent_id=agent_id,
            name="Test Agent",
            description="Mock agent for testing",
            version="1.0.0-test"
        )
        
        self.initialization_calls = 0
        self.shutdown_calls = 0
        self.message_calls = 0
        self.task_calls = 0
        
        # Mock data
        self.mock_responses = {}
        self.mock_delays = {}
        self.mock_failures = {}
    
    async def initialize(self):
        """Mock initialization"""
        self.initialization_calls += 1
        if "initialize" in self.mock_failures:
            raise Exception(self.mock_failures["initialize"])
        
        if "initialize" in self.mock_delays:
            await asyncio.sleep(self.mock_delays["initialize"])
    
    async def shutdown(self):
        """Mock shutdown"""
        self.shutdown_calls += 1
        if "shutdown" in self.mock_failures:
            raise Exception(self.mock_failures["shutdown"])
        
        if "shutdown" in self.mock_delays:
            await asyncio.sleep(self.mock_delays["shutdown"])
    
    async def process_test_message(self, message: A2AMessage) -> Dict[str, Any]:
        """Mock message processing"""
        self.message_calls += 1
        
        if "process_message" in self.mock_failures:
            raise Exception(self.mock_failures["process_message"])
        
        if "process_message" in self.mock_delays:
            await asyncio.sleep(self.mock_delays["process_message"])
        
        return self.mock_responses.get("process_message", {"success": True})
    
    def set_mock_response(self, method: str, response: Any):
        """Set mock response for a method"""
        self.mock_responses[method] = response
    
    def set_mock_delay(self, method: str, delay: float):
        """Set mock delay for a method"""
        self.mock_delays[method] = delay
    
    def set_mock_failure(self, method: str, error: str):
        """Set mock failure for a method"""
        self.mock_failures[method] = error


class RedisFixture(TestFixture):
    """Redis test fixture using fakeredis"""
    
    def __init__(self):
        self.redis_server = None
        self.redis_client = None
    
    async def setup(self) -> RedisClient:
        """Setup fake Redis server"""
        import fakeredis.aioredis
        
        # Create fake Redis server
        self.redis_server = fakeredis.FakeServer()
        fake_redis = fakeredis.aioredis.FakeRedis(server=self.redis_server)
        
        # Create Redis client wrapper
        config = RedisConfig(
            host="localhost",
            port=6379,
            db=0
        )
        self.redis_client = RedisClient(config)
        self.redis_client._redis = fake_redis
        
        return self.redis_client
    
    async def teardown(self):
        """Teardown Redis fixture"""
        if self.redis_client:
            await self.redis_client.close()
        if self.redis_server:
            self.redis_server.close()
    
    def get_name(self) -> str:
        return "redis"


class DatabaseFixture(TestFixture):
    """SQLite database test fixture"""
    
    def __init__(self):
        self.db_path = None
        self.connection = None
    
    async def setup(self) -> str:
        """Setup temporary SQLite database"""
        # Create temporary database file
        fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        # Initialize database schema
        self.connection = sqlite3.connect(self.db_path)
        await self._create_test_schema()
        
        return self.db_path
    
    async def teardown(self):
        """Teardown database fixture"""
        if self.connection:
            self.connection.close()
        if self.db_path and os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    async def _create_test_schema(self):
        """Create test database schema"""
        cursor = self.connection.cursor()
        
        # Create test tables
        cursor.execute("""
            CREATE TABLE test_agents (
                agent_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE test_tasks (
                task_id TEXT PRIMARY KEY,
                agent_id TEXT,
                status TEXT NOT NULL,
                payload TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_id) REFERENCES test_agents (agent_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE test_messages (
                message_id TEXT PRIMARY KEY,
                agent_id TEXT,
                role TEXT NOT NULL,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_id) REFERENCES test_agents (agent_id)
            )
        """)
        
        self.connection.commit()
    
    def get_name(self) -> str:
        return "database"


class HttpServerFixture(TestFixture):
    """HTTP server fixture for testing"""
    
    def __init__(self, port: int = 0):
        self.port = port
        self.server = None
        self.actual_port = None
    
    async def setup(self) -> Dict[str, Any]:
        """Setup test HTTP server"""
        from fastapi import FastAPI
        import uvicorn
        
        app = FastAPI(title="Test Server")
        
        @app.get("/health")
        async def health():
            return {"status": "healthy"}
        
        @app.post("/test-endpoint")
        async def test_endpoint(data: Dict[str, Any]):
            return {"received": data, "timestamp": datetime.utcnow().isoformat()}
        
        # Find available port
        import socket
        if self.port == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                self.actual_port = s.getsockname()[1]
        else:
            self.actual_port = self.port
        
        # Start server in background
        config = uvicorn.Config(app, host="127.0.0.1", port=self.actual_port, log_level="error")
        self.server = uvicorn.Server(config)
        
        # Start server task
        server_task = asyncio.create_task(self.server.serve())
        
        # Wait for server to start
        await asyncio.sleep(0.1)
        
        return {
            "url": f"http://127.0.0.1:{self.actual_port}",
            "port": self.actual_port,
            "server_task": server_task
        }
    
    async def teardown(self):
        """Teardown HTTP server"""
        if self.server:
            self.server.should_exit = True
            await asyncio.sleep(0.1)
    
    def get_name(self) -> str:
        return "http_server"


class AgentFixture(TestFixture):
    """A2A agent test fixture"""
    
    def __init__(self, agent_class: Type[A2AAgentBase] = MockA2AAgent, **agent_kwargs):
        self.agent_class = agent_class
        self.agent_kwargs = agent_kwargs
        self.agent = None
        self.lifecycle_manager = None
    
    async def setup(self) -> A2AAgentBase:
        """Setup agent for testing"""
        # Create agent instance
        self.agent = self.agent_class(**self.agent_kwargs)
        
        # Create lifecycle manager
        self.lifecycle_manager = StandardizedLifecycleManager()
        await self.lifecycle_manager.initialize()
        
        # Register and initialize agent
        self.lifecycle_manager.register_agent(self.agent)
        await self.lifecycle_manager.initialize_agent(self.agent.agent_id)
        
        return self.agent
    
    async def teardown(self):
        """Teardown agent fixture"""
        if self.agent and self.lifecycle_manager:
            await self.lifecycle_manager.shutdown_agent(self.agent.agent_id)
        
        if self.lifecycle_manager:
            await self.lifecycle_manager.shutdown()
    
    def get_name(self) -> str:
        return f"agent_{self.agent_class.__name__}"


class A2ATestFramework:
    """Comprehensive testing framework for A2A agents"""
    
    def __init__(self):
        self.test_cases: Dict[str, TestCase] = {}
        self.fixtures: Dict[str, TestFixture] = {}
        self.test_results: List[TestExecution] = []
        self.active_fixtures: Dict[str, Any] = {}
        
        # Register default fixtures
        self._register_default_fixtures()
        
        # Test configuration
        self.parallel_execution = True
        self.max_workers = 4
        self.default_timeout = 60
        
    def _register_default_fixtures(self):
        """Register default test fixtures"""
        self.fixtures["redis"] = RedisFixture()
        self.fixtures["database"] = DatabaseFixture()
        self.fixtures["http_server"] = HttpServerFixture()
        self.fixtures["mock_agent"] = AgentFixture(MockA2AAgent)
    
    def register_fixture(self, name: str, fixture: TestFixture):
        """Register a custom test fixture"""
        self.fixtures[name] = fixture
        logger.info(f"Registered test fixture: {name}")
    
    def register_test_case(
        self,
        name: str,
        description: str,
        test_function: Callable,
        test_level: TestLevel = TestLevel.UNIT,
        setup_function: Optional[Callable] = None,
        teardown_function: Optional[Callable] = None,
        timeout_seconds: int = 60,
        tags: List[str] = None,
        fixtures: List[str] = None
    ) -> str:
        """Register a test case"""
        
        test_id = str(uuid.uuid4())
        
        test_case = TestCase(
            test_id=test_id,
            name=name,
            description=description,
            test_level=test_level,
            test_function=test_function,
            setup_function=setup_function,
            teardown_function=teardown_function,
            timeout_seconds=timeout_seconds,
            tags=tags or [],
            fixtures=fixtures or []
        )
        
        self.test_cases[test_id] = test_case
        logger.info(f"Registered test case: {name} ({test_level})")
        
        return test_id
    
    async def setup_fixtures(self, fixture_names: List[str]) -> Dict[str, Any]:
        """Setup required fixtures"""
        fixture_instances = {}
        
        for fixture_name in fixture_names:
            if fixture_name not in self.fixtures:
                raise ValueError(f"Unknown fixture: {fixture_name}")
            
            fixture = self.fixtures[fixture_name]
            try:
                instance = await fixture.setup()
                fixture_instances[fixture_name] = instance
                self.active_fixtures[fixture_name] = fixture
                logger.debug(f"Setup fixture: {fixture_name}")
            except Exception as e:
                # Cleanup already setup fixtures
                await self.teardown_fixtures(list(fixture_instances.keys()))
                raise Exception(f"Failed to setup fixture {fixture_name}: {str(e)}")
        
        return fixture_instances
    
    async def teardown_fixtures(self, fixture_names: List[str]):
        """Teardown fixtures"""
        for fixture_name in reversed(fixture_names):  # Teardown in reverse order
            if fixture_name in self.active_fixtures:
                try:
                    await self.active_fixtures[fixture_name].teardown()
                    del self.active_fixtures[fixture_name]
                    logger.debug(f"Tore down fixture: {fixture_name}")
                except Exception as e:
                    logger.error(f"Failed to teardown fixture {fixture_name}: {str(e)}")
    
    @trace_async("execute_test")
    async def execute_test(self, test_id: str) -> TestExecution:
        """Execute a single test case"""
        
        if test_id not in self.test_cases:
            raise ValueError(f"Unknown test case: {test_id}")
        
        test_case = self.test_cases[test_id]
        
        add_span_attributes({
            "test.id": test_id,
            "test.name": test_case.name,
            "test.level": test_case.test_level.value
        })
        
        start_time = datetime.utcnow()
        
        try:
            # Setup fixtures
            fixture_instances = {}
            if test_case.fixtures:
                fixture_instances = await self.setup_fixtures(test_case.fixtures)
            
            # Run setup function
            if test_case.setup_function:
                if asyncio.iscoroutinefunction(test_case.setup_function):
                    await test_case.setup_function()
                else:
                    test_case.setup_function()
            
            # Execute test function with timeout
            test_kwargs = {}
            
            # Inject fixtures as parameters
            sig = inspect.signature(test_case.test_function)
            for param_name in sig.parameters:
                if param_name in fixture_instances:
                    test_kwargs[param_name] = fixture_instances[param_name]
            
            if asyncio.iscoroutinefunction(test_case.test_function):
                await asyncio.wait_for(
                    test_case.test_function(**test_kwargs),
                    timeout=test_case.timeout_seconds
                )
            else:
                test_case.test_function(**test_kwargs)
            
            # Test passed
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            execution = TestExecution(
                test_case=test_case,
                result=TestResult.PASSED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration
            )
            
            logger.info(f"Test PASSED: {test_case.name} ({duration:.2f}s)")
            
        except asyncio.TimeoutError:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            execution = TestExecution(
                test_case=test_case,
                result=TestResult.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                error_message=f"Test timeout after {test_case.timeout_seconds}s"
            )
            
            logger.error(f"Test TIMEOUT: {test_case.name}")
            
        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            execution = TestExecution(
                test_case=test_case,
                result=TestResult.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                error_message=str(e)
            )
            
            logger.error(f"Test FAILED: {test_case.name}: {str(e)}")
        
        finally:
            # Run teardown function
            if test_case.teardown_function:
                try:
                    if asyncio.iscoroutinefunction(test_case.teardown_function):
                        await test_case.teardown_function()
                    else:
                        test_case.teardown_function()
                except Exception as e:
                    logger.error(f"Teardown failed for {test_case.name}: {str(e)}")
            
            # Teardown fixtures
            if test_case.fixtures:
                await self.teardown_fixtures(test_case.fixtures)
        
        self.test_results.append(execution)
        return execution
    
    async def run_tests(
        self,
        test_level: Optional[TestLevel] = None,
        tags: Optional[List[str]] = None,
        test_pattern: Optional[str] = None
    ) -> List[TestExecution]:
        """Run multiple tests with filtering"""
        
        # Filter tests
        tests_to_run = []
        
        for test_case in self.test_cases.values():
            # Level filter
            if test_level and test_case.test_level != test_level:
                continue
            
            # Tags filter
            if tags and not any(tag in test_case.tags for tag in tags):
                continue
            
            # Pattern filter
            if test_pattern and test_pattern not in test_case.name:
                continue
            
            tests_to_run.append(test_case.test_id)
        
        logger.info(f"Running {len(tests_to_run)} tests")
        
        # Execute tests
        if self.parallel_execution and len(tests_to_run) > 1:
            # Run tests in parallel with semaphore
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def run_with_semaphore(test_id):
                async with semaphore:
                    return await self.execute_test(test_id)
            
            tasks = [run_with_semaphore(test_id) for test_id in tests_to_run]
            executions = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_executions = [e for e in executions if isinstance(e, TestExecution)]
            
        else:
            # Run tests sequentially
            valid_executions = []
            for test_id in tests_to_run:
                execution = await self.execute_test(test_id)
                valid_executions.append(execution)
        
        return valid_executions
    
    def generate_test_report(self, executions: List[TestExecution]) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        total_tests = len(executions)
        passed_tests = len([e for e in executions if e.result == TestResult.PASSED])
        failed_tests = len([e for e in executions if e.result == TestResult.FAILED])
        
        # Calculate statistics
        total_duration = sum(e.duration_seconds for e in executions)
        avg_duration = total_duration / total_tests if total_tests > 0 else 0
        
        # Group by test level
        by_level = {}
        for execution in executions:
            level = execution.test_case.test_level.value
            if level not in by_level:
                by_level[level] = {"total": 0, "passed": 0, "failed": 0}
            
            by_level[level]["total"] += 1
            if execution.result == TestResult.PASSED:
                by_level[level]["passed"] += 1
            else:
                by_level[level]["failed"] += 1
        
        # Failed test details
        failed_test_details = []
        for execution in executions:
            if execution.result == TestResult.FAILED:
                failed_test_details.append({
                    "test_name": execution.test_case.name,
                    "test_level": execution.test_case.test_level.value,
                    "error": execution.error_message,
                    "duration": execution.duration_seconds
                })
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "total_duration_seconds": total_duration,
                "average_duration_seconds": avg_duration
            },
            "by_level": by_level,
            "failed_tests": failed_test_details,
            "execution_timestamp": datetime.utcnow().isoformat()
        }
    
    def export_junit_report(self, executions: List[TestExecution], output_path: str):
        """Export test results in JUnit XML format"""
        import xml.etree.ElementTree as ET
        
        # Create root element
        testsuite = ET.Element("testsuite")
        testsuite.set("name", "A2A Agent Tests")
        testsuite.set("tests", str(len(executions)))
        testsuite.set("failures", str(len([e for e in executions if e.result == TestResult.FAILED])))
        testsuite.set("time", str(sum(e.duration_seconds for e in executions)))
        
        # Add test cases
        for execution in executions:
            testcase = ET.SubElement(testsuite, "testcase")
            testcase.set("name", execution.test_case.name)
            testcase.set("classname", execution.test_case.test_level.value)
            testcase.set("time", str(execution.duration_seconds))
            
            if execution.result == TestResult.FAILED:
                failure = ET.SubElement(testcase, "failure")
                failure.set("message", execution.error_message or "Test failed")
                failure.text = execution.error_message
        
        # Write to file
        tree = ET.ElementTree(testsuite)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        
        logger.info(f"JUnit report exported to: {output_path}")


# Test decorators for easy test registration
def a2a_test(
    name: str = None,
    description: str = "",
    test_level: TestLevel = TestLevel.UNIT,
    tags: List[str] = None,
    fixtures: List[str] = None,
    timeout: int = 60
):
    """Decorator to register A2A test cases"""
    def decorator(func):
        test_name = name or func.__name__
        
        # Get global test framework instance
        framework = get_test_framework()
        framework.register_test_case(
            name=test_name,
            description=description,
            test_function=func,
            test_level=test_level,
            tags=tags or [],
            fixtures=fixtures or [],
            timeout_seconds=timeout
        )
        
        return func
    return decorator


def a2a_unit_test(name: str = None, **kwargs):
    """Decorator for unit tests"""
    return a2a_test(name=name, test_level=TestLevel.UNIT, **kwargs)


def a2a_integration_test(name: str = None, **kwargs):
    """Decorator for integration tests"""
    return a2a_test(name=name, test_level=TestLevel.INTEGRATION, **kwargs)


def a2a_e2e_test(name: str = None, **kwargs):
    """Decorator for E2E tests"""
    return a2a_test(name=name, test_level=TestLevel.E2E, **kwargs)


# Global test framework instance
_test_framework = None


def get_test_framework() -> A2ATestFramework:
    """Get global test framework instance"""
    global _test_framework
    
    if _test_framework is None:
        _test_framework = A2ATestFramework()
    
    return _test_framework


# Example test cases
@a2a_unit_test(
    name="test_agent_initialization",
    description="Test agent initialization process",
    fixtures=["mock_agent"]
)
async def test_agent_initialization(mock_agent):
    """Test agent initialization"""
    assert mock_agent.initialization_calls == 1
    assert mock_agent.agent_id == "test_agent"
    assert mock_agent.name == "Test Agent"


@a2a_unit_test(
    name="test_agent_message_processing",
    description="Test agent message processing",
    fixtures=["mock_agent"]
)
async def test_agent_message_processing(mock_agent):
    """Test agent message processing"""
    # Create test message
    message = A2AMessage(
        role=MessageRole.USER,
        parts=[MessagePart(kind="text", text="Hello, agent!")]
    )
    
    # Process message
    result = await mock_agent.process_test_message(message)
    
    assert result["success"] == True
    assert mock_agent.message_calls == 1


@a2a_integration_test(
    name="test_agent_lifecycle_integration",
    description="Test complete agent lifecycle",
    fixtures=["mock_agent", "redis", "database"]
)
async def test_agent_lifecycle_integration(mock_agent, redis, database):
    """Test agent lifecycle integration"""
    # Test that agent can interact with Redis
    await redis.set("test_key", "test_value")
    value = await redis.get("test_key")
    assert value == "test_value"
    
    # Test database integration
    import sqlite3
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO test_agents (agent_id, name, status) VALUES (?, ?, ?)",
                  (mock_agent.agent_id, mock_agent.name, "active"))
    conn.commit()
    
    cursor.execute("SELECT name FROM test_agents WHERE agent_id = ?", (mock_agent.agent_id,))
    result = cursor.fetchone()
    assert result[0] == mock_agent.name
    
    conn.close()


@a2a_e2e_test(
    name="test_agent_http_api",
    description="Test agent HTTP API endpoints",
    fixtures=["mock_agent", "http_server"]
)
async def test_agent_http_api(mock_agent, http_server):
    """Test agent HTTP API"""
    base_url = http_server["url"]
    
    # Test health endpoint
    async with AsyncClient() as client:
        response = await client.get(f"{base_url}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
        # Test custom endpoint
        test_data = {"message": "Hello, API!"}
        response = await client.post(f"{base_url}/test-endpoint", json=test_data)
        assert response.status_code == 200
        result = response.json()
        assert result["received"] == test_data