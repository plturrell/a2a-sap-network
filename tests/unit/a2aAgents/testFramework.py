"""
Comprehensive integration test framework for A2A agents.
Provides production-ready testing capabilities.
"""
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pytest
import httpx
from unittest.mock import Mock, patch

logger = logging.getLogger(__name__)


class TestResult(Enum):
    """Test result types."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestSeverity(Enum):
    """Test severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TestCase:
    """Individual test case definition."""
    test_id: str
    name: str
    description: str
    agent_type: str
    test_function: callable
    setup_function: Optional[callable] = None
    teardown_function: Optional[callable] = None
    timeout: int = 30
    severity: TestSeverity = TestSeverity.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    expected_duration: Optional[float] = None


@dataclass
class TestExecution:
    """Test execution result."""
    test_case: TestCase
    result: TestResult
    duration: float
    start_time: datetime
    end_time: datetime
    output: Optional[str] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuite:
    """Collection of related test cases."""
    suite_id: str
    name: str
    description: str
    test_cases: List[TestCase] = field(default_factory=list)
    setup_suite: Optional[callable] = None
    teardown_suite: Optional[callable] = None


class A2ATestEnvironment:
    """Test environment management for A2A agents."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        blockchain_test_mode: bool = True,
        mock_external_services: bool = True
    ):
        """
        Initialize test environment.
        
        Args:
            base_url: Base URL for agent services
            blockchain_test_mode: Use test blockchain
            mock_external_services: Mock external dependencies
        """
        self.base_url = base_url
        self.blockchain_test_mode = blockchain_test_mode
        self.mock_external_services = mock_external_services
        
        # Test data
        self.test_data = {}
        self.test_agents = {}
        self.mock_services = {}
        
        # HTTP client for API calls
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
        
        # Environment state
        self._is_setup = False
        self._setup_start_time = None
    
    async def setup_environment(self):
        """Set up test environment."""
        if self._is_setup:
            return
        
        self._setup_start_time = datetime.now()
        logger.info("Setting up A2A test environment...")
        
        # Setup mock services if enabled
        if self.mock_external_services:
            await self._setup_mock_services()
        
        # Initialize test agents
        await self._initialize_test_agents()
        
        # Create test data
        await self._create_test_data()
        
        # Verify environment health
        await self._verify_environment_health()
        
        self._is_setup = True
        setup_duration = (datetime.now() - self._setup_start_time).total_seconds()
        logger.info(f"Test environment setup completed in {setup_duration:.2f}s")
    
    async def teardown_environment(self):
        """Tear down test environment."""
        if not self._is_setup:
            return
        
        logger.info("Tearing down A2A test environment...")
        
        # Cleanup test data
        await self._cleanup_test_data()
        
        # Stop test agents
        await self._stop_test_agents()
        
        # Cleanup mock services
        if self.mock_external_services:
            await self._cleanup_mock_services()
        
        # Close HTTP client
        await self.client.aclose()
        
        self._is_setup = False
        logger.info("Test environment teardown completed")
    
    async def _setup_mock_services(self):
        """Setup mock external services."""
        # Mock blockchain service
        if self.blockchain_test_mode:
            self.mock_services['blockchain'] = Mock()
            self.mock_services['blockchain'].send_transaction = Mock(
                return_value={'success': True, 'tx_hash': 'mock_hash'}
            )
        
        # Mock catalog manager
        self.mock_services['catalog'] = Mock()
        self.mock_services['catalog'].get_services = Mock(
            return_value=[
                {
                    'id': 'test_service_1',
                    'name': 'Test Service 1',
                    'endpoints': ['http://test1.example.com']
                }
            ]
        )
        
        # Mock data manager
        self.mock_services['data_manager'] = Mock()
        self.mock_services['data_manager'].store_data = Mock(
            return_value={'success': True, 'data_id': 'mock_data_id'}
        )
    
    async def _initialize_test_agents(self):
        """Initialize test agent instances."""
        # Test agents configuration
        test_agents_config = [
            {
                'type': 'data_product',
                'agent_id': 'test_data_product_agent',
                'port': 8001
            },
            {
                'type': 'qa_validation',
                'agent_id': 'test_qa_validation_agent',
                'port': 8007
            },
            {
                'type': 'calculation',
                'agent_id': 'test_calculation_agent',
                'port': 8004
            }
        ]
        
        for config in test_agents_config:
            agent_url = f"{self.base_url.replace('8000', str(config['port']))}"
            self.test_agents[config['type']] = {
                'url': agent_url,
                'agent_id': config['agent_id'],
                'config': config
            }
    
    async def _create_test_data(self):
        """Create test data for integration tests."""
        self.test_data = {
            'sample_questions': [
                {
                    'id': 'q1',
                    'text': 'What is the capital of France?',
                    'expected_answer': 'Paris',
                    'difficulty': 'easy'
                },
                {
                    'id': 'q2',
                    'text': 'Calculate the square root of 144',
                    'expected_answer': '12',
                    'difficulty': 'medium'
                }
            ],
            'sample_data_products': [
                {
                    'id': 'dp1',
                    'name': 'Test Dataset 1',
                    'data': {'values': [1, 2, 3, 4, 5]},
                    'schema': {'type': 'array', 'items': {'type': 'number'}}
                }
            ],
            'test_calculations': [
                {
                    'id': 'calc1',
                    'operation': 'sum',
                    'inputs': [10, 20, 30],
                    'expected_result': 60
                },
                {
                    'id': 'calc2',
                    'operation': 'average',
                    'inputs': [1, 2, 3, 4, 5],
                    'expected_result': 3.0
                }
            ]
        }
    
    async def _verify_environment_health(self):
        """Verify test environment health."""
        health_checks = []
        
        for agent_type, agent_info in self.test_agents.items():
            try:
                response = await self.client.get(f"{agent_info['url']}/health")
                if response.status_code == 200:
                    health_checks.append((agent_type, True, None))
                else:
                    health_checks.append((agent_type, False, f"HTTP {response.status_code}"))
            except Exception as e:
                health_checks.append((agent_type, False, str(e)))
        
        failed_checks = [check for check in health_checks if not check[1]]
        if failed_checks:
            logger.warning(f"Some agents failed health checks: {failed_checks}")
    
    async def _cleanup_test_data(self):
        """Clean up test data."""
        # Clear any persistent test data
        for agent_type, agent_info in self.test_agents.items():
            try:
                await self.client.post(f"{agent_info['url']}/test/cleanup")
            except Exception as e:
                logger.warning(f"Failed to cleanup {agent_type}: {e}")
    
    async def _stop_test_agents(self):
        """Stop test agent instances."""
        # In real implementation, would stop agent processes
        pass
    
    async def _cleanup_mock_services(self):
        """Clean up mock services."""
        self.mock_services.clear()
    
    async def call_agent_api(
        self,
        agent_type: str,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None
    ) -> httpx.Response:
        """Make API call to agent."""
        if agent_type not in self.test_agents:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_url = self.test_agents[agent_type]['url']
        url = f"{agent_url}{endpoint}"
        
        if method.upper() == "GET":
            return await self.client.get(url)
        elif method.upper() == "POST":
            return await self.client.post(url, json=data)
        elif method.upper() == "PUT":
            return await self.client.put(url, json=data)
        elif method.upper() == "DELETE":
            return await self.client.delete(url)
        else:
            raise ValueError(f"Unsupported method: {method}")


class A2ATestRunner:
    """Test runner for A2A integration tests."""
    
    def __init__(self, environment: A2ATestEnvironment):
        """Initialize test runner."""
        self.environment = environment
        self.test_suites = {}
        self.test_results = []
        self.current_execution = None
    
    def register_test_suite(self, test_suite: TestSuite):
        """Register a test suite."""
        self.test_suites[test_suite.suite_id] = test_suite
        logger.info(f"Registered test suite: {test_suite.name}")
    
    def register_test_case(
        self,
        suite_id: str,
        test_case: TestCase
    ):
        """Register a test case to a suite."""
        if suite_id not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_id}")
        
        self.test_suites[suite_id].test_cases.append(test_case)
        logger.debug(f"Registered test case: {test_case.name}")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all registered tests."""
        logger.info("Starting A2A integration test run")
        start_time = datetime.now()
        
        # Setup environment
        await self.environment.setup_environment()
        
        try:
            # Run all test suites
            suite_results = {}
            for suite_id, suite in self.test_suites.items():
                suite_results[suite_id] = await self.run_test_suite(suite)
            
            # Generate summary
            summary = self._generate_test_summary(suite_results, start_time)
            
            return {
                'summary': summary,
                'suites': suite_results,
                'environment': {
                    'base_url': self.environment.base_url,
                    'blockchain_test_mode': self.environment.blockchain_test_mode,
                    'mock_external_services': self.environment.mock_external_services
                }
            }
            
        finally:
            # Cleanup environment
            await self.environment.teardown_environment()
    
    async def run_test_suite(self, test_suite: TestSuite) -> Dict[str, Any]:
        """Run a specific test suite."""
        logger.info(f"Running test suite: {test_suite.name}")
        start_time = datetime.now()
        
        # Setup suite
        if test_suite.setup_suite:
            try:
                await test_suite.setup_suite(self.environment)
            except Exception as e:
                logger.error(f"Suite setup failed: {e}")
                return {
                    'name': test_suite.name,
                    'status': 'setup_failed',
                    'error': str(e),
                    'test_results': []
                }
        
        try:
            # Run test cases
            test_results = []
            for test_case in test_suite.test_cases:
                # Check dependencies
                if not self._check_dependencies(test_case, test_results):
                    result = TestExecution(
                        test_case=test_case,
                        result=TestResult.SKIPPED,
                        duration=0.0,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        output="Skipped due to failed dependencies"
                    )
                    test_results.append(result)
                    continue
                
                # Run test case
                result = await self.run_test_case(test_case)
                test_results.append(result)
                self.test_results.append(result)
            
            # Generate suite summary
            passed = sum(1 for r in test_results if r.result == TestResult.PASSED)
            failed = sum(1 for r in test_results if r.result == TestResult.FAILED)
            skipped = sum(1 for r in test_results if r.result == TestResult.SKIPPED)
            errors = sum(1 for r in test_results if r.result == TestResult.ERROR)
            
            return {
                'name': test_suite.name,
                'description': test_suite.description,
                'status': 'completed',
                'duration': (datetime.now() - start_time).total_seconds(),
                'summary': {
                    'total': len(test_results),
                    'passed': passed,
                    'failed': failed,
                    'skipped': skipped,
                    'errors': errors
                },
                'test_results': [
                    {
                        'test_id': r.test_case.test_id,
                        'name': r.test_case.name,
                        'result': r.result.value,
                        'duration': r.duration,
                        'error': r.error,
                        'output': r.output
                    }
                    for r in test_results
                ]
            }
            
        finally:
            # Teardown suite
            if test_suite.teardown_suite:
                try:
                    await test_suite.teardown_suite(self.environment)
                except Exception as e:
                    logger.warning(f"Suite teardown failed: {e}")
    
    async def run_test_case(self, test_case: TestCase) -> TestExecution:
        """Run a specific test case."""
        logger.debug(f"Running test case: {test_case.name}")
        start_time = datetime.now()
        
        # Setup test case
        if test_case.setup_function:
            try:
                await test_case.setup_function(self.environment)
            except Exception as e:
                return TestExecution(
                    test_case=test_case,
                    result=TestResult.ERROR,
                    duration=0.0,
                    start_time=start_time,
                    end_time=datetime.now(),
                    error=f"Setup failed: {str(e)}"
                )
        
        try:
            # Run test with timeout
            result_output = await asyncio.wait_for(
                test_case.test_function(self.environment),
                timeout=test_case.timeout
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Check expected duration
            if test_case.expected_duration and duration > test_case.expected_duration * 1.5:
                logger.warning(
                    f"Test {test_case.name} took {duration:.2f}s, "
                    f"expected ~{test_case.expected_duration:.2f}s"
                )
            
            return TestExecution(
                test_case=test_case,
                result=TestResult.PASSED,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                output=str(result_output) if result_output else None
            )
            
        except asyncio.TimeoutError:
            return TestExecution(
                test_case=test_case,
                result=TestResult.ERROR,
                duration=test_case.timeout,
                start_time=start_time,
                end_time=datetime.now(),
                error=f"Test timed out after {test_case.timeout}s"
            )
            
        except AssertionError as e:
            return TestExecution(
                test_case=test_case,
                result=TestResult.FAILED,
                duration=(datetime.now() - start_time).total_seconds(),
                start_time=start_time,
                end_time=datetime.now(),
                error=f"Assertion failed: {str(e)}"
            )
            
        except Exception as e:
            return TestExecution(
                test_case=test_case,
                result=TestResult.ERROR,
                duration=(datetime.now() - start_time).total_seconds(),
                start_time=start_time,
                end_time=datetime.now(),
                error=f"Unexpected error: {str(e)}"
            )
            
        finally:
            # Teardown test case
            if test_case.teardown_function:
                try:
                    await test_case.teardown_function(self.environment)
                except Exception as e:
                    logger.warning(f"Test teardown failed: {e}")
    
    def _check_dependencies(
        self,
        test_case: TestCase,
        completed_tests: List[TestExecution]
    ) -> bool:
        """Check if test dependencies are satisfied."""
        if not test_case.dependencies:
            return True
        
        completed_test_ids = {
            r.test_case.test_id for r in completed_tests
            if r.result == TestResult.PASSED
        }
        
        return all(dep in completed_test_ids for dep in test_case.dependencies)
    
    def _generate_test_summary(
        self,
        suite_results: Dict[str, Any],
        start_time: datetime
    ) -> Dict[str, Any]:
        """Generate test run summary."""
        total_duration = (datetime.now() - start_time).total_seconds()
        
        # Aggregate results
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        total_errors = 0
        
        for suite_result in suite_results.values():
            if 'summary' in suite_result:
                summary = suite_result['summary']
                total_tests += summary['total']
                total_passed += summary['passed']
                total_failed += summary['failed']
                total_skipped += summary['skipped']
                total_errors += summary['errors']
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        return {
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration': total_duration,
            'total_tests': total_tests,
            'passed': total_passed,
            'failed': total_failed,
            'skipped': total_skipped,
            'errors': total_errors,
            'success_rate': success_rate,
            'suites_run': len(suite_results),
            'status': 'passed' if total_failed == 0 and total_errors == 0 else 'failed'
        }


# Test decorators and utilities
def a2a_test(
    test_id: str,
    name: str,
    description: str,
    agent_type: str,
    timeout: int = 30,
    severity: TestSeverity = TestSeverity.MEDIUM,
    dependencies: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    expected_duration: Optional[float] = None
):
    """Decorator for A2A test cases."""
    def decorator(func):
        test_case = TestCase(
            test_id=test_id,
            name=name,
            description=description,
            agent_type=agent_type,
            test_function=func,
            timeout=timeout,
            severity=severity,
            dependencies=dependencies or [],
            tags=tags or [],
            expected_duration=expected_duration
        )
        
        # Store test case metadata
        func._a2a_test_case = test_case
        return func
    
    return decorator


async def assert_response_ok(response: httpx.Response, message: str = ""):
    """Assert HTTP response is OK."""
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. {message}"


async def assert_response_contains(
    response: httpx.Response,
    expected_content: str,
    message: str = ""
):
    """Assert response contains expected content."""
    content = response.text
    assert expected_content in content, f"Expected '{expected_content}' in response. {message}"


async def assert_json_field(
    response: httpx.Response,
    field_path: str,
    expected_value: Any,
    message: str = ""
):
    """Assert JSON response field has expected value."""
    try:
        json_data = response.json()
        field_parts = field_path.split('.')
        current = json_data
        
        for part in field_parts:
            current = current[part]
        
        assert current == expected_value, f"Expected {field_path}={expected_value}, got {current}. {message}"
        
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        assert False, f"Failed to access {field_path}: {e}. {message}"


def generate_test_data(data_type: str, count: int = 10) -> List[Dict[str, Any]]:
    """Generate test data of specified type."""
    generators = {
        'questions': lambda: [
            {
                'id': f'q_{i}',
                'text': f'Test question {i}?',
                'expected_answer': f'Answer {i}',
                'difficulty': ['easy', 'medium', 'hard'][i % 3]
            }
            for i in range(count)
        ],
        'calculations': lambda: [
            {
                'id': f'calc_{i}',
                'operation': ['sum', 'multiply', 'average'][i % 3],
                'inputs': [i, i+1, i+2],
                'expected_result': i * 3 + 3  # Simple calculation
            }
            for i in range(count)
        ],
        'data_products': lambda: [
            {
                'id': f'dp_{i}',
                'name': f'Test Dataset {i}',
                'data': {'values': list(range(i, i+5))},
                'schema': {'type': 'array', 'items': {'type': 'number'}}
            }
            for i in range(count)
        ]
    }
    
    generator = generators.get(data_type)
    if not generator:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return generator()