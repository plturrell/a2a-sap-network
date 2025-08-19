"""
Comprehensive MCP Integration Tests for A2A Agents
Tests cross-agent MCP functionality and end-to-end workflows
"""

import asyncio
import json
import logging
import pytest
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from unittest.mock import AsyncMock, Mock, patch

from testFramework import (
    A2ATestEnvironment, A2ATestRunner, TestSuite, TestCase, 
    TestSeverity, a2a_test, assert_response_ok, assert_json_field,
    generate_test_data
)

logger = logging.getLogger(__name__)


class MCPTestEnvironment(A2ATestEnvironment):
    """Extended test environment with MCP-specific capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mcp_agent_configs = {
            'calculation': {'port': 8010, 'mcp_enabled': True},
            'data_manager': {'port': 8011, 'mcp_enabled': True},
            'catalog_manager': {'port': 8012, 'mcp_enabled': True},
            'reasoning': {'port': 8013, 'mcp_enabled': True},
            'sql_agent': {'port': 8014, 'mcp_enabled': True}
        }
    
    async def _initialize_test_agents(self):
        """Initialize MCP-enabled test agents."""
        await super()._initialize_test_agents()
        
        # Add MCP-specific agents
        for agent_name, config in self.mcp_agent_configs.items():
            agent_url = f"{self.base_url.replace('8000', str(config['port']))}"
            self.test_agents[agent_name] = {
                'url': agent_url,
                'agent_id': f'test_{agent_name}_agent',
                'config': config,
                'mcp_enabled': True
            }
    
    async def call_mcp_tool(
        self,
        agent_type: str,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call MCP tool on agent."""
        if agent_type not in self.test_agents:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        if not self.test_agents[agent_type].get('mcp_enabled'):
            raise ValueError(f"Agent {agent_type} is not MCP-enabled")
        
        endpoint = f"/mcp/tools/{tool_name}"
        response = await self.call_agent_api(agent_type, endpoint, "POST", parameters)
        
        if response.status_code != 200:
            raise Exception(f"MCP tool call failed: {response.status_code} - {response.text}")
        
        return response.json()
    
    async def get_mcp_resource(
        self,
        agent_type: str,
        resource_uri: str
    ) -> Dict[str, Any]:
        """Get MCP resource from agent."""
        if agent_type not in self.test_agents:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        if not self.test_agents[agent_type].get('mcp_enabled'):
            raise ValueError(f"Agent {agent_type} is not MCP-enabled")
        
        endpoint = f"/mcp/resources/{resource_uri.replace('://', '_')}"
        response = await self.call_agent_api(agent_type, endpoint, "GET")
        
        if response.status_code != 200:
            raise Exception(f"MCP resource call failed: {response.status_code} - {response.text}")
        
        return response.json()
    
    async def call_mcp_prompt(
        self,
        agent_type: str,
        prompt_name: str,
        parameters: Dict[str, Any]
    ) -> str:
        """Call MCP prompt on agent."""
        if agent_type not in self.test_agents:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        if not self.test_agents[agent_type].get('mcp_enabled'):
            raise ValueError(f"Agent {agent_type} is not MCP-enabled")
        
        endpoint = f"/mcp/prompts/{prompt_name}"
        response = await self.call_agent_api(agent_type, endpoint, "POST", parameters)
        
        if response.status_code != 200:
            raise Exception(f"MCP prompt call failed: {response.status_code} - {response.text}")
        
        result = response.json()
        return result.get('response', '')


# MCP Individual Agent Tests
@a2a_test(
    test_id="mcp_001",
    name="Calculation Agent MCP Tools",
    description="Test all MCP tools for Calculation Agent",
    agent_type="calculation",
    severity=TestSeverity.HIGH,
    expected_duration=5.0
)
async def test_calculation_agent_mcp_tools(env: MCPTestEnvironment):
    """Test Calculation Agent MCP tools."""
    
    # Test calculate tool
    result = await env.call_mcp_tool('calculation', 'calculate', {
        'expression': '2 + 2 * 3',
        'explain': True
    })
    
    assert result['success'] == True
    assert 'result' in result
    assert 'explanation' in result
    
    # Test solve_equation tool
    result = await env.call_mcp_tool('calculation', 'solve_equation', {
        'equation': 'x^2 - 4 = 0',
        'variable': 'x'
    })
    
    assert result['success'] == True
    assert 'solutions' in result
    
    # Test financial_calculation tool
    result = await env.call_mcp_tool('calculation', 'financial_calculation', {
        'calculation_type': 'compound_interest',
        'principal': 1000,
        'rate': 0.05,
        'time': 2
    })
    
    assert result['success'] == True
    assert 'result' in result
    
    logger.info("✅ Calculation Agent MCP tools test passed")


@a2a_test(
    test_id="mcp_002", 
    name="Data Manager MCP Tools",
    description="Test all MCP tools for Data Manager",
    agent_type="data_manager",
    severity=TestSeverity.HIGH,
    expected_duration=7.0
)
async def test_data_manager_mcp_tools(env: MCPTestEnvironment):
    """Test Data Manager MCP tools."""
    
    # Test store_data tool
    test_data = {
        "test_key": "test_value",
        "numbers": [1, 2, 3, 4, 5],
        "metadata": {"created_at": datetime.utcnow().isoformat()}
    }
    
    result = await env.call_mcp_tool('data_manager', 'store_data', {
        'data': test_data,
        'data_type': 'test_data',
        'service_level': 'silver'
    })
    
    assert result['success'] == True
    assert 'data_id' in result
    data_id = result['data_id']
    
    # Test retrieve_data tool
    result = await env.call_mcp_tool('data_manager', 'retrieve_data', {
        'data_id': data_id,
        'verify_integrity': True
    })
    
    assert result['success'] == True
    assert result['data'] == test_data
    
    # Test query_data tool
    result = await env.call_mcp_tool('data_manager', 'query_data', {
        'data_type': 'test_data',
        'limit': 10
    })
    
    assert result['success'] == True
    assert 'results' in result
    
    # Test manage_storage tool
    result = await env.call_mcp_tool('data_manager', 'manage_storage', {
        'operation': 'stats'
    })
    
    assert result['success'] == True
    assert 'storage_statistics' in result
    
    logger.info("✅ Data Manager MCP tools test passed")


@a2a_test(
    test_id="mcp_003",
    name="SQL Agent MCP Tools", 
    description="Test all MCP tools for SQL Agent",
    agent_type="sql_agent",
    severity=TestSeverity.HIGH,
    expected_duration=6.0
)
async def test_sql_agent_mcp_tools(env: MCPTestEnvironment):
    """Test SQL Agent MCP tools."""
    
    # Test natural_to_sql tool
    result = await env.call_mcp_tool('sql_agent', 'natural_to_sql', {
        'question': 'Show me all customers with orders over $1000',
        'database_type': 'hana',
        'include_explanation': True
    })
    
    assert result['success'] == True
    assert 'sql_query' in result
    assert 'explanation' in result
    
    # Test sql_optimization tool
    sql_query = "SELECT * FROM customers WHERE amount > 1000"
    result = await env.call_mcp_tool('sql_agent', 'sql_optimization', {
        'sql_query': sql_query,
        'target_database': 'hana',
        'include_analysis': True
    })
    
    assert result['success'] == True
    assert 'optimized_query' in result
    
    # Test sql_validation tool
    result = await env.call_mcp_tool('sql_agent', 'sql_validation', {
        'sql_query': sql_query,
        'validation_level': 'comprehensive'
    })
    
    assert result['success'] == True
    assert 'is_valid' in result
    
    logger.info("✅ SQL Agent MCP tools test passed")


@a2a_test(
    test_id="mcp_004",
    name="Reasoning Agent MCP Tools",
    description="Test all MCP tools for Reasoning Agent", 
    agent_type="reasoning",
    severity=TestSeverity.HIGH,
    expected_duration=10.0
)
async def test_reasoning_agent_mcp_tools(env: MCPTestEnvironment):
    """Test Reasoning Agent MCP tools."""
    
    # Test advanced_reasoning tool
    result = await env.call_mcp_tool('reasoning', 'advanced_reasoning', {
        'question': 'What are the pros and cons of renewable energy?',
        'reasoning_architecture': 'hierarchical',
        'enable_debate': True
    })
    
    assert result['success'] == True
    assert 'answer' in result
    assert 'confidence' in result
    
    # Test hypothesis_generation tool
    result = await env.call_mcp_tool('reasoning', 'hypothesis_generation', {
        'problem': 'Why do some companies succeed while others fail?',
        'domain': 'business',
        'max_hypotheses': 3
    })
    
    assert result['success'] == True
    assert 'hypotheses' in result
    assert len(result['hypotheses']) <= 3
    
    # Test debate_orchestration tool
    result = await env.call_mcp_tool('reasoning', 'debate_orchestration', {
        'topic': 'Should AI be regulated?',
        'perspectives': ['tech_industry', 'government', 'civil_society'],
        'max_rounds': 2
    })
    
    assert result['success'] == True
    assert 'debate_results' in result
    
    logger.info("✅ Reasoning Agent MCP tools test passed")


# Cross-Agent Integration Tests
@a2a_test(
    test_id="mcp_101", 
    name="Data Pipeline MCP Integration",
    description="Test data flow between Data Manager and SQL Agent via MCP",
    agent_type="integration",
    severity=TestSeverity.CRITICAL,
    dependencies=["mcp_002", "mcp_003"],
    expected_duration=15.0
)
async def test_data_pipeline_mcp_integration(env: MCPTestEnvironment):
    """Test integrated data pipeline using MCP."""
    
    # Step 1: Store data using Data Manager
    sample_data = [
        {"id": 1, "name": "Alice", "amount": 1500, "category": "premium"},
        {"id": 2, "name": "Bob", "amount": 800, "category": "standard"},
        {"id": 3, "name": "Charlie", "amount": 2200, "category": "premium"}
    ]
    
    store_result = await env.call_mcp_tool('data_manager', 'store_data', {
        'data': sample_data,
        'data_type': 'customer_data',
        'service_level': 'gold'
    })
    
    assert store_result['success'] == True
    data_id = store_result['data_id']
    
    # Step 2: Generate SQL query for analysis
    sql_result = await env.call_mcp_tool('sql_agent', 'natural_to_sql', {
        'question': 'Find all premium customers with amount greater than 1000',
        'schema_info': {
            'tables': [
                {
                    'name': 'customers',
                    'columns': ['id', 'name', 'amount', 'category']
                }
            ]
        }
    })
    
    assert sql_result['success'] == True
    assert 'sql_query' in sql_result
    
    # Step 3: Optimize the generated SQL
    optimization_result = await env.call_mcp_tool('sql_agent', 'sql_optimization', {
        'sql_query': sql_result['sql_query'],
        'optimization_goals': ['performance']
    })
    
    assert optimization_result['success'] == True
    
    # Step 4: Retrieve and verify data
    retrieve_result = await env.call_mcp_tool('data_manager', 'retrieve_data', {
        'data_id': data_id,
        'include_metadata': True
    })
    
    assert retrieve_result['success'] == True
    assert retrieve_result['data'] == sample_data
    
    logger.info("✅ Data pipeline MCP integration test passed")


@a2a_test(
    test_id="mcp_102",
    name="Reasoning and Calculation MCP Integration", 
    description="Test reasoning workflow with calculation validation",
    agent_type="integration",
    severity=TestSeverity.HIGH,
    dependencies=["mcp_001", "mcp_004"],
    expected_duration=12.0
)
async def test_reasoning_calculation_mcp_integration(env: MCPTestEnvironment):
    """Test integrated reasoning and calculation workflow."""
    
    # Step 1: Generate hypotheses about a mathematical problem
    reasoning_result = await env.call_mcp_tool('reasoning', 'hypothesis_generation', {
        'problem': 'What factors affect compound interest calculations?',
        'domain': 'financial',
        'max_hypotheses': 3
    })
    
    assert reasoning_result['success'] == True
    hypotheses = reasoning_result['hypotheses']
    
    # Step 2: Test financial calculations based on reasoning
    calc_result = await env.call_mcp_tool('calculation', 'financial_calculation', {
        'calculation_type': 'compound_interest',
        'principal': 10000,
        'rate': 0.06,
        'time': 5,
        'compound_frequency': 12
    })
    
    assert calc_result['success'] == True
    compound_result = calc_result['result']
    
    # Step 3: Use reasoning to analyze the calculation result
    analysis_result = await env.call_mcp_tool('reasoning', 'advanced_reasoning', {
        'question': f'Given a compound interest result of {compound_result}, what does this tell us about the investment?',
        'context': {
            'calculation_details': calc_result,
            'hypotheses': hypotheses
        },
        'reasoning_architecture': 'hub_and_spoke'
    })
    
    assert analysis_result['success'] == True
    assert 'answer' in analysis_result
    
    # Step 4: Validate calculations with different parameters
    validation_result = await env.call_mcp_tool('calculation', 'calculate', {
        'expression': f'10000 * (1 + 0.06/12)^(12*5)',
        'explain': True
    })
    
    assert validation_result['success'] == True
    
    logger.info("✅ Reasoning and calculation MCP integration test passed")


@a2a_test(
    test_id="mcp_103",
    name="Full Workflow MCP Integration",
    description="Test complete workflow across all MCP agents",
    agent_type="integration", 
    severity=TestSeverity.CRITICAL,
    dependencies=["mcp_001", "mcp_002", "mcp_003", "mcp_004"],
    expected_duration=20.0
)
async def test_full_workflow_mcp_integration(env: MCPTestEnvironment):
    """Test complete workflow across all MCP-enabled agents."""
    
    # Step 1: Store business data
    business_data = {
        "quarterly_revenue": [100000, 120000, 135000, 150000],
        "expenses": [80000, 85000, 90000, 95000],
        "growth_targets": {"q1": 0.1, "q2": 0.15, "q3": 0.12, "q4": 0.18}
    }
    
    store_result = await env.call_mcp_tool('data_manager', 'store_data', {
        'data': business_data,
        'data_type': 'quarterly_financials',
        'service_level': 'gold'
    })
    
    assert store_result['success'] == True
    
    # Step 2: Generate reasoning about business performance
    reasoning_result = await env.call_mcp_tool('reasoning', 'advanced_reasoning', {
        'question': 'Based on quarterly revenue growth, what strategic recommendations should we make?',
        'context': business_data,
        'reasoning_architecture': 'hierarchical',
        'enable_debate': True
    })
    
    assert reasoning_result['success'] == True
    strategic_insights = reasoning_result['answer']
    
    # Step 3: Calculate financial metrics
    growth_calc = await env.call_mcp_tool('calculation', 'calculate', {
        'expression': '(150000 - 100000) / 100000 * 100',
        'explain': True
    })
    
    assert growth_calc['success'] == True
    
    # Step 4: Generate SQL for analysis
    sql_result = await env.call_mcp_tool('sql_agent', 'natural_to_sql', {
        'question': 'Calculate year-over-year growth rate and identify quarters exceeding target',
        'schema_info': {
            'tables': [
                {
                    'name': 'quarterly_financials',
                    'columns': ['quarter', 'revenue', 'expenses', 'growth_target']
                }
            ]
        }
    })
    
    assert sql_result['success'] == True
    
    # Step 5: Store analysis results
    analysis_results = {
        "strategic_insights": strategic_insights,
        "growth_calculation": growth_calc['result'],
        "sql_analysis": sql_result['sql_query'],
        "workflow_completed_at": datetime.utcnow().isoformat()
    }
    
    final_store = await env.call_mcp_tool('data_manager', 'store_data', {
        'data': analysis_results,
        'data_type': 'business_analysis',
        'service_level': 'gold'
    })
    
    assert final_store['success'] == True
    
    logger.info("✅ Full workflow MCP integration test passed")


# MCP Resource Tests
@a2a_test(
    test_id="mcp_201",
    name="MCP Resource Accessibility",
    description="Test all MCP resources are accessible",
    agent_type="integration",
    severity=TestSeverity.MEDIUM,
    expected_duration=8.0
)
async def test_mcp_resource_accessibility(env: MCPTestEnvironment):
    """Test that all MCP resources are accessible."""
    
    # Test Calculation Agent resources
    calc_status = await env.get_mcp_resource('calculation', 'calculation://status')
    assert 'calculation_status' in calc_status
    
    calc_capabilities = await env.get_mcp_resource('calculation', 'calculation://capabilities')
    assert 'calculation_capabilities' in calc_capabilities
    
    # Test Data Manager resources
    storage_status = await env.get_mcp_resource('data_manager', 'storage://status')
    assert 'storage_status' in storage_status
    
    storage_backends = await env.get_mcp_resource('data_manager', 'storage://backends')
    assert 'storage_backends' in storage_backends
    
    # Test SQL Agent resources
    sql_status = await env.get_mcp_resource('sql_agent', 'sql://status')
    assert 'sql_status' in sql_status
    
    sql_functions = await env.get_mcp_resource('sql_agent', 'sql://supported-functions')
    assert 'supported_functions' in sql_functions
    
    logger.info("✅ MCP resource accessibility test passed")


# MCP Prompt Tests
@a2a_test(
    test_id="mcp_301",
    name="MCP Prompt Functionality",
    description="Test MCP prompts provide helpful responses",
    agent_type="integration",
    severity=TestSeverity.MEDIUM,
    expected_duration=10.0
)
async def test_mcp_prompt_functionality(env: MCPTestEnvironment):
    """Test MCP prompts provide helpful responses."""
    
    # Test Calculation Agent prompts
    calc_response = await env.call_mcp_prompt('calculation', 'calculation_assistant', {
        'user_query': 'How do I calculate compound interest?'
    })
    
    assert len(calc_response) > 50  # Should be a meaningful response
    assert 'compound interest' in calc_response.lower()
    
    # Test Reasoning Agent prompts  
    reasoning_response = await env.call_mcp_prompt('reasoning', 'reasoning_assistant', {
        'user_query': 'Help me analyze a complex business problem'
    })
    
    assert len(reasoning_response) > 50
    assert any(word in reasoning_response.lower() for word in ['analyze', 'reasoning', 'problem'])
    
    # Test SQL Agent prompts
    sql_response = await env.call_mcp_prompt('sql_agent', 'sql_assistant', {
        'user_query': 'Convert this to SQL: find top customers'
    })
    
    assert len(sql_response) > 50
    assert 'sql' in sql_response.lower()
    
    logger.info("✅ MCP prompt functionality test passed")


# Error Handling Tests
@a2a_test(
    test_id="mcp_401",
    name="MCP Error Handling",
    description="Test MCP error handling and recovery",
    agent_type="integration",
    severity=TestSeverity.HIGH,
    expected_duration=5.0
)
async def test_mcp_error_handling(env: MCPTestEnvironment):
    """Test MCP error handling."""
    
    # Test invalid tool parameters
    calc_result = await env.call_mcp_tool('calculation', 'calculate', {
        'expression': 'invalid_expression',
        'explain': True
    })
    
    # Should handle error gracefully
    assert calc_result['success'] == False
    assert 'error' in calc_result
    
    # Test invalid data for storage
    storage_result = await env.call_mcp_tool('data_manager', 'store_data', {
        'data': None,  # Invalid data
        'data_type': 'test'
    })
    
    assert storage_result['success'] == False
    assert 'error' in storage_result
    
    logger.info("✅ MCP error handling test passed")


# Performance Tests
@a2a_test(
    test_id="mcp_501",
    name="MCP Performance Benchmarks",
    description="Test MCP performance under load",
    agent_type="integration",
    severity=TestSeverity.MEDIUM,
    expected_duration=15.0
)
async def test_mcp_performance_benchmarks(env: MCPTestEnvironment):
    """Test MCP performance under concurrent load."""
    
    # Concurrent calculation requests
    calc_tasks = []
    for i in range(10):
        task = env.call_mcp_tool('calculation', 'calculate', {
            'expression': f'2^{i} + {i} * 5',
            'explain': False
        })
        calc_tasks.append(task)
    
    start_time = datetime.utcnow()
    calc_results = await asyncio.gather(*calc_tasks, return_exceptions=True)
    calc_duration = (datetime.utcnow() - start_time).total_seconds()
    
    # Verify all succeeded
    successful_calcs = sum(1 for r in calc_results if isinstance(r, dict) and r.get('success'))
    assert successful_calcs >= 8  # Allow some failures under load
    
    # Should complete within reasonable time
    assert calc_duration < 10.0
    
    # Concurrent data operations
    data_tasks = []
    for i in range(5):
        task = env.call_mcp_tool('data_manager', 'store_data', {
            'data': {'test_id': i, 'value': f'test_value_{i}'},
            'data_type': f'performance_test_{i}'
        })
        data_tasks.append(task)
    
    start_time = datetime.utcnow()
    data_results = await asyncio.gather(*data_tasks, return_exceptions=True)
    data_duration = (datetime.utcnow() - start_time).total_seconds()
    
    successful_stores = sum(1 for r in data_results if isinstance(r, dict) and r.get('success'))
    assert successful_stores >= 4
    
    assert data_duration < 8.0
    
    logger.info(f"✅ MCP performance test passed - Calc: {calc_duration:.2f}s, Data: {data_duration:.2f}s")


# Create Test Suites
def create_mcp_test_suites() -> List[TestSuite]:
    """Create MCP test suites."""
    
    # Individual agent tests
    agent_suite = TestSuite(
        suite_id="mcp_agents",
        name="MCP Individual Agent Tests", 
        description="Test MCP functionality for each agent individually"
    )
    
    # Integration tests
    integration_suite = TestSuite(
        suite_id="mcp_integration",
        name="MCP Cross-Agent Integration Tests",
        description="Test MCP functionality across multiple agents"
    )
    
    # Resource and prompt tests
    interface_suite = TestSuite(
        suite_id="mcp_interfaces",
        name="MCP Interface Tests",
        description="Test MCP resources, prompts, and interfaces"
    )
    
    # Error and performance tests
    robustness_suite = TestSuite(
        suite_id="mcp_robustness", 
        name="MCP Robustness Tests",
        description="Test MCP error handling and performance"
    )
    
    return [agent_suite, integration_suite, interface_suite, robustness_suite]


async def run_mcp_integration_tests():
    """Run comprehensive MCP integration tests."""
    
    # Setup test environment
    test_env = MCPTestEnvironment(
        base_url="http://localhost:8000",
        blockchain_test_mode=True,
        mock_external_services=True
    )
    
    # Create test runner
    runner = A2ATestRunner(test_env)
    
    # Register test suites
    test_suites = create_mcp_test_suites()
    for suite in test_suites:
        runner.register_test_suite(suite)
    
    # Register individual test cases to appropriate suites
    test_functions = [
        (test_calculation_agent_mcp_tools, "mcp_agents"),
        (test_data_manager_mcp_tools, "mcp_agents"),
        (test_sql_agent_mcp_tools, "mcp_agents"),
        (test_reasoning_agent_mcp_tools, "mcp_agents"),
        (test_data_pipeline_mcp_integration, "mcp_integration"),
        (test_reasoning_calculation_mcp_integration, "mcp_integration"),
        (test_full_workflow_mcp_integration, "mcp_integration"),
        (test_mcp_resource_accessibility, "mcp_interfaces"),
        (test_mcp_prompt_functionality, "mcp_interfaces"),
        (test_mcp_error_handling, "mcp_robustness"),
        (test_mcp_performance_benchmarks, "mcp_robustness")
    ]
    
    for test_func, suite_id in test_functions:
        if hasattr(test_func, '_a2a_test_case'):
            runner.register_test_case(suite_id, test_func._a2a_test_case)
    
    # Run all tests
    results = await runner.run_all_tests()
    
    # Print summary
    summary = results['summary']
    print(f"\n{'='*60}")
    print(f"MCP INTEGRATION TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']} ✅")
    print(f"Failed: {summary['failed']} ❌")
    print(f"Skipped: {summary['skipped']} ⏭️")
    print(f"Errors: {summary['errors']} 💥")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Duration: {summary['duration']:.2f}s")
    print(f"Status: {summary['status'].upper()}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    # Run tests when executed directly
    asyncio.run(run_mcp_integration_tests())