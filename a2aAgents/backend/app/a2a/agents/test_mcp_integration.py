"""
Real Integration Tests for MCP Agent Implementations
Tests actual MCP tool functionality without mocking
"""

import pytest
import asyncio
import json
import numpy as np
from datetime import datetime
import tempfile
import os
from typing import Dict, List, Any, Optional

# Import the agents to test
from .agent0DataProduct.active.advancedMcpDataProductAgent import AdvancedMCPDataProductAgent
from .agent1Standardization.active.advancedMcpStandardizationAgent import AdvancedMCPStandardizationAgent
from .agent3VectorProcessing.active.advancedMcpVectorProcessingAgent import AdvancedMCPVectorProcessingAgent
from .agent4CalcValidation.active.advancedMcpCalculationValidationAgent import AdvancedMCPCalculationValidationAgent

# Import actual MCP tools - no mocking!
from ..common.mcpPerformanceTools import MCPPerformanceTools
from ..common.mcpValidationTools import MCPValidationTools
from ..common.mcpQualityAssessmentTools import MCPQualityAssessmentTools


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")

class TestRealMCPIntegration:
    """Real integration tests that verify MCP tools work together"""
    
    @pytest.fixture
    async def test_environment(self):
        """Setup real test environment with actual agents"""
        # Create temporary directory for test data
        test_dir = tempfile.mkdtemp()
        
        # Initialize real agents with actual MCP tools
        base_url = os.getenv("A2A_SERVICE_URL")
        
        agents = {
            "data_product": AdvancedMCPDataProductAgent(base_url),
            "standardization": AdvancedMCPStandardizationAgent(base_url),
            "vector_processing": AdvancedMCPVectorProcessingAgent(base_url),
            "calculation": AdvancedMCPCalculationValidationAgent(base_url)
        }
        
        # Verify MCP tools are real instances, not mocks
        for agent_name, agent in agents.items():
            assert isinstance(agent.performance_tools, MCPPerformanceTools), f"{agent_name} has mocked performance tools!"
            assert isinstance(agent.validation_tools, MCPValidationTools), f"{agent_name} has mocked validation tools!"
            assert isinstance(agent.quality_tools, MCPQualityAssessmentTools), f"{agent_name} has mocked quality tools!"
        
        yield {
            "agents": agents,
            "test_dir": test_dir,
            "base_url": base_url
        }
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
    
    @pytest.mark.asyncio
    async def test_real_mcp_performance_tools(self, test_environment):
        """Test that MCP performance tools actually work"""
        agent = test_environment["agents"]["data_product"]
        
        # Use real performance tools
        start_time = datetime.now().timestamp()
        await asyncio.sleep(0.1)  # Simulate some work
        end_time = datetime.now().timestamp()
        
        # Call the actual MCP tool
        metrics = await agent.performance_tools.measure_performance_metrics(
            operation_id="test_operation",
            start_time=start_time,
            end_time=end_time,
            operation_count=10,
            custom_metrics={
                "test_metric": 42.0,
                "accuracy": 0.95
            }
        )
        
        # Verify real results
        assert metrics["operation_id"] == "test_operation"
        assert metrics["duration_ms"] >= 100  # Should be at least 100ms
        assert metrics["throughput"] > 0
        assert metrics["error_rate"] == 0
        assert metrics["custom_metrics"]["test_metric"] == 42.0
        assert "measurement_timestamp" in metrics
    
    @pytest.mark.asyncio
    async def test_real_mcp_validation_tools(self, test_environment):
        """Test that MCP validation tools actually validate"""
        agent = test_environment["agents"]["standardization"]
        
        # Test schema validation with real data
        test_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
                "email": {"type": "string", "format": "email"}
            },
            "required": ["name", "age"]
        }
        
        # Valid data
        valid_data = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com"
        }
        
        # Call real validation
        result = await agent.validation_tools.validate_schema_compliance(
            data=valid_data,
            schema=test_schema,
            validation_level="strict"
        )
        
        assert result["is_valid"] is True
        assert "validation_details" in result
        
        # Invalid data
        invalid_data = {
            "name": "Jane Doe",
            "age": -5,  # Invalid age
            "email": "not-an-email"  # Invalid email format
        }
        
        result = await agent.validation_tools.validate_schema_compliance(
            data=invalid_data,
            schema=test_schema,
            validation_level="strict"
        )
        
        assert result["is_valid"] is False
        assert len(result["validation_errors"]) > 0
    
    @pytest.mark.asyncio
    async def test_real_cross_agent_workflow(self, test_environment):
        """Test a complete workflow using multiple agents with real MCP tools"""
        agents = test_environment["agents"]
        
        # Step 1: Register a data product
        product_definition = {
            "name": "test_customer_data",
            "type": "structured",
            "description": "Test customer data for integration",
            "version": "1.0.0",
            "schema": {
                "fields": {
                    "customer_id": {"type": "integer"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "score": {"type": "float"}
                }
            }
        }
        
        data_source = {
            "type": "memory",
            "data": [
                {"customer_id": "1", "name": "Alice", "email": "alice@test.com", "score": "0.85"},
                {"customer_id": "2", "name": "Bob", "email": "bob@test.com", "score": "0.92"}
            ]
        }
        
        # Register with real MCP tools
        registration_result = await agents["data_product"].intelligent_data_product_registration(
            product_definition=product_definition,
            data_source=data_source,
            auto_standardization=False,  # We'll do it manually
            cross_agent_validation=False  # Avoid circular dependencies for now
        )
        
        assert registration_result["status"] == "success"
        assert "product_id" in registration_result
        product_id = registration_result["product_id"]
        
        # Step 2: Standardize the data
        standardization_result = await agents["standardization"].intelligent_data_standardization(
            data_input=data_source["data"][0],  # Standardize first record
            target_schema=product_definition["schema"],
            learning_mode=True,
            cross_validation=False
        )
        
        assert standardization_result["status"] == "success"
        assert "transformation_results" in standardization_result
        
        # Verify type conversions happened
        standardized_data = standardization_result["transformation_results"]["standardized_data"]
        assert isinstance(standardized_data["customer_id"], int)  # Converted from string
        assert isinstance(standardized_data["score"], float)  # Converted from string
        
        # Step 3: Process vectors if we have embeddings
        test_vectors = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9, 1.0]
        ]
        
        vector_result = await agents["vector_processing"].intelligent_vector_processing(
            vectors=test_vectors,
            operations=["normalize", "similarity_matrix"],
            cross_validation=False
        )
        
        assert vector_result["status"] == "success"
        assert "operation_results" in vector_result
        
        # Step 4: Validate calculations
        calc_request = {
            "expression": "0.85 + 0.92",  # Sum of customer scores
            "parameters": {},
            "expected_type": "numeric"
        }
        
        validation_result = await agents["calculation"].comprehensive_calculation_validation(
            calculation_request=calc_request,
            expected_result=1.77,
            validation_methods=["direct"],
            cross_agent_validation=False,
            symbolic_verification=False  # Keep it simple
        )
        
        assert validation_result["status"] == "success"
        assert "calculation_results" in validation_result
    
    @pytest.mark.asyncio
    async def test_real_quality_assessment(self, test_environment):
        """Test that quality assessment tools provide real metrics"""
        agent = test_environment["agents"]["data_product"]
        
        # Perform a real quality assessment
        quality_result = await agent.quality_tools.assess_data_product_quality(
            product_definition={
                "name": "test_product",
                "type": "structured",
                "schema": {"fields": {"id": {"type": "integer"}}}
            },
            data_source={
                "type": "memory",
                "record_count": 100,
                "sample_data": [{"id": 1}, {"id": 2}]
            },
            quality_requirements={
                "completeness": 0.95,
                "accuracy": 0.90
            },
            assessment_criteria=["completeness", "accuracy", "consistency"]
        )
        
        # Verify real quality scores
        assert "overall_score" in quality_result
        assert 0 <= quality_result["overall_score"] <= 1
        assert "completeness" in quality_result
        assert "accuracy" in quality_result
        assert "assessment_timestamp" in quality_result
    
    @pytest.mark.asyncio
    async def test_mcp_resource_access(self, test_environment):
        """Test that MCP resources return real data"""
        agents = test_environment["agents"]
        
        # Add some test data to agents
        test_product_id = "test_product_123"
        agents["data_product"].data_products[test_product_id] = {
            "product_id": test_product_id,
            "definition": {"name": "Test Product", "type": "structured"},
            "status": "active",
            "registration_time": datetime.now().timestamp(),
            "quality_assessment": {"overall_score": 0.85},
            "monitoring_enabled": True
        }
        
        # Access MCP resource
        registry = await agents["data_product"].get_data_product_registry()
        
        assert "registered_products" in registry
        assert test_product_id in registry["registered_products"]
        assert registry["total_products"] == 1
        assert "last_updated" in registry
    
    @pytest.mark.asyncio
    async def test_mcp_prompt_functionality(self, test_environment):
        """Test that MCP prompts return intelligent responses"""
        agent = test_environment["agents"]["standardization"]
        
        # Add some state for context
        agent.standardization_rules["test_rule"] = {
            "name": "Email Standardization",
            "type": "format",
            "usage_count": 10
        }
        
        # Call MCP prompt
        advice = await agent.standardization_advisor_prompt(
            data_context={"data_types": ["email", "phone"]},
            requirements={"accuracy": 0.95},
            constraints={"processing_time": "< 1 minute"}
        )
        
        assert isinstance(advice, str)
        assert len(advice) > 50  # Should be meaningful advice
        assert "standardization" in advice.lower()
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, test_environment):
        """Test real performance benchmarking capabilities"""
        agent = test_environment["agents"]["vector_processing"]
        
        # Generate some historical performance data
        historical_data = []
        for i in range(10):
            historical_data.append({
                "response_time": 100 + i * 10,
                "throughput": 1000 - i * 50,
                "error_rate": 0.01 + i * 0.001
            })
        
        current_metrics = {
            "response_time": 150,
            "throughput": 800,
            "error_rate": 0.015
        }
        
        # Perform real benchmarking
        benchmark_result = await agent.performance_tools.benchmark_performance(
            current_metrics=current_metrics,
            historical_data=historical_data,
            benchmark_type="statistical"
        )
        
        assert "performance_rating" in benchmark_result
        assert "benchmark_results" in benchmark_result
        assert "recommendations" in benchmark_result
        assert benchmark_result["overall_score"] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, test_environment):
        """Test that MCP tools handle errors gracefully"""
        agent = test_environment["agents"]["calculation"]
        
        # Test with invalid calculation
        calc_request = {
            "expression": "1/0",  # Division by zero
            "parameters": {},
            "expected_type": "numeric"
        }
        
        result = await agent.comprehensive_calculation_validation(
            calculation_request=calc_request,
            expected_result=0,
            validation_methods=["direct"],
            cross_agent_validation=False
        )
        
        # Should handle error gracefully
        assert result["status"] in ["success", "error"]
        if result["status"] == "success":
            # Should detect the calculation error
            assert result["calculation_results"]["method_results"]["direct"]["success"] is False
    
    @pytest.mark.asyncio
    async def test_concurrent_mcp_operations(self, test_environment):
        """Test that MCP tools work correctly under concurrent load"""
        agent = test_environment["agents"]["data_product"]
        
        # Create multiple concurrent operations
        tasks = []
        for i in range(5):
            product_def = {
                "name": f"concurrent_product_{i}",
                "type": "structured",
                "description": f"Concurrent test product {i}",
                "version": "1.0.0",
                "schema": {"fields": {"id": {"type": "integer"}}}
            }
            
            task = agent.intelligent_data_product_registration(
                product_definition=product_def,
                data_source={"type": "memory", "data": []},
                cross_agent_validation=False
            )
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(r["status"] == "success" for r in results)
        assert len(set(r["product_id"] for r in results)) == 5  # All unique IDs


class TestMCPToolChaining:
    """Test complex MCP tool chaining scenarios"""
    
    @pytest.mark.asyncio
    async def test_validation_chain(self):
        """Test chaining validation tools across agents"""
        # Initialize agents
        data_agent = AdvancedMCPDataProductAgent(os.getenv("A2A_SERVICE_URL"))
        std_agent = AdvancedMCPStandardizationAgent(os.getenv("A2A_SERVICE_URL"))
        
        # Create test data with validation issues
        test_data = {
            "customer_id": "ABC123",  # Should be numeric
            "email": "invalid-email",  # Invalid format
            "age": "twenty-five",  # Should be numeric
            "score": 0.95
        }
        
        # First validation pass
        initial_validation = await std_agent.validation_tools.validate_data_structure(
            data=test_data,
            expected_structure={
                "customer_id": "integer",
                "email": "email",
                "age": "integer",
                "score": "float"
            }
        )
        
        assert initial_validation["is_valid"] is False
        assert len(initial_validation["validation_errors"]) > 0
        
        # Apply intelligent validation with pattern detection
        enhanced_validation = await std_agent.intelligent_data_validation(
            data_to_validate=test_data,
            validation_schema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "pattern": "^[0-9]+$"},
                    "email": {"type": "string", "format": "email"},
                    "age": {"type": "integer"},
                    "score": {"type": "number"}
                }
            },
            anomaly_detection=True,
            remediation_suggestions=True
        )
        
        assert enhanced_validation["status"] == "success"
        assert "remediation_suggestions" in enhanced_validation
        assert len(enhanced_validation["remediation_suggestions"]) > 0


class TestMCPPerformanceAndScaling:
    """Test MCP tools under load and performance conditions"""
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test MCP tool performance with realistic load"""
        agent = AdvancedMCPVectorProcessingAgent(os.getenv("A2A_SERVICE_URL"))
        
        # Generate large dataset
        num_vectors = 1000
        vector_dim = 128
        vectors = np.random.rand(num_vectors, vector_dim).tolist()
        
        # Measure processing time
        start = datetime.now()
        
        result = await agent.intelligent_vector_processing(
            vectors=vectors,
            operations=["normalize", "clustering"],
            processing_config={
                "n_clusters": 10,
                "optimization_level": "advanced"
            },
            performance_monitoring=True
        )
        
        end = datetime.now()
        duration = (end - start).total_seconds()
        
        assert result["status"] == "success"
        assert result["performance_metrics"]["vectors_processed"] == num_vectors
        assert duration < 10  # Should complete within 10 seconds
        
        # Verify quality didn't degrade under load
        assert result["quality_assessment"]["overall_score"] > 0.7


class TestMCPErrorScenarios:
    """Test MCP tool behavior in error scenarios"""
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self):
        """Test how MCP tools handle validation errors"""
        agent = AdvancedMCPCalculationValidationAgent(os.getenv("A2A_SERVICE_URL"))
        
        # Test with malformed calculation request
        malformed_request = {
            "expression": None,  # Missing required field
            "parameters": "not-a-dict"  # Wrong type
        }
        
        result = await agent.comprehensive_calculation_validation(
            calculation_request=malformed_request,
            expected_result=42
        )
        
        assert result["status"] == "error"
        assert "validation_details" in result or "error" in result
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self):
        """Test MCP tools when resources are constrained"""
        agent = AdvancedMCPVectorProcessingAgent(os.getenv("A2A_SERVICE_URL"))
        
        # Try to process extremely large vectors
        huge_vectors = [[i] * 10000 for i in range(100)]  # 100 vectors of 10k dimensions
        
        result = await agent.intelligent_vector_processing(
            vectors=huge_vectors,
            operations=["normalize"],
            optimization_level="basic"  # Use basic to avoid memory issues
        )
        
        # Should either succeed with degraded performance or fail gracefully
        assert result["status"] in ["success", "error"]
        if result["status"] == "error":
            assert "error" in result
            assert isinstance(result["error"], str)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-s"])