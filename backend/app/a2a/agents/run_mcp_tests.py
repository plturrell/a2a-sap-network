#!/usr/bin/env python3
"""
Run real MCP integration tests to verify functionality
"""

import asyncio
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import agents
from agent0DataProduct.active.advancedMcpDataProductAgentFixed import AdvancedMCPDataProductAgentFixed
from agent1Standardization.active.advancedMcpStandardizationAgent import AdvancedMCPStandardizationAgent
from agent3VectorProcessing.active.advancedMcpVectorProcessingAgent import AdvancedMCPVectorProcessingAgent
from agent4CalcValidation.active.advancedMcpCalculationValidationAgent import AdvancedMCPCalculationValidationAgent


async def test_mcp_tools_are_real():
    """Test that MCP tools are real instances, not mocks"""
    logger.info("=== Testing MCP Tool Reality ===")
    
    agent = AdvancedMCPDataProductAgentFixed("http://localhost:8000")
    
    # Check tool types
    from ..common.mcpPerformanceTools import MCPPerformanceTools
    from ..common.mcpValidationTools import MCPValidationTools
    from ..common.mcpQualityAssessmentTools import MCPQualityAssessmentTools
    
    assert isinstance(agent.performance_tools, MCPPerformanceTools), "Performance tools are mocked!"
    assert isinstance(agent.validation_tools, MCPValidationTools), "Validation tools are mocked!"
    assert isinstance(agent.quality_tools, MCPQualityAssessmentTools), "Quality tools are mocked!"
    
    logger.info("✓ All MCP tools are real instances")
    return True


async def test_real_performance_measurement():
    """Test real performance measurement"""
    logger.info("\n=== Testing Real Performance Measurement ===")
    
    agent = AdvancedMCPDataProductAgentFixed("http://localhost:8000")
    
    # Measure real operation
    start_time = datetime.now().timestamp()
    await asyncio.sleep(0.2)  # Simulate work
    end_time = datetime.now().timestamp()
    
    metrics = await agent.performance_tools.measure_performance_metrics(
        operation_id="test_op_001",
        start_time=start_time,
        end_time=end_time,
        operation_count=5,
        custom_metrics={
            "data_size_mb": 10.5,
            "quality_score": 0.92
        }
    )
    
    # Verify results
    assert metrics["duration_ms"] >= 200, f"Duration too short: {metrics['duration_ms']}"
    assert metrics["throughput"] > 0, "Throughput should be positive"
    assert metrics["error_rate"] == 0, "No errors should be recorded"
    assert metrics["custom_metrics"]["quality_score"] == 0.92
    
    logger.info(f"✓ Performance measurement working: {metrics['duration_ms']:.2f}ms, {metrics['throughput']:.2f} ops/sec")
    return True


async def test_real_validation():
    """Test real validation functionality"""
    logger.info("\n=== Testing Real Validation ===")
    
    agent = AdvancedMCPStandardizationAgent("http://localhost:8000")
    
    # Test schema validation
    test_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0, "maximum": 120},
            "email": {"type": "string", "pattern": "^[^@]+@[^@]+\\.[^@]+$"}
        },
        "required": ["name", "age"]
    }
    
    # Valid data
    valid_data = {
        "name": "Alice Johnson",
        "age": 28,
        "email": "alice@example.com"
    }
    
    result = await agent.validation_tools.validate_schema_compliance(
        data=valid_data,
        schema=test_schema,
        validation_level="strict"
    )
    
    assert result["is_valid"] is True
    logger.info("✓ Valid data passed validation")
    
    # Invalid data
    invalid_data = {
        "name": "Bob",
        "age": 150,  # Too old
        "email": "not-an-email"
    }
    
    result = await agent.validation_tools.validate_schema_compliance(
        data=invalid_data,
        schema=test_schema,
        validation_level="strict"
    )
    
    assert result["is_valid"] is False
    assert len(result["validation_errors"]) >= 2  # Age and email errors
    logger.info(f"✓ Invalid data caught {len(result['validation_errors'])} errors")
    
    return True


async def test_real_quality_assessment():
    """Test real quality assessment"""
    logger.info("\n=== Testing Real Quality Assessment ===")
    
    agent = AdvancedMCPDataProductAgentFixed("http://localhost:8000")
    
    # Assess data product quality
    result = await agent.quality_tools.assess_data_product_quality(
        product_definition={
            "name": "customer_analytics",
            "type": "structured",
            "version": "1.0.0",
            "schema": {
                "customer_id": {"type": "integer"},
                "score": {"type": "float"}
            }
        },
        data_source={
            "type": "memory",
            "record_count": 1000,
            "sample_data": [
                {"customer_id": 1, "score": 0.85},
                {"customer_id": 2, "score": 0.92}
            ]
        },
        quality_requirements={
            "completeness": 0.95,
            "accuracy": 0.90,
            "consistency": 0.85
        },
        assessment_criteria=["completeness", "accuracy", "consistency", "timeliness"]
    )
    
    assert "overall_score" in result
    assert 0 <= result["overall_score"] <= 1
    logger.info(f"✓ Quality assessment complete: score={result['overall_score']:.2f}")
    
    return True


async def test_cross_agent_workflow():
    """Test complete workflow across multiple agents"""
    logger.info("\n=== Testing Cross-Agent Workflow ===")
    
    # Initialize agents
    data_agent = AdvancedMCPDataProductAgentFixed("http://localhost:8000")
    std_agent = AdvancedMCPStandardizationAgent("http://localhost:8000")
    vec_agent = AdvancedMCPVectorProcessingAgent("http://localhost:8000")
    calc_agent = AdvancedMCPCalculationValidationAgent("http://localhost:8000")
    
    # Step 1: Register data product
    logger.info("Step 1: Registering data product...")
    
    product_def = {
        "name": "test_metrics",
        "type": "structured",
        "version": "1.0.0",
        "description": "Test metrics data",
        "schema": {
            "fields": {
                "metric_id": {"type": "integer"},
                "value": {"type": "string"},  # Will need standardization
                "timestamp": {"type": "string"}
            }
        }
    }
    
    data_source = {
        "type": "memory",
        "data": [
            {"metric_id": "1", "value": "85.5", "timestamp": "2024-01-18T10:00:00"},
            {"metric_id": "2", "value": "92.3", "timestamp": "2024-01-18T10:01:00"}
        ]
    }
    
    reg_result = await data_agent.intelligent_data_product_registration(
        product_definition=product_def,
        data_source=data_source,
        auto_standardization=True,
        cross_agent_validation=False  # Avoid circular deps
    )
    
    assert reg_result["status"] == "success"
    product_id = reg_result["product_id"]
    logger.info(f"✓ Product registered: {product_id}")
    
    # Step 2: Standardize data
    logger.info("Step 2: Standardizing data...")
    
    std_result = await std_agent.intelligent_data_standardization(
        data_input=data_source["data"][0],
        target_schema={
            "fields": {
                "metric_id": {"type": "integer"},
                "value": {"type": "float"},
                "timestamp": {"type": "datetime"}
            },
            "type": "structured",
            "version": "1.0.0"
        },
        learning_mode=True
    )
    
    assert std_result["status"] == "success"
    logger.info("✓ Data standardized successfully")
    
    # Step 3: Process vectors (if we had embeddings)
    logger.info("Step 3: Processing vectors...")
    
    test_vectors = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.4, 0.5, 0.6, 0.7]
    ]
    
    vec_result = await vec_agent.intelligent_vector_processing(
        vectors=test_vectors,
        operations=["normalize"],
        cross_validation=False
    )
    
    assert vec_result["status"] == "success"
    logger.info("✓ Vectors processed successfully")
    
    # Step 4: Validate calculations
    logger.info("Step 4: Validating calculations...")
    
    calc_result = await calc_agent.comprehensive_calculation_validation(
        calculation_request={
            "expression": "85.5 + 92.3",
            "parameters": {},
            "expected_type": "numeric"
        },
        expected_result=177.8,
        validation_methods=["direct"],
        cross_agent_validation=False
    )
    
    assert calc_result["status"] == "success"
    logger.info("✓ Calculation validated successfully")
    
    logger.info("\n✓ Cross-agent workflow completed successfully!")
    return True


async def test_mcp_resources():
    """Test MCP resource access"""
    logger.info("\n=== Testing MCP Resources ===")
    
    agent = AdvancedMCPDataProductAgentFixed("http://localhost:8000")
    
    # Add test product
    test_product = {
        "product_id": "test_123",
        "definition": {"name": "Test Product", "type": "structured"},
        "status": "active",
        "registration_time": datetime.now().isoformat(),
        "quality_assessment": {"overall_score": 0.88},
        "monitoring_enabled": True,
        "source_analysis": {"health_score": 0.95}
    }
    agent.data_products["test_123"] = test_product
    
    # Access registry resource
    registry = await agent.get_data_product_registry()
    
    assert "registered_products" in registry
    assert "test_123" in registry["registered_products"]
    assert registry["total_products"] == 1
    assert registry["registered_products"]["test_123"]["health_score"] == 0.95
    
    logger.info("✓ MCP resources accessible and working")
    return True


async def test_error_handling():
    """Test error handling in MCP tools"""
    logger.info("\n=== Testing Error Handling ===")
    
    agent = AdvancedMCPCalculationValidationAgent("http://localhost:8000")
    
    # Test with division by zero
    result = await agent.comprehensive_calculation_validation(
        calculation_request={
            "expression": "1/0",
            "parameters": {},
            "expected_type": "numeric"
        },
        expected_result=0,
        validation_methods=["direct"]
    )
    
    # Should handle gracefully
    assert result["status"] in ["success", "error"]
    if result["status"] == "success":
        # Check that error was detected in calculation
        direct_result = result["calculation_results"]["method_results"].get("direct", {})
        assert direct_result.get("success") is False or direct_result.get("error") is not None
    
    logger.info("✓ Error handling working correctly")
    return True


async def main():
    """Run all tests"""
    logger.info("Starting MCP Integration Tests\n")
    
    tests = [
        ("MCP Tools Reality Check", test_mcp_tools_are_real),
        ("Performance Measurement", test_real_performance_measurement),
        ("Validation Functionality", test_real_validation),
        ("Quality Assessment", test_real_quality_assessment),
        ("Cross-Agent Workflow", test_cross_agent_workflow),
        ("MCP Resources", test_mcp_resources),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
                logger.info(f"✅ {test_name}: PASSED\n")
            else:
                failed += 1
                logger.error(f"❌ {test_name}: FAILED\n")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name}: FAILED with error: {e}\n")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Summary: {passed} passed, {failed} failed")
    logger.info(f"Overall Score: {(passed/(passed+failed)*100):.1f}%")
    logger.info(f"{'='*50}")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)