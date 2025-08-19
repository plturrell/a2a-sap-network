#!/usr/bin/env python3
"""
Test script to verify MCP fixes work correctly
Run from the backend directory: python test_mcp_fixed.py
"""

import sys
import os
import asyncio
import logging
from datetime import datetime

# Add the app directory to path
app_path = os.path.join(os.path.dirname(__file__), 'app')
sys.path.insert(0, app_path)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all imports work"""
    try:
        # Test MCP tool imports
        from a2a.common.mcpPerformanceTools import MCPPerformanceTools
        from a2a.common.mcpValidationTools import MCPValidationTools
        from a2a.common.mcpQualityAssessmentTools import MCPQualityAssessmentTools
        
        # Test agent imports
        from a2a.agents.agent0DataProduct.active.advancedMcpDataProductAgent import AdvancedMCPDataProductAgent
        from a2a.agents.agent1Standardization.active.advancedMcpStandardizationAgent import AdvancedMCPStandardizationAgent
        
        logger.info("✓ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_agent_creation():
    """Test creating agents with real MCP tools"""
    try:
        from a2a.agents.agent0DataProduct.active.advancedMcpDataProductAgent import AdvancedMCPDataProductAgent
        from a2a.common.mcpPerformanceTools import MCPPerformanceTools
        from a2a.common.mcpValidationTools import MCPValidationTools
        from a2a.common.mcpQualityAssessmentTools import MCPQualityAssessmentTools
        
        # Create agent
        agent = AdvancedMCPDataProductAgent("http://localhost:8000")
        
        # Verify MCP tools are real instances
        assert isinstance(agent.performance_tools, MCPPerformanceTools), "Performance tools are mocked!"
        assert isinstance(agent.validation_tools, MCPValidationTools), "Validation tools are mocked!"
        assert isinstance(agent.quality_tools, MCPQualityAssessmentTools), "Quality tools are mocked!"
        
        logger.info("✓ Agent creation with real MCP tools successful")
        return True
    except Exception as e:
        logger.error(f"❌ Agent creation failed: {e}")
        return False

async def test_real_performance_measurement():
    """Test real performance measurement"""
    try:
        from a2a.agents.agent0DataProduct.active.advancedMcpDataProductAgent import AdvancedMCPDataProductAgent
        
        agent = AdvancedMCPDataProductAgent("http://localhost:8000")
        
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
    except Exception as e:
        logger.error(f"❌ Performance measurement failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_real_validation():
    """Test real validation functionality"""
    try:
        from a2a.agents.agent1Standardization.active.advancedMcpStandardizationAgent import AdvancedMCPStandardizationAgent
        
        agent = AdvancedMCPStandardizationAgent("http://localhost:8000")
        
        # Test schema validation
        test_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0, "maximum": 120},
                "email": {"type": "string"}
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
        assert len(result["validation_errors"]) >= 1  # At least age error
        logger.info(f"✓ Invalid data caught {len(result['validation_errors'])} errors")
        
        return True
    except Exception as e:
        logger.error(f"❌ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_data_product_registration():
    """Test complete data product registration workflow"""
    try:
        from a2a.agents.agent0DataProduct.active.advancedMcpDataProductAgent import AdvancedMCPDataProductAgent
        
        agent = AdvancedMCPDataProductAgent("http://localhost:8000")
        
        product_def = {
            "name": "test_customer_data",
            "type": "structured",
            "description": "Test customer data",
            "version": "1.0.0",
            "schema": {
                "fields": {
                    "customer_id": {"type": "integer"},
                    "name": {"type": "string"},
                    "email": {"type": "string"}
                }
            }
        }
        
        data_source = {
            "type": "memory",
            "data": [
                {"customer_id": "1", "name": "Alice", "email": "alice@test.com"},
                {"customer_id": "2", "name": "Bob", "email": "bob@test.com"}
            ]
        }
        
        # Register with real MCP tools
        result = await agent.intelligent_data_product_registration(
            product_definition=product_def,
            data_source=data_source,
            auto_standardization=False,  # Skip to avoid cross-agent calls for now
            cross_agent_validation=False
        )
        
        assert result["status"] == "success"
        assert "product_id" in result
        assert "performance_metrics" in result
        assert result["source_analysis"]["analysis_available"] is True
        
        logger.info(f"✓ Data product registration successful: {result['product_id']}")
        return True
    except Exception as e:
        logger.error(f"❌ Data product registration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_data_standardization():
    """Test data standardization with real implementation"""
    try:
        from a2a.agents.agent1Standardization.active.advancedMcpStandardizationAgent import AdvancedMCPStandardizationAgent
        
        agent = AdvancedMCPStandardizationAgent("http://localhost:8000")
        
        # Test data with type mismatches
        data_input = {
            "customer_id": "123",  # String but should be int
            "name": "John Doe",
            "age": "30",  # String but should be int
            "score": "0.85"  # String but should be float
        }
        
        target_schema = {
            "fields": {
                "customer_id": {"type": "integer"},
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "score": {"type": "float"}
            },
            "type": "structured",
            "version": "1.0.0"
        }
        
        result = await agent.intelligent_data_standardization(
            data_input=data_input,
            target_schema=target_schema,
            learning_mode=True,
            cross_validation=False
        )
        
        assert result["status"] == "success"
        assert "transformation_results" in result
        assert "standardization_rules" in result
        
        # Check that transformations were applied
        standardized_data = result["transformation_results"]["standardized_data"]
        assert isinstance(standardized_data["customer_id"], int)
        assert isinstance(standardized_data["age"], int)
        assert isinstance(standardized_data["score"], float)
        
        logger.info("✓ Data standardization with real transformations successful")
        return True
    except Exception as e:
        logger.error(f"❌ Data standardization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    logger.info("Starting comprehensive MCP implementation tests\n")
    
    # Sync tests
    sync_tests = [
        ("Import Test", test_imports),
        ("Agent Creation", test_agent_creation),
    ]
    
    # Async tests
    async_tests = [
        ("Performance Measurement", test_real_performance_measurement),
        ("Validation Functionality", test_real_validation),
        ("Data Product Registration", test_data_product_registration),
        ("Data Standardization", test_data_standardization),
    ]
    
    passed = 0
    failed = 0
    
    # Run sync tests
    for test_name, test_func in sync_tests:
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name}: PASSED\n")
            else:
                failed += 1
                logger.error(f"❌ {test_name}: FAILED\n")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name}: FAILED with exception: {e}\n")
    
    # Run async tests
    for test_name, test_func in async_tests:
        try:
            if await test_func():
                passed += 1
                logger.info(f"✅ {test_name}: PASSED\n")
            else:
                failed += 1
                logger.error(f"❌ {test_name}: FAILED\n")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name}: FAILED with exception: {e}\n")
    
    logger.info(f"{'='*50}")
    logger.info(f"Test Summary: {passed} passed, {failed} failed")
    logger.info(f"Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED - MCP Implementation is working!")
    else:
        logger.error("❌ Some tests failed - check implementation")
    
    logger.info(f"{'='*50}")
    
    return failed == 0

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)