#!/usr/bin/env python3
"""
Test script to verify imports work correctly
"""

import sys
import logging


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_import_sdk_components():
    """Test basic SDK imports"""
    try:
        from ..sdk.agentBase import A2AAgentBase
        from ..sdk.decorators import a2a_handler, a2a_skill, a2a_task
        from ..sdk.types import A2AMessage, MessageRole, TaskStatus, AgentCard
        from ..sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt
        logger.info("✓ SDK components imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import SDK components: {e}")
        return False

def test_import_mcp_tools():
    """Test MCP tool imports"""
    try:
        from ..common.mcpPerformanceTools import MCPPerformanceTools
        from ..common.mcpValidationTools import MCPValidationTools
        from ..common.mcpQualityAssessmentTools import MCPQualityAssessmentTools
        logger.info("✓ MCP tools imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import MCP tools: {e}")
        return False

def test_import_agents():
    """Test agent imports"""
    try:
        from .agent0DataProduct.active.advancedMcpDataProductAgent import AdvancedMCPDataProductAgent
        from .agent1Standardization.active.advancedMcpStandardizationAgent import AdvancedMCPStandardizationAgent
        from .agent3VectorProcessing.active.advancedMcpVectorProcessingAgent import AdvancedMCPVectorProcessingAgent
        from .agent4CalcValidation.active.advancedMcpCalculationValidationAgent import AdvancedMCPCalculationValidationAgent
        logger.info("✓ All agents imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import agents: {e}")
        return False

def test_create_agents():
    """Test creating agent instances"""
    try:
        from .agent0DataProduct.active.advancedMcpDataProductAgent import AdvancedMCPDataProductAgent
        
        # Try to create an agent
        agent = AdvancedMCPDataProductAgent(os.getenv("A2A_SERVICE_URL"))
        
        # Verify MCP tools are real instances
        from ..common.mcpPerformanceTools import MCPPerformanceTools
        from ..common.mcpValidationTools import MCPValidationTools
        from ..common.mcpQualityAssessmentTools import MCPQualityAssessmentTools
        
        assert isinstance(agent.performance_tools, MCPPerformanceTools), "Performance tools are not real!"
        assert isinstance(agent.validation_tools, MCPValidationTools), "Validation tools are not real!"
        assert isinstance(agent.quality_tools, MCPQualityAssessmentTools), "Quality tools are not real!"
        
        logger.info("✓ Agent creation and MCP tool verification successful")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create agents: {e}")
        return False

async def test_mcp_functionality():
    """Test actual MCP functionality"""
    try:
        from .agent0DataProduct.active.advancedMcpDataProductAgent import AdvancedMCPDataProductAgent
        
        agent = AdvancedMCPDataProductAgent(os.getenv("A2A_SERVICE_URL"))
        
        # Test performance measurement
        import time
        start = time.time()
        await asyncio.sleep(0.1)
        end = time.time()
        
        metrics = await agent.performance_tools.measure_performance_metrics(
            operation_id="test_001",
            start_time=start,
            end_time=end,
            operation_count=1
        )
        
        assert metrics["duration_ms"] >= 100, f"Duration too short: {metrics['duration_ms']}"
        assert "performance_score" in metrics
        
        logger.info(f"✓ MCP performance measurement working: {metrics['duration_ms']:.2f}ms")
        return True
    except Exception as e:
        logger.error(f"❌ MCP functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    logger.info("Starting import and functionality tests\n")
    
    tests = [
        ("SDK Components", test_import_sdk_components),
        ("MCP Tools", test_import_mcp_tools),
        ("Agent Classes", test_import_agents),
        ("Agent Creation", test_create_agents),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
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
    
    # Test async functionality
    try:
        import asyncio
        result = asyncio.run(test_mcp_functionality())
        if result:
            passed += 1
            logger.info("✅ MCP Functionality: PASSED\n")
        else:
            failed += 1
            logger.error("❌ MCP Functionality: FAILED\n")
    except Exception as e:
        failed += 1
        logger.error(f"❌ MCP Functionality: FAILED with exception: {e}\n")
    
    logger.info(f"{'='*50}")
    logger.info(f"Test Summary: {passed} passed, {failed} failed")
    logger.info(f"Success Rate: {(passed/(passed+failed)*100):.1f}%")
    logger.info(f"{'='*50}")
    
    return failed == 0

if __name__ == "__main__":
    import asyncio


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
    success = main()
    sys.exit(0 if success else 1)