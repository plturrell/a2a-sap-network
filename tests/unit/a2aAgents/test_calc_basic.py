#!/usr/bin/env python3
"""
Simple test for the enhanced Calculation Validation Agent
"""

import sys
import os
import asyncio
import logging

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_import():
    """Test basic imports"""
    try:
        # Test if we can import the main SDK components
        from app.a2a.sdk.types import A2AMessage, MessagePart
        logger.info("✅ Successfully imported A2A SDK types")
        
        from app.a2a.agents.agent4CalcValidation.active.calcValidationAgentSdk import CalcValidationAgentSDK
        logger.info("✅ Successfully imported CalcValidationAgentSDK")
        
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

async def test_basic_functionality():
    """Test basic agent functionality"""
    try:
        # Only test if imports work
        if not test_import():
            return False
            
        from app.a2a.agents.agent4CalcValidation.active.calcValidationAgentSdk import CalcValidationAgentSDK
        
        # Create agent instance
        agent = CalcValidationAgentSDK("http://localhost:8004")
        logger.info(f"✅ Created agent instance: {agent.name}")
        
        # Check capabilities
        status = agent.get_agent_status()
        logger.info(f"✅ Agent capabilities: {len(status['capabilities'])} found")
        for capability in status['capabilities']:
            logger.info(f"  - {capability}")
        
        # Test simple reasoning method
        analysis = agent._analyze_expression_complexity("2 + 2")
        logger.info(f"✅ Expression analysis: {analysis['complexity_level']}")
        
        # Test method selection
        method = agent._select_method("x**2 + y**2")
        logger.info(f"✅ Method selection: {method}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Main test runner"""
    logger.info("="*60)
    logger.info("ENHANCED CALC VALIDATION AGENT - SIMPLE TEST")
    logger.info("Testing imports and basic functionality")
    logger.info("="*60)
    
    # Test imports
    if test_import():
        logger.info("✅ All imports successful")
        
        # Test basic functionality
        result = asyncio.run(test_basic_functionality())
        if result:
            logger.info("🎉 All basic tests passed!")
        else:
            logger.error("❌ Basic functionality tests failed")
    else:
        logger.error("❌ Import tests failed")

if __name__ == "__main__":
    main()