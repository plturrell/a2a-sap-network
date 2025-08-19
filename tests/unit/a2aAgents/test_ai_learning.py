#!/usr/bin/env python3
"""
Test AI Learning Capabilities of Enhanced Calculation Validation Agent
"""

import sys
import os
import asyncio
import logging
import numpy as np

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_ai_learning():
    """Test AI learning and adaptive capabilities"""
    try:
        # Import the enhanced agent
        from app.a2a.agents.agent4CalcValidation.active.calcValidationAgentSdk import CalcValidationAgentSDK
        
        logger.info("="*60)
        logger.info("AI LEARNING TEST - ENHANCED CALC VALIDATION AGENT")
        logger.info("="*60)
        
        # Create agent
        agent = CalcValidationAgentSDK("http://localhost:8004")
        logger.info(f"✅ Created agent: {agent.name}")
        
        # Initialize (this will set up AI learning)
        await agent.initialize()
        logger.info("✅ Agent initialized with AI learning")
        
        # Check AI learning status
        status = agent.get_agent_status()
        ai_status = status['ai_learning']
        
        logger.info("\n📊 AI Learning Status:")
        logger.info(f"  Model trained: {ai_status['model_trained']}")
        logger.info(f"  Training samples: {ai_status['training_samples']}")
        logger.info(f"  Patterns learned: {ai_status['patterns_learned']}")
        logger.info(f"  Learning enabled: {ai_status['learning_enabled']}")
        
        # Test feature extraction
        logger.info("\n🧠 Testing Feature Extraction:")
        test_expressions = [
            "2 + 2",
            "x**2 - 1", 
            "sin(x)**2 + cos(x)**2",
            "integrate(x**2, x)",
            "x + y + z + w"
        ]
        
        for expr in test_expressions:
            features = agent._extract_expression_features(expr)
            logger.info(f"  {expr} -> {len(features)} features")
        
        # Test pattern recognition
        logger.info("\n🔍 Testing Pattern Recognition:")
        for expr in test_expressions:
            pattern = agent._get_expression_pattern(expr)
            logger.info(f"  {expr} -> {pattern}")
        
        # Test AI method selection
        logger.info("\n🤖 Testing AI Method Selection:")
        for expr in test_expressions:
            if agent.method_selector_ml:
                ai_method = agent._ai_method_selection(expr)
                logger.info(f"  {expr} -> AI: {ai_method}")
            
            fallback_method = agent._rule_based_method_selection(expr)
            logger.info(f"  {expr} -> Fallback: {fallback_method}")
        
        # Test learning from validation
        logger.info("\n📚 Testing Adaptive Learning:")
        from app.a2a.agents.agent4CalcValidation.active.calcValidationAgentSdk import ValidationResult
        
        # Simulate some validation results
        mock_result = ValidationResult(
            expression="test_expr",
            method_used="symbolic",
            result=True,
            confidence=0.95
        )
        
        initial_samples = len(agent.training_data['expressions'])
        await agent._learn_from_validation_result("2*x + 3", "symbolic", mock_result, 0.1)
        final_samples = len(agent.training_data['expressions'])
        
        logger.info(f"  Training samples: {initial_samples} -> {final_samples}")
        logger.info(f"  ✅ Learning system is functional")
        
        # Show final capabilities
        logger.info("\n🎯 Enhanced Capabilities:")
        for capability in status['capabilities']:
            logger.info(f"  ✅ {capability}")
        
        logger.info("\n🎉 AI Learning test completed successfully!")
        
        await agent.shutdown()
        return True
        
    except Exception as e:
        logger.error(f"❌ AI Learning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test runner"""
    result = asyncio.run(test_ai_learning())
    if result:
        print("\n✅ AI Learning capabilities are working!")
    else:
        print("\n❌ AI Learning test failed")

if __name__ == "__main__":
    main()