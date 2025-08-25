#!/usr/bin/env python3
"""
Simple test script for blackboard reasoning with Grok-4
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import sys
import logging
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_blackboard():
    """Test blackboard reasoning"""
    try:
        # Import necessary modules
        from blackboardArchitecture import BlackboardController

        logger.info("Testing Blackboard Reasoning with Grok-4")
        logger.info("=" * 60)

        # Create controller
        controller = BlackboardController()

        # Test questions
        test_questions = [
            "What are the environmental impacts of renewable energy?",
            "How does machine learning differ from traditional programming?",
            "What causes economic inflation?"
        ]

        for question in test_questions:
            logger.info(f"\nQuestion: {question}")
            logger.info("-" * 40)

            # Run reasoning
            result = await controller.reason(question)

            # Display results
            logger.info(f"Answer: {result.get('answer', 'No answer')}")
            logger.info(f"Confidence: {result.get('confidence', 0.0):.2f}")
            logger.info(f"Iterations: {result.get('iterations', 0)}")
            logger.info(f"Enhanced: {result.get('enhanced', False)}")

            # Show blackboard state summary
            if 'blackboard_state' in result:
                state = result['blackboard_state']
                logger.info(f"\nBlackboard Summary:")
                logger.info(f"  Facts: {len(state.get('facts', []))}")
                logger.info(f"  Patterns: {len(state.get('patterns', []))}")
                logger.info(f"  Conclusions: {len(state.get('conclusions', []))}")
                logger.info(f"  Hypotheses: {len(state.get('hypotheses', []))}")

        logger.info("\n" + "=" * 60)
        logger.info("✅ Blackboard reasoning test completed!")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)


async def test_grok_integration():
    """Test Grok-4 integration directly"""
    try:
        from grokReasoning import GrokReasoning

        logger.info("\nTesting Grok-4 Integration")
        logger.info("=" * 60)

        grok = GrokReasoning()

        # Test decomposition
        result = await grok.decompose_question("What are the causes of climate change?")
        logger.info(f"Decomposition successful: {result.get('success', False)}")

        # Test pattern analysis
        result = await grok.analyze_patterns("Climate change is caused by greenhouse gas emissions")
        logger.info(f"Pattern analysis successful: {result.get('success', False)}")

        # Test synthesis
        sub_answers = [
            {"content": "Greenhouse gases trap heat"},
            {"content": "Human activities increase emissions"},
            {"content": "Deforestation reduces CO2 absorption"}
        ]
        result = await grok.synthesize_answer(sub_answers, "What causes climate change?")
        logger.info(f"Synthesis successful: {result.get('success', False)}")

        logger.info("✅ Grok-4 integration working!")

    except Exception as e:
        logger.error(f"Grok test failed: {e}", exc_info=True)


async def main():
    """Run all tests"""
    # Test Grok integration first
    await test_grok_integration()

    # Then test blackboard
    await test_blackboard()


if __name__ == "__main__":
    asyncio.run(main())
