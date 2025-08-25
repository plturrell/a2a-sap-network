#!/usr/bin/env python3
"""
Test script to verify the blackboard reasoning integration with Grok-4
"""

import asyncio
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_blackboard_reasoning():
    """Test the blackboard reasoning integration"""
    try:
        # Import the reasoning skills
        from reasoningSkills import ReasoningOrchestrationSkills
        from app.a2a.core.trustIdentity import TrustIdentity

        # Create a trust identity for testing
        trust_identity = TrustIdentity(
            agent_id="test_reasoning_agent",
            agent_type="reasoning",
            capabilities=["blackboard_reasoning"],
            blockchain_address="0x1234567890abcdef",
            reputation_score=0.8
        )

        # Initialize the orchestration skills
        orchestration_skills = ReasoningOrchestrationSkills(trust_identity)

        # Test cases
        test_cases = [
            {
                "question": "What are the environmental impacts of renewable energy compared to fossil fuels?",
                "context": {
                    "domain": "environmental_science",
                    "comparison_type": "impact_analysis"
                }
            },
            {
                "question": "How does machine learning differ from traditional programming?",
                "context": {
                    "domain": "computer_science",
                    "focus": "methodology"
                }
            },
            {
                "question": "What causes economic inflation and what are its effects?",
                "context": {
                    "domain": "economics",
                    "analysis_type": "causal"
                }
            }
        ]

        # Run tests
        for i, test_case in enumerate(test_cases):
            logger.info(f"\n{'='*60}")
            logger.info(f"Test Case {i+1}: {test_case['question']}")
            logger.info(f"{'='*60}")

            # Create mock state and request objects
            class MockState:
                def __init__(self, question):
                    self.question = question

            class MockRequest:
                def __init__(self, question, context):
                    self.question = question
                    self.context = context

            state = MockState(test_case["question"])
            request = MockRequest(test_case["question"], test_case["context"])

            # Run blackboard reasoning
            start_time = datetime.utcnow()
            result = await orchestration_skills.blackboard_reasoning(state, request)
            end_time = datetime.utcnow()

            # Display results
            logger.info(f"\nResult:")
            logger.info(f"Answer: {result.get('answer', 'No answer')}")
            logger.info(f"Confidence: {result.get('confidence', 0.0):.2f}")
            logger.info(f"Architecture: {result.get('reasoning_architecture', 'unknown')}")
            logger.info(f"Enhanced: {result.get('enhanced', False)}")
            logger.info(f"Processing Time: {(end_time - start_time).total_seconds():.2f} seconds")

            if 'blackboard_state' in result:
                state = result['blackboard_state']
                logger.info(f"\nBlackboard State:")
                logger.info(f"  - Facts: {state.get('facts', 0)}")
                logger.info(f"  - Hypotheses: {state.get('hypotheses', 0)}")
                logger.info(f"  - Evidence: {state.get('evidence', 0)}")
                logger.info(f"  - Conclusions: {state.get('conclusions', 0)}")
                logger.info(f"  - Patterns: {state.get('patterns', 0)}")
                logger.info(f"  - Iterations: {state.get('iteration', 0)}")

            if 'error' in result:
                logger.error(f"Error occurred: {result['error']}")

    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)


async def test_direct_blackboard():
    """Test the blackboard architecture directly"""
    try:
        from blackboardArchitecture import blackboard_reasoning

        logger.info("\n" + "="*60)
        logger.info("Testing Direct Blackboard Architecture")
        logger.info("="*60)

        # Test question
        question = "What are the key factors driving climate change and their interconnections?"
        context = {
            "domain": "climate_science",
            "analysis_depth": "comprehensive",
            "include_causal_chains": True
        }

        # Run blackboard reasoning
        start_time = datetime.utcnow()
        result = await blackboard_reasoning(question, context)
        end_time = datetime.utcnow()

        # Display results
        logger.info(f"\nDirect Blackboard Result:")
        logger.info(f"Answer: {result.get('answer', 'No answer')}")
        logger.info(f"Confidence: {result.get('confidence', 0.0):.2f}")
        logger.info(f"Enhanced: {result.get('enhanced', False)}")
        logger.info(f"Iterations: {result.get('iterations', 0)}")
        logger.info(f"Processing Time: {(end_time - start_time).total_seconds():.2f} seconds")

        # Display detailed blackboard state if available
        if 'blackboard_state' in result:
            state = result['blackboard_state']
            logger.info(f"\nDetailed Blackboard State:")

            if 'facts' in state and state['facts']:
                logger.info(f"\nFacts ({len(state['facts'])}):")
                for fact in state['facts'][:3]:  # Show first 3
                    logger.info(f"  - {fact.get('content', 'N/A')} (confidence: {fact.get('confidence', 0):.2f})")

            if 'patterns' in state and state['patterns']:
                logger.info(f"\nPatterns ({len(state['patterns'])}):")
                for pattern in state['patterns'][:3]:  # Show first 3
                    logger.info(f"  - Type: {pattern.get('type', 'N/A')}, Pattern: {pattern.get('pattern', 'N/A')}")

            if 'conclusions' in state and state['conclusions']:
                logger.info(f"\nConclusions ({len(state['conclusions'])}):")
                for conclusion in state['conclusions'][:3]:  # Show first 3
                    logger.info(f"  - {conclusion.get('content', 'N/A')} (confidence: {conclusion.get('confidence', 0):.2f})")

            if 'contributions' in state and state['contributions']:
                logger.info(f"\nKnowledge Source Contributions ({len(state['contributions'])}):")
                sources = {}
                for contrib in state['contributions']:
                    source = contrib.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
                for source, count in sources.items():
                    logger.info(f"  - {source}: {count} contributions")

    except Exception as e:
        logger.error(f"Direct blackboard test failed: {e}", exc_info=True)


async def main():
    """Run all tests"""
    logger.info("Starting Blackboard Reasoning Integration Tests")

    # Test through reasoning skills
    await test_blackboard_reasoning()

    # Test direct blackboard
    await test_direct_blackboard()

    logger.info("\nAll tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
