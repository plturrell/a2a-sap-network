import asyncio
import logging
import time
import sys
from pathlib import Path

#!/usr/bin/env python3
"""
Test script for Calculation Validation Agent
Tests mathematical validation capabilities
"""

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.a2a.agents.agent4CalcValidation.active.calcValidationAgentSdk import (
    CalcValidationAgentSDK
)


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_calc_validation_agent():
    """Test the calculation validation agent"""

    # Create agent
    agent = CalcValidationAgentSDK(os.getenv("A2A_SERVICE_URL"))

    try:
        # Initialize
        logger.info("Initializing Calculation Validation Agent...")
        await agent.initialize()
        logger.info("✅ Agent initialized")

        # Test cases
        test_cases = [
            {
                'name': 'Simple Arithmetic',
                'expression': '2 + 2',
                'expected': 4,
                'method': 'numerical'
            },
            {
                'name': 'Symbolic Identity',
                'expression': 'x**2 - 1',
                'expected': '(x-1)*(x+1)',
                'method': 'symbolic'
            },
            {
                'name': 'Trigonometric Identity',
                'expression': 'sin(x)**2 + cos(x)**2',
                'expected': 1,
                'method': 'symbolic'
            },
            {
                'name': 'Statistical Expression',
                'expression': 'x + y',
                'expected': None,
                'method': 'statistical'
            },
            {
                'name': 'Complex Numerical',
                'expression': 'sqrt(2)**2',
                'expected': 2.0,
                'method': 'numerical'
            }
        ]

        logger.info(f"\n{'='*60}")
        logger.info("TESTING CALCULATION VALIDATION")
        logger.info(f"{'='*60}")

        for i, test_case in enumerate(test_cases):
            logger.info(f"\nTest {i+1}: {test_case['name']}")
            logger.info(f"Expression: {test_case['expression']}")
            logger.info(f"Method: {test_case['method']}")

            # Create message
            from app.a2a.sdk.types import A2AMessage, MessagePart

            message = A2AMessage(
                id=f"test_{i}",
                conversation_id=f"validation_test_{i}",
                parts=[
                    MessagePart(
                        kind="data",
                        data={
                            'expression': test_case['expression'],
                            'expected_result': test_case['expected'],
                            'method': test_case['method']
                        }
                    )
                ]
            )

            # Perform validation
            start_time = time.time()
            response = await agent.handle_calculation_validation(message)
            end_time = time.time()

            if response.get('success'):
                result = response['data']

                # Log results
                logger.info(f"Method used: {result.get('method_used')}")
                logger.info(f"Confidence: {result.get('confidence', 0):.3f}")
                logger.info(f"Error bound: {result.get('error_bound')}")
                logger.info(f"Processing time: {end_time - start_time:.3f}s")

                if result.get('error_message'):
                    logger.info(f"Error: {result['error_message']}")
                else:
                    logger.info(f"Result: {result.get('result')}")

                logger.info("✅ Test PASSED")
            else:
                logger.error(f"❌ Test FAILED: {response.get('error')}")

        # Test legacy interface
        logger.info(f"\nTesting legacy interface...")
        legacy_result = await agent.validate_calculation({
            'expression': '3 * 3',
            'expected_result': 9,
            'method': 'numerical'
        })
        logger.info(f"Legacy result: {legacy_result}")

        # Get agent status
        status = agent.get_agent_status()

        logger.info(f"\n{'='*60}")
        logger.info("AGENT STATUS")
        logger.info(f"{'='*60}")
        logger.info(f"Agent: {status['agent_name']} v{status['version']}")
        logger.info(f"Total validations: {status['metrics']['total_validations']}")
        logger.info(f"Cache hits: {status['metrics']['cache_hits']}")
        logger.info(f"Validation errors: {status['metrics']['validation_errors']}")

        logger.info(f"\nMethod Performance:")
        for method, perf in status['method_performance'].items():
            logger.info(f"  {method}: {perf['success_rate']:.1%} success rate ({perf['total_attempts']} attempts)")

        logger.info(f"\nCapabilities:")
        for capability in status['capabilities']:
            logger.info(f"  ✅ {capability}")

        logger.info(f"\n🎉 All tests completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

    finally:
        await agent.shutdown()


def main():
    """Main entry point"""
    logger.info("="*60)
    logger.info("CALCULATION VALIDATION AGENT TEST")
    logger.info("Testing mathematical validation capabilities")
    logger.info("="*60)

    # Run test
    asyncio.run(test_calc_validation_agent())


if __name__ == "__main__":
    main()
