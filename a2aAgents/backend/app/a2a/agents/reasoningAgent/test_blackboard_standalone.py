#!/usr/bin/env python3
"""
Standalone test for blackboard reasoning - minimal dependencies
"""

import asyncio
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

print("Testing Blackboard Architecture with Grok-4")
print("=" * 60)


async def test_blackboard_minimal():
    """Test blackboard with minimal setup"""
    try:
        # First test if Grok works
        print("\n1. Testing Grok-4 Connection...")
        from grokReasoning import GrokReasoning
        grok = GrokReasoning()

        # Simple test
        result = await grok.decompose_question("What is climate change?")
        if result.get('success'):
            print("✅ Grok-4 connection successful!")
        else:
            print("❌ Grok-4 connection failed")
            return

        # Now test blackboard
        print("\n2. Testing Blackboard Architecture...")
        from blackboardArchitecture import BlackboardController

        controller = BlackboardController()

        # Test simple question
        question = "What are the main causes of global warming?"
        print(f"\nQuestion: {question}")

        result = await controller.reason(question)

        print(f"\nAnswer: {result.get('answer', 'No answer')}")
        print(f"Confidence: {result.get('confidence', 0.0):.2f}")
        print(f"Iterations: {result.get('iterations', 0)}")

        # Show knowledge source contributions
        if 'blackboard_state' in result:
            state = result['blackboard_state']
            if 'contributions' in state:
                print(f"\nKnowledge Sources Used:")
                sources = {}
                for contrib in state['contributions']:
                    source = contrib.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
                for source, count in sources.items():
                    print(f"  - {source}: {count} contributions")

        print("\n✅ Blackboard architecture test completed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_blackboard_minimal())
