#!/usr/bin/env python3
"""
Quick test for blackboard reasoning
"""

import asyncio
import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Set API key
os.environ['XAI_API_KEY'] = 'your-xai-api-key-here'

async def quick_test():
    """Quick blackboard test"""
    try:
        from blackboardArchitecture import BlackboardController

        print("Quick Blackboard Test")
        print("=" * 40)

        controller = BlackboardController()

        question = "What causes inflation?"
        print(f"Question: {question}")

        result = await controller.reason(question)

        print(f"\nAnswer: {result.get('answer', 'No answer')[:150]}...")
        print(f"Confidence: {result.get('confidence', 0.0):.2f}")
        print(f"Iterations: {result.get('iterations', 0)}")
        print(f"Enhanced: {result.get('enhanced', False)}")

        # Show knowledge source activity
        if 'blackboard_state' in result:
            state = result['blackboard_state']
            contributions = state.get('contributions', [])
            print(f"\nKnowledge Source Activity:")
            for contrib in contributions:
                print(f"  - {contrib.get('source')}: {contrib.get('action')}")

        print("\n✅ Quick test completed!")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(quick_test())
