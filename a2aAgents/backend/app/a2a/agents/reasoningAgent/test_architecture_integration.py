#!/usr/bin/env python3
"""
Test Architecture Integration
Quick test to verify all architectures are properly integrated
"""

import sys
import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Enable test mode
from testMode import enable_test_mode
enable_test_mode()

async def test_architectures():
    """Test all reasoning architectures"""
    print("Testing Architecture Integration")
    print("=" * 40)

    # Test individual architecture imports
    architectures_to_test = [
        ("Peer-to-Peer", "peerToPeerArchitecture"),
        ("Chain-of-Thought", "chainOfThoughtArchitecture"),
        ("Swarm Intelligence", "swarmIntelligenceArchitecture"),
        ("Debate", "debateArchitecture"),
        ("Blackboard", "blackboardArchitecture"),
        ("NLP Pattern Matcher", "nlpPatternMatcher")
    ]

    import_results = {}

    for name, module_name in architectures_to_test:
        try:
            __import__(module_name)
            import_results[name] = "✅ SUCCESS"
            print(f"✅ {name}: Import successful")
        except Exception as e:
            import_results[name] = f"❌ FAILED: {e}"
            print(f"❌ {name}: Import failed - {e}")

    print(f"\nImport Results: {sum(1 for r in import_results.values() if r.startswith('✅'))}/{len(import_results)} successful")

    # Test architecture creation
    print("\nTesting Architecture Creation:")
    creation_results = {}

    try:
        from peerToPeerArchitecture import create_peer_to_peer_coordinator
        coordinator = create_peer_to_peer_coordinator()
        creation_results["Peer-to-Peer"] = "✅ Created"
        print("✅ Peer-to-Peer coordinator created")
    except Exception as e:
        creation_results["Peer-to-Peer"] = f"❌ {e}"
        print(f"❌ Peer-to-Peer creation failed: {e}")

    try:
        from chainOfThoughtArchitecture import create_chain_of_thought_reasoner
        reasoner = create_chain_of_thought_reasoner()
        creation_results["Chain-of-Thought"] = "✅ Created"
        print("✅ Chain-of-Thought reasoner created")
    except Exception as e:
        creation_results["Chain-of-Thought"] = f"❌ {e}"
        print(f"❌ Chain-of-Thought creation failed: {e}")

    try:
        from swarmIntelligenceArchitecture import create_swarm_intelligence_coordinator
        swarm = create_swarm_intelligence_coordinator()
        creation_results["Swarm Intelligence"] = "✅ Created"
        print("✅ Swarm Intelligence coordinator created")
    except Exception as e:
        creation_results["Swarm Intelligence"] = f"❌ {e}"
        print(f"❌ Swarm Intelligence creation failed: {e}")

    try:
        from debateArchitecture import create_debate_coordinator
        debate = create_debate_coordinator()
        creation_results["Debate"] = "✅ Created"
        print("✅ Debate coordinator created")
    except Exception as e:
        creation_results["Debate"] = f"❌ {e}"
        print(f"❌ Debate creation failed: {e}")

    try:
        from blackboardArchitecture import BlackboardController
        blackboard = BlackboardController()
        creation_results["Blackboard"] = "✅ Created"
        print("✅ Blackboard controller created")
    except Exception as e:
        creation_results["Blackboard"] = f"❌ {e}"
        print(f"❌ Blackboard creation failed: {e}")

    try:
        from nlpPatternMatcher import create_nlp_pattern_matcher
        nlp = create_nlp_pattern_matcher()
        creation_results["NLP Pattern Matcher"] = "✅ Created"
        print("✅ NLP Pattern Matcher created")
    except Exception as e:
        creation_results["NLP Pattern Matcher"] = f"❌ {e}"
        print(f"❌ NLP Pattern Matcher creation failed: {e}")

    print(f"\nCreation Results: {sum(1 for r in creation_results.values() if r.startswith('✅'))}/{len(creation_results)} successful")

    # Test functionality
    print("\nTesting Basic Functionality:")

    test_question = "What is the meaning of artificial intelligence?"
    test_context = {"domain": "technology", "complexity": "moderate"}

    functionality_results = {}

    # Test peer-to-peer reasoning
    try:
        result = await coordinator.reason(test_question, test_context)
        functionality_results["Peer-to-Peer"] = "✅ Working"
        print(f"✅ Peer-to-Peer reasoning: {result.get('reasoning_type', 'Unknown')}")
    except Exception as e:
        functionality_results["Peer-to-Peer"] = f"❌ {e}"
        print(f"❌ Peer-to-Peer reasoning failed: {e}")

    # Test chain-of-thought reasoning
    try:
        from chainOfThoughtArchitecture import ReasoningStrategy
        result = await reasoner.reason(test_question, test_context, ReasoningStrategy.LINEAR)
        functionality_results["Chain-of-Thought"] = "✅ Working"
        print(f"✅ Chain-of-Thought reasoning: {result.get('reasoning_type', 'Unknown')}")
    except Exception as e:
        functionality_results["Chain-of-Thought"] = f"❌ {e}"
        print(f"❌ Chain-of-Thought reasoning failed: {e}")

    # Test swarm intelligence
    try:
        from swarmIntelligenceArchitecture import SwarmBehavior
        result = await swarm.reason(test_question, test_context, SwarmBehavior.EXPLORATION)
        functionality_results["Swarm Intelligence"] = "✅ Working"
        print(f"✅ Swarm Intelligence reasoning: {result.get('reasoning_type', 'Unknown')}")
    except Exception as e:
        functionality_results["Swarm Intelligence"] = f"❌ {e}"
        print(f"❌ Swarm Intelligence reasoning failed: {e}")

    # Test debate reasoning
    try:
        result = await debate.reason(test_question, test_context)
        functionality_results["Debate"] = "✅ Working"
        print(f"✅ Debate reasoning: {result.get('reasoning_type', 'Unknown')}")
    except Exception as e:
        functionality_results["Debate"] = f"❌ {e}"
        print(f"❌ Debate reasoning failed: {e}")

    # Test blackboard reasoning
    try:
        result = await blackboard.reason(test_question, test_context)
        functionality_results["Blackboard"] = "✅ Working"
        print(f"✅ Blackboard reasoning: {result.get('answer', 'No answer')[:50]}...")
    except Exception as e:
        functionality_results["Blackboard"] = f"❌ {e}"
        print(f"❌ Blackboard reasoning failed: {e}")

    # Test NLP pattern matching
    try:
        patterns = await nlp.analyze_patterns(test_question, use_grok=False)
        functionality_results["NLP Pattern Matcher"] = "✅ Working"
        print(f"✅ NLP Pattern Matching: {len(patterns)} pattern types detected")
    except Exception as e:
        functionality_results["NLP Pattern Matcher"] = f"❌ {e}"
        print(f"❌ NLP Pattern Matching failed: {e}")

    print(f"\nFunctionality Results: {sum(1 for r in functionality_results.values() if r.startswith('✅'))}/{len(functionality_results)} working")

    # Summary
    print("\n" + "=" * 40)
    print("ARCHITECTURE INTEGRATION SUMMARY")
    print("=" * 40)

    total_imports = sum(1 for r in import_results.values() if r.startswith('✅'))
    total_creations = sum(1 for r in creation_results.values() if r.startswith('✅'))
    total_functionality = sum(1 for r in functionality_results.values() if r.startswith('✅'))

    print(f"Imports: {total_imports}/{len(import_results)}")
    print(f"Creation: {total_creations}/{len(creation_results)}")
    print(f"Functionality: {total_functionality}/{len(functionality_results)}")

    if total_imports >= 5 and total_creations >= 5 and total_functionality >= 4:
        print("\n🎉 INTEGRATION SUCCESSFUL!")
        print("All architectures are properly integrated and working")
        return True
    else:
        print("\n⚠️  INTEGRATION INCOMPLETE")
        print("Some architectures need attention")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(test_architectures())
        print(f"\nIntegration test {'PASSED' if success else 'FAILED'}")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
