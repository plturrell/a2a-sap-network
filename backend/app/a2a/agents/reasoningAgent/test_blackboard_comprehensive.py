#!/usr/bin/env python3
"""
Comprehensive test for blackboard reasoning showing all knowledge sources
"""

import asyncio
import sys
import json
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

print("Comprehensive Blackboard Architecture Test")
print("=" * 60)


async def test_blackboard_detailed():
    """Test blackboard with detailed output"""
    try:
        from blackboardArchitecture import BlackboardController
        
        controller = BlackboardController()
        
        # Test questions designed to trigger different knowledge sources
        test_cases = [
            {
                "question": "If greenhouse gases increase, what happens to global temperatures and why?",
                "context": {"analysis_type": "causal", "domain": "climate_science"}
            },
            {
                "question": "Compare the patterns between economic inflation in the 1970s and 2020s",
                "context": {"analysis_type": "pattern_recognition", "domain": "economics"}
            },
            {
                "question": "Given that quantum computers can break RSA encryption, what are the logical implications for cybersecurity?",
                "context": {"analysis_type": "logical_reasoning", "domain": "computer_science"}
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{'='*60}")
            print(f"Test Case {i+1}: {test_case['question']}")
            print(f"{'='*60}")
            
            result = await controller.reason(test_case['question'], test_case['context'])
            
            print(f"\nAnswer: {result.get('answer', 'No answer')[:200]}...")
            print(f"Confidence: {result.get('confidence', 0.0):.2f}")
            print(f"Iterations: {result.get('iterations', 0)}")
            print(f"Enhanced: {result.get('enhanced', False)}")
            
            # Detailed blackboard state
            if 'blackboard_state' in result:
                state = result['blackboard_state']
                
                print("\nBlackboard State Details:")
                print(f"  Problem: {state.get('problem', '')[:80]}...")
                
                # Facts
                facts = state.get('facts', [])
                print(f"\n  Facts ({len(facts)}):")
                for fact in facts[:3]:
                    print(f"    - {fact.get('content', 'N/A')[:60]}...")
                    
                # Patterns
                patterns = state.get('patterns', [])
                print(f"\n  Patterns ({len(patterns)}):")
                for pattern in patterns[:3]:
                    print(f"    - Type: {pattern.get('type')}, Pattern: {str(pattern.get('pattern'))[:50]}...")
                    
                # Hypotheses
                hypotheses = state.get('hypotheses', [])
                print(f"\n  Hypotheses ({len(hypotheses)}):")
                for hyp in hypotheses[:3]:
                    print(f"    - {hyp.get('content', 'N/A')[:60]}...")
                    print(f"      Evidence Score: {hyp.get('evidence_score', 0):.2f}")
                    
                # Conclusions
                conclusions = state.get('conclusions', [])
                print(f"\n  Conclusions ({len(conclusions)}):")
                for conc in conclusions[:3]:
                    print(f"    - {conc.get('content', 'N/A')[:60]}...")
                    print(f"      Type: {conc.get('type')}, Confidence: {conc.get('confidence', 0):.2f}")
                    
                # Causal chains
                causal_chains = state.get('causal_chains', [])
                print(f"\n  Causal Chains ({len(causal_chains)}):")
                for chain in causal_chains[:3]:
                    print(f"    - {chain.get('cause')} → {chain.get('effect')}")
                    print(f"      Strength: {chain.get('strength', 0):.2f}")
                    
                # Knowledge source contributions
                contributions = state.get('contributions', [])
                print(f"\n  Knowledge Source Activity:")
                source_counts = {}
                source_actions = {}
                for contrib in contributions:
                    source = contrib.get('source', 'unknown')
                    action = contrib.get('action', 'unknown')
                    source_counts[source] = source_counts.get(source, 0) + 1
                    if source not in source_actions:
                        source_actions[source] = []
                    source_actions[source].append(action)
                    
                for source, count in source_counts.items():
                    print(f"    - {source}: {count} contributions")
                    unique_actions = list(set(source_actions[source]))
                    print(f"      Actions: {', '.join(unique_actions)}")
                    
        print("\n" + "="*60)
        print("✅ All blackboard tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import os
    # Set API key
    os.environ['XAI_API_KEY'] = 'your-xai-api-key-here'
    
    asyncio.run(test_blackboard_detailed())