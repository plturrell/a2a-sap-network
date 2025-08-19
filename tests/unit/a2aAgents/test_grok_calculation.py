#!/usr/bin/env python3
"""
Test script for Grok-enhanced calculation capabilities
Demonstrates AI-powered mathematical understanding and solutions
"""

import sys
import os
import asyncio
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_grok_enhanced_calculations():
    """Test the Grok-enhanced calculation features"""
    
    try:
        from app.a2a.agents.calculationAgent.active.enhancedCalculationAgentSdk import EnhancedCalculationAgentSDK
        
        print("🤖 Testing Grok-Enhanced Mathematical Calculations")
        print("=" * 70)
        
        # Initialize the enhanced calculation agent
        agent = EnhancedCalculationAgentSDK(
            agent_id="calc_test_agent",
            name="Grok Enhanced Calc Test",
            enable_monitoring=False
        )
        
        # Initialize the agent
        await agent.initialize()
        
        if not agent.grok_math_client:
            print("❌ Grok Mathematical Client not available. Please configure API keys.")
            return
        
        print("✅ Grok Mathematical Client initialized successfully\n")
        
        # Test cases for different modes
        test_cases = [
            {
                "name": "Complex Derivative with AI Understanding",
                "input": {
                    "query": "Find the derivative of x^3 * sin(x) + e^(2x) with respect to x",
                    "mode": "solve"
                }
            },
            {
                "name": "Word Problem with Step-by-Step Solution",
                "input": {
                    "query": "A ball is thrown upward with initial velocity 20 m/s. Find the maximum height using calculus.",
                    "mode": "solve"
                }
            },
            {
                "name": "Mathematical Concept Explanation",
                "input": {
                    "query": "L'Hôpital's Rule",
                    "mode": "explain",
                    "level": "intermediate",
                    "include_practice": True
                }
            },
            {
                "name": "Interactive Teaching Mode",
                "input": {
                    "query": "Teach me how to solve integrals by substitution",
                    "mode": "teach"
                }
            },
            {
                "name": "Calculation Validation",
                "input": {
                    "query": "Verify that the integral of x^2 from 0 to 3 equals 9",
                    "mode": "validate",
                    "result": 9,
                    "steps": [
                        {"description": "Apply power rule for integration", "result": "x^3/3"},
                        {"description": "Evaluate at bounds", "result": "3^3/3 - 0^3/3 = 9"}
                    ]
                }
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*70}")
            print(f"Test {i}: {test_case['name']}")
            print(f"{'='*70}")
            
            try:
                # Execute the Grok-enhanced calculation
                result = await agent.grok_enhanced_calculation(test_case['input'])
                
                if result.get("success"):
                    print(f"✅ Success!")
                    
                    if result["mode"] == "solve":
                        print(f"\n📊 Analysis:")
                        analysis = result.get("analysis", {})
                        print(f"   Operation: {analysis.get('operation_type', 'unknown')}")
                        print(f"   Expression: {analysis.get('mathematical_expression', 'N/A')}")
                        print(f"   Confidence: {analysis.get('confidence', 0):.1%}")
                        
                        if "solution" in result:
                            solution = result["solution"]
                            if isinstance(solution, dict) and "steps" in solution:
                                print(f"\n📝 Step-by-Step Solution:")
                                for step in solution.get("steps", []):
                                    print(f"   Step {step.get('step_number', '?')}: {step.get('description', '')}")
                                    if "result" in step:
                                        print(f"      → {step['result']}")
                                
                                if "final_answer" in solution:
                                    print(f"\n✨ Final Answer: {solution['final_answer']}")
                        
                        if "calculation_result" in result and result["calculation_result"]:
                            print(f"\n🔢 Computed Result: {result['calculation_result'].get('result', 'N/A')}")
                    
                    elif result["mode"] == "explain":
                        print(f"\n📚 Concept: {result['concept']}")
                        print(f"   Level: {result['level']}")
                        if "explanation" in result:
                            print(f"\n📖 Explanation Preview:")
                            print(f"   {result['explanation'][:200]}...")
                        
                        if "practice_problems" in result and result["practice_problems"]:
                            print(f"\n📝 Practice Problems Generated: {len(result['practice_problems'])}")
                            for prob in result["practice_problems"][:2]:
                                print(f"   • {prob.get('problem_statement', 'N/A')}")
                    
                    elif result["mode"] == "validate":
                        validation = result.get("validation", {})
                        print(f"\n✅ Validation Result: {'Correct' if validation.get('is_correct') else 'Incorrect'}")
                        print(f"   Confidence: {validation.get('confidence', 0):.1%}")
                        if "verification_method" in validation:
                            print(f"   Method: {validation['verification_method']}")
                    
                    elif result["mode"] == "teach":
                        response = result.get("response", {})
                        print(f"\n🎓 Teaching Status: {response.get('status', 'unknown')}")
                        if "teaching_material" in response:
                            print(f"   Teaching materials prepared")
                        if "clarification_questions" in response:
                            print(f"   Clarification needed: {response['clarification_questions']}")
                    
                else:
                    print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                    if "suggestions" in result:
                        print(f"   Suggestions: {result['suggestions']}")
                
            except Exception as e:
                print(f"❌ Test failed with error: {e}")
        
        # Test enhanced natural language dispatching with Grok
        print(f"\n\n{'='*70}")
        print("🎯 Testing Enhanced Natural Language Dispatching with Grok")
        print(f"{'='*70}")
        
        nl_queries = [
            "What is the area under the curve y = x^2 from x = 0 to x = 5?",
            "Solve the differential equation dy/dx = 2x with initial condition y(0) = 1",
            "Find all critical points of f(x) = x^3 - 3x^2 + 2"
        ]
        
        for query in nl_queries:
            print(f"\n🔍 Query: \"{query}\"")
            
            result = await agent.intelligent_dispatch_calculation({
                "request": query,
                "auto_execute": True,
                "use_grok_enhancement": True
            })
            
            if result.get("success"):
                print(f"✅ Dispatched to: {result.get('dispatched_to', 'unknown')}")
                print(f"   Confidence: {result.get('confidence', 0):.1%}")
                if "result" in result and result["result"]:
                    print(f"   Result: {result['result'].get('result', 'N/A')}")
                if "analysis" in result and "grok_enhancement" in result["analysis"]:
                    print(f"   🤖 Grok AI Enhancement Applied")
            else:
                print(f"❌ Dispatch failed: {result.get('error', 'Unknown error')}")
        
        print(f"\n{'='*70}")
        print("✅ Grok-Enhanced Calculation Testing Complete!")
        print("\n🎉 Key Enhancements Demonstrated:")
        print("   • AI-powered natural language understanding")
        print("   • Step-by-step solution generation")
        print("   • Mathematical concept explanation")
        print("   • Interactive teaching capabilities")
        print("   • Intelligent result validation")
        print("   • Enhanced confidence scoring with AI")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Enhanced calculation agent not available")
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Grok-Enhanced Mathematical Calculation Test Suite")
    print("=" * 70)
    print("This test demonstrates the integration of Grok AI with the calculation agent")
    print("for enhanced natural language understanding and mathematical problem solving.\n")
    
    # Run the async test
    asyncio.run(test_grok_enhanced_calculations())