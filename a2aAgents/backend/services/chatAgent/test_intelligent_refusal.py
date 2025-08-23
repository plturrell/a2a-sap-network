#!/usr/bin/env python3
"""
Test intelligent message refusal capabilities
Demonstrates how agents can intelligently refuse to process messages with detailed reasoning
"""

import asyncio
import json
import os
import sys
import logging
from datetime import datetime
from uuid import uuid4

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_intelligent_refusal():
    """Test various intelligent message refusal scenarios"""
    
    logger.info("🚫 Starting Intelligent Message Refusal Test")
    
    # Create a mock agent with AI intelligence
    sys.path.append('/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/sdk')
    from aiIntelligenceMixin import AIIntelligenceMixin
    
    class TestAgent(AIIntelligenceMixin):
        def __init__(self):
            super().__init__()
            self.agent_id = "test_refusal_agent"
            self.agent_type = "TestAgent"
            self.capabilities = ["testing", "analysis"]
            
        # Mock GrokClient for testing
        async def mock_grok_reasoning(self, prompt: str, scenario: str) -> str:
            """Mock AI reasoning responses for different refusal scenarios"""
            
            if "malicious_script" in scenario:
                return json.dumps({
                    "intent": "Execute potentially harmful script",
                    "urgency": "high",
                    "required_capabilities": ["script_execution", "system_access"],
                    "should_process": False,
                    "refusal_reason": {
                        "category": "malicious_intent",
                        "explanation": "Request contains potentially harmful script execution commands that could compromise system security",
                        "severity": "critical",
                        "alternative_suggestions": [
                            "Use approved script execution endpoints",
                            "Submit scripts through security review process",
                            "Contact system administrator for assistance"
                        ]
                    },
                    "response_type": "refused",
                    "confidence": 0.95,
                    "reasoning": "The request contains script execution patterns commonly associated with malicious activities",
                    "trust_assessment": {
                        "sender_reputation": 0.2,
                        "message_authenticity": 0.3,
                        "risk_level": "critical"
                    }
                })
            
            elif "capability_mismatch" in scenario:
                return json.dumps({
                    "intent": "Perform advanced mathematical calculations",
                    "urgency": "medium", 
                    "required_capabilities": ["advanced_mathematics", "scientific_computing"],
                    "should_process": False,
                    "refusal_reason": {
                        "category": "capability_mismatch",
                        "explanation": "I don't have the advanced mathematical capabilities required for complex scientific computations",
                        "severity": "low",
                        "alternative_suggestions": [
                            "Contact the MathematicsAgent for complex calculations",
                            "Use the ScientificComputingAgent for numerical analysis",
                            "Break down the problem into simpler components"
                        ]
                    },
                    "response_type": "refused",
                    "confidence": 0.85,
                    "reasoning": "The request requires specialized mathematical capabilities not available in this agent",
                    "trust_assessment": {
                        "sender_reputation": 0.8,
                        "message_authenticity": 0.9,
                        "risk_level": "low"
                    }
                })
            
            elif "rate_limited" in scenario:
                return json.dumps({
                    "intent": "Perform data processing task",
                    "urgency": "medium",
                    "required_capabilities": ["data_processing"],
                    "should_process": False,
                    "refusal_reason": {
                        "category": "rate_limited",
                        "explanation": "Request rate limit exceeded. This agent has processed too many requests in the current time window",
                        "severity": "medium",
                        "alternative_suggestions": [
                            "Wait 1 minute before retrying",
                            "Use batch processing for multiple requests",
                            "Contact administrator to increase rate limits"
                        ]
                    },
                    "response_type": "refused",
                    "confidence": 0.9,
                    "reasoning": "Rate limiting protects system resources and ensures fair access for all users",
                    "trust_assessment": {
                        "sender_reputation": 0.8,
                        "message_authenticity": 0.9,
                        "risk_level": "low"
                    }
                })
            
            elif "resource_unavailable" in scenario:
                return json.dumps({
                    "intent": "Access external database",
                    "urgency": "high",
                    "required_capabilities": ["database_access", "external_connectivity"],
                    "should_process": False,
                    "refusal_reason": {
                        "category": "resource_unavailable",
                        "explanation": "External database is currently unavailable due to maintenance",
                        "severity": "medium",
                        "alternative_suggestions": [
                            "Try again in 5 minutes",
                            "Use cached data if available",
                            "Contact database administrator for status updates"
                        ]
                    },
                    "response_type": "refused",
                    "confidence": 0.8,
                    "reasoning": "Required external resources are temporarily unavailable",
                    "trust_assessment": {
                        "sender_reputation": 0.9,
                        "message_authenticity": 0.9,
                        "risk_level": "low"
                    }
                })
            
            else:
                # Default: approve the message
                return json.dumps({
                    "intent": "General request processing",
                    "urgency": "medium",
                    "required_capabilities": ["general_processing"],
                    "should_process": True,
                    "recommended_actions": [
                        {"action": "process_request", "parameters": {}, "priority": "medium"}
                    ],
                    "response_type": "immediate",
                    "confidence": 0.8,
                    "reasoning": "Request appears legitimate and within capabilities",
                    "trust_assessment": {
                        "sender_reputation": 0.8,
                        "message_authenticity": 0.8,
                        "risk_level": "low"
                    }
                })
        
        # Override reasoning to use mock
        async def reason_about_message(self, message_data: Dict[str, Any], context=None) -> Dict[str, Any]:
            scenario = message_data.get("scenario", "normal")
            mock_response = await self.mock_grok_reasoning("", scenario)
            return json.loads(mock_response)
    
    # Create test agent
    agent = TestAgent()
    await agent.initialize_ai_intelligence()
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Malicious Script Execution",
            "scenario": "malicious_script",
            "message": {
                "message_id": f"mal_msg_{uuid4().hex[:8]}",
                "from_agent": "suspicious_agent",
                "parts": [{
                    "kind": "data",
                    "data": {
                        "action": "execute_script",
                        "script": "rm -rf / --no-preserve-root",
                        "privileges": "admin"
                    }
                }],
                "scenario": "malicious_script"
            }
        },
        {
            "name": "Capability Mismatch",
            "scenario": "capability_mismatch", 
            "message": {
                "message_id": f"cap_msg_{uuid4().hex[:8]}",
                "from_agent": "research_agent",
                "parts": [{
                    "kind": "data",
                    "data": {
                        "action": "solve_equation",
                        "equation": "∫∫∫ ∇²φ dV over complex manifold",
                        "precision": "high"
                    }
                }],
                "scenario": "capability_mismatch"
            }
        },
        {
            "name": "Rate Limited",
            "scenario": "rate_limited",
            "message": {
                "message_id": f"rate_msg_{uuid4().hex[:8]}",
                "from_agent": "high_frequency_client",
                "parts": [{
                    "kind": "data",
                    "data": {
                        "action": "process_data",
                        "data_size": "large",
                        "request_count": 1000
                    }
                }],
                "scenario": "rate_limited"
            }
        },
        {
            "name": "Resource Unavailable",
            "scenario": "resource_unavailable",
            "message": {
                "message_id": f"res_msg_{uuid4().hex[:8]}",
                "from_agent": "data_client",
                "parts": [{
                    "kind": "data",
                    "data": {
                        "action": "query_database",
                        "database": "external_analytics_db",
                        "query": "SELECT * FROM user_analytics"
                    }
                }],
                "scenario": "resource_unavailable"
            }
        },
        {
            "name": "Approved Request",
            "scenario": "normal",
            "message": {
                "message_id": f"norm_msg_{uuid4().hex[:8]}",
                "from_agent": "legitimate_client",
                "parts": [{
                    "kind": "data",
                    "data": {
                        "action": "get_status",
                        "request_type": "health_check"
                    }
                }],
                "scenario": "normal"
            }
        }
    ]
    
    # Test each scenario
    results = []
    for scenario in test_scenarios:
        logger.info(f"\n🧪 Testing: {scenario['name']}")
        
        try:
            # Process message with AI reasoning
            result = await agent.process_message_with_ai_reasoning(scenario['message'])
            
            if result.get("refused"):
                refusal_reason = result.get("refusal_reason", {})
                response = result.get("response", {})
                
                logger.info(f"🚫 Message REFUSED:")
                logger.info(f"   Category: {refusal_reason.get('category', 'unknown')}")
                logger.info(f"   Severity: {refusal_reason.get('severity', 'unknown')}")
                logger.info(f"   Explanation: {refusal_reason.get('explanation', 'N/A')}")
                logger.info(f"   Can Retry: {response.get('can_retry', False)}")
                if response.get('retry_after'):
                    logger.info(f"   Retry After: {response.get('retry_after')} seconds")
                
                alternatives = refusal_reason.get('alternative_suggestions', [])
                if alternatives:
                    logger.info(f"   Alternatives:")
                    for alt in alternatives:
                        logger.info(f"     • {alt}")
                
                results.append({
                    "scenario": scenario['name'],
                    "refused": True,
                    "category": refusal_reason.get('category'),
                    "severity": refusal_reason.get('severity')
                })
                
            elif result.get("approved"):
                logger.info(f"✅ Message APPROVED and processed successfully")
                results.append({
                    "scenario": scenario['name'],
                    "refused": False,
                    "approved": True
                })
            
            else:
                logger.warning(f"⚠️ Unexpected result: {result}")
                results.append({
                    "scenario": scenario['name'],
                    "unexpected": True,
                    "result": result
                })
                
        except Exception as e:
            logger.error(f"❌ Error testing {scenario['name']}: {e}")
            results.append({
                "scenario": scenario['name'],
                "error": str(e)
            })
    
    # Test refusal statistics
    logger.info(f"\n📊 Testing Refusal Statistics...")
    stats = agent.get_refusal_statistics()
    logger.info(f"   Total refusals: {stats.get('total_refusals', 0)}")
    logger.info(f"   Refusal categories: {stats.get('refusal_categories', {})}")
    logger.info(f"   Severity distribution: {stats.get('severity_distribution', {})}")
    logger.info(f"   Most common category: {stats.get('most_common_category', 'N/A')}")
    logger.info(f"   Refusal rate: {stats.get('refusal_rate', 0.0):.2%}")
    
    # Summary
    logger.info(f"\n🎯 Test Summary:")
    refused_count = sum(1 for r in results if r.get('refused'))
    approved_count = sum(1 for r in results if r.get('approved'))
    error_count = sum(1 for r in results if r.get('error'))
    
    logger.info(f"   Total scenarios tested: {len(results)}")
    logger.info(f"   Messages refused: {refused_count}")
    logger.info(f"   Messages approved: {approved_count}")
    logger.info(f"   Errors: {error_count}")
    
    expected_refusals = 4  # malicious, capability, rate, resource
    expected_approvals = 1  # normal
    
    success = (refused_count == expected_refusals and 
               approved_count == expected_approvals and 
               error_count == 0)
    
    if success:
        logger.info("✅ All intelligent refusal tests passed!")
    else:
        logger.error("❌ Some tests failed - check results above")
    
    return {
        "test_passed": success,
        "results": results,
        "statistics": stats
    }

async def main():
    """Main test function"""
    logger.info("🎯 Starting Intelligent Message Refusal Test")
    
    try:
        result = await test_intelligent_refusal()
        
        if result["test_passed"]:
            logger.info("🎉 Intelligent Message Refusal Test completed successfully!")
        else:
            logger.error("❌ Intelligent Message Refusal Test failed")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"test_passed": False, "error": str(e)}

if __name__ == "__main__":
    asyncio.run(main())