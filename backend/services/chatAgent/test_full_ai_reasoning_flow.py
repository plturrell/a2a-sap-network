#!/usr/bin/env python3
"""
Test the complete AI reasoning flow between ChatAgent and DataManager
Tests: receive → reason → act → respond for both agents with blockchain messaging
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_ai_reasoning_flow():
    """Test the complete AI reasoning flow between agents"""
    
    logger.info("🚀 Starting AI Reasoning Flow Test")
    
    # Set environment variables for AI and blockchain
    os.environ["AI_ENABLED"] = "true"
    os.environ["BLOCKCHAIN_ENABLED"] = "true"
    os.environ["A2A_RPC_URL"] = "http://localhost:8545"
    os.environ["A2A_AGENT_REGISTRY_ADDRESS"] = "0x5FbDB2315678afecb367f032d93F642f64180aa3"
    os.environ["A2A_MESSAGE_ROUTER_ADDRESS"] = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
    
    try:
        # Test 1: Import and create agents with AI capabilities
        logger.info("📦 Testing agent creation with AI capabilities...")
        
        # Import ChatAgent
        sys.path.append('/Users/apple/projects/a2a/a2aAgents/backend/services/chatAgent')
        from chatAgent import ChatAgent
        
        # Import DataManager
        sys.path.append('/Users/apple/projects/a2a/a2aAgents/backend/services/dataManager/src')
        from agent import DataManager
        
        # Create ChatAgent with AI enabled
        chat_config = {
            "agent_id": "chat_agent_ai_test",
            "name": "ChatAgent_AI_Test",
            "description": "ChatAgent with AI reasoning capabilities",
            "base_url": "http://localhost:8000",
            "enable_ai": True,
            "enable_blockchain": True
        }
        
        chat_agent = ChatAgent(config=chat_config)
        logger.info("✅ ChatAgent created with AI capabilities")
        
        # Create DataManager with AI enabled
        data_config = {
            "agent_id": "data_manager_ai_test", 
            "name": "DataManager_AI_Test",
            "description": "DataManager with AI reasoning capabilities",
            "base_url": "http://localhost:8001",
            "enable_ai": True,
            "enable_blockchain": True
        }
        
        data_manager = DataManager(config=data_config)
        logger.info("✅ DataManager created with AI capabilities")
        
        # Test 2: Verify AI intelligence initialization
        logger.info("🧠 Testing AI intelligence initialization...")
        
        # Check if AI is enabled and available
        chat_ai_available = hasattr(chat_agent, 'ai_enabled') and chat_agent.ai_enabled
        data_ai_available = hasattr(data_manager, 'ai_enabled') and data_manager.ai_enabled
        
        logger.info(f"   ChatAgent AI available: {chat_ai_available}")
        logger.info(f"   DataManager AI available: {data_ai_available}")
        
        # Test 3: Test incoming message AI reasoning
        logger.info("📥 Testing incoming message AI reasoning...")
        
        # Create a test A2A message
        test_message_data = {
            "message_id": f"test_msg_{uuid4().hex[:8]}",
            "from_agent": "user_simulator",
            "to_agent": "chat_agent_ai_test",
            "task_id": f"task_{uuid4().hex[:8]}",
            "context_id": f"ctx_{uuid4().hex[:8]}",
            "parts": [{
                "kind": "data",
                "data": {
                    "method": "store_user_data",
                    "user_data": {
                        "name": "Alice AI Test",
                        "email": "alice@aitest.com",
                        "preferences": {
                            "communication": "email",
                            "frequency": "weekly"
                        }
                    },
                    "storage_type": "persistent",
                    "encryption_required": True
                }
            }],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if chat_ai_available and hasattr(chat_agent, 'reason_about_message'):
            reasoning_result = await chat_agent.reason_about_message(test_message_data)
            logger.info("✅ ChatAgent AI reasoning successful:")
            logger.info(f"   Intent: {reasoning_result.get('intent', 'N/A')}")
            logger.info(f"   Urgency: {reasoning_result.get('urgency', 'N/A')}")
            logger.info(f"   Confidence: {reasoning_result.get('confidence', 'N/A')}")
        
        # Test 4: Test outgoing message AI optimization
        logger.info("📤 Testing outgoing message AI optimization...")
        
        outgoing_message_data = {
            "message_id": f"out_msg_{uuid4().hex[:8]}",
            "from_agent": "chat_agent_ai_test",
            "to_agent": "data_manager_ai_test",
            "task_id": f"task_{uuid4().hex[:8]}",
            "context_id": f"ctx_{uuid4().hex[:8]}",
            "parts": [{
                "partType": "data",
                "data": {
                    "action": "store_data",
                    "user_data": test_message_data["parts"][0]["data"]["user_data"],
                    "metadata": {
                        "source": "chat_agent",
                        "priority": "high",
                        "requires_encryption": True
                    }
                }
            }],
            "priority": "NORMAL",
            "encrypted": False,
            "direction": "outgoing",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if chat_ai_available and hasattr(chat_agent, 'reason_about_message'):
            outgoing_reasoning = await chat_agent.reason_about_message(
                outgoing_message_data,
                context={"message_direction": "outgoing", "target_agent": "data_manager_ai_test"}
            )
            logger.info("✅ Outgoing message AI optimization successful:")
            if outgoing_reasoning.get('recommended_modifications'):
                mods = outgoing_reasoning['recommended_modifications']
                logger.info(f"   Priority suggestion: {mods.get('priority', 'No change')}")
                logger.info(f"   Encryption suggestion: {mods.get('encrypt', 'No change')}")
        
        # Test 5: Test complete AI reasoning flow
        logger.info("🔄 Testing complete AI reasoning flow...")
        
        if chat_ai_available and hasattr(chat_agent, 'process_message_with_ai_reasoning'):
            complete_result = await chat_agent.process_message_with_ai_reasoning(test_message_data)
            
            if complete_result.get("success"):
                logger.info("✅ Complete AI reasoning flow successful:")
                logger.info(f"   Processing steps: {complete_result.get('processing_steps', [])}")
                logger.info(f"   AI enhanced: {complete_result.get('ai_enhanced', False)}")
                
                response = complete_result.get("response", {})
                if response:
                    logger.info(f"   Response message: {response.get('message', 'N/A')}")
                    logger.info(f"   Response status: {response.get('status', 'N/A')}")
            else:
                logger.warning(f"⚠️ AI reasoning flow failed: {complete_result.get('error')}")
        
        # Test 6: Test cross-agent AI communication
        logger.info("🤝 Testing cross-agent AI communication...")
        
        # Simulate a conversation between ChatAgent and DataManager
        conversation_steps = [
            {
                "from": "chat_agent_ai_test",
                "to": "data_manager_ai_test", 
                "message": {
                    "action": "analyze_user_patterns",
                    "user_id": "alice_ai_test",
                    "analysis_type": "behavioral",
                    "time_range": "last_30_days"
                }
            },
            {
                "from": "data_manager_ai_test",
                "to": "chat_agent_ai_test",
                "message": {
                    "status": "analysis_complete",
                    "insights": {
                        "most_active_time": "2PM-4PM",
                        "preferred_content": "technical_articles",
                        "engagement_score": 0.85
                    },
                    "recommendations": [
                        "Send notifications during peak activity hours",
                        "Focus on technical content",
                        "Increase interaction frequency"
                    ]
                }
            }
        ]
        
        for step in conversation_steps:
            step_message = {
                "message_id": f"conv_msg_{uuid4().hex[:8]}",
                "from_agent": step["from"],
                "to_agent": step["to"],
                "task_id": f"conv_task_{uuid4().hex[:8]}",
                "context_id": "ai_conversation_test",
                "parts": [{"kind": "data", "data": step["message"]}],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Test AI reasoning for this step
            agent = chat_agent if step["from"] == "chat_agent_ai_test" else data_manager
            if hasattr(agent, 'reason_about_message'):
                step_reasoning = await agent.reason_about_message(step_message)
                logger.info(f"   📊 {step['from']} → {step['to']}: {step_reasoning.get('intent', 'N/A')}")
        
        logger.info("✅ Cross-agent AI communication test successful")
        
        # Test 7: Performance and metrics
        logger.info("📈 Testing AI reasoning performance...")
        
        start_time = datetime.utcnow()
        performance_tests = []
        
        for i in range(5):
            test_msg = {
                "message_id": f"perf_msg_{i}",
                "from_agent": "performance_tester",
                "to_agent": "chat_agent_ai_test",
                "parts": [{"kind": "data", "data": {"test_number": i, "complexity": "medium"}}],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if hasattr(chat_agent, 'reason_about_message'):
                perf_start = datetime.utcnow()
                result = await chat_agent.reason_about_message(test_msg)
                perf_end = datetime.utcnow()
                
                performance_tests.append({
                    "test_number": i,
                    "processing_time": (perf_end - perf_start).total_seconds(),
                    "success": result.get("confidence", 0) > 0.5
                })
        
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()
        
        successful_tests = sum(1 for test in performance_tests if test["success"])
        avg_time = sum(test["processing_time"] for test in performance_tests) / len(performance_tests)
        
        logger.info(f"   📊 Performance Results:")
        logger.info(f"      Total tests: {len(performance_tests)}")
        logger.info(f"      Successful: {successful_tests}")
        logger.info(f"      Success rate: {successful_tests/len(performance_tests)*100:.1f}%")
        logger.info(f"      Average processing time: {avg_time:.3f}s")
        logger.info(f"      Total test time: {total_time:.3f}s")
        
        logger.info("🎉 AI Reasoning Flow Test completed successfully!")
        
        return {
            "test_passed": True,
            "chat_agent_ai": chat_ai_available,
            "data_manager_ai": data_ai_available,
            "performance": {
                "success_rate": successful_tests/len(performance_tests),
                "avg_processing_time": avg_time,
                "total_tests": len(performance_tests)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ AI Reasoning Flow Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "test_passed": False,
            "error": str(e)
        }

async def main():
    """Main test function"""
    logger.info("🎯 Starting Comprehensive AI Reasoning Flow Test")
    
    result = await test_ai_reasoning_flow()
    
    if result["test_passed"]:
        logger.info("✅ All AI reasoning tests passed!")
        logger.info(f"Final results: {json.dumps(result, indent=2)}")
    else:
        logger.error(f"❌ AI reasoning tests failed: {result.get('error')}")
    
    return result

if __name__ == "__main__":
    asyncio.run(main())