#!/usr/bin/env python3
"""
Test script to validate message conversion between blockchain and A2A formats
This test validates the critical message conversion fixes made to Agent 17
"""

import asyncio
import json
import sys
import os
from typing import Dict, Any
from uuid import uuid4

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

from a2aAgents.backend.app.a2a.sdk.types import A2AMessage, MessagePart, MessageRole
from a2aAgents.backend.app.a2a.agents.agent17ChatAgent.active.agent17ChatAgentSdk import Agent17ChatAgent

async def test_blockchain_to_a2a_conversion():
    """Test blockchain message to A2A message conversion"""
    print("Testing blockchain to A2A message conversion...")
    
    agent = Agent17ChatAgent()
    
    # Test 1: JSON string content
    blockchain_msg_1 = {
        "id": "test_msg_1",
        "content": json.dumps({
            "prompt": "Hello, what's the weather?",
            "context_id": "test_context_1",
            "from_agent": "user_agent"
        }),
        "task_id": "task_123"
    }
    
    a2a_msg_1 = agent._blockchain_to_a2a_message(blockchain_msg_1)
    
    # Validate conversion
    assert a2a_msg_1.messageId == "test_msg_1"
    assert a2a_msg_1.role == MessageRole.USER
    assert len(a2a_msg_1.parts) == 1
    assert a2a_msg_1.parts[0].kind == "data"
    assert a2a_msg_1.parts[0].data["prompt"] == "Hello, what's the weather?"
    assert a2a_msg_1.parts[0].text == "Hello, what's the weather?"
    assert a2a_msg_1.taskId == "task_123"
    assert a2a_msg_1.contextId == "test_context_1"
    
    print("✓ Test 1 passed: JSON string content conversion")
    
    # Test 2: Raw string content
    blockchain_msg_2 = {
        "id": "test_msg_2",
        "content": "Simple text message",
        "context_id": "test_context_2"
    }
    
    a2a_msg_2 = agent._blockchain_to_a2a_message(blockchain_msg_2)
    
    assert a2a_msg_2.messageId == "test_msg_2"
    assert a2a_msg_2.parts[0].data["raw_content"] == "Simple text message"
    assert a2a_msg_2.parts[0].text == "Simple text message"
    assert a2a_msg_2.contextId == "test_context_2"
    
    print("✓ Test 2 passed: Raw string content conversion")
    
    # Test 3: Malformed JSON (error handling)
    blockchain_msg_3 = {
        "content": "{invalid json",
        "context_id": "test_context_3"
    }
    
    a2a_msg_3 = agent._blockchain_to_a2a_message(blockchain_msg_3)
    
    assert a2a_msg_3.parts[0].data["raw_content"] == "{invalid json"
    assert a2a_msg_3.contextId == "test_context_3"
    
    print("✓ Test 3 passed: Malformed JSON error handling")
    
    # Test 4: Missing fields (should generate UUID)
    blockchain_msg_4 = {
        "content": "Test message"
    }
    
    a2a_msg_4 = agent._blockchain_to_a2a_message(blockchain_msg_4)
    
    assert a2a_msg_4.messageId is not None
    assert len(a2a_msg_4.messageId) > 0
    assert a2a_msg_4.contextId is not None
    
    print("✓ Test 4 passed: Missing fields handled with defaults")
    
    print("🎉 All blockchain to A2A conversion tests passed!")
    return True

async def test_a2a_to_blockchain_routing():
    """Test A2A message routing to blockchain format"""
    print("Testing A2A to blockchain message routing...")
    
    # Mock blockchain client for testing
    class MockBlockchainClient:
        def __init__(self):
            self.sent_messages = []
        
        async def send_message(self, to_address: str, content: str, message_type: str):
            # Validate parameters match expected blockchain client interface
            assert isinstance(to_address, str), "to_address must be string"
            assert isinstance(content, str), "content must be JSON string"
            assert isinstance(message_type, str), "message_type must be string"
            
            # Parse content to validate it's valid JSON
            try:
                parsed_content = json.loads(content)
                assert isinstance(parsed_content, dict), "Content must be JSON object"
            except json.JSONDecodeError:
                raise ValueError("Content must be valid JSON string")
            
            self.sent_messages.append({
                "to_address": to_address,
                "content": content,
                "message_type": message_type,
                "parsed_content": parsed_content
            })
            
            return f"tx_hash_{len(self.sent_messages)}"
    
    agent = Agent17ChatAgent()
    agent.blockchain_client = MockBlockchainClient()
    
    # Test routing with proper parameter conversion
    prompt = "Analyze this data for insights"
    target_agents = ["agent9_reasoning", "agent5_data"]
    context_id = "test_context_routing"
    
    results = await agent._route_via_blockchain(prompt, target_agents, context_id)
    
    # Validate results
    assert len(results) == 2, "Should have 2 routing results"
    
    # Check first message
    sent_msg_1 = agent.blockchain_client.sent_messages[0]
    assert sent_msg_1["to_address"] == "agent9_reasoning"
    assert sent_msg_1["message_type"] == "a2a_agent_request"
    
    # Validate message content structure
    parsed_1 = sent_msg_1["parsed_content"]
    assert parsed_1["operation"] == "chat_message"
    assert parsed_1["prompt"] == prompt
    assert parsed_1["context_id"] == context_id
    assert parsed_1["from_agent"] == agent.AGENT_ID
    
    print("✓ Test 1 passed: Proper parameter names (to_address, not to_agent)")
    print("✓ Test 2 passed: Content serialized to JSON string")
    print("✓ Test 3 passed: Message structure preserved")
    
    # Check statistics updated
    assert agent.stats["blockchain_messages_sent"] == 2
    assert agent.stats["successful_routings"] == 2
    
    print("✓ Test 4 passed: Statistics updated correctly")
    
    print("🎉 All A2A to blockchain routing tests passed!")
    return True

async def test_message_format_compatibility():
    """Test compatibility between message formats"""
    print("Testing message format compatibility...")
    
    agent = Agent17ChatAgent()
    
    # Create a complex message that goes through full conversion cycle
    original_data = {
        "prompt": "Complex query with multiple parameters",
        "context_id": "complex_context",
        "user_id": "test_user",
        "metadata": {
            "priority": "high",
            "requires_ai": True,
            "encrypt": False
        }
    }
    
    # Simulate blockchain message format
    blockchain_msg = {
        "id": "complex_msg_id",
        "content": json.dumps(original_data),
        "task_id": "complex_task",
        "context_id": "blockchain_context"  # This should be overridden by content
    }
    
    # Convert to A2A format
    a2a_msg = agent._blockchain_to_a2a_message(blockchain_msg)
    
    # Validate data integrity
    assert a2a_msg.messageId == "complex_msg_id"
    assert a2a_msg.taskId == "complex_task"
    assert a2a_msg.contextId == "complex_context"  # From content, not blockchain level
    
    # Validate all original data preserved
    msg_data = a2a_msg.parts[0].data
    assert msg_data["prompt"] == original_data["prompt"]
    assert msg_data["context_id"] == original_data["context_id"]
    assert msg_data["user_id"] == original_data["user_id"]
    assert msg_data["metadata"]["priority"] == "high"
    assert msg_data["metadata"]["requires_ai"] is True
    
    print("✓ Test 1 passed: Complex data structure preserved")
    
    # Test edge case: Empty content
    empty_blockchain_msg = {
        "id": "empty_msg",
        "content": ""
    }
    
    empty_a2a_msg = agent._blockchain_to_a2a_message(empty_blockchain_msg)
    assert empty_a2a_msg.messageId == "empty_msg"
    assert empty_a2a_msg.parts[0].data["raw_content"] == ""
    
    print("✓ Test 2 passed: Empty content handled correctly")
    
    # Test Unicode and special characters
    unicode_data = {
        "prompt": "Test with émojis 🚀 and spëcial chars: àáâãäå",
        "language": "multibyte"
    }
    
    unicode_blockchain_msg = {
        "content": json.dumps(unicode_data, ensure_ascii=False)
    }
    
    unicode_a2a_msg = agent._blockchain_to_a2a_message(unicode_blockchain_msg)
    recovered_data = unicode_a2a_msg.parts[0].data
    
    assert recovered_data["prompt"] == unicode_data["prompt"]
    assert "🚀" in recovered_data["prompt"]
    assert "émojis" in recovered_data["prompt"]
    
    print("✓ Test 3 passed: Unicode and special characters preserved")
    
    print("🎉 All message format compatibility tests passed!")
    return True

async def main():
    """Run all message conversion tests"""
    print("=" * 60)
    print("A2A Agent 17 Message Conversion Validation Tests")
    print("=" * 60)
    
    try:
        # Run all test suites
        test1_passed = await test_blockchain_to_a2a_conversion()
        print()
        
        test2_passed = await test_a2a_to_blockchain_routing()
        print()
        
        test3_passed = await test_message_format_compatibility()
        print()
        
        # Final results
        if test1_passed and test2_passed and test3_passed:
            print("🎊 ALL TESTS PASSED! Message conversion is working correctly.")
            print("\n✅ VALIDATION SUMMARY:")
            print("  • Blockchain → A2A conversion: WORKING")
            print("  • A2A → Blockchain routing: WORKING")  
            print("  • Parameter names fixed: to_address ✓")
            print("  • Content serialization fixed: JSON string ✓")
            print("  • Field names fixed: camelCase ✓")
            print("  • Data integrity preserved: ✓")
            print("  • Error handling robust: ✓")
            return True
        else:
            print("❌ Some tests failed")
            return False
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)