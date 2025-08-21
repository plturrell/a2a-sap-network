#!/usr/bin/env python3
"""
Test real communication between ChatAgent and DataManager
"""

import asyncio
import json
import os
from datetime import datetime
from uuid import uuid4

# Set environment variable for DataManager
os.environ['DATA_PROCESSOR_URL'] = os.getenv("A2A_SERVICE_URL")

# Import ChatAgent
from chatAgent import ChatAgent


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
async def test_real_communication():
    """Test real agent-to-agent communication"""
    print("🚀 Testing real ChatAgent to DataManager communication...\n")
    
    # Initialize ChatAgent
    config = {
        "environment": "development",
        "database": {
            "type": "sqlite",
            "connection_string": "sqlite+aiosqlite:///test_chat.db"
        }
    }
    
    chat_agent = ChatAgent(base_url=os.getenv("A2A_SERVICE_URL"), config=config)
    await chat_agent.initialize()
    
    print("✅ ChatAgent initialized successfully")
    print(f"   Agent ID: {chat_agent.agent_id}")
    print(f"   Data Processor URL: {chat_agent.agent_registry.get('data-processor', {}).get('endpoint')}\n")
    
    # Create a test message that should route to DataManager
    test_message = "Please store this customer information: Name is Bob Johnson, Email is bob@example.com, Customer ID is CUST-789"
    
    # Create A2A message
    from a2aCommon import A2AMessage, MessageRole
    
    a2a_message = A2AMessage(
        role=MessageRole.USER,
        content={
            'data': {
                'prompt': test_message,
                'user_id': 'test_user',
                'session_id': 'test_session'
            }
        },
        context_id=f"test_{int(datetime.now().timestamp())}"
    )
    
    print(f"📤 Sending message: {test_message}\n")
    
    # Process the message synchronously to see the result
    result = await chat_agent.handle_chat_message(a2a_message, a2a_message.context_id)
    
    print("📥 Initial Response:")
    print(json.dumps(result, indent=2))
    
    if result.get('success') and result.get('data', {}).get('task_id'):
        task_id = result['data']['task_id']
        print(f"\n⏳ Task created: {task_id}")
        
        # Wait a bit for processing
        await asyncio.sleep(2)
        
        # Check if the message was routed to DataManager
        # We'll check the routing stats
        print(f"\n📊 Routing Statistics:")
        print(f"   Total messages: {chat_agent.routing_stats.get('total_messages', 0)}")
        print(f"   Successful routings: {chat_agent.routing_stats.get('successful_routings', 0)}")
        print(f"   Failed routings: {chat_agent.routing_stats.get('failed_routings', 0)}")
        print(f"   Popular agents: {chat_agent.routing_stats.get('popular_agents', {})}")
        
        # Try to directly route to data-processor
        print("\n🔄 Testing direct routing to data-processor...")
        
        routing_result = await chat_agent.route_to_agent({
            "prompt": "Store this test data: key=test123, value=success",
            "target_agent": "data-processor",
            "context_id": "direct_test"
        })
        
        print("📥 Direct Routing Result:")
        print(json.dumps(routing_result, indent=2))
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_real_communication())