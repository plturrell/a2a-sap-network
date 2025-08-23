#!/usr/bin/env python3
"""
Test script for the A2A Chat Agent - demonstrates real communication with all 16 agents
"""

import asyncio
import json
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chatAgent import create_chat_agent
from a2aCommon import A2AMessage, MessageRole


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_chat_agent():
    """Test the Chat Agent with various prompts"""
    
    # Create agent
    agent = create_chat_agent(
        base_url=os.getenv("A2A_SERVICE_URL"),
        config={
            "enable_blockchain": False,
            "enable_persistence": True
        }
    )
    
    # Initialize
    await agent.initialize()
    
    print("\n=== A2A Chat Agent Test Suite ===\n")
    
    # Test 1: Simple data analysis request
    print("Test 1: Data Analysis Request")
    print("-" * 50)
    
    message1 = A2AMessage(
        role=MessageRole.USER,
        content={
            "data": {
                "prompt": "Analyze the sales data for Q4 and identify trends",
                "user_id": "test_user_1",
                "conversation_id": "conv_001"
            }
        },
        context_id="test_context_1"
    )
    
    response1 = await agent.handle_chat_message(message1, "test_context_1")
    print(f"Response: {json.dumps(response1, indent=2)}")
    
    # Wait for async processing
    await asyncio.sleep(2)
    
    # Test 2: Multi-agent query
    print("\n\nTest 2: Multi-Agent Coordination")
    print("-" * 50)
    
    message2 = A2AMessage(
        role=MessageRole.USER,
        content={
            "data": {
                "query": "Scrape cryptocurrency prices and analyze market trends",
                "target_agents": ["web-scraper", "crypto-trader", "analytics-agent"],
                "coordination_type": "parallel"
            }
        },
        context_id="test_context_2"
    )
    
    response2 = await agent.handle_multi_agent_query(message2, "test_context_2")
    print(f"Response: {json.dumps(response2, indent=2)}")
    
    # Test 3: Complex workflow
    print("\n\nTest 3: Complex Workflow Request")
    print("-" * 50)
    
    message3 = A2AMessage(
        role=MessageRole.USER,
        content={
            "data": {
                "prompt": "Schedule a daily backup of my database and send notifications when complete",
                "user_id": "test_user_1",
                "conversation_id": "conv_002"
            }
        },
        context_id="test_context_3"
    )
    
    response3 = await agent.handle_chat_message(message3, "test_context_3")
    print(f"Response: {json.dumps(response3, indent=2)}")
    
    # Test 4: Intent analysis
    print("\n\nTest 4: Intent Analysis")
    print("-" * 50)
    
    prompts = [
        "Translate this document to Spanish and analyze the sentiment",
        "Check my code for security vulnerabilities",
        "Create a machine learning model to predict sales",
        "Monitor my API endpoints and alert on failures",
        "Generate a comprehensive analytics dashboard"
    ]
    
    for prompt in prompts:
        result = await agent.analyze_intent({"prompt": prompt})
        print(f"\nPrompt: '{prompt}'")
        print(f"Recommended agents: {result['recommended_agents']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Intent type: {result['intent_type']}")
    
    # Test 5: Direct agent routing
    print("\n\nTest 5: Direct Agent Routing")
    print("-" * 50)
    
    # Route to specific agents
    agents_to_test = [
        ("data-processor", "Calculate the average of these numbers: 10, 20, 30, 40"),
        ("nlp-agent", "What is the sentiment of this text: 'I love this product!'"),
        ("security-agent", "Scan my system for vulnerabilities"),
        ("ml-agent", "Train a model to classify customer feedback"),
        ("workflow-agent", "Create a pipeline for data processing")
    ]
    
    for agent_id, prompt in agents_to_test:
        try:
            result = await agent.route_to_agent({
                "prompt": prompt,
                "target_agent": agent_id,
                "context_id": f"direct_test_{agent_id}"
            })
            print(f"\n{agent_id}: {prompt}")
            print(f"Success: {result['success']}")
            if result['success']:
                print(f"Response: {result['response']}")
        except Exception as e:
            print(f"Error routing to {agent_id}: {e}")
    
    # Test 6: Conversation history
    print("\n\nTest 6: Conversation History")
    print("-" * 50)
    
    history = await agent.get_conversation_history({
        "conversation_id": "conv_001",
        "user_id": "test_user_1",
        "limit": 10
    })
    print(f"Conversation history: {json.dumps(history, indent=2)}")
    
    # Test 7: Active conversations (MCP resource)
    print("\n\nTest 7: Active Conversations")
    print("-" * 50)
    
    if hasattr(agent, 'get_active_conversations_resource'):
        active = await agent.get_active_conversations_resource()
        print(f"Active conversations: {json.dumps(active, indent=2)}")
    
    # Display routing statistics
    print("\n\nRouting Statistics")
    print("-" * 50)
    print(f"Total messages: {agent.routing_stats['total_messages']}")
    print(f"Successful routings: {agent.routing_stats['successful_routings']}")
    print(f"Failed routings: {agent.routing_stats['failed_routings']}")
    print(f"Popular agents: {agent.routing_stats['popular_agents']}")
    
    # Shutdown
    await agent.shutdown()
    
    print("\n=== Test Complete ===")


async def test_real_agent_communication():
    """Test real communication with actual running agents"""
    
    print("\n=== Testing Real Agent Communication ===\n")
    
    # This assumes other agents are running on their respective ports
    agent = create_chat_agent(
        base_url=os.getenv("A2A_SERVICE_URL"),
        config={
            "enable_blockchain": False,
            "enable_persistence": True
        }
    )
    
    await agent.initialize()
    
    # Test prompts that should route to different agents
    test_cases = [
        {
            "prompt": "Analyze this dataset: [1,2,3,4,5,6,7,8,9,10] and calculate mean, median, mode",
            "expected_agent": "data-processor"
        },
        {
            "prompt": "Translate 'Hello World' to Spanish, French, and German",
            "expected_agent": "nlp-agent"
        },
        {
            "prompt": "What's the current Bitcoin price and market trend?",
            "expected_agent": "crypto-trader"
        },
        {
            "prompt": "Create a backup of my important files",
            "expected_agent": "backup-agent"
        },
        {
            "prompt": "Scan this code for security issues: function test() { eval(userInput); }",
            "expected_agent": "code-reviewer"
        },
        {
            "prompt": "Create a workflow to process images and extract text",
            "expected_agent": "workflow-agent"
        },
        {
            "prompt": "Schedule a task to run every day at 9 AM",
            "expected_agent": "scheduler-agent"
        },
        {
            "prompt": "Send me an email notification when the task completes",
            "expected_agent": "notification-agent"
        },
        {
            "prompt": "Query the database for all users created this month",
            "expected_agent": "database-agent"
        },
        {
            "prompt": "Create an API endpoint for user management",
            "expected_agent": "api-agent"
        },
        {
            "prompt": "Generate analytics report for website traffic",
            "expected_agent": "analytics-agent"
        },
        {
            "prompt": "Train a machine learning model for spam detection",
            "expected_agent": "ml-agent"
        },
        {
            "prompt": "Extract data from https://example.com",
            "expected_agent": "web-scraper"
        },
        {
            "prompt": "Analyze this image and extract any text",
            "expected_agent": "image-processor"
        },
        {
            "prompt": "Check for security vulnerabilities in my system",
            "expected_agent": "security-agent"
        },
        {
            "prompt": "Upload and organize these documents in the cloud",
            "expected_agent": "file-manager"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test_case['expected_agent']}")
        print("-" * 50)
        print(f"Prompt: {test_case['prompt']}")
        
        # Send message
        message = A2AMessage(
            role=MessageRole.USER,
            content={
                "data": {
                    "prompt": test_case['prompt'],
                    "user_id": "test_user",
                    "conversation_id": f"test_conv_{i}"
                }
            },
            context_id=f"test_{i}"
        )
        
        response = await agent.handle_chat_message(message, f"test_{i}")
        print(f"Initial response: {response}")
        
        # Wait for processing
        await asyncio.sleep(1)
        
        # Check task status
        if 'task_id' in response:
            task_status = agent.get_task_status(response['task_id'])
            if task_status:
                print(f"Task status: {task_status['status']}")
                if task_status['status'] == 'completed':
                    print(f"Routed to: {task_status.get('result', {}).get('routed_to', [])}")
    
    await agent.shutdown()
    print("\n=== Real Communication Test Complete ===")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_chat_agent())
    
    # Uncomment to test real agent communication
    # asyncio.run(test_real_agent_communication())