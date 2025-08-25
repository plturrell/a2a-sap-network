"""
Example usage of the A2A Chat Agent
"""

import asyncio
import logging
from typing import Dict, Any

from ..agents.chatAgent import ChatAgent
from ..sdk.types import A2AMessage, MessagePart, MessageRole

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Example demonstrating ChatAgent usage"""
    
    # Configuration for blockchain connection (optional)
    blockchain_config = {
        "rpc_url": "http://localhost:8545",
        "contract_addresses": {
            "message_router": "0x1234567890123456789012345678901234567890",
            "agent_registry": "0x0987654321098765432109876543210987654321"
        }
        # Add private_key if you want to send transactions
    }
    
    # Initialize ChatAgent
    chat_agent = ChatAgent(
        agent_id="chat-agent-example",
        name="Example Chat Agent",
        description="Example A2A Chat Agent for demonstrations",
        base_url="http://localhost:8001",
        blockchain_config=blockchain_config
    )
    
    await chat_agent.initialize()
    
    try:
        # Example 1: Send a simple prompt to a specific agent
        print("\n=== Example 1: Send prompt to specific agent ===")
        result1 = await chat_agent.send_prompt_to_agent({
            "prompt": "Analyze the performance data from last week",
            "target_agent": "data-processor",
            "use_blockchain": False  # Use HTTP for demo
        })
        print(f"Result: {result1}")
        
        # Example 2: Handle a chat message (simulates user interaction)
        print("\n=== Example 2: Handle chat message ===")
        user_message = A2AMessage(
            role=MessageRole.USER,
            parts=[MessagePart(kind="text", text="I need help analyzing some cryptocurrency trading data")]
        )
        
        result2 = await chat_agent.handle_chat_message(user_message, "conversation-123")
        print(f"Chat response: {result2}")
        
        # Example 3: Start a group conversation
        print("\n=== Example 3: Start group conversation ===")
        result3 = await chat_agent.start_conversation({
            "participants": ["data-processor", "crypto-trader", "analytics-agent"],
            "initial_message": "Let's collaborate on analyzing the latest market trends",
            "conversation_type": "group"
        })
        print(f"Conversation started: {result3}")
        
        # Example 4: Broadcast a message to agents with specific capabilities
        print("\n=== Example 4: Broadcast message ===")
        result4 = await chat_agent.broadcast_message({
            "message": "Please report your current status and any anomalies",
            "filter_by_capability": ["analysis", "monitoring"],
            "max_agents": 3
        })
        print(f"Broadcast result: {result4}")
        
        # Example 5: List available agents
        print("\n=== Example 5: List available agents ===")
        result5 = await chat_agent.list_available_agents({})
        print(f"Available agents: {result5}")
        
        # Example 6: Get conversation history
        print("\n=== Example 6: Get conversation history ===")
        history_message = A2AMessage(
            role=MessageRole.SYSTEM,
            parts=[MessagePart(kind="data", data={"method": "get_conversation_history"})]
        )
        result6 = await chat_agent.get_conversation_history(history_message, "conversation-123")
        print(f"Conversation history: {result6}")
        
    finally:
        await chat_agent.shutdown()


async def blockchain_example():
    """Example demonstrating blockchain integration"""
    
    # This example requires a running blockchain node and deployed contracts
    blockchain_config = {
        "rpc_url": "http://localhost:8545",
        "private_key": "0x..." # Your private key for sending transactions
        "contract_addresses": {
            "message_router": "0x...",  # Deployed MessageRouter contract
            "agent_registry": "0x...",  # Deployed AgentRegistry contract
        }
    }
    
    chat_agent = ChatAgent(
        agent_id="blockchain-chat-agent",
        blockchain_config=blockchain_config
    )
    
    await chat_agent.initialize()
    
    try:
        # Send message via blockchain
        result = await chat_agent.send_prompt_to_agent({
            "prompt": "Process this data on the blockchain",
            "target_agent": "data-processor",
            "use_blockchain": True  # This will use blockchain messaging
        })
        print(f"Blockchain message result: {result}")
        
    except Exception as e:
        logger.error(f"Blockchain example failed: {e}")
        print("Note: This example requires a running blockchain node with deployed A2A contracts")
    
    finally:
        await chat_agent.shutdown()


def create_fastapi_server():
    """Create a FastAPI server for the ChatAgent"""
    
    chat_agent = ChatAgent(
        agent_id="chat-api-server",
        base_url="http://localhost:8002"
    )
    
    # Get the FastAPI app with standard A2A endpoints
    app = chat_agent.create_fastapi_app()
    
    # Add custom chat endpoints
    @app.post("/chat")
    async def chat_endpoint(request: dict):
        """Custom chat endpoint for easy integration"""
        message = A2AMessage(
            role=MessageRole.USER,
            parts=[MessagePart(kind="text", text=request.get("message", ""))]
        )
        
        context_id = request.get("conversation_id", "default")
        result = await chat_agent.handle_chat_message(message, context_id)
        
        return {
            "response": result.get("response", ""),
            "success": result.get("success", False),
            "conversation_id": context_id,
            "routed_to": result.get("routed_to", [])
        }
    
    @app.post("/send-to-agent")
    async def send_to_agent_endpoint(request: dict):
        """Direct agent communication endpoint"""
        result = await chat_agent.send_prompt_to_agent(request)
        return result
    
    return app


if __name__ == "__main__":
    # Run the basic example
    print("Running A2A ChatAgent Example...")
    asyncio.run(main())
    
    # Uncomment to run blockchain example
    # asyncio.run(blockchain_example())
    
    # To run as a server:
    # app = create_fastapi_server()
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8002)