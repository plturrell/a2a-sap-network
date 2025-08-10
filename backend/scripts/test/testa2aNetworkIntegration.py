#!/usr/bin/env python3
"""
Test A2A Network Integration
Tests the integration between A2A Developer Portal and A2A Network
"""

import asyncio
import httpx
import json
from datetime import datetime

# Test configuration
PORTAL_BASE_URL = "http://localhost:3001"
A2A_NETWORK_API = f"{PORTAL_BASE_URL}/api/a2a-network"

# Test data
TEST_NETWORK_CONFIG = {
    "network": "local",
    "rpc_url": "http://localhost:8545",
    "private_key": None,  # Will use default from environment
    "websocket_url": None
}

TEST_AGENT = {
    "name": "Test Portal Agent",
    "description": "Agent created from Developer Portal integration test",
    "endpoint": "http://localhost:8000/agents/test-portal",
    "capabilities": {
        "messaging": True,
        "dataProcessing": True,
        "workflow": True,
        "analytics": False,
        "ai": False
    },
    "metadata": json.dumps({
        "version": "1.0.0",
        "created_by": "Portal Integration Test"
    })
}

TEST_WEBHOOK = {
    "event_type": "agent_registered",
    "webhook_url": "http://localhost:8000/webhooks/test",
    "filters": {},
    "active": True
}

async def test_connection():
    """Test connection to A2A Network"""
    print("\n1. Testing A2A Network Connection...")
    
    async with httpx.AsyncClient() as client:
        # Check initial status
        response = await client.get(f"{A2A_NETWORK_API}/status")
        print(f"Initial status: {response.json()}")
        
        # Connect to network
        print(f"Connecting to {TEST_NETWORK_CONFIG['network']} network...")
        response = await client.post(
            f"{A2A_NETWORK_API}/connect",
            json=TEST_NETWORK_CONFIG
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Connected successfully!")
            print(f"   Network: {data['network']}")
            print(f"   Chain ID: {data['chain_id']}")
            print(f"   Address: {data['address']}")
            print(f"   Contracts: {data['contracts']}")
            return True
        else:
            print(f"❌ Connection failed: {response.text}")
            return False

async def test_agent_management():
    """Test agent management features"""
    print("\n2. Testing Agent Management...")
    
    async with httpx.AsyncClient() as client:
        # Register agent
        print("Registering test agent...")
        response = await client.post(
            f"{A2A_NETWORK_API}/agents/register",
            json=TEST_AGENT
        )
        
        if response.status_code == 200:
            data = response.json()
            agent_id = data['agent_id']
            print(f"✅ Agent registered!")
            print(f"   Agent ID: {agent_id}")
            print(f"   Transaction: {data['transaction_hash']}")
            
            # Get agent details
            print(f"\nFetching agent details...")
            response = await client.get(f"{A2A_NETWORK_API}/agents/{agent_id}")
            if response.status_code == 200:
                agent = response.json()
                print(f"✅ Agent details retrieved!")
                print(f"   Name: {agent['name']}")
                print(f"   Owner: {agent['owner']}")
                print(f"   Status: {'Active' if agent['isActive'] else 'Inactive'}")
                
                # Get agent profile
                print(f"\nFetching agent profile...")
                response = await client.get(f"{A2A_NETWORK_API}/agents/{agent_id}/profile")
                if response.status_code == 200:
                    profile = response.json()
                    print(f"✅ Agent profile retrieved!")
                    print(f"   Reputation Score: {profile['reputation']['score']}")
                    print(f"   Total Tasks: {profile['reputation']['totalTasks']}")
                
                return agent_id
            else:
                print(f"❌ Failed to get agent details: {response.text}")
        else:
            print(f"❌ Agent registration failed: {response.text}")
    
    return None

async def test_agent_search():
    """Test agent search functionality"""
    print("\n3. Testing Agent Search...")
    
    async with httpx.AsyncClient() as client:
        # List all agents
        print("Fetching all agents...")
        response = await client.get(f"{A2A_NETWORK_API}/agents?limit=10")
        
        if response.status_code == 200:
            data = response.json()
            agents = data['data']['agents']
            print(f"✅ Found {len(agents)} agents")
            
            for agent in agents[:3]:  # Show first 3
                print(f"   - {agent['name']} (ID: {agent['id']})")
            
            # Search by skills
            print("\nSearching agents with messaging capability...")
            response = await client.post(
                f"{A2A_NETWORK_API}/agents/search",
                json={
                    "skills": ["messaging"],
                    "limit": 5
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Found {len(data['data']['agents'])} agents with messaging skill")
        else:
            print(f"❌ Failed to fetch agents: {response.text}")

async def test_messaging(agent_id):
    """Test messaging functionality"""
    print("\n4. Testing Messaging...")
    
    if not agent_id:
        print("⚠️  Skipping messaging test - no agent ID available")
        return
    
    async with httpx.AsyncClient() as client:
        # Send test message
        message = {
            "recipient_id": agent_id,
            "content": "Hello from Portal Integration Test!",
            "message_type": "text",
            "metadata": {
                "test": True,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        print(f"Sending message to agent {agent_id}...")
        response = await client.post(
            f"{A2A_NETWORK_API}/messages/send",
            json=message
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Message sent!")
            print(f"   Message ID: {data['message_id']}")
            print(f"   Transaction: {data['transaction_hash']}")
            
            # Get message history
            print(f"\nFetching message history...")
            response = await client.get(f"{A2A_NETWORK_API}/messages/{agent_id}?limit=5")
            
            if response.status_code == 200:
                messages = response.json()
                print(f"✅ Retrieved {len(messages['data'])} messages")
        else:
            print(f"❌ Failed to send message: {response.text}")

async def test_webhooks():
    """Test webhook functionality"""
    print("\n5. Testing Webhooks...")
    
    async with httpx.AsyncClient() as client:
        # Subscribe to webhook
        print("Creating webhook subscription...")
        response = await client.post(
            f"{A2A_NETWORK_API}/webhooks/subscribe",
            json=TEST_WEBHOOK
        )
        
        if response.status_code == 200:
            data = response.json()
            subscription_id = data['subscription_id']
            print(f"✅ Webhook subscription created!")
            print(f"   Subscription ID: {subscription_id}")
            print(f"   Event Type: {TEST_WEBHOOK['event_type']}")
            
            # List subscriptions
            print("\nListing webhook subscriptions...")
            response = await client.get(f"{A2A_NETWORK_API}/webhooks/subscriptions")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Found {len(data['subscriptions'])} subscriptions")
                
                # Test webhook
                print(f"\nTesting webhook delivery...")
                response = await client.post(
                    f"{A2A_NETWORK_API}/webhooks/test/{subscription_id}",
                    json={"test_data": "Portal integration test"}
                )
                
                if response.status_code == 200:
                    print(f"✅ Test webhook sent!")
                
                # Delete subscription
                print(f"\nDeleting webhook subscription...")
                response = await client.delete(
                    f"{A2A_NETWORK_API}/webhooks/subscriptions/{subscription_id}"
                )
                
                if response.status_code == 200:
                    print(f"✅ Webhook subscription deleted")
        else:
            print(f"❌ Failed to create webhook: {response.text}")

async def test_analytics():
    """Test analytics endpoints"""
    print("\n6. Testing Analytics...")
    
    async with httpx.AsyncClient() as client:
        # Get network analytics
        print("Fetching network analytics...")
        response = await client.get(f"{A2A_NETWORK_API}/analytics/network")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Network analytics retrieved!")
            print(f"   Network Info: {data['network']}")
            
        # Get reputation leaderboard
        print("\nFetching reputation leaderboard...")
        response = await client.get(f"{A2A_NETWORK_API}/reputation/leaderboard?limit=5")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Reputation leaderboard retrieved!")
            if data['data']:
                print(f"   Top agents by reputation:")
                for i, agent in enumerate(data['data'][:3]):
                    print(f"   {i+1}. Agent {agent['agentId']} - Score: {agent['score']}")

async def test_disconnect():
    """Test disconnection"""
    print("\n7. Testing Disconnection...")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{A2A_NETWORK_API}/disconnect")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Disconnected successfully!")
            print(f"   Status: {data['status']}")
        else:
            print(f"❌ Disconnection failed: {response.text}")

async def main():
    """Run all integration tests"""
    print("=" * 60)
    print("A2A Network Integration Test Suite")
    print("=" * 60)
    
    try:
        # Test connection
        connected = await test_connection()
        
        if connected:
            # Test agent management
            agent_id = await test_agent_management()
            
            # Test agent search
            await test_agent_search()
            
            # Test messaging
            await test_messaging(agent_id)
            
            # Test webhooks
            await test_webhooks()
            
            # Test analytics
            await test_analytics()
            
            # Test disconnection
            await test_disconnect()
        
        print("\n" + "=" * 60)
        print("✅ Integration tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting A2A Network Integration Tests...")
    print("Make sure the Developer Portal is running on http://localhost:3001")
    print("And A2A Network is accessible\n")
    
    asyncio.run(main())