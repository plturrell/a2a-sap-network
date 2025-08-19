"""
A2A Network Integration API
Connects the Developer Portal to A2A Network smart contracts
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from web3 import Web3
from eth_account import Account
import httpx

# Import A2A Network SDK
import sys
from a2a.client import A2AClient
from a2a.utils.errors import A2AError

logger = logging.getLogger(__name__)

# API Router
router = APIRouter(prefix="/api/a2a-network", tags=["A2A Network Integration"])

# Global A2A client instance
a2a_client: Optional[A2AClient] = None

# Webhook subscriptions
webhook_subscriptions = {}

# Request/Response Models
class NetworkConfig(BaseModel):
    """A2A Network configuration"""
    network: str = Field(default="mainnet", description="Network to connect to")
    rpc_url: Optional[str] = Field(None, description="Custom RPC URL")
    private_key: Optional[str] = Field(None, description="Private key for transactions")
    websocket_url: Optional[str] = Field(None, description="WebSocket URL for real-time updates")

class AgentRegistrationRequest(BaseModel):
    """Agent registration request"""
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    endpoint: str = Field(..., description="Agent endpoint URL")
    capabilities: Dict[str, bool] = Field(..., description="Agent capabilities")
    metadata: Optional[str] = Field("{}", description="Additional metadata JSON")

class MessageRequest(BaseModel):
    """Message sending request"""
    recipient_id: str = Field(..., description="Recipient agent ID")
    content: str = Field(..., description="Message content")
    message_type: str = Field(default="text", description="Message type")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Message metadata")

class WebhookSubscription(BaseModel):
    """Webhook subscription configuration"""
    event_type: str = Field(..., description="Event type to subscribe to")
    webhook_url: str = Field(..., description="URL to receive webhook calls")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Event filters")
    active: bool = Field(default=True, description="Subscription status")

class AgentSearchCriteria(BaseModel):
    """Agent search criteria"""
    skills: Optional[List[str]] = Field(None, description="Required skills")
    min_reputation: Optional[int] = Field(None, description="Minimum reputation score")
    max_response_time: Optional[int] = Field(None, description="Maximum response time")
    limit: int = Field(default=20, ge=1, le=100, description="Result limit")
    offset: int = Field(default=0, ge=0, description="Result offset")

# Initialization
async def initialize_a2a_client(config: NetworkConfig):
    """Initialize A2A Network client"""
    global a2a_client
    
    try:
        client_config = {
            "network": config.network,
            "api_timeout": 30,
            "retry_attempts": 3,
            "auto_reconnect": True
        }
        
        if config.rpc_url:
            client_config["rpc_url"] = config.rpc_url
        if config.private_key:
            client_config["private_key"] = config.private_key
        if config.websocket_url:
            client_config["websocket_url"] = config.websocket_url
        
        # Create and connect client
        a2a_client = A2AClient(client_config)
        await a2a_client.connect()
        
        # Setup event listeners for webhooks
        await setup_event_listeners()
        
        logger.info(f"Connected to A2A Network on {config.network}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize A2A client: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to connect to A2A Network: {str(e)}")

async def setup_event_listeners():
    """Setup event listeners for webhook notifications"""
    if not a2a_client:
        return
    
    # Agent events
    a2a_client.on('agent_registered', lambda data: asyncio.create_task(handle_webhook_event('agent_registered', data)))
    a2a_client.on('agent_updated', lambda data: asyncio.create_task(handle_webhook_event('agent_updated', data)))
    a2a_client.on('agent_status_changed', lambda data: asyncio.create_task(handle_webhook_event('agent_status_changed', data)))
    
    # Message events  
    a2a_client.on('message_sent', lambda data: asyncio.create_task(handle_webhook_event('message_sent', data)))
    a2a_client.on('message_received', lambda data: asyncio.create_task(handle_webhook_event('message_received', data)))
    
    # Network events
    a2a_client.on('agent_event', lambda data: asyncio.create_task(handle_webhook_event('agent_event', data)))
    a2a_client.on('message_event', lambda data: asyncio.create_task(handle_webhook_event('message_event', data)))

async def handle_webhook_event(event_type: str, data: Dict[str, Any]):
    """Handle webhook events and notify subscribers"""
    try:
        # Find active subscriptions for this event type
        for sub_id, subscription in webhook_subscriptions.items():
            if subscription['event_type'] == event_type and subscription['active']:
                # Apply filters if any
                if subscription.get('filters'):
                    if not match_filters(data, subscription['filters']):
                        continue
                
                # Send webhook notification
                await send_webhook_notification(subscription['webhook_url'], {
                    'event_type': event_type,
                    'data': data,
                    'timestamp': datetime.utcnow().isoformat(),
                    'subscription_id': sub_id
                })
                
    except Exception as e:
        logger.error(f"Error handling webhook event {event_type}: {e}")

async def send_webhook_notification(webhook_url: str, payload: Dict[str, Any]):
    """Send webhook notification to subscriber"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                webhook_url,
                json=payload,
                timeout=10.0,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code >= 400:
                logger.warning(f"Webhook delivery failed to {webhook_url}: {response.status_code}")
                
    except Exception as e:
        logger.error(f"Failed to send webhook to {webhook_url}: {e}")

def match_filters(data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Check if event data matches subscription filters"""
    for key, value in filters.items():
        if key not in data:
            return False
        if isinstance(value, list):
            if data[key] not in value:
                return False
        elif data[key] != value:
            return False
    return True

# API Endpoints

@router.post("/connect")
async def connect_to_network(config: NetworkConfig):
    """Connect to A2A Network"""
    try:
        await initialize_a2a_client(config)
        network_info = await a2a_client.get_network_info()
        
        return {
            "status": "connected",
            "network": config.network,
            "chain_id": network_info['chain_id'],
            "block_number": network_info['block_number'],
            "contracts": network_info['contracts'],
            "address": a2a_client.get_address()
        }
        
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/disconnect")
async def disconnect_from_network():
    """Disconnect from A2A Network"""
    global a2a_client
    
    if a2a_client:
        await a2a_client.disconnect()
        a2a_client = None
        return {"status": "disconnected"}
    
    return {"status": "not_connected"}

@router.get("/status")
async def get_connection_status():
    """Get current connection status"""
    if not a2a_client:
        return {
            "connected": False,
            "status": "disconnected"
        }
    
    try:
        health = await a2a_client.health_check()
        return {
            "connected": True,
            "status": health['status'],
            "details": health['details']
        }
    except Exception as e:
        return {
            "connected": False,
            "status": "error",
            "error": str(e)
        }

# Agent Management

@router.post("/agents/register")
async def register_agent(request: AgentRegistrationRequest):
    """Register a new agent on A2A Network"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        result = await a2a_client.agents.register({
            "name": request.name,
            "description": request.description,
            "endpoint": request.endpoint,
            "capabilities": request.capabilities,
            "metadata": request.metadata
        })
        
        return {
            "success": True,
            "agent_id": result['agentId'],
            "transaction_hash": result['transactionHash'],
            "message": f"Agent {request.name} registered successfully"
        }
        
    except A2AError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Agent registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents")
async def get_agents(limit: int = 20, offset: int = 0, search: Optional[str] = None):
    """Get agents from A2A Network"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        result = await a2a_client.agents.get_agents({
            "limit": limit,
            "offset": offset,
            "search": search
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get specific agent details"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        agent = await a2a_client.agents.get_agent(agent_id)
        return agent
        
    except Exception as e:
        logger.error(f"Failed to fetch agent {agent_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

@router.get("/agents/{agent_id}/profile")
async def get_agent_profile(agent_id: str):
    """Get agent profile with reputation"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        profile = await a2a_client.agents.get_agent_profile(agent_id)
        return profile
        
    except Exception as e:
        logger.error(f"Failed to fetch agent profile {agent_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Agent profile not found")

@router.put("/agents/{agent_id}")
async def update_agent(agent_id: str, updates: Dict[str, Any]):
    """Update agent information"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        result = await a2a_client.agents.update(agent_id, updates)
        return {
            "success": True,
            "transaction_hash": result['transactionHash'],
            "message": f"Agent {agent_id} updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to update agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/agents/{agent_id}/status")
async def set_agent_status(agent_id: str, is_active: bool):
    """Set agent active/inactive status"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        result = await a2a_client.agents.set_status(agent_id, is_active)
        return {
            "success": True,
            "transaction_hash": result['transactionHash'],
            "status": "active" if is_active else "inactive"
        }
        
    except Exception as e:
        logger.error(f"Failed to update agent status {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/search")
async def search_agents(criteria: AgentSearchCriteria):
    """Search agents by criteria"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        result = await a2a_client.agents.search_agents(criteria.dict(exclude_none=True))
        return result
        
    except Exception as e:
        logger.error(f"Agent search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/owner/{address}")
async def get_agents_by_owner(address: str):
    """Get agents owned by address"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        agents = await a2a_client.agents.get_agents_by_owner(address)
        return {
            "owner": address,
            "agents": agents,
            "total": len(agents)
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch agents for owner {address}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Messaging

@router.post("/messages/send")
async def send_message(message: MessageRequest):
    """Send message through A2A Network"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        result = await a2a_client.messages.send({
            "recipientId": message.recipient_id,
            "content": message.content,
            "messageType": message.message_type,
            "metadata": message.metadata
        })
        
        return {
            "success": True,
            "message_id": result['messageId'],
            "transaction_hash": result['transactionHash']
        }
        
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/messages/{agent_id}")
async def get_message_history(agent_id: str, limit: int = 50, offset: int = 0):
    """Get message history for agent"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        messages = await a2a_client.messages.get_message_history(agent_id, {
            "limit": limit,
            "offset": offset
        })
        
        return messages
        
    except Exception as e:
        logger.error(f"Failed to fetch message history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Reputation

@router.get("/reputation/{agent_id}")
async def get_agent_reputation(agent_id: str):
    """Get agent reputation score"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        reputation = await a2a_client.reputation.get_reputation(agent_id)
        return reputation
        
    except Exception as e:
        logger.error(f"Failed to fetch reputation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reputation/leaderboard")
async def get_reputation_leaderboard(limit: int = 10):
    """Get reputation leaderboard"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        leaderboard = await a2a_client.reputation.get_reputation_leaderboard(limit)
        return leaderboard
        
    except Exception as e:
        logger.error(f"Failed to fetch leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Token Operations

@router.get("/tokens/balance/{address}")
async def get_token_balance(address: str):
    """Get A2A token balance"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        balance = await a2a_client.tokens.get_balance(address)
        return balance
        
    except Exception as e:
        logger.error(f"Failed to fetch token balance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tokens/info")
async def get_token_info():
    """Get A2A token information"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        info = await a2a_client.tokens.get_token_info()
        return info
        
    except Exception as e:
        logger.error(f"Failed to fetch token info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Governance

@router.get("/governance/proposals")
async def get_proposals(status: str = "active"):
    """Get governance proposals"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        proposals = await a2a_client.governance.get_proposals(status)
        return proposals
        
    except Exception as e:
        logger.error(f"Failed to fetch proposals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/governance/proposals/{proposal_id}")
async def get_proposal(proposal_id: str):
    """Get specific proposal details"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        proposal = await a2a_client.governance.get_proposal(proposal_id)
        return proposal
        
    except Exception as e:
        logger.error(f"Failed to fetch proposal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Webhook Management

@router.post("/webhooks/subscribe")
async def subscribe_webhook(subscription: WebhookSubscription):
    """Subscribe to A2A Network events"""
    import uuid
    subscription_id = str(uuid.uuid4())
    
    webhook_subscriptions[subscription_id] = {
        "event_type": subscription.event_type,
        "webhook_url": subscription.webhook_url,
        "filters": subscription.filters,
        "active": subscription.active,
        "created_at": datetime.utcnow().isoformat()
    }
    
    return {
        "subscription_id": subscription_id,
        "status": "active" if subscription.active else "paused",
        "message": f"Webhook subscription created for {subscription.event_type}"
    }

@router.get("/webhooks/subscriptions")
async def get_webhook_subscriptions():
    """Get all webhook subscriptions"""
    return {
        "subscriptions": [
            {
                "subscription_id": sub_id,
                **sub_data
            }
            for sub_id, sub_data in webhook_subscriptions.items()
        ]
    }

@router.delete("/webhooks/subscriptions/{subscription_id}")
async def unsubscribe_webhook(subscription_id: str):
    """Unsubscribe from webhook"""
    if subscription_id not in webhook_subscriptions:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    del webhook_subscriptions[subscription_id]
    
    return {
        "subscription_id": subscription_id,
        "status": "unsubscribed"
    }

@router.patch("/webhooks/subscriptions/{subscription_id}")
async def update_webhook_subscription(subscription_id: str, updates: Dict[str, Any]):
    """Update webhook subscription"""
    if subscription_id not in webhook_subscriptions:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    allowed_updates = ["webhook_url", "filters", "active"]
    for key, value in updates.items():
        if key in allowed_updates:
            webhook_subscriptions[subscription_id][key] = value
    
    webhook_subscriptions[subscription_id]["updated_at"] = datetime.utcnow().isoformat()
    
    return {
        "subscription_id": subscription_id,
        "status": "updated",
        "subscription": webhook_subscriptions[subscription_id]
    }

# Analytics

@router.get("/analytics/network")
async def get_network_analytics():
    """Get network analytics and statistics"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        # Get network overview
        network_info = await a2a_client.get_network_info()
        
        # Try to get additional stats
        stats = {
            "network": network_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to fetch network analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/agents/{agent_id}/activity")
async def get_agent_activity(agent_id: str, days: int = 30):
    """Get agent activity analytics"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        # Get agent statistics
        stats = await a2a_client.agents.get_statistics(agent_id)
        
        return {
            "agent_id": agent_id,
            "period_days": days,
            "statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch agent activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Contract Information

@router.get("/contracts")
async def get_contract_addresses():
    """Get deployed contract addresses"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        config = a2a_client.get_config()
        network_info = await a2a_client.get_network_info()
        
        return {
            "network": config['network'],
            "chain_id": network_info['chain_id'],
            "contracts": network_info.get('contracts', []),
            "block_number": network_info['block_number']
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch contract info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Gas Estimation

@router.post("/estimate-gas/register-agent")
async def estimate_registration_gas(request: AgentRegistrationRequest):
    """Estimate gas for agent registration"""
    if not a2a_client:
        raise HTTPException(status_code=503, detail="Not connected to A2A Network")
    
    try:
        gas_estimate = await a2a_client.agents.estimate_registration_gas({
            "name": request.name,
            "description": request.description,
            "endpoint": request.endpoint,
            "capabilities": request.capabilities,
            "metadata": request.metadata
        })
        
        return {
            "gas_estimate": str(gas_estimate),
            "operation": "agent_registration"
        }
        
    except Exception as e:
        logger.error(f"Failed to estimate gas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Test webhook endpoint (for debugging)
@router.post("/webhooks/test/{subscription_id}")
async def test_webhook(subscription_id: str, test_data: Optional[Dict[str, Any]] = None):
    """Test webhook delivery"""
    if subscription_id not in webhook_subscriptions:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    subscription = webhook_subscriptions[subscription_id]
    
    test_payload = {
        "event_type": "test_event",
        "data": test_data or {"test": True, "message": "Test webhook delivery"},
        "timestamp": datetime.utcnow().isoformat(),
        "subscription_id": subscription_id
    }
    
    try:
        await send_webhook_notification(subscription['webhook_url'], test_payload)
        return {
            "status": "sent",
            "webhook_url": subscription['webhook_url'],
            "payload": test_payload
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }

# Cleanup function
async def cleanup():
    """Cleanup resources on shutdown"""
    global a2a_client
    
    if a2a_client:
        try:
            await a2a_client.disconnect()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            a2a_client = None