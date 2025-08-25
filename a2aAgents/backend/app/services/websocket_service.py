"""
WebSocket service for real-time marketplace updates
Handles real-time communication between marketplace components and users
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import dataclass, asdict
from enum import Enum

from app.services.l3DatabaseCache import L3DatabaseCache
from app.models.marketplace import MarketplaceEvent, WebSocketMessage

logger = logging.getLogger(__name__)

class MessageType(str, Enum):
    # Client to Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"
    REQUEST_STATUS = "request_status"
    
    # Server to Client
    MARKETPLACE_UPDATE = "marketplace_update"
    SERVICE_UPDATE = "service_update" 
    DATA_PRODUCT_UPDATE = "data_product_update"
    AGENT_STATUS_CHANGE = "agent_status_change"
    SERVICE_REQUEST_UPDATE = "service_request_update"
    RECOMMENDATION_UPDATE = "recommendation_update"
    CHECKOUT_COMPLETED = "checkout_completed"
    SUBSCRIPTION_CONFIRMED = "subscription_confirmed"
    PONG = "pong"
    ERROR = "error"

class SubscriptionType(str, Enum):
    MARKETPLACE_STATS = "marketplace_stats"
    AGENT_UPDATES = "agent_updates"
    SERVICE_REQUESTS = "service_requests"
    DATA_PRODUCTS = "data_products" 
    RECOMMENDATIONS = "recommendations"
    USER_ORDERS = "user_orders"
    SYSTEM_ALERTS = "system_alerts"

@dataclass
class WebSocketConnection:
    websocket: WebSocket
    user_id: str
    subscriptions: Set[SubscriptionType]
    connected_at: datetime
    last_ping: datetime
    metadata: Dict[str, Any]

class WebSocketManager:
    """Manages WebSocket connections and message routing for the marketplace"""
    
    def __init__(self, db_cache: L3DatabaseCache):
        self.db_cache = db_cache
        self.active_connections: Dict[str, WebSocketConnection] = {}
        self.topic_subscribers: Dict[SubscriptionType, Set[str]] = {
            topic: set() for topic in SubscriptionType
        }
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
        # Start background tasks
        asyncio.create_task(self._process_message_queue())
        asyncio.create_task(self._periodic_cleanup())
        asyncio.create_task(self._heartbeat_monitor())
    
    async def connect(self, websocket: WebSocket, user_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        
        connection = WebSocketConnection(
            websocket=websocket,
            user_id=user_id,
            subscriptions=set(),
            connected_at=datetime.now(),
            last_ping=datetime.now(),
            metadata=metadata or {}
        )
        
        self.active_connections[user_id] = connection
        
        # Send connection confirmation
        await self._send_to_user(user_id, WebSocketMessage(
            type=MessageType.SUBSCRIPTION_CONFIRMED,
            data={
                "status": "connected",
                "user_id": user_id,
                "server_time": datetime.now().isoformat()
            }
        ))
        
        # Load user's subscription preferences
        await self._load_user_subscriptions(user_id)
        
        logger.info(f"WebSocket connected: user_id={user_id}")
    
    def disconnect(self, user_id: str):
        """Handle WebSocket disconnection"""
        if user_id in self.active_connections:
            connection = self.active_connections[user_id]
            
            # Remove from topic subscriptions
            for topic in connection.subscriptions:
                if topic in self.topic_subscribers:
                    self.topic_subscribers[topic].discard(user_id)
            
            # Remove connection
            del self.active_connections[user_id]
            
            logger.info(f"WebSocket disconnected: user_id={user_id}")
    
    async def handle_message(self, user_id: str, message_data: Dict[str, Any]):
        """Handle incoming message from client"""
        try:
            message_type = message_data.get("type")
            data = message_data.get("data", {})
            
            if user_id not in self.active_connections:
                return
            
            connection = self.active_connections[user_id]
            connection.last_ping = datetime.now()
            
            if message_type == MessageType.SUBSCRIBE:
                await self._handle_subscription(user_id, data)
                
            elif message_type == MessageType.UNSUBSCRIBE:
                await self._handle_unsubscription(user_id, data)
                
            elif message_type == MessageType.PING:
                await self._send_to_user(user_id, WebSocketMessage(
                    type=MessageType.PONG,
                    data={"timestamp": datetime.now().isoformat()}
                ))
                
            elif message_type == MessageType.REQUEST_STATUS:
                await self._handle_status_request(user_id, data)
                
            else:
                logger.warning(f"Unknown message type: {message_type} from user {user_id}")
        
        except Exception as e:
            logger.error(f"Error handling message from user {user_id}: {str(e)}")
            await self._send_error(user_id, f"Error processing message: {str(e)}")
    
    async def broadcast_marketplace_update(self, update_data: Dict[str, Any]):
        """Broadcast general marketplace updates to all subscribed users"""
        message = WebSocketMessage(
            type=MessageType.MARKETPLACE_UPDATE,
            data=update_data
        )
        
        await self._broadcast_to_topic(SubscriptionType.MARKETPLACE_STATS, message)
    
    async def notify_service_update(self, service_id: str, update_data: Dict[str, Any]):
        """Notify about service updates"""
        message = WebSocketMessage(
            type=MessageType.SERVICE_UPDATE,
            data={
                "service_id": service_id,
                "update": update_data,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await self._broadcast_to_topic(SubscriptionType.AGENT_UPDATES, message)
    
    async def notify_data_product_update(self, product_id: str, update_data: Dict[str, Any]):
        """Notify about data product updates"""
        message = WebSocketMessage(
            type=MessageType.DATA_PRODUCT_UPDATE,
            data={
                "product_id": product_id,
                "update": update_data,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await self._broadcast_to_topic(SubscriptionType.DATA_PRODUCTS, message)
    
    async def notify_agent_status_change(self, agent_id: str, old_status: str, new_status: str):
        """Notify about agent status changes"""
        message = WebSocketMessage(
            type=MessageType.AGENT_STATUS_CHANGE,
            data={
                "agent_id": agent_id,
                "old_status": old_status,
                "new_status": new_status,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await self._broadcast_to_topic(SubscriptionType.AGENT_UPDATES, message)
    
    async def notify_service_request_update(self, user_id: str, request_data: Dict[str, Any]):
        """Notify specific user about their service request updates"""
        message = WebSocketMessage(
            type=MessageType.SERVICE_REQUEST_UPDATE,
            data=request_data,
            user_id=user_id
        )
        
        await self._send_to_user(user_id, message)
    
    async def notify_checkout_completed(self, user_id: str, transaction_data: Dict[str, Any]):
        """Notify user about completed checkout"""
        message = WebSocketMessage(
            type=MessageType.CHECKOUT_COMPLETED,
            data=transaction_data,
            user_id=user_id
        )
        
        await self._send_to_user(user_id, message)
    
    async def send_recommendation_update(self, user_id: str, recommendations: List[Dict[str, Any]]):
        """Send personalized recommendations to user"""
        message = WebSocketMessage(
            type=MessageType.RECOMMENDATION_UPDATE,
            data={
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            },
            user_id=user_id
        )
        
        await self._send_to_user(user_id, message)
    
    async def _handle_subscription(self, user_id: str, data: Dict[str, Any]):
        """Handle subscription request"""
        subscription_types = data.get("subscriptions", [])
        
        if user_id not in self.active_connections:
            return
        
        connection = self.active_connections[user_id]
        
        for sub_type_str in subscription_types:
            try:
                sub_type = SubscriptionType(sub_type_str)
                connection.subscriptions.add(sub_type)
                self.topic_subscribers[sub_type].add(user_id)
                
            except ValueError:
                logger.warning(f"Invalid subscription type: {sub_type_str}")
        
        # Save subscription preferences
        await self._save_user_subscriptions(user_id, list(connection.subscriptions))
        
        # Confirm subscription
        await self._send_to_user(user_id, WebSocketMessage(
            type=MessageType.SUBSCRIPTION_CONFIRMED,
            data={
                "subscriptions": list(connection.subscriptions),
                "timestamp": datetime.now().isoformat()
            }
        ))
        
        logger.info(f"User {user_id} subscribed to: {subscription_types}")
    
    async def _handle_unsubscription(self, user_id: str, data: Dict[str, Any]):
        """Handle unsubscription request"""
        subscription_types = data.get("subscriptions", [])
        
        if user_id not in self.active_connections:
            return
        
        connection = self.active_connections[user_id]
        
        for sub_type_str in subscription_types:
            try:
                sub_type = SubscriptionType(sub_type_str)
                connection.subscriptions.discard(sub_type)
                self.topic_subscribers[sub_type].discard(user_id)
                
            except ValueError:
                logger.warning(f"Invalid subscription type: {sub_type_str}")
        
        # Save updated preferences
        await self._save_user_subscriptions(user_id, list(connection.subscriptions))
        
        logger.info(f"User {user_id} unsubscribed from: {subscription_types}")
    
    async def _handle_status_request(self, user_id: str, data: Dict[str, Any]):
        """Handle status request"""
        requested_items = data.get("items", [])
        status_data = {}
        
        for item in requested_items:
            item_type = item.get("type")
            item_id = item.get("id")
            
            # Mock status data - replace with actual status lookup
            if item_type == "service":
                status_data[item_id] = {
                    "status": "available",
                    "active_requests": 5,
                    "avg_response_time": "2.3s"
                }
            elif item_type == "agent":
                status_data[item_id] = {
                    "status": "online",
                    "cpu_usage": 45.2,
                    "memory_usage": 67.8,
                    "active_services": 12
                }
        
        await self._send_to_user(user_id, WebSocketMessage(
            type="status_response",
            data=status_data
        ))
    
    async def _send_to_user(self, user_id: str, message: WebSocketMessage):
        """Send message to specific user"""
        if user_id not in self.active_connections:
            return
        
        connection = self.active_connections[user_id]
        
        try:
            message_dict = {
                "type": message.type,
                "data": message.data,
                "timestamp": message.timestamp.isoformat()
            }
            
            if message.user_id:
                message_dict["user_id"] = message.user_id
            
            await connection.websocket.send_text(json.dumps(message_dict))
            
        except Exception as e:
            logger.error(f"Error sending message to user {user_id}: {str(e)}")
            self.disconnect(user_id)
    
    async def _broadcast_to_topic(self, topic: SubscriptionType, message: WebSocketMessage):
        """Broadcast message to all users subscribed to a topic"""
        if topic not in self.topic_subscribers:
            return
        
        subscribers = list(self.topic_subscribers[topic])  # Copy to avoid concurrent modification
        
        for user_id in subscribers:
            await self._send_to_user(user_id, message)
    
    async def _send_error(self, user_id: str, error_message: str):
        """Send error message to user"""
        await self._send_to_user(user_id, WebSocketMessage(
            type=MessageType.ERROR,
            data={"error": error_message}
        ))
    
    async def _load_user_subscriptions(self, user_id: str):
        """Load user's saved subscription preferences"""
        try:
            subscriptions_data = await self.db_cache.get_async(f"websocket_subscriptions:{user_id}")
            if subscriptions_data:
                subscription_types = json.loads(subscriptions_data)
                
                if user_id in self.active_connections:
                    connection = self.active_connections[user_id]
                    
                    for sub_type_str in subscription_types:
                        try:
                            sub_type = SubscriptionType(sub_type_str)
                            connection.subscriptions.add(sub_type)
                            self.topic_subscribers[sub_type].add(user_id)
                        except ValueError:
                            continue
                    
                    logger.info(f"Loaded subscriptions for user {user_id}: {subscription_types}")
        
        except Exception as e:
            logger.error(f"Error loading subscriptions for user {user_id}: {str(e)}")
    
    async def _save_user_subscriptions(self, user_id: str, subscriptions: List[SubscriptionType]):
        """Save user's subscription preferences"""
        try:
            await self.db_cache.set_async(
                f"websocket_subscriptions:{user_id}",
                json.dumps([str(sub) for sub in subscriptions])
            )
        except Exception as e:
            logger.error(f"Error saving subscriptions for user {user_id}: {str(e)}")
    
    async def _process_message_queue(self):
        """Background task to process queued messages"""
        while True:
            try:
                # Process any queued messages
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
                # Here you could implement message queuing logic
                # For now, we'll just yield control
                
            except Exception as e:
                logger.error(f"Error in message queue processor: {str(e)}")
                await asyncio.sleep(1)
    
    async def _periodic_cleanup(self):
        """Background task for periodic cleanup"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Clean up stale connections
                current_time = datetime.now()
                stale_connections = []
                
                for user_id, connection in self.active_connections.items():
                    # Mark as stale if no ping for 10 minutes
                    if (current_time - connection.last_ping).total_seconds() > 600:
                        stale_connections.append(user_id)
                
                for user_id in stale_connections:
                    logger.info(f"Removing stale connection: {user_id}")
                    self.disconnect(user_id)
                
                # Log connection stats
                logger.info(f"Active WebSocket connections: {len(self.active_connections)}")
                
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {str(e)}")
    
    async def _heartbeat_monitor(self):
        """Background task to monitor connection health"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                current_time = datetime.now()
                
                # Send ping to connections that haven't been active recently
                for user_id, connection in list(self.active_connections.items()):
                    if (current_time - connection.last_ping).total_seconds() > 60:
                        try:
                            await connection.websocket.ping()
                        except:
                            logger.info(f"Ping failed, removing connection: {user_id}")
                            self.disconnect(user_id)
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {str(e)}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        return {
            "active_connections": len(self.active_connections),
            "topic_subscriber_counts": {
                topic.value: len(subscribers) 
                for topic, subscribers in self.topic_subscribers.items()
            },
            "total_subscriptions": sum(
                len(connection.subscriptions) 
                for connection in self.active_connections.values()
            )
        }

# Global WebSocket manager instance
websocket_manager: Optional[WebSocketManager] = None

def get_websocket_manager(db_cache: L3DatabaseCache = None) -> WebSocketManager:
    """Get or create WebSocket manager instance"""
    global websocket_manager
    
    if websocket_manager is None and db_cache:
        websocket_manager = WebSocketManager(db_cache)
    
    return websocket_manager

# Utility functions for easy access
async def broadcast_marketplace_update(update_data: Dict[str, Any]):
    """Utility function to broadcast marketplace updates"""
    if websocket_manager:
        await websocket_manager.broadcast_marketplace_update(update_data)

async def notify_user_checkout(user_id: str, transaction_data: Dict[str, Any]):
    """Utility function to notify user about checkout completion"""
    if websocket_manager:
        await websocket_manager.notify_checkout_completed(user_id, transaction_data)

async def notify_service_status_change(service_id: str, update_data: Dict[str, Any]):
    """Utility function to notify about service status changes"""
    if websocket_manager:
        await websocket_manager.notify_service_update(service_id, update_data)

async def send_user_recommendations(user_id: str, recommendations: List[Dict[str, Any]]):
    """Utility function to send recommendations to user"""
    if websocket_manager:
        await websocket_manager.send_recommendation_update(user_id, recommendations)