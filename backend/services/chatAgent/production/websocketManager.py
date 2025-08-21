"""
Production WebSocket Manager for A2A Chat Agent
Provides real-time bidirectional communication
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
from uuid import uuid4
import websockets
from fastapi import WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections and message routing
    """
    
    def __init__(self):
        # Active connections by conversation ID
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        # User ID to WebSocket mapping
        self.user_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        # Message queues for offline users
        self.offline_queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # Connection stats
        self.stats = {
            'total_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0
        }
        
    async def connect(
        self, 
        websocket: WebSocket, 
        user_id: str,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        
        # Store connection metadata
        self.connection_metadata[websocket] = {
            'user_id': user_id,
            'conversation_id': conversation_id,
            'connected_at': datetime.utcnow(),
            'metadata': metadata or {}
        }
        
        # Add to user connections
        self.user_connections[user_id].add(websocket)
        
        # Add to conversation if specified
        if conversation_id:
            self.active_connections[conversation_id].add(websocket)
        
        self.stats['total_connections'] += 1
        
        # Send any queued offline messages
        await self._send_offline_messages(user_id, websocket)
        
        # Send connection acknowledgment
        await self.send_personal_message(
            {
                'type': 'connection_established',
                'user_id': user_id,
                'conversation_id': conversation_id,
                'timestamp': datetime.utcnow().isoformat()
            },
            websocket
        )
        
        logger.info(f"WebSocket connected: user={user_id}, conversation={conversation_id}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        metadata = self.connection_metadata.get(websocket, {})
        user_id = metadata.get('user_id')
        conversation_id = metadata.get('conversation_id')
        
        # Remove from conversation
        if conversation_id and conversation_id in self.active_connections:
            self.active_connections[conversation_id].discard(websocket)
            if not self.active_connections[conversation_id]:
                del self.active_connections[conversation_id]
        
        # Remove from user connections
        if user_id and user_id in self.user_connections:
            self.user_connections[user_id].discard(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        # Clean up metadata
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
        
        logger.info(f"WebSocket disconnected: user={user_id}, conversation={conversation_id}")
    
    async def send_personal_message(
        self, 
        message: Dict[str, Any], 
        websocket: WebSocket
    ):
        """Send message to specific WebSocket connection"""
        try:
            await websocket.send_json(message)
            self.stats['messages_sent'] += 1
        except (WebSocketDisconnect, ConnectionClosed) as e:
            logger.error(f"Failed to send message: {e}")
            self.disconnect(websocket)
            self.stats['errors'] += 1
        except Exception as e:
            logger.error(f"Unexpected error sending message: {e}")
            self.stats['errors'] += 1
    
    async def send_to_user(
        self, 
        user_id: str, 
        message: Dict[str, Any],
        queue_if_offline: bool = True
    ):
        """Send message to all connections of a user"""
        connections = self.user_connections.get(user_id, set())
        
        if not connections and queue_if_offline:
            # Queue message for offline user
            self.offline_queues[user_id].append({
                'message': message,
                'timestamp': datetime.utcnow(),
                'attempts': 0
            })
            # Limit queue size
            if len(self.offline_queues[user_id]) > 100:
                self.offline_queues[user_id] = self.offline_queues[user_id][-100:]
            return
        
        # Send to all user connections
        disconnected = []
        for connection in connections:
            try:
                await connection.send_json(message)
                self.stats['messages_sent'] += 1
            except (WebSocketDisconnect, ConnectionClosed):
                disconnected.append(connection)
            except Exception as e:
                logger.error(f"Error sending to user {user_id}: {e}")
                self.stats['errors'] += 1
        
        # Clean up disconnected connections
        for conn in disconnected:
            self.disconnect(conn)
    
    async def broadcast_to_conversation(
        self, 
        conversation_id: str, 
        message: Dict[str, Any],
        exclude_websocket: Optional[WebSocket] = None
    ):
        """Broadcast message to all participants in a conversation"""
        connections = self.active_connections.get(conversation_id, set())
        
        disconnected = []
        for connection in connections:
            if connection == exclude_websocket:
                continue
                
            try:
                await connection.send_json(message)
                self.stats['messages_sent'] += 1
            except (WebSocketDisconnect, ConnectionClosed):
                disconnected.append(connection)
            except Exception as e:
                logger.error(f"Error broadcasting to conversation {conversation_id}: {e}")
                self.stats['errors'] += 1
        
        # Clean up disconnected connections
        for conn in disconnected:
            self.disconnect(conn)
    
    async def handle_message(
        self, 
        websocket: WebSocket, 
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process incoming WebSocket message"""
        self.stats['messages_received'] += 1
        
        metadata = self.connection_metadata.get(websocket, {})
        user_id = metadata.get('user_id')
        
        # Add metadata to message
        message['user_id'] = user_id
        message['timestamp'] = datetime.utcnow().isoformat()
        
        # Message type handlers
        message_type = message.get('type', 'chat')
        
        if message_type == 'ping':
            return {'type': 'pong', 'timestamp': message['timestamp']}
        
        elif message_type == 'join_conversation':
            conversation_id = message.get('conversation_id')
            if conversation_id:
                self.active_connections[conversation_id].add(websocket)
                metadata['conversation_id'] = conversation_id
                await self.broadcast_to_conversation(
                    conversation_id,
                    {
                        'type': 'user_joined',
                        'user_id': user_id,
                        'conversation_id': conversation_id,
                        'timestamp': message['timestamp']
                    },
                    exclude_websocket=websocket
                )
                return {
                    'type': 'joined_conversation',
                    'conversation_id': conversation_id,
                    'timestamp': message['timestamp']
                }
        
        elif message_type == 'leave_conversation':
            conversation_id = metadata.get('conversation_id')
            if conversation_id and conversation_id in self.active_connections:
                self.active_connections[conversation_id].discard(websocket)
                metadata['conversation_id'] = None
                await self.broadcast_to_conversation(
                    conversation_id,
                    {
                        'type': 'user_left',
                        'user_id': user_id,
                        'conversation_id': conversation_id,
                        'timestamp': message['timestamp']
                    }
                )
                return {
                    'type': 'left_conversation',
                    'conversation_id': conversation_id,
                    'timestamp': message['timestamp']
                }
        
        elif message_type == 'typing':
            conversation_id = metadata.get('conversation_id')
            if conversation_id:
                await self.broadcast_to_conversation(
                    conversation_id,
                    {
                        'type': 'user_typing',
                        'user_id': user_id,
                        'conversation_id': conversation_id,
                        'timestamp': message['timestamp']
                    },
                    exclude_websocket=websocket
                )
        
        return message
    
    async def _send_offline_messages(
        self, 
        user_id: str, 
        websocket: WebSocket
    ):
        """Send queued offline messages to newly connected user"""
        if user_id not in self.offline_queues:
            return
        
        messages = self.offline_queues[user_id]
        if not messages:
            return
        
        # Send offline message notification
        await self.send_personal_message(
            {
                'type': 'offline_messages',
                'count': len(messages),
                'timestamp': datetime.utcnow().isoformat()
            },
            websocket
        )
        
        # Send each queued message
        for queued in messages:
            await self.send_personal_message(queued['message'], websocket)
        
        # Clear the queue
        del self.offline_queues[user_id]
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get current connection statistics"""
        return {
            'active_connections': sum(len(conns) for conns in self.active_connections.values()),
            'active_users': len(self.user_connections),
            'active_conversations': len(self.active_connections),
            'queued_messages': sum(len(msgs) for msgs in self.offline_queues.values()),
            'total_stats': self.stats
        }
    
    def get_user_status(self, user_id: str) -> Dict[str, Any]:
        """Get connection status for a user"""
        connections = self.user_connections.get(user_id, set())
        if not connections:
            return {
                'online': False,
                'connections': 0,
                'queued_messages': len(self.offline_queues.get(user_id, []))
            }
        
        return {
            'online': True,
            'connections': len(connections),
            'conversations': [
                meta.get('conversation_id') 
                for conn, meta in self.connection_metadata.items() 
                if meta.get('user_id') == user_id and meta.get('conversation_id')
            ]
        }


class WebSocketEndpoint:
    """
    FastAPI WebSocket endpoint handler
    """
    
    def __init__(
        self, 
        connection_manager: ConnectionManager,
        auth_manager: Any,
        chat_agent: Any
    ):
        self.manager = connection_manager
        self.auth_manager = auth_manager
        self.chat_agent = chat_agent
    
    async def websocket_endpoint(
        self, 
        websocket: WebSocket,
        token: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Main WebSocket endpoint handler
        """
        # Authenticate connection
        try:
            if token:
                user_data = self.auth_manager.verify_jwt_token(token)
            elif api_key:
                user_data = await self.auth_manager._validate_api_key(api_key)
                if not user_data:
                    await websocket.close(code=4001, reason="Invalid API key")
                    return
            else:
                await websocket.close(code=4001, reason="Authentication required")
                return
            
            user_id = user_data['user_id']
            
        except Exception as e:
            logger.error(f"WebSocket authentication failed: {e}")
            await websocket.close(code=4001, reason="Authentication failed")
            return
        
        # Accept connection
        await self.manager.connect(websocket, user_id)
        
        try:
            # Message handling loop
            while True:
                # Receive message
                data = await websocket.receive_json()
                
                # Handle message
                response = await self.manager.handle_message(websocket, data)
                
                # Process chat messages through agent
                if data.get('type') == 'chat' and data.get('content'):
                    # Create A2A message
                    from a2aCommon import A2AMessage, MessageRole
                    
                    message = A2AMessage(
                        role=MessageRole.USER,
                        content={
                            'data': {
                                'prompt': data['content'],
                                'user_id': user_id,
                                'conversation_id': data.get('conversation_id'),
                                'websocket': True
                            }
                        },
                        context_id=f"ws_{user_id}_{uuid4().hex[:8]}_{int(time.time())}"
                    )
                    
                    # Process through chat agent
                    agent_response = await self.chat_agent.handle_chat_message(
                        message, 
                        message.context_id
                    )
                    
                    # Send response back
                    await self.manager.send_personal_message(
                        {
                            'type': 'agent_response',
                            'response': agent_response,
                            'request_id': data.get('request_id'),
                            'timestamp': datetime.utcnow().isoformat()
                        },
                        websocket
                    )
                    
                    # Broadcast to conversation if applicable
                    if data.get('conversation_id'):
                        await self.manager.broadcast_to_conversation(
                            data['conversation_id'],
                            {
                                'type': 'new_message',
                                'user_id': user_id,
                                'content': data['content'],
                                'timestamp': datetime.utcnow().isoformat()
                            },
                            exclude_websocket=websocket
                        )
                
                # Send response if needed
                elif response and response != data:
                    await self.manager.send_personal_message(response, websocket)
                    
        except WebSocketDisconnect:
            self.manager.disconnect(websocket)
            logger.info(f"WebSocket disconnected: {user_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.manager.disconnect(websocket)
            try:
                await websocket.close(code=4000, reason="Internal error")
            except:
                pass


class RealtimeNotificationService:
    """
    Service for sending real-time notifications
    """
    
    def __init__(self, connection_manager: ConnectionManager):
        self.manager = connection_manager
    
    async def notify_task_completion(
        self, 
        user_id: str, 
        task_id: str, 
        result: Dict[str, Any]
    ):
        """Notify user of task completion"""
        await self.manager.send_to_user(
            user_id,
            {
                'type': 'task_completed',
                'task_id': task_id,
                'result': result,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    async def notify_agent_response(
        self, 
        conversation_id: str, 
        agent_id: str, 
        response: Dict[str, Any]
    ):
        """Notify conversation participants of agent response"""
        await self.manager.broadcast_to_conversation(
            conversation_id,
            {
                'type': 'agent_response',
                'agent_id': agent_id,
                'response': response,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    async def notify_error(
        self, 
        user_id: str, 
        error: str, 
        details: Optional[Dict[str, Any]] = None
    ):
        """Notify user of an error"""
        await self.manager.send_to_user(
            user_id,
            {
                'type': 'error',
                'error': error,
                'details': details or {},
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    async def broadcast_system_message(
        self, 
        message: str, 
        level: str = 'info',
        target_users: Optional[List[str]] = None
    ):
        """Broadcast system message to users"""
        notification = {
            'type': 'system_message',
            'message': message,
            'level': level,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if target_users:
            # Send to specific users
            for user_id in target_users:
                await self.manager.send_to_user(user_id, notification)
        else:
            # Broadcast to all connected users
            for user_id in list(self.manager.user_connections.keys()):
                await self.manager.send_to_user(user_id, notification)


# Factory functions
def create_connection_manager() -> ConnectionManager:
    """Create WebSocket connection manager"""
    return ConnectionManager()


def create_websocket_endpoint(
    connection_manager: ConnectionManager,
    auth_manager: Any,
    chat_agent: Any
) -> WebSocketEndpoint:
    """Create WebSocket endpoint handler"""
    return WebSocketEndpoint(connection_manager, auth_manager, chat_agent)


def create_notification_service(
    connection_manager: ConnectionManager
) -> RealtimeNotificationService:
    """Create real-time notification service"""
    return RealtimeNotificationService(connection_manager)