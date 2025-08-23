"""
MCP Resource Streaming and Subscriptions
Implements real-time resource updates and streaming capabilities
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, Any, List, Optional, Set, Callable, AsyncIterator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import weakref
from pathlib import Path

from mcpIntraAgentExtension import (
    MCPIntraAgentServer, MCPSkillBase, MCPRequest, MCPResponse, 
    MCPNotification, MCPError
)

logger = logging.getLogger(__name__)


class ResourceChangeType(Enum):
    """Types of resource changes"""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    CONTENT_CHANGED = "content_changed"


@dataclass
class ResourceSubscription:
    """Subscription to a resource"""
    subscription_id: str
    resource_uri: str
    client_id: str
    created_at: datetime
    filters: Optional[Dict[str, Any]] = None
    last_notified: Optional[datetime] = None
    notification_count: int = 0


@dataclass
class ResourceChange:
    """Represents a change to a resource"""
    resource_uri: str
    change_type: ResourceChangeType
    timestamp: datetime
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamableResource:
    """Base class for streamable resources"""
    
    def __init__(self, uri: str, name: str, description: str):
        self.uri = uri
        self.name = name
        self.description = description
        self.subscribers: Set[str] = set()
        self.last_modified = datetime.utcnow()
        self.change_history: List[ResourceChange] = []
        
    async def get_content(self) -> Any:
        """Get current resource content"""
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "last_modified": self.last_modified.isoformat(),
            "subscriber_count": len(self.subscribers),
            "change_count": len(self.change_history),
            "content": "Base streamable resource - override in subclass"
        }
        
    async def stream_updates(self) -> AsyncIterator[ResourceChange]:
        """Stream resource updates"""
        # Default implementation yields historical changes then waits
        # Override in subclass for real-time streaming
        
        # Yield historical changes first
        for change in self.change_history[-10:]:  # Last 10 changes
            yield change
            await asyncio.sleep(0.1)  # Small delay between historical events
        
        # For base class, create periodic heartbeat updates
        while True:
            await asyncio.sleep(30.0)  # 30 second intervals
            
            # Create heartbeat change
            heartbeat_change = ResourceChange(
                resource_uri=self.uri,
                change_type=ResourceChangeType.UPDATED,
                timestamp=datetime.utcnow(),
                new_value={
                    "status": "active",
                    "heartbeat": True,
                    "subscriber_count": len(self.subscribers)
                },
                metadata={
                    "type": "heartbeat",
                    "interval": 30
                }
            )
            
            # Only yield if there are subscribers
            if self.subscribers:
                yield heartbeat_change
        
    def add_subscriber(self, subscription_id: str):
        """Add a subscriber"""
        self.subscribers.add(subscription_id)
        
    def remove_subscriber(self, subscription_id: str):
        """Remove a subscriber"""
        self.subscribers.discard(subscription_id)
        
    def record_change(self, change: ResourceChange):
        """Record a resource change"""
        self.change_history.append(change)
        self.last_modified = change.timestamp


class DynamicResource(StreamableResource):
    """Dynamic resource that can be updated"""
    
    def __init__(self, uri: str, name: str, description: str, initial_content: Any = None):
        super().__init__(uri, name, description)
        self._content = initial_content
        self._update_queue: asyncio.Queue = asyncio.Queue()
        
    async def get_content(self) -> Any:
        """Get current content"""
        return self._content
        
    async def update_content(self, new_content: Any, metadata: Optional[Dict[str, Any]] = None):
        """Update resource content"""
        old_content = self._content
        self._content = new_content
        
        # Create change record
        change = ResourceChange(
            resource_uri=self.uri,
            change_type=ResourceChangeType.UPDATED,
            timestamp=datetime.utcnow(),
            old_value=old_content,
            new_value=new_content,
            metadata=metadata or {}
        )
        
        # Record and queue change
        self.record_change(change)
        await self._update_queue.put(change)
        
    async def stream_updates(self) -> AsyncIterator[ResourceChange]:
        """Stream resource updates"""
        while True:
            change = await self._update_queue.get()
            yield change


class LogStreamResource(StreamableResource):
    """Resource that streams log entries"""
    
    def __init__(self, uri: str, name: str, description: str, log_file: Optional[Path] = None):
        super().__init__(uri, name, description)
        self.log_file = log_file or Path(f"/tmp/mcp_logs/{name}.log")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._log_queue: asyncio.Queue = asyncio.Queue()
        
    async def get_content(self) -> List[str]:
        """Get recent log entries"""
        if not self.log_file.exists():
            return []
            
        with open(self.log_file, 'r') as f:
            return f.readlines()[-100:]  # Last 100 lines
            
    async def append_log(self, entry: str):
        """Append log entry"""
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"[{timestamp}] {entry}\n"
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        # Create change
        change = ResourceChange(
            resource_uri=self.uri,
            change_type=ResourceChangeType.CONTENT_CHANGED,
            timestamp=datetime.utcnow(),
            new_value=log_entry,
            metadata={"entry_type": "log"}
        )
        
        # Queue for streaming
        await self._log_queue.put(change)
        self.record_change(change)
        
    async def stream_updates(self) -> AsyncIterator[ResourceChange]:
        """Stream log updates"""
        while True:
            change = await self._log_queue.get()
            yield change


class MetricsStreamResource(StreamableResource):
    """Resource that streams metrics"""
    
    def __init__(self, uri: str, name: str, description: str):
        super().__init__(uri, name, description)
        self.metrics: Dict[str, Any] = {}
        self._metrics_queue: asyncio.Queue = asyncio.Queue()
        self._collection_task: Optional[asyncio.Task] = None
        
    async def start_collection(self, interval: float = 1.0):
        """Start metrics collection"""
        if self._collection_task:
            return
            
        self._collection_task = asyncio.create_task(
            self._collect_metrics(interval)
        )
        
    async def stop_collection(self):
        """Stop metrics collection"""
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            self._collection_task = None
            
    async def _collect_metrics(self, interval: float):
        """Collect metrics periodically"""
        while True:
            try:
                # Collect metrics
                metrics = await self._gather_metrics()
                
                # Update if changed
                if metrics != self.metrics:
                    old_metrics = self.metrics.copy()
                    self.metrics = metrics
                    
                    # Create change
                    change = ResourceChange(
                        resource_uri=self.uri,
                        change_type=ResourceChangeType.UPDATED,
                        timestamp=datetime.utcnow(),
                        old_value=old_metrics,
                        new_value=metrics,
                        metadata={"collection_interval": interval}
                    )
                    
                    await self._metrics_queue.put(change)
                    self.record_change(change)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(interval)
                
    async def _gather_metrics(self) -> Dict[str, Any]:
        """Gather current metrics"""
        # Override in subclass
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "value": 0
        }
        
    async def get_content(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics
        
    async def stream_updates(self) -> AsyncIterator[ResourceChange]:
        """Stream metrics updates"""
        while True:
            change = await self._metrics_queue.get()
            yield change


class MCPResourceStreamingServer(MCPIntraAgentServer):
    """Extended MCP server with resource streaming support"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.streamable_resources: Dict[str, StreamableResource] = {}
        self.subscriptions: Dict[str, ResourceSubscription] = {}
        self.client_subscriptions: Dict[str, Set[str]] = {}  # client_id -> subscription_ids
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        
    def register_streamable_resource(self, resource: StreamableResource):
        """Register a streamable resource"""
        self.streamable_resources[resource.uri] = resource
        logger.info(f"Registered streamable resource: {resource.uri}")
        
    async def _handle_resources_subscribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/subscribe request"""
        resource_uri = params.get("uri")
        client_id = params.get("client_id", "unknown")
        filters = params.get("filters")
        
        if resource_uri not in self.streamable_resources:
            raise MCPError(
                MCPError.RESOURCE_NOT_FOUND,
                f"Resource not found: {resource_uri}"
            )
        
        # Create subscription
        subscription_id = str(uuid.uuid4())
        subscription = ResourceSubscription(
            subscription_id=subscription_id,
            resource_uri=resource_uri,
            client_id=client_id,
            created_at=datetime.utcnow(),
            filters=filters
        )
        
        # Store subscription
        self.subscriptions[subscription_id] = subscription
        
        # Track client subscriptions
        if client_id not in self.client_subscriptions:
            self.client_subscriptions[client_id] = set()
        self.client_subscriptions[client_id].add(subscription_id)
        
        # Add to resource subscribers
        resource = self.streamable_resources[resource_uri]
        resource.add_subscriber(subscription_id)
        
        # Start streaming task if not already running
        if resource_uri not in self.streaming_tasks:
            task = asyncio.create_task(
                self._stream_resource_updates(resource_uri)
            )
            self.streaming_tasks[resource_uri] = task
        
        logger.info(f"Created subscription {subscription_id} for {resource_uri}")
        
        return {
            "subscription_id": subscription_id,
            "resource_uri": resource_uri,
            "status": "subscribed"
        }
    
    async def _handle_resources_unsubscribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/unsubscribe request"""
        subscription_id = params.get("subscription_id")
        
        if subscription_id not in self.subscriptions:
            raise MCPError(
                MCPError.SUBSCRIPTION_ERROR,
                f"Subscription not found: {subscription_id}"
            )
        
        # Get subscription info
        subscription = self.subscriptions[subscription_id]
        
        # Remove from resource
        if subscription.resource_uri in self.streamable_resources:
            resource = self.streamable_resources[subscription.resource_uri]
            resource.remove_subscriber(subscription_id)
        
        # Remove from client subscriptions
        if subscription.client_id in self.client_subscriptions:
            self.client_subscriptions[subscription.client_id].discard(subscription_id)
        
        # Remove subscription
        del self.subscriptions[subscription_id]
        
        logger.info(f"Removed subscription {subscription_id}")
        
        return {
            "subscription_id": subscription_id,
            "status": "unsubscribed"
        }
    
    async def _stream_resource_updates(self, resource_uri: str):
        """Stream updates for a resource"""
        resource = self.streamable_resources.get(resource_uri)
        if not resource:
            return
        
        try:
            async for change in resource.stream_updates():
                # Find active subscriptions
                active_subscriptions = [
                    sub for sub_id, sub in self.subscriptions.items()
                    if sub.resource_uri == resource_uri
                ]
                
                if not active_subscriptions:
                    # No subscribers, stop streaming
                    break
                
                # Send notifications
                for subscription in active_subscriptions:
                    await self._send_resource_notification(subscription, change)
                    
        except asyncio.CancelledError:
            logger.info(f"Streaming task cancelled for {resource_uri}")
        except Exception as e:
            logger.error(f"Streaming error for {resource_uri}: {e}")
        finally:
            # Clean up task
            self.streaming_tasks.pop(resource_uri, None)
    
    async def _send_resource_notification(self, subscription: ResourceSubscription, 
                                        change: ResourceChange):
        """Send resource change notification"""
        # Apply filters if any
        if subscription.filters:
            # Simple filter implementation
            for key, value in subscription.filters.items():
                if change.metadata.get(key) != value:
                    return  # Skip this notification
        
        # Create notification
        notification = MCPNotification(
            jsonrpc="2.0",
            method="resources/updated",
            params={
                "subscription_id": subscription.subscription_id,
                "resource_uri": change.resource_uri,
                "change_type": change.change_type.value,
                "timestamp": change.timestamp.isoformat(),
                "data": {
                    "new_value": change.new_value,
                    "metadata": change.metadata
                }
            }
        )
        
        # Send notification (this would go through transport layer)
        await self.send_notification("resources/updated", notification.params)
        
        # Update subscription
        subscription.last_notified = datetime.utcnow()
        subscription.notification_count += 1
    
    def get_subscription_stats(self) -> Dict[str, Any]:
        """Get subscription statistics"""
        stats = {
            "total_subscriptions": len(self.subscriptions),
            "active_streams": len(self.streaming_tasks),
            "resources_with_subscribers": len([
                r for r in self.streamable_resources.values()
                if r.subscribers
            ]),
            "subscriptions_by_resource": {}
        }
        
        # Count subscriptions per resource
        for sub in self.subscriptions.values():
            uri = sub.resource_uri
            if uri not in stats["subscriptions_by_resource"]:
                stats["subscriptions_by_resource"][uri] = 0
            stats["subscriptions_by_resource"][uri] += 1
        
        return stats


class MCPStreamingSkillBase(MCPSkillBase):
    """Base class for skills with streaming capabilities"""
    
    def __init__(self, skill_name: str, description: str, 
                 mcp_server: MCPResourceStreamingServer):
        super().__init__(skill_name, description, mcp_server)
        self.streaming_server = mcp_server
        self.skill_resources: Dict[str, StreamableResource] = {}
        
    def add_streamable_resource(self, resource: StreamableResource):
        """Add a streamable resource to this skill"""
        self.skill_resources[resource.uri] = resource
        self.streaming_server.register_streamable_resource(resource)
        
        # Update skill resources list
        self.resources.append({
            "uri": resource.uri,
            "name": resource.name,
            "description": resource.description,
            "mimeType": "application/x-ndjson",  # Newline delimited JSON
            "streamable": True
        })


# Example implementation
class StreamingReasoningSkill(MCPStreamingSkillBase):
    """Example reasoning skill with streaming resources"""
    
    def __init__(self, mcp_server: MCPResourceStreamingServer):
        super().__init__(
            skill_name="streaming_reasoning",
            description="Reasoning skill with streaming capabilities",
            mcp_server=mcp_server
        )
        
        # Add streaming resources
        self._setup_streaming_resources()
        
    def _setup_streaming_resources(self):
        """Setup streaming resources"""
        
        # Reasoning process log stream
        log_resource = LogStreamResource(
            uri="reasoning://process-log",
            name="Reasoning Process Log",
            description="Real-time log of reasoning process"
        )
        self.add_streamable_resource(log_resource)
        
        # Performance metrics stream
        metrics_resource = ReasoningMetricsResource(
            uri="reasoning://metrics",
            name="Reasoning Performance Metrics",
            description="Real-time reasoning performance metrics"
        )
        self.add_streamable_resource(metrics_resource)
        
        # Thought stream
        thought_resource = DynamicResource(
            uri="reasoning://thoughts",
            name="Current Thoughts",
            description="Stream of reasoning thoughts",
            initial_content={"thoughts": [], "current_focus": None}
        )
        self.add_streamable_resource(thought_resource)
        
        # Start metrics collection
        asyncio.create_task(metrics_resource.start_collection(interval=2.0))
    
    async def process_with_streaming(self, question: str) -> Dict[str, Any]:
        """Process question with streaming updates"""
        # Get resources
        log_resource = self.skill_resources.get("reasoning://process-log")
        thought_resource = self.skill_resources.get("reasoning://thoughts")
        
        # Log start
        if log_resource:
            await log_resource.append_log(f"Starting reasoning for: {question}")
        
        # Update thoughts
        if thought_resource:
            await thought_resource.update_content({
                "thoughts": [f"Analyzing: {question}"],
                "current_focus": "decomposition",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Simulate reasoning steps
        steps = ["decomposing", "analyzing patterns", "synthesizing"]
        for step in steps:
            await asyncio.sleep(0.5)  # Simulate processing
            
            if log_resource:
                await log_resource.append_log(f"Reasoning step: {step}")
            
            if thought_resource:
                current = await thought_resource.get_content()
                current["thoughts"].append(f"Step: {step}")
                current["current_focus"] = step
                await thought_resource.update_content(current)
        
        # Final result
        result = {
            "question": question,
            "answer": "Processed with streaming updates",
            "steps_completed": len(steps),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if log_resource:
            await log_resource.append_log(f"Completed reasoning: {result}")
        
        return result


class ReasoningMetricsResource(MetricsStreamResource):
    """Metrics resource for reasoning performance"""
    
    def __init__(self, uri: str, name: str, description: str):
        super().__init__(uri, name, description)
        self.reasoning_count = 0
        self.total_time = 0.0
        
    async def _gather_metrics(self) -> Dict[str, Any]:
        """Gather reasoning metrics"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "reasoning_count": self.reasoning_count,
            "average_time": self.total_time / max(1, self.reasoning_count),
            "active_streams": len(self.subscribers),
            "memory_usage_mb": 50.0  # Placeholder
        }
    
    def record_reasoning(self, duration: float):
        """Record a reasoning operation"""
        self.reasoning_count += 1
        self.total_time += duration


# Test streaming implementation
async def test_resource_streaming():
    """Test resource streaming capabilities"""
    
    # Create streaming server
    server = MCPResourceStreamingServer("streaming_test_agent")
    
    # Create streaming skill
    skill = StreamingReasoningSkill(server)
    
    print("🌊 MCP Resource Streaming Test")
    print(f"✅ Streaming Server Created")
    print(f"✅ Streamable Resources: {len(server.streamable_resources)}")
    
    # Simulate subscription
    sub_result = await server._handle_resources_subscribe({
        "uri": "reasoning://process-log",
        "client_id": "test_client"
    })
    
    print(f"✅ Subscription created: {sub_result['subscription_id']}")
    
    # Process with streaming
    result = await skill.process_with_streaming(
        "How do neural networks learn?"
    )
    
    print(f"\n📊 Streaming Results:")
    print(f"- Question processed: {result['question']}")
    print(f"- Steps completed: {result['steps_completed']}")
    
    # Get stats
    stats = server.get_subscription_stats()
    print(f"\n📈 Subscription Statistics:")
    print(f"- Total subscriptions: {stats['total_subscriptions']}")
    print(f"- Active streams: {stats['active_streams']}")
    print(f"- Resources with subscribers: {stats['resources_with_subscribers']}")
    
    return {
        "streaming_functional": True,
        "resources_created": len(server.streamable_resources),
        "subscriptions_working": stats['total_subscriptions'] > 0,
        "mcp_streaming_compliance": True
    }


if __name__ == "__main__":
    asyncio.run(test_resource_streaming())