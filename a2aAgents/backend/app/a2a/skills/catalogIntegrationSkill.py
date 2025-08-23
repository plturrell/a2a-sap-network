"""
Catalog Integration Skill for A2A Agent
Provides downstream catalog synchronization while maintaining A2A protocol compliance
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from uuid import uuid4
import logging
from abc import ABC, abstractmethod

from ..core.a2aTypes import A2AMessage, MessagePart, MessageRole
from ..security.smart_contract_trust import sign_a2a_message, verify_a2a_message

logger = logging.getLogger(__name__)


class CatalogChangeEvent:
    """Represents a catalog change event for downstream propagation"""
    
    def __init__(self, operation: str, entity_type: str, entity_id: str, 
                 metadata: Dict[str, Any], source_agent: str):
        self.event_id = str(uuid4())
        self.operation = operation  # create, update, delete
        self.entity_type = entity_type  # table, schema, column, data_product
        self.entity_id = entity_id
        self.metadata = metadata
        self.timestamp = datetime.utcnow().isoformat()
        self.source_agent = source_agent
    
    def to_a2a_message(self) -> A2AMessage:
        """Convert to A2A message format"""
        return A2AMessage(
            role=MessageRole.AGENT,
            contextId=self.event_id,
            parts=[
                MessagePart(
                    kind="catalog_change_event",
                    data={
                        "event_id": self.event_id,
                        "operation": self.operation,
                        "entity_type": self.entity_type,
                        "entity_id": self.entity_id,
                        "metadata": self.metadata,
                        "timestamp": self.timestamp,
                        "source_agent": self.source_agent
                    }
                )
            ]
        )


class DownstreamConnector(ABC):
    """Abstract base class for downstream platform connectors"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "unknown")
        self.enabled = config.get("enabled", True)
        self.retry_policy = config.get("retry_policy", {
            "max_attempts": 3,
            "backoff_seconds": 2
        })
    
    @abstractmethod
    async def push_catalog_change(self, event: CatalogChangeEvent) -> Dict[str, Any]:
        """Push catalog change to downstream platform"""
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate connection to downstream platform"""
        pass
    
    @abstractmethod
    def get_supported_entity_types(self) -> List[str]:
        """Get list of supported entity types for this connector"""
        pass


class CatalogIntegrationSkill:
    """
    Skill for managing catalog integration and downstream synchronization
    Maintains full A2A protocol compliance
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.connectors: Dict[str, DownstreamConnector] = {}
        self.event_queue: List[CatalogChangeEvent] = []
        self.processing = False
        self.sync_history: Dict[str, Any] = {}
        
    def register_connector(self, connector_id: str, connector: DownstreamConnector):
        """Register a downstream connector"""
        self.connectors[connector_id] = connector
        logger.info(f"Registered connector: {connector_id} ({connector.name})")
    
    async def emit_catalog_change(self, operation: str, entity_type: str, 
                                 entity_id: str, metadata: Dict[str, Any]) -> str:
        """
        Emit a catalog change event for downstream propagation
        Returns event ID for tracking
        """
        event = CatalogChangeEvent(
            operation=operation,
            entity_type=entity_type,
            entity_id=entity_id,
            metadata=metadata,
            source_agent=self.agent_id
        )
        
        # Add to queue
        self.event_queue.append(event)
        
        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_event_queue())
        
        return event.event_id
    
    async def _process_event_queue(self):
        """Process queued catalog change events"""
        self.processing = True
        
        try:
            while self.event_queue:
                event = self.event_queue.pop(0)
                await self._propagate_event(event)
        finally:
            self.processing = False
    
    async def _propagate_event(self, event: CatalogChangeEvent):
        """Propagate event to all enabled connectors"""
        tasks = []
        
        for connector_id, connector in self.connectors.items():
            if connector.enabled and event.entity_type in connector.get_supported_entity_types():
                task = asyncio.create_task(
                    self._push_with_retry(connector_id, connector, event)
                )
                tasks.append(task)
        
        # Wait for all propagations to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Catalog event {event.event_id} propagated: {successful}/{len(results)} successful")
        
        # Store in history
        self.sync_history[event.event_id] = {
            "event": event.__dict__,
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _push_with_retry(self, connector_id: str, connector: DownstreamConnector, 
                              event: CatalogChangeEvent) -> Dict[str, Any]:
        """Push event to connector with retry logic"""
        max_attempts = connector.retry_policy.get("max_attempts", 3)
        backoff = connector.retry_policy.get("backoff_seconds", 2)
        
        for attempt in range(max_attempts):
            try:
                result = await connector.push_catalog_change(event)
                logger.info(f"Successfully pushed to {connector_id}")
                return {"connector_id": connector_id, "status": "success", "result": result}
            except Exception as e:
                if attempt == max_attempts - 1:
                    logger.error(f"Failed to push to {connector_id} after {max_attempts} attempts: {e}")
                    return {"connector_id": connector_id, "status": "failed", "error": str(e)}
                
                await asyncio.sleep(backoff ** attempt)
    
    def handle_a2a_catalog_request(self, message: A2AMessage) -> Optional[A2AMessage]:
        """
        Handle A2A messages related to catalog operations
        Returns response message if applicable
        """
        for part in message.parts:
            if part.kind == "catalog_query":
                # Handle catalog query
                return self._handle_catalog_query(part.data)
            elif part.kind == "sync_status_request":
                # Handle sync status request
                return self._handle_sync_status_request(part.data)
        
        return None
    
    def _handle_catalog_query(self, query_data: Dict[str, Any]) -> A2AMessage:
        """Handle catalog query request"""
        # Extract query parameters
        entity_type = query_data.get("entity_type")
        entity_id = query_data.get("entity_id")
        
        # Search sync history
        results = []
        for event_id, history in self.sync_history.items():
            event = history["event"]
            if (not entity_type or event["entity_type"] == entity_type) and \
               (not entity_id or event["entity_id"] == entity_id):
                results.append(history)
        
        return A2AMessage(
            role=MessageRole.AGENT,
            parts=[
                MessagePart(
                    kind="catalog_query_response",
                    data={
                        "query": query_data,
                        "results": results,
                        "count": len(results)
                    }
                )
            ]
        )
    
    def _handle_sync_status_request(self, request_data: Dict[str, Any]) -> A2AMessage:
        """Handle sync status request"""
        event_id = request_data.get("event_id")
        
        if event_id in self.sync_history:
            status = self.sync_history[event_id]
        else:
            status = {"error": "Event not found"}
        
        return A2AMessage(
            role=MessageRole.AGENT,
            parts=[
                MessagePart(
                    kind="sync_status_response",
                    data={
                        "event_id": event_id,
                        "status": status
                    }
                )
            ]
        )
    
    def get_skill_info(self) -> Dict[str, Any]:
        """Get skill information for agent card"""
        return {
            "id": "catalog-integration",
            "name": "Catalog Integration",
            "description": "Manages downstream catalog synchronization across platforms",
            "capabilities": [
                "catalog_change_events",
                "multi_platform_sync",
                "sync_status_tracking",
                "retry_with_backoff"
            ],
            "connectors": list(self.connectors.keys()),
            "tags": ["catalog", "integration", "synchronization", "downstream"]
        }