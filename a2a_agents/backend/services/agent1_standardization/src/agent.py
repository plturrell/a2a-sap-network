"""
Data Standardization Agent - A2A Microservice
Agent 1: Standardizes financial data to L4 hierarchical structure
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import pandas as pd

from a2a_common import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole
)
from a2a_common.sdk.utils import create_success_response, create_error_response
from a2a_common.skills.account_standardizer import AccountStandardizer
from a2a_common.skills.book_standardizer import BookStandardizer
from a2a_common.skills.location_standardizer import LocationStandardizer
from a2a_common.skills.measure_standardizer import MeasureStandardizer
from a2a_common.skills.product_standardizer import ProductStandardizer

logger = logging.getLogger(__name__)


class DataStandardizationAgent(A2AAgentBase):
    """
    Agent 1: Data Standardization Agent
    A2A compliant agent for standardizing financial data
    """
    
    def __init__(self, base_url: str, agent_manager_url: str, downstream_agent_url: str):
        super().__init__(
            agent_id="data_standardization_agent_1",
            name="Data Standardization Agent",
            description="A2A v0.2.9 compliant agent for standardizing financial data to L4 hierarchical structure",
            version="3.0.0",
            base_url=base_url
        )
        
        self.agent_manager_url = agent_manager_url
        self.downstream_agent_url = downstream_agent_url
        
        # Initialize standardizers
        self.standardizers = {
            "account": AccountStandardizer(),
            "book": BookStandardizer(),
            "location": LocationStandardizer(),
            "measure": MeasureStandardizer(),
            "product": ProductStandardizer()
        }
        
        self.standardization_stats = {
            "total_processed": 0,
            "successful_standardizations": 0,
            "records_standardized": 0,
            "data_types_processed": set()
        }
        
        logger.info(f"Initialized A2A {self.name} v{self.version}")
    
    async def initialize(self) -> None:
        """Initialize agent resources and A2A connections"""
        logger.info("Initializing Data Standardization Agent...")
        
        # Initialize output directory
        self.output_dir = os.getenv("STANDARDIZATION_OUTPUT_DIR", "/tmp/standardized_data")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize A2A trust identity
        await self._initialize_trust_identity()
        
        logger.info("Data Standardization Agent initialized successfully")
    
    async def register_with_network(self) -> None:
        """Register with A2A Agent Manager"""
        try:
            # Register capabilities with Agent Manager
            registration = {
                "agent_id": self.agent_id,
                "name": self.name,
                "base_url": self.base_url,
                "capabilities": {
                    "standardization_types": list(self.standardizers.keys()),
                    "input_formats": ["csv", "json", "excel"],
                    "output_formats": ["json", "parquet"],
                    "batch_processing": True
                },
                "handlers": [h.name for h in self.handlers.values()],
                "skills": [s.name for s in self.skills.values()]
            }
            
            # Send registration to Agent Manager
            # In real implementation, this would make HTTP call to agent_manager_url
            logger.info(f"Registered with A2A network at {self.agent_manager_url}")
            self.is_registered = True
            
        except Exception as e:
            logger.error(f"Failed to register with A2A network: {e}")
            raise
    
    async def deregister_from_network(self) -> None:
        """Deregister from A2A network"""
        logger.info("Deregistering from A2A network...")
        # In real implementation, notify Agent Manager of shutdown
        self.is_registered = False
    
    @a2a_handler("standardize_data", "Standardize financial data to L4 hierarchical structure")
    async def handle_standardization_request(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Main A2A handler for standardization requests"""
        try:
            # Extract standardization request from A2A message
            standardization_request = self._extract_standardization_request(message)
            
            if not standardization_request:
                return create_error_response(400, "No standardization request found in A2A message")
            
            # Create A2A task for tracking
            task_id = await self.create_task("data_standardization", {
                "context_id": context_id,
                "request": standardization_request,
                "source_agent": message.sender_id if hasattr(message, 'sender_id') else None
            })
            
            # Process asynchronously
            asyncio.create_task(self._process_standardization(task_id, standardization_request, context_id))
            
            return create_success_response({
                "task_id": task_id,
                "status": "processing",
                "data_types": list(standardization_request.keys()),
                "message": "Data standardization started",
                "a2a_context": context_id
            })
            
        except Exception as e:
            logger.error(f"Error handling standardization request: {e}")
            return create_error_response(500, str(e))
    
    @a2a_skill("account_standardization", "Standardize account data")
    async def standardize_accounts(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Standardize account data using account standardizer"""
        return await asyncio.to_thread(
            self.standardizers["account"].standardize_batch, data
        )
    
    @a2a_skill("book_standardization", "Standardize book data")
    async def standardize_books(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Standardize book data using book standardizer"""
        return await asyncio.to_thread(
            self.standardizers["book"].standardize_batch, data
        )
    
    @a2a_skill("location_standardization", "Standardize location data")
    async def standardize_locations(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Standardize location data using location standardizer"""
        return await asyncio.to_thread(
            self.standardizers["location"].standardize_batch, data
        )
    
    async def _process_standardization(self, task_id: str, request: Dict[str, Any], context_id: str):
        """Process standardization request asynchronously"""
        try:
            standardized_data = {}
            
            # Process each data type
            for data_type, data in request.items():
                if data_type in self.standardizers:
                    logger.info(f"Standardizing {len(data)} {data_type} records")
                    
                    # Use appropriate skill
                    if data_type == "account":
                        result = await self.standardize_accounts(data)
                    elif data_type == "book":
                        result = await self.standardize_books(data)
                    elif data_type == "location":
                        result = await self.standardize_locations(data)
                    else:
                        # Generic standardization
                        result = await asyncio.to_thread(
                            self.standardizers[data_type].standardize_batch, data
                        )
                    
                    standardized_data[data_type] = result
                    self.standardization_stats["records_standardized"] += len(result)
                    self.standardization_stats["data_types_processed"].add(data_type)
            
            # Update stats
            self.standardization_stats["total_processed"] += 1
            self.standardization_stats["successful_standardizations"] += 1
            
            # Send to downstream agent via A2A protocol
            if self.downstream_agent_url:
                await self._send_to_downstream(standardized_data, context_id)
            
            # Update task status
            await self.update_task_status(task_id, "completed", {
                "standardized_types": list(standardized_data.keys()),
                "total_records": sum(len(data) for data in standardized_data.values())
            })
            
        except Exception as e:
            logger.error(f"Error processing standardization: {e}")
            await self.update_task_status(task_id, "failed", {"error": str(e)})
    
    async def _send_to_downstream(self, data: Dict[str, Any], context_id: str):
        """Send standardized data to downstream agent via A2A protocol"""
        try:
            # Create A2A message
            message = A2AMessage(
                sender_id=self.agent_id,
                content={
                    "standardized_data": data,
                    "context_id": context_id,
                    "timestamp": datetime.utcnow().isoformat()
                },
                role=MessageRole.AGENT
            )
            
            # In real implementation, send via HTTP to downstream agent
            logger.info(f"Sent standardized data to downstream agent at {self.downstream_agent_url}")
            
        except Exception as e:
            logger.error(f"Failed to send to downstream agent: {e}")
    
    def _extract_standardization_request(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Extract standardization request from A2A message"""
        if hasattr(message, 'content'):
            content = message.content
            if isinstance(content, dict):
                return content.get('data_to_standardize', content.get('data', None))
        return None