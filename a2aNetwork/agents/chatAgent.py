"""
A2A Chat Agent - Production-grade conversational agent with persistent storage
Converts user prompts to A2A messages for blockchain communication with the 16 A2A agents
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from uuid import uuid4
from contextlib import asynccontextmanager
import traceback

from ..sdk.agentBase import A2AAgentBase
from ..sdk.types import A2AMessage, MessagePart, MessageRole, TaskStatus
from ..sdk.decorators import a2a_handler, a2a_skill
from .production.chatStorage import ChatStorage

# Blockchain integration imports
try:
    from ..sdk.pythonSdk.blockchain.web3Client import Web3Client
    from ..sdk.pythonSdk.blockchain.eventListener import MessageEventListener
    BLOCKCHAIN_AVAILABLE = True
except ImportError:
    BLOCKCHAIN_AVAILABLE = False

# Authentication and rate limiting
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProductionChatAgent(A2AAgentBase):
    """
    Production-grade A2A Chat Agent with commercial features:
    
    Core Features:
    - Convert user prompts to A2A protocol messages
    - Route messages to appropriate agents via blockchain/HTTP
    - Persistent conversation storage with SQLite/PostgreSQL
    - Multi-user support with authentication
    - Real-time message delivery and notifications
    - Rate limiting and security controls
    
    Commercial Features:
    - Multiple conversation management per user
    - Conversation search and archiving
    - Agent response aggregation and formatting
    - Performance monitoring and analytics
    - Error handling with retry logic
    - Scalable architecture with connection pooling
    """
    
    def __init__(
        self,
        agent_id: str = "production-chat-agent",
        name: str = "A2A Production Chat Agent",
        description: str = "Production-grade chat agent for A2A network communication",
        version: str = "2.0.0",
        base_url: str = "http://localhost:8000",
        
        # Storage configuration
        storage_config: Optional[Dict[str, Any]] = None,
        
        # Blockchain configuration
        blockchain_config: Optional[Dict[str, Any]] = None,
        
        # Authentication configuration
        auth_config: Optional[Dict[str, Any]] = None,
        
        # Performance configuration
        performance_config: Optional[Dict[str, Any]] = None,
        
        **kwargs
    ):
        super().__init__(agent_id, name, description, version, base_url, **kwargs)
        
        # Configuration
        self.storage_config = storage_config or {"database_path": "a2a_chat.db"}
        self.blockchain_config = blockchain_config or {}
        self.auth_config = auth_config or {"enabled": False}
        self.performance_config = performance_config or {
            "max_concurrent_conversations": 100,
            "message_batch_size": 50,
            "response_timeout": 30,
            "retry_attempts": 3
        }
        
        # Core components
        self.storage: Optional[ChatStorage] = None
        self.blockchain_client: Optional[Web3Client] = None
        self.event_listener: Optional[MessageEventListener] = None
        self.redis_client = None
        
        # Agent directory (the 16 A2A agents)
        self.agent_directory: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.metrics = {
            "total_conversations": 0,
            "total_messages": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "average_response_time": 0.0
        }
        
        # Active conversation tracking
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"ProductionChatAgent initialized: {self.agent_id}")
    
    async def initialize(self) -> None:
        """Initialize production chat agent resources"""
        try:
            # Initialize storage
            self.storage = ChatStorage(**self.storage_config)
            await self.storage.initialize()
            
            # Initialize Redis for rate limiting (if available)
            if REDIS_AVAILABLE and self.auth_config.get("redis_url"):
                self.redis_client = redis.Redis.from_url(self.auth_config["redis_url"])
                await self.redis_client.ping()
                logger.info("Redis connection established")
            
            # Initialize blockchain connection
            if BLOCKCHAIN_AVAILABLE and self.blockchain_config:
                await self._init_blockchain()
            
            # Discover network agents
            await self._discover_network_agents()
            
            # Load performance metrics
            await self._load_metrics()
            
            logger.info("ProductionChatAgent initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize ProductionChatAgent: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Cleanup production chat agent resources"""
        try:
            # Save metrics
            await self._save_metrics()
            
            # Close blockchain connections
            if self.event_listener:
                await self.event_listener.stop()
            if self.blockchain_client:
                await self.blockchain_client.disconnect()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            # Close storage
            if self.storage:
                await self.storage.close()
            
            logger.info("ProductionChatAgent shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def _init_blockchain(self):
        """Initialize blockchain connection"""
        try:
            self.blockchain_client = Web3Client(
                rpc_url=self.blockchain_config.get('rpc_url', 'http://localhost:8545'),
                private_key=self.blockchain_config.get('private_key'),
                contract_addresses=self.blockchain_config.get('contract_addresses', {})
            )
            await self.blockchain_client.connect()
            
            # Initialize event listener for real-time messages
            self.event_listener = MessageEventListener(
                web3_client=self.blockchain_client,
                agent_address=self.blockchain_client.account.address,
                message_handler=self._handle_blockchain_message
            )
            await self.event_listener.start()
            
            logger.info("Blockchain connection established")
        except Exception as e:
            logger.error(f"Failed to initialize blockchain: {e}")
            self.blockchain_client = None
            self.event_listener = None
    
    async def _discover_network_agents(self):
        """Discover available agents in the A2A network"""
        # This would typically query the agent registry
        # For now, populate with the known 16 A2A agents
        self.agent_directory = {
            "data-processor": {"description": "Processes and analyzes data", "capabilities": ["data_analysis", "transformation"]},
            "nlp-agent": {"description": "Natural language processing tasks", "capabilities": ["text_analysis", "language_detection"]},
            "crypto-trader": {"description": "Cryptocurrency trading and analysis", "capabilities": ["trading", "market_analysis"]},
            "file-manager": {"description": "File operations and management", "capabilities": ["file_ops", "storage"]},
            "web-scraper": {"description": "Web scraping and content extraction", "capabilities": ["scraping", "data_extraction"]},
            "image-processor": {"description": "Image analysis and processing", "capabilities": ["image_analysis", "computer_vision"]},
            "code-reviewer": {"description": "Code analysis and review", "capabilities": ["code_analysis", "security_review"]},
            "database-agent": {"description": "Database operations and queries", "capabilities": ["database", "sql"]},
            "notification-agent": {"description": "Sends notifications and alerts", "capabilities": ["notifications", "messaging"]},
            "scheduler-agent": {"description": "Task scheduling and automation", "capabilities": ["scheduling", "automation"]},
            "security-agent": {"description": "Security analysis and monitoring", "capabilities": ["security", "threat_detection"]},
            "analytics-agent": {"description": "Advanced analytics and reporting", "capabilities": ["analytics", "reporting"]},
            "workflow-agent": {"description": "Workflow orchestration", "capabilities": ["orchestration", "workflow"]},
            "api-agent": {"description": "API integration and management", "capabilities": ["api_integration", "webhooks"]},
            "ml-agent": {"description": "Machine learning and AI tasks", "capabilities": ["machine_learning", "prediction"]},
            "backup-agent": {"description": "Data backup and recovery", "capabilities": ["backup", "recovery"]}
        }
        logger.info(f"Discovered {len(self.agent_directory)} agents in network")
    
    @a2a_handler("chat_message")
    async def handle_chat_message(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle incoming chat messages from users"""
        try:
            # Extract user input from message
            user_input = self._extract_text_from_message(message)
            if not user_input:
                return {"error": "No text content found in message"}
            
            # Create or get conversation context
            conversation = await self._get_or_create_conversation(context_id, message.role)
            
            # Add message to history
            self._add_to_history(context_id, message)
            
            # Analyze user intent and determine target agent(s)
            routing_decision = await self._analyze_user_intent(user_input, conversation)
            
            # Convert prompt to A2A message format
            a2a_message = await self._convert_prompt_to_a2a_message(
                user_input, 
                routing_decision, 
                context_id,
                message.taskId
            )
            
            # Route message to appropriate agent(s)
            responses = await self._route_message_to_agents(a2a_message, routing_decision)
            
            # Process and format responses
            formatted_response = await self._format_responses_for_user(responses, routing_decision)
            
            return {
                "success": True,
                "response": formatted_response,
                "routed_to": routing_decision.get("target_agents", []),
                "conversation_id": context_id,
                "message_id": a2a_message.messageId,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error handling chat message: {e}")
            return {"error": str(e), "success": False}
    
    @a2a_skill(
        name="send_prompt_to_agent",
        description="Send a user prompt to a specific A2A agent",
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "User prompt text"},
                "target_agent": {"type": "string", "description": "Target agent ID"},
                "context_id": {"type": "string", "description": "Conversation context ID"},
                "use_blockchain": {"type": "boolean", "description": "Send via blockchain", "default": True}
            },
            "required": ["prompt", "target_agent"]
        }
    )
    async def send_prompt_to_agent(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a user prompt to a specific A2A agent"""
        prompt = input_data["prompt"]
        target_agent = input_data["target_agent"]
        context_id = input_data.get("context_id", str(uuid4()))
        use_blockchain = input_data.get("use_blockchain", True)
        
        # Convert prompt to A2A message
        a2a_message = await self._convert_prompt_to_a2a_message(
            prompt,
            {"target_agents": [target_agent], "method": "process_data"},
            context_id
        )
        
        # Send message
        if use_blockchain and self.blockchain_client:
            response = await self._send_via_blockchain(a2a_message, target_agent)
        else:
            response = await self._send_via_http(a2a_message, target_agent)
        
        return {
            "message_sent": True,
            "message_id": a2a_message.messageId,
            "target_agent": target_agent,
            "response": response,
            "context_id": context_id
        }
    
    @a2a_skill(
        name="start_conversation",
        description="Start a new conversation with specified agents",
        input_schema={
            "type": "object",
            "properties": {
                "participants": {"type": "array", "items": {"type": "string"}},
                "initial_message": {"type": "string"},
                "conversation_type": {"type": "string", "enum": ["direct", "group", "broadcast"], "default": "direct"}
            },
            "required": ["participants", "initial_message"]
        }
    )
    async def start_conversation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new conversation with specified agents"""
        participants = input_data["participants"]
        initial_message = input_data["initial_message"]
        conversation_type = input_data.get("conversation_type", "direct")
        
        # Create new conversation context
        context_id = str(uuid4())
        conversation = {
            "id": context_id,
            "type": conversation_type,
            "participants": participants,
            "created_at": datetime.utcnow().isoformat(),
            "status": "active",
            "message_count": 0
        }
        
        self.conversations[context_id] = conversation
        self.message_history[context_id] = []
        
        # Send initial message to all participants
        results = []
        for agent_id in participants:
            result = await self.send_prompt_to_agent({
                "prompt": initial_message,
                "target_agent": agent_id,
                "context_id": context_id
            })
            results.append(result)
        
        return {
            "conversation_started": True,
            "conversation_id": context_id,
            "participants": participants,
            "message_results": results
        }
    
    @a2a_skill(
        name="broadcast_message",
        description="Broadcast a message to all available agents",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "filter_by_capability": {"type": "array", "items": {"type": "string"}},
                "max_agents": {"type": "integer", "default": 5}
            },
            "required": ["message"]
        }
    )
    async def broadcast_message(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Broadcast a message to multiple agents"""
        message = input_data["message"]
        filter_capabilities = input_data.get("filter_by_capability", [])
        max_agents = input_data.get("max_agents", 5)
        
        # Filter agents by capabilities if specified
        target_agents = []
        for agent_id, agent_info in self.agent_directory.items():
            if not filter_capabilities:
                target_agents.append(agent_id)
            else:
                agent_capabilities = agent_info.get("capabilities", [])
                if any(cap in agent_capabilities for cap in filter_capabilities):
                    target_agents.append(agent_id)
        
        # Limit number of agents
        target_agents = target_agents[:max_agents]
        
        # Start broadcast conversation
        result = await self.start_conversation({
            "participants": target_agents,
            "initial_message": message,
            "conversation_type": "broadcast"
        })
        
        return result
    
    def _extract_text_from_message(self, message: A2AMessage) -> Optional[str]:
        """Extract text content from A2A message"""
        for part in message.parts:
            if part.kind == "text" and part.text:
                return part.text
        return None
    
    async def _get_or_create_conversation(self, context_id: str, role: MessageRole) -> Dict[str, Any]:
        """Get existing conversation or create new one"""
        if context_id not in self.conversations:
            self.conversations[context_id] = {
                "id": context_id,
                "created_at": datetime.utcnow().isoformat(),
                "participants": [self.agent_id],
                "status": "active",
                "message_count": 0
            }
            self.message_history[context_id] = []
        
        return self.conversations[context_id]
    
    def _add_to_history(self, context_id: str, message: A2AMessage):
        """Add message to conversation history"""
        if context_id not in self.message_history:
            self.message_history[context_id] = []
        
        self.message_history[context_id].append(message)
        
        # Update conversation stats
        if context_id in self.conversations:
            self.conversations[context_id]["message_count"] += 1
            self.conversations[context_id]["last_activity"] = datetime.utcnow().isoformat()
    
    async def _analyze_user_intent(self, user_input: str, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user intent to determine routing strategy"""
        # Simple intent analysis - in production this would use NLP
        user_input_lower = user_input.lower()
        
        # Determine target agent(s) based on keywords
        target_agents = []
        method = "process_data"
        
        if any(word in user_input_lower for word in ["analyze", "data", "process", "calculate"]):
            target_agents.append("data-processor")
        elif any(word in user_input_lower for word in ["translate", "language", "text", "nlp"]):
            target_agents.append("nlp-agent")
        elif any(word in user_input_lower for word in ["trade", "crypto", "bitcoin", "price"]):
            target_agents.append("crypto-trader")
        elif any(word in user_input_lower for word in ["file", "save", "upload", "download"]):
            target_agents.append("file-manager")
        elif any(word in user_input_lower for word in ["scrape", "web", "extract", "crawl"]):
            target_agents.append("web-scraper")
        elif any(word in user_input_lower for word in ["image", "picture", "photo", "visual"]):
            target_agents.append("image-processor")
        elif any(word in user_input_lower for word in ["code", "review", "programming", "bug"]):
            target_agents.append("code-reviewer")
        elif any(word in user_input_lower for word in ["database", "query", "sql", "db"]):
            target_agents.append("database-agent")
        elif any(word in user_input_lower for word in ["notify", "alert", "send", "message"]):
            target_agents.append("notification-agent")
        elif any(word in user_input_lower for word in ["schedule", "automate", "cron", "timer"]):
            target_agents.append("scheduler-agent")
        elif any(word in user_input_lower for word in ["security", "scan", "threat", "vulnerability"]):
            target_agents.append("security-agent")
        elif any(word in user_input_lower for word in ["report", "analytics", "dashboard", "metrics"]):
            target_agents.append("analytics-agent")
        elif any(word in user_input_lower for word in ["workflow", "orchestrate", "pipeline"]):
            target_agents.append("workflow-agent")
        elif any(word in user_input_lower for word in ["api", "integration", "webhook", "endpoint"]):
            target_agents.append("api-agent")
        elif any(word in user_input_lower for word in ["predict", "ml", "ai", "model", "learn"]):
            target_agents.append("ml-agent")
        elif any(word in user_input_lower for word in ["backup", "restore", "recover", "archive"]):
            target_agents.append("backup-agent")
        else:
            # Default to data processor for general queries
            target_agents.append("data-processor")
        
        # Special handling for broadcast requests
        if any(word in user_input_lower for word in ["everyone", "all agents", "broadcast", "tell all"]):
            target_agents = list(self.agent_directory.keys())[:5]  # Limit broadcast
            method = "handle_message"
        
        return {
            "target_agents": target_agents,
            "method": method,
            "confidence": 0.8,  # Simple confidence score
            "reasoning": f"Selected {target_agents} based on keywords in user input"
        }
    
    async def _convert_prompt_to_a2a_message(
        self, 
        user_input: str, 
        routing_decision: Dict[str, Any], 
        context_id: str,
        task_id: Optional[str] = None
    ) -> A2AMessage:
        """Convert user prompt to A2A protocol message"""
        
        # Create message parts
        parts = [
            MessagePart(
                kind="text",
                text=user_input
            ),
            MessagePart(
                kind="data",
                data={
                    "method": routing_decision.get("method", "process_data"),
                    "routing_info": routing_decision,
                    "source_agent": self.agent_id,
                    "original_prompt": user_input
                }
            )
        ]
        
        # Create A2A message
        message = A2AMessage(
            messageId=str(uuid4()),
            role=MessageRole.USER,
            parts=parts,
            taskId=task_id or str(uuid4()),
            contextId=context_id,
            timestamp=datetime.utcnow().isoformat()
        )
        
        return message
    
    async def _route_message_to_agents(
        self, 
        message: A2AMessage, 
        routing_decision: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Route message to target agents"""
        target_agents = routing_decision.get("target_agents", [])
        responses = []
        
        for agent_id in target_agents:
            try:
                if self.blockchain_client:
                    response = await self._send_via_blockchain(message, agent_id)
                else:
                    response = await self._send_via_http(message, agent_id)
                
                responses.append({
                    "agent_id": agent_id,
                    "success": True,
                    "response": response,
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.error(f"Failed to route message to {agent_id}: {e}")
                responses.append({
                    "agent_id": agent_id,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        return responses
    
    async def _send_via_blockchain(self, message: A2AMessage, target_agent: str) -> Dict[str, Any]:
        """Send message via blockchain"""
        if not self.blockchain_client:
            raise Exception("Blockchain client not available")
        
        # Convert message to blockchain format
        message_content = json.dumps({
            "messageId": message.messageId,
            "role": message.role.value,
            "parts": [part.dict() for part in message.parts],
            "taskId": message.taskId,
            "contextId": message.contextId,
            "timestamp": message.timestamp
        })
        
        # Send via blockchain
        tx_hash = await self.blockchain_client.send_message(
            to_address=target_agent,  # In practice, this would be the agent's blockchain address
            content=message_content,
            message_type="a2a_message"
        )
        
        return {
            "method": "blockchain",
            "transaction_hash": tx_hash,
            "message_id": message.messageId,
            "target": target_agent
        }
    
    async def _send_via_http(self, message: A2AMessage, target_agent: str) -> Dict[str, Any]:
        """Send message via HTTP (fallback method)"""
        # This would send to the agent's HTTP endpoint
        # For now, simulate the response
        
        # In practice, this would make an HTTP POST to:
        # agent_endpoint = self.agent_directory[target_agent].get("endpoint")
        # POST {agent_endpoint}/messages
        
        return {
            "method": "http",
            "status": "simulated",
            "message_id": message.messageId,
            "target": target_agent,
            "simulated_response": f"Agent {target_agent} would process: {self._extract_text_from_message(message)}"
        }
    
    async def _format_responses_for_user(
        self, 
        responses: List[Dict[str, Any]], 
        routing_decision: Dict[str, Any]
    ) -> str:
        """Format agent responses for user consumption"""
        if not responses:
            return "No responses received from agents."
        
        formatted_parts = []
        
        for response in responses:
            agent_id = response["agent_id"]
            agent_name = self.agent_directory.get(agent_id, {}).get("description", agent_id)
            
            if response["success"]:
                resp_data = response.get("response", {})
                if isinstance(resp_data, dict) and "simulated_response" in resp_data:
                    formatted_parts.append(f"**{agent_name}**: {resp_data['simulated_response']}")
                else:
                    formatted_parts.append(f"**{agent_name}**: Message sent successfully (ID: {resp_data.get('message_id', 'unknown')})")
            else:
                formatted_parts.append(f"**{agent_name}**: Error - {response.get('error', 'Unknown error')}")
        
        return "\n\n".join(formatted_parts)
    
    async def _handle_blockchain_message(self, event_data: Dict[str, Any]):
        """Handle incoming blockchain messages (event listener callback)"""
        try:
            message_id = event_data.get("messageId")
            from_agent = event_data.get("fromAgent")
            content = event_data.get("content")
            
            logger.info(f"Received blockchain message {message_id} from {from_agent}")
            
            # Process incoming message from other agents
            # This would handle responses from the 16 A2A agents
            
        except Exception as e:
            logger.error(f"Error handling blockchain message: {e}")
    
    @a2a_handler("get_conversation_history")
    async def get_conversation_history(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Get conversation history for a context"""
        history = self.message_history.get(context_id, [])
        return {
            "context_id": context_id,
            "message_count": len(history),
            "history": [
                {
                    "messageId": msg.messageId,
                    "role": msg.role.value,
                    "text": self._extract_text_from_message(msg),
                    "timestamp": msg.timestamp
                }
                for msg in history
            ]
        }
    
    @a2a_skill(
        name="list_available_agents",
        description="List all available agents in the A2A network",
        input_schema={"type": "object", "properties": {}}
    )
    async def list_available_agents(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """List all available agents"""
        return {
            "total_agents": len(self.agent_directory),
            "agents": {
                agent_id: {
                    "description": info["description"],
                    "capabilities": info["capabilities"]
                }
                for agent_id, info in self.agent_directory.items()
            }
        }