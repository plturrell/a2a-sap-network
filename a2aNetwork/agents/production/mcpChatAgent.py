"""
MCP-enabled Production Chat Agent for A2A Network
Integrates Model Context Protocol for external tool access
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from uuid import uuid4
from contextlib import asynccontextmanager
import traceback

from ...sdk.agentBase import A2AAgentBase
from ...sdk.types import A2AMessage, MessagePart, MessageRole, TaskStatus
from ...sdk.decorators import a2a_handler, a2a_skill
from .chatStorage import ChatStorage

# MCP imports
try:
    from ....a2aAgents.backend.app.a2a.sdk.mcpServer import MCPServerMixin
    from ....a2aAgents.backend.app.a2a.sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt
    from ....a2aAgents.backend.app.a2a.sdk.mcpTypes import McpToolSchema, McpResourceInfo, McpPromptInfo
    MCP_AVAILABLE = True
except ImportError:
    # Fallback for missing MCP
    MCP_AVAILABLE = False
    class MCPServerMixin:
        pass
    def mcp_tool(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def mcp_resource(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def mcp_prompt(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Blockchain integration imports
try:
    from ...sdk.pythonSdk.blockchain.web3Client import Web3Client
    from ...sdk.pythonSdk.blockchain.eventListener import MessageEventListener
    BLOCKCHAIN_AVAILABLE = True
except ImportError:
    BLOCKCHAIN_AVAILABLE = False

# Authentication and rate limiting
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MCPChatAgent(A2AAgentBase, MCPServerMixin):
    """
    Production-grade A2A Chat Agent with MCP integration:
    
    Core Features:
    - Convert user prompts to A2A protocol messages
    - Route messages to appropriate agents via blockchain/HTTP
    - Persistent conversation storage with SQLite/PostgreSQL
    - Multi-user support with authentication
    - Real-time message delivery and notifications
    - Rate limiting and security controls
    
    MCP Features:
    - Expose chat operations as MCP tools
    - Provide conversation resources via MCP
    - Support external MCP clients
    - Cross-agent coordination through MCP
    - Real-time conversation monitoring
    """
    
    def __init__(
        self,
        agent_id: str = "mcp-chat-agent",
        name: str = "A2A MCP Chat Agent",
        description: str = "MCP-enabled chat agent for A2A network communication",
        version: str = "2.1.0",
        base_url: str = "http://localhost:8000",
        
        # Storage configuration
        storage_config: Optional[Dict[str, Any]] = None,
        
        # Blockchain configuration
        blockchain_config: Optional[Dict[str, Any]] = None,
        
        # Authentication configuration
        auth_config: Optional[Dict[str, Any]] = None,
        
        # Performance configuration
        performance_config: Optional[Dict[str, Any]] = None,
        
        # MCP configuration
        mcp_config: Optional[Dict[str, Any]] = None,
        
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
        self.mcp_config = mcp_config or {
            "host": "localhost",
            "port": 8080,
            "enable_http": True,
            "enable_websocket": True,
            "max_concurrent_mcp_requests": 20
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
            "average_response_time": 0.0,
            "mcp_requests": 0,
            "mcp_tool_calls": 0,
            "uptime_start": datetime.utcnow().isoformat()
        }
        
        # Active conversation tracking
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"MCPChatAgent initialized: {self.agent_id}")
    
    async def initialize(self) -> None:
        """Initialize MCP chat agent resources"""
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
            
            # Initialize MCP server if available
            if MCP_AVAILABLE:
                await self._init_mcp_server()
            
            # Load performance metrics
            await self._load_metrics()
            
            logger.info("MCPChatAgent initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCPChatAgent: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Cleanup MCP chat agent resources"""
        try:
            # Save metrics
            await self._save_metrics()
            
            # Stop MCP server
            if MCP_AVAILABLE and hasattr(self, 'mcp_server'):
                await self.stop_mcp_server()
            
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
            
            logger.info("MCPChatAgent shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def _init_mcp_server(self):
        """Initialize MCP server with chat-specific configuration"""
        if not MCP_AVAILABLE:
            logger.warning("MCP not available, skipping MCP server initialization")
            return
        
        try:
            # Initialize MCP server mixin
            self.initialize_mcp_server(
                host=self.mcp_config.get("host", "localhost"),
                port=self.mcp_config.get("port", 8080),
                enable_http=self.mcp_config.get("enable_http", True),
                enable_websocket=self.mcp_config.get("enable_websocket", True),
                max_concurrent_requests=self.mcp_config.get("max_concurrent_mcp_requests", 20)
            )
            
            # Start MCP server
            await self.start_mcp_server()
            
            logger.info(f"MCP server started on {self.mcp_config['host']}:{self.mcp_config['port']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            # Don't fail initialization if MCP server fails
    
    async def _init_blockchain(self):
        """Initialize blockchain connection with retry logic"""
        for attempt in range(self.performance_config["retry_attempts"]):
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
                return
                
            except Exception as e:
                logger.warning(f"Blockchain connection attempt {attempt + 1} failed: {e}")
                if attempt < self.performance_config["retry_attempts"] - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error("Failed to establish blockchain connection after all attempts")
                    self.blockchain_client = None
                    self.event_listener = None
    
    async def _discover_network_agents(self):
        """Discover available agents in the A2A network"""
        self.agent_directory = {
            "data-processor": {
                "description": "Advanced data processing and analytics",
                "capabilities": ["data_analysis", "transformation", "aggregation", "cleaning"],
                "specialties": ["large_datasets", "real_time_processing", "statistical_analysis"],
                "endpoint": "http://localhost:8001",
                "blockchain_address": "0x1234567890123456789012345678901234567890",
                "performance_score": 4.8,
                "availability": "99.9%",
                "mcp_enabled": True
            },
            "nlp-agent": {
                "description": "Natural language processing and understanding",
                "capabilities": ["text_analysis", "language_detection", "sentiment_analysis", "translation"],
                "specialties": ["multilingual", "context_understanding", "entity_extraction"],
                "endpoint": "http://localhost:8002",
                "blockchain_address": "0x2345678901234567890123456789012345678901",
                "performance_score": 4.7,
                "availability": "99.8%",
                "mcp_enabled": True
            },
            "crypto-trader": {
                "description": "Cryptocurrency trading and market analysis",
                "capabilities": ["trading", "market_analysis", "portfolio_management", "risk_assessment"],
                "specialties": ["DeFi", "automated_trading", "technical_analysis"],
                "endpoint": "http://localhost:8003",
                "blockchain_address": "0x3456789012345678901234567890123456789012",
                "performance_score": 4.9,
                "availability": "99.7%",
                "mcp_enabled": True
            },
            "file-manager": {
                "description": "File operations and document management",
                "capabilities": ["file_ops", "storage", "compression", "format_conversion"],
                "specialties": ["cloud_integration", "version_control", "security"],
                "endpoint": "http://localhost:8004",
                "blockchain_address": "0x4567890123456789012345678901234567890123",
                "performance_score": 4.6,
                "availability": "99.9%",
                "mcp_enabled": False
            },
            "web-scraper": {
                "description": "Web scraping and content extraction",
                "capabilities": ["scraping", "data_extraction", "crawling", "monitoring"],
                "specialties": ["dynamic_content", "anti_bot_bypass", "structured_data"],
                "endpoint": "http://localhost:8005",
                "blockchain_address": "0x5678901234567890123456789012345678901234",
                "performance_score": 4.5,
                "availability": "99.6%",
                "mcp_enabled": True
            },
            "image-processor": {
                "description": "Image analysis and computer vision",
                "capabilities": ["image_analysis", "computer_vision", "ocr", "enhancement"],
                "specialties": ["object_detection", "facial_recognition", "medical_imaging"],
                "endpoint": "http://localhost:8006",
                "blockchain_address": "0x6789012345678901234567890123456789012345",
                "performance_score": 4.8,
                "availability": "99.5%",
                "mcp_enabled": True
            },
            "code-reviewer": {
                "description": "Code analysis and security review",
                "capabilities": ["code_analysis", "security_review", "bug_detection", "optimization"],
                "specialties": ["vulnerability_scanning", "performance_analysis", "best_practices"],
                "endpoint": "http://localhost:8007",
                "blockchain_address": "0x7890123456789012345678901234567890123456",
                "performance_score": 4.7,
                "availability": "99.8%",
                "mcp_enabled": True
            },
            "database-agent": {
                "description": "Database operations and query optimization",
                "capabilities": ["database", "sql", "optimization", "migration"],
                "specialties": ["multi_db_support", "performance_tuning", "data_modeling"],
                "endpoint": "http://localhost:8008",
                "blockchain_address": "0x8901234567890123456789012345678901234567",
                "performance_score": 4.6,
                "availability": "99.9%",
                "mcp_enabled": False
            },
            "notification-agent": {
                "description": "Multi-channel notifications and alerts",
                "capabilities": ["notifications", "messaging", "alerts", "scheduling"],
                "specialties": ["multi_channel", "smart_routing", "escalation"],
                "endpoint": "http://localhost:8009",
                "blockchain_address": "0x9012345678901234567890123456789012345678",
                "performance_score": 4.5,
                "availability": "99.7%",
                "mcp_enabled": True
            },
            "scheduler-agent": {
                "description": "Task scheduling and workflow automation",
                "capabilities": ["scheduling", "automation", "workflow", "cron"],
                "specialties": ["complex_workflows", "conditional_logic", "retry_mechanisms"],
                "endpoint": "http://localhost:8010",
                "blockchain_address": "0x0123456789012345678901234567890123456789",
                "performance_score": 4.8,
                "availability": "99.9%",
                "mcp_enabled": True
            },
            "security-agent": {
                "description": "Security monitoring and threat detection",
                "capabilities": ["security", "threat_detection", "monitoring", "incident_response"],
                "specialties": ["real_time_monitoring", "AI_threat_detection", "forensics"],
                "endpoint": "http://localhost:8011",
                "blockchain_address": "0x1234567890123456789012345678901234567891",
                "performance_score": 4.9,
                "availability": "99.8%",
                "mcp_enabled": True
            },
            "analytics-agent": {
                "description": "Advanced analytics and business intelligence",
                "capabilities": ["analytics", "reporting", "visualization", "predictions"],
                "specialties": ["real_time_dashboards", "predictive_analytics", "custom_reports"],
                "endpoint": "http://localhost:8012",
                "blockchain_address": "0x2345678901234567890123456789012345678902",
                "performance_score": 4.7,
                "availability": "99.6%",
                "mcp_enabled": True
            },
            "workflow-agent": {
                "description": "Enterprise workflow orchestration",
                "capabilities": ["orchestration", "workflow", "integration", "monitoring"],
                "specialties": ["enterprise_integration", "parallel_processing", "error_handling"],
                "endpoint": "http://localhost:8013",
                "blockchain_address": "0x3456789012345678901234567890123456789013",
                "performance_score": 4.8,
                "availability": "99.7%",
                "mcp_enabled": True
            },
            "api-agent": {
                "description": "API integration and management",
                "capabilities": ["api_integration", "webhooks", "rate_limiting", "transformation"],
                "specialties": ["REST_GraphQL", "authentication", "load_balancing"],
                "endpoint": "http://localhost:8014",
                "blockchain_address": "0x4567890123456789012345678901234567890124",
                "performance_score": 4.6,
                "availability": "99.8%",
                "mcp_enabled": False
            },
            "ml-agent": {
                "description": "Machine learning and AI model operations",
                "capabilities": ["machine_learning", "prediction", "training", "inference"],
                "specialties": ["deep_learning", "model_deployment", "AutoML"],
                "endpoint": "http://localhost:8015",
                "blockchain_address": "0x5678901234567890123456789012345678901235",
                "performance_score": 4.9,
                "availability": "99.5%",
                "mcp_enabled": True
            },
            "backup-agent": {
                "description": "Data backup and disaster recovery",
                "capabilities": ["backup", "recovery", "replication", "archiving"],
                "specialties": ["incremental_backup", "cross_region", "encryption"],
                "endpoint": "http://localhost:8016",
                "blockchain_address": "0x6789012345678901234567890123456789012346",
                "performance_score": 4.7,
                "availability": "99.9%",
                "mcp_enabled": False
            }
        }
        logger.info(f"Discovered {len(self.agent_directory)} agents in network")
    
    async def _load_metrics(self):
        """Load performance metrics from storage"""
        # Implementation would load from persistent storage
        pass
    
    async def _save_metrics(self):
        """Save performance metrics to storage"""
        # Implementation would save to persistent storage
        pass
    
    async def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limits"""
        if not self.redis_client:
            return True  # No rate limiting if Redis unavailable
        
        try:
            # Check messages per minute
            key = f"rate_limit:{user_id}:messages"
            current = await self.redis_client.get(key)
            
            if current and int(current) >= 30:  # 30 messages per minute
                logger.warning(f"Rate limit exceeded for user {user_id}")
                return False
            
            # Increment counter
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, 60)  # 1 minute TTL
            await pipe.execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Fail open
    
    # ===========================================
    # MCP TOOLS
    # ===========================================
    
    @mcp_tool(
        name="send_message_to_agent",
        description="Send a message to a specific A2A agent and get response",
        input_schema={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message content to send to the agent"
                },
                "target_agent": {
                    "type": "string",
                    "description": "ID of the target agent",
                    "enum": list(["data-processor", "nlp-agent", "crypto-trader", "file-manager", "web-scraper", "image-processor", "code-reviewer", "database-agent", "notification-agent", "scheduler-agent", "security-agent", "analytics-agent", "workflow-agent", "api-agent", "ml-agent", "backup-agent"])
                },
                "user_id": {
                    "type": "string",
                    "description": "User ID for conversation tracking"
                },
                "context_id": {
                    "type": "string",
                    "description": "Optional conversation context ID"
                },
                "use_blockchain": {
                    "type": "boolean",
                    "description": "Whether to send via blockchain",
                    "default": False
                }
            },
            "required": ["message", "target_agent", "user_id"]
        }
    )
    async def send_message_to_agent_mcp(
        self, 
        message: str, 
        target_agent: str, 
        user_id: str,
        context_id: Optional[str] = None,
        use_blockchain: bool = False
    ) -> Dict[str, Any]:
        """MCP tool to send message to specific A2A agent"""
        self.metrics["mcp_tool_calls"] += 1
        
        try:
            # Check rate limits
            if not await self._check_rate_limit(user_id):
                return {"error": "Rate limit exceeded", "success": False}
            
            # Validate target agent
            if target_agent not in self.agent_directory:
                return {"error": f"Unknown agent: {target_agent}", "success": False}
            
            # Use existing send_prompt_to_agent skill
            result = await self.send_prompt_to_agent({
                "prompt": message,
                "target_agent": target_agent,
                "context_id": context_id or str(uuid4()),
                "use_blockchain": use_blockchain
            })
            
            # Store in conversation if context provided
            if context_id and self.storage:
                # Create message objects for storage
                user_message = A2AMessage(
                    role=MessageRole.USER,
                    parts=[MessagePart(kind="text", text=message)],
                    contextId=context_id
                )
                await self.storage.save_message(context_id, user_message, "sent")
                
                # Create agent response message
                if result.get("response"):
                    agent_message = A2AMessage(
                        role=MessageRole.AGENT,
                        parts=[MessagePart(kind="text", text=str(result["response"]))],
                        contextId=context_id
                    )
                    await self.storage.save_message(context_id, agent_message, "received")
            
            return {
                "success": True,
                "agent_response": result,
                "target_agent": target_agent,
                "message_sent": message,
                "context_id": context_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"MCP send_message_to_agent failed: {e}")
            return {"error": str(e), "success": False}
    
    @mcp_tool(
        name="start_conversation",
        description="Start a new conversation with one or more A2A agents",
        input_schema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User ID starting the conversation"
                },
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of agent IDs to include in conversation"
                },
                "initial_message": {
                    "type": "string",
                    "description": "Initial message to send to all participants"
                },
                "conversation_title": {
                    "type": "string",
                    "description": "Optional title for the conversation"
                },
                "conversation_type": {
                    "type": "string",
                    "enum": ["direct", "group", "broadcast"],
                    "description": "Type of conversation",
                    "default": "group"
                }
            },
            "required": ["user_id", "participants", "initial_message"]
        }
    )
    async def start_conversation_mcp(
        self,
        user_id: str,
        participants: List[str],
        initial_message: str,
        conversation_title: Optional[str] = None,
        conversation_type: str = "group"
    ) -> Dict[str, Any]:
        """MCP tool to start a new conversation"""
        self.metrics["mcp_tool_calls"] += 1
        
        try:
            # Check rate limits
            if not await self._check_rate_limit(user_id):
                return {"error": "Rate limit exceeded", "success": False}
            
            # Validate participants
            invalid_agents = [agent for agent in participants if agent not in self.agent_directory]
            if invalid_agents:
                return {"error": f"Unknown agents: {invalid_agents}", "success": False}
            
            # Create conversation
            conversation_id = await self.storage.create_conversation(
                user_id=user_id,
                title=conversation_title or f"Conversation with {', '.join(participants)}",
                conversation_type=conversation_type,
                participants=participants
            )
            
            # Track active conversation
            self.active_conversations[conversation_id] = {
                "user_id": user_id,
                "participants": participants,
                "created_at": datetime.utcnow().isoformat(),
                "message_count": 0,
                "last_activity": datetime.utcnow().isoformat()
            }
            
            # Send initial message to all participants
            results = []
            for agent_id in participants:
                try:
                    result = await self.send_message_to_agent_mcp(
                        message=initial_message,
                        target_agent=agent_id,
                        user_id=user_id,
                        context_id=conversation_id
                    )
                    results.append({"agent_id": agent_id, "result": result})
                except Exception as e:
                    results.append({"agent_id": agent_id, "error": str(e)})
            
            self.metrics["total_conversations"] += 1
            
            return {
                "success": True,
                "conversation_id": conversation_id,
                "participants": participants,
                "initial_message": initial_message,
                "agent_responses": results,
                "created_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"MCP start_conversation failed: {e}")
            return {"error": str(e), "success": False}
    
    @mcp_tool(
        name="broadcast_to_agents",
        description="Broadcast a message to multiple A2A agents based on capabilities",
        input_schema={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to broadcast"
                },
                "user_id": {
                    "type": "string",
                    "description": "User ID sending the broadcast"
                },
                "capability_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter agents by capabilities"
                },
                "max_agents": {
                    "type": "integer",
                    "description": "Maximum number of agents to broadcast to",
                    "default": 5,
                    "maximum": 10
                },
                "mcp_only": {
                    "type": "boolean",
                    "description": "Only broadcast to MCP-enabled agents",
                    "default": False
                }
            },
            "required": ["message", "user_id"]
        }
    )
    async def broadcast_to_agents_mcp(
        self,
        message: str,
        user_id: str,
        capability_filter: Optional[List[str]] = None,
        max_agents: int = 5,
        mcp_only: bool = False
    ) -> Dict[str, Any]:
        """MCP tool to broadcast message to multiple agents"""
        self.metrics["mcp_tool_calls"] += 1
        
        try:
            # Check rate limits
            if not await self._check_rate_limit(user_id):
                return {"error": "Rate limit exceeded", "success": False}
            
            # Filter agents
            target_agents = []
            for agent_id, agent_info in self.agent_directory.items():
                # Filter by MCP capability if requested
                if mcp_only and not agent_info.get("mcp_enabled", False):
                    continue
                
                # Filter by capabilities if specified
                if capability_filter:
                    agent_capabilities = agent_info.get("capabilities", [])
                    if not any(cap in agent_capabilities for cap in capability_filter):
                        continue
                
                target_agents.append(agent_id)
                
                if len(target_agents) >= max_agents:
                    break
            
            if not target_agents:
                return {"error": "No agents match the specified criteria", "success": False}
            
            # Create broadcast conversation
            conversation_id = await self.storage.create_conversation(
                user_id=user_id,
                title=f"Broadcast to {len(target_agents)} agents",
                conversation_type="broadcast",
                participants=target_agents
            )
            
            # Send to all target agents concurrently
            tasks = []
            for agent_id in target_agents:
                task = self.send_message_to_agent_mcp(
                    message=message,
                    target_agent=agent_id,
                    user_id=user_id,
                    context_id=conversation_id
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            agent_responses = []
            successful_count = 0
            for i, result in enumerate(results):
                agent_id = target_agents[i]
                if isinstance(result, Exception):
                    agent_responses.append({
                        "agent_id": agent_id,
                        "success": False,
                        "error": str(result)
                    })
                else:
                    agent_responses.append({
                        "agent_id": agent_id,
                        "success": result.get("success", False),
                        "response": result.get("agent_response")
                    })
                    if result.get("success"):
                        successful_count += 1
            
            return {
                "success": True,
                "conversation_id": conversation_id,
                "message": message,
                "target_agents": target_agents,
                "total_agents": len(target_agents),
                "successful_responses": successful_count,
                "agent_responses": agent_responses,
                "filters_applied": {
                    "capability_filter": capability_filter,
                    "mcp_only": mcp_only,
                    "max_agents": max_agents
                }
            }
            
        except Exception as e:
            logger.error(f"MCP broadcast_to_agents failed: {e}")
            return {"error": str(e), "success": False}
    
    @mcp_tool(
        name="coordinate_multi_agent_task",
        description="Coordinate a complex task across multiple A2A agents with different strategies",
        input_schema={
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "Description of the task to coordinate"
                },
                "user_id": {
                    "type": "string",
                    "description": "User ID coordinating the task"
                },
                "agent_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of specific agents to coordinate"
                },
                "coordination_strategy": {
                    "type": "string",
                    "enum": ["sequential", "parallel", "pipeline"],
                    "description": "How to coordinate the agents",
                    "default": "parallel"
                },
                "task_data": {
                    "type": "object",
                    "description": "Additional task-specific data"
                }
            },
            "required": ["task_description", "user_id", "agent_list"]
        }
    )
    async def coordinate_multi_agent_task_mcp(
        self,
        task_description: str,
        user_id: str,
        agent_list: List[str],
        coordination_strategy: str = "parallel",
        task_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """MCP tool for complex multi-agent task coordination"""
        self.metrics["mcp_tool_calls"] += 1
        
        try:
            # Check rate limits
            if not await self._check_rate_limit(user_id):
                return {"error": "Rate limit exceeded", "success": False}
            
            # Validate agents
            invalid_agents = [agent for agent in agent_list if agent not in self.agent_directory]
            if invalid_agents:
                return {"error": f"Unknown agents: {invalid_agents}", "success": False}
            
            # Create coordination conversation
            conversation_id = await self.storage.create_conversation(
                user_id=user_id,
                title=f"Multi-agent task: {task_description[:50]}...",
                conversation_type="group",
                participants=agent_list,
                settings={"coordination_strategy": coordination_strategy}
            )
            
            start_time = time.time()
            agent_results = {}
            
            if coordination_strategy == "sequential":
                # Execute agents one by one
                for agent_id in agent_list:
                    result = await self.send_message_to_agent_mcp(
                        message=f"Task: {task_description}\nData: {json.dumps(task_data or {})}",
                        target_agent=agent_id,
                        user_id=user_id,
                        context_id=conversation_id
                    )
                    agent_results[agent_id] = result
                    
                    # If this agent failed, decide whether to continue
                    if not result.get("success"):
                        break
            
            elif coordination_strategy == "parallel":
                # Execute all agents simultaneously
                tasks = []
                for agent_id in agent_list:
                    task = self.send_message_to_agent_mcp(
                        message=f"Task: {task_description}\nData: {json.dumps(task_data or {})}",
                        target_agent=agent_id,
                        user_id=user_id,
                        context_id=conversation_id
                    )
                    tasks.append((agent_id, task))
                
                results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
                
                for i, result in enumerate(results):
                    agent_id = tasks[i][0]
                    if isinstance(result, Exception):
                        agent_results[agent_id] = {"success": False, "error": str(result)}
                    else:
                        agent_results[agent_id] = result
            
            elif coordination_strategy == "pipeline":
                # Execute agents in pipeline (output of one becomes input of next)
                current_data = task_data or {}
                
                for i, agent_id in enumerate(agent_list):
                    message = f"Task: {task_description}\nPipeline stage {i+1}\nData: {json.dumps(current_data)}"
                    
                    result = await self.send_message_to_agent_mcp(
                        message=message,
                        target_agent=agent_id,
                        user_id=user_id,
                        context_id=conversation_id
                    )
                    
                    agent_results[agent_id] = result
                    
                    # Extract output for next stage
                    if result.get("success") and result.get("agent_response"):
                        # Try to extract structured data from response
                        try:
                            response_data = result["agent_response"].get("response", {})
                            if isinstance(response_data, dict):
                                current_data.update(response_data)
                        except:
                            # If no structured data, pass the raw response
                            current_data["previous_response"] = str(result["agent_response"])
                    else:
                        # Pipeline failed
                        break
            
            execution_time = time.time() - start_time
            successful_agents = sum(1 for result in agent_results.values() if result.get("success"))
            
            return {
                "success": True,
                "conversation_id": conversation_id,
                "task_description": task_description,
                "coordination_strategy": coordination_strategy,
                "agent_list": agent_list,
                "agent_results": agent_results,
                "execution_time_seconds": execution_time,
                "successful_agents": successful_agents,
                "total_agents": len(agent_list),
                "completion_rate": successful_agents / len(agent_list) if agent_list else 0
            }
            
        except Exception as e:
            logger.error(f"MCP coordinate_multi_agent_task failed: {e}")
            return {"error": str(e), "success": False}
    
    @mcp_tool(
        name="get_conversation_history",
        description="Retrieve conversation history for a user",
        input_schema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User ID to get conversations for"
                },
                "conversation_id": {
                    "type": "string",
                    "description": "Specific conversation ID (optional)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of conversations to return",
                    "default": 20,
                    "maximum": 100
                },
                "include_messages": {
                    "type": "boolean",
                    "description": "Include message content in response",
                    "default": False
                }
            },
            "required": ["user_id"]
        }
    )
    async def get_conversation_history_mcp(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        limit: int = 20,
        include_messages: bool = False
    ) -> Dict[str, Any]:
        """MCP tool to get conversation history"""
        self.metrics["mcp_tool_calls"] += 1
        
        try:
            if conversation_id:
                # Get specific conversation
                messages = await self.storage.get_conversation_messages(
                    conversation_id=conversation_id,
                    include_responses=True
                )
                return {
                    "success": True,
                    "conversation_id": conversation_id,
                    "messages": messages,
                    "message_count": len(messages)
                }
            else:
                # Get user's conversations
                conversations = await self.storage.get_conversations(
                    user_id=user_id,
                    limit=min(limit, 100)
                )
                
                # Optionally include messages
                if include_messages:
                    for conv in conversations:
                        conv["messages"] = await self.storage.get_conversation_messages(
                            conversation_id=conv["conversation_id"],
                            limit=10  # Limit messages per conversation
                        )
                
                return {
                    "success": True,
                    "user_id": user_id,
                    "conversations": conversations,
                    "total_conversations": len(conversations),
                    "include_messages": include_messages
                }
                
        except Exception as e:
            logger.error(f"MCP get_conversation_history failed: {e}")
            return {"error": str(e), "success": False}
    
    # ===========================================
    # MCP RESOURCES
    # ===========================================
    
    @mcp_resource(
        uri="conversation://active",
        name="Active Conversations",
        description="Real-time information about currently active conversations",
        mime_type="application/json"
    )
    async def get_active_conversations_resource(self) -> Dict[str, Any]:
        """MCP resource for active conversations"""
        return {
            "active_conversations": self.active_conversations,
            "total_active": len(self.active_conversations),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @mcp_resource(
        uri="agents://directory",
        name="Agent Directory",
        description="Information about all available A2A agents",
        mime_type="application/json"
    )
    async def get_agent_directory_resource(self) -> Dict[str, Any]:
        """MCP resource for agent directory"""
        return {
            "agents": self.agent_directory,
            "total_agents": len(self.agent_directory),
            "mcp_enabled_agents": len([a for a in self.agent_directory.values() if a.get("mcp_enabled")]),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @mcp_resource(
        uri="system://metrics",
        name="System Metrics",
        description="Performance and usage metrics for the chat agent",
        mime_type="application/json"
    )
    async def get_system_metrics_resource(self) -> Dict[str, Any]:
        """MCP resource for system metrics"""
        uptime = datetime.utcnow() - datetime.fromisoformat(self.metrics["uptime_start"])
        
        return {
            "metrics": self.metrics,
            "uptime_seconds": uptime.total_seconds(),
            "active_conversations": len(self.active_conversations),
            "system_status": {
                "blockchain_connected": self.blockchain_client is not None,
                "redis_connected": self.redis_client is not None,
                "storage_connected": self.storage is not None,
                "mcp_enabled": MCP_AVAILABLE
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # ===========================================
    # MCP PROMPTS
    # ===========================================
    
    @mcp_prompt(
        name="agent_coordination_prompt",
        description="Generate a prompt for coordinating multiple agents on a task",
        arguments=[
            {"name": "task", "description": "The task to coordinate", "required": True},
            {"name": "agents", "description": "List of agent IDs", "required": True},
            {"name": "strategy", "description": "Coordination strategy", "required": False}
        ]
    )
    async def agent_coordination_prompt(self, task: str, agents: List[str], strategy: str = "parallel") -> str:
        """Generate a coordination prompt for multi-agent tasks"""
        agent_info = []
        for agent_id in agents:
            if agent_id in self.agent_directory:
                info = self.agent_directory[agent_id]
                agent_info.append(f"- {agent_id}: {info['description']} (capabilities: {', '.join(info['capabilities'])})")
        
        return f"""Multi-Agent Task Coordination

Task: {task}
Strategy: {strategy}

Available Agents:
{chr(10).join(agent_info)}

Coordination Instructions:
- Each agent should focus on their specialized capabilities
- {'Execute agents sequentially, passing results between stages' if strategy == 'pipeline' else 'Execute agents in parallel for maximum efficiency' if strategy == 'parallel' else 'Execute agents one by one'}
- Aggregate results from all agents into a comprehensive response
- Handle any agent failures gracefully

Begin coordination..."""
    
    @mcp_prompt(
        name="conversation_summary_prompt",
        description="Generate a summary prompt for a conversation",
        arguments=[
            {"name": "conversation_id", "description": "Conversation ID to summarize", "required": True},
            {"name": "max_messages", "description": "Maximum messages to include", "required": False}
        ]
    )
    async def conversation_summary_prompt(self, conversation_id: str, max_messages: int = 50) -> str:
        """Generate a conversation summary prompt"""
        try:
            messages = await self.storage.get_conversation_messages(
                conversation_id=conversation_id,
                limit=max_messages,
                include_responses=True
            )
            
            summary_content = []
            for msg in messages[-10:]:  # Last 10 messages
                role = msg['role'].upper()
                content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
                summary_content.append(f"{role}: {content}")
            
            return f"""Conversation Summary Request

Conversation ID: {conversation_id}
Total Messages: {len(messages)}
Recent Messages:
{chr(10).join(summary_content)}

Please provide a concise summary of this conversation, highlighting:
1. Main topics discussed
2. Key decisions or outcomes
3. Participants involved
4. Current status or next steps"""
        
        except Exception as e:
            return f"Error generating conversation summary: {str(e)}"
    
    # ===========================================
    # EXISTING A2A HANDLERS (for backward compatibility)
    # ===========================================
    
    @a2a_handler("start_chat")
    async def start_new_chat(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """A2A handler to start new chat (calls MCP tool internally)"""
        # Extract data from message
        user_data = None
        initial_prompt = ""
        
        for part in message.parts:
            if part.kind == "data" and part.data:
                user_data = part.data
            elif part.kind == "text" and part.text:
                initial_prompt = part.text
        
        if not user_data or "user_id" not in user_data:
            return {"error": "user_id required to start chat", "success": False}
        
        # Use MCP tool
        result = await self.start_conversation_mcp(
            user_id=user_data["user_id"],
            participants=user_data.get("participants", ["data-processor"]),
            initial_message=initial_prompt or "Hello, I'd like to start a conversation",
            conversation_title=user_data.get("title"),
            conversation_type=user_data.get("type", "direct")
        )
        
        return result
    
    @a2a_handler("send_message")
    async def send_chat_message(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """A2A handler to send chat message (calls MCP tool internally)"""
        # Extract user data
        user_data = None
        message_text = ""
        
        for part in message.parts:
            if part.kind == "data" and part.data:
                user_data = part.data
            elif part.kind == "text" and part.text:
                message_text = part.text
        
        if not user_data or "user_id" not in user_data:
            return {"error": "user_id required", "success": False}
        
        target_agent = user_data.get("target_agent", "data-processor")
        
        # Use MCP tool
        result = await self.send_message_to_agent_mcp(
            message=message_text,
            target_agent=target_agent,
            user_id=user_data["user_id"],
            context_id=context_id,
            use_blockchain=user_data.get("use_blockchain", False)
        )
        
        return result
    
    async def _handle_blockchain_message(self, event_data: Dict[str, Any]):
        """Handle incoming blockchain messages"""
        try:
            message_id = event_data.get("messageId")
            from_agent = event_data.get("fromAgent")
            content = event_data.get("content")
            
            logger.info(f"Received blockchain message {message_id} from {from_agent}")
            
            # Process incoming message from other agents
            # This would handle responses from the 16 A2A agents
            
        except Exception as e:
            logger.error(f"Error handling blockchain message: {e}")
    
    def _extract_text_from_message(self, message: A2AMessage) -> Optional[str]:
        """Extract text content from A2A message"""
        for part in message.parts:
            if part.kind == "text" and part.text:
                return part.text
        return None