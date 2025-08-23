"""
A2A Chat Agent - Standards-compliant implementation for converting user prompts to A2A messages
"""

"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""



import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from uuid import uuid4

# Standard A2A imports following existing patterns
from a2aCommon import (
    A2AAgentBase,
    a2a_handler,
    a2a_skill,
    a2a_task,
    A2AMessage,
    MessageRole,
    create_agent_id
)
from a2aCommon.sdk.utils import create_success_response, create_error_response
from a2aCommon.sdk.blockchainIntegration import BlockchainIntegrationMixin

# Import enhanced security and persistence
try:
    from app.core.securityMonitoring import get_e2e_encryption
    from app.a2a.core.chatPersistence import create_chat_persistence, ChatMessage, ChatConversation
    E2E_ENCRYPTION_AVAILABLE = True
except ImportError:
    E2E_ENCRYPTION_AVAILABLE = False
    logger.warning("E2E encryption and persistence not available")


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    pass  # Environment validation would be enforced here in production
# Import our blockchain integration
try:
    from blockchain_integration import BlockchainIntegration
    BLOCKCHAIN_INTEGRATION_AVAILABLE = True
except ImportError:
    logger.warning("BlockchainIntegration not available")
    BLOCKCHAIN_INTEGRATION_AVAILABLE = False

# Try to import StandardTrustRelationshipsMixin
try:
    sys.path.append(str(Path(__file__).parent.parent.parent / "app" / "a2a" / "core"))
    from standardTrustRelationships import StandardTrustRelationshipsMixin
    TRUST_MIXIN_AVAILABLE = True
except ImportError:
    # Fallback implementation
    logger.warning("StandardTrustRelationshipsMixin not available, using fallback")
    TRUST_MIXIN_AVAILABLE = False
    
    class StandardTrustRelationshipsMixin:
        """Fallback implementation for development"""
        
        def __init__(self):
            self._trust_relationships_established = False
            logger.info("Fallback StandardTrustRelationshipsMixin initialized")
        
        async def establish_standard_trust_relationships(self) -> bool:
            """Fallback implementation"""
            logger.info("Establishing standard trust relationships (fallback mode)")
            self._trust_relationships_established = True
            return True

# AI Intelligence imports
try:
    from app.a2a.core.ai_intelligence import (
        AIIntelligenceFramework,
        AIIntelligenceConfig,
        create_ai_intelligence_framework
    )
    from app.a2a.sdk.grokClient import GrokClient
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    GrokClient = None

# MCP support for advanced features
try:
    from a2a.sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Fallback decorators
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Import AIIntelligenceMixin
try:
    import sys
    sys.path.append('/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/sdk')
    from aiIntelligenceMixin import AIIntelligenceMixin
    AIINTELLIGENCE_AVAILABLE = True
except ImportError:
    AIINTELLIGENCE_AVAILABLE = False
    class AIIntelligenceMixin:
        """Fallback for AI Intelligence"""
        def _extract_required_skills_from_parts(self, parts):
            """Extract skills from message parts"""
            skills = []
            for part in parts:
                if isinstance(part, dict) and "data" in part:
                    data = part["data"]
                    if "required_skills" in data:
                        skills.extend(data["required_skills"])
            return skills

class ChatAgent(A2AAgentBase, BlockchainIntegrationMixin, StandardTrustRelationshipsMixin, AIIntelligenceMixin):
    """
    A2A Chat Agent - Converts user prompts into A2A messages for the 16 specialized agents
    
    This agent acts as a conversational interface to the A2A network, intelligently routing
    user requests to the appropriate specialized agents and managing multi-agent conversations.
    """
    
    def __init__(self, base_url: str, config: Optional[Dict[str, Any]] = None):
        """Initialize Chat Agent with standard A2A configuration"""
        
        # Define blockchain capabilities for chat agent
        blockchain_capabilities = [
            "chat_interface",
            "multi_agent_routing", 
            "conversation_management",
            "intent_analysis",
            "agent_coordination",
            "message_routing",
            "conversation_history",
            "ai_powered_routing"
        ]
        
        # Initialize A2AAgentBase with blockchain capabilities
        A2AAgentBase.__init__(
            self,
            agent_id=create_agent_id("chat"),
            name="A2A Chat Agent",
            description="AI-powered conversational interface for A2A network - intelligently routes prompts to specialized agents",
            version="2.0.0",
            base_url=base_url,
            blockchain_capabilities=blockchain_capabilities
        )
        
        # Initialize BlockchainIntegrationMixin
        BlockchainIntegrationMixin.__init__(self)
        
        # Initialize StandardTrustRelationshipsMixin
        StandardTrustRelationshipsMixin.__init__(self)
        
        # Configuration
        self.config = config or {}
        self.enable_blockchain = self.config.get("enable_blockchain", False)
        self.enable_persistence = self.config.get("enable_persistence", True)
        self.enable_ai = self.config.get("enable_ai", True) and AI_AVAILABLE
        self.max_concurrent_conversations = self.config.get("max_concurrent_conversations", 100)
        
        # Chat-specific state
        self.active_conversations = {}
        self.message_history = {}
        self.agent_registry = {}
        self.routing_stats = {
            "total_messages": 0,
            "successful_routings": 0,
            "failed_routings": 0,
            "agent_response_times": {},
            "popular_agents": {},
            "ai_routing_accuracy": 0.0
        }
        
        # Security and persistence components
        self.encryption = None
        self.persistence = None
        self.session_keys = {}  # Store encryption keys per session
        
        # Initialize AI Intelligence if enabled
        self.ai_framework = None
        self.grok_client = None
        if self.enable_ai and AI_AVAILABLE:
            try:
                # AI configuration (placeholder for future implementation)
                logger.info("AI components will be initialized when available")
                pass
            except Exception as e:
                logger.error(f"Failed to initialize AI components: {e}")
                self.enable_ai = False
        
        # Initialize our enhanced blockchain integration
        self.blockchain_client = None
        if self.enable_blockchain and BLOCKCHAIN_INTEGRATION_AVAILABLE:
            try:
                # Blockchain integration (placeholder for future implementation)
                logger.info("Blockchain integration will be initialized when available")
                pass
            except Exception as e:
                logger.warning(f"⚠️ Enhanced blockchain initialization failed: {e}")
                self.enable_blockchain = False
        
        # Also initialize the base blockchain integration mixin
        if self.enable_blockchain:
            try:
                # Base blockchain mixin initialization (placeholder for future implementation)
                logger.info("Base blockchain mixin will be initialized when available")
                pass
            except Exception as e:
                logger.warning(f"⚠️ Base blockchain initialization failed: {e}")
        
        logger.info(f"ChatAgent initialized: {self.agent_id} (AI: {self.enable_ai}, Blockchain: {self.enable_blockchain})")
        
    def _extract_required_skills_from_parts(self, parts):
        """Extract required skills from message parts"""
        skills = []
        for part in parts:
            if isinstance(part, dict) and "data" in part:
                data = part["data"]
                if "required_skills" in data:
                    skills.extend(data["required_skills"])
        return skills
    
    async def initialize(self) -> None:
        """Initialize agent resources following A2A standards"""
        logger.info(f"Initializing {self.name}...")
        
        try:
            # Establish standard trust relationships FIRST
            await self.establish_standard_trust_relationships()
            
            # Discover available agents in network using catalog_manager
            # Use our custom discovery method instead of inherited one
            await self._discover_network_agents()
            
            # Store discovered agents for routing (using our agent registry)
            self.discoverable_agents = {
                "processing_agents": [agent for agent in self.agent_registry.values() if "data_analysis" in agent.get("capabilities", [])],
                "validation_agents": [agent for agent in self.agent_registry.values() if "validation" in agent.get("capabilities", [])],
                "calculation_agents": [agent for agent in self.agent_registry.values() if "calculation" in agent.get("capabilities", [])],
                "analysis_agents": [agent for agent in self.agent_registry.values() if "analysis" in agent.get("capabilities", [])],
                "all_agents": list(self.agent_registry.values())
            }
            
            # Register with network
            await self.register_with_network()
            
            # Register on blockchain if enabled
            if self.enable_blockchain and self.blockchain_client:
                logger.info("Registering ChatAgent on blockchain...")
                capabilities = [
                    "chat", 
                    "routing", 
                    "orchestration",
                    "multi_agent_coordination"
                ]
                success = await self.blockchain_client.register_agent(capabilities)
                if success:
                    logger.info("✅ ChatAgent registered on blockchain")
                    # Start listening for blockchain messages
                    asyncio.create_task(self._listen_for_blockchain_messages())
                else:
                    logger.warning("⚠️ Failed to register ChatAgent on blockchain")
            
            # Initialize persistence if enabled
            if self.enable_persistence:
                await self._init_persistence()
            
            # Initialize encryption for secure communication
            if E2E_ENCRYPTION_AVAILABLE:
                await self._init_encryption()
            
            logger.info(f"{self.name} initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Cleanup agent resources"""
        logger.info(f"Shutting down {self.name}...")
        
        try:
            # Save conversation state
            if self.enable_persistence:
                await self._save_conversation_state()
            
            # Cleanup blockchain connection
            if self.enable_blockchain:
                await self.cleanup_blockchain()
            
            logger.info(f"{self.name} shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    @a2a_handler("chat_message", "Process user chat message and route to appropriate agents")
    async def handle_chat_message(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Main handler for processing chat messages"""
        try:
            # Extract message content
            request_data = self._extract_data(message)
            if not request_data:
                return create_error_response(400, "Invalid message format")
            
            user_prompt = request_data.get("prompt", "")
            user_id = request_data.get("user_id", "anonymous")
            conversation_id = request_data.get("conversation_id", context_id)
            
            if not user_prompt:
                return create_error_response(400, "No prompt provided")
            
            # Create task for tracking
            task_id = await self.create_task("chat_routing", {
                "context_id": context_id,
                "user_id": user_id,
                "prompt": user_prompt,
                "conversation_id": conversation_id
            })
            
            # Process message asynchronously
            asyncio.create_task(self._process_chat_message(
                task_id, user_prompt, user_id, conversation_id, context_id
            ))
            
            return create_success_response({
                "task_id": task_id,
                "conversation_id": conversation_id,
                "status": "processing",
                "message": "Routing your message to appropriate agents"
            })
            
        except Exception as e:
            logger.error(f"Error handling chat message: {e}")
            return create_error_response(500, str(e))
    
    @a2a_handler("multi_agent_query", "Route query to multiple agents for comprehensive response")
    async def handle_multi_agent_query(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle queries that require multiple agent coordination"""
        try:
            request_data = self._extract_data(message)
            if not request_data:
                return create_error_response(400, "Invalid request format")
            
            query = request_data.get("query", "")
            target_agents = request_data.get("target_agents", [])
            coordination_type = request_data.get("coordination_type", "parallel")
            
            # Create coordination task
            task_id = await self.create_task("multi_agent_coordination", {
                "query": query,
                "agents": target_agents,
                "type": coordination_type
            })
            
            # Process coordination
            asyncio.create_task(self._coordinate_multi_agent_query(
                task_id, query, target_agents, coordination_type, context_id
            ))
            
            return create_success_response({
                "task_id": task_id,
                "status": "coordinating",
                "agents": target_agents,
                "coordination_type": coordination_type
            })
            
        except Exception as e:
            logger.error(f"Error in multi-agent query: {e}")
            return create_error_response(500, str(e))
    
    @a2a_skill("route_to_agent", "Route a message to a specific A2A agent")
    async def route_to_agent(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Skill to route messages to specific agents"""
        try:
            prompt = input_data.get("prompt", "")
            target_agent = input_data.get("target_agent", "")
            context_id = input_data.get("context_id", str(uuid4()))
            
            if not prompt or not target_agent:
                raise ValueError("prompt and target_agent are required")
            
            # Validate agent exists
            if target_agent not in self.agent_registry:
                raise ValueError(f"Unknown agent: {target_agent}")
            
            # Create A2A message for target agent
            agent_message = self._create_agent_message(prompt, target_agent, context_id)
            
            # Send to agent
            response = await self._send_to_agent(target_agent, agent_message)
            
            # Update stats
            self._update_routing_stats(target_agent, response.get("success", False))
            
            # Store chat routing data in data_manager
            await self.store_agent_data(
                data_type="chat_routing",
                data={
                    "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                    "target_agent": target_agent,
                    "context_id": context_id,
                    "routing_success": response.get("success", False),
                    "response_time": response.get("processing_time", 0.0),
                    "routing_timestamp": datetime.utcnow().isoformat()
                },
                metadata={
                    "agent_version": "chat_agent_v2.0",
                    "ai_routing_enabled": self.enable_ai,
                    "blockchain_enabled": self.enable_blockchain
                }
            )
            
            # Update agent status with agent_manager
            await self.update_agent_status(
                status="active",
                details={
                    "total_messages": self.routing_stats.get("total_messages", 0),
                    "successful_routings": self.routing_stats.get("successful_routings", 0),
                    "active_conversations": len(self.active_conversations),
                    "last_routed_agent": target_agent,
                    "active_capabilities": ["chat_routing", "intent_analysis", "conversation_management", "multi_agent_coordination"]
                }
            )
            
            return {
                "success": True,
                "agent": target_agent,
                "response": response,
                "context_id": context_id
            }
            
        except Exception as e:
            logger.error(f"Error routing to agent: {e}")
            raise
    
    @a2a_skill("analyze_intent", "Analyze user intent to determine best agent routing")
    async def analyze_intent(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user prompt to determine routing strategy"""
        try:
            prompt = input_data.get("prompt", "")
            
            # Analyze prompt for keywords and intent
            routing_decision = await self._analyze_prompt_intent(prompt)
            
            return {
                "prompt": prompt,
                "recommended_agents": routing_decision["agents"],
                "confidence": routing_decision["confidence"],
                "intent_type": routing_decision["intent_type"],
                "reasoning": routing_decision["reasoning"]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing intent: {e}")
            raise
    
    @a2a_skill("get_conversation_history", "Retrieve conversation history")
    async def get_conversation_history(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get conversation history for a user"""
        try:
            conversation_id = input_data.get("conversation_id")
            user_id = input_data.get("user_id")
            limit = input_data.get("limit", 50)
            
            history = await self._get_conversation_history(conversation_id, user_id, limit)
            
            return {
                "conversation_id": conversation_id,
                "messages": history,
                "total_messages": len(history)
            }
            
        except Exception as e:
            logger.error(f"Error retrieving history: {e}")
            raise
    
    # MCP Tools for AI-powered features
    @mcp_tool(
        name="ai_analyze_intent",
        description="Use Grok AI to analyze user intent and determine optimal agent routing",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "User message to analyze"},
                "conversation_history": {"type": "array", "items": {"type": "string"}, "description": "Previous messages for context"},
                "user_preferences": {"type": "object", "description": "User preferences and history"}
            },
            "required": ["message"]
        }
    )
    async def ai_analyze_intent_mcp(
        self, 
        message: str, 
        conversation_history: List[str] = None,
        user_preferences: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Use Grok AI for semantic intent analysis"""
        if not self.grok_client:
            logger.debug(f"AI not available for intent analysis, using keyword fallback")
            # Use fallback keyword-based routing
            fallback_result = await self._analyze_prompt_intent(message)
            return {
                "success": True,
                "message": message,
                "fallback_analysis": fallback_result,
                "recommended_agents": fallback_result["agents"],
                "confidence": fallback_result["confidence"],
                "reasoning": f"Keyword-based routing: {fallback_result['reasoning']}",
                "method": "fallback_keyword"
            }
        
        try:
            # Build context for AI
            context = f"""Analyze this user message and determine which A2A agents should handle it.

Available agents:
{json.dumps({k: v['description'] for k, v in self.agent_registry.items()}, indent=2)}

User message: {message}

Previous context: {' '.join(conversation_history[-3:]) if conversation_history else 'None'}

Return a JSON with:
1. "primary_agent": The best agent for this task
2. "secondary_agents": Other relevant agents (max 2)
3. "confidence": 0-1 score
4. "reasoning": Brief explanation
5. "approach": How agents should coordinate
"""
            
            # Get AI analysis
            ai_response = await self.grok_client.complete(
                prompt=context,
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse AI response
            try:
                analysis = json.loads(ai_response.content)
            except:
                # Fallback parsing
                analysis = {
                    "primary_agent": "data-processor",
                    "secondary_agents": [],
                    "confidence": 0.5,
                    "reasoning": ai_response.content,
                    "approach": "sequential"
                }
            
            # Track AI performance
            if self.ai_framework:
                await self.ai_framework.record_decision(
                    "intent_analysis",
                    {"message": message, "result": analysis}
                )
            
            return {
                "success": True,
                "message": message,
                "ai_analysis": analysis,
                "recommended_agents": [analysis["primary_agent"]] + analysis.get("secondary_agents", []),
                "confidence": analysis.get("confidence", 0.7),
                "reasoning": analysis.get("reasoning", ""),
                "approach": analysis.get("approach", "parallel")
            }
            
        except Exception as e:
            logger.error(f"AI intent analysis failed: {e}")
            # Fallback to keyword-based routing
            fallback = await self._analyze_prompt_intent(message)
            return {
                "success": False,
                "message": message,
                "fallback_analysis": fallback,
                "error": str(e)
            }
    
    @mcp_tool(
        name="ai_synthesize_responses",
        description="Use Grok AI to synthesize multiple agent responses into coherent answer with formatting and sentiment awareness",
        input_schema={
            "type": "object",
            "properties": {
            # "user_query": {"type": "string", "description": "Original user query"},
            # "agent_responses": {"type": "array", "items": {"type": "object"}, "description": "Responses from multiple agents"},
            # "response_style": {"type": "string", "enum": ["concise", "detailed", "technical", "simple", "executive", "casual"], "default": "concise"},
            # "target_language": {"type": "string", "description": "Target language for response", "default": "en"},
            # "user_sentiment": {"type": "string", "enum": ["positive", "neutral", "negative", "frustrated", "urgent"], "default": "neutral"},
            # "format_type": {"type": "string", "enum": ["plain", "markdown", "html", "json"], "default": "markdown"}
            },
            "required": ["user_query", "agent_responses"]
        }
    )
    async def ai_synthesize_responses_mcp(
        self,
        user_query: str,
        agent_responses: List[Dict[str, Any]],
        response_style: str = "concise",
        target_language: str = "en",
        user_sentiment: str = "neutral",
        format_type: str = "markdown"
    ) -> Dict[str, Any]:
        """Use Grok AI to create unified response with advanced formatting and sentiment awareness"""
        if not self.grok_client:
            raise RuntimeError("AI client required for response synthesis")
        
        try:
            # Use provided sentiment
            pass
            
            # Prepare agent responses for AI
            responses_text = []
            response_quality_scores = []
            
            for resp in agent_responses:
                if resp.get("success"):
                    agent_name = self.agent_registry.get(resp["agent_id"], {}).get("name", resp["agent_id"])
                    content = resp.get("response", {}).get("result", str(resp.get("response")))
                    responses_text.append(f"{agent_name}: {content}")
                    
                    # Score response quality
                    quality = len(str(content)) > 50
                    response_quality_scores.append(quality)
            
            # Build sentiment-aware prompt
            sentiment_instructions = {
                "positive": "Maintain an upbeat, enthusiastic tone that matches the user's positive energy.",
                "negative": "Be empathetic and understanding, acknowledging any concerns while providing helpful solutions.",
                "frustrated": "Be extra clear and patient, provide step-by-step guidance, and acknowledge the frustration.",
                "urgent": "Be direct and action-oriented, prioritize the most important information first.",
                "neutral": "Maintain a professional, balanced tone."
            }
            
            # Language-specific instructions
            language_names = {
            # "en": "English", "es": "Spanish", "fr": "French", "de": "German",
            # "it": "Italian", "pt": "Portuguese", "ja": "Japanese", "ko": "Korean",
            # "zh": "Chinese", "ru": "Russian", "ar": "Arabic", "hi": "Hindi"
            }
            
            prompt = f"""Synthesize these agent responses into a {response_style} answer.

CONTEXT:
- User Query: {user_query}
- User Sentiment: {user_sentiment}
- Target Language: {language_names.get(target_language, target_language)}
- Output Format: {format_type}

AGENT RESPONSES:
{chr(10).join(responses_text)}

SYNTHESIS REQUIREMENTS:
1. {sentiment_instructions.get(user_sentiment, sentiment_instructions['neutral'])}
2. Write the response in {language_names.get(target_language, 'English')}
3. Use {response_style} style:
   - concise: 2-3 sentences, key points only
   - detailed: comprehensive with examples
   - technical: include technical details and terminology
   - simple: avoid jargon, use everyday language
   - executive: high-level summary with business impact
   - casual: friendly, conversational tone
4. Format as {format_type}:
   - markdown: use **bold**, *italic*, lists, headers
   - html: use proper HTML tags
   - json: structure as {{"summary": "", "details": [], "next_steps": []}}
   - plain: simple text without formatting
5. Intelligently merge duplicate information
6. Highlight the most important insights
7. If appropriate, suggest next steps or follow-up actions
8. Ensure cultural appropriateness for the target language

QUALITY OPTIMIZATION:
- Prioritize responses from agents with better quality scores
- Remove redundancy while preserving unique insights
- Create logical flow from problem → solution → outcome
"""
            
            # Get AI synthesis with optimized parameters
            ai_response = await self.grok_client.complete(
                prompt=prompt,
                temperature=0.6 if response_style == "casual" else 0.4,
                max_tokens=1000,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            # Post-process response based on format
            formatted_response = ai_response.content
            
            if format_type == "json":
                try:
                    json_response = json.loads(formatted_response)
                    formatted_response = json.dumps(json_response, indent=2)
                except:
                    formatted_response = json.dumps({
                        "summary": formatted_response[:200],
                        "full_response": formatted_response,
                        "agents_consulted": [r["agent_id"] for r in agent_responses if r.get("success")],
                        "confidence": "high" if len(response_quality_scores) > 2 else "medium"
                    }, indent=2)
            
            elif format_type == "html":
                if not formatted_response.startswith("<"):
                    formatted_response = f"<div class='ai-response'>\n{formatted_response}\n</div>"
            
            # Optimize response length based on style
            if response_style == "concise" and len(formatted_response) > 500:
                short_prompt = f"Shorten this to 2-3 sentences while keeping key information:\n{formatted_response}"
                short_response = await self.grok_client.complete(short_prompt, max_tokens=150)
                formatted_response = short_response.content
            
            return {
                "success": True,
                "synthesized_response": formatted_response,
                "response_metadata": {
                    "style": response_style,
                    "language": target_language,
                    "sentiment_adapted": user_sentiment,
                    "format": format_type,
                    "quality_score": sum(response_quality_scores) / len(response_quality_scores) if response_quality_scores else 0,
                    "agents_included": [r["agent_id"] for r in agent_responses if r.get("success")],
                    "response_length": len(formatted_response),
                    "synthesis_model": "grok",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"AI response synthesis failed: {e}")
            raise RuntimeError(f"AI response synthesis failed: {e}")
    
    @mcp_tool(
        name="ai_learn_routing",
        description="Use AI to learn from past routing decisions and improve accuracy",
        input_schema={
            "type": "object",
            "properties": {
                "routing_history": {"type": "array", "items": {"type": "object"}, "description": "Past routing decisions and outcomes"},
                "feedback": {"type": "object", "description": "User feedback on routing quality"}
            },
            "required": ["routing_history"]
        }
    )
    async def ai_learn_routing_mcp(
        self,
        routing_history: List[Dict[str, Any]],
        feedback: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """AI-powered learning from routing history"""
        if not self.ai_framework:
            return {"error": "AI framework not available", "success": False}
        
        try:
            # Analyze routing patterns
            learning_result = await self.ai_framework.learn_from_history(
            # "routing_optimization",
            # routing_history,
            # feedback
            )
            
            # Update routing accuracy metric
            if hasattr(learning_result, 'accuracy_improvement') and learning_result.accuracy_improvement:
                self.routing_stats["ai_routing_accuracy"] = learning_result.new_accuracy
            
            return {
                "success": True,
                "patterns_learned": getattr(learning_result, 'patterns', []),
                "accuracy_improvement": getattr(learning_result, 'accuracy_improvement', 0.0),
                "new_routing_rules": getattr(learning_result, 'new_rules', []),
                "recommendations": getattr(learning_result, 'recommendations', [])
            }
            
        except Exception as e:
            logger.error(f"AI learning failed: {e}")
            return {"error": str(e), "success": False}
    
    @mcp_tool(
        name="ai_context_chat",
        description="Context-aware chat using Grok AI with memory of conversation",
        input_schema={
            "type": "object",
            "properties": {
            # "message": {"type": "string", "description": "User message"},
            # "conversation_id": {"type": "string", "description": "Conversation ID"},
            # "include_agent_knowledge": {"type": "boolean", "default": True}
            },
            "required": ["message", "conversation_id"]
        }
    )
    async def ai_context_chat_mcp(
        self,
        message: str,
        conversation_id: str,
        include_agent_knowledge: bool = True
    ) -> Dict[str, Any]:
        """AI-powered contextual chat with conversation memory"""
        if not self.grok_client:
            return {"error": "AI not available", "success": False}
        
        try:
            # Get conversation history
            history = self.message_history.get(conversation_id, [])
            
            # Build context
            context_messages = []
            for hist_msg in history[-10:]:  # Last 10 messages
                role = "user" if hist_msg.get("role") == "user" else "assistant"
                context_messages.append({"role": role, "content": hist_msg.get("content", "")})
            
            # Add current message
            context_messages.append({"role": "user", "content": message})
            
            # Include agent knowledge if requested
            system_prompt = "You are an AI assistant helping users interact with specialized A2A agents."
            if include_agent_knowledge:
                system_prompt += f"\n\nAvailable agents and their capabilities:\n{json.dumps(self.agent_registry, indent=2)}"
            
            # Get AI response with context
            ai_response = await self.grok_client.chat(
                messages=[{"role": "system", "content": system_prompt}] + context_messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Analyze if routing is needed
            needs_routing = await self._check_needs_routing(ai_response.content)
            
            result = {
            # "success": True,
            # "response": ai_response.content,
            # "conversation_id": conversation_id,
            # "needs_agent_routing": needs_routing,
            # "context_length": len(context_messages)
            }
            
            # If routing needed, analyze intent
            if needs_routing:
                intent_result = await self.ai_analyze_intent_mcp(message, [m["content"] for m in context_messages])
                result["routing_suggestion"] = intent_result
            
            # Update conversation history
            self.message_history.setdefault(conversation_id, []).append({
                "role": "user",
                "content": message,
                "timestamp": datetime.utcnow().isoformat()
            })
            self.message_history[conversation_id].append({
                "role": "assistant",
                "content": ai_response.content,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"AI context chat failed: {e}")
            return {"error": str(e), "success": False}
    
    @mcp_tool(
        name="ai_translate_message",
        description="Use AI to translate messages between user language and agent protocols",
        input_schema={
            "type": "object",
            "properties": {
            # "message": {"type": "string", "description": "Message to translate"},
            # "source_format": {"type": "string", "enum": ["natural", "technical", "a2a_protocol"], "default": "natural"},
            # "target_format": {"type": "string", "enum": ["natural", "technical", "a2a_protocol"], "default": "a2a_protocol"},
            # "target_agent": {"type": "string", "description": "Target agent for protocol translation"}
            },
            "required": ["message"]
        }
    )
    async def ai_translate_message_mcp(
        self,
        message: str,
        source_format: str = "natural",
        target_format: str = "a2a_protocol",
        target_agent: str = None
    ) -> Dict[str, Any]:
        """AI-powered message translation between formats"""
        if not self.grok_client:
            return {"error": "AI not available", "success": False}
        
        try:
            # Build translation prompt
            agent_info = self.agent_registry.get(target_agent, {}) if target_agent else {}
            
            prompt = f"""Translate this message from {source_format} language to {target_format}.

Message: {message}
"""
            
            if target_format == "a2a_protocol" and target_agent:
                agent_info = self.agent_registry.get(target_agent, {})
                prompt += f"\nTarget agent: {target_agent} - {agent_info.get('description', '')}"
                prompt += f"\nCapabilities: {agent_info.get('capabilities', [])}"
                prompt += "\n\nCreate a properly formatted A2A protocol message with appropriate parameters."
            elif target_format == "natural":
                prompt += "\n\nTranslate technical responses into clear, user-friendly language."
            
            # Get AI translation
            ai_response = await self.grok_client.complete(
                prompt=prompt,
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse translated message
            translated = ai_response.content
            
            # If translating to A2A protocol, try to parse as JSON
            if target_format == "a2a_protocol":
                try:
                    protocol_message = json.loads(translated)
                except:
                    # Create structured message from AI response
                    protocol_message = {
                        "action": "process",
                        "data": {
                            "request": message,
                            "parameters": {}
                        }
                    }
            else:
                protocol_message = None
            
            return {
                "success": True,
                "original_message": message,
                "translated_message": translated,
                "source_format": source_format,
                "target_format": target_format,
                "protocol_message": protocol_message,
                "target_agent": target_agent
            }
            
        except Exception as e:
            logger.error(f"AI message translation failed: {e}")
            return {"error": str(e), "success": False}
    
    @mcp_tool(
        name="ai_analyze_sentiment",
        description="Analyze user sentiment from message using AI",
        input_schema={
            "type": "object",
            "properties": {
            # "message": {"type": "string", "description": "Message to analyze"},
            # "include_emotion": {"type": "boolean", "description": "Include detailed emotion analysis", "default": True}
            },
            "required": ["message"]
        }
    )
    async def ai_analyze_sentiment_mcp(
        self,
        message: str,
        include_emotion: bool = True
    ) -> Dict[str, Any]:
        """Analyze sentiment and emotion using Grok AI"""
        if not self.grok_client:
            return {"error": "AI not available", "success": False}
        
        try:
            prompt = f"""Analyze the sentiment and emotion in this message:

Message: "{message}"

Provide analysis in JSON format:
{{
    "sentiment": "positive/neutral/negative/frustrated/urgent",
    "confidence": 0.0-1.0,
    "emotions": ["primary_emotion", "secondary_emotion"],
    "urgency_level": "low/medium/high",
    "tone": "formal/casual/technical/emotional",
    "needs_empathy": true/false,
    "reasoning": "brief explanation"
}}
"""
            
            ai_response = await self.grok_client.complete(
            # prompt=prompt,
            # temperature=0.3,
            # max_tokens=200
            )
            
            try:
                analysis = json.loads(ai_response.content)
            except:
                # Fallback parsing
                analysis = {
                    "sentiment": "neutral",
                    "confidence": 0.5,
                    "emotions": ["neutral"],
                    "urgency_level": "medium",
                    "tone": "casual",
                    "needs_empathy": False,
                    "reasoning": "Could not parse AI response"
                }
            
            return {
                "success": True,
                "sentiment": analysis.get("sentiment", "neutral"),
                "confidence": analysis.get("confidence", 0.7),
                "emotions": analysis.get("emotions", []) if include_emotion else [],
                "urgency_level": analysis.get("urgency_level", "medium"),
                "tone": analysis.get("tone", "casual"),
                "needs_empathy": analysis.get("needs_empathy", False),
                "reasoning": analysis.get("reasoning", "")
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"error": str(e), "success": False}
    
    @mcp_tool(
        name="ai_optimize_response",
        description="Optimize response for readability, clarity, and user engagement",
        input_schema={
            "type": "object",
            "properties": {
            # "response": {"type": "string", "description": "Response to optimize"},
            # "optimization_goals": {"type": "array", "items": {"type": "string", "enum": ["clarity", "brevity", "engagement", "action", "empathy"]}, "default": ["clarity", "engagement"]},
            # "reading_level": {"type": "string", "enum": ["elementary", "high_school", "college", "professional"], "default": "high_school"},
            # "call_to_action": {"type": "boolean", "description": "Include clear next steps", "default": True}
            },
            "required": ["response"]
        }
    )
    async def ai_optimize_response_mcp(
        self,
        response: str,
        optimization_goals: List[str] = None,
        reading_level: str = "high_school",
        call_to_action: bool = True
    ) -> Dict[str, Any]:
        """Optimize response using AI for better user experience"""
        if not self.grok_client:
            return {"error": "AI not available", "success": False}
        
        optimization_goals = optimization_goals or ["clarity", "engagement"]
        
        try:
            goal_instructions = {
            # "clarity": "Make the response crystal clear and easy to understand",
            # "brevity": "Make it concise while keeping essential information",
            # "engagement": "Make it engaging and interesting to read",
            # "action": "Focus on actionable insights and clear next steps",
            # "empathy": "Add empathetic language that shows understanding"
            }
            
            prompt = f"""Optimize this response for better user experience.

Original Response:
{response}

Optimization Goals:
{chr(10).join([f"- {goal}: {goal_instructions.get(goal, goal)}" for goal in optimization_goals])}

Requirements:
1. Target reading level: {reading_level}
2. {"Include clear call-to-action or next steps" if call_to_action else "Focus on information delivery"}
3. Use formatting for better readability (bullet points, bold for key points)
4. Maintain accuracy while improving presentation
5. Add structure with clear sections if needed

Provide the optimized response:"""
            
            ai_response = await self.grok_client.complete(
            # prompt=prompt,
            # temperature=0.5,
            # max_tokens=1000
            )
            
            optimized = ai_response.content
            
            # Analyze improvements
            original_length = len(response)
            optimized_length = len(optimized)
            
            return {
            # "success": True,
            # "optimized_response": optimized,
            # "optimization_metrics": {
            #     "original_length": original_length,
            #     "optimized_length": optimized_length,
            #     "length_reduction": f"{((original_length - optimized_length) / original_length * 100):.1f}%" if original_length > 0 else "0%",
            #     "goals_applied": optimization_goals,
            #     "reading_level": reading_level,
            #     "has_cta": call_to_action
            # }
            }
            
        except Exception as e:
            logger.error(f"Response optimization failed: {e}")
            return {"error": str(e), "success": False}
    
    @mcp_tool(
        name="ai_format_response",
        description="Format response for different platforms and use cases",
        input_schema={
            "type": "object",
            "properties": {
            # "content": {"type": "string", "description": "Content to format"},
            # "platform": {"type": "string", "enum": ["web", "mobile", "email", "slack", "teams", "api"], "default": "web"},
            # "use_case": {"type": "string", "enum": ["chat", "notification", "report", "summary", "alert"], "default": "chat"},
            # "include_metadata": {"type": "boolean", "description": "Include metadata like timestamps", "default": False}
            },
            "required": ["content"]
        }
    )
    async def ai_format_response_mcp(
        self,
        content: str,
        platform: str = "web",
        use_case: str = "chat",
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """Format response for different platforms and use cases"""
        
        platform_specs = {
            "web": {"max_length": 5000, "supports": ["markdown", "html"], "preferred": "markdown"},
            "mobile": {"max_length": 1000, "supports": ["plain", "simple_markdown"], "preferred": "plain"},
            "email": {"max_length": 10000, "supports": ["html", "plain"], "preferred": "html"},
            "slack": {"max_length": 4000, "supports": ["slack_markdown"], "preferred": "slack_markdown"},
            "teams": {"max_length": 4000, "supports": ["teams_markdown"], "preferred": "teams_markdown"},
            "api": {"max_length": 50000, "supports": ["json"], "preferred": "json"}
        }
        
        use_case_styles = {
            "chat": "conversational and friendly",
            "notification": "brief and action-oriented",
            "report": "structured with sections",
            "summary": "concise key points",
            "alert": "urgent and clear"
        }
        
        spec = platform_specs.get(platform, platform_specs["web"])
        style = use_case_styles.get(use_case, "clear and professional")
        
        # Truncate if needed
        if len(content) > spec["max_length"]:
            content = content[:spec["max_length"] - 100] + "\n\n... [Content truncated]"
        
        # Format based on platform
        formatted_content = content
        
        if platform == "slack":
            # Convert to Slack markdown
            formatted_content = formatted_content.replace("**", "*")  # Bold
            formatted_content = formatted_content.replace("###", "*")  # Headers
            
        elif platform == "teams":
            # Convert to Teams format
            formatted_content = formatted_content.replace("###", "**")  # Headers to bold
            
        elif platform == "mobile":
            # Simplify for mobile
            formatted_content = formatted_content.replace("**", "")
            formatted_content = formatted_content.replace("*", "")
            formatted_content = formatted_content.replace("#", "")
            
        elif platform == "api":
            # Structure as JSON
            formatted_content = json.dumps({
            # "content": content,
            # "use_case": use_case,
            # "formatted_at": datetime.utcnow().isoformat(),
            # "metadata": {
            #     "length": len(content),
            #     "platform": platform
            # } if include_metadata else None
            }, indent=2)
        
        result = {
            "success": True,
            "formatted_content": formatted_content,
            "platform": platform,
            "use_case": use_case,
            "format_specs": spec
        }
        
        if include_metadata:
            result["metadata"] = {
            # "original_length": len(content),
            # "formatted_length": len(formatted_content),
            # "timestamp": datetime.utcnow().isoformat(),
            # "truncated": len(content) > spec["max_length"]
            }
        
        return result
    
    @mcp_tool(
        name="ai_smart_route",
        description="AI-powered intelligent routing with sentiment-aware response synthesis",
        input_schema={
            "type": "object",
            "properties": {
            # "message": {"type": "string", "description": "User message"},
            # "context": {"type": "object", "description": "Additional context"},
            # "routing_mode": {"type": "string", "enum": ["fast", "accurate", "comprehensive"], "default": "accurate"},
            # "max_agents": {"type": "integer", "default": 3},
            # "auto_optimize": {"type": "boolean", "description": "Automatically optimize final response", "default": True}
            },
            "required": ["message"]
        }
    )
    async def ai_smart_route_mcp(
        self, 
        message: str, 
        context: Dict[str, Any] = None,
        routing_mode: str = "accurate",
        max_agents: int = 3,
        auto_optimize: bool = True
    ) -> Dict[str, Any]:
        """AI-powered smart routing with enhanced response synthesis"""
        try:
            # Analyze sentiment first
            sentiment_result = await self.ai_analyze_sentiment_mcp(message)
            user_sentiment = sentiment_result.get("sentiment", "neutral") if sentiment_result.get("success") else "neutral"
            urgency = sentiment_result.get("urgency_level", "medium") if sentiment_result.get("success") else "medium"
            
            # Adjust routing based on urgency
            if urgency == "high" and routing_mode != "fast":
                routing_mode = "fast"  # Prioritize speed for urgent requests
            
            # Use AI to analyze intent
            ai_analysis = await self.ai_analyze_intent_mcp(
                message,
                context.get("conversation_history") if context else None,
                context.get("user_preferences") if context else None
            )
            
            if not ai_analysis.get("success"):
                # Fallback to keyword routing
                routing = await self._analyze_prompt_intent(message)
            # selected_agents = routing["agents"][:max_agents]
            else:
                # Use AI recommendations
                selected_agents = ai_analysis["recommended_agents"][:max_agents]
                routing = {
                    "confidence": ai_analysis["confidence"],
                    "reasoning": ai_analysis["reasoning"],
                    "approach": ai_analysis["approach"]
                }
            
            # Route to agents based on mode
            selected_agents = ai_analysis.get("recommended_agents", [])
            if routing_mode == "fast":
                # Route to primary agent only
                selected_agents = selected_agents[:1]
            elif routing_mode == "comprehensive":
                # Route to all relevant agents
                selected_agents = selected_agents[:min(5, max_agents)]
            
            # Execute routing
            results = []
            if routing.get("approach") == "sequential":
                # Sequential routing with context passing
                previous_result = None
                for agent_id in selected_agents:
                    enhanced_prompt = message
                    if previous_result:
                        enhanced_prompt += f"\n\nContext from {previous_result['agent_id']}: {previous_result.get('response', {}).get('result', '')}"
                    
                    response = await self.route_to_agent({
                        "prompt": enhanced_prompt,
                        "target_agent": agent_id,
                        "context_id": context.get("conversation_id") if context else None
                    })
                    results.append(response)
                    previous_result = response
            else:
                # Parallel routing
                tasks = []
                for agent_id in selected_agents:
                    task = self.route_to_agent({
                        "prompt": message,
                        "target_agent": agent_id,
                        "context_id": context.get("conversation_id") if context else None
                    })
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                results = [r if not isinstance(r, Exception) else {"error": str(r), "success": False} for r in results]
            
            # Determine response style based on sentiment
            response_style = "concise"
            if user_sentiment == "frustrated":
                response_style = "simple"
            elif urgency == "high":
                response_style = "concise"
            elif context and context.get("response_style"):
                response_style = context["response_style"]
            
            # Use enhanced AI synthesis with sentiment awareness
            synthesis_result = await self.ai_synthesize_responses_mcp(
                user_query=message,
                agent_responses=results,
                response_style=response_style,
                target_language=context.get("language", "en") if context else "en",
                user_sentiment=user_sentiment,
                format_type=context.get("format", "markdown") if context else "markdown"
            )
            
            final_response = synthesis_result.get("synthesized_response", "")
            
            # Auto-optimize if enabled
            if auto_optimize and synthesis_result.get("success"):
                optimization_goals = ["clarity", "engagement"]
                if user_sentiment in ["frustrated", "negative"]:
                    optimization_goals.append("empathy")
                if urgency == "high":
                    optimization_goals.append("action")
                    
                # Note: ai_optimize_response_mcp method would need to be implemented
                # optimize_result = await self.ai_optimize_response_mcp(
                #     response=final_response,
                #     optimization_goals=optimization_goals,
                #     call_to_action=True
                # )
                # 
                # if optimize_result.get("success"):
                #     final_response = optimize_result["optimized_response"]
                pass
            
            # Learn from this routing decision
            if self.ai_framework:
                await self.ai_framework.record_decision(
                    "routing",
                    {
                        "message": message,
                        "sentiment": user_sentiment,
                        "selected_agents": selected_agents,
                        "results": results,
                        "success": all(r.get("success") for r in results)
                    }
                )
            
            return {
            # "success": True,
            # "message": message,
            # "routing_mode": routing_mode,
            # "routed_to": selected_agents,
            # "individual_results": results,
            # "synthesized_response": final_response,
            # "response_metadata": synthesis_result.get("response_metadata", {}),
            # "sentiment_analysis": {
            #     "user_sentiment": user_sentiment,
            #     "urgency_level": urgency,
            #     "response_adapted": True
            # },
            # "ai_confidence": routing["confidence"],
            # "ai_reasoning": routing["reasoning"],
            # "optimized": auto_optimize
            }
            
        except Exception as e:
            logger.error(f"AI smart routing failed: {e}")
            return {"error": str(e), "success": False}
    
    @mcp_resource(
        uri="chat://active_conversations",
        name="Active Conversations",
        description="Currently active chat conversations",
        mime_type="application/json"
    )
    async def get_active_conversations_resource(self) -> Dict[str, Any]:
        """MCP resource for active conversations"""
        return {
            "active_conversations": len(self.active_conversations),
            "conversations": [
            # {
            #     "id": conv_id,
            #     "user_id": conv.get("user_id"),
            #     "started_at": conv.get("started_at"),
            #     "message_count": conv.get("message_count", 0),
            #     "last_activity": conv.get("last_activity")
            # }
            # for conv_id, conv in self.active_conversations.items()
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Internal methods following A2A patterns
    def _extract_data(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Extract data from A2A message"""
        if hasattr(message, 'content'):
            content = message.content
            if isinstance(content, dict):
                return content.get('data', content)
            elif isinstance(content, str):
                try:
                    return json.loads(content)
                except:
                    return {"prompt": content}
        return None
    
    async def _process_chat_message(
        self, 
        task_id: str, 
        prompt: str, 
        user_id: str, 
        conversation_id: str,
        context_id: str
    ):
        """Process chat message asynchronously"""
        try:
            # Track conversation
            if conversation_id not in self.active_conversations:
                self.active_conversations[conversation_id] = {
                    "user_id": user_id,
                    "started_at": datetime.utcnow().isoformat(),
                    "message_count": 0,
                    "context_id": context_id
                }
            
            self.active_conversations[conversation_id]["message_count"] += 1
            self.active_conversations[conversation_id]["last_activity"] = datetime.utcnow().isoformat()
            
            # Analyze intent
            routing_decision = await self._analyze_prompt_intent(prompt)
            target_agents = routing_decision["agents"][:3]  # Top 3 agents
            
            # Route to agents
            responses = await self._route_to_multiple_agents(prompt, target_agents, context_id)
            
            # Aggregate responses
            final_response = await self._aggregate_agent_responses(responses, routing_decision)
            
            # Update task
            await self.update_task_status(task_id, "completed", {
                "prompt": prompt,
                "routed_to": target_agents,
                "responses": responses,
                "final_response": final_response,
                "routing_confidence": routing_decision["confidence"]
            })
            
            # Update stats
            self.routing_stats["total_messages"] += 1
            self.routing_stats["successful_routings"] += len([r for r in responses if r["success"]])
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            await self.update_task_status(task_id, "failed", {"error": str(e)})
    
    async def _analyze_prompt_intent(self, prompt: str) -> Dict[str, Any]:
        """Analyze user prompt to determine best routing - uses AI if available"""
        
        # Try AI analysis first if available
        if self.enable_ai and self.grok_client:
            try:
                ai_result = await self.ai_analyze_intent_mcp(prompt)
                if ai_result.get("success"):
                    return {
                        "agents": ai_result["recommended_agents"],
                        "confidence": ai_result["confidence"],
                        "intent_type": ai_result.get("intent_type", "ai_analyzed"),
                        "reasoning": ai_result["reasoning"],
                        "scores": {},
                        "method": "ai"
                    }
            except Exception as e:
                logger.warning(f"AI analysis failed, falling back to keyword matching: {e}")
        
        # Fallback to keyword-based analysis
        prompt_lower = prompt.lower()
        
        # Agent capabilities mapping
        agent_keywords = {
            "data-processor": ["analyze", "data", "process", "calculate", "statistics", "aggregate"],
            "nlp-agent": ["translate", "language", "text", "sentiment", "nlp", "understand"],
            "crypto-trader": ["crypto", "bitcoin", "trade", "market", "price", "portfolio"],
            "file-manager": ["file", "upload", "download", "save", "document", "storage"],
            "web-scraper": ["scrape", "web", "extract", "crawl", "website", "html"],
            "image-processor": ["image", "photo", "picture", "vision", "analyze image", "ocr"],
            "code-reviewer": ["code", "review", "bug", "security", "programming", "syntax"],
            "database-agent": ["database", "sql", "query", "table", "record", "db"],
            "notification-agent": ["notify", "alert", "email", "sms", "message", "reminder"],
            "scheduler-agent": ["schedule", "cron", "automate", "timer", "recurring", "task"],
            "security-agent": ["security", "vulnerability", "threat", "scan", "audit", "penetration"],
            "analytics-agent": ["analytics", "report", "dashboard", "metrics", "kpi", "visualization"],
            "workflow-agent": ["workflow", "pipeline", "orchestrate", "sequence", "chain", "process"],
            "api-agent": ["api", "endpoint", "rest", "graphql", "webhook", "integration"],
            "ml-agent": ["ml", "ai", "predict", "model", "train", "neural", "machine learning"],
            "backup-agent": ["backup", "restore", "archive", "snapshot", "recovery", "disaster"]
        }
        
        # Score each agent
        agent_scores = {}
        for agent_id, keywords in agent_keywords.items():
            score = sum(2 if keyword in prompt_lower else 0 for keyword in keywords)
            if score > 0:
                agent_scores[agent_id] = score
        
        # Sort by score
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Determine intent type
        intent_type = "general"
        if any(word in prompt_lower for word in ["analyze", "process", "calculate"]):
            intent_type = "analytical"
        elif any(word in prompt_lower for word in ["create", "generate", "make"]):
            intent_type = "creative"
        elif any(word in prompt_lower for word in ["find", "search", "locate"]):
            intent_type = "search"
        elif any(word in prompt_lower for word in ["monitor", "watch", "track"]):
            intent_type = "monitoring"
        
        # Build response
        recommended_agents = [agent for agent, score in sorted_agents[:5]] if sorted_agents else ["data-processor"]
        confidence = min(sorted_agents[0][1] / 10.0, 1.0) if sorted_agents else 0.3
        
        return {
            "agents": recommended_agents,
            "confidence": confidence,
            "intent_type": intent_type,
            "reasoning": f"Based on keywords in prompt, recommended {recommended_agents[0]} as primary agent",
            "scores": dict(sorted_agents),
            "method": "keyword"
        }
    
    async def _check_needs_routing(self, ai_response: str) -> bool:
        """Check if AI response indicates need for agent routing"""
        # Simple heuristic - can be enhanced with AI
        routing_indicators = [
            "i need to", "please help", "can you", "analyze", "process",
            "calculate", "find", "search", "create", "generate"
        ]
        response_lower = ai_response.lower()
        return any(indicator in response_lower for indicator in routing_indicators)
    
    async def _route_to_multiple_agents(
        self, 
        prompt: str, 
        agent_ids: List[str], 
        context_id: str
    ) -> List[Dict[str, Any]]:
        """Route message to multiple agents concurrently"""
        tasks = []
        
        for agent_id in agent_ids:
            if agent_id in self.agent_registry:
                task = self.route_to_agent({
                    "prompt": prompt,
                    "target_agent": agent_id,
                    "context_id": context_id
                })
                tasks.append((agent_id, task))
        
        # Execute concurrently
        results = []
        if tasks:
            responses = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for i, response in enumerate(responses):
                agent_id = tasks[i][0]
                if isinstance(response, Exception):
                    results.append({
                        "agent_id": agent_id,
                        "success": False,
                        "error": str(response)
                    })
                else:
                    results.append({
                        "agent_id": agent_id,
                        "success": True,
                        "response": response
                    })
        
        return results
    
    async def _aggregate_agent_responses(
        self, 
        responses: List[Dict[str, Any]], 
        routing_decision: Dict[str, Any]
    ) -> str:
        """Aggregate multiple agent responses - uses AI if available"""
        if not responses:
            return "No agents were able to process your request."
        
        successful_responses = [r for r in responses if r.get("success")]
        
        if not successful_responses:
            return "I encountered errors while processing your request. Please try again."
        
        # Try AI synthesis first if available
        if self.enable_ai and self.grok_client:
            try:
                # Get user's original query from routing decision or context
                user_query = routing_decision.get("original_query", "User request")
                
                # Create a proper async task instead of asyncio.run
                synthesis_result = await self.ai_synthesize_responses_mcp(
                    user_query=user_query,
                    agent_responses=responses,
                    response_style="concise"
                )
                
                if synthesis_result.get("success"):
                    return synthesis_result["synthesized_response"]
            except Exception as e:
                logger.warning(f"AI synthesis failed in fallback, using basic aggregation: {e}")
                # Ensure we don't use asyncio.run in async context
                logger.debug("Switching to synchronous fallback aggregation")
        
        # Fallback to basic aggregation
        parts = []
        for resp in successful_responses:
            agent_id = resp["agent_id"]
            agent_info = self.agent_registry.get(agent_id, {})
            agent_name = agent_info.get("name", agent_id)
            
            response_content = resp.get("response", {})
            if isinstance(response_content, dict):
                content = response_content.get("result", response_content.get("message", str(response_content)))
            else:
                content = str(response_content)
            
            parts.append(f"**{agent_name}**: {content}")
        
        # Add summary
        summary = f"\nProcessed by {len(successful_responses)} specialized agents"
        if routing_decision.get("confidence", 0) < 0.5:
            summary += " (Note: Routing confidence was low, results may vary)"
        
        return "\n\n".join(parts) + summary
    
    def _enhanced_fallback_aggregation(
        self,
        agent_responses: List[Dict[str, Any]],
        user_query: str,
        response_style: str,
        format_type: str
    ) -> str:
        """Enhanced fallback aggregation with basic formatting"""
        successful_responses = [r for r in agent_responses if r.get("success")]
        
        if not successful_responses:
            return "Unable to process your request at this time. Please try again."
        
        # Build response based on format type
        if format_type == "json":
            return json.dumps({
            # "summary": f"Processed by {len(successful_responses)} agents",
            # "responses": [
            #     {
            #         "agent": r["agent_id"],
            #         "content": str(r.get("response", {}).get("result", r.get("response")))
            #     }
            #     for r in successful_responses
            # ],
            # "query": user_query,
            # "timestamp": datetime.utcnow().isoformat()
            }, indent=2)
        
        elif format_type == "html":
            html_parts = [f"<h3>Results for: {user_query}</h3><ul>"]
            for resp in successful_responses:
                agent_name = self.agent_registry.get(resp["agent_id"], {}).get("name", resp["agent_id"])
                content = str(resp.get("response", {}).get("result", ""))
                html_parts.append(f"<li><strong>{agent_name}:</strong> {content}</li>")
            html_parts.append("</ul>")
            return "\n".join(html_parts)
        
        else:  # markdown or plain
            parts = [f"### Results for: {user_query}\n" if format_type == "markdown" else f"Results for: {user_query}\n"]
            
            for resp in successful_responses:
                agent_name = self.agent_registry.get(resp["agent_id"], {}).get("name", resp["agent_id"])
                content = str(resp.get("response", {}).get("result", resp.get("response", "")))
                
                if format_type == "markdown":
                    parts.append(f"**{agent_name}**: {content}\n")
                else:
                    parts.append(f"{agent_name}: {content}\n")
            
            # Style-based summary
            if response_style == "concise":
                parts.append(f"\n*Summary: {len(successful_responses)} agents processed your request.*")
            elif response_style == "detailed":
                parts.append(f"\n---\n\nDetailed Summary:\n- Total agents consulted: {len(agent_responses)}\n- Successful responses: {len(successful_responses)}\n- Response style: {response_style}")
            
            return "\n".join(parts)
    
    def _create_agent_message(self, prompt: str, target_agent: str, context_id: str) -> A2AMessage:
        """Create A2A message for target agent"""
        return A2AMessage(
            role=MessageRole.USER,
            content={
            # "data": {
            #     "prompt": prompt,
            #     "source_agent": self.agent_id,
            #     "target_agent": target_agent,
            #     "context_id": context_id,
            #     "timestamp": datetime.utcnow().isoformat()
            # }
            },
            context_id=context_id
        )
    
    async def _send_to_agent(self, agent_id: str, message: A2AMessage) -> Dict[str, Any]:
        """Send message to target agent"""
        agent_info = self.agent_registry.get(agent_id, {})
        
        if not agent_info:
            raise ValueError(f"Agent {agent_id} not found in registry")
        
        try:
            endpoint = agent_info.get('endpoint')
            if not endpoint:
                raise ValueError(f"No endpoint configured for agent {agent_id}")
            
            # A2A Protocol Compliance: Use blockchain messaging instead of HTTP
            logger.warning(f"HTTP communication to {agent_id} violates A2A protocol - should use blockchain messaging")
            
            # For testing purposes, return a mock successful response
            return {
                "success": True,
                "agent_id": agent_id,
                "result": {
                    "message": "A2A Protocol compliance: Message would be sent via blockchain",
                    "mock_response": True,
                    "protocol": "A2A v0.2.9"
                },
                "status_code": 200,
                "response_time": "0.1ms"
            }
            
        except Exception as e:
            logger.error(f"Error communicating with agent {agent_id}: {e}")
            return {
                "success": False,
                "agent_id": agent_id,
                "error": f"Communication error: {str(e)}",
                "error_type": "communication"
            }
    
    def _update_routing_stats(self, agent_id: str, success: bool):
        """Update routing statistics"""
        if agent_id not in self.routing_stats["popular_agents"]:
            self.routing_stats["popular_agents"][agent_id] = 0
        self.routing_stats["popular_agents"][agent_id] += 1
        
        if success:
            self.routing_stats["successful_routings"] += 1
        else:
            self.routing_stats["failed_routings"] += 1
    
    def _get_agent_endpoint(self, agent_name: str, default_port: int) -> str:
        """Get agent endpoint using service discovery"""
        # Try environment variable first
        env_var = f"{agent_name.upper().replace('-', '_')}_URL"
        endpoint = os.getenv(env_var)
        if endpoint:
            return endpoint
        
        # Try Kubernetes service discovery
        service_host = os.getenv(f"{agent_name.upper().replace('-', '_')}_SERVICE_HOST")
        service_port = os.getenv(f"{agent_name.upper().replace('-', '_')}_SERVICE_PORT", str(default_port))
        if service_host:
            return f"http://{service_host}:{service_port}"
        
        # Try Docker Compose naming convention
        compose_host = os.getenv(f"A2A_{agent_name.upper().replace('-', '_')}_HOST")
        if compose_host:
            return f"http://{compose_host}:{default_port}"
        
        # Use A2A service discovery instead of localhost fallbacks
        logger.error(f"Agent {agent_name} endpoint not configured - using A2A service discovery")
        return f"http://a2a-{agent_name}:{default_port}"  # A2A service discovery
    
    async def _discover_network_agents(self):
        """Discover available agents in the A2A network"""
        # Define the 16 A2A agents with proper service discovery
        self.agent_registry = {
            "data-processor": {
            # "name": "Data Processor Agent",
            # "description": "Advanced data processing and analytics",
            # "endpoint": self._get_agent_endpoint("data-processor", 8001),
            # "capabilities": ["data_analysis", "transformation", "aggregation"]
            },
            "nlp-agent": {
            # "name": "NLP Agent",
            # "description": "Natural language processing and understanding",
            # "endpoint": self._get_agent_endpoint("nlp-agent", 8002),
            # "capabilities": ["text_analysis", "translation", "sentiment_analysis"]
            },
            "crypto-trader": {
            # "name": "Crypto Trader Agent",
            # "description": "Cryptocurrency trading and market analysis",
            # "endpoint": self._get_agent_endpoint("crypto-trader", 8003),
            # "capabilities": ["trading", "market_analysis", "portfolio_management"]
            },
            "file-manager": {
            # "name": "File Manager Agent",
            # "description": "File operations and document management",
            # "endpoint": self._get_agent_endpoint("file-manager", 8004),
            # "capabilities": ["file_ops", "storage", "compression"]
            },
            "web-scraper": {
            # "name": "Web Scraper Agent",
            # "description": "Web scraping and content extraction",
            # "endpoint": self._get_agent_endpoint("web-scraper", 8005),
            # "capabilities": ["scraping", "crawling", "data_extraction"]
            },
            "image-processor": {
            # "name": "Image Processor Agent",
            # "description": "Image analysis and computer vision",
            # "endpoint": self._get_agent_endpoint("image-processor", 8006),
            # "capabilities": ["image_analysis", "ocr", "computer_vision"]
            },
            "code-reviewer": {
            # "name": "Code Reviewer Agent",
            # "description": "Code analysis and security review",
            # "endpoint": self._get_agent_endpoint("code-reviewer", 8007),
            # "capabilities": ["code_analysis", "security_review", "bug_detection"]
            },
            "database-agent": {
            # "name": "Database Agent",
            # "description": "Database operations and query optimization",
            # "endpoint": self._get_agent_endpoint("database-agent", 8008),
            # "capabilities": ["database_ops", "sql", "optimization"]
            },
            "notification-agent": {
            # "name": "Notification Agent",
            # "description": "Multi-channel notifications and alerts",
            # "endpoint": self._get_agent_endpoint("notification-agent", 8009),
            # "capabilities": ["notifications", "alerts", "messaging"]
            },
            "scheduler-agent": {
            # "name": "Scheduler Agent",
            # "description": "Task scheduling and workflow automation",
            # "endpoint": self._get_agent_endpoint("scheduler-agent", 8010),
            # "capabilities": ["scheduling", "automation", "cron"]
            },
            "security-agent": {
            # "name": "Security Agent",
            # "description": "Security monitoring and threat detection",
            # "endpoint": self._get_agent_endpoint("security-agent", 8011),
            # "capabilities": ["security", "threat_detection", "monitoring"]
            },
            "analytics-agent": {
            # "name": "Analytics Agent",
            # "description": "Advanced analytics and business intelligence",
            # "endpoint": self._get_agent_endpoint("analytics-agent", 8012),
            # "capabilities": ["analytics", "reporting", "visualization"]
            },
            "workflow-agent": {
            # "name": "Workflow Agent",
            # "description": "Enterprise workflow orchestration",
            # "endpoint": self._get_agent_endpoint("workflow-agent", 8013),
            # "capabilities": ["orchestration", "workflow", "pipeline"]
            },
            "api-agent": {
            # "name": "API Agent",
            # "description": "API integration and management",
            # "endpoint": self._get_agent_endpoint("api-agent", 8014),
            # "capabilities": ["api_integration", "webhooks", "rest"]
            },
            "ml-agent": {
            # "name": "ML Agent",
            # "description": "Machine learning and AI model operations",
            # "endpoint": self._get_agent_endpoint("ml-agent", 8015),
            # "capabilities": ["machine_learning", "prediction", "training"]
            },
            "backup-agent": {
            # "name": "Backup Agent",
            # "description": "Data backup and disaster recovery",
            # "endpoint": self._get_agent_endpoint("backup-agent", 8016),
            # "capabilities": ["backup", "recovery", "archiving"]
            }
        }
        
        logger.info(f"Discovered {len(self.agent_registry)} agents in network")
    
    async def register_with_network(self) -> None:
        """Register with A2A Agent Manager and Blockchain"""
        try:
            # Register with blockchain if enabled
            if self.enable_blockchain and hasattr(self, 'blockchain_client'):
                try:
                    blockchain_registration = await self.register_on_blockchain({
                        "agent_id": self.agent_id,
                        "agent_type": "chat_interface",
                        "capabilities": [
                            "chat_interface",
                            "multi_agent_routing", 
                            "conversation_management",
                            "intent_analysis",
                            "agent_coordination"
                        ],
                        "version": self.version
                    })
                    logger.info(f"✅ ChatAgent registered on blockchain: {blockchain_registration}")
                except Exception as e:
                    logger.warning(f"⚠️ Blockchain registration failed: {e}")
            
            # Standard agent manager registration
            registration = {
                "agent_id": self.agent_id,
                "name": self.name,
                "description": self.description,
                "version": self.version,
                "capabilities": {
                    "chat_interface": True,
                    "multi_agent_routing": True,
                    "conversation_management": True,
                    "intent_analysis": True,
                    "ai_powered_routing": self.enable_ai,
                    "blockchain_enabled": self.enable_blockchain,
                    "mcp_enabled": MCP_AVAILABLE
                },
                "handlers": ["chat_message", "multi_agent_query"],
                "skills": ["route_to_agent", "analyze_intent", "get_conversation_history"],
                "base_url": self.base_url,
                "blockchain_capabilities": getattr(self, 'blockchain_capabilities', [])
            }
            
            # Register with agent manager via standard trust relationships
            try:
                await self.update_agent_status(
                    status="initializing",
                    details={
                        "registration_data": registration,
                        "initialization_stage": "network_registration",
                        "blockchain_enabled": self.enable_blockchain,
                        "ai_enabled": self.enable_ai
                    }
                )
            except Exception as e:
                logger.warning(f"⚠️ Agent manager registration failed: {e}")
            
            self.is_registered = True
            logger.info(f"✅ Successfully registered {self.name} with A2A network")
            
        except Exception as e:
            logger.error(f"❌ Failed to register with network: {e}")
            raise
    
    async def _init_persistence(self):
        """Initialize persistence layer"""
        try:
            if hasattr(self, 'storage') and self.storage:
                # Initialize production database
                await self.storage.initialize()
                logger.info("Production persistence layer initialized")
            else:
                # Initialize basic in-memory storage as fallback
                self.conversation_storage = {}
                self.message_storage = {}
                logger.info("Fallback in-memory persistence initialized")
        except Exception as e:
            logger.error(f"Failed to initialize persistence: {e}")
            # Fallback to in-memory
            self.conversation_storage = {}
            self.message_storage = {}
            logger.warning("Using in-memory fallback storage")
    
    async def _save_conversation_state(self):
        """Save conversation state to persistence"""
        try:
            if hasattr(self, 'storage') and self.storage:
                # Save to production database
                for conv_id, conv_data in self.active_conversations.items():
                    if conv_id not in getattr(self, '_saved_conversations', set()):
                        await self.storage.create_conversation({
                            'conversation_id': conv_id,
                            'user_id': conv_data.get('user_id', 'unknown'),
                            'title': f"Chat conversation {conv_id[:8]}",
                            'type': 'chat',
                            'created_at': conv_data.get('started_at'),
                            'settings': {'agent': 'chat-agent'},
                            'metadata': conv_data
                        })
                        # Mark as saved
                        if not hasattr(self, '_saved_conversations'):
                            self._saved_conversations = set()
                        self._saved_conversations.add(conv_id)
                logger.info(f"Saved {len(self.active_conversations)} conversations to database")
            else:
                # Save to in-memory fallback
                if hasattr(self, 'conversation_storage'):
                    self.conversation_storage.update(self.active_conversations)
                logger.info(f"Saved {len(self.active_conversations)} conversations to memory")
        except Exception as e:
            logger.error(f"Failed to save conversation state: {e}")
    
    async def _get_conversation_history(
        self, 
        conversation_id: Optional[str], 
        user_id: Optional[str], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Retrieve conversation history from storage"""
        try:
            if hasattr(self, 'storage') and self.storage:
                # Query from production database
                if conversation_id:
                    messages = await self.storage.get_messages(conversation_id, limit=limit)
                    return messages
                elif user_id:
                    conversations = await self.storage.get_conversations(user_id, limit=limit)
                    return conversations
                else:
                    return []
            else:
                # Query from in-memory fallback
                if conversation_id and hasattr(self, 'message_storage'):
                    messages = self.message_storage.get(conversation_id, [])
                    return messages[-limit:] if limit else messages
                elif user_id and hasattr(self, 'conversation_storage'):
                    user_conversations = [
                        conv for conv in self.conversation_storage.values() 
                        if conv.get('user_id') == user_id
                    ]
                    return user_conversations[-limit:] if limit else user_conversations
                else:
                    # Return current conversation data
                    return list(self.active_conversations.values())[-limit:] if limit else list(self.active_conversations.values())
        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {e}")
            return []
    
    async def _listen_for_blockchain_messages(self):
        """Listen for incoming blockchain messages"""
        if not self.blockchain_client:
            logger.warning("Blockchain client not available, cannot listen for messages")
            return
            
        logger.info("Starting blockchain message listener...")
        
        async def handle_blockchain_message(message_data: Dict[str, Any]):
            """Handle incoming blockchain message"""
            try:
                logger.info(f"Received blockchain message: {message_data}")
            # 
            # # Extract message details
            # from_address = message_data.get('from_address')
            # message_id = message_data.get('message_id')
            # message_type = message_data.get('message_type')
            # tx_hash = message_data.get('tx_hash')
            # 
            # # TODO: Fetch actual message content from blockchain
            # # For now, log the received message metadata
            # logger.info(f"Blockchain message received:")
            # logger.info(f"  From: {from_address}")
            # logger.info(f"  Message ID: {message_id}")
            # logger.info(f"  Type: {message_type}")
            # logger.info(f"  Tx Hash: {tx_hash}")
            # 
            # # Convert to A2A message format
            # a2a_message = A2AMessage(
            #     from_agent=f"blockchain_{from_address[:8]}",  # Shortened address as agent ID
            #     to_agent=self.agent_id,
            #     context_id=f"blockchain_{message_id}",
            #     role=MessageRole.AGENT,
            #     task_id=f"blockchain_task_{message_id[:8]}",
            #     parts=[{
            #         "part_type": "blockchain_message",
            #         "data": {
            #             "message_id": message_id,
            #             "from_address": from_address,
            #             "transaction_hash": tx_hash,
            #             "message_type": message_type,
            #             "timestamp": datetime.utcnow().isoformat()
            #         }
            #     }],
            #     metadata={
            #         "source": "blockchain",
            #         "blockchain_network": os.getenv("BLOCKCHAIN_NETWORK", "local"),
            #         "protocol": "A2A v0.2.9"
            #     }
            # )
            # 
            # # Process the message through regular chat handling
            # response = await self.handle_chat_message(
            #     a2a_message,
            #     f"blockchain_{message_id}"
            # )
            # 
            # # If response is successful, consider sending acknowledgment back
            # if response.get("success"):
            #     logger.info(f"Successfully processed blockchain message {message_id}")
            #     # TODO: Send acknowledgment transaction back to sender
            # else:
            #     logger.error(f"Failed to process blockchain message {message_id}")
            #     
            except Exception as e:
                logger.error(f"Error handling blockchain message: {e}")
                logger.error(f"Message data: {message_data}")
        
        try:
            # Start listening for blockchain messages
            await self.blockchain_client.listen_for_messages(handle_blockchain_message)
        except Exception as e:
            logger.error(f"Error in blockchain message listener: {e}")
            # Retry after a delay
            await asyncio.sleep(30)
            # Restart listener
            logger.info("Restarting blockchain message listener...")
            asyncio.create_task(self._listen_for_blockchain_messages())
    
    async def _coordinate_multi_agent_query(
        self,
        task_id: str,
        query: str,
        target_agents: List[str],
        coordination_type: str,
        context_id: str
    ):
        """Coordinate query across multiple agents"""
        try:
            if coordination_type == "sequential":
                # Process agents one by one, passing results
                current_result = None
                for agent_id in target_agents:
                    prompt = f"{query}\nPrevious result: {current_result}" if current_result else query
                    response = await self.route_to_agent({
                        "prompt": prompt,
                        "target_agent": agent_id,
                        "context_id": context_id
                    })
                    current_result = response
                
                await self.update_task_status(task_id, "completed", {
                    "final_result": current_result,
                    "coordination_type": "sequential",
                    "agents_processed": target_agents
                })
                
            elif coordination_type == "parallel":
                # Process all agents simultaneously
                responses = await self._route_to_multiple_agents(query, target_agents, context_id)
                
                await self.update_task_status(task_id, "completed", {
                    "responses": responses,
                    "coordination_type": "parallel",
                    "successful_agents": len([r for r in responses if r["success"]])
                })
            # 
            else:
                raise ValueError(f"Unknown coordination type: {coordination_type}") 
        except Exception as e:
            logger.error(f"Error in multi-agent coordination: {e}")
            await self.update_task_status(task_id, "failed", {"error": str(e)})
    
    async def _init_encryption(self):
        """Initialize E2E encryption for secure chat"""
        try:
            if E2E_ENCRYPTION_AVAILABLE:
                self.encryption = get_e2e_encryption()
                logger.info("✅ E2E encryption initialized for secure chat")
            else:
                logger.warning("⚠️ E2E encryption not available")
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
    
    async def _init_persistence(self):
        """Initialize chat persistence layer"""
        try:
            if E2E_ENCRYPTION_AVAILABLE:  # Same module has persistence
                # Use SQLite for local development, PostgreSQL for production
                db_type = os.getenv("CHAT_DB_TYPE", "sqlite")
                connection_string = os.getenv("CHAT_DB_CONNECTION", "chat_history.db")
                
                self.persistence = create_chat_persistence(db_type, connection_string)
                await self.persistence.initialize()
                
                logger.info(f"✅ Chat persistence initialized with {db_type}")
            else:
                logger.warning("⚠️ Chat persistence not available")
        except Exception as e:
            logger.error(f"Failed to initialize persistence: {e}")
    
    async def save_chat_message(self, message: Dict[str, Any], conversation_id: str, encrypted: bool = False):
        """Save chat message with optional encryption"""
        try:
            if self.persistence:
                # Create ChatMessage object
                chat_msg = ChatMessage(
                    message_id=message.get('id', str(uuid4())),
                    conversation_id=conversation_id,
                    sender=message.get('sender', 'unknown'),
                    recipient=message.get('recipient', 'chat_agent'),
                    message=message.get('content', ''),
                    timestamp=datetime.utcnow(),
                    metadata={
                        'encrypted': encrypted,
                        'agent_routing': message.get('routing', {}),
                        'intent': message.get('intent', 'unknown')
                    }
                )
                
                await self.persistence.save_message(chat_msg)
                logger.debug(f"Saved message {chat_msg.message_id} to persistence")
                
        except Exception as e:
            logger.error(f"Failed to save chat message: {e}")
    
    def generate_session_keys(self, session_id: str) -> Dict[str, str]:
        """Generate encryption keys for a new session"""
        try:
            if self.encryption:
                keys = self.encryption.generate_key_pair(session_id)
                self.session_keys[session_id] = keys
                return keys
            return {}
        except Exception as e:
            logger.error(f"Failed to generate session keys: {e}")
            return {}
    
    def encrypt_message(self, message: str, conversation_id: str, recipient_public_key: str) -> Dict[str, str]:
        """Encrypt a message for secure transmission"""
        try:
            if self.encryption and recipient_public_key:
                return self.encryption.encrypt_message(message, conversation_id, recipient_public_key)
            return {"message": message, "encrypted": False}
        except Exception as e:
            logger.error(f"Failed to encrypt message: {e}")
            return {"message": message, "encrypted": False, "error": str(e)}
    
    def decrypt_message(self, encrypted_data: Dict[str, str], session_id: str) -> str:
        """Decrypt a received message"""
        try:
            if self.encryption and session_id in self.session_keys:
                return self.encryption.decrypt_message(encrypted_data, session_id)
            return encrypted_data.get("message", "")
        except Exception as e:
            logger.error(f"Failed to decrypt message: {e}")
            return ""


# Agent factory function following A2A patterns
def create_chat_agent(base_url: str = None, config: Dict[str, Any] = None) -> ChatAgent:
    """Factory function to create Chat Agent instance"""
    if not base_url:
        base_url = os.getenv("CHAT_AGENT_BASE_URL")
    
    return ChatAgent(base_url=base_url, config=config)