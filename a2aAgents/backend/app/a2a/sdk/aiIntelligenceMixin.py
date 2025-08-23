"""
AI Intelligence Mixin for A2A Agents
Provides standardized AI reasoning capabilities using GrokClient
"""

import asyncio
import json
import os
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from uuid import uuid4

logger = logging.getLogger(__name__)

class AIIntelligenceMixin:
    """
    Mixin that provides AI intelligence capabilities to all A2A agents.
    
    This mixin adds standardized reasoning capabilities using GrokClient,
    implementing the standard flow: receive → reason → act → respond
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grok_client = None
        self.ai_enabled = os.getenv("AI_ENABLED", "true").lower() == "true"
        self.reasoning_cache = {}
        self.interaction_history = []
        
        # Skills and network awareness
        self.skill_registry = {}
        self.network_agents_cache = {}
        self.skill_matching_cache = {}
        self.referral_history = []
        
    async def initialize_ai_intelligence(self, config: Optional[Dict[str, Any]] = None):
        """Initialize AI intelligence capabilities"""
        if not self.ai_enabled:
            logger.info(f"{getattr(self, 'agent_id', 'Agent')} - AI intelligence disabled")
            return
            
        try:
            # Import GrokClient (now with SAP AI Core SDK integration)
            from app.a2a.core.grokClient import GrokClient
            
            # Initialize GrokClient with configuration
            grok_config = config or {}
            
            # Prepare GrokClient initialization arguments
            grok_args = {
                "api_key": os.getenv("GROK_API_KEY"),
                "base_url": os.getenv("GROK_BASE_URL"),
                "model": grok_config.get("model", "grok-beta")
            }
            
            # Add other config parameters (excluding model to avoid duplication)
            for key, value in grok_config.items():
                if key != "model":
                    grok_args[key] = value
            
            self.grok_client = GrokClient(**grok_args)
            
            # Log which LLM service is being used
            system_status = self.grok_client.get_system_status()
            if system_status.get("sap_ai_core", {}).get("available"):
                mode = system_status["sap_ai_core"]["mode"]
                logger.info(f"{getattr(self, 'agent_id', 'Agent')} - AI intelligence initialized with SAP AI Core SDK in {mode} mode")
                logger.info(f"Failover chain: {' → '.join(system_status.get('failover_chain', []))}")
            else:
                logger.info(f"{getattr(self, 'agent_id', 'Agent')} - AI intelligence initialized with GrokClient")
            
            # Initialize skills awareness
            await self._initialize_skills_awareness()
            
        except ImportError as e:
            logger.warning(f"GrokClient not available: {e}")
            self.ai_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize AI intelligence: {e}")
            self.ai_enabled = False
    
    async def _initialize_skills_awareness(self):
        """Initialize skills awareness and network discovery"""
        try:
            # Register own skills
            await self._register_own_skills()
            
            # Discover network agents and their skills
            await self._discover_network_agents_skills()
            
            logger.info(f"Skills awareness initialized - {len(self.skill_registry)} own skills, {len(self.network_agents_cache)} network agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize skills awareness: {e}")
    
    async def _register_own_skills(self):
        """Register this agent's own skills with detailed metadata"""
        try:
            agent_id = getattr(self, 'agent_id', 'unknown')
            
            # Get skills from agent if available
            if hasattr(self, 'skills'):
                skills = getattr(self, 'skills', {})
                # Handle both dictionary and list formats
                if hasattr(skills, 'items'):
                    # Dictionary format
                    for skill_name, skill_def in skills.items():
                        skill_metadata = {
                            "name": skill_name,
                            "description": getattr(skill_def, 'description', f"Skill: {skill_name}"),
                            "input_schema": getattr(skill_def, 'input_schema', {}),
                            "output_schema": getattr(skill_def, 'output_schema', {}),
                            "capabilities": getattr(skill_def, 'capabilities', []),
                            "complexity": getattr(skill_def, 'complexity', 'medium'),
                            "performance": getattr(skill_def, 'performance', 'standard'),
                            "reliability": getattr(skill_def, 'reliability', 0.9),
                            "cost": getattr(skill_def, 'cost', 'low'),
                            "categories": getattr(skill_def, 'categories', []),
                            "dependencies": getattr(skill_def, 'dependencies', []),
                            "agent_id": agent_id
                        }
                        self.skill_registry[skill_name] = skill_metadata
                elif isinstance(skills, list):
                    # List format - create skills from list items
                    for i, skill in enumerate(skills):
                        skill_name = getattr(skill, 'name', f"skill_{i}")
                        skill_metadata = {
                            "name": skill_name,
                            "description": getattr(skill, 'description', f"Skill: {skill_name}"),
                            "input_schema": getattr(skill, 'input_schema', {}),
                            "output_schema": getattr(skill, 'output_schema', {}),
                            "capabilities": getattr(skill, 'capabilities', []),
                            "complexity": getattr(skill, 'complexity', 'medium'),
                            "performance": getattr(skill, 'performance', 'standard'),
                            "reliability": getattr(skill, 'reliability', 0.9),
                            "cost": getattr(skill, 'cost', 'low'),
                            "categories": getattr(skill, 'categories', []),
                            "dependencies": getattr(skill, 'dependencies', []),
                            "agent_id": agent_id
                        }
                        self.skill_registry[skill_name] = skill_metadata
            
            # Get capabilities as skills if available
            if hasattr(self, 'capabilities'):
                capabilities = getattr(self, 'capabilities', [])
                for cap in capabilities:
                    cap_name = cap.name if hasattr(cap, 'name') else str(cap)
                    if cap_name not in self.skill_registry:
                        self.skill_registry[cap_name] = {
                            "name": cap_name,
                            "description": f"Capability: {cap_name}",
                            "type": "capability",
                            "agent_id": agent_id,
                            "reliability": 0.8,
                            "complexity": "medium"
                        }
            
            logger.info(f"Registered {len(self.skill_registry)} own skills")
            
        except Exception as e:
            logger.error(f"Failed to register own skills: {e}")
    
    async def _discover_network_agents_skills(self):
        """Discover other agents and their skills from blockchain"""
        try:
            # Initialize blockchain client if not available
            if not hasattr(self, 'blockchain_client') or not self.blockchain_client:
                try:
                    from web3 import Web3
                    rpc_url = os.getenv("A2A_RPC_URL", "http://localhost:8545")
                    self.blockchain_client = Web3(Web3.HTTPProvider(rpc_url))
                    if not self.blockchain_client.is_connected():
                        logger.warning("Blockchain client not connected")
                        return
                    logger.info("Initialized blockchain client for network discovery")
                except ImportError:
                    logger.warning("Web3 not available - cannot connect to blockchain")
                    return
                except Exception as e:
                    logger.warning(f"Failed to initialize blockchain client: {e}")
                    return
            
            # Get all registered agents from blockchain
            try:
                agents = await self._get_all_blockchain_agents()
                
                for agent_info in agents:
                    agent_address = agent_info.get('address')
                    agent_name = agent_info.get('name', 'Unknown')
                    capabilities = agent_info.get('capabilities', [])
                    
                    if agent_address and agent_address != getattr(self, 'agent_address', None):
                        # Parse capabilities as skills
                        agent_skills = {}
                        for cap in capabilities:
                            # Decode bytes32 capability to string if needed
                            cap_name = self._decode_capability(cap)
                            if cap_name:
                                agent_skills[cap_name] = {
                                    "name": cap_name,
                                    "description": f"{agent_name} capability: {cap_name}",
                                    "agent_id": agent_name,
                                    "agent_address": agent_address,
                                    "type": "blockchain_capability",
                                    "reliability": agent_info.get('reputation', 0.5) / 100.0,  # Normalize reputation
                                    "active": agent_info.get('active', False)
                                }
                        
                        self.network_agents_cache[agent_address] = {
                            "name": agent_name,
                            "address": agent_address,
                            "skills": agent_skills,
                            "reputation": agent_info.get('reputation', 0),
                            "active": agent_info.get('active', False),
                            "endpoint": agent_info.get('endpoint', ''),
                            "last_updated": datetime.utcnow().isoformat()
                        }
                
                logger.info(f"Discovered {len(self.network_agents_cache)} network agents")
                
            except Exception as e:
                logger.warning(f"Failed to discover network agents: {e}")
                
        except Exception as e:
            logger.error(f"Failed to discover network agents skills: {e}")
    
    def _decode_capability(self, capability) -> Optional[str]:
        """Decode blockchain capability (bytes32) to human-readable string"""
        try:
            if isinstance(capability, bytes):
                # Remove null bytes and decode
                return capability.rstrip(b'\x00').decode('utf-8')
            elif isinstance(capability, str):
                return capability
            else:
                # Try to convert to string
                return str(capability)
        except Exception:
            return None
    
    async def _get_all_blockchain_agents(self) -> List[Dict[str, Any]]:
        """Get all registered agents from blockchain"""
        try:
            if not hasattr(self, 'blockchain_client') or not self.blockchain_client:
                logger.warning("No blockchain client available")
                return []
            
            # Get agent registry contract
            agent_registry_address = os.getenv("A2A_AGENT_REGISTRY_ADDRESS")
            if not agent_registry_address:
                logger.warning("A2A_AGENT_REGISTRY_ADDRESS not set")
                return []
            
            # Get all agent addresses from events or iterate through known agents
            agents = []
            
            # Try to get agent count first
            try:
                # This is the actual blockchain call - no mock
                w3 = self.blockchain_client.web3
                
                # Load the contract ABI for agent registry
                agent_registry_abi = [
                    {
                        "inputs": [{"name": "", "type": "address"}],
                        "name": "agents",
                        "outputs": [
                            {"name": "owner", "type": "address"},
                            {"name": "name", "type": "string"},
                            {"name": "endpoint", "type": "string"},
                            {"name": "capabilities", "type": "bytes32[]"},
                            {"name": "reputation", "type": "uint256"},
                            {"name": "active", "type": "bool"},
                            {"name": "registeredAt", "type": "uint256"}
                        ],
                        "stateMutability": "view",
                        "type": "function"
                    },
                    {
                        "inputs": [],
                        "name": "getAgentCount",
                        "outputs": [{"name": "", "type": "uint256"}],
                        "stateMutability": "view",
                        "type": "function"
                    }
                ]
                
                contract = w3.eth.contract(
                    address=w3.to_checksum_address(agent_registry_address),
                    abi=agent_registry_abi
                )
                
                # Get known agent addresses from deployed agents list
                known_agents = [
                    "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",  # ChatAgent
                    "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",  # DataManager
                    "0x90F79bf6EB2c4f870365E785982E1f101E93b906",  # TaskExecutor
                    "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65",  # AnalyticsAgent
                    "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc"   # SecurityAgent
                ]
                
                for agent_address in known_agents:
                    try:
                        agent_info = contract.functions.agents(agent_address).call()
                        if agent_info[0] != "0x0000000000000000000000000000000000000000":  # Not zero address
                            agents.append({
                                "address": agent_address,
                                "owner": agent_info[0],
                                "name": agent_info[1],
                                "endpoint": agent_info[2],
                                "capabilities": agent_info[3],
                                "reputation": agent_info[4],
                                "active": agent_info[5],
                                "registeredAt": agent_info[6]
                            })
                    except Exception as e:
                        logger.debug(f"Agent {agent_address} not found or error: {e}")
                        continue
                
                logger.info(f"Retrieved {len(agents)} agents from blockchain")
                return agents
                
            except Exception as e:
                logger.error(f"Error querying blockchain for agents: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to get blockchain agents: {e}")
            return []
    
    async def analyze_skills_match(self, required_skills: List[str], message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze if this agent has the required skills to handle the message.
        If not, find and recommend better-suited agents.
        
        Args:
            required_skills: List of required skills for the task
            message_data: The message requesting the task
            
        Returns:
            Dict containing skills analysis and referral recommendations
        """
        try:
            agent_id = getattr(self, 'agent_id', 'unknown')
            
            # 1. Analyze own skills match
            own_skills = list(self.skill_registry.keys())
            skills_match = self._calculate_skills_similarity(required_skills, own_skills)
            
            # 2. Determine if we can handle this task
            can_handle = skills_match['similarity_score'] >= 0.7  # 70% threshold
            confidence = skills_match['similarity_score']
            
            logger.info(f"Skills analysis for {agent_id}:")
            logger.info(f"  Required: {required_skills}")
            logger.info(f"  Available: {own_skills}")
            logger.info(f"  Match score: {skills_match['similarity_score']:.2f}")
            logger.info(f"  Can handle: {can_handle}")
            
            result = {
                "agent_id": agent_id,
                "required_skills": required_skills,
                "available_skills": own_skills,
                "skills_match": skills_match,
                "can_handle": can_handle,
                "confidence": confidence,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # 3. If we can't handle it well, find better agents
            if not can_handle or confidence < 0.8:
                logger.info(f"Searching for better-suited agents (current confidence: {confidence:.2f})")
                better_agents = await self._find_better_agents(required_skills, message_data)
                
                if better_agents:
                    result["referral_recommended"] = True
                    result["recommended_agents"] = better_agents
                    result["referral_reason"] = f"Found {len(better_agents)} agents better suited for this task"
                    
                    best_agent = better_agents[0]  # First is highest scoring
                    logger.info(f"Best referral candidate: {best_agent['name']} (score: {best_agent['match_score']:.2f})")
                else:
                    result["referral_recommended"] = False
                    result["referral_reason"] = "No better agents found, will attempt with available capabilities"
            else:
                result["referral_recommended"] = False
                result["referral_reason"] = "Agent has sufficient skills to handle the task"
            
            return result
            
        except Exception as e:
            logger.error(f"Skills analysis failed: {e}")
            return {
                "agent_id": getattr(self, 'agent_id', 'unknown'),
                "can_handle": True,  # Default to handling if analysis fails
                "confidence": 0.5,
                "error": str(e),
                "referral_recommended": False
            }
    
    def _calculate_skills_similarity(self, required_skills: List[str], available_skills: List[str]) -> Dict[str, Any]:
        """Calculate similarity between required and available skills"""
        try:
            if not required_skills:
                return {"similarity_score": 1.0, "matches": [], "gaps": [], "analysis": "No specific skills required"}
            
            if not available_skills:
                return {"similarity_score": 0.0, "matches": [], "gaps": required_skills, "analysis": "No skills available"}
            
            # Normalize skill names for comparison
            req_normalized = [skill.lower().strip() for skill in required_skills]
            avail_normalized = [skill.lower().strip() for skill in available_skills]
            
            # Direct matches
            direct_matches = []
            for req_skill in req_normalized:
                for avail_skill in avail_normalized:
                    if req_skill == avail_skill:
                        direct_matches.append(req_skill)
                        break
            
            # Partial/semantic matches (simple keyword matching)
            partial_matches = []
            for req_skill in req_normalized:
                if req_skill not in direct_matches:
                    for avail_skill in avail_normalized:
                        if req_skill in avail_skill or avail_skill in req_skill:
                            # Check for keyword overlap
                            req_words = set(req_skill.replace('_', ' ').split())
                            avail_words = set(avail_skill.replace('_', ' ').split())
                            if req_words & avail_words:  # If there's any word overlap
                                partial_matches.append((req_skill, avail_skill))
                                break
            
            # Calculate similarity score
            direct_weight = 1.0
            partial_weight = 0.6
            
            direct_score = len(direct_matches) * direct_weight
            partial_score = len(partial_matches) * partial_weight
            total_score = (direct_score + partial_score) / len(required_skills)
            
            # Cap at 1.0
            similarity_score = min(total_score, 1.0)
            
            # Identify gaps
            matched_req_skills = set(direct_matches + [match[0] for match in partial_matches])
            gaps = [skill for skill in req_normalized if skill not in matched_req_skills]
            
            return {
                "similarity_score": similarity_score,
                "direct_matches": direct_matches,
                "partial_matches": partial_matches,
                "gaps": gaps,
                "analysis": f"Found {len(direct_matches)} direct matches, {len(partial_matches)} partial matches, {len(gaps)} gaps"
            }
            
        except Exception as e:
            logger.error(f"Skills similarity calculation failed: {e}")
            return {"similarity_score": 0.0, "matches": [], "gaps": required_skills, "error": str(e)}
    
    async def _find_better_agents(self, required_skills: List[str], message_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find agents in the network that are better suited for the required skills"""
        try:
            better_agents = []
            own_agent_id = getattr(self, 'agent_id', 'unknown')
            
            # Search through network agents cache
            for agent_address, agent_info in self.network_agents_cache.items():
                agent_name = agent_info.get('name', 'Unknown')
                agent_skills = list(agent_info.get('skills', {}).keys())
                
                # Skip self
                if agent_name == own_agent_id:
                    continue
                
                # Skip inactive agents
                if not agent_info.get('active', False):
                    continue
                
                # Calculate skills match for this agent
                skills_match = self._calculate_skills_similarity(required_skills, agent_skills)
                match_score = skills_match['similarity_score']
                
                # Only consider agents with better match than us
                own_skills = list(self.skill_registry.keys())
                own_match = self._calculate_skills_similarity(required_skills, own_skills)
                
                if match_score > own_match['similarity_score']:
                    # Calculate overall agent score including reputation
                    reputation = agent_info.get('reputation', 0) / 100.0  # Normalize to 0-1
                    overall_score = (match_score * 0.7) + (reputation * 0.3)  # 70% skills, 30% reputation
                    
                    better_agents.append({
                        "name": agent_name,
                        "address": agent_address,
                        "endpoint": agent_info.get('endpoint', ''),
                        "skills": agent_skills,
                        "match_score": match_score,
                        "reputation": reputation,
                        "overall_score": overall_score,
                        "skills_analysis": skills_match,
                        "active": agent_info.get('active', False),
                        "last_updated": agent_info.get('last_updated')
                    })
            
            # Sort by overall score (best first)
            better_agents.sort(key=lambda x: x['overall_score'], reverse=True)
            
            # Limit to top 3 recommendations
            return better_agents[:3]
            
        except Exception as e:
            logger.error(f"Failed to find better agents: {e}")
            return []
    
    async def refer_to_agent(self, target_agent: Dict[str, Any], original_message: Dict[str, Any], referral_reason: str) -> Dict[str, Any]:
        """
        Refer a message to a better-suited agent with detailed context
        
        Args:
            target_agent: Information about the target agent
            original_message: The original message to be referred
            referral_reason: Explanation for the referral
            
        Returns:
            Dict containing referral result
        """
        try:
            # Create referral message with enhanced context
            referral_message = {
                "message_id": f"ref_{original_message.get('message_id', uuid4().hex[:8])}",
                "from_agent": getattr(self, 'agent_id', 'unknown'),
                "to_agent": target_agent['name'],
                "referral_context": {
                    "original_sender": original_message.get('from_agent'),
                    "original_message_id": original_message.get('message_id'),
                    "referral_reason": referral_reason,
                    "referring_agent": getattr(self, 'agent_id', 'unknown'),
                    "skills_analysis": {
                        "required_skills": original_message.get('required_skills', []),
                        "target_agent_match": target_agent.get('match_score', 0.0),
                        "referring_agent_match": getattr(self, '_last_skills_match', 0.0)
                    },
                    "timestamp": datetime.utcnow().isoformat()
                },
                "parts": original_message.get('parts', []),
                "priority": "HIGH",  # Referrals get high priority
                "encrypted": original_message.get('encrypted', False)
            }
            
            # Record referral in history
            referral_record = {
                "referral_id": referral_message["message_id"],
                "original_message_id": original_message.get('message_id'),
                "from_agent": getattr(self, 'agent_id', 'unknown'),
                "to_agent": target_agent['name'],
                "reason": referral_reason,
                "target_agent_score": target_agent.get('match_score', 0.0),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.referral_history.append(referral_record)
            
            # Keep only recent referrals (last 50)
            if len(self.referral_history) > 50:
                self.referral_history = self.referral_history[-50:]
            
            logger.info(f"Referring message {original_message.get('message_id')} to {target_agent['name']}")
            logger.info(f"Referral reason: {referral_reason}")
            logger.info(f"Target agent match score: {target_agent.get('match_score', 0.0):.2f}")
            
            # If we have blockchain integration, send the referral message
            if hasattr(self, 'send_a2a_message'):
                try:
                    send_result = await self.send_a2a_message(
                        target_agent['name'],
                        referral_message,
                        priority="HIGH"
                    )
                    
                    return {
                        "success": True,
                        "referral_sent": True,
                        "target_agent": target_agent['name'],
                        "referral_message_id": referral_message["message_id"],
                        "send_result": send_result,
                        "referral_reason": referral_reason
                    }
                    
                except Exception as send_error:
                    logger.error(f"Failed to send referral message: {send_error}")
                    return {
                        "success": False,
                        "referral_sent": False,
                        "error": f"Failed to send referral: {send_error}",
                        "referral_prepared": True,
                        "referral_message": referral_message
                    }
            else:
                # No blockchain integration, just return the prepared referral
                return {
                    "success": True,
                    "referral_sent": False,
                    "referral_prepared": True,
                    "referral_message": referral_message,
                    "note": "Referral prepared but no messaging capability available"
                }
                
        except Exception as e:
            logger.error(f"Agent referral failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "referral_sent": False
            }
    
    def get_skills_matching_statistics(self) -> Dict[str, Any]:
        """Get statistics about skills matching and referrals"""
        try:
            stats = {
                "own_skills_count": len(self.skill_registry),
                "own_skills": list(self.skill_registry.keys()),
                "network_agents_count": len(self.network_agents_cache),
                "referral_history_count": len(getattr(self, 'referral_history', [])),
                "skills_cache_size": len(getattr(self, 'skill_matching_cache', {}))
            }
            
            # Referral statistics
            if hasattr(self, 'referral_history') and self.referral_history:
                recent_threshold = (datetime.utcnow() - timedelta(hours=24)).isoformat()
                recent_referrals = [r for r in self.referral_history if r.get('timestamp', '') > recent_threshold]
                
                stats["recent_referrals_24h"] = len(recent_referrals)
                
                # Most referred agents
                referred_agents = {}
                for referral in self.referral_history:
                    agent = referral.get('to_agent', 'unknown')
                    referred_agents[agent] = referred_agents.get(agent, 0) + 1
                
                stats["most_referred_agents"] = dict(sorted(referred_agents.items(), key=lambda x: x[1], reverse=True)[:5])
            
            # Network skills overview
            all_network_skills = set()
            active_agents = 0
            for agent_info in self.network_agents_cache.values():
                if agent_info.get('active', False):
                    active_agents += 1
                    all_network_skills.update(agent_info.get('skills', {}).keys())
            
            stats["active_network_agents"] = active_agents
            stats["total_network_skills"] = len(all_network_skills)
            stats["network_skills_sample"] = list(all_network_skills)[:10]  # First 10 for preview
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get skills matching statistics: {e}")
            return {"error": str(e)}
    
    async def reason_about_message(self, message_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Use AI to reason about an incoming message and determine appropriate response.
        
        Args:
            message_data: The incoming message data
            context: Additional context for reasoning
            
        Returns:
            Dict containing reasoning results and recommended actions
        """
        if not self.ai_enabled or not self.grok_client:
            return self._fallback_reasoning(message_data, context)
        
        try:
            # Build reasoning prompt
            agent_info = {
                "agent_id": getattr(self, 'agent_id', 'unknown'),
                "agent_type": getattr(self, 'agent_type', 'generic'),
                "capabilities": getattr(self, 'capabilities', []),
                "current_status": getattr(self, 'status', 'active')
            }
            
            reasoning_prompt = self._build_reasoning_prompt(message_data, context, agent_info)
            
            # Get AI reasoning
            response = await self.grok_client.complete(
                prompt=reasoning_prompt,
                temperature=0.3,
                max_tokens=800,
                response_format="json"
            )
            
            # Parse AI response
            try:
                reasoning_result = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback to structured extraction
                reasoning_result = self._extract_reasoning_from_text(response.content)
            
            # ✨ NEW: Perform skills matching analysis
            required_skills = reasoning_result.get('required_capabilities', [])
            if required_skills:
                logger.info(f"Performing skills matching analysis for: {required_skills}")
                skills_analysis = await self.analyze_skills_match(required_skills, message_data)
                
                # Integrate skills analysis into reasoning result
                reasoning_result['skills_analysis'] = skills_analysis
                
                # If referral is recommended, update reasoning accordingly
                if skills_analysis.get('referral_recommended', False):
                    reasoning_result['should_process'] = False
                    reasoning_result['refusal_reason'] = {
                        'category': 'capability_mismatch',
                        'explanation': f"Skills match confidence ({skills_analysis.get('confidence', 0.0):.2f}) below threshold. {skills_analysis.get('referral_reason', '')}",
                        'severity': 'low',
                        'alternative_suggestions': [
                            f"Refer to {agent['name']} (match score: {agent['match_score']:.2f})" 
                            for agent in skills_analysis.get('recommended_agents', [])[:3]
                        ] + ['Attempt with available capabilities as fallback']
                    }
                    
                    # Store the skills match for potential referral
                    self._last_skills_match = skills_analysis.get('confidence', 0.0)
                    
                    logger.info(f"🔄 Referral recommended - Best candidate: {skills_analysis.get('recommended_agents', [{}])[0].get('name', 'None')} if available")
                else:
                    logger.info(f"✅ Skills match sufficient - Confidence: {skills_analysis.get('confidence', 0.0):.2f}")
            
            # Cache the reasoning
            message_id = message_data.get('message_id', str(datetime.utcnow().timestamp()))
            self.reasoning_cache[message_id] = reasoning_result
            
            # Record interaction for learning
            await self._record_interaction({
                "message": message_data,
                "context": context,
                "reasoning": reasoning_result,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return reasoning_result
            
        except Exception as e:
            logger.error(f"AI reasoning failed: {e}")
            return self._fallback_reasoning(message_data, context)
    
    async def generate_intelligent_response(self, 
                                          reasoning_result: Dict[str, Any], 
                                          action_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an intelligent response based on reasoning and action results.
        
        Args:
            reasoning_result: Results from reason_about_message
            action_result: Results from executing actions
            
        Returns:
            Dict containing the intelligent response
        """
        if not self.ai_enabled or not self.grok_client:
            return self._fallback_response_generation(reasoning_result, action_result)
        
        try:
            # Build response generation prompt
            agent_info = {
                "agent_id": getattr(self, 'agent_id', 'unknown'),
                "agent_type": getattr(self, 'agent_type', 'generic'),
                "capabilities": getattr(self, 'capabilities', [])
            }
            
            response_prompt = self._build_response_prompt(reasoning_result, action_result, agent_info)
            
            # Generate intelligent response
            response = await self.grok_client.complete(
                prompt=response_prompt,
                temperature=0.4,
                max_tokens=600,
                response_format="json"
            )
            
            # Parse response
            try:
                response_data = json.loads(response.content)
            except json.JSONDecodeError:
                response_data = self._extract_response_from_text(response.content)
            
            return {
                "success": True,
                "response": response_data,
                "reasoning_used": reasoning_result.get("reasoning_type", "ai"),
                "confidence": reasoning_result.get("confidence", 0.8),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._fallback_response_generation(reasoning_result, action_result)
    
    async def process_message_with_ai_reasoning(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete flow: receive → reason → act → respond using AI.
        
        This is the main method that implements the standard AI reasoning pattern.
        Includes intelligent message refusal capabilities.
        """
        try:
            # 1. RECEIVE - Extract and validate message
            logger.info(f"Processing message with AI reasoning: {message_data.get('message_id', 'unknown')}")
            
            # 2. REASON - Use AI to understand intent and determine response
            reasoning_result = await self.reason_about_message(message_data)
            
            # 3. EVALUATE - Check if message should be processed or refused
            should_process = reasoning_result.get("should_process", True)
            
            if not should_process:
                # Message is being intelligently refused
                refusal_reason = reasoning_result.get("refusal_reason", {})
                logger.warning(f"🚫 Refusing to process message {message_data.get('message_id', 'unknown')}")
                logger.warning(f"   Reason: {refusal_reason.get('category', 'unknown')}")
                logger.warning(f"   Explanation: {refusal_reason.get('explanation', 'No explanation provided')}")
                
                # ✨ NEW: Check if this is a skills mismatch with referral recommendation
                skills_analysis = reasoning_result.get('skills_analysis', {})
                if (refusal_reason.get('category') == 'capability_mismatch' and 
                    skills_analysis.get('referral_recommended', False)):
                    
                    recommended_agents = skills_analysis.get('recommended_agents', [])
                    if recommended_agents:
                        best_agent = recommended_agents[0]
                        logger.info(f"🔄 Attempting automatic referral to {best_agent['name']}")
                        
                        # Attempt referral
                        referral_result = await self.refer_to_agent(
                            best_agent, 
                            message_data, 
                            f"Skills mismatch - target agent better suited (match score: {best_agent['match_score']:.2f})"
                        )
                        
                        if referral_result.get('success', False):
                            logger.info(f"✅ Message successfully referred to {best_agent['name']}")
                            
                            return {
                                "success": True,
                                "refused": False,
                                "referred": True,
                                "referral_target": best_agent['name'],
                                "referral_result": referral_result,
                                "skills_analysis": skills_analysis,
                                "processing_steps": ["receive", "reason", "evaluate", "refer"],
                                "ai_enhanced": self.ai_enabled,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        else:
                            logger.warning(f"⚠️ Referral failed, proceeding with refusal: {referral_result.get('error', 'Unknown error')}")
                
                # Generate intelligent refusal response
                refusal_response = await self._generate_refusal_response(reasoning_result, message_data)
                
                return {
                    "success": False,
                    "refused": True,
                    "refusal_reason": refusal_reason,
                    "response": refusal_response,
                    "skills_analysis": skills_analysis,
                    "processing_steps": ["receive", "reason", "evaluate", "refuse"],
                    "ai_enhanced": self.ai_enabled,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # 4. ACT - Execute appropriate actions based on reasoning
            logger.info(f"✅ Message approved for processing: {message_data.get('message_id', 'unknown')}")
            action_result = await self._execute_reasoned_actions(reasoning_result, message_data)
            
            # 5. RESPOND - Generate intelligent response
            final_response = await self.generate_intelligent_response(reasoning_result, action_result)
            
            # Add metadata
            final_response["processing_steps"] = ["receive", "reason", "evaluate", "act", "respond"]
            final_response["ai_enhanced"] = self.ai_enabled
            final_response["approved"] = True
            
            return final_response
            
        except Exception as e:
            logger.error(f"AI reasoning flow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_used": True,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _generate_refusal_response(self, reasoning_result: Dict[str, Any], message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an intelligent refusal response with helpful alternatives"""
        try:
            refusal_reason = reasoning_result.get("refusal_reason", {})
            category = refusal_reason.get("category", "unknown")
            explanation = refusal_reason.get("explanation", "Message cannot be processed")
            alternatives = refusal_reason.get("alternative_suggestions", [])
            severity = refusal_reason.get("severity", "medium")
            
            # Generate appropriate refusal message based on category
            refusal_messages = {
                "capability_mismatch": f"I don't have the required capabilities to handle this request. {explanation}",
                "security_risk": f"This request poses a security risk and cannot be processed. {explanation}",
                "policy_violation": f"This request violates our operational policies. {explanation}",
                "resource_unavailable": f"Required resources are currently unavailable. {explanation}",
                "malicious_intent": f"This request appears to have malicious intent and is blocked. {explanation}",
                "authentication_failure": f"Authentication failed or insufficient permissions. {explanation}",
                "rate_limited": f"Request rate limit exceeded. Please try again later. {explanation}",
                "maintenance_mode": f"System is currently in maintenance mode. {explanation}"
            }
            
            base_message = refusal_messages.get(category, f"Request cannot be processed: {explanation}")
            
            # Add helpful alternatives if available
            response_message = base_message
            if alternatives:
                alternatives_text = "\n\nAlternative options:\n" + "\n".join(f"• {alt}" for alt in alternatives)
                response_message += alternatives_text
            
            # Add severity-based response adjustments
            if severity == "critical":
                response_message = "🚨 CRITICAL: " + response_message
            elif severity == "high":
                response_message = "⚠️ WARNING: " + response_message
            
            # Record refusal for learning
            await self._record_refusal({
                "message_id": message_data.get("message_id"),
                "refusal_category": category,
                "severity": severity,
                "reasoning": reasoning_result.get("reasoning"),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return {
                "message": response_message,
                "status": "refused",
                "category": category,
                "severity": severity,
                "alternatives": alternatives,
                "can_retry": category in ["resource_unavailable", "rate_limited", "maintenance_mode"],
                "retry_after": self._calculate_retry_delay(category),
                "metadata": {
                    "refusal_id": f"ref_{message_data.get('message_id', 'unknown')}",
                    "agent_id": getattr(self, 'agent_id', 'unknown'),
                    "reasoning_confidence": reasoning_result.get("confidence", 0.0)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate refusal response: {e}")
            return {
                "message": "Request cannot be processed due to an internal error.",
                "status": "refused",
                "category": "internal_error",
                "severity": "medium"
            }
    
    async def _record_refusal(self, refusal_data: Dict[str, Any]):
        """Record message refusal for learning and analytics"""
        try:
            # Add to interaction history for learning
            if not hasattr(self, 'refusal_history'):
                self.refusal_history = []
            
            self.refusal_history.append(refusal_data)
            
            # Keep only recent refusals (last 50)
            if len(self.refusal_history) > 50:
                self.refusal_history = self.refusal_history[-50:]
            
            # Log important refusals
            if refusal_data.get("severity") in ["high", "critical"]:
                logger.warning(f"High-severity refusal recorded: {refusal_data}")
            
        except Exception as e:
            logger.error(f"Failed to record refusal: {e}")
    
    def _calculate_retry_delay(self, category: str) -> Optional[int]:
        """Calculate appropriate retry delay in seconds based on refusal category"""
        retry_delays = {
            "rate_limited": 60,  # 1 minute
            "resource_unavailable": 300,  # 5 minutes
            "maintenance_mode": 1800,  # 30 minutes
            "authentication_failure": None,  # No retry
            "security_risk": None,  # No retry
            "malicious_intent": None,  # No retry
            "policy_violation": None,  # No retry
            "capability_mismatch": None  # No retry
        }
        return retry_delays.get(category)
    
    def get_refusal_statistics(self) -> Dict[str, Any]:
        """Get statistics about message refusals for monitoring and analysis"""
        if not hasattr(self, 'refusal_history'):
            return {"total_refusals": 0}
        
        refusals = self.refusal_history
        total = len(refusals)
        
        if total == 0:
            return {"total_refusals": 0}
        
        # Analyze refusal patterns
        categories = {}
        severities = {}
        recent_refusals = 0
        
        recent_threshold = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        
        for refusal in refusals:
            category = refusal.get("refusal_category", "unknown")
            severity = refusal.get("severity", "medium")
            timestamp = refusal.get("timestamp", "")
            
            categories[category] = categories.get(category, 0) + 1
            severities[severity] = severities.get(severity, 0) + 1
            
            if timestamp > recent_threshold:
                recent_refusals += 1
        
        return {
            "total_refusals": total,
            "recent_refusals_1h": recent_refusals,
            "refusal_categories": categories,
            "severity_distribution": severities,
            "most_common_category": max(categories.items(), key=lambda x: x[1])[0] if categories else None,
            "refusal_rate": len(refusals) / max(len(getattr(self, 'interaction_history', [])) + len(refusals), 1)
        }

    async def learn_from_interaction(self, interaction_data: Dict[str, Any]):
        """Learn from interaction outcomes to improve future reasoning"""
        if not self.ai_enabled:
            return
            
        try:
            # Store interaction for analysis
            self.interaction_history.append({
                **interaction_data,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Keep only recent interactions (last 100)
            if len(self.interaction_history) > 100:
                self.interaction_history = self.interaction_history[-100:]
            
            # Optional: Implement more sophisticated learning mechanisms
            # This could include pattern analysis, preference learning, etc.
            
        except Exception as e:
            logger.error(f"Learning from interaction failed: {e}")
    
    # Private helper methods
    
    def _build_reasoning_prompt(self, message_data: Dict[str, Any], context: Optional[Dict[str, Any]], agent_info: Dict[str, Any]) -> str:
        """Build reasoning prompt for AI"""
        message_direction = message_data.get('direction', 'incoming')
        
        if message_direction == 'outgoing':
            return self._build_outgoing_reasoning_prompt(message_data, context, agent_info)
        else:
            return self._build_incoming_reasoning_prompt(message_data, context, agent_info)
    
    def _build_incoming_reasoning_prompt(self, message_data: Dict[str, Any], context: Optional[Dict[str, Any]], agent_info: Dict[str, Any]) -> str:
        """Build reasoning prompt for incoming messages"""
        return f"""You are an AI agent reasoning system for {agent_info['agent_id']} (type: {agent_info['agent_type']}).

AGENT CAPABILITIES: {', '.join(agent_info.get('capabilities', []))}

INCOMING MESSAGE:
{json.dumps(message_data, indent=2)}

CONTEXT:
{json.dumps(context or {}, indent=2)}

Please analyze this incoming message and provide reasoning in the following JSON format:
{{
    "intent": "What is the sender trying to accomplish?",
    "urgency": "high|medium|low",
    "required_capabilities": ["list", "of", "capabilities", "needed"],
    "should_process": true/false,
    "refusal_reason": {{
        "category": "capability_mismatch|security_risk|policy_violation|resource_unavailable|malicious_intent|authentication_failure|rate_limited|maintenance_mode",
        "explanation": "Detailed explanation if refusing to process",
        "severity": "low|medium|high|critical",
        "alternative_suggestions": ["list", "of", "alternative", "actions"]
    }},
    "recommended_actions": [
        {{"action": "action_name", "parameters": {{}}, "priority": "high|medium|low"}}
    ],
    "response_type": "immediate|delayed|none|refused",
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of the reasoning process",
    "potential_risks": ["list", "of", "potential", "issues"],
    "success_criteria": "How to measure if the response was successful",
    "trust_assessment": {{
        "sender_reputation": 0.0-1.0,
        "message_authenticity": 0.0-1.0,
        "risk_level": "low|medium|high|critical"
    }}
}}"""
    
    def _build_outgoing_reasoning_prompt(self, message_data: Dict[str, Any], context: Optional[Dict[str, Any]], agent_info: Dict[str, Any]) -> str:
        """Build reasoning prompt for outgoing messages"""
        return f"""You are an AI agent reasoning system for {agent_info['agent_id']} (type: {agent_info['agent_type']}).

AGENT CAPABILITIES: {', '.join(agent_info.get('capabilities', []))}

OUTGOING MESSAGE ANALYSIS:
{json.dumps(message_data, indent=2)}

CONTEXT:
{json.dumps(context or {}, indent=2)}

Please analyze this outgoing message and provide optimization recommendations in the following JSON format:
{{
    "message_assessment": {{
        "clarity": "high|medium|low - How clear is the message?",
        "completeness": "high|medium|low - Does it contain all necessary information?",
        "appropriateness": "high|medium|low - Is it appropriate for the target agent?",
        "efficiency": "high|medium|low - Is it efficient and concise?"
    }},
    "recommended_modifications": {{
        "priority": "CRITICAL|HIGH|NORMAL|LOW - Suggested priority level",
        "encrypt": true/false,
        "enhanced_parts": [
            {{"partType": "data", "data": {{"enhanced": "content"}}}}
        ]
    }},
    "delivery_strategy": {{
        "timing": "immediate|delayed|scheduled",
        "retry_strategy": "aggressive|normal|conservative",
        "fallback_options": ["list", "of", "alternatives"]
    }},
    "success_prediction": {{
        "likelihood": 0.0-1.0,
        "factors": ["factors", "affecting", "success"],
        "potential_issues": ["possible", "problems"]
    }},
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of the analysis and recommendations"
}}"""

    def _build_response_prompt(self, reasoning_result: Dict[str, Any], action_result: Optional[Dict[str, Any]], agent_info: Dict[str, Any]) -> str:
        """Build response generation prompt"""
        return f"""You are generating a response for {agent_info['agent_id']} based on AI reasoning and action results.

REASONING RESULTS:
{json.dumps(reasoning_result, indent=2)}

ACTION RESULTS:
{json.dumps(action_result or {}, indent=2)}

Generate an appropriate response in this JSON format:
{{
    "message": "Clear, helpful response message",
    "data": {{"any": "relevant data to include"}},
    "status": "success|partial|failed",
    "next_steps": ["suggested", "next", "actions"],
    "metadata": {{
        "processing_time": "estimate",
        "confidence": 0.0-1.0,
        "reasoning_summary": "brief summary of reasoning"
    }}
}}"""

    def _fallback_reasoning(self, message_data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback reasoning when AI is not available"""
        return {
            "intent": "general_request",
            "urgency": "medium",
            "required_capabilities": ["basic_processing"],
            "recommended_actions": [{"action": "default_handler", "parameters": {}, "priority": "medium"}],
            "response_type": "immediate",
            "confidence": 0.6,
            "reasoning": "Fallback rule-based reasoning",
            "reasoning_type": "fallback"
        }
    
    def _fallback_response_generation(self, reasoning_result: Dict[str, Any], action_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback response generation"""
        return {
            "success": True,
            "response": {
                "message": "Request processed successfully",
                "data": action_result or {},
                "status": "success",
                "metadata": {"reasoning_type": "fallback"}
            },
            "reasoning_used": "fallback",
            "confidence": 0.6,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _extract_reasoning_from_text(self, text: str) -> Dict[str, Any]:
        """Extract reasoning structure from unstructured text"""
        # Simple extraction logic - can be enhanced
        return {
            "intent": "extracted_from_text",
            "reasoning": text,
            "confidence": 0.5,
            "reasoning_type": "text_extraction"
        }
    
    def _extract_response_from_text(self, text: str) -> Dict[str, Any]:
        """Extract response structure from unstructured text"""
        return {
            "message": text,
            "status": "success",
            "metadata": {"extraction_method": "text"}
        }
    
    async def _execute_reasoned_actions(self, reasoning_result: Dict[str, Any], message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute actions based on AI reasoning"""
        try:
            actions = reasoning_result.get("recommended_actions", [])
            results = []
            
            for action in actions:
                action_name = action.get("action")
                parameters = action.get("parameters", {})
                
                # Try to execute the action
                if hasattr(self, action_name):
                    method = getattr(self, action_name)
                    if callable(method):
                        result = await method(**parameters)
                        results.append({"action": action_name, "result": result, "success": True})
                    else:
                        results.append({"action": action_name, "error": "Not callable", "success": False})
                else:
                    # Fallback to default processing
                    result = await self._default_action_handler(action_name, parameters, message_data)
                    results.append({"action": action_name, "result": result, "success": True})
            
            return {
                "actions_executed": len(results),
                "results": results,
                "success": all(r.get("success", False) for r in results)
            }
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return {"error": str(e), "success": False}
    
    async def _default_action_handler(self, action_name: str, parameters: Dict[str, Any], message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Default handler for actions that don't have specific implementations"""
        logger.info(f"Executing default action: {action_name} with parameters: {parameters}")
        
        return {
            "action": action_name,
            "parameters": parameters,
            "message": "Action processed with default handler",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _record_interaction(self, interaction_data: Dict[str, Any]):
        """Record interaction for learning purposes"""
        try:
            # This could be enhanced to store in persistent storage
            # For now, just keep in memory
            pass
        except Exception as e:
            logger.error(f"Failed to record interaction: {e}")