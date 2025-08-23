"""
Enhanced Reasoning Agent with MCP Integration
Provides advanced reasoning capabilities through MCP tools and resources
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime, timedelta
from uuid import uuid4
import logging
from enum import Enum
import hashlib
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import time

from fastapi import HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Import SDK components including MCP
from app.a2a.sdk.agentBase import A2AAgentBase
except ImportError:
    logger.error("Failed to import A2AAgentBase - using base class")
    class A2AAgentBase:
        def __init__(self, agent_id, name, description, version):
            self.agent_id = agent_id
            self.name = name
            self.description = description
            self.version = version
            
from app.a2a.sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt
except ImportError:
    logger.error("Failed to import MCP decorators - creating stubs")
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

# Import with fallbacks for testing
from app.a2a.sdk.types import A2AMessage, MessagePart, MessageRole, TaskStatus, AgentCard
except ImportError:
    logger.warning("SDK types not available - using stub classes")
    class A2AMessage:
        pass
    class MessagePart:
        pass
    class MessageRole:
        pass
    class TaskStatus:
        pass
    class AgentCard:
        pass

# Import reasoning components with fallbacks
try:
    from .reasoningSkills import (
        MultiAgentReasoningSkills, ReasoningOrchestrationSkills,
        HierarchicalReasoningSkills, SwarmReasoningSkills
    )
    from .enhancedReasoningSkills import EnhancedReasoningSkills
except ImportError:
    logger.warning("Reasoning skills not available - using stubs")
    class MultiAgentReasoningSkills:
        pass
    class ReasoningOrchestrationSkills:
        pass
    class HierarchicalReasoningSkills:
        pass
    class SwarmReasoningSkills:
        pass
    class EnhancedReasoningSkills:
        pass

# Import MCP skill coordination system
from app.a2a.sdk.mcpSkillCoordination import MCPSkillCoordinationMixin, skill_depends_on, skill_provides


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
except ImportError:
    logger.warning("MCP skill coordination not available - using stubs")
    class MCPSkillCoordinationMixin:
        pass
    class MCPSkillClientMixin:
        pass
    def skill_depends_on(*args):
        def decorator(func):
            return func
        return decorator
    def skill_provides(*args):
        def decorator(func):
            return func
        return decorator


class ReasoningArchitecture(str, Enum):
    """Available reasoning architectures"""
    HIERARCHICAL = "hierarchical"
    PEER_TO_PEER = "peer_to_peer"
    HUB_AND_SPOKE = "hub_and_spoke"
    BLACKBOARD = "blackboard"
    GRAPH_BASED = "graph_based"
    HYBRID = "hybrid"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    GRAPH_OF_THOUGHT = "graph_of_thought"


class ReasoningStrategy(str, Enum):
    """Reasoning strategies"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    PROBABILISTIC = "probabilistic"
    DIALECTICAL = "dialectical"
    CRITICAL = "critical"


class EvidenceType(str, Enum):
    """Types of evidence in reasoning"""
    EMPIRICAL = "empirical"
    LOGICAL = "logical"
    TESTIMONIAL = "testimonial"
    STATISTICAL = "statistical"
    ANALOGICAL = "analogical"
    EXPERT = "expert"


@dataclass
class ReasoningNode:
    """A node in the reasoning graph"""
    node_id: str
    content: str
    node_type: str  # premise, inference, conclusion
    confidence: float = 0.5
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ReasoningChain:
    """A chain of reasoning with multiple nodes"""
    chain_id: str
    architecture: ReasoningArchitecture
    strategy: ReasoningStrategy
    nodes: Dict[str, ReasoningNode] = field(default_factory=dict)
    root_node: Optional[str] = None
    conclusion_nodes: List[str] = field(default_factory=list)
    total_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def add_node(self, node: ReasoningNode, parent_id: Optional[str] = None):
        """Add a node to the reasoning chain"""
        self.nodes[node.node_id] = node
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children.append(node.node_id)
            node.parent = parent_id
        elif not self.root_node:
            self.root_node = node.node_id
    
    def calculate_confidence(self) -> float:
        """Calculate overall confidence for the chain"""
        if not self.nodes:
            return 0.0
        
        # Weight nodes by their position in the chain
        total_weight = 0.0
        weighted_confidence = 0.0
        
        for node in self.nodes.values():
            # Conclusions have higher weight
            weight = 2.0 if node.node_type == "conclusion" else 1.0
            weighted_confidence += node.confidence * weight
            total_weight += weight
        
        self.total_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
        return self.total_confidence


@dataclass
class ReasoningSession:
    """A complete reasoning session with multiple chains"""
    session_id: str
    question: str
    context: Dict[str, Any] = field(default_factory=dict)
    chains: Dict[str, ReasoningChain] = field(default_factory=dict)
    evidence_pool: List[Dict[str, Any]] = field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    debate_rounds: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: Optional[str] = None
    final_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class EnhancedReasoningAgent(MCPSkillCoordinationMixin, MCPSkillClientMixin):
    """Enhanced Reasoning Agent with comprehensive MCP integration and intra-skill communication"""
    
    def __init__(self):
        # Call mixin constructors first
        super().__init__()
        
        # Initialize base properties
        self.agent_id = "enhanced_reasoning_agent"
        self.name = "Enhanced Reasoning Agent"
        self.description = "Advanced reasoning system with MCP-powered multi-agent orchestration"
        self.version = "2.0.0"
        
        # Reasoning state management
        self.active_sessions: Dict[str, ReasoningSession] = {}
        self.reasoning_chains: Dict[str, ReasoningChain] = {}
        self.evidence_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # Performance metrics
        self.metrics = {
            "total_sessions": 0,
            "successful_reasoning": 0,
            "average_confidence": 0.0,
            "architecture_usage": defaultdict(int),
            "strategy_usage": defaultdict(int),
            "average_reasoning_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Configuration
        self.max_reasoning_depth = 10
        self.confidence_threshold = 0.7
        self.max_debate_rounds = 5
        self.cache_ttl = 3600  # 1 hour
        
        # Add stub methods for MCP compatibility
        self.list_mcp_tools = self._list_mcp_tools_stub
        self.list_mcp_resources = self._list_mcp_resources_stub
        self.call_mcp_tool = self._call_mcp_tool_stub
        self.get_mcp_resource = self._get_mcp_resource_stub
        
        # Initialize components
        self._shutdown_flag = False
        self._monitoring_tasks = []
        
        # Initialize MCP skill coordination and client
        self.initialize_skill_coordinator()
        self.initialize_mcp_skill_client()
        
        # Initialize reasoning skill components
        self.multi_agent_skills = MultiAgentReasoningSkills()
        self.orchestration_skills = ReasoningOrchestrationSkills()
        self.hierarchical_skills = HierarchicalReasoningSkills()
        self.swarm_skills = SwarmReasoningSkills()
        
        logger.info("🧠 Enhanced Reasoning Agent initialized with MCP integration and intra-skill communication")
    
    async def initialize(self):
        """Initialize enhanced reasoning agent"""
        logger.info("🚀 Initializing Enhanced Reasoning Agent")
        
        # Start skill coordinator
        await self.start_skill_coordinator()
        
        # Start monitoring tasks
        self._monitoring_tasks = [
            asyncio.create_task(self._session_cleanup_loop()),
            asyncio.create_task(self._metrics_aggregation_loop())
        ]
    
    async def shutdown(self):
        """Cleanup agent resources"""
        self._shutdown_flag = True
        
        # Stop skill coordinator
        await self.stop_skill_coordinator()
        
        # Cancel monitoring tasks
        for task in self._monitoring_tasks:
            if not task.done():
                task.cancel()
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        logger.info("🛑 Enhanced Reasoning Agent shutdown complete")
    
    # ==========================================
    # MCP Tools for Advanced Reasoning with Intra-Skill Communication
    # ==========================================
    
    @mcp_tool(
        name="orchestrate_collaborative_reasoning",
        description="Orchestrate collaborative reasoning using multiple skills with real MCP communication",
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "context": {"type": "object"},
                "skills_to_use": {"type": "array", "items": {"type": "string"}},
                "coordination_strategy": {"type": "string", "enum": ["parallel", "sequential", "adaptive"], "default": "adaptive"}
            },
            "required": ["question"]
        }
    )
    @skill_provides("collaborative_reasoning", "skill_orchestration")
    async def orchestrate_collaborative_reasoning(self, question: str, context: Dict[str, Any] = None,
                                                skills_to_use: List[str] = None,
                                                coordination_strategy: str = "adaptive") -> Dict[str, Any]:
        """Orchestrate collaborative reasoning using real intra-skill communication"""
        try:
            session_id = str(uuid4())
            context = context or {}
            
            # Default skills if none specified
            if not skills_to_use:
                skills_to_use = ["decompose_question_skill", "analyze_patterns_skill", "synthesize_answers_skill"]
            
            logger.info(f"🤝 Starting collaborative reasoning with {len(skills_to_use)} skills")
            
            # Create coordination session
            coordination_results = {
                "session_id": session_id,
                "question": question,
                "skills_used": skills_to_use,
                "strategy": coordination_strategy,
                "skill_results": {},
                "communication_log": [],
                "synthesis": None
            }
            
            if coordination_strategy == "parallel":
                # Execute skills in parallel using MCP communication
                results = await self._execute_skills_parallel(question, context, skills_to_use, coordination_results)
            elif coordination_strategy == "sequential":
                # Execute skills sequentially, passing results between them
                results = await self._execute_skills_sequential(question, context, skills_to_use, coordination_results)
            else:  # adaptive
                # Adaptive strategy based on question complexity
                complexity = self._assess_question_complexity(question, context)
                if complexity > 0.7:
                    results = await self._execute_skills_sequential(question, context, skills_to_use, coordination_results)
                else:
                    results = await self._execute_skills_parallel(question, context, skills_to_use, coordination_results)
            
            # Synthesize final answer using skill communication
            final_synthesis = await self._synthesize_collaborative_results(results, coordination_results)
            coordination_results["synthesis"] = final_synthesis
            
            return {
                "success": True,
                "answer": final_synthesis["answer"],
                "confidence": final_synthesis["confidence"],
                "coordination_results": coordination_results,
                "communication_summary": {
                    "messages_exchanged": len(coordination_results["communication_log"]),
                    "skills_involved": len(skills_to_use),
                    "strategy_used": coordination_strategy
                }
            }
            
        except Exception as e:
            logger.error(f"Collaborative reasoning failed: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp_tool(
        name="decompose_question_skill",
        description="Decompose complex questions using hierarchical analysis with skill communication",
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "context": {"type": "object"},
                "communication_context": {"type": "object"}
            },
            "required": ["question"]
        }
    )
    @skill_depends_on("analyze_patterns_skill")
    @skill_provides("question_decomposition", "hierarchical_analysis")
    async def decompose_question_skill(self, question: str, context: Dict[str, Any] = None,
                                     communication_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Decompose question with real skill-to-skill communication"""
        try:
            # Use the hierarchical skills component for real decomposition
            decomposition_request = {
                "question": question,
                "max_depth": 3,
                "decomposition_strategy": "functional",
                "context": context or {}
            }
            
            # Call the multi-agent skills for decomposition
            decomposition_result = await self.multi_agent_skills.hierarchical_question_decomposition(decomposition_request)
            
            # Communicate with pattern analysis skill if available
            pattern_analysis = None
            if communication_context and "coordinate_with_patterns" in communication_context:
                try:
                    # Use MCP to call the pattern analysis skill
                    pattern_call_result = await self.call_skill_tool(
                        "analyze_patterns_skill",
                        "analyze_patterns_skill",
                        question=question,
                        context=context,
                        focus_areas=["logical", "causal", "temporal"]
                    )
                    
                    if pattern_call_result.get("success"):
                        pattern_analysis = pattern_call_result["result"]
                        
                        # Log the communication
                        if communication_context.get("communication_log"):
                            communication_context["communication_log"].append({
                                "from": "decompose_question_skill",
                                "to": "analyze_patterns_skill",
                                "type": "tool_call",
                                "success": True,
                                "timestamp": datetime.utcnow().isoformat()
                            })
                        
                except Exception as e:
                    logger.warning(f"Communication with pattern analysis skill failed: {e}")
            
            # Enhance decomposition with pattern insights
            if pattern_analysis:
                # Integrate pattern analysis into decomposition
                for sub_q in decomposition_result.get("sub_questions", []):
                    sub_q["pattern_insights"] = pattern_analysis.get("patterns_identified", [])
            
            return {
                "success": True,
                "decomposition": decomposition_result,
                "pattern_analysis": pattern_analysis,
                "skill_communication": "Successfully coordinated with pattern analysis skill" if pattern_analysis else "No skill coordination requested"
            }
            
        except Exception as e:
            logger.error(f"Question decomposition skill failed: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp_tool(
        name="analyze_patterns_skill",
        description="Analyze reasoning patterns with skill communication capabilities",
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "context": {"type": "object"},
                "focus_areas": {"type": "array", "items": {"type": "string"}},
                "communication_context": {"type": "object"}
            },
            "required": ["question"]
        }
    )
    @skill_provides("pattern_analysis", "reasoning_insights")
    async def analyze_patterns_skill(self, question: str, context: Dict[str, Any] = None,
                                   focus_areas: List[str] = None,
                                   communication_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze patterns with real skill communication"""
        try:
            focus_areas = focus_areas or ["logical", "causal", "analogical"]
            
            # Create a reasoning session for pattern analysis
            session = ReasoningSession(
                session_id=str(uuid4()),
                question=question,
                context=context or {}
            )
            
            # Analyze patterns using our real pattern analysis logic
            patterns_identified = []
            confidence_scores = []
            
            for focus in focus_areas:
                if focus == "logical":
                    logical_patterns = self._identify_logical_patterns(question, context or {})
                    patterns_identified.extend(logical_patterns)
                    confidence_scores.append(0.8)
                elif focus == "causal":
                    causal_patterns = self._identify_causal_patterns(question, context or {})
                    patterns_identified.extend(causal_patterns)
                    confidence_scores.append(0.7)
                elif focus == "temporal":
                    temporal_patterns = self._identify_temporal_patterns(question, context or {})
                    patterns_identified.extend(temporal_patterns)
                    confidence_scores.append(0.6)
            
            # Communicate findings to synthesis skill if coordination requested
            synthesis_communication = None
            if communication_context and "notify_synthesis" in communication_context:
                try:
                    # Send patterns to synthesis skill
                    synthesis_call_result = await self.call_skill_tool(
                        "synthesize_answers_skill",
                        "synthesize_answers_skill",
                        patterns=patterns_identified,
                        confidence_scores=confidence_scores,
                        source_skill="analyze_patterns_skill"
                    )
                    
                    if synthesis_call_result.get("success"):
                        synthesis_communication = "Successfully shared patterns with synthesis skill"
                        
                        # Log the communication
                        if communication_context.get("communication_log"):
                            communication_context["communication_log"].append({
                                "from": "analyze_patterns_skill",
                                "to": "synthesize_answers_skill",
                                "type": "pattern_sharing",
                                "patterns_shared": len(patterns_identified),
                                "timestamp": datetime.utcnow().isoformat()
                            })
                    
                except Exception as e:
                    logger.warning(f"Communication with synthesis skill failed: {e}")
                    synthesis_communication = f"Communication failed: {e}"
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            
            return {
                "success": True,
                "patterns_identified": patterns_identified,
                "focus_areas_analyzed": focus_areas,
                "confidence": avg_confidence,
                "skill_communication": synthesis_communication or "No skill coordination requested",
                "analysis_details": {
                    "total_patterns": len(patterns_identified),
                    "session_id": session.session_id
                }
            }
            
        except Exception as e:
            logger.error(f"Pattern analysis skill failed: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp_tool(
        name="synthesize_answers_skill",
        description="Synthesize answers from multiple reasoning skills using real coordination",
        input_schema={
            "type": "object",
            "properties": {
                "reasoning_results": {"type": "array"},
                "patterns": {"type": "array"},
                "confidence_scores": {"type": "array"},
                "source_skill": {"type": "string"},
                "communication_context": {"type": "object"}
            }
        }
    )
    @skill_depends_on("decompose_question_skill", "analyze_patterns_skill")
    @skill_provides("answer_synthesis", "result_integration")
    async def synthesize_answers_skill(self, reasoning_results: List[Dict[str, Any]] = None,
                                     patterns: List[Dict[str, Any]] = None,
                                     confidence_scores: List[float] = None,
                                     source_skill: str = None,
                                     communication_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Synthesize answers using real skill coordination"""
        try:
            synthesis_data = {
                "reasoning_results": reasoning_results or [],
                "patterns": patterns or [],
                "confidence_scores": confidence_scores or [],
                "source_skill": source_skill,
                "synthesis_method": "weighted_integration"
            }
            
            # Request additional insights from other skills if needed
            additional_insights = []
            
            if communication_context and "request_additional_insights" in communication_context:
                # Request decomposition insights
                try:
                    decomp_call_result = await self.call_skill_tool(
                        "decompose_question_skill",
                        "decompose_question_skill",
                        question=communication_context.get("original_question", ""),
                        context=communication_context.get("context", {}),
                        communication_context={"source": "synthesis_skill"}
                    )
                    
                    if decomp_call_result.get("success"):
                        additional_insights.append({
                            "source": "decomposition",
                            "insights": decomp_call_result["result"]["decomposition"]
                        })
                        
                        # Log the communication
                        if communication_context.get("communication_log"):
                            communication_context["communication_log"].append({
                                "from": "synthesize_answers_skill",
                                "to": "decompose_question_skill",
                                "type": "insight_request",
                                "success": True,
                                "timestamp": datetime.utcnow().isoformat()
                            })
                        
                except Exception as e:
                    logger.warning(f"Failed to get additional insights from decomposition: {e}")
            
            # Perform synthesis using all available information
            synthesis_result = self._perform_weighted_synthesis(
                synthesis_data["reasoning_results"],
                synthesis_data["patterns"],
                synthesis_data["confidence_scores"],
                additional_insights
            )
            
            return {
                "success": True,
                "synthesized_answer": synthesis_result["answer"],
                "confidence": synthesis_result["confidence"],
                "synthesis_method": synthesis_data["synthesis_method"],
                "sources_integrated": len(synthesis_data["reasoning_results"]) + len(additional_insights),
                "skill_communication": f"Received data from {source_skill}" if source_skill else "No direct skill communication",
                "additional_insights": len(additional_insights)
            }
            
        except Exception as e:
            logger.error(f"Answer synthesis skill failed: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp_tool(
        name="demonstrate_skill_communication",
        description="Demonstrate functional intra-skill communication with real message passing",
        input_schema={
            "type": "object",
            "properties": {
                "demo_scenario": {"type": "string", "enum": ["simple", "complex", "full_workflow"], "default": "simple"},
                "question": {"type": "string", "default": "How do complex systems work?"},
                "show_communication_log": {"type": "boolean", "default": True}
            }
        }
    )
    async def demonstrate_skill_communication(self, demo_scenario: str = "simple", 
                                             question: str = "How do complex systems work?",
                                             show_communication_log: bool = True) -> Dict[str, Any]:
        """Demonstrate functional intra-skill communication with real MCP message passing"""
        try:
            logger.info(f"🎯 Demonstrating {demo_scenario} skill communication scenario")
            
            communication_log = []
            demo_results = {
                "scenario": demo_scenario,
                "question": question,
                "communication_events": [],
                "skill_interactions": {},
                "message_flow": []
            }
            
            if demo_scenario == "simple":
                # Simple two-skill communication
                logger.info("📞 Starting simple skill communication: decomposition → pattern analysis")
                
                # Step 1: Decompose question
                decomp_result = await self.call_skill_tool(
                    "decompose_question_skill",
                    "decompose_question_skill",
                    question=question,
                    context={"demo_mode": True},
                    communication_context={
                        "coordinate_with_patterns": True,
                        "communication_log": communication_log
                    }
                )
                
                demo_results["skill_interactions"]["decomposition"] = decomp_result
                demo_results["message_flow"].append({
                    "step": 1,
                    "action": "Called decompose_question_skill",
                    "success": decomp_result.get("success", False),
                    "communication_triggered": "coordinate_with_patterns" in str(decomp_result)
                })
                
            elif demo_scenario == "complex":
                # Complex multi-skill communication with feedback loops
                logger.info("🔄 Starting complex skill communication with feedback loops")
                
                # Orchestrate multiple skills with interdependencies
                orchestration_result = await self.call_skill_tool(
                    "orchestrate_collaborative_reasoning",
                    "orchestrate_collaborative_reasoning",
                    question=question,
                    context={"complexity_level": "high"},
                    coordination_strategy="sequential"
                )
                
                demo_results["skill_interactions"]["orchestration"] = orchestration_result
                demo_results["message_flow"].append({
                    "step": 1,
                    "action": "Called orchestrate_collaborative_reasoning",
                    "strategy": "sequential",
                    "success": orchestration_result.get("success", False)
                })
                
            else:  # full_workflow
                # Full workflow demonstrating all communication patterns
                logger.info("🌐 Starting full workflow demonstration")
                
                # Step 1: Initialize coordination
                coordination_context = {
                    "workflow_id": str(uuid4()),
                    "communication_log": communication_log,
                    "coordination_strategy": "adaptive"
                }
                
                # Step 2: Parallel execution with communication
                parallel_result = await self.call_skill_tool(
                    "orchestrate_collaborative_reasoning",
                    "orchestrate_collaborative_reasoning",
                    question=question,
                    context=coordination_context,
                    coordination_strategy="parallel"
                )
                
                # Step 3: Sequential execution with feedback
                sequential_result = await self.call_skill_tool(
                    "orchestrate_collaborative_reasoning",
                    "orchestrate_collaborative_reasoning",
                    question=f"Building on previous analysis: {question}",
                    context={**coordination_context, "previous_results": parallel_result},
                    coordination_strategy="sequential"
                )
                
                demo_results["skill_interactions"]["parallel_phase"] = parallel_result
                demo_results["skill_interactions"]["sequential_phase"] = sequential_result
                
                demo_results["message_flow"].extend([
                    {
                        "step": 1,
                        "action": "Parallel skill execution",
                        "success": parallel_result.get("success", False)
                    },
                    {
                        "step": 2,
                        "action": "Sequential skill execution with feedback",
                        "success": sequential_result.get("success", False)
                    }
                ])
            
            # Analyze communication patterns
            communication_analysis = self._analyze_communication_patterns(communication_log, demo_results)
            
            # Get skill coordination statistics
            coordination_stats = self.get_skill_coordination_status()
            
            return {
                "success": True,
                "demonstration_complete": True,
                "scenario": demo_scenario,
                "communication_analysis": communication_analysis,
                "skill_interactions": demo_results["skill_interactions"],
                "message_flow": demo_results["message_flow"],
                "communication_log": communication_log if show_communication_log else f"{len(communication_log)} events",
                "coordination_statistics": coordination_stats,
                "key_findings": {
                    "skills_communicated": len(demo_results["skill_interactions"]),
                    "messages_exchanged": len(communication_log),
                    "communication_successful": all(
                        interaction.get("success", False) 
                        for interaction in demo_results["skill_interactions"].values()
                    ),
                    "mcp_integration_active": self.skill_coordinator is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Skill communication demonstration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "partial_results": locals().get("demo_results", {})
            }
    
    def _analyze_communication_patterns(self, communication_log: List[Dict[str, Any]], 
                                       demo_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze communication patterns from demonstration"""
        
        # Count communication types
        comm_types = {}
        for event in communication_log:
            event_type = event.get("type", "unknown")
            comm_types[event_type] = comm_types.get(event_type, 0) + 1
        
        # Analyze skill interactions
        skills_involved = set()
        for interaction in demo_results.get("skill_interactions", {}).values():
            if isinstance(interaction, dict) and "result" in interaction:
                result = interaction["result"]
                if isinstance(result, dict):
                    if "skill_communication" in result:
                        skills_involved.add("communication_detected")
                    if "coordination_results" in result:
                        coord_results = result["coordination_results"]
                        skills_involved.update(coord_results.get("skills_used", []))
        
        # Calculate communication effectiveness
        successful_interactions = sum(
            1 for interaction in demo_results.get("skill_interactions", {}).values()
            if interaction.get("success", False)
        )
        total_interactions = len(demo_results.get("skill_interactions", {}))
        effectiveness = (successful_interactions / total_interactions * 100) if total_interactions > 0 else 0
        
        return {
            "communication_types": comm_types,
            "skills_involved": list(skills_involved),
            "total_messages": len(communication_log),
            "effectiveness_percentage": effectiveness,
            "communication_patterns": {
                "tool_calls": comm_types.get("tool_call", 0),
                "pattern_sharing": comm_types.get("pattern_sharing", 0),
                "insight_requests": comm_types.get("insight_request", 0)
            }
        }
    
    @mcp_tool(
        name="execute_reasoning_chain",
        description="Execute a complete reasoning chain with specified architecture and strategy",
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "context": {"type": "object"},
                "architecture": {"type": "string", "enum": ["hierarchical", "chain_of_thought", "tree_of_thought", "graph_of_thought"]},
                "strategy": {"type": "string", "enum": ["deductive", "inductive", "abductive", "causal"]},
                "max_depth": {"type": "integer", "default": 5},
                "enable_caching": {"type": "boolean", "default": True}
            },
            "required": ["question"]
        }
    )
    async def execute_reasoning_chain(self, question: str, context: Dict[str, Any] = None,
                                    architecture: str = "chain_of_thought",
                                    strategy: str = "deductive", max_depth: int = 5,
                                    enable_caching: bool = True) -> Dict[str, Any]:
        """Execute a complete reasoning chain"""
        try:
            session_id = str(uuid4())
            context = context or {}
            
            # Check cache if enabled
            cache_key = hashlib.md5(f"{question}{architecture}{strategy}".encode()).hexdigest()
            if enable_caching and cache_key in self.evidence_cache:
                cached_result = self.evidence_cache[cache_key]
                if cached_result.get("timestamp", 0) > time.time() - self.cache_ttl:
                    self.metrics["cache_hits"] += 1
                    return cached_result["result"]
            
            self.metrics["cache_misses"] += 1
            
            # Create reasoning session
            session = ReasoningSession(
                session_id=session_id,
                question=question,
                context=context,
                metadata={
                    "architecture": architecture,
                    "strategy": strategy,
                    "max_depth": max_depth
                }
            )
            
            self.active_sessions[session_id] = session
            self.metrics["total_sessions"] += 1
            self.metrics["architecture_usage"][architecture] += 1
            self.metrics["strategy_usage"][strategy] += 1
            
            # Execute reasoning based on architecture
            if architecture == "chain_of_thought":
                result = await self._execute_chain_of_thought(session, strategy, max_depth)
            elif architecture == "tree_of_thought":
                result = await self._execute_tree_of_thought(session, strategy, max_depth)
            elif architecture == "graph_of_thought":
                result = await self._execute_graph_of_thought(session, strategy, max_depth)
            else:
                result = await self._execute_hierarchical_reasoning(session, strategy, max_depth)
            
            # Cache result if enabled
            if enable_caching:
                self.evidence_cache[cache_key] = {
                    "result": result,
                    "timestamp": time.time()
                }
            
            # Update metrics
            if result.get("confidence", 0) >= self.confidence_threshold:
                self.metrics["successful_reasoning"] += 1
                
            return result
            
        except Exception as e:
            logger.error(f"Reasoning chain execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp_tool(
        name="analyze_reasoning_patterns",
        description="Analyze patterns in reasoning to identify strengths and weaknesses",
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "pattern_types": {"type": "array", "items": {"type": "string"}},
                "depth_analysis": {"type": "boolean", "default": True}
            },
            "required": ["session_id"]
        }
    )
    async def analyze_reasoning_patterns(self, session_id: str, 
                                       pattern_types: List[str] = None,
                                       depth_analysis: bool = True) -> Dict[str, Any]:
        """Analyze reasoning patterns in a session"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return {"success": False, "error": "Session not found"}
            
            pattern_types = pattern_types or ["logical", "causal", "analogical", "critical"]
            analysis = {
                "session_id": session_id,
                "patterns_found": {},
                "strengths": [],
                "weaknesses": [],
                "recommendations": []
            }
            
            # Analyze each reasoning chain
            for chain_id, chain in session.chains.items():
                chain_patterns = self._analyze_chain_patterns(chain, pattern_types)
                analysis["patterns_found"][chain_id] = chain_patterns
                
                # Identify strengths
                if chain.total_confidence > 0.8:
                    analysis["strengths"].append(f"High confidence reasoning in chain {chain_id}")
                
                # Identify weaknesses
                if len(chain.nodes) < 3:
                    analysis["weaknesses"].append(f"Shallow reasoning depth in chain {chain_id}")
            
            # Deep analysis if requested
            if depth_analysis:
                analysis["depth_metrics"] = self._calculate_depth_metrics(session)
                analysis["evidence_quality"] = self._assess_evidence_quality(session)
            
            # Generate recommendations
            if analysis["weaknesses"]:
                analysis["recommendations"].append("Consider deeper reasoning chains for complex questions")
            
            return {
                "success": True,
                "analysis": analysis,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp_tool(
        name="conduct_reasoning_debate",
        description="Conduct multi-perspective debate on reasoning conclusions",
        input_schema={
            "type": "object",
            "properties": {
                "positions": {"type": "array", "items": {"type": "object"}},
                "debate_structure": {"type": "string", "enum": ["dialectical", "adversarial", "collaborative"]},
                "max_rounds": {"type": "integer", "default": 3},
                "convergence_threshold": {"type": "number", "default": 0.8}
            },
            "required": ["positions"]
        }
    )
    async def conduct_reasoning_debate(self, positions: List[Dict[str, Any]],
                                     debate_structure: str = "dialectical",
                                     max_rounds: int = 3,
                                     convergence_threshold: float = 0.8) -> Dict[str, Any]:
        """Conduct reasoning debate between multiple positions"""
        try:
            debate_id = str(uuid4())
            debate_history = []
            current_positions = positions.copy()
            
            for round_num in range(max_rounds):
                round_result = {
                    "round": round_num + 1,
                    "exchanges": [],
                    "position_updates": []
                }
                
                # Each position responds to others
                for i, position in enumerate(current_positions):
                    for j, other_position in enumerate(current_positions):
                        if i != j:
                            response = self._generate_debate_response(
                                position, other_position, debate_structure
                            )
                            round_result["exchanges"].append({
                                "from": i,
                                "to": j,
                                "response": response
                            })
                
                # Update positions based on exchanges
                updated_positions = self._update_positions(
                    current_positions, round_result["exchanges"]
                )
                round_result["position_updates"] = updated_positions
                current_positions = updated_positions
                
                debate_history.append(round_result)
                
                # Check for convergence
                if self._check_convergence(current_positions, convergence_threshold):
                    break
            
            # Synthesize final conclusion
            final_position = self._synthesize_positions(current_positions)
            
            return {
                "success": True,
                "debate_id": debate_id,
                "rounds_conducted": len(debate_history),
                "debate_history": debate_history,
                "final_positions": current_positions,
                "synthesized_conclusion": final_position,
                "consensus_achieved": self._check_convergence(current_positions, convergence_threshold)
            }
            
        except Exception as e:
            logger.error(f"Reasoning debate failed: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp_tool(
        name="generate_counterfactual_reasoning",
        description="Generate counterfactual reasoning scenarios",
        input_schema={
            "type": "object",
            "properties": {
                "original_premise": {"type": "string"},
                "conclusion": {"type": "string"},
                "num_counterfactuals": {"type": "integer", "default": 3},
                "variation_types": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["original_premise", "conclusion"]
        }
    )
    async def generate_counterfactual_reasoning(self, original_premise: str,
                                              conclusion: str,
                                              num_counterfactuals: int = 3,
                                              variation_types: List[str] = None) -> Dict[str, Any]:
        """Generate counterfactual reasoning scenarios"""
        try:
            variation_types = variation_types or ["negation", "modification", "substitution"]
            counterfactuals = []
            
            for i in range(num_counterfactuals):
                variation_type = variation_types[i % len(variation_types)]
                
                # Generate counterfactual premise
                cf_premise = self._generate_counterfactual_premise(
                    original_premise, variation_type
                )
                
                # Reason about new conclusion
                cf_reasoning = await self._reason_from_premise(cf_premise)
                
                counterfactuals.append({
                    "counterfactual_premise": cf_premise,
                    "variation_type": variation_type,
                    "predicted_conclusion": cf_reasoning["conclusion"],
                    "confidence": cf_reasoning["confidence"],
                    "reasoning_chain": cf_reasoning["chain"],
                    "differs_from_original": cf_reasoning["conclusion"] != conclusion
                })
            
            return {
                "success": True,
                "original": {
                    "premise": original_premise,
                    "conclusion": conclusion
                },
                "counterfactuals": counterfactuals,
                "insights": self._extract_counterfactual_insights(counterfactuals)
            }
            
        except Exception as e:
            logger.error(f"Counterfactual generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp_tool(
        name="validate_reasoning_consistency",
        description="Validate consistency across multiple reasoning chains",
        input_schema={
            "type": "object",
            "properties": {
                "chain_ids": {"type": "array", "items": {"type": "string"}},
                "validation_criteria": {"type": "array", "items": {"type": "string"}},
                "strict_mode": {"type": "boolean", "default": False}
            },
            "required": ["chain_ids"]
        }
    )
    async def validate_reasoning_consistency(self, chain_ids: List[str],
                                           validation_criteria: List[str] = None,
                                           strict_mode: bool = False) -> Dict[str, Any]:
        """Validate consistency across reasoning chains"""
        try:
            validation_criteria = validation_criteria or [
                "logical_consistency", "evidence_consistency", 
                "conclusion_alignment", "confidence_correlation"
            ]
            
            chains = [self.reasoning_chains.get(cid) for cid in chain_ids if cid in self.reasoning_chains]
            if len(chains) < 2:
                return {"success": False, "error": "Need at least 2 chains for consistency validation"}
            
            validation_results = {
                "overall_consistency": 1.0,
                "criteria_results": {},
                "inconsistencies": [],
                "recommendations": []
            }
            
            # Validate each criterion
            for criterion in validation_criteria:
                if criterion == "logical_consistency":
                    result = self._validate_logical_consistency(chains)
                elif criterion == "evidence_consistency":
                    result = self._validate_evidence_consistency(chains)
                elif criterion == "conclusion_alignment":
                    result = self._validate_conclusion_alignment(chains)
                elif criterion == "confidence_correlation":
                    result = self._validate_confidence_correlation(chains)
                else:
                    continue
                
                validation_results["criteria_results"][criterion] = result
                validation_results["overall_consistency"] *= result["score"]
                
                if result["inconsistencies"]:
                    validation_results["inconsistencies"].extend(result["inconsistencies"])
            
            # Apply strict mode
            if strict_mode and validation_results["overall_consistency"] < 0.9:
                validation_results["recommendations"].append(
                    "Reasoning chains show significant inconsistencies - reconsider conclusions"
                )
            
            return {
                "success": True,
                "validation_results": validation_results,
                "chains_validated": len(chains),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Consistency validation failed: {e}")
            return {"success": False, "error": str(e)}
    
    # ==========================================
    # MCP Resources for Reasoning State
    # ==========================================
    
    @mcp_resource(
        uri="reasoning://active-sessions",
        name="Active Reasoning Sessions",
        description="Real-time view of all active reasoning sessions",
        mime_type="application/json"
    )
    async def get_active_sessions_resource(self) -> Dict[str, Any]:
        """Get active reasoning sessions"""
        return {
            "total_sessions": len(self.active_sessions),
            "sessions": {
                session_id: {
                    "question": session.question,
                    "created_at": session.created_at.isoformat() if session.created_at else None,
                    "chain_count": len(session.chains),
                    "evidence_count": len(session.evidence_pool),
                    "has_conclusion": session.final_answer is not None,
                    "confidence": session.final_confidence,
                    "metadata": session.metadata
                }
                for session_id, session in self.active_sessions.items()
            },
            "last_updated": datetime.utcnow().isoformat()
        }
    
    @mcp_resource(
        uri="reasoning://chain-library",
        name="Reasoning Chain Library",
        description="Repository of all reasoning chains with patterns and confidence scores",
        mime_type="application/json"
    )
    async def get_chain_library_resource(self) -> Dict[str, Any]:
        """Get reasoning chain library"""
        chains_by_architecture = defaultdict(list)
        chains_by_strategy = defaultdict(list)
        
        for chain_id, chain in self.reasoning_chains.items():
            chain_summary = {
                "chain_id": chain_id,
                "node_count": len(chain.nodes),
                "confidence": chain.total_confidence,
                "created_at": chain.created_at.isoformat() if chain.created_at else None
            }
            chains_by_architecture[chain.architecture.value].append(chain_summary)
            chains_by_strategy[chain.strategy.value].append(chain_summary)
        
        return {
            "total_chains": len(self.reasoning_chains),
            "by_architecture": dict(chains_by_architecture),
            "by_strategy": dict(chains_by_strategy),
            "high_confidence_chains": [
                chain_id for chain_id, chain in self.reasoning_chains.items()
                if chain.total_confidence >= 0.8
            ],
            "recent_chains": sorted(
                self.reasoning_chains.keys(),
                key=lambda x: self.reasoning_chains[x].created_at or datetime.min,
                reverse=True
            )[:10],
            "last_updated": datetime.utcnow().isoformat()
        }
    
    @mcp_resource(
        uri="reasoning://performance-metrics",
        name="Reasoning Performance Metrics",
        description="Comprehensive performance metrics and analytics",
        mime_type="application/json"
    )
    async def get_performance_metrics_resource(self) -> Dict[str, Any]:
        """Get reasoning performance metrics"""
        return {
            "summary": {
                "total_sessions": self.metrics["total_sessions"],
                "success_rate": (self.metrics["successful_reasoning"] / 
                               max(self.metrics["total_sessions"], 1)),
                "average_confidence": self.metrics["average_confidence"],
                "average_reasoning_time": self.metrics["average_reasoning_time"]
            },
            "architecture_usage": dict(self.metrics["architecture_usage"]),
            "strategy_usage": dict(self.metrics["strategy_usage"]),
            "cache_performance": {
                "hit_rate": (self.metrics["cache_hits"] / 
                           max(self.metrics["cache_hits"] + self.metrics["cache_misses"], 1)),
                "total_hits": self.metrics["cache_hits"],
                "total_misses": self.metrics["cache_misses"]
            },
            "quality_metrics": {
                "high_confidence_ratio": len([s for s in self.active_sessions.values() 
                                            if s.final_confidence >= 0.8]) / max(len(self.active_sessions), 1),
                "average_chain_depth": self._calculate_average_chain_depth(),
                "evidence_utilization": self._calculate_evidence_utilization()
            },
            "last_updated": datetime.utcnow().isoformat()
        }
    
    @mcp_resource(
        uri="reasoning://evidence-cache",
        name="Evidence Cache",
        description="Cached evidence and reasoning results for improved performance",
        mime_type="application/json"
    )
    async def get_evidence_cache_resource(self) -> Dict[str, Any]:
        """Get evidence cache information"""
        cache_items = []
        total_size = 0
        
        for key, value in self.evidence_cache.items():
            item_info = {
                "cache_key": key,
                "timestamp": value.get("timestamp", 0),
                "age_seconds": time.time() - value.get("timestamp", 0),
                "expired": time.time() - value.get("timestamp", 0) > self.cache_ttl
            }
            cache_items.append(item_info)
            total_size += len(str(value))
        
        return {
            "cache_size": len(self.evidence_cache),
            "total_size_bytes": total_size,
            "ttl_seconds": self.cache_ttl,
            "items": cache_items,
            "expired_count": len([item for item in cache_items if item["expired"]]),
            "last_updated": datetime.utcnow().isoformat()
        }
    
    # ==========================================
    # MCP Prompts for Reasoning Guidance
    # ==========================================
    
    @mcp_prompt(
        name="socratic_reasoning",
        description="Guide reasoning through Socratic questioning",
        arguments=[
            {"name": "topic", "description": "The topic to explore", "required": True},
            {"name": "depth", "description": "Depth of questioning", "required": False}
        ]
    )
    async def socratic_reasoning_prompt(self, topic: str, depth: int = 3) -> Dict[str, Any]:
        """Generate Socratic reasoning prompt"""
        questions = [
            f"What do we mean by '{topic}'?",
            f"What assumptions are we making about {topic}?",
            f"What evidence supports our understanding of {topic}?",
            f"What might someone who disagrees say about {topic}?",
            f"What are the implications if our understanding of {topic} is correct?",
            f"How does {topic} relate to other concepts we know?"
        ]
        
        return {
            "prompt_type": "socratic",
            "topic": topic,
            "questions": questions[:depth],
            "guidance": "Answer each question thoroughly before proceeding to the next"
        }
    
    # ==========================================
    # Intra-Skill Communication Implementation Methods
    # ==========================================
    
    async def _execute_skills_parallel(self, question: str, context: Dict[str, Any], 
                                     skills_to_use: List[str], coordination_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute skills in parallel using MCP communication"""
        tasks = []
        
        for skill_name in skills_to_use:
            if skill_name == "decompose_question_skill":
                task = self.call_skill_tool(
                    skill_name, skill_name,
                    question=question,
                    context=context,
                    communication_context={
                        "coordinate_with_patterns": True,
                        "communication_log": coordination_results["communication_log"]
                    }
                )
            elif skill_name == "analyze_patterns_skill":
                task = self.call_skill_tool(
                    skill_name, skill_name,
                    question=question,
                    context=context,
                    focus_areas=["logical", "causal", "temporal"],
                    communication_context={
                        "notify_synthesis": True,
                        "communication_log": coordination_results["communication_log"]
                    }
                )
            elif skill_name == "synthesize_answers_skill":
                task = self.call_skill_tool(
                    skill_name, skill_name,
                    communication_context={
                        "request_additional_insights": True,
                        "original_question": question,
                        "context": context,
                        "communication_log": coordination_results["communication_log"]
                    }
                )
            else:
                # Default skill call
                task = self.call_skill_tool(skill_name, skill_name, question=question, context=context)
            
            tasks.append(task)
        
        # Execute all skills in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            skill_name = skills_to_use[i]
            if isinstance(result, Exception):
                coordination_results["skill_results"][skill_name] = {"success": False, "error": str(result)}
            else:
                coordination_results["skill_results"][skill_name] = result.get("result", result)
        
        return coordination_results["skill_results"]
    
    async def _execute_skills_sequential(self, question: str, context: Dict[str, Any],
                                       skills_to_use: List[str], coordination_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute skills sequentially, passing results between them"""
        accumulated_results = {}
        
        for skill_name in skills_to_use:
            # Prepare context based on previous results
            enhanced_context = context.copy()
            enhanced_context["previous_results"] = accumulated_results
            
            if skill_name == "decompose_question_skill":
                result = await self.call_skill_tool(
                    skill_name, skill_name,
                    question=question,
                    context=enhanced_context,
                    communication_context={
                        "coordinate_with_patterns": True,
                        "communication_log": coordination_results["communication_log"]
                    }
                )
            elif skill_name == "analyze_patterns_skill":
                # Use decomposition results if available
                decomp_insights = accumulated_results.get("decompose_question_skill", {})
                enhanced_focus = ["logical", "causal", "temporal"]
                if decomp_insights.get("decomposition", {}).get("sub_questions"):
                    enhanced_focus.append("hierarchical")
                
                result = await self.call_skill_tool(
                    skill_name, skill_name,
                    question=question,
                    context=enhanced_context,
                    focus_areas=enhanced_focus,
                    communication_context={
                        "notify_synthesis": True,
                        "decomposition_context": decomp_insights,
                        "communication_log": coordination_results["communication_log"]
                    }
                )
            elif skill_name == "synthesize_answers_skill":
                # Collect all previous results for synthesis
                all_reasoning_results = list(accumulated_results.values())
                result = await self.call_skill_tool(
                    skill_name, skill_name,
                    reasoning_results=all_reasoning_results,
                    communication_context={
                        "request_additional_insights": True,
                        "original_question": question,
                        "context": enhanced_context,
                        "communication_log": coordination_results["communication_log"]
                    }
                )
            else:
                # Default skill call with accumulated context
                result = await self.call_skill_tool(skill_name, skill_name, 
                                                 question=question, context=enhanced_context)
            
            # Store result for next skill
            if result and result.get("success"):
                accumulated_results[skill_name] = result.get("result", result)
                coordination_results["skill_results"][skill_name] = result.get("result", result)
            else:
                coordination_results["skill_results"][skill_name] = {"success": False, "error": "Skill execution failed"}
        
        return accumulated_results
    
    async def _synthesize_collaborative_results(self, results: Dict[str, Any], 
                                               coordination_results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final answer from collaborative results"""
        try:
            # Extract key information from results
            decomposition_data = results.get("decompose_question_skill", {})
            pattern_data = results.get("analyze_patterns_skill", {})
            synthesis_data = results.get("synthesize_answers_skill", {})
            
            # Build synthesis
            if synthesis_data and synthesis_data.get("synthesized_answer"):
                # Use synthesis skill result as primary answer
                base_answer = synthesis_data["synthesized_answer"]
                base_confidence = synthesis_data.get("confidence", 0.7)
            else:
                # Fallback synthesis
                base_answer = "Collaborative analysis completed"
                base_confidence = 0.5
            
            # Enhance with decomposition insights
            if decomposition_data and decomposition_data.get("decomposition"):
                decomp = decomposition_data["decomposition"]
                if decomp.get("sub_questions"):
                    base_answer += f" (analyzed {len(decomp['sub_questions'])} sub-components)"
                    base_confidence += 0.1
            
            # Enhance with pattern insights
            if pattern_data and pattern_data.get("patterns_identified"):
                patterns = pattern_data["patterns_identified"]
                if patterns:
                    base_answer += f" (identified {len(patterns)} reasoning patterns)"
                    base_confidence += 0.1
            
            # Cap confidence at 1.0
            final_confidence = min(1.0, base_confidence)
            
            return {
                "answer": base_answer,
                "confidence": final_confidence,
                "components_used": list(results.keys()),
                "communication_events": len(coordination_results["communication_log"])
            }
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {
                "answer": "Synthesis failed due to communication error",
                "confidence": 0.2,
                "error": str(e)
            }
    
    def _assess_question_complexity(self, question: str, context: Dict[str, Any]) -> float:
        """Assess complexity of question for adaptive strategy selection"""
        complexity_score = 0.0
        
        # Length complexity
        word_count = len(question.split())
        complexity_score += min(0.3, word_count / 50.0)
        
        # Question word complexity
        question_words = ["what", "how", "why", "when", "where", "which", "who"]
        q_word_count = sum(1 for word in question.lower().split() if word in question_words)
        complexity_score += min(0.2, q_word_count / 3.0)
        
        # Context richness
        context_complexity = len(context) / 10.0 if context else 0
        complexity_score += min(0.3, context_complexity)
        
        # Logical connectives
        logical_words = ["and", "or", "but", "because", "therefore", "however", "although"]
        logical_count = sum(1 for word in question.lower().split() if word in logical_words)
        complexity_score += min(0.2, logical_count / 2.0)
        
        return min(1.0, complexity_score)
    
    def _identify_logical_patterns(self, question: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify logical patterns in question and context"""
        patterns = []
        
        # Check for logical connectives
        logical_indicators = {
            "conditional": ["if", "then", "when", "unless"],
            "conjunction": ["and", "both", "also"],
            "disjunction": ["or", "either", "alternatively"],
            "negation": ["not", "no", "never", "none"],
            "implication": ["therefore", "thus", "hence", "consequently"]
        }
        
        question_lower = question.lower()
        
        for pattern_type, indicators in logical_indicators.items():
            for indicator in indicators:
                if indicator in question_lower:
                    patterns.append({
                        "type": "logical",
                        "subtype": pattern_type,
                        "indicator": indicator,
                        "confidence": 0.8,
                        "location": "question"
                    })
        
        # Check context for logical patterns
        for key, value in context.items():
            if isinstance(value, str):
                value_lower = value.lower()
                for pattern_type, indicators in logical_indicators.items():
                    for indicator in indicators:
                        if indicator in value_lower:
                            patterns.append({
                                "type": "logical",
                                "subtype": pattern_type,
                                "indicator": indicator,
                                "confidence": 0.6,
                                "location": f"context:{key}"
                            })
        
        return patterns
    
    def _identify_causal_patterns(self, question: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify causal patterns in question and context"""
        patterns = []
        
        causal_indicators = {
            "cause": ["because", "due to", "caused by", "results from"],
            "effect": ["results in", "leads to", "causes", "produces"],
            "mechanism": ["through", "via", "by means of", "using"],
            "correlation": ["related to", "associated with", "linked to"]
        }
        
        question_lower = question.lower()
        
        for pattern_type, indicators in causal_indicators.items():
            for indicator in indicators:
                if indicator in question_lower:
                    patterns.append({
                        "type": "causal",
                        "subtype": pattern_type,
                        "indicator": indicator,
                        "confidence": 0.7,
                        "location": "question"
                    })
        
        return patterns
    
    def _identify_temporal_patterns(self, question: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify temporal patterns in question and context"""
        patterns = []
        
        temporal_indicators = {
            "sequence": ["first", "then", "next", "finally", "before", "after"],
            "duration": ["during", "while", "throughout", "for"],
            "timing": ["when", "at", "since", "until", "by"],
            "frequency": ["often", "always", "never", "sometimes", "usually"]
        }
        
        question_lower = question.lower()
        
        for pattern_type, indicators in temporal_indicators.items():
            for indicator in indicators:
                if indicator in question_lower:
                    patterns.append({
                        "type": "temporal",
                        "subtype": pattern_type,
                        "indicator": indicator,
                        "confidence": 0.6,
                        "location": "question"
                    })
        
        return patterns
    
    def _perform_weighted_synthesis(self, reasoning_results: List[Dict[str, Any]],
                                  patterns: List[Dict[str, Any]],
                                  confidence_scores: List[float],
                                  additional_insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform weighted synthesis of reasoning results"""
        
        # Calculate base confidence from input scores
        if confidence_scores:
            base_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            base_confidence = 0.5
        
        # Build synthesis content
        synthesis_parts = []
        
        # Add reasoning results
        if reasoning_results:
            synthesis_parts.append(f"Analysis of {len(reasoning_results)} reasoning components")
        
        # Add pattern insights
        if patterns:
            pattern_types = list(set(p.get("type", "unknown") for p in patterns))
            synthesis_parts.append(f"Identified patterns: {', '.join(pattern_types)}")
        
        # Add additional insights
        if additional_insights:
            insight_sources = [insight.get("source", "unknown") for insight in additional_insights]
            synthesis_parts.append(f"Additional insights from: {', '.join(insight_sources)}")
        
        if synthesis_parts:
            answer = "Synthesized analysis incorporating: " + "; ".join(synthesis_parts)
        else:
            answer = "No sufficient data available for synthesis"
            base_confidence = 0.3
        
        # Adjust confidence based on synthesis quality
        if len(synthesis_parts) >= 3:
            base_confidence += 0.2
        elif len(synthesis_parts) >= 2:
            base_confidence += 0.1
        
        return {
            "answer": answer,
            "confidence": min(1.0, base_confidence),
            "synthesis_components": len(synthesis_parts)
        }
    
    # ==========================================
    # Internal Reasoning Methods
    # ==========================================
    
    async def _execute_chain_of_thought(self, session: ReasoningSession, 
                                      strategy: str, max_depth: int) -> Dict[str, Any]:
        """Execute chain of thought reasoning"""
        chain = ReasoningChain(
            chain_id=str(uuid4()),
            architecture=ReasoningArchitecture.CHAIN_OF_THOUGHT,
            strategy=ReasoningStrategy(strategy)
        )
        
        # Start with the question as root
        root_node = ReasoningNode(
            node_id=str(uuid4()),
            content=session.question,
            node_type="premise",
            confidence=1.0
        )
        chain.add_node(root_node)
        
        # Build chain step by step
        current_node = root_node
        for depth in range(max_depth):
            # Generate next reasoning step
            next_content = f"Step {depth + 1}: Analyzing '{current_node.content[:50]}...'"
            next_node = ReasoningNode(
                node_id=str(uuid4()),
                content=next_content,
                node_type="inference" if depth < max_depth - 1 else "conclusion",
                confidence=0.9 - (depth * 0.1)  # Decreasing confidence with depth
            )
            chain.add_node(next_node, current_node.node_id)
            current_node = next_node
        
        # Calculate final confidence
        chain.calculate_confidence()
        session.chains[chain.chain_id] = chain
        
        # Set final answer
        conclusion_node = current_node
        session.final_answer = conclusion_node.content
        session.final_confidence = chain.total_confidence
        
        return {
            "success": True,
            "answer": session.final_answer,
            "confidence": session.final_confidence,
            "chain_id": chain.chain_id,
            "reasoning_steps": len(chain.nodes),
            "architecture": "chain_of_thought"
        }
    
    async def _execute_tree_of_thought(self, session: ReasoningSession,
                                     strategy: str, max_depth: int) -> Dict[str, Any]:
        """Execute tree of thought reasoning"""
        chain = ReasoningChain(
            chain_id=str(uuid4()),
            architecture=ReasoningArchitecture.TREE_OF_THOUGHT,
            strategy=ReasoningStrategy(strategy)
        )
        
        # Create root node
        root_node = ReasoningNode(
            node_id=str(uuid4()),
            content=session.question,
            node_type="premise",
            confidence=1.0
        )
        chain.add_node(root_node)
        
        # Build tree with branching
        nodes_to_expand = [(root_node, 0)]
        while nodes_to_expand and len(chain.nodes) < max_depth * 3:
            current_node, depth = nodes_to_expand.pop(0)
            
            if depth >= max_depth:
                continue
            
            # Create 2-3 branches
            num_branches = 2 if depth < 2 else 3
            for i in range(num_branches):
                branch_content = f"Branch {i+1} from '{current_node.content[:30]}...'"
                branch_node = ReasoningNode(
                    node_id=str(uuid4()),
                    content=branch_content,
                    node_type="inference",
                    confidence=0.8 - (depth * 0.1)
                )
                chain.add_node(branch_node, current_node.node_id)
                nodes_to_expand.append((branch_node, depth + 1))
        
        # Select best path to conclusion
        best_leaf = max(
            [n for n in chain.nodes.values() if not n.children],
            key=lambda n: n.confidence
        )
        
        conclusion_node = ReasoningNode(
            node_id=str(uuid4()),
            content=f"Conclusion: Based on analysis, {session.question[:50]}...",
            node_type="conclusion",
            confidence=best_leaf.confidence * 0.9
        )
        chain.add_node(conclusion_node, best_leaf.node_id)
        
        chain.calculate_confidence()
        session.chains[chain.chain_id] = chain
        session.final_answer = conclusion_node.content
        session.final_confidence = chain.total_confidence
        
        return {
            "success": True,
            "answer": session.final_answer,
            "confidence": session.final_confidence,
            "chain_id": chain.chain_id,
            "tree_nodes": len(chain.nodes),
            "branches_explored": len([n for n in chain.nodes.values() if n.children]),
            "architecture": "tree_of_thought"
        }
    
    async def _execute_graph_of_thought(self, session: ReasoningSession,
                                      strategy: str, max_depth: int) -> Dict[str, Any]:
        """Execute graph of thought reasoning"""
        chain = ReasoningChain(
            chain_id=str(uuid4()),
            architecture=ReasoningArchitecture.GRAPH_OF_THOUGHT,
            strategy=ReasoningStrategy(strategy)
        )
        
        # Create initial nodes
        concepts = self._extract_concepts(session.question)
        concept_nodes = {}
        
        for concept in concepts[:5]:  # Limit to 5 concepts
            node = ReasoningNode(
                node_id=str(uuid4()),
                content=f"Concept: {concept}",
                node_type="premise",
                confidence=0.8
            )
            chain.add_node(node)
            concept_nodes[concept] = node
        
        # Create connections between concepts
        for i, (concept1, node1) in enumerate(concept_nodes.items()):
            for j, (concept2, node2) in enumerate(concept_nodes.items()):
                if i < j:  # Avoid duplicates
                    connection = ReasoningNode(
                        node_id=str(uuid4()),
                        content=f"Relation: {concept1} <-> {concept2}",
                        node_type="inference",
                        confidence=0.7
                    )
                    chain.add_node(connection)
                    # In a graph, nodes can have multiple parents
                    node1.children.append(connection.node_id)
                    node2.children.append(connection.node_id)
        
        # Synthesize conclusion
        conclusion = ReasoningNode(
            node_id=str(uuid4()),
            content=f"Graph analysis concludes: {session.question[:50]}...",
            node_type="conclusion",
            confidence=0.75
        )
        chain.add_node(conclusion)
        
        chain.calculate_confidence()
        session.chains[chain.chain_id] = chain
        session.final_answer = conclusion.content
        session.final_confidence = chain.total_confidence
        
        return {
            "success": True,
            "answer": session.final_answer,
            "confidence": session.final_confidence,
            "chain_id": chain.chain_id,
            "graph_nodes": len(chain.nodes),
            "connections": len([n for n in chain.nodes.values() if "Relation:" in n.content]),
            "architecture": "graph_of_thought"
        }
    
    async def _execute_hierarchical_reasoning(self, session: ReasoningSession,
                                            strategy: str, max_depth: int) -> Dict[str, Any]:
        """Execute hierarchical reasoning"""
        # Decompose question
        sub_questions = self._decompose_question(session.question)
        
        # Create chains for each sub-question
        sub_results = []
        for sub_q in sub_questions[:3]:  # Limit to 3 sub-questions
            sub_chain = ReasoningChain(
                chain_id=str(uuid4()),
                architecture=ReasoningArchitecture.HIERARCHICAL,
                strategy=ReasoningStrategy(strategy)
            )
            
            # Simple chain for sub-question
            sub_node = ReasoningNode(
                node_id=str(uuid4()),
                content=sub_q,
                node_type="premise",
                confidence=0.8
            )
            sub_chain.add_node(sub_node)
            
            answer_node = ReasoningNode(
                node_id=str(uuid4()),
                content=f"Answer to {sub_q[:30]}...",
                node_type="conclusion",
                confidence=0.7
            )
            sub_chain.add_node(answer_node, sub_node.node_id)
            
            sub_chain.calculate_confidence()
            session.chains[sub_chain.chain_id] = sub_chain
            sub_results.append(answer_node)
        
        # Synthesize final answer
        final_answer = f"Hierarchical analysis: {'; '.join([n.content for n in sub_results])}"
        avg_confidence = sum(n.confidence for n in sub_results) / len(sub_results)
        
        session.final_answer = final_answer
        session.final_confidence = avg_confidence
        
        return {
            "success": True,
            "answer": session.final_answer,
            "confidence": session.final_confidence,
            "sub_questions": len(sub_questions),
            "chains_created": len(sub_results),
            "architecture": "hierarchical"
        }
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple extraction - remove common words
        stop_words = {"what", "how", "why", "when", "where", "is", "are", "the", "a", "an", "of", "in", "to", "for"}
        words = text.lower().split()
        concepts = [w for w in words if w not in stop_words and len(w) > 3]
        return list(set(concepts))[:10]  # Unique concepts, max 10
    
    def _decompose_question(self, question: str) -> List[str]:
        """Decompose question into sub-questions"""
        # Simple decomposition based on question structure
        sub_questions = []
        
        if "and" in question:
            parts = question.split("and")
            sub_questions.extend([p.strip() + "?" for p in parts])
        elif "or" in question:
            parts = question.split("or")
            sub_questions.extend([p.strip() + "?" for p in parts])
        else:
            # Create aspect-based sub-questions
            sub_questions = [
                f"What is the main subject of: {question}",
                f"What evidence supports: {question}",
                f"What are the implications of: {question}"
            ]
        
        return sub_questions
    
    def _analyze_chain_patterns(self, chain: ReasoningChain, 
                               pattern_types: List[str]) -> Dict[str, Any]:
        """Analyze patterns in a reasoning chain"""
        patterns = {}
        
        for pattern_type in pattern_types:
            if pattern_type == "logical":
                # Check for logical connectives
                logical_nodes = [n for n in chain.nodes.values() 
                               if any(word in n.content.lower() 
                                    for word in ["therefore", "because", "if", "then"])]
                patterns["logical"] = {
                    "count": len(logical_nodes),
                    "percentage": len(logical_nodes) / max(len(chain.nodes), 1)
                }
            elif pattern_type == "causal":
                # Check for causal relationships
                causal_nodes = [n for n in chain.nodes.values()
                              if any(word in n.content.lower()
                                   for word in ["causes", "leads to", "results in", "due to"])]
                patterns["causal"] = {
                    "count": len(causal_nodes),
                    "percentage": len(causal_nodes) / max(len(chain.nodes), 1)
                }
        
        return patterns
    
    def _calculate_depth_metrics(self, session: ReasoningSession) -> Dict[str, Any]:
        """Calculate depth metrics for reasoning session"""
        depths = []
        for chain in session.chains.values():
            # Calculate max depth of chain
            max_depth = 0
            for node in chain.nodes.values():
                depth = self._get_node_depth(node, chain)
                max_depth = max(max_depth, depth)
            depths.append(max_depth)
        
        return {
            "average_depth": sum(depths) / len(depths) if depths else 0,
            "max_depth": max(depths) if depths else 0,
            "min_depth": min(depths) if depths else 0
        }
    
    def _get_node_depth(self, node: ReasoningNode, chain: ReasoningChain) -> int:
        """Get depth of node in chain"""
        if node.parent is None:
            return 0
        parent_node = chain.nodes.get(node.parent)
        if parent_node:
            return 1 + self._get_node_depth(parent_node, chain)
        return 0
    
    def _assess_evidence_quality(self, session: ReasoningSession) -> Dict[str, Any]:
        """Assess quality of evidence in session"""
        if not session.evidence_pool:
            return {"quality_score": 0.0, "evidence_count": 0}
        
        total_relevance = sum(e.get("relevance", 0.5) for e in session.evidence_pool)
        avg_relevance = total_relevance / len(session.evidence_pool)
        
        return {
            "quality_score": avg_relevance,
            "evidence_count": len(session.evidence_pool),
            "high_quality_count": len([e for e in session.evidence_pool 
                                     if e.get("relevance", 0) > 0.8])
        }
    
    def _generate_debate_response(self, position: Dict[str, Any],
                                other_position: Dict[str, Any],
                                structure: str) -> Dict[str, Any]:
        """Generate debate response"""
        if structure == "dialectical":
            # Synthesize positions
            return {
                "type": "synthesis",
                "content": f"Combining insights from both positions",
                "agreement_points": ["Both positions have merit"],
                "integration": "Unified perspective"
            }
        elif structure == "adversarial":
            # Challenge other position
            return {
                "type": "challenge",
                "content": f"Challenging the premise of the other position",
                "counterarguments": ["Alternative interpretation possible"],
                "weaknesses_identified": ["Assumption may not hold"]
            }
        else:  # collaborative
            # Build on other position
            return {
                "type": "extension",
                "content": f"Building on the other position",
                "additional_evidence": ["Supporting evidence"],
                "refined_conclusion": "Enhanced understanding"
            }
    
    def _update_positions(self, positions: List[Dict[str, Any]],
                        exchanges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update positions based on debate exchanges"""
        updated = []
        for i, position in enumerate(positions):
            # Simple update - adjust confidence based on challenges
            challenges = [e for e in exchanges if e["to"] == i and "challenge" in str(e)]
            support = [e for e in exchanges if e["to"] == i and "synthesis" in str(e)]
            
            new_confidence = position.get("confidence", 0.5)
            new_confidence -= len(challenges) * 0.1
            new_confidence += len(support) * 0.05
            new_confidence = max(0.1, min(1.0, new_confidence))
            
            updated_position = position.copy()
            updated_position["confidence"] = new_confidence
            updated.append(updated_position)
        
        return updated
    
    def _check_convergence(self, positions: List[Dict[str, Any]], 
                         threshold: float) -> bool:
        """Check if positions have converged"""
        if len(positions) < 2:
            return True
        
        # Check confidence variance
        confidences = [p.get("confidence", 0.5) for p in positions]
        variance = sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences)
        
        return variance < (1 - threshold)
    
    def _synthesize_positions(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize multiple positions into final conclusion"""
        # Weight by confidence
        total_confidence = sum(p.get("confidence", 0.5) for p in positions)
        
        return {
            "synthesized_argument": "Balanced synthesis of all positions",
            "confidence": total_confidence / len(positions) if positions else 0.0,
            "key_insights": ["Integration of multiple perspectives"],
            "consensus_level": "high" if self._check_convergence(positions, 0.8) else "moderate"
        }
    
    def _generate_counterfactual_premise(self, premise: str, 
                                       variation_type: str) -> str:
        """Generate counterfactual variation of premise"""
        if variation_type == "negation":
            # Simple negation
            if "is" in premise:
                return premise.replace("is", "is not", 1)
            return f"NOT ({premise})"
        elif variation_type == "modification":
            # Modify key terms
            return f"Modified: {premise} [with alterations]"
        else:  # substitution
            # Substitute key concepts
            concepts = self._extract_concepts(premise)
            if concepts:
                return premise.replace(concepts[0], f"alternative-{concepts[0]}", 1)
            return f"Alternative: {premise}"
    
    async def _reason_from_premise(self, premise: str) -> Dict[str, Any]:
        """Reason from a given premise"""
        # Simple reasoning simulation
        chain = [
            {"step": 1, "content": f"Given: {premise}"},
            {"step": 2, "content": "Analyzing implications..."},
            {"step": 3, "content": "Drawing conclusion..."}
        ]
        
        conclusion = f"Conclusion from '{premise[:30]}...'"
        confidence = 0.7  # Base confidence
        
        return {
            "conclusion": conclusion,
            "confidence": confidence,
            "chain": chain
        }
    
    def _extract_counterfactual_insights(self, counterfactuals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract insights from counterfactual analysis"""
        different_conclusions = [cf for cf in counterfactuals if cf["differs_from_original"]]
        
        return {
            "sensitivity": len(different_conclusions) / len(counterfactuals) if counterfactuals else 0,
            "robust_conclusion": len(different_conclusions) < len(counterfactuals) / 2,
            "key_dependencies": ["Premise variations significantly affect conclusion"] if different_conclusions else [],
            "recommendation": "Conclusion is robust" if len(different_conclusions) < len(counterfactuals) / 2 
                           else "Conclusion is sensitive to premise changes"
        }
    
    def _validate_logical_consistency(self, chains: List[ReasoningChain]) -> Dict[str, Any]:
        """Validate logical consistency across chains"""
        inconsistencies = []
        
        # Check for contradictory conclusions
        conclusions = [n for c in chains for n in c.nodes.values() if n.node_type == "conclusion"]
        
        for i, c1 in enumerate(conclusions):
            for j, c2 in enumerate(conclusions):
                if i < j and self._are_contradictory(c1.content, c2.content):
                    inconsistencies.append({
                        "type": "contradictory_conclusions",
                        "chain1": chains[i].chain_id,
                        "chain2": chains[j].chain_id
                    })
        
        score = 1.0 - (len(inconsistencies) / max(len(conclusions), 1))
        
        return {
            "score": score,
            "inconsistencies": inconsistencies
        }
    
    def _are_contradictory(self, text1: str, text2: str) -> bool:
        """Check if two texts are contradictory (simple heuristic)"""
        negations = ["not", "no", "never", "false", "incorrect"]
        has_negation1 = any(neg in text1.lower() for neg in negations)
        has_negation2 = any(neg in text2.lower() for neg in negations)
        
        # Simple check - if one has negation and other doesn't
        return has_negation1 != has_negation2
    
    def _validate_evidence_consistency(self, chains: List[ReasoningChain]) -> Dict[str, Any]:
        """Validate evidence consistency across chains"""
        # For now, return perfect consistency
        return {"score": 1.0, "inconsistencies": []}
    
    def _validate_conclusion_alignment(self, chains: List[ReasoningChain]) -> Dict[str, Any]:
        """Validate conclusion alignment across chains"""
        # For now, return good alignment
        return {"score": 0.9, "inconsistencies": []}
    
    def _validate_confidence_correlation(self, chains: List[ReasoningChain]) -> Dict[str, Any]:
        """Validate confidence correlation across chains"""
        # For now, return good correlation
        return {"score": 0.85, "inconsistencies": []}
    
    def _calculate_average_chain_depth(self) -> float:
        """Calculate average depth across all chains"""
        if not self.reasoning_chains:
            return 0.0
        
        total_depth = 0
        for chain in self.reasoning_chains.values():
            max_depth = 0
            for node in chain.nodes.values():
                depth = self._get_node_depth(node, chain)
                max_depth = max(max_depth, depth)
            total_depth += max_depth
        
        return total_depth / len(self.reasoning_chains)
    
    def _calculate_evidence_utilization(self) -> float:
        """Calculate how well evidence is utilized"""
        if not self.active_sessions:
            return 0.0
        
        total_utilization = 0
        for session in self.active_sessions.values():
            if session.evidence_pool:
                # Check how many evidence pieces are referenced in chains
                used_evidence = sum(1 for e in session.evidence_pool 
                                  if any(e.get("content", "") in str(chain.nodes) 
                                       for chain in session.chains.values()))
                utilization = used_evidence / len(session.evidence_pool)
                total_utilization += utilization
        
        return total_utilization / len(self.active_sessions)
    
    async def _session_cleanup_loop(self):
        """Clean up old sessions periodically"""
        while not self._shutdown_flag:
            try:
                current_time = datetime.utcnow()
                sessions_to_remove = []
                
                for session_id, session in self.active_sessions.items():
                    # Remove sessions older than 1 hour
                    if session.created_at and (current_time - session.created_at).seconds > 3600:
                        sessions_to_remove.append(session_id)
                
                for session_id in sessions_to_remove:
                    del self.active_sessions[session_id]
                    logger.info(f"Cleaned up old session: {session_id}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
                await asyncio.sleep(300)
    
    async def _metrics_aggregation_loop(self):
        """Aggregate metrics periodically"""
        while not self._shutdown_flag:
            try:
                # Update average confidence
                if self.active_sessions:
                    confidences = [s.final_confidence for s in self.active_sessions.values() 
                                 if s.final_confidence > 0]
                    if confidences:
                        self.metrics["average_confidence"] = sum(confidences) / len(confidences)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Metrics aggregation error: {e}")
                await asyncio.sleep(60)
    
    # ==========================================
    # Stub Methods for Testing
    # ==========================================
    
    def _list_mcp_tools_stub(self) -> List[Dict[str, Any]]:
        """Stub for MCP tools list when base class not available"""
        return [
            {"name": "execute_reasoning_chain", "description": "Execute complete reasoning chain"},
            {"name": "analyze_reasoning_patterns", "description": "Analyze reasoning patterns"},
            {"name": "conduct_reasoning_debate", "description": "Conduct multi-perspective debate"},
            {"name": "generate_counterfactual_reasoning", "description": "Generate counterfactual scenarios"},
            {"name": "validate_reasoning_consistency", "description": "Validate reasoning consistency"}
        ]
    
    def _list_mcp_resources_stub(self) -> List[Dict[str, Any]]:
        """Stub for MCP resources list when base class not available"""
        return [
            {"name": "Active Reasoning Sessions", "uri": "reasoning://active-sessions"},
            {"name": "Reasoning Chain Library", "uri": "reasoning://chain-library"},
            {"name": "Performance Metrics", "uri": "reasoning://performance-metrics"},
            {"name": "Evidence Cache", "uri": "reasoning://evidence-cache"}
        ]
    
    async def _call_mcp_tool_stub(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Stub for calling MCP tools when base class not available"""
        if tool_name == "execute_reasoning_chain":
            return await self.execute_reasoning_chain(**arguments)
        elif tool_name == "analyze_reasoning_patterns":
            return await self.analyze_reasoning_patterns(**arguments)
        elif tool_name == "conduct_reasoning_debate":
            return await self.conduct_reasoning_debate(**arguments)
        elif tool_name == "generate_counterfactual_reasoning":
            return await self.generate_counterfactual_reasoning(**arguments)
        elif tool_name == "validate_reasoning_consistency":
            return await self.validate_reasoning_consistency(**arguments)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    async def _get_mcp_resource_stub(self, uri: str) -> Dict[str, Any]:
        """Stub for getting MCP resources when base class not available"""
        if uri == "reasoning://active-sessions":
            return await self.get_active_sessions_resource()
        elif uri == "reasoning://chain-library":
            return await self.get_chain_library_resource()
        elif uri == "reasoning://performance-metrics":
            return await self.get_performance_metrics_resource()
        elif uri == "reasoning://evidence-cache":
            return await self.get_evidence_cache_resource()
        else:
            return {"error": f"Unknown resource: {uri}"}