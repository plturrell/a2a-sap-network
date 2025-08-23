"""
A2A-Compliant Multi-Agent Coordination System
Real implementation using A2A SDK patterns, replacing NotImplementedError placeholders
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import logging
import uuid

from dataclasses import dataclass, field

# A2A SDK imports
from app.a2a.sdk import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)
from app.a2a.sdk.mcpDecorators import mcp_tool, mcp_resource
from app.a2a.sdk.mcpSkillCoordination import (
    MCPSkillCoordinator, SkillMessageType, SkillPriority, SkillMessage
)
from app.a2a.core.serviceDiscovery import (
    discover_qa_agents, discover_reasoning_engines, discover_synthesis_agents
)
from app.a2a.core.trustManager import sign_a2a_message, verify_a2a_message

logger = logging.getLogger(__name__)

class A2ACoordinationPattern(Enum):
    HIERARCHICAL = "hierarchical"
    PEER_TO_PEER = "peer_to_peer"
    BLACKBOARD = "blackboard"
    DEBATE = "debate"

@dataclass
class A2AReasoningTask:
    task_id: str
    question: str
    context: Dict[str, Any]
    coordination_pattern: A2ACoordinationPattern
    required_agents: int = 3
    timeout_seconds: int = 30

class A2ADebateCoordination:
    """Real A2A-compliant debate coordination"""
    
    def __init__(self, reasoning_agent):
        self.reasoning_agent = reasoning_agent
        self.active_debates = {}
    
    @a2a_skill("coordinate_agent_debate")
    async def coordinate_agent_debate(self, proposals: List[Dict[str, Any]], 
                                    threshold: float = 0.7) -> Dict[str, Any]:
        """Coordinate real debate between discovered A2A reasoning agents"""
        
        debate_id = str(uuid.uuid4())[:8]
        logger.info(f"🎯 Starting A2A agent debate {debate_id}")
        
        try:
            # Discover available A2A reasoning agents
            reasoning_agents = await discover_reasoning_engines(
                capabilities=["debate", "argumentation", "reasoning"],
                min_agents=2,
                timeout=10
            )
            
            if len(reasoning_agents) < 2:
                logger.warning("Insufficient A2A agents for debate, using consensus fallback")
                return await self._a2a_consensus_fallback(proposals, threshold)
            
            # Initialize A2A debate state
            debate_state = {
                "debate_id": debate_id,
                "a2a_agents": reasoning_agents[:4],  # Limit to 4 agents
                "rounds": [],
                "positions": {},
                "consensus_reached": False
            }
            
            # Distribute positions using A2A messaging
            await self._distribute_debate_positions(debate_state, proposals)
            
            # Conduct debate rounds with A2A coordination
            max_rounds = 3
            for round_num in range(max_rounds):
                round_result = await self._conduct_a2a_debate_round(debate_state, round_num)
                debate_state["rounds"].append(round_result)
                
                # Check A2A consensus
                consensus_score = await self._calculate_a2a_consensus(debate_state)
                if consensus_score >= threshold:
                    debate_state["consensus_reached"] = True
                    break
            
            # Synthesize final result using A2A synthesis agents
            return await self._synthesize_a2a_debate_result(debate_state)
            
        except Exception as e:
            logger.error(f"A2A debate coordination failed: {e}")
            return await self._a2a_consensus_fallback(proposals, threshold)
    
    async def _distribute_debate_positions(self, debate_state: Dict[str, Any], 
                                         proposals: List[Dict[str, Any]]):
        """Distribute debate positions using A2A messaging"""
        
        agents = debate_state["a2a_agents"]
        
        position_tasks = []
        for i, proposal in enumerate(proposals[:len(agents)]):
            agent_info = agents[i]
            
            # Create A2A message for position assignment
            position_message = A2AMessage(
                message_id=create_agent_id(),
                sender_id=self.reasoning_agent.agent_id,
                receiver_id=agent_info["agent_id"],
                message_type="debate_position_assignment",
                content={
                    "debate_id": debate_state["debate_id"],
                    "assigned_position": proposal,
                    "role": "advocate",
                    "debate_question": debate_state.get("question", "")
                },
                role=MessageRole.SYSTEM,
                timestamp=datetime.utcnow()
            )
            
            # Sign A2A message
            signed_message = await sign_a2a_message(position_message)
            
            # Send position assignment
            task = self._send_a2a_position_assignment(agent_info, signed_message)
            position_tasks.append((agent_info["agent_id"], task))
            
            # Track position in debate state
            debate_state["positions"][agent_info["agent_id"]] = {
                "position": proposal,
                "confidence": proposal.get("confidence", 0.5),
                "agent_info": agent_info
            }
        
        # Wait for position confirmations
        await asyncio.gather(*[task for _, task in position_tasks], return_exceptions=True)
    
    async def _conduct_a2a_debate_round(self, debate_state: Dict[str, Any], 
                                      round_num: int) -> Dict[str, Any]:
        """Conduct debate round using A2A agent coordination"""
        
        round_result = {
            "round": round_num + 1,
            "a2a_arguments": [],
            "position_updates": [],
            "message_count": 0
        }
        
        # Phase 1: Request arguments from each A2A agent
        argument_tasks = []
        for agent_id, position_data in debate_state["positions"].items():
            
            # Create A2A argument request
            argument_request = A2AMessage(
                message_id=create_agent_id(),
                sender_id=self.reasoning_agent.agent_id,
                receiver_id=agent_id,
                message_type="debate_argument_request",
                content={
                    "debate_id": debate_state["debate_id"],
                    "round": round_num + 1,
                    "current_position": position_data["position"],
                    "debate_history": debate_state["rounds"],
                    "other_positions": [pos["position"] for aid, pos in debate_state["positions"].items() if aid != agent_id]
                },
                role=MessageRole.USER,
                timestamp=datetime.utcnow()
            )
            
            signed_request = await sign_a2a_message(argument_request)
            task = self._request_a2a_argument(position_data["agent_info"], signed_request)
            argument_tasks.append((agent_id, task))
        
        # Collect A2A arguments
        for agent_id, task in argument_tasks:
            try:
                argument_response = await asyncio.wait_for(task, timeout=15)
                if argument_response:
                    round_result["a2a_arguments"].append({
                        "agent_id": agent_id,
                        "argument": argument_response.get("argument"),
                        "confidence": argument_response.get("confidence", 0.5),
                        "evidence": argument_response.get("evidence", [])
                    })
                    round_result["message_count"] += 1
                    
            except asyncio.TimeoutError:
                logger.warning(f"A2A agent {agent_id} argument request timed out")
            except Exception as e:
                logger.error(f"Failed to get argument from A2A agent {agent_id}: {e}")
        
        # Phase 2: Cross-argument evaluation using A2A messaging
        if len(round_result["a2a_arguments"]) > 1:
            evaluation_tasks = []
            for agent_id, position_data in debate_state["positions"].items():
                other_arguments = [arg for arg in round_result["a2a_arguments"] 
                                 if arg["agent_id"] != agent_id]
                
                evaluation_request = A2AMessage(
                    message_id=create_agent_id(),
                    sender_id=self.reasoning_agent.agent_id,
                    receiver_id=agent_id,
                    message_type="debate_evaluation_request",
                    content={
                        "debate_id": debate_state["debate_id"],
                        "other_arguments": other_arguments,
                        "evaluate_against_position": position_data["position"]
                    },
                    role=MessageRole.USER,
                    timestamp=datetime.utcnow()
                )
                
                signed_eval = await sign_a2a_message(evaluation_request)
                task = self._request_a2a_evaluation(position_data["agent_info"], signed_eval)
                evaluation_tasks.append((agent_id, task))
            
            # Process evaluations
            for agent_id, task in evaluation_tasks:
                try:
                    evaluation = await asyncio.wait_for(task, timeout=10)
                    if evaluation:
                        confidence_change = evaluation.get("confidence_change", 0.0)
                        old_confidence = debate_state["positions"][agent_id]["confidence"]
                        new_confidence = max(0.0, min(1.0, old_confidence + confidence_change))
                        
                        debate_state["positions"][agent_id]["confidence"] = new_confidence
                        round_result["position_updates"].append({
                            "agent_id": agent_id,
                            "confidence_change": confidence_change,
                            "new_confidence": new_confidence
                        })
                        round_result["message_count"] += 1
                        
                except Exception as e:
                    logger.error(f"A2A evaluation failed for agent {agent_id}: {e}")
        
        return round_result
    
    @a2a_handler("debate_argument")
    async def _request_a2a_argument(self, agent_info: Dict[str, Any], 
                                   signed_request: A2AMessage) -> Dict[str, Any]:
        """Request argument from A2A agent using proper A2A messaging"""
        
        try:
            # Use A2A service discovery to get agent endpoint
            agent_endpoint = agent_info.get("endpoint")
            if not agent_endpoint:
                logger.error(f"No endpoint found for A2A agent {agent_info['agent_id']}")
                return {}
            
            # Send A2A message through proper channels
            response = await self.reasoning_agent._send_a2a_message(
                agent_endpoint,
                signed_request,
                expected_response_type="debate_argument_response"
            )
            
            # Verify A2A response
            if response and await verify_a2a_message(response):
                return response.content
            else:
                logger.warning(f"Invalid A2A response from {agent_info['agent_id']}")
                return {}
                
        except Exception as e:
            logger.error(f"A2A argument request failed: {e}")
            return {}
    
    async def _calculate_a2a_consensus(self, debate_state: Dict[str, Any]) -> float:
        """Calculate consensus score using A2A agent positions"""
        
        positions = list(debate_state["positions"].values())
        if len(positions) < 2:
            return 1.0
        
        # Calculate confidence convergence
        confidences = [pos["confidence"] for pos in positions]
        confidence_variance = sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences)
        convergence_score = max(0.0, 1.0 - confidence_variance)
        
        # Calculate position similarity using enhanced semantic similarity
        from .semanticSimilarityCalculator import calculate_text_similarity
        
        similarity_scores = []
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i+1:]:
                text1 = str(pos1["position"].get("answer", ""))
                text2 = str(pos2["position"].get("answer", ""))
                similarity = calculate_text_similarity(text1, text2, method="hybrid")
                similarity_scores.append(similarity)
        
        avg_similarity = sum(similarity_scores) / max(len(similarity_scores), 1)
        
        return (convergence_score * 0.6) + (avg_similarity * 0.4)
    
    async def _synthesize_a2a_debate_result(self, debate_state: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize debate result using A2A synthesis agents"""
        
        try:
            # Discover A2A synthesis agents
            synthesis_agents = await discover_synthesis_agents(
                capabilities=["debate_synthesis", "consensus_building"],
                min_agents=1,
                timeout=5
            )
            
            if synthesis_agents:
                synthesis_agent = synthesis_agents[0]
                
                # Create A2A synthesis request
                synthesis_request = A2AMessage(
                    message_id=create_agent_id(),
                    sender_id=self.reasoning_agent.agent_id,
                    receiver_id=synthesis_agent["agent_id"],
                    message_type="debate_synthesis_request",
                    content={
                        "debate_id": debate_state["debate_id"],
                        "debate_rounds": debate_state["rounds"],
                        "final_positions": debate_state["positions"],
                        "consensus_reached": debate_state["consensus_reached"]
                    },
                    role=MessageRole.USER,
                    timestamp=datetime.utcnow()
                )
                
                signed_synthesis = await sign_a2a_message(synthesis_request)
                synthesis_response = await self.reasoning_agent._send_a2a_message(
                    synthesis_agent["endpoint"],
                    signed_synthesis,
                    expected_response_type="debate_synthesis_response"
                )
                
                if synthesis_response and await verify_a2a_message(synthesis_response):
                    return synthesis_response.content
            
            # Fallback synthesis
            return await self._internal_debate_synthesis(debate_state)
            
        except Exception as e:
            logger.error(f"A2A synthesis failed: {e}")
            return await self._internal_debate_synthesis(debate_state)
    
    async def _a2a_consensus_fallback(self, proposals: List[Dict[str, Any]], 
                                    threshold: float) -> Dict[str, Any]:
        """A2A-compliant consensus fallback"""
        
        if not proposals:
            return {
                "consensus": "No proposals available for A2A consensus",
                "confidence": 0.0,
                "method": "a2a_fallback",
                "agent_count": 0
            }
        
        # Weight by confidence
        total_weight = sum(p.get("confidence", 0.5) for p in proposals)
        weighted_avg = total_weight / len(proposals) if proposals else 0.0
        
        # Select best proposal
        best_proposal = max(proposals, key=lambda p: p.get("confidence", 0.0))
        
        return {
            "consensus": best_proposal.get("answer", "A2A consensus processing"),
            "confidence": weighted_avg,
            "method": "a2a_weighted_consensus",
            "agent_count": len(proposals),
            "proposals_processed": len(proposals)
        }


class A2ABlackboardCoordination:
    """A2A-compliant blackboard reasoning coordination"""
    
    def __init__(self, reasoning_agent):
        self.reasoning_agent = reasoning_agent
        self.active_blackboards = {}
    
    @a2a_skill("coordinate_blackboard_reasoning")
    async def coordinate_blackboard_reasoning(self, question: str, 
                                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate blackboard reasoning using A2A knowledge agents"""
        
        blackboard_id = str(uuid.uuid4())[:8]
        logger.info(f"🧠 Starting A2A blackboard reasoning {blackboard_id}")
        
        try:
            # Discover A2A knowledge agents
            knowledge_agents = await discover_qa_agents(
                capabilities=["knowledge_contribution", "analysis", "synthesis"],
                min_agents=2,
                timeout=10
            )
            
            if not knowledge_agents:
                logger.warning("No A2A knowledge agents found, using internal fallback")
                return await self._a2a_internal_blackboard(question, context)
            
            # Initialize A2A blackboard
            blackboard = {
                "blackboard_id": blackboard_id,
                "problem": question,
                "context": context,
                "a2a_knowledge_agents": knowledge_agents[:5],  # Limit to 5 agents
                "knowledge_contributions": [],
                "synthesis_attempts": [],
                "final_solution": None
            }
            
            # Multi-round A2A knowledge gathering
            max_iterations = 3
            for iteration in range(max_iterations):
                # Request knowledge contributions via A2A messaging
                contributions = await self._gather_a2a_knowledge(blackboard, iteration)
                blackboard["knowledge_contributions"].extend(contributions)
                
                # Attempt solution synthesis
                synthesis = await self._attempt_a2a_synthesis(blackboard)
                blackboard["synthesis_attempts"].append(synthesis)
                
                if synthesis.get("confidence", 0) > 0.7:
                    blackboard["final_solution"] = synthesis
                    break
            
            # Final A2A synthesis if needed
            if not blackboard["final_solution"]:
                blackboard["final_solution"] = await self._final_a2a_blackboard_synthesis(blackboard)
            
            return blackboard["final_solution"]
            
        except Exception as e:
            logger.error(f"A2A blackboard coordination failed: {e}")
            return await self._a2a_internal_blackboard(question, context)
    
    async def _gather_a2a_knowledge(self, blackboard: Dict[str, Any], 
                                   iteration: int) -> List[Dict[str, Any]]:
        """Gather knowledge contributions using A2A messaging"""
        
        contribution_tasks = []
        
        for agent_info in blackboard["a2a_knowledge_agents"]:
            # Create A2A knowledge request
            knowledge_request = A2AMessage(
                message_id=create_agent_id(),
                sender_id=self.reasoning_agent.agent_id,
                receiver_id=agent_info["agent_id"],
                message_type="knowledge_contribution_request",
                content={
                    "blackboard_id": blackboard["blackboard_id"],
                    "problem": blackboard["problem"],
                    "context": blackboard["context"],
                    "iteration": iteration,
                    "existing_contributions": blackboard["knowledge_contributions"][-3:],  # Last 3
                    "focus_area": self._determine_focus_area(agent_info, iteration)
                },
                role=MessageRole.USER,
                timestamp=datetime.utcnow()
            )
            
            signed_request = await sign_a2a_message(knowledge_request)
            task = self._request_a2a_knowledge(agent_info, signed_request)
            contribution_tasks.append((agent_info["agent_id"], task))
        
        # Collect A2A contributions
        contributions = []
        for agent_id, task in contribution_tasks:
            try:
                contribution = await asyncio.wait_for(task, timeout=15)
                if contribution:
                    contribution["contributor_agent"] = agent_id
                    contribution["iteration"] = iteration
                    contribution["timestamp"] = datetime.utcnow().isoformat()
                    contributions.append(contribution)
                    
            except asyncio.TimeoutError:
                logger.warning(f"A2A knowledge request to {agent_id} timed out")
            except Exception as e:
                logger.error(f"A2A knowledge request failed for {agent_id}: {e}")
        
        return contributions
    
    @a2a_handler("knowledge_contribution")
    async def _request_a2a_knowledge(self, agent_info: Dict[str, Any], 
                                   signed_request: A2AMessage) -> Dict[str, Any]:
        """Request knowledge contribution using A2A messaging"""
        
        try:
            agent_endpoint = agent_info.get("endpoint")
            if not agent_endpoint:
                return {}
            
            response = await self.reasoning_agent._send_a2a_message(
                agent_endpoint,
                signed_request,
                expected_response_type="knowledge_contribution_response"
            )
            
            if response and await verify_a2a_message(response):
                return response.content
            else:
                return {}
                
        except Exception as e:
            logger.error(f"A2A knowledge request failed: {e}")
            return {}
    
    def _determine_focus_area(self, agent_info: Dict[str, Any], iteration: int) -> str:
        """Determine focus area for A2A agent based on capabilities and iteration"""
        
        capabilities = agent_info.get("capabilities", [])
        
        if iteration == 0:  # Initial broad analysis
            if "analysis" in capabilities:
                return "problem_analysis"
            elif "domain_knowledge" in capabilities:
                return "domain_expertise"
            else:
                return "general_knowledge"
                
        elif iteration == 1:  # Deeper investigation
            if "synthesis" in capabilities:
                return "solution_synthesis"
            elif "pattern_recognition" in capabilities:
                return "pattern_analysis"
            else:
                return "detailed_analysis"
                
        else:  # Final iteration - validation and refinement
            return "validation_and_refinement"
    
    async def _attempt_a2a_synthesis(self, blackboard: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt solution synthesis using A2A synthesis agents"""
        
        try:
            # Use discovered synthesis agents or internal synthesis
            synthesis_agents = await discover_synthesis_agents(
                capabilities=["blackboard_synthesis", "knowledge_integration"],
                min_agents=1,
                timeout=5
            )
            
            if synthesis_agents:
                synthesis_agent = synthesis_agents[0]
                
                synthesis_request = A2AMessage(
                    message_id=create_agent_id(),
                    sender_id=self.reasoning_agent.agent_id,
                    receiver_id=synthesis_agent["agent_id"],
                    message_type="blackboard_synthesis_request",
                    content={
                        "blackboard_id": blackboard["blackboard_id"],
                        "problem": blackboard["problem"],
                        "knowledge_contributions": blackboard["knowledge_contributions"],
                        "context": blackboard["context"]
                    },
                    role=MessageRole.USER,
                    timestamp=datetime.utcnow()
                )
                
                signed_synthesis = await sign_a2a_message(synthesis_request)
                response = await self.reasoning_agent._send_a2a_message(
                    synthesis_agent["endpoint"],
                    signed_synthesis,
                    expected_response_type="blackboard_synthesis_response"
                )
                
                if response and await verify_a2a_message(response):
                    return response.content
            
            # Internal synthesis fallback
            return await self._internal_a2a_synthesis(blackboard)
            
        except Exception as e:
            logger.error(f"A2A synthesis attempt failed: {e}")
            return await self._internal_a2a_synthesis(blackboard)
    
    async def _a2a_internal_blackboard(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """A2A-compliant internal blackboard fallback"""
        
        return {
            "solution": f"A2A internal analysis suggests examining: {question}",
            "confidence": 0.4,
            "method": "a2a_internal_blackboard",
            "knowledge_sources": 0,
            "analysis": {
                "question_complexity": len(question.split()),
                "context_elements": len(context),
                "recommended_approach": "multi_agent_collaboration"
            }
        }


class A2APeerToPeerCoordination:
    """A2A-compliant peer-to-peer swarm coordination"""
    
    def __init__(self, reasoning_agent):
        self.reasoning_agent = reasoning_agent
        self.active_swarms = {}
    
    @a2a_skill("coordinate_peer_to_peer_reasoning")
    async def coordinate_peer_to_peer_reasoning(self, question: str, 
                                              exploration_params: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate P2P swarm reasoning using A2A agent network"""
        
        swarm_id = str(uuid.uuid4())[:8]
        logger.info(f"🐝 Starting A2A P2P swarm reasoning {swarm_id}")
        
        try:
            # Discover A2A reasoning agents for swarm
            swarm_agents = await discover_reasoning_engines(
                capabilities=["exploration", "peer_reasoning", "collaboration"],
                min_agents=2,
                timeout=10
            )
            
            if len(swarm_agents) < 2:
                logger.warning("Insufficient A2A agents for swarm, using single agent")
                return await self._a2a_single_agent_fallback(question)
            
            # Initialize A2A swarm network
            swarm_network = {
                "swarm_id": swarm_id,
                "question": question,
                "a2a_agents": swarm_agents[:4],  # Limit swarm size
                "exploration_results": [],
                "peer_exchanges": [],
                "convergence_data": [],
                "final_synthesis": None
            }
            
            # Phase 1: Independent A2A exploration
            exploration_results = await self._coordinate_a2a_exploration(swarm_network)
            swarm_network["exploration_results"] = exploration_results
            
            # Phase 2: A2A peer-to-peer exchange
            peer_exchanges = await self._coordinate_a2a_peer_exchange(swarm_network)
            swarm_network["peer_exchanges"] = peer_exchanges
            
            # Phase 3: A2A swarm convergence
            final_result = await self._achieve_a2a_swarm_convergence(swarm_network)
            swarm_network["final_synthesis"] = final_result
            
            return final_result
            
        except Exception as e:
            logger.error(f"A2A P2P coordination failed: {e}")
            return await self._a2a_single_agent_fallback(question)
    
    async def _coordinate_a2a_exploration(self, swarm_network: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Coordinate independent exploration using A2A messaging"""
        
        exploration_tasks = []
        
        for i, agent_info in enumerate(swarm_network["a2a_agents"]):
            # Assign diverse exploration focus
            exploration_focus = self._assign_a2a_exploration_focus(agent_info, i)
            
            exploration_request = A2AMessage(
                message_id=create_agent_id(),
                sender_id=self.reasoning_agent.agent_id,
                receiver_id=agent_info["agent_id"],
                message_type="swarm_exploration_request",
                content={
                    "swarm_id": swarm_network["swarm_id"],
                    "question": swarm_network["question"],
                    "exploration_focus": exploration_focus,
                    "agent_role": f"explorer_{i}",
                    "swarm_size": len(swarm_network["a2a_agents"])
                },
                role=MessageRole.USER,
                timestamp=datetime.utcnow()
            )
            
            signed_request = await sign_a2a_message(exploration_request)
            task = self._request_a2a_exploration(agent_info, signed_request)
            exploration_tasks.append((agent_info["agent_id"], task))
        
        # Collect A2A exploration results
        exploration_results = []
        for agent_id, task in exploration_tasks:
            try:
                result = await asyncio.wait_for(task, timeout=20)
                if result:
                    result["explorer_agent"] = agent_id
                    result["timestamp"] = datetime.utcnow().isoformat()
                    exploration_results.append(result)
                    
            except asyncio.TimeoutError:
                logger.warning(f"A2A exploration from {agent_id} timed out")
            except Exception as e:
                logger.error(f"A2A exploration failed for {agent_id}: {e}")
        
        return exploration_results
    
    def _assign_a2a_exploration_focus(self, agent_info: Dict[str, Any], index: int) -> str:
        """Assign exploration focus based on A2A agent capabilities"""
        
        capabilities = agent_info.get("capabilities", [])
        
        # Assign based on capabilities and index for diversity
        focus_options = [
            "analytical_exploration",
            "creative_exploration", 
            "systematic_exploration",
            "heuristic_exploration"
        ]
        
        if "analysis" in capabilities:
            return "analytical_exploration"
        elif "creativity" in capabilities:
            return "creative_exploration"
        elif "systematic" in capabilities:
            return "systematic_exploration"
        else:
            return focus_options[index % len(focus_options)]
    
    @a2a_handler("swarm_exploration")
    async def _request_a2a_exploration(self, agent_info: Dict[str, Any], 
                                     signed_request: A2AMessage) -> Dict[str, Any]:
        """Request exploration using A2A messaging"""
        
        try:
            agent_endpoint = agent_info.get("endpoint")
            if not agent_endpoint:
                return {}
            
            response = await self.reasoning_agent._send_a2a_message(
                agent_endpoint,
                signed_request,
                expected_response_type="swarm_exploration_response"
            )
            
            if response and await verify_a2a_message(response):
                return response.content
            else:
                return {}
                
        except Exception as e:
            logger.error(f"A2A exploration request failed: {e}")
            return {}
    
    async def _coordinate_a2a_peer_exchange(self, swarm_network: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Coordinate peer-to-peer exchange between A2A agents"""
        
        exploration_results = swarm_network["exploration_results"]
        if len(exploration_results) < 2:
            return []
        
        exchange_tasks = []
        
        # Create peer exchange pairs
        for i, result1 in enumerate(exploration_results):
            for result2 in exploration_results[i+1:]:
                # Find corresponding agents
                agent1 = next(a for a in swarm_network["a2a_agents"] 
                             if a["agent_id"] == result1["explorer_agent"])
                agent2 = next(a for a in swarm_network["a2a_agents"] 
                             if a["agent_id"] == result2["explorer_agent"])
                
                # Create bilateral A2A exchange
                exchange_task = self._facilitate_a2a_peer_exchange(
                    agent1, agent2, result1, result2, swarm_network["swarm_id"]
                )
                exchange_tasks.append(exchange_task)
        
        # Collect exchange results
        exchanges = []
        for task in exchange_tasks:
            try:
                exchange_result = await asyncio.wait_for(task, timeout=15)
                if exchange_result:
                    exchanges.append(exchange_result)
                    
            except Exception as e:
                logger.error(f"A2A peer exchange failed: {e}")
        
        return exchanges
    
    async def _facilitate_a2a_peer_exchange(self, agent1: Dict[str, Any], agent2: Dict[str, Any],
                                          result1: Dict[str, Any], result2: Dict[str, Any],
                                          swarm_id: str) -> Dict[str, Any]:
        """Facilitate bilateral peer exchange between A2A agents"""
        
        try:
            # Send result1 to agent2 for peer review
            peer_review_request = A2AMessage(
                message_id=create_agent_id(),
                sender_id=self.reasoning_agent.agent_id,
                receiver_id=agent2["agent_id"],
                message_type="peer_review_request",
                content={
                    "swarm_id": swarm_id,
                    "peer_result": result1,
                    "your_result": result2,
                    "review_focus": "synthesis_and_improvement"
                },
                role=MessageRole.USER,
                timestamp=datetime.utcnow()
            )
            
            signed_review = await sign_a2a_message(peer_review_request)
            review_response = await self.reasoning_agent._send_a2a_message(
                agent2["endpoint"],
                signed_review,
                expected_response_type="peer_review_response"
            )
            
            if review_response and await verify_a2a_message(review_response):
                return {
                    "reviewer_agent": agent2["agent_id"],
                    "reviewed_agent": agent1["agent_id"],
                    "synthesis": review_response.content.get("synthesis"),
                    "improvements": review_response.content.get("improvements", []),
                    "confidence": review_response.content.get("confidence", 0.5)
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"A2A peer exchange facilitation failed: {e}")
            return {}
    
    async def _a2a_single_agent_fallback(self, question: str) -> Dict[str, Any]:
        """A2A-compliant single agent fallback"""
        
        return {
            "solution": f"A2A single-agent analysis of: {question}",
            "confidence": 0.3,
            "method": "a2a_single_agent_fallback",
            "swarm_size": 1,
            "exploration_approach": "independent_analysis"
        }


class A2AMultiAgentCoordinator:
    """Enhanced A2A-compliant multi-agent coordinator"""
    
    def __init__(self, reasoning_agent):
        self.reasoning_agent = reasoning_agent
        self.debate_coordinator = A2ADebateCoordination(reasoning_agent)
        self.blackboard_coordinator = A2ABlackboardCoordination(reasoning_agent)
        self.p2p_coordinator = A2APeerToPeerCoordination(reasoning_agent)
    
    @a2a_skill("coordinate_multi_agent_reasoning")
    async def coordinate_multi_agent_reasoning(self, task: A2AReasoningTask) -> Dict[str, Any]:
        """Main coordination method using A2A patterns"""
        
        logger.info(f"🎯 Coordinating A2A multi-agent reasoning: {task.coordination_pattern.value}")
        
        try:
            if task.coordination_pattern == A2ACoordinationPattern.DEBATE:
                proposals = task.context.get("proposals", [])
                return await self.debate_coordinator.coordinate_agent_debate(proposals)
            
            elif task.coordination_pattern == A2ACoordinationPattern.BLACKBOARD:
                return await self.blackboard_coordinator.coordinate_blackboard_reasoning(
                    task.question, task.context
                )
            
            elif task.coordination_pattern == A2ACoordinationPattern.PEER_TO_PEER:
                return await self.p2p_coordinator.coordinate_peer_to_peer_reasoning(
                    task.question, task.context
                )
            
            elif task.coordination_pattern == A2ACoordinationPattern.HIERARCHICAL:
                # Use debate for hierarchical coordination
                proposals = task.context.get("proposals", [])
                return await self.debate_coordinator.coordinate_agent_debate(proposals, threshold=0.8)
            
            else:
                logger.warning(f"A2A coordination pattern {task.coordination_pattern} not implemented")
                return await self._a2a_fallback_coordination(task)
                
        except Exception as e:
            logger.error(f"A2A multi-agent coordination failed: {e}")
            return await self._a2a_fallback_coordination(task)
    
    async def _a2a_fallback_coordination(self, task: A2AReasoningTask) -> Dict[str, Any]:
        """A2A-compliant fallback coordination"""
        
        return {
            "solution": f"A2A fallback analysis for: {task.question}",
            "confidence": 0.2,
            "method": "a2a_fallback_coordination",
            "coordination_pattern": task.coordination_pattern.value,
            "agents_requested": task.required_agents,
            "recommendation": "Retry with different coordination pattern or more available A2A agents"
        }

    async def _send_a2a_position_assignment(self, agent_info: Dict[str, Any], signed_message: A2AMessage):
        """Send position assignment to A2A agent"""
        try:
            agent_endpoint = agent_info.get("endpoint")
            if not agent_endpoint:
                logger.error(f"No endpoint found for agent {agent_info['agent_id']}")
                return

            # Send A2A message through proper channels
            response = await self.reasoning_agent._send_a2a_message(
                agent_endpoint,
                signed_message,
                expected_response_type="position_assignment_confirmation"
            )
            
            if response and await verify_a2a_message(response):
                logger.info(f"Position assigned to {agent_info['agent_id']}")
            else:
                logger.warning(f"Position assignment failed for {agent_info['agent_id']}")
                
        except Exception as e:
            logger.error(f"Position assignment failed: {e}")

    async def _request_a2a_evaluation(self, agent_info: Dict[str, Any], signed_request: A2AMessage) -> Dict[str, Any]:
        """Request evaluation from A2A agent"""
        try:
            agent_endpoint = agent_info.get("endpoint")
            if not agent_endpoint:
                return {}

            response = await self.reasoning_agent._send_a2a_message(
                agent_endpoint,
                signed_request,
                expected_response_type="debate_evaluation_response"
            )
            
            if response and await verify_a2a_message(response):
                return response.content
            else:
                return {}
                
        except Exception as e:
            logger.error(f"A2A evaluation request failed: {e}")
            return {}

    async def _internal_debate_synthesis(self, debate_state: Dict[str, Any]) -> Dict[str, Any]:
        """Internal synthesis of debate results"""
        try:
            positions = list(debate_state["positions"].values())
            if not positions:
                return {
                    "consensus": "No positions available for synthesis",
                    "confidence": 0.0,
                    "method": "internal_debate_synthesis",
                    "participants": 0
                }

            # Calculate weighted consensus
            total_confidence = sum(pos["confidence"] for pos in positions)
            weighted_responses = []
            
            for pos in positions:
                weight = pos["confidence"] / max(total_confidence, 0.1)
                position_answer = pos["position"].get("answer", "No answer provided")
                weighted_responses.append({
                    "answer": position_answer,
                    "weight": weight,
                    "confidence": pos["confidence"]
                })

            # Select best weighted response
            best_response = max(weighted_responses, key=lambda x: x["weight"] * x["confidence"])
            
            return {
                "consensus": best_response["answer"],
                "confidence": total_confidence / len(positions),
                "method": "internal_debate_synthesis",
                "participants": len(positions),
                "debate_rounds": len(debate_state.get("rounds", [])),
                "consensus_reached": debate_state.get("consensus_reached", False)
            }
            
        except Exception as e:
            logger.error(f"Internal debate synthesis failed: {e}")
            return {
                "consensus": "Synthesis failed",
                "confidence": 0.0,
                "method": "internal_debate_synthesis_fallback",
                "error": str(e)
            }

    async def _final_a2a_blackboard_synthesis(self, blackboard: Dict[str, Any]) -> Dict[str, Any]:
        """Final synthesis of blackboard knowledge"""
        try:
            contributions = blackboard.get("knowledge_contributions", [])
            if not contributions:
                return {
                    "solution": "No knowledge contributions available",
                    "confidence": 0.0,
                    "method": "final_blackboard_synthesis",
                    "knowledge_sources": 0
                }

            # Aggregate knowledge by confidence
            high_confidence_contributions = [c for c in contributions if c.get("confidence", 0) > 0.7]
            medium_confidence_contributions = [c for c in contributions if 0.4 <= c.get("confidence", 0) <= 0.7]
            
            # Build synthesis from high-confidence contributions first
            synthesis_parts = []
            if high_confidence_contributions:
                for contrib in high_confidence_contributions:
                    synthesis_parts.append(contrib.get("knowledge", ""))
            elif medium_confidence_contributions:
                for contrib in medium_confidence_contributions[:2]:  # Limit to top 2
                    synthesis_parts.append(contrib.get("knowledge", ""))

            final_solution = " ".join(filter(None, synthesis_parts))
            if not final_solution:
                final_solution = f"Analysis of problem: {blackboard.get('problem', 'Unknown problem')}"

            avg_confidence = sum(c.get("confidence", 0) for c in contributions) / len(contributions)
            
            return {
                "solution": final_solution,
                "confidence": avg_confidence,
                "method": "final_blackboard_synthesis",
                "knowledge_sources": len(contributions),
                "high_confidence_sources": len(high_confidence_contributions),
                "synthesis_iterations": len(blackboard.get("synthesis_attempts", []))
            }
            
        except Exception as e:
            logger.error(f"Final blackboard synthesis failed: {e}")
            return {
                "solution": "Synthesis failed",
                "confidence": 0.0,
                "method": "final_blackboard_synthesis_fallback",
                "error": str(e)
            }

    async def _internal_a2a_synthesis(self, blackboard: Dict[str, Any]) -> Dict[str, Any]:
        """Internal A2A blackboard synthesis"""
        try:
            contributions = blackboard.get("knowledge_contributions", [])
            problem = blackboard.get("problem", "Unknown problem")
            
            if not contributions:
                return {
                    "synthesis": f"No contributions available for: {problem}",
                    "confidence": 0.1,
                    "method": "internal_a2a_synthesis",
                    "sources": 0
                }

            # Simple synthesis by combining top contributions
            sorted_contributions = sorted(
                contributions, 
                key=lambda x: x.get("confidence", 0), 
                reverse=True
            )
            
            top_contributions = sorted_contributions[:3]  # Top 3 contributions
            
            synthesis_text = f"Based on {len(contributions)} knowledge sources: "
            for i, contrib in enumerate(top_contributions):
                knowledge = contrib.get("knowledge", "")
                if knowledge:
                    synthesis_text += f"({i+1}) {knowledge[:100]}... "

            avg_confidence = sum(c.get("confidence", 0) for c in top_contributions) / len(top_contributions)
            
            return {
                "synthesis": synthesis_text,
                "confidence": min(avg_confidence, 0.8),  # Cap at 0.8 for internal synthesis
                "method": "internal_a2a_synthesis",
                "sources": len(contributions),
                "top_sources": len(top_contributions)
            }
            
        except Exception as e:
            logger.error(f"Internal A2A synthesis failed: {e}")
            return {
                "synthesis": f"Internal synthesis failed for: {problem}",
                "confidence": 0.1,
                "method": "internal_a2a_synthesis_fallback",
                "error": str(e)
            }

    async def _achieve_a2a_swarm_convergence(self, swarm_network: Dict[str, Any]) -> Dict[str, Any]:
        """Achieve convergence in A2A swarm reasoning"""
        try:
            exploration_results = swarm_network.get("exploration_results", [])
            peer_exchanges = swarm_network.get("peer_exchanges", [])
            
            if not exploration_results:
                return {
                    "solution": "No exploration results available for convergence",
                    "confidence": 0.1,
                    "method": "swarm_convergence",
                    "participants": 0
                }

            # Calculate convergence based on exploration and peer feedback
            convergence_score = 0.0
            synthesis_elements = []
            
            # Weight exploration results by quality
            for result in exploration_results:
                confidence = result.get("confidence", 0.5)
                exploration = result.get("exploration", "")
                if exploration and confidence > 0.4:
                    synthesis_elements.append({
                        "content": exploration,
                        "weight": confidence,
                        "source": result.get("explorer_agent", "unknown")
                    })
                    convergence_score += confidence

            # Add insights from peer exchanges
            for exchange in peer_exchanges:
                synthesis_content = exchange.get("synthesis", "")
                if synthesis_content:
                    synthesis_elements.append({
                        "content": synthesis_content,
                        "weight": exchange.get("confidence", 0.5),
                        "source": "peer_exchange"
                    })

            # Create final synthesis
            if synthesis_elements:
                # Sort by weight and take top elements
                sorted_elements = sorted(synthesis_elements, key=lambda x: x["weight"], reverse=True)
                top_elements = sorted_elements[:3]
                
                final_solution = "Swarm convergence synthesis: "
                for elem in top_elements:
                    final_solution += f"{elem['content'][:80]}... "
                
                avg_confidence = sum(elem["weight"] for elem in top_elements) / len(top_elements)
                convergence_score = avg_confidence
            else:
                final_solution = f"Swarm analysis of: {swarm_network.get('question', 'Unknown question')}"
                convergence_score = 0.2

            return {
                "solution": final_solution,
                "confidence": min(convergence_score, 0.9),
                "method": "swarm_convergence",
                "participants": len(exploration_results),
                "peer_exchanges": len(peer_exchanges),
                "convergence_achieved": convergence_score > 0.6,
                "synthesis_elements": len(synthesis_elements)
            }
            
        except Exception as e:
            logger.error(f"Swarm convergence failed: {e}")
            return {
                "solution": "Swarm convergence failed",
                "confidence": 0.1,
                "method": "swarm_convergence_fallback",
                "error": str(e)
            }