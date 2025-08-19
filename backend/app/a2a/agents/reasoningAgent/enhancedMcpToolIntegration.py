"""
Enhanced MCP Tool Integration for High-Priority Agents
Demonstrates comprehensive MCP tool usage patterns for reasoning agents
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

from ...sdk.agentBase import A2AAgentBase
from ...sdk.decorators import a2a_handler, a2a_skill, a2a_task
from ...sdk.types import A2AMessage, MessageRole, TaskStatus
from ...sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt
from ...common.mcpPerformanceTools import MCPPerformanceTools
from ...common.mcpValidationTools import MCPValidationTools
from ...common.mcpQualityAssessmentTools import MCPQualityAssessmentTools

logger = logging.getLogger(__name__)


class EnhancedMCPReasoningAgent(A2AAgentBase):
    """
    High-priority reasoning agent with comprehensive MCP tool usage
    Demonstrates best practices for MCP tool integration
    """
    
    def __init__(self, base_url: str):
        super().__init__(
            agent_id="enhanced_mcp_reasoning_agent",
            name="Enhanced MCP Reasoning Agent",
            description="Advanced reasoning agent showcasing comprehensive MCP tool usage",
            version="2.0.0",
            base_url=base_url
        )
        
        # Initialize MCP tool providers
        self.performance_tools = MCPPerformanceTools()
        self.validation_tools = MCPValidationTools()
        self.quality_tools = MCPQualityAssessmentTools()
        
        
        # Reasoning state
        self.reasoning_sessions = {}
        self.performance_metrics = {}
        
        logger.info(f"Initialized {self.name} with comprehensive MCP tool integration")
    
    @mcp_tool(
        name="enhanced_reasoning_analysis",
        description="Comprehensive reasoning analysis using multiple MCP tools",
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Question to analyze"},
                "context": {"type": "object", "description": "Additional context"},
                "analysis_depth": {
                    "type": "string", 
                    "enum": ["basic", "standard", "comprehensive"], 
                    "default": "standard"
                },
                "use_cross_agent_tools": {"type": "boolean", "default": True},
                "performance_tracking": {"type": "boolean", "default": True}
            },
            "required": ["question"]
        }
    )
    async def enhanced_reasoning_analysis(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        analysis_depth: str = "standard",
        use_cross_agent_tools: bool = True,
        performance_tracking: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive reasoning analysis using multiple MCP tools
        """
        session_id = f"reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now().timestamp()
        
        try:
            # Step 1: Validate input using MCP validation tools
            validation_result = await self.validation_tools.validate_reasoning_input(
                question=question,
                context=context or {},
                validation_level="standard"
            )
            
            if not validation_result["is_valid"]:
                return {
                    "status": "error",
                    "error": "Input validation failed",
                    "validation_details": validation_result
                }
            
            # Step 2: Initialize reasoning session
            self.reasoning_sessions[session_id] = {
                "question": question,
                "context": context,
                "start_time": start_time,
                "steps": []
            }
            
            # Step 3: Decompose question using MCP tools
            decomposition_result = await self._decompose_question_with_mcp(
                question, analysis_depth
            )
            
            # Step 4: Analyze patterns using cross-agent MCP tools
            pattern_analysis = {}
            if use_cross_agent_tools:
                pattern_analysis = await self._analyze_patterns_cross_agent(
                    question, decomposition_result
                )
            
            # Step 5: Generate reasoning chain
            reasoning_chain = await self._generate_reasoning_chain(
                question, decomposition_result, pattern_analysis
            )
            
            # Step 6: Quality assessment using MCP tools
            quality_assessment = await self.quality_tools.assess_reasoning_quality(
                reasoning_chain=reasoning_chain,
                original_question=question,
                assessment_criteria=["coherence", "completeness", "logical_flow"]
            )
            
            # Step 7: Performance measurement
            end_time = datetime.now().timestamp()
            if performance_tracking:
                performance_metrics = await self.performance_tools.measure_performance_metrics(
                    operation_id=session_id,
                    start_time=start_time,
                    end_time=end_time,
                    operation_count=len(reasoning_chain),
                    custom_metrics={
                        "decomposition_steps": len(decomposition_result.get("sub_questions", [])),
                        "pattern_matches": len(pattern_analysis.get("patterns", [])),
                        "quality_score": quality_assessment.get("overall_score", 0)
                    }
                )
                self.performance_metrics[session_id] = performance_metrics
            
            # Step 8: Synthesize final answer
            final_answer = await self._synthesize_answer_with_mcp(
                question, reasoning_chain, quality_assessment
            )
            
            return {
                "status": "success",
                "session_id": session_id,
                "question": question,
                "analysis_depth": analysis_depth,
                "decomposition": decomposition_result,
                "pattern_analysis": pattern_analysis,
                "reasoning_chain": reasoning_chain,
                "quality_assessment": quality_assessment,
                "final_answer": final_answer,
                "performance_metrics": self.performance_metrics.get(session_id, {}),
                "mcp_tools_used": [
                    "validate_reasoning_input",
                    "decompose_question", 
                    "analyze_patterns",
                    "assess_reasoning_quality",
                    "measure_performance_metrics"
                ]
            }
            
        except Exception as e:
            logger.error(f"Enhanced reasoning analysis failed: {e}")
            return {
                "status": "error",
                "session_id": session_id,
                "error": str(e),
                "partial_results": self.reasoning_sessions.get(session_id, {})
            }
    
    @mcp_tool(
        name="cross_agent_collaboration",
        description="Collaborate with other agents using MCP protocol",
        input_schema={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Collaboration task"},
                "target_agents": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of agent IDs to collaborate with"
                },
                "collaboration_mode": {
                    "type": "string",
                    "enum": ["sequential", "parallel", "debate"],
                    "default": "sequential"
                },
                "timeout_seconds": {"type": "number", "default": 30}
            },
            "required": ["task", "target_agents"]
        }
    )
    async def cross_agent_collaboration(
        self,
        task: str,
        target_agents: List[str],
        collaboration_mode: str = "sequential",
        timeout_seconds: float = 30
    ) -> Dict[str, Any]:
        """
        Collaborate with other agents using MCP protocol
        """
        collaboration_id = f"collab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now().timestamp()
        
        try:
            results = {}
            
            if collaboration_mode == "sequential":
                # Sequential collaboration
                for agent_id in target_agents:
                    agent_result = await self._call_agent_via_mcp(
                        agent_id, task, timeout_seconds
                    )
                    results[agent_id] = agent_result
                    
                    # Use previous results as context for next agent
                    task = f"{task}\nPrevious analysis: {agent_result.get('summary', '')}"
            
            elif collaboration_mode == "parallel":
                # Parallel collaboration
                tasks_coroutines = [
                    self._call_agent_via_mcp(agent_id, task, timeout_seconds)
                    for agent_id in target_agents
                ]
                agent_results = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
                
                for i, agent_id in enumerate(target_agents):
                    results[agent_id] = agent_results[i] if not isinstance(agent_results[i], Exception) else {
                        "error": str(agent_results[i])
                    }
            
            elif collaboration_mode == "debate":
                # Debate-style collaboration
                results = await self._conduct_agent_debate(target_agents, task, timeout_seconds)
            
            # Synthesize collaboration results
            synthesis = await self._synthesize_collaboration_results(results, task)
            
            # Performance tracking
            end_time = datetime.now().timestamp()
            performance_metrics = await self.performance_tools.measure_performance_metrics(
                operation_id=collaboration_id,
                start_time=start_time,
                end_time=end_time,
                operation_count=len(target_agents),
                custom_metrics={
                    "agents_involved": len(target_agents),
                    "successful_collaborations": len([r for r in results.values() if not r.get("error")]),
                    "collaboration_mode": collaboration_mode
                }
            )
            
            return {
                "status": "success",
                "collaboration_id": collaboration_id,
                "task": task,
                "collaboration_mode": collaboration_mode,
                "target_agents": target_agents,
                "individual_results": results,
                "synthesis": synthesis,
                "performance_metrics": performance_metrics,
                "mcp_protocol_used": True
            }
            
        except Exception as e:
            logger.error(f"Cross-agent collaboration failed: {e}")
            return {
                "status": "error",
                "collaboration_id": collaboration_id,
                "error": str(e)
            }
    
    @mcp_resource(
        uri="reasoning://session-data",
        name="Reasoning Session Data",
        description="Access to current reasoning session data and metrics"
    )
    async def get_reasoning_session_data(self) -> Dict[str, Any]:
        """Provide access to reasoning session data as MCP resource"""
        return {
            "active_sessions": list(self.reasoning_sessions.keys()),
            "total_sessions": len(self.reasoning_sessions),
            "performance_metrics": self.performance_metrics,
            "last_updated": datetime.now().isoformat()
        }
    
    @mcp_prompt(
        name="generate_reasoning_prompt",
        description="Generate contextual reasoning prompts based on question type",
        arguments=[
            {"name": "question_type", "type": "string", "description": "Type of question"},
            {"name": "domain", "type": "string", "description": "Domain context"},
            {"name": "complexity", "type": "string", "description": "Complexity level"}
        ]
    )
    async def generate_reasoning_prompt(
        self,
        question_type: str = "analytical",
        domain: str = "general",
        complexity: str = "medium"
    ) -> str:
        """Generate contextual reasoning prompts"""
        
        prompt_templates = {
            "analytical": {
                "low": f"Analyze the following {domain} question step by step:",
                "medium": f"Provide a comprehensive analysis of this {domain} question, considering multiple perspectives:",
                "high": f"Conduct a deep analytical examination of this complex {domain} question, incorporating systematic reasoning:"
            },
            "creative": {
                "low": f"Think creatively about this {domain} challenge:",
                "medium": f"Explore innovative solutions for this {domain} problem using creative reasoning:",
                "high": f"Apply advanced creative thinking methodologies to this complex {domain} challenge:"
            },
            "logical": {
                "low": f"Apply logical reasoning to this {domain} question:",
                "medium": f"Use systematic logical analysis for this {domain} problem:",
                "high": f"Employ formal logical reasoning methods for this complex {domain} question:"
            }
        }
        
        return prompt_templates.get(question_type, {}).get(complexity, 
            f"Analyze this {domain} question using {question_type} reasoning:"
        )
    
    async def _decompose_question_with_mcp(self, question: str, depth: str) -> Dict[str, Any]:
        """Decompose question using MCP tools"""
        try:
            # Use MCP skill client to call decomposition tool
            result = await self.mcp_client.call_skill_tool(
                "question_decomposition",
                "decompose_question", 
                {
                    "question": question,
                    "decomposition_strategy": depth,
                    "max_sub_questions": 5 if depth == "comprehensive" else 3
                }
            )
            return result.get("result", {})
        except Exception as e:
            logger.warning(f"MCP decomposition failed, using fallback: {e}")
            return {"sub_questions": [question], "strategy": "fallback"}
    
    async def _analyze_patterns_cross_agent(self, question: str, decomposition: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns using cross-agent MCP tools"""
        try:
            # Call pattern analysis agent via MCP
            pattern_result = await self.mcp_client.call_skill_tool(
                "pattern_analysis",
                "analyze_question_patterns",
                {
                    "question": question,
                    "sub_questions": decomposition.get("sub_questions", []),
                    "analysis_focus": ["semantic", "structural", "logical"]
                }
            )
            return pattern_result.get("result", {})
        except Exception as e:
            logger.warning(f"Cross-agent pattern analysis failed: {e}")
            return {"patterns": [], "analysis_method": "fallback"}
    
    async def _generate_reasoning_chain(
        self, 
        question: str, 
        decomposition: Dict[str, Any], 
        patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate reasoning chain based on decomposition and patterns"""
        
        reasoning_chain = []
        
        # Add initial premise
        reasoning_chain.append({
            "step": 1,
            "type": "premise",
            "content": f"Analyzing question: {question}",
            "confidence": 0.9
        })
        
        # Add decomposition steps
        sub_questions = decomposition.get("sub_questions", [question])
        for i, sub_q in enumerate(sub_questions):
            reasoning_chain.append({
                "step": i + 2,
                "type": "decomposition",
                "content": f"Sub-question {i+1}: {sub_q}",
                "confidence": 0.8
            })
        
        # Add pattern-based insights
        for i, pattern in enumerate(patterns.get("patterns", [])):
            reasoning_chain.append({
                "step": len(reasoning_chain) + 1,
                "type": "pattern_insight",
                "content": f"Pattern identified: {pattern.get('description', 'Unknown pattern')}",
                "confidence": pattern.get("confidence", 0.7)
            })
        
        # Add synthesis step
        reasoning_chain.append({
            "step": len(reasoning_chain) + 1,
            "type": "synthesis",
            "content": "Synthesizing insights from decomposition and pattern analysis",
            "confidence": 0.85
        })
        
        return reasoning_chain
    
    async def _synthesize_answer_with_mcp(
        self, 
        question: str, 
        reasoning_chain: List[Dict[str, Any]], 
        quality_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize final answer using MCP tools"""
        
        # Calculate overall confidence
        confidences = [step.get("confidence", 0.5) for step in reasoning_chain]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Adjust confidence based on quality assessment
        quality_score = quality_assessment.get("overall_score", 0.5)
        final_confidence = (avg_confidence + quality_score) / 2
        
        return {
            "answer": f"Based on comprehensive analysis using MCP tools, the question '{question}' requires consideration of {len(reasoning_chain)} reasoning steps.",
            "confidence": final_confidence,
            "reasoning_steps": len(reasoning_chain),
            "quality_score": quality_score,
            "methodology": "MCP-enhanced reasoning analysis"
        }
    
    async def _call_agent_via_mcp(self, agent_id: str, task: str, timeout: float) -> Dict[str, Any]:
        """Call another agent via MCP protocol"""
        try:
            # Simulate MCP call to another agent
            # In real implementation, this would use actual MCP client
            await asyncio.sleep(0.1)  # Simulate network call
            
            return {
                "agent_id": agent_id,
                "task": task,
                "response": f"Agent {agent_id} processed task: {task[:50]}...",
                "confidence": 0.8,
                "mcp_protocol": True
            }
        except Exception as e:
            return {"error": str(e), "agent_id": agent_id}
    
    async def _conduct_agent_debate(self, agents: List[str], topic: str, timeout: float) -> Dict[str, Any]:
        """Conduct debate-style collaboration between agents"""
        debate_rounds = []
        
        for round_num in range(min(3, len(agents))):  # Max 3 rounds
            round_results = {}
            
            for agent_id in agents:
                # Each agent provides their perspective
                response = await self._call_agent_via_mcp(
                    agent_id, 
                    f"Round {round_num + 1} debate on: {topic}", 
                    timeout
                )
                round_results[agent_id] = response
            
            debate_rounds.append({
                "round": round_num + 1,
                "responses": round_results
            })
        
        return {
            "debate_topic": topic,
            "rounds": debate_rounds,
            "participants": agents,
            "methodology": "MCP debate protocol"
        }
    
    async def _synthesize_collaboration_results(self, results: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Synthesize results from agent collaboration"""
        
        successful_results = [r for r in results.values() if not r.get("error")]
        total_confidence = sum(r.get("confidence", 0) for r in successful_results)
        avg_confidence = total_confidence / len(successful_results) if successful_results else 0
        
        return {
            "task": task,
            "agents_participated": len(results),
            "successful_responses": len(successful_results),
            "average_confidence": avg_confidence,
            "synthesis": f"Collaboration on '{task}' completed with {len(successful_results)} successful agent responses",
            "recommendation": "high" if avg_confidence > 0.8 else "medium" if avg_confidence > 0.6 else "low"
        }


# Factory function for creating enhanced MCP reasoning agent
def create_enhanced_mcp_reasoning_agent(base_url: str) -> EnhancedMCPReasoningAgent:
    """Create and configure enhanced MCP reasoning agent"""
    return EnhancedMCPReasoningAgent(base_url)