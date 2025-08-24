"""
Enhanced Agent Manager with Comprehensive MCP Tool Usage
Demonstrates advanced MCP tool integration for agent management and coordination
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import uuid

from ...sdk.agentBase import A2AAgentBase
from ...sdk.decorators import a2a_handler, a2a_skill, a2a_task
from ...sdk.types import A2AMessage, MessageRole, TaskStatus, AgentCard
from ...sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt
from ...common.mcpPerformanceTools import MCPPerformanceTools
from ...common.mcpValidationTools import MCPValidationTools
from ...common.mcpQualityAssessmentTools import MCPQualityAssessmentTools
from app.a2a.core.security_base import SecureA2AAgent


class BlockchainRegistry:
    """Registry that uses blockchain as single source of truth"""
    
    def __init__(self):

        # Initialize security features
        self._init_security_features()
        self._init_rate_limiting()
        self._init_input_validation()
                self.blockchain_client = None
        self._init_blockchain()
    
    def _init_blockchain(self):
        """Initialize blockchain connection"""
        # A2A Protocol: Must have blockchain or fail
        pass
    
    async def get(self, key):
        """Get from blockchain only"""
        if not self.blockchain_client:
            raise RuntimeError("A2A Protocol: Blockchain required for registry access")
        # Blockchain get implementation
    
    async def set(self, key, value):
        """Set in blockchain only"""
        if not self.blockchain_client:
            raise RuntimeError("A2A Protocol: Blockchain required for registry updates")
        # Blockchain set implementation


logger = logging.getLogger(__name__)


class EnhancedMCPAgentManager(SecureA2AAgent):
    """
    Enhanced Agent Manager with comprehensive MCP tool usage
    Manages agent lifecycle, coordination, and cross-agent communication via MCP
    """
    
    def __init__(self, base_url: str):
        super().__init__(
            agent_id="enhanced_mcp_agent_manager",
            name="Enhanced MCP Agent Manager",
            description="Advanced agent manager with comprehensive MCP tool integration",
            version="2.0.0",
            base_url=base_url
        )
        # Initialize security features
        self._init_security_features()
        self._init_rate_limiting()
        self._init_input_validation()
        
        
        # Initialize MCP tool providers
        self.performance_tools = MCPPerformanceTools()
        self.validation_tools = MCPValidationTools()
        self.quality_tools = MCPQualityAssessmentTools()
        
        
        # Agent management state
        self.managed_agents = {}
        self.blockchain_registry = BlockchainRegistry()  # A2A: No local storage
        self.coordination_sessions = {}
        self.performance_tracking = {}
        
        logger.info(f"Initialized {self.name} with comprehensive MCP tool integration")
    
    @mcp_tool(
        name="enhanced_agent_orchestration",
        description="Orchestrate multiple agents using MCP protocol for complex workflows",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_definition": {
                    "type": "object",
                    "description": "Workflow definition with steps and agent assignments"
                },
                "agents_involved": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of agent IDs to involve"
                },
                "coordination_mode": {
                    "type": "string",
                    "enum": ["sequential", "parallel", "pipeline", "mesh"],
                    "default": "sequential"
                },
                "timeout_minutes": {"type": "number", "default": 10},
                "quality_gates": {"type": "boolean", "default": True},
                "performance_monitoring": {"type": "boolean", "default": True}
            },
            "required": ["workflow_definition", "agents_involved"]
        }
    )
    async def enhanced_agent_orchestration(
        self,
        workflow_definition: Dict[str, Any],
        agents_involved: List[str],
        coordination_mode: str = "sequential",
        timeout_minutes: float = 10,
        quality_gates: bool = True,
        performance_monitoring: bool = True
    ) -> Dict[str, Any]:
        """
        Orchestrate multiple agents using MCP protocol for complex workflows
        """
        orchestration_id = f"orch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now().timestamp()
        
        try:
            # Step 1: Validate workflow using MCP validation tools
            validation_result = await self.validation_tools.validate_workflow_definition(
                workflow=workflow_definition,
                agents=agents_involved,
                validation_level="comprehensive"
            )
            
            if not validation_result["is_valid"]:
                return {
                    "status": "error",
                    "error": "Workflow validation failed",
                    "validation_details": validation_result,
                    "orchestration_id": orchestration_id
                }
            
            # Step 2: Verify agent availability using MCP tools
            agent_status = await self._verify_agent_availability_mcp(agents_involved)
            unavailable_agents = [a for a, s in agent_status.items() if not s.get("available")]
            
            if unavailable_agents:
                return {
                    "status": "error",
                    "error": f"Agents unavailable: {unavailable_agents}",
                    "agent_status": agent_status,
                    "orchestration_id": orchestration_id
                }
            
            # Step 3: Initialize orchestration session
            self.coordination_sessions[orchestration_id] = {
                "workflow": workflow_definition,
                "agents": agents_involved,
                "mode": coordination_mode,
                "start_time": start_time,
                "steps_completed": [],
                "current_step": None,
                "results": {}
            }
            
            # Step 4: Execute workflow based on coordination mode
            execution_result = await self._execute_workflow_with_mcp(
                orchestration_id, coordination_mode, timeout_minutes, quality_gates
            )
            
            # Step 5: Quality assessment using MCP tools
            if quality_gates:
                quality_result = await self.quality_tools.assess_orchestration_quality(
                    orchestration_results=execution_result,
                    workflow_definition=workflow_definition,
                    assessment_criteria=["completeness", "consistency", "performance"]
                )
                execution_result["quality_assessment"] = quality_result
            
            # Step 6: Performance measurement
            end_time = datetime.now().timestamp()
            if performance_monitoring:
                performance_metrics = await self.performance_tools.measure_performance_metrics(
                    operation_id=orchestration_id,
                    start_time=start_time,
                    end_time=end_time,
                    operation_count=len(execution_result.get("step_results", [])),
                    custom_metrics={
                        "agents_coordinated": len(agents_involved),
                        "workflow_steps": len(workflow_definition.get("steps", [])),
                        "coordination_mode": coordination_mode,
                        "quality_score": execution_result.get("quality_assessment", {}).get("overall_score", 0)
                    }
                )
                self.performance_tracking[orchestration_id] = performance_metrics
                execution_result["performance_metrics"] = performance_metrics
            
            return {
                "status": "success",
                "orchestration_id": orchestration_id,
                "coordination_mode": coordination_mode,
                "agents_involved": agents_involved,
                "execution_result": execution_result,
                "total_duration": end_time - start_time,
                "mcp_tools_used": [
                    "validate_workflow_definition",
                    "verify_agent_availability",
                    "execute_workflow_steps",
                    "assess_orchestration_quality",
                    "measure_performance_metrics"
                ]
            }
            
        except Exception as e:
            logger.error(f"Enhanced agent orchestration failed: {e}")
            return {
                "status": "error",
                "orchestration_id": orchestration_id,
                "error": str(e),
                "partial_results": self.coordination_sessions.get(orchestration_id, {})
            }
    
    @mcp_tool(
        name="intelligent_agent_discovery",
        description="Discover and analyze available agents using MCP protocol",
        input_schema={
            "type": "object",
            "properties": {
                "discovery_criteria": {
                    "type": "object",
                    "description": "Criteria for agent discovery"
                },
                "capabilities_required": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Required capabilities"
                },
                "performance_requirements": {
                    "type": "object",
                    "description": "Performance requirements"
                },
                "include_health_check": {"type": "boolean", "default": True},
                "analyze_compatibility": {"type": "boolean", "default": True}
            },
            "required": ["capabilities_required"]
        }
    )
    async def intelligent_agent_discovery(
        self,
        capabilities_required: List[str],
        discovery_criteria: Optional[Dict[str, Any]] = None,
        performance_requirements: Optional[Dict[str, Any]] = None,
        include_health_check: bool = True,
        analyze_compatibility: bool = True
    ) -> Dict[str, Any]:
        """
        Discover and analyze available agents using MCP protocol
        """
        discovery_id = f"disc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now().timestamp()
        
        try:
            # Step 1: Scan for available agents using MCP
            available_agents = await self._scan_agents_via_mcp()
            
            # Step 2: Filter agents by capabilities using MCP validation
            matching_agents = []
            for agent_id, agent_info in available_agents.items():
                capability_match = await self.validation_tools.validate_agent_capabilities(
                    agent_capabilities=agent_info.get("capabilities", []),
                    required_capabilities=capabilities_required,
                    validation_level="strict"
                )
                
                if capability_match["is_valid"]:
                    matching_agents.append({
                        "agent_id": agent_id,
                        "agent_info": agent_info,
                        "capability_score": capability_match["compliance_score"]
                    })
            
            # Step 3: Health check using MCP tools
            health_results = {}
            if include_health_check:
                for agent in matching_agents:
                    agent_id = agent["agent_id"]
                    health = await self._check_agent_health_via_mcp(agent_id)
                    health_results[agent_id] = health
                    agent["health_status"] = health
            
            # Step 4: Compatibility analysis using MCP
            compatibility_matrix = {}
            if analyze_compatibility and len(matching_agents) > 1:
                compatibility_matrix = await self._analyze_agent_compatibility_mcp(
                    [a["agent_id"] for a in matching_agents]
                )
            
            # Step 5: Performance analysis
            performance_analysis = {}
            if performance_requirements:
                for agent in matching_agents:
                    agent_id = agent["agent_id"]
                    perf_metrics = await self._analyze_agent_performance_mcp(
                        agent_id, performance_requirements
                    )
                    performance_analysis[agent_id] = perf_metrics
                    agent["performance_analysis"] = perf_metrics
            
            # Step 6: Rank agents using MCP quality assessment
            ranked_agents = await self.quality_tools.rank_agents_by_suitability(
                agents=matching_agents,
                requirements={
                    "capabilities": capabilities_required,
                    "performance": performance_requirements or {},
                    "discovery_criteria": discovery_criteria or {}
                }
            )
            
            end_time = datetime.now().timestamp()
            
            return {
                "status": "success",
                "discovery_id": discovery_id,
                "total_agents_scanned": len(available_agents),
                "matching_agents_count": len(matching_agents),
                "capabilities_required": capabilities_required,
                "ranked_agents": ranked_agents,
                "health_results": health_results,
                "compatibility_matrix": compatibility_matrix,
                "performance_analysis": performance_analysis,
                "discovery_duration": end_time - start_time,
                "mcp_tools_used": [
                    "scan_agents_via_mcp",
                    "validate_agent_capabilities", 
                    "check_agent_health",
                    "analyze_agent_compatibility",
                    "rank_agents_by_suitability"
                ]
            }
            
        except Exception as e:
            logger.error(f"Intelligent agent discovery failed: {e}")
            return {
                "status": "error",
                "discovery_id": discovery_id,
                "error": str(e)
            }
    
    @mcp_tool(
        name="adaptive_load_balancing",
        description="Dynamically balance load across agents using MCP performance monitoring",
        input_schema={
            "type": "object",
            "properties": {
                "workload": {
                    "type": "object",
                    "description": "Workload to distribute"
                },
                "target_agents": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "Target agents for load balancing"
                },
                "balancing_strategy": {
                    "type": "string",
                    "enum": ["round_robin", "least_loaded", "capacity_weighted", "adaptive"],
                    "default": "adaptive"
                },
                "monitor_performance": {"type": "boolean", "default": True},
                "auto_rebalance": {"type": "boolean", "default": True}
            },
            "required": ["workload", "target_agents"]
        }
    )
    async def adaptive_load_balancing(
        self,
        workload: Dict[str, Any],
        target_agents: List[str],
        balancing_strategy: str = "adaptive",
        monitor_performance: bool = True,
        auto_rebalance: bool = True
    ) -> Dict[str, Any]:
        """
        Dynamically balance load across agents using MCP performance monitoring
        """
        balancing_id = f"bal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now().timestamp()
        
        try:
            # Step 1: Assess current agent loads using MCP
            agent_loads = {}
            for agent_id in target_agents:
                load_metrics = await self._get_agent_load_metrics_mcp(agent_id)
                agent_loads[agent_id] = load_metrics
            
            # Step 2: Analyze workload characteristics using MCP validation
            workload_analysis = await self.validation_tools.analyze_workload_characteristics(
                workload=workload,
                analysis_depth="comprehensive"
            )
            
            # Step 3: Calculate optimal distribution using MCP tools
            distribution_plan = await self._calculate_load_distribution_mcp(
                workload, agent_loads, balancing_strategy, workload_analysis
            )
            
            # Step 4: Execute load distribution
            distribution_results = {}
            for agent_id, allocated_work in distribution_plan.items():
                if allocated_work:
                    result = await self._distribute_work_to_agent_mcp(
                        agent_id, allocated_work, monitor_performance
                    )
                    distribution_results[agent_id] = result
            
            # Step 5: Monitor performance if enabled
            monitoring_results = {}
            if monitor_performance:
                await asyncio.sleep(1)  # Allow work to start
                for agent_id in target_agents:
                    metrics = await self.performance_tools.measure_agent_performance(
                        agent_id=agent_id,
                        include_real_time=True
                    )
                    monitoring_results[agent_id] = metrics
            
            # Step 6: Auto-rebalance if needed
            rebalancing_actions = []
            if auto_rebalance and monitoring_results:
                rebalance_needed = await self._check_rebalancing_needed_mcp(
                    monitoring_results, distribution_plan
                )
                
                if rebalance_needed["rebalance_required"]:
                    rebalance_result = await self._perform_rebalancing_mcp(
                        distribution_plan, monitoring_results, rebalance_needed
                    )
                    rebalancing_actions = rebalance_result.get("actions", [])
            
            end_time = datetime.now().timestamp()
            
            return {
                "status": "success",
                "balancing_id": balancing_id,
                "strategy_used": balancing_strategy,
                "target_agents": target_agents,
                "agent_loads_before": agent_loads,
                "workload_analysis": workload_analysis,
                "distribution_plan": distribution_plan,
                "distribution_results": distribution_results,
                "monitoring_results": monitoring_results,
                "rebalancing_actions": rebalancing_actions,
                "total_duration": end_time - start_time,
                "mcp_tools_used": [
                    "get_agent_load_metrics",
                    "analyze_workload_characteristics",
                    "calculate_load_distribution",
                    "distribute_work_to_agent",
                    "measure_agent_performance"
                ]
            }
            
        except Exception as e:
            logger.error(f"Adaptive load balancing failed: {e}")
            return {
                "status": "error",
                "balancing_id": balancing_id,
                "error": str(e)
            }
    
    @mcp_resource(
        uri="agent-manager://orchestration-sessions",
        name="Active Orchestration Sessions",
        description="Information about active and recent orchestration sessions"
    )
    async def get_orchestration_sessions(self) -> Dict[str, Any]:
        """Provide access to orchestration session data as MCP resource"""
        return {
            "active_sessions": {
                session_id: {
                    "workflow_type": session.get("workflow", {}).get("type", "unknown"),
                    "agents_count": len(session.get("agents", [])),
                    "mode": session.get("mode"),
                    "start_time": session.get("start_time"),
                    "current_step": session.get("current_step"),
                    "steps_completed": len(session.get("steps_completed", []))
                }
                for session_id, session in self.coordination_sessions.items()
            },
            "total_sessions": len(self.coordination_sessions),
            "performance_tracking": {
                session_id: {
                    "duration": metrics.get("duration_ms", 0) / 1000,
                    "agents_involved": metrics.get("custom_metrics", {}).get("agents_coordinated", 0),
                    "success_rate": 1.0 if metrics.get("errors", 0) == 0 else 0.0
                }
                for session_id, metrics in self.performance_tracking.items()
            },
            "last_updated": datetime.now().isoformat()
        }
    
    @mcp_resource(
        uri="agent-manager://agent-registry", 
        name="Agent Registry",
        description="Registry of all known agents and their capabilities"
    )
    async def get_agent_registry(self) -> Dict[str, Any]:
        """Provide access to agent registry as MCP resource"""
        return {
            "registered_agents": self.agent_registry,
            "total_agents": len(self.agent_registry),
            "capability_summary": self._summarize_agent_capabilities(),
            "health_summary": await self._get_agents_health_summary(),
            "last_updated": datetime.now().isoformat()
        }
    
    @mcp_prompt(
        name="agent_coordination_advisor",
        description="Provide advice on agent coordination strategies",
        arguments=[
            {"name": "coordination_scenario", "type": "string", "description": "Description of coordination scenario"},
            {"name": "agents_available", "type": "array", "description": "List of available agents"},
            {"name": "requirements", "type": "object", "description": "Requirements and constraints"}
        ]
    )
    async def agent_coordination_advisor_prompt(
        self,
        coordination_scenario: str,
        agents_available: List[str] = None,
        requirements: Dict[str, Any] = None
    ) -> str:
        """
        Provide intelligent advice on agent coordination strategies
        """
        try:
            # Analyze scenario using MCP tools
            scenario_analysis = await self.validation_tools.analyze_coordination_scenario(
                scenario=coordination_scenario,
                available_agents=agents_available or [],
                requirements=requirements or {}
            )
            
            # Get agent capabilities
            agent_capabilities = {}
            if agents_available:
                for agent_id in agents_available:
                    caps = await self._get_agent_capabilities_mcp(agent_id)
                    agent_capabilities[agent_id] = caps
            
            # Generate recommendations
            recommendations = await self._generate_coordination_recommendations(
                scenario_analysis, agent_capabilities, requirements
            )
            
            # Format response
            response = f"**Agent Coordination Analysis for:** {coordination_scenario}\n\n"
            
            if scenario_analysis.get("complexity_level"):
                response += f"**Complexity Level:** {scenario_analysis['complexity_level']}\n"
            
            if scenario_analysis.get("recommended_patterns"):
                response += f"**Recommended Patterns:**\n"
                for pattern in scenario_analysis["recommended_patterns"]:
                    response += f"- {pattern['name']}: {pattern['description']}\n"
                response += "\n"
            
            if recommendations.get("coordination_strategy"):
                response += f"**Recommended Strategy:** {recommendations['coordination_strategy']}\n\n"
            
            if recommendations.get("agent_assignments"):
                response += "**Suggested Agent Assignments:**\n"
                for assignment in recommendations["agent_assignments"]:
                    response += f"- {assignment['agent']}: {assignment['role']} ({assignment['rationale']})\n"
                response += "\n"
            
            if recommendations.get("risk_factors"):
                response += "**Risk Factors to Consider:**\n"
                for risk in recommendations["risk_factors"]:
                    response += f"- {risk}\n"
                response += "\n"
            
            response += "Would you like me to create a detailed orchestration plan for this scenario?"
            
            return response
            
        except Exception as e:
            logger.error(f"Agent coordination advisor failed: {e}")
            return f"I'm having trouble analyzing that coordination scenario. Error: {str(e)}"
    
    # Private helper methods for MCP operations
    
    async def _verify_agent_availability_mcp(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Verify agent availability using MCP protocol"""
        availability_status = {}
        
        for agent_id in agent_ids:
            try:
                # Use MCP to check agent health
                health_result = await self.mcp_client.call_skill_tool(
                    "health_check",
                    "check_agent_health",
                    {"agent_id": agent_id, "timeout": 5}
                )
                
                availability_status[agent_id] = {
                    "available": health_result.get("success", False),
                    "health_score": health_result.get("result", {}).get("health_score", 0),
                    "response_time": health_result.get("duration", float('inf'))
                }
            except Exception as e:
                availability_status[agent_id] = {
                    "available": False,
                    "error": str(e),
                    "response_time": float('inf')
                }
        
        return availability_status
    
    async def _execute_workflow_with_mcp(
        self, 
        orchestration_id: str, 
        mode: str, 
        timeout_minutes: float,
        quality_gates: bool
    ) -> Dict[str, Any]:
        """Execute workflow using MCP protocol"""
        
        session = self.coordination_sessions[orchestration_id]
        workflow_steps = session["workflow"].get("steps", [])
        
        execution_results = {
            "step_results": [],
            "mode": mode,
            "total_steps": len(workflow_steps),
            "completed_steps": 0,
            "failed_steps": 0
        }
        
        if mode == "sequential":
            for i, step in enumerate(workflow_steps):
                step_result = await self._execute_workflow_step_mcp(
                    step, session["agents"], timeout_minutes, quality_gates
                )
                execution_results["step_results"].append(step_result)
                
                if step_result.get("success"):
                    execution_results["completed_steps"] += 1
                    session["steps_completed"].append(i)
                else:
                    execution_results["failed_steps"] += 1
                    if step.get("critical", False):
                        break
        
        elif mode == "parallel":
            step_tasks = [
                self._execute_workflow_step_mcp(step, session["agents"], timeout_minutes, quality_gates)
                for step in workflow_steps
            ]
            step_results = await asyncio.gather(*step_tasks, return_exceptions=True)
            
            for i, result in enumerate(step_results):
                if isinstance(result, Exception):
                    execution_results["step_results"].append({
                        "success": False,
                        "error": str(result),
                        "step_index": i
                    })
                    execution_results["failed_steps"] += 1
                else:
                    execution_results["step_results"].append(result)
                    if result.get("success"):
                        execution_results["completed_steps"] += 1
                        session["steps_completed"].append(i)
                    else:
                        execution_results["failed_steps"] += 1
        
        return execution_results
    
    async def _execute_workflow_step_mcp(
        self, 
        step: Dict[str, Any], 
        available_agents: List[str],
        timeout_minutes: float,
        quality_gates: bool
    ) -> Dict[str, Any]:
        """Execute a single workflow step using MCP"""
        
        step_start = datetime.now().timestamp()
        
        try:
            # Select best agent for this step using MCP
            agent_selection = await self._select_best_agent_for_step_mcp(step, available_agents)
            selected_agent = agent_selection["selected_agent"]
            
            # Execute step on selected agent via MCP
            step_result = await self.mcp_client.call_skill_tool(
                selected_agent,
                step.get("skill", "execute_task"),
                {
                    "task": step.get("task"),
                    "parameters": step.get("parameters", {}),
                    "timeout": timeout_minutes * 60
                }
            )
            
            # Quality gate check if enabled
            quality_check = {"passed": True}
            if quality_gates and step_result.get("success"):
                quality_check = await self.quality_tools.validate_step_output(
                    step_output=step_result.get("result"),
                    expected_criteria=step.get("quality_criteria", {}),
                    validation_level="standard"
                )
            
            step_end = datetime.now().timestamp()
            
            return {
                "success": step_result.get("success", False) and quality_check["passed"],
                "agent_used": selected_agent,
                "agent_selection_rationale": agent_selection.get("rationale"),
                "result": step_result.get("result"),
                "quality_check": quality_check,
                "duration": step_end - step_start,
                "step_name": step.get("name", "unnamed_step")
            }
            
        except Exception as e:
            step_end = datetime.now().timestamp()
            return {
                "success": False,
                "error": str(e),
                "duration": step_end - step_start,
                "step_name": step.get("name", "unnamed_step")
            }
    
    async def _select_best_agent_for_step_mcp(
        self, 
        step: Dict[str, Any], 
        available_agents: List[str]
    ) -> Dict[str, Any]:
        """Select the best agent for a workflow step using MCP analysis"""
        
        # Analyze step requirements
        required_capabilities = step.get("required_capabilities", [])
        performance_requirements = step.get("performance_requirements", {})
        
        # Score each agent
        agent_scores = {}
        for agent_id in available_agents:
            try:
                # Get agent capabilities via MCP
                capabilities = await self._get_agent_capabilities_mcp(agent_id)
                
                # Get current performance via MCP
                performance = await self._get_agent_load_metrics_mcp(agent_id)
                
                # Calculate suitability score
                capability_score = self._calculate_capability_match(
                    capabilities, required_capabilities
                )
                performance_score = self._calculate_performance_score(
                    performance, performance_requirements
                )
                
                agent_scores[agent_id] = {
                    "total_score": (capability_score + performance_score) / 2,
                    "capability_score": capability_score,
                    "performance_score": performance_score
                }
            except Exception as e:
                agent_scores[agent_id] = {
                    "total_score": 0.0,
                    "error": str(e)
                }
        
        # Select best agent
        best_agent = max(agent_scores.keys(), key=lambda a: agent_scores[a]["total_score"])
        
        return {
            "selected_agent": best_agent,
            "scores": agent_scores,
            "rationale": f"Selected based on capability match ({agent_scores[best_agent]['capability_score']:.2f}) and performance ({agent_scores[best_agent]['performance_score']:.2f})"
        }
    
    async def _scan_agents_via_mcp(self) -> Dict[str, Any]:
        """Scan for available agents using MCP protocol"""
        # This would typically scan the network, blockchain, or service registry
        # For now, return simulated agent data
        return {
            "agent_0_data_product": {
                "capabilities": ["data_management", "product_registration"],
                "health_score": 0.95,
                "load": 0.3
            },
            "agent_1_standardization": {
                "capabilities": ["data_standardization", "validation"],
                "health_score": 0.88,
                "load": 0.6
            },
            "calculation_agent": {
                "capabilities": ["mathematical_computation", "financial_calculation"],
                "health_score": 0.92,
                "load": 0.4
            },
            "reasoning_agent": {
                "capabilities": ["logical_reasoning", "pattern_analysis"],
                "health_score": 0.89,
                "load": 0.5
            }
        }
    
    async def _get_agent_capabilities_mcp(self, agent_id: str) -> List[str]:
        """Get agent capabilities via MCP"""
        try:
            result = await self.mcp_client.access_skill_resource(
                f"agent://{agent_id}/capabilities"
            )
            return result.get("result", {}).get("capabilities", [])
        except Exception:
            return []
    
    async def _get_agent_load_metrics_mcp(self, agent_id: str) -> Dict[str, Any]:
        """Get agent load metrics via MCP"""
        try:
            result = await self.mcp_client.call_skill_tool(
                "performance_monitor",
                "get_agent_metrics",
                {"agent_id": agent_id}
            )
            return result.get("result", {})
        except Exception:
            return {"cpu_usage": 0.5, "memory_usage": 0.5, "active_tasks": 0}
    
    def _calculate_capability_match(self, agent_capabilities: List[str], required: List[str]) -> float:
        """Calculate how well agent capabilities match requirements"""
        if not required:
            return 1.0
        
        matches = sum(1 for cap in required if cap in agent_capabilities)
        return matches / len(required)
    
    def _calculate_performance_score(self, performance: Dict[str, Any], requirements: Dict[str, Any]) -> float:
        """Calculate performance score based on current metrics and requirements"""
        # Simple scoring based on load (lower is better)
        cpu_usage = performance.get("cpu_usage", 0.5)
        memory_usage = performance.get("memory_usage", 0.5)
        
        # Score is inverse of average usage
        avg_usage = (cpu_usage + memory_usage) / 2
        return max(0.1, 1.0 - avg_usage)
    
    async def _check_agent_health_via_mcp(self, agent_id: str) -> Dict[str, Any]:
        """Check agent health via MCP"""
        try:
            result = await self.mcp_client.call_skill_tool(
                "health_monitor",
                "check_health",
                {"agent_id": agent_id}
            )
            return result.get("result", {"healthy": False})
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def _summarize_agent_capabilities(self) -> Dict[str, Any]:
        """Summarize capabilities across all registered agents"""
        capabilities_count = {}
        for agent_info in self.agent_registry.values():
            for cap in agent_info.get("capabilities", []):
                capabilities_count[cap] = capabilities_count.get(cap, 0) + 1
        
        return {
            "total_capabilities": len(capabilities_count),
            "most_common": sorted(capabilities_count.items(), key=lambda x: x[1], reverse=True)[:5],
            "capabilities_distribution": capabilities_count
        }
    
    async def _get_agents_health_summary(self) -> Dict[str, Any]:
        """Get health summary of all registered agents"""
        healthy_count = 0
        total_count = len(self.agent_registry)
        
        for agent_id in self.agent_registry.keys():
            health = await self._check_agent_health_via_mcp(agent_id)
            if health.get("healthy", False):
                healthy_count += 1
        
        return {
            "total_agents": total_count,
            "healthy_agents": healthy_count,
            "health_percentage": (healthy_count / total_count * 100) if total_count > 0 else 0
        }


# Factory function for creating enhanced MCP agent manager
def create_enhanced_mcp_agent_manager(base_url: str) -> EnhancedMCPAgentManager:
    """Create and configure enhanced MCP agent manager"""
    return EnhancedMCPAgentManager(base_url)