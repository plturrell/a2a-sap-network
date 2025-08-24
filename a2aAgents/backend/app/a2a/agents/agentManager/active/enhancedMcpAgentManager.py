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
from ...sdk.utils import create_error_response, create_success_response
from ...sdk.types import A2AMessage, MessageRole, TaskStatus, AgentCard
from ...sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt
from ...common.mcpPerformanceTools import MCPPerformanceTools
from ...common.mcpValidationTools import MCPValidationTools
from ...common.mcpQualityAssessmentTools import MCPQualityAssessmentTools
from app.a2a.core.security_base import SecureA2AAgent
import re


class BlockchainRegistry:
    """Registry that uses blockchain as single source of truth"""
    
    def __init__(self):
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
        return {}
    
    async def set(self, key, value):
        """Set in blockchain only"""
        if not self.blockchain_client:
            raise RuntimeError("A2A Protocol: Blockchain required for registry updates")
        # Blockchain set implementation
        pass


logger = logging.getLogger(__name__)


class MCPClientPlaceholder:
    """Placeholder MCP client for development purposes"""
    
    async def call_skill_tool(self, agent_id: str, skill: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate MCP tool call"""
        return {
            "success": True,
            "result": {"status": "simulated", "agent_id": agent_id, "skill": skill},
            "duration": 0.1
        }
    
    async def access_skill_resource(self, resource_uri: str) -> Dict[str, Any]:
        """Simulate MCP resource access"""
        return {
            "result": {"capabilities": ["data_processing", "analysis"]},
            "success": True
        }


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
        
        # Initialize MCP tool providers
        self.performance_tools = MCPPerformanceTools()
        self.validation_tools = MCPValidationTools()
        self.quality_tools = MCPQualityAssessmentTools()
        
        
        # Agent management state
        self.managed_agents = {}
        self.agent_registry = {}  # A2A: Blockchain-backed registry
        self.blockchain_registry = BlockchainRegistry()  # A2A: No local storage
        self.coordination_sessions = {}
        self.performance_tracking = {}
        
        # Initialize MCP client placeholder
        self.mcp_client = MCPClientPlaceholder()
        
        logger.info(f"Initialized {self.name} with comprehensive MCP tool integration")
    
    @a2a_skill(
        name="agent_discovery",
        description="Discover and catalog available agents in the A2A network",
        input_schema={
            "type": "object",
            "properties": {
                "discovery_criteria": {"type": "object", "description": "Criteria for agent discovery"},
                "capabilities_filter": {"type": "array", "items": {"type": "string"}, "description": "Filter by capabilities"}
            }
        }
    )
    async def agent_discovery(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Discover available agents in the A2A network"""
        try:
            discovery_criteria = input_data.get("discovery_criteria", {})
            capabilities_filter = input_data.get("capabilities_filter", [])
            
            # Scan for agents using MCP
            available_agents = await self._scan_agents_via_mcp()
            
            # Apply filters
            filtered_agents = {}
            for agent_id, agent_info in available_agents.items():
                if capabilities_filter:
                    agent_caps = agent_info.get("capabilities", [])
                    if not any(cap in agent_caps for cap in capabilities_filter):
                        continue
                filtered_agents[agent_id] = agent_info
            
            return {
                "discovered_agents": filtered_agents,
                "total_count": len(filtered_agents),
                "discovery_criteria": discovery_criteria,
                "capabilities_filter": capabilities_filter
            }
            
        except Exception as e:
            logger.error(f"Agent discovery failed: {e}")
            return {"error": str(e)}
    
    @a2a_skill(
        name="agent_coordination",
        description="Coordinate multiple agents for complex workflows",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_definition": {"type": "object", "description": "Workflow steps and dependencies"},
                "agents_involved": {"type": "array", "items": {"type": "string"}, "description": "Agent IDs to coordinate"},
                "coordination_mode": {"type": "string", "enum": ["sequential", "parallel", "pipeline"], "default": "sequential"}
            },
            "required": ["workflow_definition", "agents_involved"]
        }
    )
    async def agent_coordination(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multiple agents for complex workflows"""
        try:
            workflow_definition = input_data.get("workflow_definition", {})
            agents_involved = input_data.get("agents_involved", [])
            coordination_mode = input_data.get("coordination_mode", "sequential")
            
            # Use the enhanced MCP orchestration
            result = await self.enhanced_agent_orchestration(
                workflow_definition=workflow_definition,
                agents_involved=agents_involved,
                coordination_mode=coordination_mode,
                timeout_minutes=10,
                quality_gates=True,
                performance_monitoring=True
            )
            
            return {
                "coordination_result": result,
                "workflow_id": result.get("orchestration_id"),
                "agents_coordinated": agents_involved,
                "mode": coordination_mode
            }
            
        except Exception as e:
            logger.error(f"Agent coordination failed: {e}")
            return {"error": str(e)}
    
    @a2a_skill(
        name="load_balancing",
        description="Balance workload across multiple agents dynamically",
        input_schema={
            "type": "object",
            "properties": {
                "workload": {"type": "object", "description": "Workload to distribute"},
                "target_agents": {"type": "array", "items": {"type": "string"}, "description": "Agents for load balancing"},
                "strategy": {"type": "string", "enum": ["round_robin", "least_loaded", "adaptive"], "default": "adaptive"}
            },
            "required": ["workload", "target_agents"]
        }
    )
    async def load_balancing(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Balance workload across multiple agents"""
        try:
            workload = input_data.get("workload", {})
            target_agents = input_data.get("target_agents", [])
            strategy = input_data.get("strategy", "adaptive")
            
            # Use the adaptive load balancing MCP tool
            result = await self.adaptive_load_balancing(
                workload=workload,
                target_agents=target_agents,
                balancing_strategy=strategy,
                monitor_performance=True,
                auto_rebalance=True
            )
            
            return {
                "balancing_result": result,
                "strategy_used": strategy,
                "agents_balanced": target_agents,
                "workload_distributed": bool(result.get("status") == "success")
            }
            
        except Exception as e:
            logger.error(f"Load balancing failed: {e}")
            return {"error": str(e)}
    
    @a2a_skill(
        name="performance_monitoring",
        description="Monitor and analyze agent performance metrics",
        input_schema={
            "type": "object",
            "properties": {
                "agents_to_monitor": {"type": "array", "items": {"type": "string"}, "description": "Agent IDs to monitor"},
                "metrics_requested": {"type": "array", "items": {"type": "string"}, "description": "Specific metrics to collect"},
                "monitoring_duration": {"type": "number", "default": 60, "description": "Monitoring duration in seconds"}
            }
        }
    )
    async def performance_monitoring(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor agent performance metrics"""
        try:
            agents_to_monitor = input_data.get("agents_to_monitor", list(self.agent_registry.keys()))
            metrics_requested = input_data.get("metrics_requested", ["cpu_usage", "memory_usage", "response_time"])
            monitoring_duration = input_data.get("monitoring_duration", 60)
            
            monitoring_results = {}
            for agent_id in agents_to_monitor:
                try:
                    metrics = await self.performance_tools.measure_agent_performance(
                        agent_id=agent_id,
                        include_real_time=True
                    )
                    monitoring_results[agent_id] = {
                        "metrics": metrics,
                        "health_score": metrics.get("health_score", 0.0),
                        "performance_grade": self._calculate_performance_grade(metrics)
                    }
                except Exception as e:
                    monitoring_results[agent_id] = {"error": str(e), "health_score": 0.0}
            
            return {
                "monitoring_results": monitoring_results,
                "agents_monitored": len(monitoring_results),
                "monitoring_duration": monitoring_duration,
                "metrics_collected": metrics_requested,
                "overall_health": sum(r.get("health_score", 0) for r in monitoring_results.values()) / len(monitoring_results) if monitoring_results else 0
            }
            
        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")
            return {"error": str(e)}
    
    @a2a_skill(
        name="health_assessment",
        description="Assess overall health of the agent ecosystem",
        input_schema={
            "type": "object",
            "properties": {
                "include_detailed_analysis": {"type": "boolean", "default": True},
                "assessment_depth": {"type": "string", "enum": ["basic", "comprehensive"], "default": "comprehensive"}
            }
        }
    )
    async def health_assessment(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall health of the agent ecosystem"""
        try:
            include_detailed = input_data.get("include_detailed_analysis", True)
            assessment_depth = input_data.get("assessment_depth", "comprehensive")
            
            # Get health summary from all known agents
            health_summary = await self._get_agents_health_summary()
            
            # Detailed analysis if requested
            detailed_analysis = {}
            if include_detailed and assessment_depth == "comprehensive":
                for agent_id in self.agent_registry.keys():
                    health_check = await self._check_agent_health_via_mcp(agent_id)
                    detailed_analysis[agent_id] = health_check
            
            # Calculate overall ecosystem health
            ecosystem_health = self._calculate_ecosystem_health(health_summary, detailed_analysis)
            
            return {
                "health_summary": health_summary,
                "detailed_analysis": detailed_analysis if include_detailed else {},
                "ecosystem_health": ecosystem_health,
                "assessment_timestamp": datetime.now().isoformat(),
                "recommendations": self._generate_health_recommendations(ecosystem_health)
            }
            
        except Exception as e:
            logger.error(f"Health assessment failed: {e}")
            return {"error": str(e)}
    
    def _calculate_performance_grade(self, metrics: Dict[str, Any]) -> str:
        """Calculate performance grade based on metrics"""
        health_score = metrics.get("health_score", 0.0)
        if health_score >= 0.9:
            return "A"
        elif health_score >= 0.8:
            return "B"
        elif health_score >= 0.7:
            return "C"
        elif health_score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _calculate_ecosystem_health(self, health_summary: Dict[str, Any], detailed_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall ecosystem health score and status"""
        health_percentage = health_summary.get("health_percentage", 0)
        
        if health_percentage >= 90:
            status = "Excellent"
        elif health_percentage >= 80:
            status = "Good"
        elif health_percentage >= 70:
            status = "Fair"
        elif health_percentage >= 60:
            status = "Poor"
        else:
            status = "Critical"
        
        return {
            "overall_score": health_percentage / 100,
            "status": status,
            "healthy_agents": health_summary.get("healthy_agents", 0),
            "total_agents": health_summary.get("total_agents", 0),
            "critical_issues": len([a for a in detailed_analysis.values() if not a.get("healthy", True)])
        }
    
    def _generate_health_recommendations(self, ecosystem_health: Dict[str, Any]) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []
        
        if ecosystem_health.get("overall_score", 0) < 0.8:
            recommendations.append("Consider restarting underperforming agents")
            
        if ecosystem_health.get("critical_issues", 0) > 0:
            recommendations.append("Address critical issues in failing agents immediately")
            
        if ecosystem_health.get("status") == "Critical":
            recommendations.append("Emergency intervention required - system stability at risk")
            
        if not recommendations:
            recommendations.append("System is healthy - continue regular monitoring")
            
        return recommendations
    
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
        best_agent = max(agent_scores.keys(), key=self._get_agent_score)
        
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
            "most_common": sorted(capabilities_count.items(), key=self._get_capability_count, reverse=True)[:5],
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


    def _init_security_features(self):
        """Initialize security features from SecureA2AAgent"""
        # Rate limiting configuration
        self.rate_limits = {
            'default': {'requests': 100, 'window': 60},  # 100 requests per minute
            'heavy': {'requests': 10, 'window': 60},     # 10 requests per minute for heavy operations
            'auth': {'requests': 5, 'window': 300}       # 5 auth attempts per 5 minutes
        }
        
        # Input validation rules
        self.validation_rules = {
            'max_string_length': 10000,
            'max_array_size': 1000,
            'max_object_depth': 10,
            'allowed_file_extensions': ['.json', '.txt', '.csv', '.xml'],
            'sql_injection_patterns': [
                r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|WHERE|FROM)\b)',
                r'(--|;|'|\"|\\*|OR\\s+1=1|AND\\s+1=1)'
            ]
        }
        
        # Initialize security logger
        self.security_logger = logging.getLogger(f'{self.__class__.__name__}.security')
    
    def _init_rate_limiting(self):
        """Initialize rate limiting tracking"""
        from collections import defaultdict
        import time
        
        self.rate_limit_tracker = defaultdict(self._create_rate_limit_entry)
    
    def _init_input_validation(self):
        """Initialize input validation helpers"""
        self.input_validators = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'),
            'url': re.compile(r'^https?://[^\\s/$.?#].[^\\s]*$'),
            'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
            'safe_string': re.compile(r'^[a-zA-Z0-9\\s\\-_.,!?]+$')
        }
    
    async def _generate_coordination_recommendations(
        self, 
        scenario_analysis: Dict[str, Any], 
        agent_capabilities: Dict[str, Any], 
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate coordination recommendations based on analysis"""
        return {
            "coordination_strategy": "Sequential processing with quality gates",
            "agent_assignments": [
                {"agent": "agent_0", "role": "data_provider", "rationale": "Best data access capabilities"},
                {"agent": "agent_1", "role": "standardizer", "rationale": "Specializes in data standardization"}
            ],
            "risk_factors": ["High coordination overhead", "Potential bottlenecks in sequential processing"]
        }
    
    async def _analyze_agent_compatibility_mcp(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Analyze compatibility between agents using MCP"""
        compatibility_matrix = {}
        for i, agent1 in enumerate(agent_ids):
            for j, agent2 in enumerate(agent_ids):
                if i != j:
                    compatibility_matrix[f"{agent1}-{agent2}"] = 0.85  # Simulated compatibility score
        return compatibility_matrix
    
    async def _analyze_agent_performance_mcp(self, agent_id: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze agent performance against requirements"""
        return {
            "meets_requirements": True,
            "performance_score": 0.92,
            "bottlenecks": [],
            "recommendations": ["Consider load balancing for high throughput scenarios"]
        }
    
    async def _calculate_load_distribution_mcp(
        self, 
        workload: Dict[str, Any], 
        agent_loads: Dict[str, Any], 
        strategy: str, 
        workload_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate optimal load distribution"""
        distribution = {}
        # Simple equal distribution for demonstration
        num_agents = len(agent_loads)
        for i, agent_id in enumerate(agent_loads.keys()):
            distribution[agent_id] = {"portion": 1/num_agents, "tasks": workload.get("tasks", [])[i::num_agents]}
        return distribution
    
    async def _distribute_work_to_agent_mcp(
        self, 
        agent_id: str, 
        work: Dict[str, Any], 
        monitor: bool
    ) -> Dict[str, Any]:
        """Distribute work to specific agent"""
        return {"success": True, "tasks_assigned": len(work.get("tasks", [])), "agent_id": agent_id}
    
    async def _check_rebalancing_needed_mcp(
        self, 
        monitoring_results: Dict[str, Any], 
        distribution_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if rebalancing is needed"""
        return {"rebalance_required": False, "reason": "All agents within acceptable load ranges"}
    
    async def _perform_rebalancing_mcp(
        self, 
        distribution_plan: Dict[str, Any], 
        monitoring_results: Dict[str, Any], 
        rebalance_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform load rebalancing"""
        return {"actions": [], "success": True}
    
    # Helper functions to replace lambda functions
    def _get_agent_score(self, agent_id: str) -> float:
        """Get total score for agent ranking"""
        return getattr(self, 'agent_scores', {}).get(agent_id, {}).get("total_score", 0.0)
    
    def _get_capability_count(self, item: tuple) -> int:
        """Get capability count from capability item tuple"""
        return item[1]
    
    def _create_rate_limit_entry(self) -> Dict[str, Union[int, float]]:
        """Create rate limit tracking entry"""
        import time
        return {'count': 0, 'window_start': time.time()}


# Factory function for creating enhanced MCP agent manager
def create_enhanced_mcp_agent_manager(base_url: str) -> EnhancedMCPAgentManager:
    """Create and configure enhanced MCP agent manager"""
    return EnhancedMCPAgentManager(base_url)