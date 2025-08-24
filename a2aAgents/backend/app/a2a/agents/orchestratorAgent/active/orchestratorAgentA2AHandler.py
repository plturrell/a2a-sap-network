"""
A2A-Compliant Message Handler for Orchestrator Agent
Replaces REST endpoints with blockchain-based messaging

A2A PROTOCOL COMPLIANCE:
This handler ensures all agent communication goes through the A2A blockchain
messaging system. No direct HTTP endpoints are exposed.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from ....core.a2aTypes import A2AMessage, MessagePart, MessageRole
from ....core.secure_agent_base import SecureA2AAgent, SecureAgentConfig
from ....sdk.a2aNetworkClient import A2ANetworkClient
from .comprehensiveOrchestratorAgentSdk import ComprehensiveOrchestratorAgentSDK
from ....storage.distributedStorage import get_distributed_storage
from ....network.agentRegistration import get_registration_service

logger = logging.getLogger(__name__)


class OrchestratorAgentA2AHandler(SecureA2AAgent):
    """
    A2A-compliant handler for Orchestrator Agent
    All communication through blockchain messaging only
    """
    
    def __init__(self, agent_sdk: ComprehensiveOrchestratorAgentSDK):
        """Initialize A2A handler with agent SDK"""
        # Configure secure agent
        config = SecureAgentConfig(
            agent_id="orchestrator_agent",
            agent_name="Orchestrator Agent",
            agent_version="1.0.0",
            allowed_operations={
                "orchestrator_info",
                "workflow_orchestration",
                "task_scheduling",
                "pipeline_management",
                "coordination_services",
                "execution_monitoring",
                "create_workflow",
                "execute_workflow",
                "monitor_workflow",
                "schedule_tasks",
                "coordinate_agents",
                "manage_pipelines",
                "optimize_execution",
                "handle_failures",
                "resource_allocation",
                "load_balancing",
                "health_check",
                "set_agent_goals",
                "get_agent_goals",
                "track_goal_progress",
                "update_goal_status",
                "get_goal_analytics",
                "goal_assignment_acknowledged"
            },
            enable_authentication=True,
            enable_rate_limiting=True,
            enable_input_validation=True,
            rate_limit_requests=200,
            rate_limit_window=60
        )
        
        super().__init__(config)
        
        self.agent_sdk = agent_sdk
        
        # Initialize A2A blockchain client
        self.a2a_client = A2ANetworkClient(
            agent_id=config.agent_id,
            private_key=os.getenv('A2A_PRIVATE_KEY'),
            rpc_url=os.getenv('A2A_RPC_URL', 'http://localhost:8545')
        )
        
        # Goal management storage - now with persistent backend
        self.storage = get_distributed_storage()
        self.registry_service = get_registration_service()
        
        # Initialize goal management storage
        self.agent_goals = {}
        self.goal_progress = {}
        self.goal_history = {}
        
        # Initialize persistent storage
        self.persistent_storage_path = os.path.join(
            os.path.dirname(__file__), 
            "..", "..", "..", "..", "data", "orchestrator_goals.json"
        )
        
        # Load existing goals from persistent storage
        asyncio.create_task(self._load_persistent_goals())
        
        # Initialize AI integration (will be done after handler registration)
        self.ai_integration_pending = True
        
        # Register message handlers
        self._register_handlers()
        
        # Initialize AI integration after handlers are registered
        asyncio.create_task(self._initialize_ai_integration())
        
        logger.info(f"A2A-compliant handler initialized for {config.agent_name} with persistent storage")
    
    async def _load_persistent_goals(self):
        """Load existing goals from persistent storage"""
        try:
            # Load goals from distributed storage
            goals_data = await self.storage.get("orchestrator:agent_goals")
            if goals_data:
                self.agent_goals = json.loads(goals_data)
                logger.info(f"Loaded {len(self.agent_goals)} agent goals from persistent storage")
            
            # Load progress data
            progress_data = await self.storage.get("orchestrator:goal_progress")
            if progress_data:
                self.goal_progress = json.loads(progress_data)
                logger.info(f"Loaded progress data for {len(self.goal_progress)} agents")
            
            # Load history data
            history_data = await self.storage.get("orchestrator:goal_history")
            if history_data:
                self.goal_history = json.loads(history_data)
                logger.info(f"Loaded goal history for {len(self.goal_history)} agents")
                
        except Exception as e:
            logger.warning(f"Failed to load persistent goals: {e}")
    
    async def _save_persistent_goals(self):
        """Save goals to persistent storage"""
        try:
            # Save goals
            await self.storage.set("orchestrator:agent_goals", json.dumps(self.agent_goals))
            
            # Save progress
            await self.storage.set("orchestrator:goal_progress", json.dumps(self.goal_progress))
            
            # Save history
            await self.storage.set("orchestrator:goal_history", json.dumps(self.goal_history))
            
            logger.debug("Saved goal data to persistent storage")
        except Exception as e:
            logger.error(f"Failed to save persistent goals: {e}")
    
    async def _update_agent_registry_metadata(self, agent_id: str, goal_summary: Dict[str, Any]):
        """Update agent registry with goal metadata"""
        try:
            # Get current agent record
            agent_record = await self.registry_service.get_agent(agent_id)
            if not agent_record:
                logger.warning(f"Agent {agent_id} not found in registry")
                return
            
            # Update metadata with goal information
            if not agent_record.agent_card.metadata:
                agent_record.agent_card.metadata = {}
            
            agent_record.agent_card.metadata["goals"] = {
                "has_goals": True,
                "goal_status": goal_summary.get("status", "active"),
                "overall_progress": goal_summary.get("overall_progress", 0.0),
                "objectives_count": goal_summary.get("objectives_count", 0),
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Update the registry
            await self.registry_service.update_agent(agent_id, agent_record.agent_card)
            logger.info(f"Updated registry metadata for agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to update registry metadata for {agent_id}: {e}")
    
    def _register_handlers(self):
        """Register A2A message handlers"""

        @self.secure_handler("orchestrator_info")
        async def handle_orchestrator_info(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle orchestrator_info operation"""
            try:
                # Get orchestrator agent information and capabilities
                result = await self.agent_sdk.get_agent_info(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="orchestrator_info",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to orchestrator_info: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("workflow_orchestration")
        async def handle_workflow_orchestration(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle workflow_orchestration operation"""
            try:
                # Orchestrate complex workflows across agents
                result = await self.agent_sdk.workflow_orchestration(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="workflow_orchestration",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to workflow_orchestration: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("task_scheduling")
        async def handle_task_scheduling(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle task_scheduling operation"""
            try:
                # Schedule tasks across the agent network
                result = await self.agent_sdk.task_scheduling(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="task_scheduling",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to task_scheduling: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("pipeline_management")
        async def handle_pipeline_management(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle pipeline_management operation"""
            try:
                # Manage data processing pipelines
                result = await self.agent_sdk.pipeline_management(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="pipeline_management",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to pipeline_management: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("coordination_services")
        async def handle_coordination_services(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle coordination_services operation"""
            try:
                # Coordinate services across agents
                result = await self.agent_sdk.coordination_services(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="coordination_services",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to coordination_services: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("execution_monitoring")
        async def handle_execution_monitoring(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle execution_monitoring operation"""
            try:
                # Monitor workflow and task execution
                result = await self.agent_sdk.execution_monitoring(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="execution_monitoring",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to execution_monitoring: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("create_workflow")
        async def handle_create_workflow(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle create_workflow operation"""
            try:
                # Create new workflow definitions
                result = await self.agent_sdk.create_workflow(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="create_workflow",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to create_workflow: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("execute_workflow")
        async def handle_execute_workflow(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle execute_workflow operation"""
            try:
                # Execute defined workflows
                result = await self.agent_sdk.execute_workflow(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="execute_workflow",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to execute_workflow: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("monitor_workflow")
        async def handle_monitor_workflow(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle monitor_workflow operation"""
            try:
                # Monitor workflow execution status
                result = await self.agent_sdk.monitor_workflow(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="monitor_workflow",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to monitor_workflow: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("schedule_tasks")
        async def handle_schedule_tasks(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle schedule_tasks operation"""
            try:
                # Schedule tasks with timing and dependencies
                result = await self.agent_sdk.schedule_tasks(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="schedule_tasks",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to schedule_tasks: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("coordinate_agents")
        async def handle_coordinate_agents(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle coordinate_agents operation"""
            try:
                # Coordinate multiple agents for complex tasks
                result = await self.agent_sdk.coordinate_agents(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="coordinate_agents",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to coordinate_agents: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("manage_pipelines")
        async def handle_manage_pipelines(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle manage_pipelines operation"""
            try:
                # Manage data processing pipelines
                result = await self.agent_sdk.manage_pipelines(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="manage_pipelines",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to manage_pipelines: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("optimize_execution")
        async def handle_optimize_execution(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle optimize_execution operation"""
            try:
                # Optimize workflow and task execution
                result = await self.agent_sdk.optimize_execution(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="optimize_execution",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to optimize_execution: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("handle_failures")
        async def handle_handle_failures(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle handle_failures operation"""
            try:
                # Handle and recover from workflow failures
                result = await self.agent_sdk.handle_failures(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="handle_failures",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to handle_failures: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("resource_allocation")
        async def handle_resource_allocation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle resource_allocation operation"""
            try:
                # Allocate resources across agents and tasks
                result = await self.agent_sdk.resource_allocation(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="resource_allocation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to resource_allocation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("load_balancing")
        async def handle_load_balancing(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle load_balancing operation"""
            try:
                # Balance load across agent network
                result = await self.agent_sdk.load_balancing(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="load_balancing",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to load_balancing: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("health_check")
        async def handle_health_check(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Health check for agent"""
            try:
                health_status = {
                    "status": "healthy",
                    "agent": self.config.agent_name,
                    "version": self.config.agent_version,
                    "timestamp": datetime.utcnow().isoformat(),
                    "a2a_compliant": True,
                    "blockchain_connected": await self._check_blockchain_connection(),
                    "orchestration_active": True
                }
                result = health_status
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="health_check",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to health_check: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("set_agent_goals")
        async def handle_set_agent_goals(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Set goals for a specific agent"""
            try:
                agent_id = data.get("agent_id")
                goals_data = data.get("goals", {})
                
                if not agent_id:
                    return self.create_secure_response("agent_id is required", status="error")
                
                # Validate goals structure
                required_fields = ["primary_objectives", "success_criteria", "purpose_statement"]
                for field in required_fields:
                    if field not in goals_data:
                        return self.create_secure_response(f"Missing required field: {field}", status="error")
                
                # Store goals with metadata
                goal_record = {
                    "agent_id": agent_id,
                    "goals": goals_data,
                    "created_at": datetime.utcnow().isoformat(),
                    "created_by": message.sender_id,
                    "status": "active",
                    "version": goals_data.get("version", "1.0")
                }
                
                self.agent_goals[agent_id] = goal_record
                
                # Initialize progress tracking
                self.goal_progress[agent_id] = {
                    "overall_progress": 0.0,
                    "objective_progress": {},
                    "last_updated": datetime.utcnow().isoformat(),
                    "milestones_achieved": []
                }
                
                # Initialize history
                if agent_id not in self.goal_history:
                    self.goal_history[agent_id] = []
                
                self.goal_history[agent_id].append({
                    "action": "goals_set",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {"goals_count": len(goals_data.get("primary_objectives", []))}
                })
                
                # Save to persistent storage
                await self._save_persistent_goals()
                
                # Update agent registry metadata
                goal_summary = {
                    "status": "active",
                    "overall_progress": 0.0,
                    "objectives_count": len(goals_data.get("primary_objectives", []))
                }
                await self._update_agent_registry_metadata(agent_id, goal_summary)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="set_agent_goals",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data({"agent_id": agent_id, "status": "success"}),
                    context_id=context_id
                )
                
                result = {
                    "status": "success",
                    "message": f"Goals set successfully for agent {agent_id}",
                    "data": {
                        "agent_id": agent_id,
                        "goals_count": len(goals_data.get("primary_objectives", [])),
                        "kpis_count": len(goals_data.get("kpis", []))
                    }
                }
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to set_agent_goals: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_agent_goals")
        async def handle_get_agent_goals(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Get goals for a specific agent or all agents"""
            try:
                agent_id = data.get("agent_id")
                include_progress = data.get("include_progress", True)
                include_history = data.get("include_history", False)
                
                if agent_id:
                    # Get specific agent goals
                    if agent_id not in self.agent_goals:
                        return self.create_secure_response(f"No goals found for agent: {agent_id}", status="error")
                    
                    result = {
                        "agent_id": agent_id,
                        "goals": self.agent_goals[agent_id]
                    }
                    
                    if include_progress and agent_id in self.goal_progress:
                        result["progress"] = self.goal_progress[agent_id]
                    
                    if include_history and agent_id in self.goal_history:
                        result["history"] = self.goal_history[agent_id]
                        
                else:
                    # Get all agent goals
                    result = {
                        "total_agents": len(self.agent_goals),
                        "agents": {}
                    }
                    
                    for aid, goals in self.agent_goals.items():
                        agent_data = {"goals": goals}
                        
                        if include_progress and aid in self.goal_progress:
                            agent_data["progress"] = self.goal_progress[aid]
                        
                        if include_history and aid in self.goal_history:
                            agent_data["history"] = self.goal_history[aid]
                        
                        result["agents"][aid] = agent_data
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_agent_goals",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to get_agent_goals: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("track_goal_progress")
        async def handle_track_goal_progress(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Update goal progress for an agent"""
            try:
                agent_id = data.get("agent_id")
                progress_data = data.get("progress", {})
                
                if not agent_id:
                    return self.create_secure_response("agent_id is required", status="error")
                
                if agent_id not in self.agent_goals:
                    return self.create_secure_response(f"No goals found for agent: {agent_id}", status="error")
                
                # Update progress
                if agent_id not in self.goal_progress:
                    self.goal_progress[agent_id] = {
                        "overall_progress": 0.0,
                        "objective_progress": {},
                        "last_updated": datetime.utcnow().isoformat(),
                        "milestones_achieved": []
                    }
                
                current_progress = self.goal_progress[agent_id]
                
                # Update fields
                if "overall_progress" in progress_data:
                    current_progress["overall_progress"] = progress_data["overall_progress"]
                
                if "objective_progress" in progress_data:
                    current_progress["objective_progress"].update(progress_data["objective_progress"])
                
                if "milestones_achieved" in progress_data:
                    for milestone in progress_data["milestones_achieved"]:
                        if milestone not in current_progress["milestones_achieved"]:
                            current_progress["milestones_achieved"].append(milestone)
                
                current_progress["last_updated"] = datetime.utcnow().isoformat()
                
                # Add to history
                self.goal_history[agent_id].append({
                    "action": "progress_updated",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": progress_data
                })
                
                # Save to persistent storage
                await self._save_persistent_goals()
                
                # Update agent registry metadata
                goal_summary = {
                    "status": self.agent_goals[agent_id]["status"],
                    "overall_progress": current_progress["overall_progress"],
                    "objectives_count": len(self.agent_goals[agent_id]["goals"].get("primary_objectives", []))
                }
                await self._update_agent_registry_metadata(agent_id, goal_summary)
                
                result = {
                    "status": "success",
                    "agent_id": agent_id,
                    "progress_updated": True,
                    "current_progress": current_progress,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="track_goal_progress",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to track_goal_progress: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("update_goal_status")
        async def handle_update_goal_status(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Update goal status (active, paused, completed, cancelled)"""
            try:
                agent_id = data.get("agent_id")
                new_status = data.get("status")
                reason = data.get("reason", "")
                
                if not agent_id or not new_status:
                    return self.create_secure_response("agent_id and status are required", status="error")
                
                if agent_id not in self.agent_goals:
                    return self.create_secure_response(f"No goals found for agent: {agent_id}", status="error")
                
                valid_statuses = ["active", "paused", "completed", "cancelled"]
                if new_status not in valid_statuses:
                    return self.create_secure_response(f"Invalid status. Must be one of: {valid_statuses}", status="error")
                
                # Update status
                old_status = self.agent_goals[agent_id]["status"]
                self.agent_goals[agent_id]["status"] = new_status
                self.agent_goals[agent_id]["status_updated_at"] = datetime.utcnow().isoformat()
                
                # Add to history
                self.goal_history[agent_id].append({
                    "action": "status_updated",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "old_status": old_status,
                        "new_status": new_status,
                        "reason": reason
                    }
                })
                
                result = {
                    "status": "success",
                    "agent_id": agent_id,
                    "old_status": old_status,
                    "new_status": new_status,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="update_goal_status",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to update_goal_status: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_goal_analytics")
        async def handle_get_goal_analytics(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Get analytics for agent goals"""
            try:
                agent_id = data.get("agent_id")
                
                if agent_id and agent_id in self.agent_goals:
                    # Agent-specific analytics
                    goal_record = self.agent_goals[agent_id]
                    progress_data = self.goal_progress.get(agent_id, {})
                    
                    analytics = {
                        "agent_id": agent_id,
                        "total_goals": len(goal_record.get("goals", {}).get("primary_objectives", [])),
                        "overall_progress": progress_data.get("overall_progress", 0.0),
                        "goals_created": goal_record.get("created_at"),
                        "last_updated": progress_data.get("last_updated"),
                        "milestones_achieved": len(progress_data.get("milestones_achieved", [])),
                        "objective_progress": progress_data.get("objective_progress", {}),
                        "success_criteria_met": self._calculate_success_criteria_met(agent_id)
                    }
                else:
                    # System-wide analytics
                    total_agents = len(self.agent_goals)
                    total_progress = sum(
                        self.goal_progress.get(aid, {}).get("overall_progress", 0.0)
                        for aid in self.agent_goals.keys()
                    )
                    avg_progress = total_progress / max(total_agents, 1)
                    
                    analytics = {
                        "system_analytics": {
                            "total_agents_with_goals": total_agents,
                            "average_progress": avg_progress,
                            "total_milestones": sum(
                                len(self.goal_progress.get(aid, {}).get("milestones_achieved", []))
                                for aid in self.agent_goals.keys()
                            ),
                            "agents_above_50_percent": len([
                                aid for aid in self.agent_goals.keys()
                                if self.goal_progress.get(aid, {}).get("overall_progress", 0.0) > 50.0
                            ])
                        }
                    }
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_goal_analytics",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(analytics),
                    context_id=context_id
                )
                
                return self.create_secure_response(analytics)
                
            except Exception as e:
                logger.error(f"Failed to get_goal_analytics: {e}")
                return self.create_secure_response(str(e), status="error")
    
    async def _initialize_ai_integration(self):
        """Initialize AI integration for enhanced goal management"""
        try:
            from .aiGoalIntegration import AIGoalIntegrationMixin
            
            # Mix in AI capabilities
            self.__class__ = type(
                self.__class__.__name__ + 'WithAI',
                (self.__class__, AIGoalIntegrationMixin),
                {}
            )
            
            # Initialize AI integration
            AIGoalIntegrationMixin.__init__(self)
            await self.initialize_ai_integration()
            
            # Register AI handlers
            await self._register_ai_handlers()
            
            logger.info("AI goal integration successfully initialized")
            self.ai_integration_pending = False
            
        except Exception as e:
            logger.error(f"Failed to initialize AI integration: {e}")
            self.ai_integration_pending = False
    
    async def process_a2a_message(self, message: A2AMessage) -> Dict[str, Any]:
        """
        Main entry point for A2A messages
        Routes messages to appropriate handlers based on operation
        """
        try:
            # Extract operation from message
            operation = None
            data = {}
            
            if message.parts and len(message.parts) > 0:
                part = message.parts[0]
                if part.data:
                    operation = part.data.get("operation")
                    data = part.data.get("data", {})
            
            if not operation:
                return self.create_secure_response(
                    "No operation specified in message",
                    status="error"
                )
            
            # Get handler for operation
            handler = self.handlers.get(operation)
            if not handler:
                return self.create_secure_response(
                    f"Unknown operation: {operation}",
                    status="error"
                )
            
            # Create context ID
            context_id = f"{message.sender_id}:{operation}:{datetime.utcnow().timestamp()}"
            
            # Process through handler
            return await handler(message, context_id, data)
            
        except Exception as e:
            logger.error(f"Failed to process A2A message: {e}")
            return self.create_secure_response(str(e), status="error")
    
    async def _log_blockchain_transaction(self, operation: str, data_hash: str, result_hash: str, context_id: str):
        """Log transaction to blockchain for audit trail"""
        try:
            transaction_data = {
                "agent_id": self.config.agent_id,
                "operation": operation,
                "data_hash": data_hash,
                "result_hash": result_hash,
                "context_id": context_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send to blockchain through A2A client
            await self.a2a_client.log_transaction(transaction_data)
            
        except Exception as e:
            logger.error(f"Failed to log blockchain transaction: {e}")
    
    def _hash_data(self, data: Any) -> str:
        """Create hash of data for blockchain logging"""
        import hashlib
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    async def _check_blockchain_connection(self) -> bool:
        """Check if blockchain connection is active"""
        try:
            return await self.a2a_client.is_connected()
        except Exception:
            return False
    
    async def start(self):
        """Start the A2A handler"""
        logger.info(f"Starting A2A handler for {self.config.agent_name}")
        
        # Connect to blockchain
        await self.a2a_client.connect()
        
        # Register agent on blockchain
        await self.a2a_client.register_agent({
            "agent_id": self.config.agent_id,
            "agent_name": self.config.agent_name,
            "capabilities": list(self.config.allowed_operations),
            "version": self.config.agent_version
        })
        
        logger.info(f"A2A handler started and registered on blockchain")
    
    async def stop(self):
        """Stop the A2A handler"""
        logger.info(f"Stopping A2A handler for {self.config.agent_name}")
        
        # Unregister from blockchain
        await self.a2a_client.unregister_agent(self.config.agent_id)
        
        # Disconnect
        await self.a2a_client.disconnect()
        
        # Parent cleanup
        await self.shutdown()
        
        logger.info(f"A2A handler stopped")


# Factory function to create A2A handler
def create_orchestrator_agent_a2a_handler(agent_sdk: ComprehensiveOrchestratorAgentSDK) -> OrchestratorAgentA2AHandler:
    """Create A2A-compliant handler for Orchestrator Agent"""
    return OrchestratorAgentA2AHandler(agent_sdk)


# Example usage for migration
"""
To migrate from REST endpoints to A2A messaging:

1. Replace router initialization:
   # OLD: router = APIRouter(...)
   # NEW: 
   handler = create_orchestrator_agent_a2a_handler(orchestrator_agent_sdk)

2. Replace FastAPI app with A2A listener:
   # OLD: app.include_router(router)
   # NEW:
   await handler.start()
   
3. Process messages through A2A:
   # Messages arrive through blockchain
   result = await handler.process_a2a_message(a2a_message)
"""