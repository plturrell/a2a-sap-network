"""
A2A-Compliant Message Handler for Orchestrator Agent
Replaces REST endpoints with blockchain-based messaging

A2A PROTOCOL COMPLIANCE:
This handler ensures all agent communication goes through the A2A blockchain
messaging system. No direct HTTP endpoints are exposed.
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, List

from ....core.a2aTypes import A2AMessage, MessagePart, MessageRole
from ....core.secure_agent_base import SecureA2AAgent, SecureAgentConfig
from ....core.networkClient import A2ANetworkClient
from .comprehensiveOrchestratorAgentSdk import ComprehensiveOrchestratorAgentSDK
from ....storage.distributedStorage import get_distributed_storage
from ....network.agentRegistration import get_registration_service

logger = logging.getLogger(__name__)


class MockA2ANetworkClient:
    """Mock A2A Network Client for development mode"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.connected = False
        self.registered = False

    async def connect(self):
        """Mock connect method"""
        logger.info(f"Mock A2A client: connecting agent {self.agent_id}")
        self.connected = True
        return True

    async def disconnect(self):
        """Mock disconnect method"""
        logger.info(f"Mock A2A client: disconnecting agent {self.agent_id}")
        self.connected = False
        return True

    async def register_agent(self, agent_info: Dict[str, Any]):
        """Mock register agent method"""
        logger.info(f"Mock A2A client: registering agent {agent_info.get('agent_id', self.agent_id)}")
        self.registered = True
        return {"status": "success", "message": "Agent registered (mock)"}

    async def unregister_agent(self, agent_id: str):
        """Mock unregister agent method"""
        logger.info(f"Mock A2A client: unregistering agent {agent_id}")
        self.registered = False
        return {"status": "success", "message": "Agent unregistered (mock)"}

    async def is_connected(self):
        """Mock connection check"""
        return self.connected


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
            description="A2A-compliant workflow orchestration and coordination system",
            base_url=os.getenv("A2A_BASE_URL", "http://localhost:4004"),
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
                "goal_assignment_acknowledged",
                "check_goal_exists",
                "check_goal_conflicts",
                "add_goal_dependency",
                "remove_goal_dependency",
                "get_goal_dependencies",
                "validate_goal_dependencies",
                "get_dependency_graph"
            },
            enable_authentication=True,
            enable_rate_limiting=True,
            enable_input_validation=True,
            rate_limit_requests=200,
            rate_limit_window=60
        )

        super().__init__(config)

        self.agent_sdk = agent_sdk

        # Initialize A2A blockchain client (with development fallback)
        try:
            # Check if we're in development mode
            if os.getenv('A2A_DEV_MODE', '').lower() == 'true' or os.getenv('BLOCKCHAIN_DISABLED', '').lower() == 'true':
                # Development mode - create mock client
                logger.info("Running in development mode - creating mock A2A client")
                self.a2a_client = MockA2ANetworkClient(agent_id=config.agent_id)
            else:
                # Production mode - use real client
                from ....core.networkClient import A2ANetworkClient
                self.a2a_client = A2ANetworkClient(
                    agent_id=config.agent_id,
                    blockchain_client=None,  # Use None for dev mode
                    config=None
                )
        except Exception as e:
            logger.warning(f"Failed to create A2A network client: {e}")
            # Ultimate fallback - create mock client
            self.a2a_client = MockA2ANetworkClient(agent_id=config.agent_id)
            logger.info("Using mock A2A client for development")

        # Goal management storage - defer async initialization
        self.storage = None
        self.registry_service = None

        # Initialize goal management storage
        self.agent_goals = {}
        self.goal_progress = {}
        self.goal_history = {}

        # Initialize goal dependency tracking
        self.goal_dependencies = {}  # goal_id -> list of dependency goal_ids
        self.reverse_dependencies = {}  # goal_id -> list of dependent goal_ids
        self.goal_conflicts = {}  # goal_id -> list of conflicting goal_ids

        # Initialize persistent storage
        self.persistent_storage_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "..", "data", "orchestrator_goals.json"
        )

        # Load existing goals from persistent storage (deferred to avoid event loop issues)
        self._goals_loaded = False

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

            # Load dependency data
            deps_data = await self.storage.get("orchestrator:goal_dependencies")
            if deps_data:
                self.goal_dependencies = json.loads(deps_data)
                logger.info(f"Loaded {len(self.goal_dependencies)} goal dependencies")

            reverse_deps_data = await self.storage.get("orchestrator:reverse_dependencies")
            if reverse_deps_data:
                self.reverse_dependencies = json.loads(reverse_deps_data)
                logger.info(f"Loaded {len(self.reverse_dependencies)} reverse dependencies")

            # Load conflicts data
            conflicts_data = await self.storage.get("orchestrator:goal_conflicts")
            if conflicts_data:
                self.goal_conflicts = json.loads(conflicts_data)
                logger.info(f"Loaded {len(self.goal_conflicts)} goal conflicts")

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

            # Save dependencies
            await self.storage.set("orchestrator:goal_dependencies", json.dumps(self.goal_dependencies))
            await self.storage.set("orchestrator:reverse_dependencies", json.dumps(self.reverse_dependencies))

            # Save conflicts
            await self.storage.set("orchestrator:goal_conflicts", json.dumps(self.goal_conflicts))

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
                    "version": goals_data.get("version", "1.0"),
                    "dependencies": goals_data.get("dependencies", []),
                    "collaborative_agents": goals_data.get("collaborative_agents", [])
                }

                # Process dependencies if provided
                if "dependencies" in goals_data:
                    for goal_obj in goals_data.get("primary_objectives", []):
                        goal_id = goal_obj.get("goal_id")
                        if goal_id and "dependencies" in goal_obj:
                            self.goal_dependencies[goal_id] = goal_obj["dependencies"]
                            # Update reverse dependencies
                            for dep_id in goal_obj["dependencies"]:
                                if dep_id not in self.reverse_dependencies:
                                    self.reverse_dependencies[dep_id] = []
                                self.reverse_dependencies[dep_id].append(goal_id)

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

        @self.secure_handler("check_goal_exists")
        async def handle_check_goal_exists(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Check if a goal exists"""
            try:
                goal_id = data.get("goal_id")
                if not goal_id:
                    return self.create_secure_response("goal_id is required", status="error")

                # Check in all agent goals
                exists = False
                owning_agent = None
                for agent_id, goal_record in self.agent_goals.items():
                    goals = goal_record.get("goals", {}).get("primary_objectives", [])
                    for goal in goals:
                        if goal.get("goal_id") == goal_id:
                            exists = True
                            owning_agent = agent_id
                            break
                    if exists:
                        break

                result = {
                    "exists": exists,
                    "goal_id": goal_id,
                    "owning_agent": owning_agent
                }

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to check_goal_exists: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("check_goal_conflicts")
        async def handle_check_goal_conflicts(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Check for conflicts with existing goals"""
            try:
                agent_id = data.get("agent_id")
                new_goal = data.get("new_goal", {})

                if not agent_id or not new_goal:
                    return self.create_secure_response("agent_id and new_goal are required", status="error")

                conflicts = []

                # Check for resource conflicts
                if agent_id in self.agent_goals:
                    existing_goals = self.agent_goals[agent_id].get("goals", {}).get("primary_objectives", [])
                    for existing_goal in existing_goals:
                        if existing_goal.get("status") == "active":
                            # Check for overlapping KPIs
                            new_kpis = set(new_goal.get("measurable", {}).keys())
                            existing_kpis = set(existing_goal.get("measurable", {}).keys())

                            if new_kpis & existing_kpis:  # Intersection
                                conflicts.append({
                                    "goal_id": existing_goal.get("goal_id"),
                                    "conflict_type": "resource",
                                    "details": f"Overlapping KPIs: {list(new_kpis & existing_kpis)}"
                                })

                            # Check for timeline conflicts
                            new_end = new_goal.get("time_bound")
                            existing_end = existing_goal.get("time_bound")
                            if new_end == existing_end:
                                conflicts.append({
                                    "goal_id": existing_goal.get("goal_id"),
                                    "conflict_type": "timeline",
                                    "details": "Same target completion date"
                                })

                # Check for cross-agent conflicts
                if "collaborative_agents" in new_goal:
                    for collab_agent in new_goal["collaborative_agents"]:
                        if collab_agent in self.agent_goals:
                            # Check if collaborative agent has conflicting goals
                            collab_goals = self.agent_goals[collab_agent].get("goals", {}).get("primary_objectives", [])
                            for cg in collab_goals:
                                if cg.get("priority") == "critical" and new_goal.get("priority") == "critical":
                                    conflicts.append({
                                        "goal_id": cg.get("goal_id"),
                                        "conflict_type": "priority",
                                        "details": f"Both goals have critical priority for agent {collab_agent}"
                                    })

                result = {
                    "has_conflicts": len(conflicts) > 0,
                    "conflicting_goals": [c["goal_id"] for c in conflicts],
                    "conflicts": conflicts
                }

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to check_goal_conflicts: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("add_goal_dependency")
        async def handle_add_goal_dependency(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Add a dependency between goals"""
            try:
                goal_id = data.get("goal_id")
                depends_on_goal_id = data.get("depends_on_goal_id")
                dependency_type = data.get("dependency_type", "prerequisite")

                if not goal_id or not depends_on_goal_id:
                    return self.create_secure_response("goal_id and depends_on_goal_id are required", status="error")

                # Validate both goals exist
                goal_exists = await self._check_goal_exists_internal(goal_id)
                dependency_exists = await self._check_goal_exists_internal(depends_on_goal_id)

                if not goal_exists:
                    return self.create_secure_response(f"Goal {goal_id} does not exist", status="error")
                if not dependency_exists:
                    return self.create_secure_response(f"Dependency goal {depends_on_goal_id} does not exist", status="error")

                # Check for circular dependencies
                if await self._would_create_circular_dependency(goal_id, depends_on_goal_id):
                    return self.create_secure_response("Cannot add dependency: would create circular dependency", status="error")

                # Add dependency
                if goal_id not in self.goal_dependencies:
                    self.goal_dependencies[goal_id] = []

                if depends_on_goal_id not in self.goal_dependencies[goal_id]:
                    self.goal_dependencies[goal_id].append(depends_on_goal_id)

                # Update reverse dependencies
                if depends_on_goal_id not in self.reverse_dependencies:
                    self.reverse_dependencies[depends_on_goal_id] = []

                if goal_id not in self.reverse_dependencies[depends_on_goal_id]:
                    self.reverse_dependencies[depends_on_goal_id].append(goal_id)

                # Save to persistent storage
                await self._save_persistent_goals()

                result = {
                    "status": "success",
                    "goal_id": goal_id,
                    "depends_on_goal_id": depends_on_goal_id,
                    "dependency_type": dependency_type,
                    "total_dependencies": len(self.goal_dependencies.get(goal_id, []))
                }

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="add_goal_dependency",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to add_goal_dependency: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("remove_goal_dependency")
        async def handle_remove_goal_dependency(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Remove a dependency between goals"""
            try:
                goal_id = data.get("goal_id")
                depends_on_goal_id = data.get("depends_on_goal_id")

                if not goal_id or not depends_on_goal_id:
                    return self.create_secure_response("goal_id and depends_on_goal_id are required", status="error")

                # Remove dependency
                if goal_id in self.goal_dependencies and depends_on_goal_id in self.goal_dependencies[goal_id]:
                    self.goal_dependencies[goal_id].remove(depends_on_goal_id)
                    if not self.goal_dependencies[goal_id]:
                        del self.goal_dependencies[goal_id]

                # Update reverse dependencies
                if depends_on_goal_id in self.reverse_dependencies and goal_id in self.reverse_dependencies[depends_on_goal_id]:
                    self.reverse_dependencies[depends_on_goal_id].remove(goal_id)
                    if not self.reverse_dependencies[depends_on_goal_id]:
                        del self.reverse_dependencies[depends_on_goal_id]

                # Save to persistent storage
                await self._save_persistent_goals()

                result = {
                    "status": "success",
                    "goal_id": goal_id,
                    "depends_on_goal_id": depends_on_goal_id,
                    "remaining_dependencies": len(self.goal_dependencies.get(goal_id, []))
                }

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to remove_goal_dependency: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_goal_dependencies")
        async def handle_get_goal_dependencies(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Get dependencies for a goal"""
            try:
                goal_id = data.get("goal_id")
                include_transitive = data.get("include_transitive", False)

                if not goal_id:
                    return self.create_secure_response("goal_id is required", status="error")

                direct_dependencies = self.goal_dependencies.get(goal_id, [])
                dependent_goals = self.reverse_dependencies.get(goal_id, [])

                result = {
                    "goal_id": goal_id,
                    "direct_dependencies": direct_dependencies,
                    "dependent_goals": dependent_goals
                }

                if include_transitive:
                    result["transitive_dependencies"] = await self._get_transitive_dependencies(goal_id)
                    result["is_circular"] = await self._has_circular_dependency(goal_id)

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to get_goal_dependencies: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("validate_goal_dependencies")
        async def handle_validate_goal_dependencies(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Validate all goal dependencies are satisfied"""
            try:
                goal_id = data.get("goal_id")

                if not goal_id:
                    return self.create_secure_response("goal_id is required", status="error")

                dependencies = self.goal_dependencies.get(goal_id, [])
                validation_results = []
                all_satisfied = True

                for dep_id in dependencies:
                    dep_status = await self._get_goal_status(dep_id)
                    dep_progress = await self._get_goal_progress(dep_id)

                    is_satisfied = dep_status == "completed" or dep_progress >= 100.0

                    validation_results.append({
                        "dependency_goal_id": dep_id,
                        "status": dep_status,
                        "progress": dep_progress,
                        "is_satisfied": is_satisfied
                    })

                    if not is_satisfied:
                        all_satisfied = False

                result = {
                    "goal_id": goal_id,
                    "all_dependencies_satisfied": all_satisfied,
                    "dependency_count": len(dependencies),
                    "validation_results": validation_results
                }

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to validate_goal_dependencies: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_dependency_graph")
        async def handle_get_dependency_graph(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Get the full dependency graph for visualization"""
            try:
                agent_filter = data.get("agent_id")

                nodes = []
                edges = []

                # Build nodes from all goals
                for agent_id, goal_record in self.agent_goals.items():
                    if agent_filter and agent_id != agent_filter:
                        continue

                    goals = goal_record.get("goals", {}).get("primary_objectives", [])
                    for goal in goals:
                        goal_id = goal.get("goal_id")
                        if goal_id:
                            nodes.append({
                                "id": goal_id,
                                "agent_id": agent_id,
                                "label": goal.get("specific", "")[:50] + "...",
                                "status": goal_record.get("status", "active"),
                                "progress": self.goal_progress.get(agent_id, {}).get("overall_progress", 0)
                            })

                # Build edges from dependencies
                for goal_id, deps in self.goal_dependencies.items():
                    for dep_id in deps:
                        edges.append({
                            "source": goal_id,
                            "target": dep_id,
                            "type": "depends_on"
                        })

                result = {
                    "nodes": nodes,
                    "edges": edges,
                    "statistics": {
                        "total_goals": len(nodes),
                        "total_dependencies": len(edges),
                        "goals_with_dependencies": len(self.goal_dependencies),
                        "goals_as_dependencies": len(self.reverse_dependencies)
                    }
                }

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to get_dependency_graph: {e}")
                return self.create_secure_response(str(e), status="error")

    # Helper methods for dependency management
    async def _check_goal_exists_internal(self, goal_id: str) -> bool:
        """Internal method to check if a goal exists"""
        for agent_id, goal_record in self.agent_goals.items():
            goals = goal_record.get("goals", {}).get("primary_objectives", [])
            for goal in goals:
                if goal.get("goal_id") == goal_id:
                    return True
        return False

    async def _would_create_circular_dependency(self, goal_id: str, new_dependency: str) -> bool:
        """Check if adding a dependency would create a circular dependency"""
        # Use DFS to check for cycles
        visited = set()

        def has_path(current: str, target: str) -> bool:
            if current == target:
                return True
            if current in visited:
                return False
            visited.add(current)

            for dep in self.goal_dependencies.get(current, []):
                if has_path(dep, target):
                    return True
            return False

        # Check if new_dependency can reach goal_id
        return has_path(new_dependency, goal_id)

    async def _get_transitive_dependencies(self, goal_id: str) -> List[str]:
        """Get all transitive dependencies of a goal"""
        all_deps = set()
        to_process = [goal_id]

        while to_process:
            current = to_process.pop()
            for dep in self.goal_dependencies.get(current, []):
                if dep not in all_deps:
                    all_deps.add(dep)
                    to_process.append(dep)

        return list(all_deps)

    async def _has_circular_dependency(self, goal_id: str) -> bool:
        """Check if a goal has circular dependencies"""
        return await self._would_create_circular_dependency(goal_id, goal_id)

    async def _get_goal_status(self, goal_id: str) -> Optional[str]:
        """Get the status of a goal"""
        for agent_id, goal_record in self.agent_goals.items():
            goals = goal_record.get("goals", {}).get("primary_objectives", [])
            for goal in goals:
                if goal.get("goal_id") == goal_id:
                    return goal_record.get("status", "unknown")
        return None

    async def _get_goal_progress(self, goal_id: str) -> float:
        """Get the progress of a goal"""
        for agent_id, goal_record in self.agent_goals.items():
            goals = goal_record.get("goals", {}).get("primary_objectives", [])
            for goal in goals:
                if goal.get("goal_id") == goal_id:
                    return self.goal_progress.get(agent_id, {}).get("overall_progress", 0.0)
        return 0.0

    def _calculate_success_criteria_met(self, agent_id: str) -> int:
        """Calculate how many success criteria have been met"""
        if agent_id not in self.agent_goals:
            return 0

        goal_record = self.agent_goals[agent_id]
        success_criteria = goal_record.get("goals", {}).get("success_criteria", [])
        progress_data = self.goal_progress.get(agent_id, {})
        objective_progress = progress_data.get("objective_progress", {})

        met_count = 0
        for criterion in success_criteria:
            metric_name = criterion.get("metric_name")
            target_value = criterion.get("target_value")
            comparison = criterion.get("comparison", ">=")

            if metric_name in objective_progress:
                current_value = objective_progress[metric_name]

                if comparison == ">=" and current_value >= target_value:
                    met_count += 1
                elif comparison == ">" and current_value > target_value:
                    met_count += 1
                elif comparison == "<=" and current_value <= target_value:
                    met_count += 1
                elif comparison == "<" and current_value < target_value:
                    met_count += 1
                elif comparison == "==" and current_value == target_value:
                    met_count += 1

        return met_count

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

    async def initialize(self) -> None:
        """Initialize agent-specific resources"""
        logger.info(f"Initializing A2A handler for {self.config.agent_name}")

        try:
            if self.a2a_client:
                # Connect to blockchain
                await self.a2a_client.connect()

                # Register agent on blockchain
                await self.a2a_client.register_agent({
                    "agent_id": self.config.agent_id,
                    "agent_name": self.config.agent_name,
                    "capabilities": list(self.config.allowed_operations),
                    "version": self.config.agent_version
                })

                logger.info(f"A2A handler initialized and registered on blockchain")
            else:
                logger.info(f"A2A handler initialized in offline mode")
        except Exception as e:
            logger.warning(f"A2A handler initialization failed: {e}")

    async def shutdown(self) -> None:
        """Cleanup agent resources"""
        logger.info(f"Shutting down A2A handler for {self.config.agent_name}")

        if self.a2a_client:
            try:
                # Unregister from blockchain
                await self.a2a_client.unregister_agent(self.config.agent_id)
            except Exception as e:
                logger.warning(f"Failed to unregister from blockchain: {e}")

            try:
                # Disconnect
                await self.a2a_client.disconnect()
            except Exception as e:
                logger.warning(f"Failed to disconnect from blockchain: {e}")

        logger.info(f"A2A handler shutdown complete")

    async def stop(self):
        """Stop the A2A handler"""
        logger.info(f"Stopping A2A handler for {self.config.agent_name}")

        # Call shutdown instead of self.shutdown()
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
