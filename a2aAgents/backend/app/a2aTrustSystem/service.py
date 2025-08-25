"""
A2A Trust System Service
Trust relationship management service for A2A agents
"""

import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .models import (
    TrustLevel, AgentStatus, WorkflowStatus,
    TrustCommitment, TrustScore, InteractionRecord, TrustWorkflow,
    TrustAgentRegistrationRequest, TrustAgentRegistrationResponse,
    TrustScoreResponse, TrustWorkflowRequest, TrustWorkflowResponse,
    SLACreationRequest, SystemHealth, TrustMetrics
)

logger = logging.getLogger(__name__)


class TrustSystemService:
    """Trust relationship management service for A2A agents"""

    def __init__(self, registry_service=None):
        self.registry = registry_service

        # In-memory storage for trust relationships
        self.trust_agents: Dict[str, Dict] = {}
        self.trust_scores: Dict[str, TrustScore] = {}
        self.workflows: Dict[str, TrustWorkflow] = {}
        self.interactions: List[InteractionRecord] = []
        self.sla_contracts: Dict[str, Dict] = {}

        logger.info("TrustSystemService initialized")

    def _generate_trust_id(self, agent_card: Dict[str, Any]) -> str:
        """Generate deterministic trust agent ID"""
        agent_data = f"{agent_card.get('name', '')}{agent_card.get('url', '')}"
        return "0x" + hashlib.sha256(agent_data.encode()).hexdigest()

    def _generate_workflow_id(self, workflow_definition: Dict[str, Any]) -> str:
        """Generate deterministic workflow ID"""
        workflow_data = f"{workflow_definition.get('workflow_name', '')}{int(time.time())}"
        return "0x" + hashlib.sha256(workflow_data.encode()).hexdigest()

    async def register_agent_with_trust(
        self,
        request: TrustAgentRegistrationRequest
    ) -> TrustAgentRegistrationResponse:
        """Register agent with trust system"""

        logger.info(f"Registering agent with trust system: {request.agent_card.get('name')}")

        try:
            # 1. Register in A2A Registry first
            if not self.registry:
                raise ValueError("A2A Registry service is required for agent registration")

            # Create proper AgentRegistrationRequest for A2A Registry
            from app.a2a_registry.models import AgentRegistrationRequest, AgentCard

            # Convert dict to AgentCard if needed
            agent_card = request.agent_card
            if isinstance(agent_card, dict):
                agent_card = AgentCard(**agent_card)

            registry_request = AgentRegistrationRequest(
                agent_card=agent_card,
                registered_by="trust_system"
            )

            registry_result = await self.registry.register_agent(registry_request)
            registry_agent_id = registry_result.agent_id

            # 2. Generate trust identifiers
            trust_agent_id = self._generate_trust_id(request.agent_card)

            # 3. Store trust registration
            trust_registration = {
                "agent_id": trust_agent_id,
                "registry_id": registry_agent_id,
                "commitment_level": request.commitment_level,
                "registration_timestamp": int(time.time()),
                "status": AgentStatus.ACTIVE,
                "agent_card": request.agent_card
            }

            self.trust_agents[trust_agent_id] = trust_registration

            # 4. Initialize trust score
            initial_score = 1.0 if request.commitment_level == TrustLevel.LOW else 2.0
            if request.commitment_level == TrustLevel.HIGH:
                initial_score = 3.0
            elif request.commitment_level == TrustLevel.CRITICAL:
                initial_score = 4.0

            trust_score = TrustScore(
                agent_id=trust_agent_id,
                total_interactions=0,
                successful_interactions=0,
                trust_rating=initial_score,
                last_updated=datetime.utcnow()
            )
            self.trust_scores[trust_agent_id] = trust_score

            response = TrustAgentRegistrationResponse(
                success=True,
                registry_agent_id=registry_agent_id,
                trust_agent_id=trust_agent_id,
                commitment_level=request.commitment_level,
                initial_trust_score=initial_score
            )

            logger.info(f"Agent registered successfully: {trust_agent_id}")
            return response

        except Exception as e:
            logger.error(f"Failed to register agent with trust system: {e}")
            raise

    async def get_agent_trust_score(self, agent_id: str) -> TrustScoreResponse:
        """Get comprehensive trust score"""

        # Get trust score
        trust_score = self.trust_scores.get(agent_id)
        if not trust_score:
            trust_score = TrustScore(
                agent_id=agent_id,
                total_interactions=0,
                successful_interactions=0,
                trust_rating=0.0,
                last_updated=datetime.utcnow()
            )

        # Use only actual trust rating, no simulated metrics
        overall_score = trust_score.trust_rating

        return TrustScoreResponse(
            agent_id=agent_id,
            overall_trust_score=overall_score,
            trust_metrics=trust_score,
            registry_metrics={},  # No simulated metrics
            last_updated=trust_score.last_updated
        )

    async def record_interaction(
        self,
        provider_id: str,
        interaction: InteractionRecord
    ) -> bool:
        """Record agent interaction for trust scoring"""

        logger.info(f"Recording interaction: {provider_id} -> {interaction.consumer_id}")

        try:
            # Store interaction record
            self.interactions.append(interaction)

            # Update trust score
            trust_score = self.trust_scores.get(provider_id)
            if not trust_score:
                trust_score = TrustScore(agent_id=provider_id)
                self.trust_scores[provider_id] = trust_score

            # Update metrics
            trust_score.total_interactions += 1

            # Calculate new trust rating
            if trust_score.trust_rating == 0.0:
                trust_score.trust_rating = float(interaction.rating)
            else:
                trust_score.trust_rating = (
                    trust_score.trust_rating + interaction.rating
                ) / 2

            # Update skill rating
            if interaction.skill_used not in trust_score.skill_ratings:
                trust_score.skill_ratings[interaction.skill_used] = float(interaction.rating)
            else:
                trust_score.skill_ratings[interaction.skill_used] = (
                    trust_score.skill_ratings[interaction.skill_used] + interaction.rating
                ) / 2

            # Mark as successful if rating >= 4
            if interaction.rating >= 4 and not interaction.error_occurred:
                trust_score.successful_interactions += 1

            trust_score.last_updated = datetime.utcnow()

            logger.info(f"Trust score updated for {provider_id}: {trust_score.trust_rating:.2f}")
            return True

        except Exception as e:
            logger.error(f"Failed to record interaction: {e}")
            return False

    async def create_trust_workflow(
        self,
        request: TrustWorkflowRequest
    ) -> TrustWorkflowResponse:
        """Create workflow with trust verification"""

        logger.info(f"Creating trust workflow: {request.workflow_definition.get('workflow_name')}")

        try:
            # 1. Create workflow in A2A Registry
            if not self.registry:
                raise ValueError("A2A Registry service is required for workflow creation")

            # Create proper WorkflowPlanRequest for A2A Registry
            from app.a2a_registry.models import WorkflowPlanRequest, WorkflowPlanStage

            # Convert workflow definition to WorkflowPlanRequest format
            stages = []
            for stage in request.workflow_definition.get('stages', []):
                plan_stage = WorkflowPlanStage(
                    name=stage.get('name'),
                    required_capabilities=stage.get('required_capabilities', []),
                    depends_on=stage.get('depends_on')
                )
                stages.append(plan_stage)

            workflow_request = WorkflowPlanRequest(
                workflow_name=request.workflow_definition.get('workflow_name', ''),
                description=request.workflow_definition.get('description'),
                stages=stages
            )

            registry_workflow = await self.registry.create_workflow_plan(workflow_request)
            registry_workflow_id = registry_workflow.workflow_id

            # 2. Generate trust workflow ID
            trust_workflow_id = self._generate_workflow_id(request.workflow_definition)

            # 3. Create trust workflow
            workflow = TrustWorkflow(
                workflow_id=trust_workflow_id,
                initiator="trust_system",
                total_trust_required=sum(request.trust_requirements.values()),
                current_stage=0,
                status=WorkflowStatus.CREATED,
                created_at=datetime.utcnow()
            )

            self.workflows[trust_workflow_id] = workflow

            response = TrustWorkflowResponse(
                workflow_id=registry_workflow_id,
                trust_workflow_id=trust_workflow_id,
                trust_requirements=request.trust_requirements
            )

            logger.info(f"Trust workflow created: {trust_workflow_id}")
            return response

        except Exception as e:
            logger.error(f"Failed to create trust workflow: {e}")
            raise

    async def create_sla_contract(self, request: SLACreationRequest) -> str:
        """Create Service Level Agreement"""

        logger.info(f"Creating SLA: {request.provider_id} <-> {request.consumer_id}")

        try:
            sla_id = "0x" + hashlib.sha256(
                f"sla_{request.provider_id}_{request.consumer_id}_{int(time.time())}".encode()
            ).hexdigest()

            sla = {
                "sla_id": sla_id,
                "provider_id": request.provider_id,
                "consumer_id": request.consumer_id,
                "terms": request.terms.dict(),
                "valid_until": datetime.utcnow() + timedelta(hours=request.validity_hours),
                "active": True,
                "violation_count": 0,
                "creation_timestamp": datetime.utcnow()
            }

            self.sla_contracts[sla_id] = sla

            logger.info(f"SLA contract created: {sla_id}")
            return sla_id

        except Exception as e:
            logger.error(f"Failed to create SLA contract: {e}")
            raise

    async def get_system_health(self) -> SystemHealth:
        """Get trust system health status"""

        active_workflows = sum(1 for w in self.workflows.values() if w.status == WorkflowStatus.EXECUTING)

        return SystemHealth(
            status="healthy",
            total_registered_agents=len(self.trust_agents),
            active_workflows=active_workflows,
            total_trust_interactions=len(self.interactions)
        )

    async def get_trust_metrics(self, period: str = "24h") -> TrustMetrics:
        """Get trust system metrics"""

        # Calculate metrics based on period
        cutoff_time = datetime.utcnow() - timedelta(hours=24 if period == "24h" else 168)

        recent_interactions = [
            i for i in self.interactions
            if i.timestamp >= cutoff_time
        ]

        workflows_completed = sum(
            1 for w in self.workflows.values()
            if w.status == WorkflowStatus.COMPLETED and w.completion_timestamp and w.completion_timestamp >= cutoff_time
        )

        total_workflows = len([w for w in self.workflows.values() if w.created_at >= cutoff_time])

        return TrustMetrics(
            period=period,
            agent_registrations=len(self.trust_agents),
            trust_updates=len(recent_interactions),
            workflows_created=total_workflows,
            workflows_completed=workflows_completed,
            average_trust_score=sum(ts.trust_rating for ts in self.trust_scores.values()) / max(len(self.trust_scores), 1),
            workflow_success_rate=workflows_completed / max(total_workflows, 1)
        )

    async def get_trust_leaderboard(self, limit: int = 10) -> List[Dict]:
        """Get trust score leaderboard"""

        # Sort agents by trust score
        sorted_agents = sorted(
            self.trust_scores.values(),
            key=lambda x: x.trust_rating,
            reverse=True
        )[:limit]

        leaderboard = [
            {
                "agent_id": agent.agent_id,
                "trust_score": agent.trust_rating,
                "total_interactions": agent.total_interactions,
                "success_rate": agent.success_rate,
                "last_updated": agent.last_updated.isoformat()
            }
            for agent in sorted_agents
        ]

        return leaderboard
