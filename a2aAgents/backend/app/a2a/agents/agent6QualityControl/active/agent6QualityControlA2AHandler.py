"""
A2A-Compliant Message Handler for Agent 6 - Quality Control Manager
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
from typing import Any, Dict, Optional

from ....core.a2aTypes import A2AMessage, MessagePart, MessageRole
from ....core.secure_agent_base import SecureA2AAgent, SecureAgentConfig
from ....sdk.a2aNetworkClient import A2ANetworkClient
from .comprehensiveQualityControlSdk import ComprehensiveQualityControlSDK

logger = logging.getLogger(__name__)


class Agent6QualityControlA2AHandler(SecureA2AAgent):
    """
    A2A-compliant handler for Agent 6 - Quality Control Manager
    All communication through blockchain messaging only
    """

    def __init__(self, agent_sdk: ComprehensiveQualityControlSDK):
        """Initialize A2A handler with agent SDK"""
        # Configure secure agent
        config = SecureAgentConfig(
            agent_id="quality_control_manager",
            agent_name="Quality Control Manager Agent",
            agent_version="2.0.0",
            allowed_operations={
                # Registry capabilities
                "quality_assessment",
                "routing_decision",
                "improvement_recommendations",
                "workflow_control",
                "trust_verification",
                # Enhanced operations
                "assess_quality",
                "make_routing_decision",
                "generate_recommendations",
                "control_workflow",
                "verify_trust",
                "detect_anomalies",
                "monitor_trends",
                "continuous_improvement",
                "compliance_audit",
                # Base operations
                "health_check",
                "get_supported_formats"
            },
            enable_authentication=True,
            enable_rate_limiting=True,
            enable_input_validation=True,
            rate_limit_requests=100,
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

        # Register message handlers
        self._register_handlers()

        logger.info(f"A2A-compliant handler initialized for {config.agent_name}")

    def _register_handlers(self):
        """Register A2A message handlers"""

        @self.secure_handler("get_supported_formats")
        async def handle_get_supported_formats(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_supported_formats operation"""
            try:
                result = {
                    "supported_formats": [
                        "json", "xml", "yaml", "csv", "metrics"
                    ],
                    "quality_dimensions": [
                        "accuracy", "reliability", "performance", "security",
                        "usability", "maintainability", "compliance", "robustness"
                    ],
                    "quality_standards": [
                        "iso_9001", "six_sigma", "lean", "cmmi", "itil"
                    ]
                }

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_supported_formats",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to get_supported_formats: {e}")
                return self.create_secure_response(str(e), status="error")

        # Registry capability handlers
        @self.secure_handler("quality_assessment")
        async def handle_quality_assessment(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle comprehensive quality assessment"""
            try:
                result = await self.agent_sdk.assess_quality(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="quality_assessment",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to quality_assessment: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("routing_decision")
        async def handle_routing_decision(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle intelligent routing decisions based on quality metrics"""
            try:
                # Make routing decision based on quality assessment
                routing_options = data.get("routing_options", [])

                # Assess quality of each option
                best_route = None
                best_score = 0.0

                for option in routing_options:
                    assessment = await self.agent_sdk.assess_quality({
                        "target": option.get("target", "unknown"),
                        "metrics": option.get("metrics", {})
                    })
                    if assessment.get("success") and assessment["data"]["overall_score"] > best_score:
                        best_score = assessment["data"]["overall_score"]
                        best_route = option

                result = {
                    "selected_route": best_route,
                    "confidence_score": best_score,
                    "routing_rationale": f"Selected based on highest quality score: {best_score:.2f}",
                    "alternatives_evaluated": len(routing_options)
                }

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="routing_decision",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to routing_decision: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("improvement_recommendations")
        async def handle_improvement_recommendations(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle generation of improvement recommendations"""
            try:
                result = await self.agent_sdk.continuous_improvement(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="improvement_recommendations",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to improvement_recommendations: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("workflow_control")
        async def handle_workflow_control(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle workflow control based on quality thresholds"""
            try:
                workflow_data = data.get("workflow", {})

                # Monitor workflow quality
                quality_check = await self.agent_sdk.monitor_trends({
                    "target": workflow_data.get("workflow_id", "unknown"),
                    "time_period": "1h"
                })

                # Make control decision based on quality trends
                control_action = "continue"
                if quality_check.get("success"):
                    trends = quality_check["data"].get("trend_anomalies", [])
                    if len(trends) > 0:
                        control_action = "investigate"
                        if any(t.get("severity") == "critical" for t in trends):
                            control_action = "halt"

                result = {
                    "workflow_id": workflow_data.get("workflow_id"),
                    "control_action": control_action,
                    "quality_status": quality_check.get("data", {}),
                    "action_reason": f"Based on quality trend analysis: {len(trends) if quality_check.get('success') else 0} anomalies detected"
                }

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="workflow_control",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to workflow_control: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("trust_verification")
        async def handle_trust_verification(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle trust verification through quality metrics"""
            try:
                agent_id = data.get("agent_id")
                trust_metrics = data.get("trust_metrics", {})
                verification_level = data.get("verification_level", "standard")

                # Assess trust based on quality metrics
                trust_assessment = await self.agent_sdk.assess_quality({
                    "target": f"agent_{agent_id}",
                    "metrics": trust_metrics,
                    "standards": ["iso_9001"] if verification_level == "high" else []
                })

                # Determine trust score
                trust_verified = False
                trust_score = 0.0

                if trust_assessment.get("success"):
                    trust_score = trust_assessment["data"]["overall_score"]
                    threshold = 0.9 if verification_level == "high" else 0.7
                    trust_verified = trust_score >= threshold

                result = {
                    "agent_id": agent_id,
                    "trust_verified": trust_verified,
                    "trust_score": trust_score,
                    "verification_level": verification_level,
                    "verification_details": trust_assessment.get("data", {})
                }

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="trust_verification",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to trust_verification: {e}")
                return self.create_secure_response(str(e), status="error")

        # Enhanced operation handlers
        @self.secure_handler("assess_quality")
        async def handle_assess_quality(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle comprehensive quality assessment with ML"""
            try:
                result = await self.agent_sdk.assess_quality(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="assess_quality",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to assess_quality: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("make_routing_decision")
        async def handle_make_routing_decision(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle intelligent routing decisions"""
            try:
                result = await self.handle_routing_decision(message, context_id, data)
                return result

            except Exception as e:
                logger.error(f"Failed to make_routing_decision: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("generate_recommendations")
        async def handle_generate_recommendations(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle generation of quality improvement recommendations"""
            try:
                result = await self.agent_sdk.continuous_improvement(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="generate_recommendations",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to generate_recommendations: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("control_workflow")
        async def handle_control_workflow(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle workflow control operations"""
            try:
                result = await self.handle_workflow_control(message, context_id, data)
                return result

            except Exception as e:
                logger.error(f"Failed to control_workflow: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("verify_trust")
        async def handle_verify_trust(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle trust verification operations"""
            try:
                result = await self.handle_trust_verification(message, context_id, data)
                return result

            except Exception as e:
                logger.error(f"Failed to verify_trust: {e}")
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
                    "quality_control_active": True,
                    "ml_models_loaded": True,
                    "blockchain_audit_enabled": True,
                    "available_operations": list(self.config.allowed_operations)
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

        @self.secure_handler("detect_anomalies")
        async def handle_detect_anomalies(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle anomaly detection operations"""
            try:
                result = await self.agent_sdk.detect_anomalies(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="detect_anomalies",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to detect_anomalies: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("monitor_trends")
        async def handle_monitor_trends(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle quality trend monitoring"""
            try:
                result = await self.agent_sdk.monitor_trends(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="monitor_trends",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to monitor_trends: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("continuous_improvement")
        async def handle_continuous_improvement(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle continuous improvement operations"""
            try:
                result = await self.agent_sdk.continuous_improvement(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="continuous_improvement",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to continuous_improvement: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("compliance_audit")
        async def handle_compliance_audit(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle compliance audit operations"""
            try:
                result = await self.agent_sdk.compliance_audit(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="compliance_audit",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to compliance_audit: {e}")
                return self.create_secure_response(str(e), status="error")

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
def create_agent6_quality_control_a2a_handler(agent_sdk: ComprehensiveQualityControlSDK) -> Agent6QualityControlA2AHandler:
    """Create A2A-compliant handler for Agent 6 - Quality Control Manager"""
    return Agent6QualityControlA2AHandler(agent_sdk)


# Example usage for migration
"""
To migrate from REST endpoints to A2A messaging:

1. Replace router initialization:
   # OLD: router = APIRouter(...)
   # NEW:
   handler = create_agent6_quality_control_a2a_handler(quality_control_sdk)

2. Replace FastAPI app with A2A listener:
   # OLD: app.include_router(router)
   # NEW:
   await handler.start()

3. Process messages through A2A:
   # Messages arrive through blockchain
   result = await handler.process_a2a_message(a2a_message)
"""
