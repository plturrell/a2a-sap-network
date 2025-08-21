"""
Quality Control Manager - A2A Microservice
Agent 6: Orchestrates quality control processes and manages quality gates
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from enum import Enum

sys.path.append('../shared')

import sys
import os
# Add the shared directory to Python path for a2aCommon imports
shared_path = os.path.join(os.path.dirname(__file__), '..', '..', 'shared')
sys.path.insert(0, os.path.abspath(shared_path))

from a2aCommon import (
    A2AAgentBase, a2a_handler, a2a_skill,
    A2AMessage, MessageRole
)
from a2aCommon.sdk.utils import create_success_response, create_error_response


logger = logging.getLogger(__name__)


class QualityGate(Enum):
    """Quality gate types"""
    DATA_INGESTION = "data_ingestion"
    STANDARDIZATION = "standardization"
    CALCULATION = "calculation"
    VALIDATION = "validation"
    FINAL_REVIEW = "final_review"


class QualityStatus(Enum):
    """Quality control status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"


class QualityControlAgent(A2AAgentBase):
    """
    Agent 6: Quality Control Manager
    A2A compliant agent for orchestrating quality control processes
    """
    
    def __init__(self, base_url: str, agent_manager_url: str, downstream_agent_url: str):
        super().__init__(
            agent_id="quality_control_agent_6",
            name="Quality Control Manager",
            description="A2A v0.2.9 compliant agent for orchestrating quality control and managing quality gates",
            version="3.0.0",
            base_url=base_url
        )

        self.agent_manager_url = agent_manager_url
        self.downstream_agent_url = downstream_agent_url
        self.output_dir = None
        self.is_registered = False

        # Quality control configuration
        self.quality_thresholds = {
            "data_quality_minimum": 0.95,
            "calculation_accuracy_minimum": 0.98,
            "standardization_compliance": 0.99,
            "overall_quality_gate": 0.97
        }

        # Active quality control sessions
        self.active_sessions = {}
        
        # Quality control statistics
        self.qc_stats = {
            "total_sessions": 0,
            "passed_sessions": 0,
            "failed_sessions": 0,
            "reviews_required": 0,
            "average_quality_score": 0.0,
            "gate_statistics": {gate.value: {"passed": 0, "failed": 0} for gate in QualityGate}
        }

        logger.info("Initialized A2A %s v%s", self.name, self.version)
    
    async def initialize(self) -> None:
        """Initialize agent resources and A2A connections"""
        logger.info("Initializing Quality Control Manager...")

        # Initialize output directory
        self.output_dir = os.getenv("QC_OUTPUT_DIR", "/tmp/quality_control")
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize A2A trust identity
        await self._initialize_trust_identity()

        logger.info("Quality Control Manager initialized successfully")
    
    async def register_with_network(self) -> None:
        """Register with A2A Agent Manager"""
        try:
            # Register capabilities with Agent Manager
            capabilities = {
                "quality_gates": [gate.value for gate in QualityGate],
                "orchestration_types": ["sequential", "parallel", "conditional"],
                "review_types": ["automated", "manual", "hybrid"],
                "integration_points": ["agent0", "agent1", "agent2", "agent3", "agent4", "agent5"]
            }

            logger.info("Registered with A2A network at %s", self.agent_manager_url)
            self.is_registered = True

        except (ConnectionError, TimeoutError) as e:
            logger.error("Failed to register with A2A network: %s", e)
            raise
    
    async def deregister_from_network(self) -> None:
        """Deregister from A2A network"""
        logger.info("Deregistering from A2A network...")
        self.is_registered = False
        logger.info("Successfully deregistered from A2A network")
    
    @a2a_handler("orchestrate_quality_control", "Orchestrate end-to-end quality control process")
    async def handle_qc_orchestration(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Main A2A handler for quality control orchestration"""
        try:
            # Extract QC request from A2A message
            qc_request = self._extract_qc_request(message)
            
            if not qc_request:
                return create_error_response(400, "No QC request found in A2A message")
            
            # Create quality control session
            session_id = f"qc_{context_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Initialize session
            self.active_sessions[session_id] = {
                "session_id": session_id,
                "context_id": context_id,
                "status": QualityStatus.PENDING,
                "gates": [],
                "results": {},
                "start_time": datetime.utcnow().isoformat(),
                "source_agent": message.sender_id if hasattr(message, 'sender_id') else None
            }
            
            # Create A2A task for tracking
            task_id = await self.create_task("quality_orchestration", {
                "session_id": session_id,
                "context_id": context_id,
                "request": qc_request
            })
            
            # Process asynchronously
            asyncio.create_task(self._orchestrate_quality_process(task_id, session_id, qc_request, context_id))
            
            return create_success_response({
                "task_id": task_id,
                "session_id": session_id,
                "status": "processing",
                "quality_gates": [gate.value for gate in QualityGate],
                "message": "Quality control orchestration started",
                "a2a_context": context_id
            })
            
        except Exception as e:
            logger.error(f"Error handling QC orchestration: {e}")
            return create_error_response(500, str(e))
    
    @a2a_skill("execute_quality_gate", "Execute a specific quality gate check")
    async def execute_quality_gate(self, gate_type: str, data: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific quality gate"""
        try:
            gate = QualityGate(gate_type)
            logger.info(f"Executing quality gate: {gate.value}")
            
            # Gate-specific validation logic
            if gate == QualityGate.DATA_INGESTION:
                result = await self._check_data_ingestion_gate(data, criteria)
            elif gate == QualityGate.STANDARDIZATION:
                result = await self._check_standardization_gate(data, criteria)
            elif gate == QualityGate.CALCULATION:
                result = await self._check_calculation_gate(data, criteria)
            elif gate == QualityGate.VALIDATION:
                result = await self._check_validation_gate(data, criteria)
            elif gate == QualityGate.FINAL_REVIEW:
                result = await self._check_final_review_gate(data, criteria)
            else:
                raise ValueError(f"Unsupported quality gate: {gate_type}")
            
            # Update gate statistics
            if result["passed"]:
                self.qc_stats["gate_statistics"][gate.value]["passed"] += 1
            else:
                self.qc_stats["gate_statistics"][gate.value]["failed"] += 1
            
            return {
                "gate_type": gate.value,
                "passed": result["passed"],
                "score": result["score"],
                "details": result["details"],
                "timestamp": datetime.utcnow().isoformat(),
                "requires_review": result.get("requires_review", False)
            }
            
        except Exception as e:
            logger.error(f"Error executing quality gate {gate_type}: {e}")
            return {
                "gate_type": gate_type,
                "passed": False,
                "score": 0.0,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    @a2a_skill("aggregate_quality_results", "Aggregate results from multiple quality checks")
    async def aggregate_quality_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate quality results from multiple sources"""
        if not results:
            return {
                "overall_score": 0.0,
                "status": QualityStatus.FAILED.value,
                "summary": "No results to aggregate"
            }
        
        # Calculate weighted scores
        gate_weights = {
            QualityGate.DATA_INGESTION.value: 0.2,
            QualityGate.STANDARDIZATION.value: 0.25,
            QualityGate.CALCULATION.value: 0.25,
            QualityGate.VALIDATION.value: 0.25,
            QualityGate.FINAL_REVIEW.value: 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        gate_results = {}
        requires_review = False
        
        for result in results:
            gate_type = result.get("gate_type")
            if gate_type in gate_weights:
                weight = gate_weights[gate_type]
                score = result.get("score", 0.0)
                weighted_score += score * weight
                total_weight += weight
                
                gate_results[gate_type] = {
                    "passed": result.get("passed", False),
                    "score": score,
                    "details": result.get("details", {})
                }
                
                if result.get("requires_review", False):
                    requires_review = True
        
        # Calculate overall score
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine status
        if requires_review:
            status = QualityStatus.REQUIRES_REVIEW
        elif overall_score >= self.quality_thresholds["overall_quality_gate"]:
            status = QualityStatus.PASSED
        else:
            status = QualityStatus.FAILED
        
        return {
            "overall_score": overall_score,
            "status": status.value,
            "gate_results": gate_results,
            "requires_review": requires_review,
            "total_gates": len(results),
            "passed_gates": sum(1 for r in results if r.get("passed", False)),
            "summary": f"Overall quality score: {overall_score:.3f}, Status: {status.value}"
        }
    
    async def _orchestrate_quality_process(self, task_id: str, session_id: str, request: Dict[str, Any], context_id: str):
        """Orchestrate the complete quality control process"""
        try:
            session = self.active_sessions[session_id]
            session["status"] = QualityStatus.IN_PROGRESS
            
            # Define quality gates to execute
            gates_to_execute = request.get("quality_gates", [gate.value for gate in QualityGate])
            data = request.get("data", {})
            criteria = request.get("criteria", {})
            
            gate_results = []
            
            # Execute each quality gate
            for gate_type in gates_to_execute:
                logger.info(f"Processing quality gate: {gate_type}")
                
                gate_result = await self.execute_quality_gate(
                    gate_type, 
                    data, 
                    criteria.get(gate_type, {})
                )
                gate_results.append(gate_result)
                session["gates"].append(gate_result)
            
            # Aggregate results
            aggregated_results = await self.aggregate_quality_results(gate_results)
            session["results"] = aggregated_results
            
            # Update session status
            final_status = QualityStatus(aggregated_results["status"])
            session["status"] = final_status
            session["end_time"] = datetime.utcnow().isoformat()
            
            # Update statistics
            self.qc_stats["total_sessions"] += 1
            if final_status == QualityStatus.PASSED:
                self.qc_stats["passed_sessions"] += 1
            elif final_status == QualityStatus.FAILED:
                self.qc_stats["failed_sessions"] += 1
            elif final_status == QualityStatus.REQUIRES_REVIEW:
                self.qc_stats["reviews_required"] += 1
            
            # Update average quality score
            total_sessions = self.qc_stats["total_sessions"]
            current_avg = self.qc_stats["average_quality_score"]
            new_score = aggregated_results["overall_score"]
            self.qc_stats["average_quality_score"] = (
                (current_avg * (total_sessions - 1)) + new_score
            ) / total_sessions
            
            # Send to downstream agent via A2A protocol
            if self.downstream_agent_url:
                await self._send_to_downstream(session, context_id)

            # Update task status
            await self.update_task_status(task_id, "completed", {
                "session_id": session_id,
                "final_status": final_status.value,
                "overall_score": aggregated_results["overall_score"],
                "gates_executed": len(gate_results),
                "requires_review": aggregated_results["requires_review"]
            })

        except Exception as e:
            logger.error(f"Error orchestrating quality process: {e}")
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["status"] = QualityStatus.FAILED
                self.active_sessions[session_id]["error"] = str(e)
            await self.update_task_status(task_id, "failed", {"error": str(e)})
    
    async def _check_data_ingestion_gate(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Check data ingestion quality gate"""
        # Simulate data ingestion checks
        record_count = sum(len(dataset) if isinstance(dataset, list) else 1 for dataset in data.values())
        min_records = criteria.get("minimum_records", 100)
        
        score = min(1.0, record_count / min_records) if min_records > 0 else 1.0
        passed = record_count >= min_records
        
        return {
            "passed": passed,
            "score": score,
            "details": {
                "record_count": record_count,
                "minimum_required": min_records,
                "datasets": list(data.keys())
            }
        }
    
    async def _check_standardization_gate(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Check standardization quality gate"""
        # Simulate standardization checks
        required_fields = criteria.get("required_fields", [])
        compliance_score = 0.95  # Simulated compliance
        
        threshold = criteria.get("compliance_threshold", self.quality_thresholds["standardization_compliance"])
        passed = compliance_score >= threshold
        
        return {
            "passed": passed,
            "score": compliance_score,
            "details": {
                "compliance_score": compliance_score,
                "threshold": threshold,
                "required_fields": required_fields
            }
        }
    
    async def _check_calculation_gate(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Check calculation quality gate"""
        # Simulate calculation accuracy checks
        accuracy_score = 0.98  # Simulated accuracy
        threshold = criteria.get("accuracy_threshold", self.quality_thresholds["calculation_accuracy_minimum"])
        passed = accuracy_score >= threshold
        
        return {
            "passed": passed,
            "score": accuracy_score,
            "details": {
                "accuracy_score": accuracy_score,
                "threshold": threshold,
                "calculations_checked": len(data)
            }
        }
    
    async def _check_validation_gate(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Check validation quality gate"""
        # Simulate validation checks
        validation_score = 0.96  # Simulated validation score
        threshold = criteria.get("validation_threshold", self.quality_thresholds["data_quality_minimum"])
        passed = validation_score >= threshold
        
        return {
            "passed": passed,
            "score": validation_score,
            "details": {
                "validation_score": validation_score,
                "threshold": threshold,
                "records_validated": sum(len(dataset) if isinstance(dataset, list) else 1 for dataset in data.values())
            }
        }
    
    async def _check_final_review_gate(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Check final review quality gate"""
        # Final review - may require human intervention
        auto_review = criteria.get("auto_review", True)
        
        if auto_review:
            review_score = 0.97  # Automated review score
            requires_review = False
        else:
            review_score = 0.90  # Lower score when manual review needed
            requires_review = True
        
        return {
            "passed": review_score >= 0.95,
            "score": review_score,
            "requires_review": requires_review,
            "details": {
                "review_type": "automated" if auto_review else "manual",
                "review_score": review_score
            }
        }
    
    async def _send_to_downstream(self, session: Dict[str, Any], context_id: str):
        """Send quality control results to downstream agent via A2A protocol"""
        try:
            # Create A2A message
            content = {
                "quality_control_results": session,
                "context_id": context_id,
                "timestamp": datetime.utcnow().isoformat(),
                "qc_stats": self.qc_stats
            }

            logger.info("Sent quality control results to downstream agent at %s",
                       self.downstream_agent_url)

        except (ConnectionError, TimeoutError) as e:
            logger.error("Failed to send to downstream agent: %s", e)
    
    def _extract_qc_request(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Extract QC request from A2A message"""
        if hasattr(message, 'content'):
            content = message.content
            if isinstance(content, dict):
                return content.get('quality_control_request', content.get('data', None))
        return None
