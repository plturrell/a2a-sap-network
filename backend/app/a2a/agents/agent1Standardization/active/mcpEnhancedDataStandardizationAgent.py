"""
MCP-Enhanced Data Standardization Agent
Example implementation showing how to integrate MCP tools into existing agents
"""

import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import asyncio
import json
import logging
import os
import hashlib

# Import SDK components
from app.a2a.sdk import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)

# Import performance monitoring
from app.a2a.core.performanceMonitor import AlertThresholds, monitor_performance

# Import MCP tools
from ...common.mcpQualityAssessmentTools import mcp_quality_assessment
from ...common.mcpValidationTools import mcp_validation_tools
from ...common.mcpPerformanceTools import mcp_performance_tools
from ..reasoningAgent.mcpReasoningConfidenceCalculator import mcp_confidence_calculator

# Import standardizers
from ..common.standardizers.accountStandardizer import AccountStandardizer
from ..common.standardizers.locationStandardizer import LocationStandardizer
from ..common.standardizers.productStandardizer import ProductStandardizer

logger = logging.getLogger(__name__)


class MCPEnhancedDataStandardizationAgent(A2AAgentBase):
    """
    Enhanced Data Standardization Agent with MCP tool integration
    Demonstrates how to use MCP tools for quality assessment, validation, and performance monitoring
    """
    
    def __init__(self, base_url: str, enable_monitoring: bool = True):
        super().__init__(
            agent_id="mcp_enhanced_standardization_agent",
            name="MCP-Enhanced Data Standardization Agent",
            description="L4 hierarchical data standardization with MCP tool integration",
            version="5.0.0",  # MCP-enhanced version
            base_url=base_url
        )
        
        self.enable_monitoring = enable_monitoring
        
        # Initialize MCP tools
        self.mcp_quality_assessment = mcp_quality_assessment
        self.mcp_validation_tools = mcp_validation_tools
        self.mcp_performance_tools = mcp_performance_tools
        self.mcp_confidence_calculator = mcp_confidence_calculator
        
        # Initialize standardizers
        self.standardizers = {
            "account": AccountStandardizer(),
            "location": LocationStandardizer(),
            "product": ProductStandardizer()
        }
        
        # L4 schema definitions
        self.l4_schemas = {
            "account": {
                "type": "object",
                "required": ["account_id", "l1_category", "l2_subcategory", "l3_classification", "l4_specific"],
                "properties": {
                    "account_id": {"type": "string"},
                    "l1_category": {"type": "string"},
                    "l2_subcategory": {"type": "string"},
                    "l3_classification": {"type": "string"},
                    "l4_specific": {"type": "string"},
                    "account_name": {"type": "string"},
                    "account_type": {"type": "string"}
                }
            },
            "location": {
                "type": "object", 
                "required": ["location_id", "l1_region", "l2_country", "l3_state", "l4_city"],
                "properties": {
                    "location_id": {"type": "string"},
                    "l1_region": {"type": "string"},
                    "l2_country": {"type": "string"},
                    "l3_state": {"type": "string"},
                    "l4_city": {"type": "string"},
                    "location_name": {"type": "string"}
                }
            },
            "product": {
                "type": "object",
                "required": ["product_id", "l1_category", "l2_family", "l3_line", "l4_specific"],
                "properties": {
                    "product_id": {"type": "string"},
                    "l1_category": {"type": "string"},
                    "l2_family": {"type": "string"},
                    "l3_line": {"type": "string"},
                    "l4_specific": {"type": "string"},
                    "product_name": {"type": "string"}
                }
            }
        }
        
        # Statistics
        self.standardization_stats = {
            "total_processed": 0,
            "records_standardized": 0,
            "quality_assessments": 0,
            "validation_checks": 0,
            "performance_measurements": 0
        }
        
        logger.info(f"Initialized {self.name} with MCP tool integration")
    
    async def initialize(self) -> None:
        """Initialize agent with MCP tools and performance monitoring"""
        logger.info("Initializing MCP-Enhanced Data Standardization Agent...")
        
        # Initialize base agent
        await super().initialize()
        
        # Initialize output directory
        self.output_dir = os.getenv("STANDARDIZATION_OUTPUT_DIR", "/tmp/mcp_standardized_data")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Enable performance monitoring
        if self.enable_monitoring:
            alert_thresholds = AlertThresholds(
                cpu_threshold=70.0,
                memory_threshold=75.0,
                response_time_threshold=5000.0,
                error_rate_threshold=0.02,
                queue_size_threshold=100
            )
            self.enable_performance_monitoring(
                alert_thresholds=alert_thresholds,
                metrics_port=8012  # Unique port for MCP-enhanced agent
            )
        
        logger.info("MCP-Enhanced Data Standardization Agent initialized successfully")
    
    @a2a_handler("mcp_enhanced_standardization")
    @monitor_performance("mcp_enhanced_standardization")
    async def handle_mcp_enhanced_standardization(self, message: A2AMessage) -> Dict[str, Any]:
        """Enhanced standardization handler using MCP tools"""
        start_time = time.time()
        
        try:
            # Extract standardization request
            request = self._extract_standardization_request(message)
            if not request:
                return self._create_error_response("No standardization request found")
            
            # Phase 1: Quality Assessment using MCP
            quality_result = await self._assess_data_quality_with_mcp(request)
            
            # Phase 2: Validation using MCP
            validation_result = await self._validate_data_with_mcp(request)
            
            # Phase 3: Standardization with confidence calculation
            standardization_result = await self._standardize_with_confidence(request)
            
            # Phase 4: Performance measurement using MCP
            end_time = time.time()
            performance_result = await self.mcp_performance_tools.measure_performance_metrics(
                operation_id="mcp_enhanced_standardization",
                start_time=start_time,
                end_time=end_time,
                operation_count=sum(len(items) for items in request.values()),
                errors=standardization_result.get("errors", 0)
            )
            
            # Update statistics
            self.standardization_stats["total_processed"] += 1
            self.standardization_stats["quality_assessments"] += 1
            self.standardization_stats["validation_checks"] += 1
            self.standardization_stats["performance_measurements"] += 1
            
            return self._create_success_response({
                "standardization_id": f"mcp_std_{int(end_time)}",
                "quality_assessment": quality_result,
                "validation_result": validation_result,
                "standardization_result": standardization_result,
                "performance_metrics": performance_result,
                "mcp_tools_used": ["quality_assessment", "validation", "confidence_calculation", "performance_measurement"]
            })
            
        except Exception as e:
            logger.error(f"MCP-enhanced standardization failed: {e}")
            return self._create_error_response(f"Standardization failed: {str(e)}")
    
    @a2a_skill(
        name="mcp_account_standardization",
        description="Account standardization with MCP quality assessment and validation",
        capabilities=["mcp-enhanced", "quality-assured", "validated"],
        domain="financial-data"
    )
    async def mcp_account_standardization_skill(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Account standardization with comprehensive MCP integration"""
        
        accounts = input_data.get("items", [])
        if not accounts:
            return {"error": "No account data provided"}
        
        # Step 1: Pre-standardization quality assessment
        quality_result = await self.mcp_quality_assessment.assess_data_quality(
            dataset=accounts,
            quality_dimensions=["completeness", "accuracy", "consistency"],
            schema={"required": ["account_id", "account_name"]}
        )
        
        # Only proceed if quality is sufficient
        if quality_result["overall_score"] < 0.6:
            return {
                "status": "quality_insufficient",
                "quality_result": quality_result,
                "recommendation": "Improve data quality before standardization"
            }
        
        # Step 2: Schema validation
        validation_result = await self.mcp_validation_tools.validate_schema_compliance(
            data={"accounts": accounts},
            schema={
                "type": "object",
                "properties": {
                    "accounts": {
                        "type": "array",
                        "items": {"type": "object", "required": ["account_id"]}
                    }
                }
            },
            validation_level="standard"
        )
        
        # Step 3: Standardization with confidence calculation
        standardized_accounts = []
        confidence_scores = []
        
        for account in accounts:
            try:
                # Standardize the account
                standardized = self.standardizers["account"].standardize(account)
                
                # Calculate confidence in standardization
                confidence_context = {
                    "evidence": [
                        {"source_type": "data_quality", "score": quality_result["overall_score"]},
                        {"source_type": "validation", "score": validation_result["compliance_score"]}
                    ],
                    "validation_results": {
                        "logical_consistency": 0.9,
                        "completeness": quality_result["dimension_scores"].get("completeness", 0.8)
                    }
                }
                
                confidence_result = await self.mcp_confidence_calculator.calculate_reasoning_confidence_mcp(
                    reasoning_context=confidence_context,
                    include_explanation=True
                )
                
                standardized_accounts.append({
                    "original": account,
                    "standardized": standardized,
                    "confidence": confidence_result["confidence"],
                    "quality_factors": confidence_result["factor_breakdown"]
                })
                
                confidence_scores.append(confidence_result["confidence"])
                
            except Exception as e:
                logger.warning(f"Failed to standardize account {account}: {e}")
                standardized_accounts.append({
                    "original": account,
                    "standardized": None,
                    "error": str(e),
                    "confidence": 0.0
                })
                confidence_scores.append(0.0)
        
        # Step 4: Post-standardization validation
        standardized_data = [item["standardized"] for item in standardized_accounts if item["standardized"]]
        
        if standardized_data:
            post_validation = await self.mcp_validation_tools.validate_schema_compliance(
                data=standardized_data[0],  # Validate structure
                schema=self.l4_schemas["account"],
                validation_level="strict"
            )
        else:
            post_validation = {"is_valid": False, "compliance_score": 0.0}
        
        # Calculate overall metrics
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        success_rate = len([a for a in standardized_accounts if a.get("standardized")]) / len(accounts)
        
        # Update statistics
        self.standardization_stats["records_standardized"] += len(standardized_accounts)
        
        return {
            "data_type": "account",
            "total_records": len(accounts),
            "successful_records": len(standardized_data),
            "success_rate": success_rate,
            "average_confidence": avg_confidence,
            "standardized_data": standardized_accounts,
            "quality_assessment": quality_result,
            "pre_validation": validation_result,
            "post_validation": post_validation,
            "mcp_enhanced": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @a2a_skill(
        name="mcp_batch_standardization",
        description="Batch standardization with comprehensive MCP integration",
        capabilities=["batch-processing", "mcp-enhanced", "performance-monitored"],
        domain="financial-data"
    )
    async def mcp_batch_standardization_skill(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Batch standardization with full MCP integration"""
        
        start_time = time.time()
        results = {}
        total_records = 0
        successful_records = 0
        total_confidence = 0.0
        confidence_count = 0
        
        # Overall batch quality assessment
        all_data = []
        for data_type, items in input_data.items():
            if data_type in self.standardizers and isinstance(items, list):
                all_data.extend(items)
        
        if all_data:
            batch_quality = await self.mcp_quality_assessment.assess_data_quality(
                dataset=all_data,
                quality_dimensions=["completeness", "accuracy", "consistency", "uniqueness"]
            )
        else:
            batch_quality = {"overall_score": 0.0, "message": "No data to assess"}
        
        # Process each data type
        for data_type, items in input_data.items():
            if data_type in self.standardizers and isinstance(items, list):
                # Use MCP-enhanced skill for each data type
                skill_name = f"mcp_{data_type}_standardization"
                
                if skill_name in self.skills:
                    skill_result = await self.execute_skill(skill_name, {"items": items})
                    
                    if skill_result["success"]:
                        result_data = skill_result["result"]
                        results[data_type] = result_data
                        total_records += result_data["total_records"]
                        successful_records += result_data["successful_records"]
                        
                        # Aggregate confidence scores
                        if "average_confidence" in result_data:
                            total_confidence += result_data["average_confidence"] * result_data["total_records"]
                            confidence_count += result_data["total_records"]
                    else:
                        results[data_type] = {"error": skill_result["error"]}
        
        # Calculate performance metrics
        end_time = time.time()
        performance_metrics = await self.mcp_performance_tools.measure_performance_metrics(
            operation_id="mcp_batch_standardization",
            start_time=start_time,
            end_time=end_time,
            operation_count=total_records,
            errors=total_records - successful_records
        )
        
        # Calculate overall confidence
        overall_confidence = total_confidence / confidence_count if confidence_count > 0 else 0.0
        
        # Generate comprehensive report
        batch_summary = {
            "batch_id": f"mcp_batch_{int(end_time)}",
            "data_types_processed": len(results),
            "total_records": total_records,
            "successful_records": successful_records,
            "success_rate": successful_records / total_records if total_records > 0 else 0,
            "overall_confidence": overall_confidence,
            "batch_quality_assessment": batch_quality,
            "performance_metrics": performance_metrics,
            "processing_time_ms": performance_metrics["duration_ms"],
            "mcp_tools_used": [
                "quality_assessment",
                "validation_tools", 
                "confidence_calculation",
                "performance_measurement"
            ]
        }
        
        return {
            "batch_results": results,
            "batch_summary": batch_summary,
            "mcp_enhanced": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @a2a_task(
        task_type="mcp_enhanced_standardization_workflow",
        description="Complete MCP-enhanced standardization workflow with quality gates",
        timeout=900,
        retry_attempts=2
    )
    async def mcp_enhanced_standardization_workflow(self, 
                                              request: Dict[str, Any], 
                                              context_id: str) -> Dict[str, Any]:
        """Complete workflow with MCP quality gates and monitoring"""
        
        workflow_id = f"mcp_workflow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        start_time = time.time()
        
        try:
            # Phase 1: Initial Quality Gate
            quality_gate_1 = await self._quality_gate_assessment(request, "initial")
            
            if not quality_gate_1["passed"]:
                return {
                    "workflow_successful": False,
                    "workflow_id": workflow_id,
                    "failed_at": "initial_quality_gate",
                    "quality_gate_result": quality_gate_1,
                    "recommendation": "Improve data quality before processing"
                }
            
            # Phase 2: Standardization Processing
            if len(request) > 1:
                standardization_result = await self.execute_skill(
                    "mcp_batch_standardization", request
                )
            else:
                # Single data type processing
                data_type = list(request.keys())[0]
                skill_name = f"mcp_{data_type}_standardization"
                standardization_result = await self.execute_skill(
                    skill_name, {"items": request[data_type]}
                )
            
            if not standardization_result["success"]:
                return {
                    "workflow_successful": False,
                    "workflow_id": workflow_id,
                    "failed_at": "standardization_processing",
                    "error": standardization_result["error"]
                }
            
            # Phase 3: Post-processing Quality Gate
            quality_gate_2 = await self._quality_gate_assessment(
                standardization_result["result"], "post_processing"
            )
            
            # Phase 4: Performance Assessment and SLA Compliance
            end_time = time.time()
            sla_targets = {
                "response_time": 5000,  # 5 seconds
                "error_rate": 0.02,     # 2%
                "throughput": 100       # records/second
            }
            
            current_metrics = {
                "response_time": (end_time - start_time) * 1000,
                "error_rate": standardization_result["result"].get("success_rate", 1.0),
                "throughput": standardization_result["result"].get("total_records", 0) / (end_time - start_time)
            }
            
            sla_compliance = await self.mcp_performance_tools.calculate_sla_compliance(
                metrics=current_metrics,
                sla_targets=sla_targets,
                operation_id="mcp_standardization_workflow"
            )
            
            # Phase 5: Save Results with Quality Metadata
            results = {
                "workflow_id": workflow_id,
                "context_id": context_id,
                "standardization_result": standardization_result["result"],
                "quality_gates": {
                    "initial": quality_gate_1,
                    "post_processing": quality_gate_2
                },
                "sla_compliance": sla_compliance,
                "mcp_enhanced": True
            }
            
            await self._save_mcp_enhanced_results(results)
            
            return {
                "workflow_successful": True,
                "workflow_id": workflow_id,
                "results": results,
                "quality_assured": quality_gate_1["passed"] and quality_gate_2["passed"],
                "sla_compliant": sla_compliance["sla_status"] == "COMPLIANT",
                "mcp_tools_utilized": 5  # Number of different MCP tools used
            }
            
        except Exception as e:
            logger.error(f"MCP-enhanced workflow failed: {e}")
            return {
                "workflow_successful": False,
                "workflow_id": workflow_id,
                "error": str(e),
                "failed_at": "workflow_execution"
            }
    
    # MCP Integration Helper Methods
    
    async def _assess_data_quality_with_mcp(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality using MCP quality assessment tools"""
        
        # Flatten all data for assessment
        all_data = []
        for data_type, items in request.items():
            if isinstance(items, list):
                all_data.extend(items)
        
        if not all_data:
            return {"error": "No data to assess"}
        
        # Use MCP quality assessment
        quality_result = await self.mcp_quality_assessment.assess_data_quality(
            dataset=all_data,
            quality_dimensions=["completeness", "accuracy", "consistency", "validity"],
            dimension_weights={
                "completeness": 0.3,
                "accuracy": 0.3,
                "consistency": 0.2,
                "validity": 0.2
            }
        )
        
        return quality_result
    
    async def _validate_data_with_mcp(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data using MCP validation tools"""
        
        validation_results = {}
        
        for data_type, items in request.items():
            if data_type in self.l4_schemas:
                # Validate against expected schema
                validation_result = await self.mcp_validation_tools.validate_schema_compliance(
                    data={"items": items},
                    schema={
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {"type": "object"}
                            }
                        }
                    },
                    validation_level="standard"
                )
                
                validation_results[data_type] = validation_result
        
        return validation_results
    
    async def _standardize_with_confidence(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform standardization with confidence calculation"""
        
        standardization_results = {}
        
        for data_type, items in request.items():
            if data_type in self.standardizers:
                skill_name = f"mcp_{data_type}_standardization"
                
                if skill_name in self.skills:
                    result = await self.execute_skill(skill_name, {"items": items})
                    standardization_results[data_type] = result["result"] if result["success"] else {"error": result["error"]}
        
        return standardization_results
    
    async def _quality_gate_assessment(self, data: Dict[str, Any], gate_type: str) -> Dict[str, Any]:
        """Perform quality gate assessment"""
        
        if gate_type == "initial":
            # Initial data quality assessment
            if isinstance(data, dict) and any(isinstance(v, list) for v in data.values()):
                all_items = []
                for items in data.values():
                    if isinstance(items, list):
                        all_items.extend(items)
                
                quality_result = await self.mcp_quality_assessment.assess_data_quality(
                    dataset=all_items,
                    quality_dimensions=["completeness", "accuracy"]
                )
                
                passed = quality_result["overall_score"] >= 0.7
                
                return {
                    "gate_type": gate_type,
                    "passed": passed,
                    "score": quality_result["overall_score"],
                    "threshold": 0.7,
                    "details": quality_result
                }
        
        elif gate_type == "post_processing":
            # Post-processing quality assessment
            success_rate = data.get("success_rate", 0.0)
            avg_confidence = data.get("average_confidence", 0.0)
            
            combined_score = (success_rate + avg_confidence) / 2
            passed = combined_score >= 0.8
            
            return {
                "gate_type": gate_type,
                "passed": passed,
                "score": combined_score,
                "threshold": 0.8,
                "details": {
                    "success_rate": success_rate,
                    "average_confidence": avg_confidence
                }
            }
        
        return {"gate_type": gate_type, "passed": False, "error": "Unknown gate type"}
    
    async def _save_mcp_enhanced_results(self, results: Dict[str, Any]):
        """Save MCP-enhanced results with quality metadata"""
        
        workflow_id = results["workflow_id"]
        output_file = os.path.join(self.output_dir, f"mcp_enhanced_{workflow_id}.json")
        
        # Add MCP metadata
        enhanced_results = {
            **results,
            "mcp_metadata": {
                "tools_used": [
                    "quality_assessment",
                    "validation_tools",
                    "confidence_calculation", 
                    "performance_measurement"
                ],
                "quality_assured": True,
                "enhanced_version": "5.0.0",
                "processing_timestamp": datetime.utcnow().isoformat()
            }
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(enhanced_results, f, indent=2, default=str)
            logger.info(f"Saved MCP-enhanced results to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save MCP-enhanced results: {e}")
    
    # Utility methods
    
    def _extract_standardization_request(self, message: A2AMessage) -> Dict[str, Any]:
        """Extract standardization request from message"""
        request = {}
        for part in message.parts:
            if part.kind == "data" and part.data:
                data_type = part.data.get("type")
                items = part.data.get("items", [])
                if data_type and items:
                    request[data_type] = items
                elif not data_type:
                    # Check for batch data
                    for key, value in part.data.items():
                        if key in self.standardizers and isinstance(value, list):
                            request[key] = value
        return request
    
    def _create_success_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create success response"""
        return {
            "success": True,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": self.agent_id,
            "mcp_enhanced": True
        }
    
    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "success": False,
            "error": message,
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": self.agent_id
        }
    
    async def get_agent_health(self) -> Dict[str, Any]:
        """Get comprehensive agent health with MCP tool status"""
        
        health = await super().get_agent_health()
        
        # Add MCP tool status
        health["mcp_tools"] = {
            "quality_assessment": "available",
            "validation_tools": "available", 
            "performance_tools": "available",
            "confidence_calculator": "available"
        }
        
        # Add agent-specific metrics
        health["agent_metrics"] = self.standardization_stats.copy()
        
        # Add MCP enhancement indicators
        health["enhancements"] = {
            "mcp_integrated": True,
            "quality_gates": True,
            "confidence_scoring": True,
            "performance_monitoring": True,
            "version": "5.0.0"
        }
        
        return health


# Example usage and integration guide
async def example_mcp_integration():
    """
    Example showing how to use the MCP-enhanced agent
    """
    
    # Initialize the MCP-enhanced agent
    agent = MCPEnhancedDataStandardizationAgent("http://localhost:8000")
    await agent.initialize()
    
    # Example standardization request with MCP integration
    sample_data = {
        "account": [
            {"account_id": "ACC001", "account_name": "Cash", "type": "asset"},
            {"account_id": "ACC002", "account_name": "Revenue", "type": "income"}
        ]
    }
    
    # Process with MCP enhancements
    result = await agent.mcp_account_standardization_skill({"items": sample_data["account"]})
    
    print(f"Standardization completed with confidence: {result['average_confidence']:.2%}")
    print(f"Quality score: {result['quality_assessment']['overall_score']:.2%}")
    print(f"Validation passed: {result['post_validation']['is_valid']}")
    
    await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(example_mcp_integration())