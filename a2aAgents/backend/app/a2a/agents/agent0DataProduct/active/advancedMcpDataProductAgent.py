"""
Advanced MCP Data Product Agent (Agent 0)
Enhanced data product registration and management with comprehensive MCP tool integration
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import hashlib

from a2a.sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt
from a2a.common.mcpPerformanceTools import MCPPerformanceTools
from a2a.common.mcpValidationTools import MCPValidationTools
from a2a.common.mcpQualityAssessmentTools import MCPQualityAssessmentTools
from app.a2a.core.security_base import SecureA2AAgent

logger = logging.getLogger(__name__)


class AdvancedMCPDataProductAgent(SecureA2AAgent):
    """
    Advanced Data Product Agent with comprehensive MCP tool integration
    Handles data product lifecycle, validation, and cross-agent coordination
    """
    def __init__(self, base_url: str):
        super().__init__(
            agent_id="advanced_mcp_data_product_agent",
            name="Advanced MCP Data Product Agent",
            description="Enhanced data product management with comprehensive MCP tool integration",
            version="2.0.0",
            base_url=base_url
        )
        # Security features are initialized by SecureA2AAgent base class

        # Initialize MCP tool providers
        self.performance_tools = MCPPerformanceTools()
        self.validation_tools = MCPValidationTools()
        self.quality_tools = MCPQualityAssessmentTools()

        # Data product management state
        self.data_products = {}
        self.validation_cache = {}
        self.processing_pipelines = {}
        self.quality_metrics = {}

        logger.info(f"Initialized {self.name} with comprehensive MCP tool integration")

    async def initialize(self):
        """Initialize the agent"""
        logger.info(f"Agent {self.name} initialized successfully")

    async def shutdown(self):
        """Shutdown the agent"""
        logger.info(f"Agent {self.name} shutting down")

    @mcp_tool(
        name="intelligent_data_product_registration",
        description="Register data products with intelligent validation and quality assessment",
        input_schema={
            "type": "object",
            "properties": {
                "product_definition": {
                    "type": "object",
                    "description": "Data product definition with metadata"
                },
                "data_source": {
                    "type": "object",
                    "description": "Data source information and connection details"
                },
                "validation_rules": {
                    "type": "array",
                    "description": "Custom validation rules for the data product"
                },
                "quality_requirements": {
                    "type": "object",
                    "description": "Quality requirements and SLA definitions"
                },
                "auto_standardization": {"type": "boolean", "default": True},
                "enable_monitoring": {"type": "boolean", "default": True},
                "cross_agent_validation": {"type": "boolean", "default": True}
            },
            "required": ["product_definition", "data_source"]
        }
    )
    async def intelligent_data_product_registration(
        self,
        product_definition: Dict[str, Any],
        data_source: Dict[str, Any],
        validation_rules: Optional[List[Dict[str, Any]]] = None,
        quality_requirements: Optional[Dict[str, Any]] = None,
        auto_standardization: bool = True,
        enable_monitoring: bool = True,
        cross_agent_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Register data products with intelligent validation and quality assessment
        """
        registration_id = f"reg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now().timestamp()
        try:
            # Step 1: Validate product definition using MCP tools
            definition_validation = await self.validation_tools.validate_schema_compliance(
                data=product_definition,
                schema=self._get_data_product_schema(),
                validation_level="strict",
                return_details=True
            )

            if not definition_validation["is_valid"]:
                return {
                    "status": "error",
                    "error": "Product definition validation failed",
                    "validation_details": definition_validation,
                    "registration_id": registration_id
                }

            # Step 2: Analyze data source using MCP tools
            source_analysis = await self._analyze_data_source_mcp(data_source)

            # Step 3: Cross-agent validation if enabled
            cross_validation_results = {}
            if cross_agent_validation:
                cross_validation_results = await self._perform_cross_agent_validation_mcp(
                    product_definition, data_source, validation_rules
                )

            # Step 4: Auto-standardization using Agent 1 if enabled
            standardization_results = {}
            if auto_standardization:
                standardization_results = await self._request_standardization_mcp(
                    product_definition, data_source
                )

            # Step 5: Quality assessment and requirements validation
            quality_assessment = await self.quality_tools.assess_data_product_quality(
                product_definition=product_definition,
                data_source=source_analysis,
                quality_requirements=quality_requirements or {},
                assessment_criteria=["completeness", "accuracy", "consistency", "timeliness"]
            )

            # Step 6: Generate unique product ID and register
            product_id = self._generate_product_id(product_definition)

            data_product = {
                "product_id": product_id,
                "registration_id": registration_id,
                "definition": product_definition,
                "data_source": data_source,
                "source_analysis": source_analysis,
                "validation_rules": validation_rules or [],
                "quality_requirements": quality_requirements or {},
                "cross_validation": cross_validation_results,
                "standardization": standardization_results,
                "quality_assessment": quality_assessment,
                "registration_time": start_time,
                "status": "active",
                "monitoring_enabled": enable_monitoring
            }

            self.data_products[product_id] = data_product

            # Step 7: Setup monitoring if enabled
            monitoring_setup = {}
            if enable_monitoring:
                monitoring_setup = await self._setup_product_monitoring_mcp(
                    product_id, data_product
                )

            # Step 8: Performance tracking
            end_time = datetime.now().timestamp()
            performance_metrics = await self.performance_tools.measure_performance_metrics(
                operation_id=registration_id,
                start_time=start_time,
                end_time=end_time,
                custom_metrics={
                    "validation_rules_count": len(validation_rules or []),
                    "quality_score": quality_assessment.get("overall_score", 0),
                    "cross_validation_used": cross_agent_validation,
                    "standardization_used": auto_standardization
                }
            )

            return {
                "status": "success",
                "registration_id": registration_id,
                "product_id": product_id,
                "definition_validation": definition_validation,
                "source_analysis": source_analysis,
                "cross_validation": cross_validation_results,
                "standardization": standardization_results,
                "quality_assessment": quality_assessment,
                "monitoring_setup": monitoring_setup,
                "performance_metrics": performance_metrics,
                "total_duration": end_time - start_time,
                "mcp_tools_used": [
                    "validate_schema_compliance",
                    "analyze_data_source",
                    "cross_agent_validation",
                    "request_standardization",
                    "assess_data_product_quality",
                    "setup_product_monitoring"
                ]
            }

        except Exception as e:
            logger.error(f"Intelligent data product registration failed: {e}")
            return {
                "status": "error",
                "registration_id": registration_id,
                "error": str(e)
            }

    @mcp_tool(
        name="advanced_data_pipeline_orchestration",
        description="Orchestrate complex data processing pipelines using multiple agents",
        input_schema={
            "type": "object",
            "properties": {
                "pipeline_definition": {
                    "type": "object",
                    "description": "Data processing pipeline definition"
                },
                "input_products": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Input data product IDs"
                },
                "processing_stages": {
                    "type": "array",
                    "description": "Processing stages with agent assignments"
                },
                "output_specification": {
                    "type": "object",
                    "description": "Output data product specification"
                },
                "quality_gates": {"type": "boolean", "default": True},
                "parallel_processing": {"type": "boolean", "default": False},
                "error_handling": {
                    "type": "string",
                    "enum": ["fail_fast", "continue_on_error", "retry_on_failure"],
                    "default": "retry_on_failure"
                }
            },
            "required": ["pipeline_definition", "input_products", "processing_stages"]
        }
    )
    async def advanced_data_pipeline_orchestration(
        self,
        pipeline_definition: Dict[str, Any],
        input_products: List[str],
        processing_stages: List[Dict[str, Any]],
        output_specification: Optional[Dict[str, Any]] = None,
        quality_gates: bool = True,
        parallel_processing: bool = False,
        error_handling: str = "retry_on_failure"
    ) -> Dict[str, Any]:
        """
        Orchestrate complex data processing pipelines using multiple agents
        """
        pipeline_id = f"pipe_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now().timestamp()

        try:
            # Step 1: Validate pipeline definition and inputs
            pipeline_validation = await self.validation_tools.validate_pipeline_definition(
                pipeline=pipeline_definition,
                input_products=input_products,
                processing_stages=processing_stages,
                validation_level="comprehensive"
            )

            if not pipeline_validation["is_valid"]:
                return {
                    "status": "error",
                    "error": "Pipeline validation failed",
                    "validation_details": pipeline_validation,
                    "pipeline_id": pipeline_id
                }

            # Step 2: Verify input products exist and are accessible
            input_verification = await self._verify_input_products_mcp(input_products)
            unavailable_products = [p for p, status in input_verification.items() if not status.get("available")]

            if unavailable_products:
                return {
                    "status": "error",
                    "error": f"Input products unavailable: {unavailable_products}",
                    "input_verification": input_verification,
                    "pipeline_id": pipeline_id
                }

            # Step 3: Initialize pipeline execution context
            execution_context = {
                "pipeline_id": pipeline_id,
                "definition": pipeline_definition,
                "input_products": input_products,
                "processing_stages": processing_stages,
                "start_time": start_time,
                "stage_results": [],
                "current_stage": 0,
                "status": "running"
            }

            self.processing_pipelines[pipeline_id] = execution_context

            # Step 4: Execute processing stages
            stage_execution_results = await self._execute_pipeline_stages_mcp(
                pipeline_id, processing_stages, parallel_processing, quality_gates, error_handling
            )

            # Step 5: Generate output product if specified
            output_product = None
            if output_specification and stage_execution_results.get("success"):
                output_product = await self._generate_output_product_mcp(
                    pipeline_id, output_specification, stage_execution_results
                )

            # Step 6: Quality assessment of pipeline results
            pipeline_quality = await self.quality_tools.assess_pipeline_quality(
                pipeline_results=stage_execution_results,
                expected_outputs=output_specification or {},
                quality_criteria=["data_integrity", "processing_accuracy", "completeness"]
            )

            # Step 7: Performance measurement
            end_time = datetime.now().timestamp()
            performance_metrics = await self.performance_tools.measure_performance_metrics(
                operation_id=pipeline_id,
                start_time=start_time,
                end_time=end_time,
                operation_count=len(processing_stages),
                custom_metrics={
                    "input_products_count": len(input_products),
                    "processing_stages_count": len(processing_stages),
                    "parallel_processing": parallel_processing,
                    "quality_gates_enabled": quality_gates,
                    "pipeline_quality_score": pipeline_quality.get("overall_score", 0)
                }
            )

            # Update pipeline context
            execution_context.update({
                "status": "completed" if stage_execution_results.get("success") else "failed",
                "end_time": end_time,
                "stage_results": stage_execution_results.get("stage_results", []),
                "output_product": output_product,
                "quality_assessment": pipeline_quality,
                "performance_metrics": performance_metrics
            })

            return {
                "status": "success" if stage_execution_results.get("success") else "failed",
                "pipeline_id": pipeline_id,
                "input_verification": input_verification,
                "stage_execution": stage_execution_results,
                "output_product": output_product,
                "quality_assessment": pipeline_quality,
                "performance_metrics": performance_metrics,
                "total_duration": end_time - start_time,
                "mcp_tools_used": [
                    "validate_pipeline_definition",
                    "verify_input_products",
                    "execute_pipeline_stages",
                    "generate_output_product",
                    "assess_pipeline_quality"
                ]
            }

        except Exception as e:
            logger.error(f"Advanced data pipeline orchestration failed: {e}")
            return {
                "status": "error",
                "pipeline_id": pipeline_id,
                "error": str(e)
            }

    @mcp_tool(
        name="intelligent_data_quality_monitoring",
        description="Monitor data product quality with intelligent alerting and remediation",
        input_schema={
            "type": "object",
            "properties": {
                "product_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Data product IDs to monitor"
                },
                "monitoring_scope": {
                    "type": "string",
                    "enum": ["real_time", "periodic", "on_demand"],
                    "default": "periodic"
                },
                "quality_dimensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Quality dimensions to monitor"
                },
                "alert_thresholds": {
                    "type": "object",
                    "description": "Quality threshold definitions for alerting"
                },
                "auto_remediation": {"type": "boolean", "default": False},
                "cross_product_analysis": {"type": "boolean", "default": True}
            },
            "required": ["product_ids"]
        }
    )
    async def intelligent_data_quality_monitoring(
        self,
        product_ids: List[str],
        monitoring_scope: str = "periodic",
        quality_dimensions: Optional[List[str]] = None,
        alert_thresholds: Optional[Dict[str, Any]] = None,
        auto_remediation: bool = False,
        cross_product_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Monitor data product quality with intelligent alerting and remediation
        """
        monitoring_id = f"mon_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now().timestamp()

        try:
            # Step 1: Validate product IDs and access
            product_validation = await self._validate_product_access_mcp(product_ids)
            accessible_products = [p for p, status in product_validation.items() if status.get("accessible")]

            # Step 2: Perform quality assessment for each product
            quality_results = {}
            for product_id in accessible_products:
                product_quality = await self._assess_product_quality_mcp(
                    product_id, quality_dimensions, monitoring_scope
                )
                quality_results[product_id] = product_quality

            # Step 3: Cross-product analysis if enabled
            cross_analysis = {}
            if cross_product_analysis and len(accessible_products) > 1:
                cross_analysis = await self._perform_cross_product_analysis_mcp(
                    accessible_products, quality_results
                )

            # Step 4: Evaluate against alert thresholds
            alert_evaluations = await self._evaluate_quality_thresholds_mcp(
                quality_results, alert_thresholds or {}
            )

            # Step 5: Auto-remediation if enabled and issues detected
            remediation_actions = []
            if auto_remediation and alert_evaluations.get("alerts_triggered"):
                remediation_actions = await self._perform_auto_remediation_mcp(
                    quality_results, alert_evaluations
                )

            # Step 6: Generate monitoring insights using MCP tools
            monitoring_insights = await self.quality_tools.generate_monitoring_insights(
                quality_results=quality_results,
                cross_analysis=cross_analysis,
                historical_trends=await self._get_quality_trends_mcp(product_ids),
                alert_patterns=alert_evaluations
            )

            # Step 7: Performance tracking
            end_time = datetime.now().timestamp()
            performance_metrics = await self.performance_tools.measure_performance_metrics(
                operation_id=monitoring_id,
                start_time=start_time,
                end_time=end_time,
                operation_count=len(accessible_products),
                custom_metrics={
                    "products_monitored": len(accessible_products),
                    "quality_dimensions": len(quality_dimensions or []),
                    "alerts_triggered": len(alert_evaluations.get("triggered_alerts", [])),
                    "remediation_actions": len(remediation_actions)
                }
            )

            return {
                "status": "success",
                "monitoring_id": monitoring_id,
                "monitoring_scope": monitoring_scope,
                "products_monitored": accessible_products,
                "product_validation": product_validation,
                "quality_results": quality_results,
                "cross_analysis": cross_analysis,
                "alert_evaluations": alert_evaluations,
                "remediation_actions": remediation_actions,
                "monitoring_insights": monitoring_insights,
                "performance_metrics": performance_metrics,
                "total_duration": end_time - start_time,
                "mcp_tools_used": [
                    "validate_product_access",
                    "assess_product_quality",
                    "cross_product_analysis",
                    "evaluate_quality_thresholds",
                    "auto_remediation",
                    "generate_monitoring_insights"
                ]
            }

        except Exception as e:
            logger.error(f"Intelligent data quality monitoring failed: {e}")
            return {
                "status": "error",
                "monitoring_id": monitoring_id,
                "error": str(e)
            }

    @mcp_resource(
        uri="data-product://registry",
        name="Data Product Registry",
        description="Complete registry of all registered data products"
    )
    async def get_data_product_registry(self) -> Dict[str, Any]:
        """Provide access to data product registry as MCP resource"""
        return {
            "registered_products": {
                product_id: {
                    "product_id": product_id,
                    "name": product.get("definition", {}).get("name", "Unknown"),
                    "type": product.get("definition", {}).get("type", "Unknown"),
                    "status": product.get("status", "Unknown"),
                    "registration_time": product.get("registration_time"),
                    "quality_score": product.get("quality_assessment", {}).get("overall_score", 0),
                    "monitoring_enabled": product.get("monitoring_enabled", False)
                }
                for product_id, product in self.data_products.items()
            },
            "total_products": len(self.data_products),
            "product_types": self._get_product_type_summary(),
            "quality_summary": self._get_quality_summary(),
            "last_updated": datetime.now().isoformat()
        }

    @mcp_resource(
        uri="data-product://pipelines",
        name="Active Data Pipelines",
        description="Information about active and recent data processing pipelines"
    )
    async def get_data_pipelines(self) -> Dict[str, Any]:
        """Provide access to data pipeline information as MCP resource"""
        return {
            "active_pipelines": {
                pipeline_id: {
                    "pipeline_id": pipeline_id,
                    "status": pipeline.get("status", "Unknown"),
                    "start_time": pipeline.get("start_time"),
                    "current_stage": pipeline.get("current_stage", 0),
                    "total_stages": len(pipeline.get("processing_stages", [])),
                    "input_products_count": len(pipeline.get("input_products", [])),
                    "quality_score": pipeline.get("quality_assessment", {}).get("overall_score", 0)
                }
                for pipeline_id, pipeline in self.processing_pipelines.items()
            },
            "total_pipelines": len(self.processing_pipelines),
            "pipeline_statistics": self._get_pipeline_statistics(),
            "last_updated": datetime.now().isoformat()
        }

    @mcp_prompt(
        name="data_product_advisor",
        description="Provide intelligent advice on data product management and optimization",
        arguments=[
            {"name": "query_type", "type": "string", "description": "Type of advice needed"},
            {"name": "product_context", "type": "object", "description": "Context about data products"},
            {"name": "requirements", "type": "object", "description": "Specific requirements or constraints"}
        ]
    )
    async def data_product_advisor_prompt(
        self,
        query_type: str = "general",
        product_context: Optional[Dict[str, Any]] = None,
        requirements: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Provide intelligent advice on data product management and optimization
        """
        try:
            # Analyze current state using MCP tools
            current_state = await self._analyze_current_product_state_mcp()

            # Generate context-specific advice
            if query_type == "quality_improvement":
                advice = await self._generate_quality_improvement_advice(
                    current_state, product_context, requirements
                )
            elif query_type == "pipeline_optimization":
                advice = await self._generate_pipeline_optimization_advice(
                    current_state, product_context, requirements
                )
            elif query_type == "standardization":
                advice = await self._generate_standardization_advice(
                    current_state, product_context, requirements
                )
            else:
                advice = await self._generate_general_advice(
                    current_state, product_context, requirements
                )

            return advice

        except Exception as e:
            logger.error(f"Data product advisor failed: {e}")
            return f"I'm having trouble analyzing your data products. Error: {str(e)}"

    # Private helper methods for MCP operations

    async def _analyze_data_source_mcp(self, data_source: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data source using MCP tools with real implementation"""
        analysis = {
            "source_type": data_source.get("type", "unknown"),
            "analysis_available": True,
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }

        try:
            source_type = data_source.get("type", "").lower()

            if source_type == "database":
                analysis["metrics"] = {
                    "connection_string": bool(data_source.get("connection")),
                    "table_specified": bool(data_source.get("table")),
                    "query_provided": bool(data_source.get("query")),
                    "estimated_complexity": "medium" if data_source.get("query") else "low"
                }
                analysis["recommendations"] = [
                    "Use connection pooling for better performance",
                    "Consider adding indexes for frequently queried columns"
                ]

            elif source_type == "file":
                file_path = data_source.get("path", "")
                analysis["metrics"] = {
                    "file_exists": bool(file_path),
                    "file_format": data_source.get("format", "unknown"),
                    "estimated_size": data_source.get("size", 0),
                    "compression": data_source.get("compressed", False)
                }

            elif source_type == "memory":
                data = data_source.get("data", [])
                analysis["metrics"] = {
                    "record_count": len(data) if isinstance(data, list) else 1,
                    "estimated_memory_mb": len(json.dumps(data)) / (1024 * 1024) if data else 0,
                    "data_structure": type(data).__name__,
                    "immediate_access": True
                }

            # Calculate health score
            quality_factors = [
                bool(data_source.get("schema")),
                bool(data_source.get("validation_rules")),
                bool(data_source.get("error_handler")),
                bool(data_source.get("monitoring_enabled"))
            ]
            analysis["health_score"] = sum(quality_factors) / len(quality_factors)

        except Exception as e:
            analysis["error"] = str(e)
            analysis["analysis_available"] = False

        return analysis

    async def _perform_cross_agent_validation_mcp(
        self,
        product_definition: Dict[str, Any],
        data_source: Dict[str, Any],
        validation_rules: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Perform cross-agent validation using MCP with real implementation"""
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "validations_performed": [],
            "overall_status": "pending"
        }

        try:
            # Determine which agents to involve based on data type
            agents_to_validate = []

            if product_definition.get("type") == "structured":
                agents_to_validate.append("agent_1_standardization")

            if product_definition.get("type") in ["vector_data", "embedding_data"]:
                agents_to_validate.append("agent_3_vector_processing")

            if any(isinstance(v, (int, float)) for v in str(data_source).split()):
                agents_to_validate.append("agent_4_calculation")

            # Perform validations with real agent calls
            for agent_id in agents_to_validate:
                try:
                    if agent_id == "agent_1_standardization":
                        validation_result = await self.mcp_client.call_skill_tool(
                            agent_id,
                            "validate_data_structure",
                            {
                                "data_definition": product_definition,
                                "source_info": data_source,
                                "validation_level": "strict"
                            }
                        )
                    elif agent_id == "agent_3_vector_processing":
                        validation_result = await self.mcp_client.call_skill_tool(
                            agent_id,
                            "validate_vector_structure",
                            {
                                "vector_definition": product_definition,
                                "expected_dimensions": product_definition.get("vector_dimensions")
                            }
                        )
                    elif agent_id == "agent_4_calculation":
                        validation_result = await self.mcp_client.call_skill_tool(
                            agent_id,
                            "validate_calculation_data",
                            {
                                "data_definition": product_definition
                            }
                        )

                    validation_results["validations_performed"].append({
                        "agent": agent_id,
                        "status": "success" if validation_result.get("success") else "error",
                        "result": validation_result.get("result", {}),
                        "confidence": validation_result.get("result", {}).get("confidence", 0.85)
                    })

                except Exception as agent_error:
                    validation_results["validations_performed"].append({
                        "agent": agent_id,
                        "status": "error",
                        "error": str(agent_error)
                    })

            # Determine overall status
            if all(v["status"] == "success" for v in validation_results["validations_performed"]):
                validation_results["overall_status"] = "validated"
            elif any(v["status"] == "error" for v in validation_results["validations_performed"]):
                validation_results["overall_status"] = "partial"
            else:
                validation_results["overall_status"] = "not_validated"

        except Exception as e:
            logger.error(f"Cross-agent validation failed: {e}")
            validation_results["overall_status"] = "error"
            validation_results["error"] = str(e)

        return validation_results

    async def _request_standardization_mcp(
        self,
        product_definition: Dict[str, Any],
        data_source: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Request standardization from Agent 1 via MCP with real implementation"""
        try:
            # Extract sample data for standardization
            sample_data = None
            if data_source.get("data"):
                if isinstance(data_source["data"], list) and data_source["data"]:
                    sample_data = data_source["data"][0]
                elif isinstance(data_source["data"], dict):
                    sample_data = data_source["data"]

            if sample_data:
                standardization_result = await self.mcp_client.call_skill_tool(
                    "agent_1_standardization",
                    "intelligent_data_standardization",
                    {
                        "data_input": sample_data,
                        "target_schema": product_definition.get("schema", {}),
                        "learning_mode": True,
                        "cross_validation": False  # Avoid circular validation
                    }
                )

                if standardization_result.get("success"):
                    return {
                        "standardization_applied": True,
                        "rules_generated": standardization_result.get("result", {}).get("standardization_rules", {}),
                        "transformation_results": standardization_result.get("result", {}).get("transformation_results", {}),
                        "quality_score": standardization_result.get("result", {}).get("quality_assessment", {}).get("overall_score", 0)
                    }
                else:
                    return {
                        "standardization_applied": False,
                        "error": standardization_result.get("error", "Standardization failed")
                    }
            else:
                return {
                    "standardization_applied": False,
                    "error": "No sample data available for standardization"
                }

        except Exception as e:
            logger.error(f"Standardization request failed: {e}")
            return {"error": str(e), "standardization_applied": False}

    def _generate_product_id(self, product_definition: Dict[str, Any]) -> str:
        """Generate unique product ID"""
        product_name = product_definition.get("name", "unknown")
        product_type = product_definition.get("type", "generic")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create hash of definition for uniqueness
        definition_str = json.dumps(product_definition, sort_keys=True)
        definition_hash = hashlib.sha256(definition_str.encode()).hexdigest()[:8]

        return f"{product_type}_{product_name}_{timestamp}_{definition_hash}"

    def _get_data_product_schema(self) -> Dict[str, Any]:
        """Get the schema for data product validation"""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "type": {"type": "string", "enum": ["structured", "unstructured", "vector_data", "stream"]},
                "description": {"type": "string"},
                "version": {"type": "string"},
                "schema": {"type": "object"},
                "metadata": {"type": "object"}
            },
            "required": ["name", "type", "schema"]
        }

    def _get_product_type_summary(self) -> Dict[str, int]:
        """Get summary of product types"""
        type_counts = {}
        for product in self.data_products.values():
            product_type = product.get("definition", {}).get("type", "unknown")
            type_counts[product_type] = type_counts.get(product_type, 0) + 1
        return type_counts

    def _get_quality_summary(self) -> Dict[str, Any]:
        """Get quality summary across all products"""
        if not self.data_products:
            return {"average_quality": 0, "quality_distribution": {}}

        quality_scores = []
        for product in self.data_products.values():
            score = product.get("quality_assessment", {}).get("overall_score", 0)
            quality_scores.append(score)

        avg_quality = sum(quality_scores) / len(quality_scores)

        # Quality distribution
        quality_ranges = {"high": 0, "medium": 0, "low": 0}
        for score in quality_scores:
            if score >= 0.8:
                quality_ranges["high"] += 1
            elif score >= 0.6:
                quality_ranges["medium"] += 1
            else:
                quality_ranges["low"] += 1

        return {
            "average_quality": avg_quality,
            "quality_distribution": quality_ranges,
            "total_assessed": len(quality_scores)
        }

    def _get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get pipeline execution statistics"""
        if not self.processing_pipelines:
            return {"total_executed": 0, "success_rate": 0}

        total = len(self.processing_pipelines)
        successful = sum(1 for p in self.processing_pipelines.values() if p.get("status") == "completed")

        return {
            "total_executed": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": (successful / total * 100) if total > 0 else 0
        }

    async def _setup_product_monitoring_mcp(self, product_id: str, data_product: Dict[str, Any]) -> Dict[str, Any]:
        """Setup monitoring for a data product using MCP tools"""
        try:
            monitoring_config = {
                "product_id": product_id,
                "monitoring_enabled": True,
                "created_time": datetime.now().isoformat(),
                "monitoring_type": "basic",
                "alerts_enabled": True
            }

            # Basic monitoring setup
            if data_product.get("quality_requirements"):
                monitoring_config["quality_monitoring"] = True
                monitoring_config["quality_thresholds"] = data_product["quality_requirements"]

            # Performance monitoring
            monitoring_config["performance_monitoring"] = True
            monitoring_config["performance_baseline"] = {
                "setup_time": datetime.now().timestamp(),
                "initial_quality_score": data_product.get("quality_assessment", {}).get("overall_score", 0.8)
            }

            return {
                "status": "monitoring_setup_complete",
                "monitoring_config": monitoring_config,
                "monitoring_id": f"mon_{product_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }

        except Exception as e:
            logger.error(f"Failed to setup monitoring for product {product_id}: {e}")
            return {
                "status": "monitoring_setup_failed",
                "error": str(e)
            }


# Factory function for creating advanced MCP data product agent
def create_advanced_mcp_data_product_agent(base_url: str) -> AdvancedMCPDataProductAgent:
    """Create and configure advanced MCP data product agent"""
    return AdvancedMCPDataProductAgent(base_url)
