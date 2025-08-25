"""
Advanced MCP Data Standardization Agent (Agent 1)
Enhanced data standardization with comprehensive MCP tool integration
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import uuid
import pandas as pd
import numpy as np
from collections import defaultdict
import re

from a2a.sdk.agentBase import A2AAgentBase
from a2a.sdk.decorators import a2a_handler, a2a_skill, a2a_task
from a2a.sdk.types import A2AMessage, MessageRole, TaskStatus, AgentCard
from a2a.sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt
from a2a.common.mcpPerformanceTools import MCPPerformanceTools
from a2a.common.mcpValidationTools import MCPValidationTools
from a2a.common.mcpQualityAssessmentTools import MCPQualityAssessmentTools
from app.a2a.core.security_base import SecureA2AAgent

logger = logging.getLogger(__name__)


class AdvancedMCPStandardizationAgent(SecureA2AAgent):
    """
    Advanced Data Standardization Agent with comprehensive MCP tool integration
    Handles intelligent data standardization, validation, and cross-agent coordination
    """

    def __init__(self, base_url: str):
        super().__init__(
            agent_id="advanced_mcp_standardization_agent",
            name="Advanced MCP Data Standardization Agent",
            description="Enhanced data standardization with comprehensive MCP tool integration",
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


        # Standardization management state
        self.standardization_rules = {}
        self.schema_registry = {}
        self.transformation_cache = {}
        self.validation_patterns = {}
        self.quality_metrics = {}

        # Initialize built-in standardization patterns
        self._initialize_standardization_patterns()

        logger.info(f"Initialized {self.name} with comprehensive MCP tool integration")

    async def initialize(self):
        """Initialize the agent"""
        logger.info(f"Agent {self.name} initialized successfully")

    async def shutdown(self):
        """Shutdown the agent"""
        logger.info(f"Agent {self.name} shutting down")

    @mcp_tool(
        name="intelligent_data_standardization",
        description="Perform intelligent data standardization with adaptive rule learning",
        input_schema={
            "type": "object",
            "properties": {
                "data_input": {
                    "type": "object",
                    "description": "Input data to standardize"
                },
                "standardization_config": {
                    "type": "object",
                    "description": "Standardization configuration and rules"
                },
                "target_schema": {
                    "type": "object",
                    "description": "Target schema for standardization"
                },
                "quality_requirements": {
                    "type": "object",
                    "description": "Quality requirements for standardized data"
                },
                "learning_mode": {"type": "boolean", "default": True},
                "cross_validation": {"type": "boolean", "default": True},
                "performance_optimization": {"type": "boolean", "default": True}
            },
            "required": ["data_input", "target_schema"]
        }
    )
    async def intelligent_data_standardization(
        self,
        data_input: Dict[str, Any],
        target_schema: Dict[str, Any],
        standardization_config: Optional[Dict[str, Any]] = None,
        quality_requirements: Optional[Dict[str, Any]] = None,
        learning_mode: bool = True,
        cross_validation: bool = True,
        performance_optimization: bool = True
    ) -> Dict[str, Any]:
        """
        Perform intelligent data standardization with adaptive rule learning
        """
        standardization_id = f"std_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now().timestamp()

        try:
            # Step 1: Validate input data and schema using MCP tools
            input_validation = await self.validation_tools.validate_data_structure(
                data=data_input,
                expected_structure=standardization_config.get("input_schema") if standardization_config else None,
                validation_level="comprehensive"
            )

            schema_validation = await self.validation_tools.validate_schema_compliance(
                data=target_schema,
                schema=self._get_schema_validation_schema(),
                validation_level="strict"
            )

            if not input_validation["is_valid"] or not schema_validation["is_valid"]:
                return {
                    "status": "error",
                    "error": "Input or schema validation failed",
                    "input_validation": input_validation,
                    "schema_validation": schema_validation,
                    "standardization_id": standardization_id
                }

            # Step 2: Analyze data patterns and characteristics
            data_analysis = await self._analyze_data_patterns_mcp(data_input)

            # Step 3: Generate or retrieve standardization rules
            standardization_rules = await self._generate_standardization_rules_mcp(
                data_input, target_schema, standardization_config, data_analysis, learning_mode
            )

            # Step 4: Apply standardization transformations
            transformation_results = await self._apply_standardization_transformations_mcp(
                data_input, standardization_rules, target_schema, performance_optimization
            )

            # Step 5: Cross-validation with other agents if enabled
            cross_validation_results = {}
            if cross_validation:
                cross_validation_results = await self._perform_cross_agent_validation_mcp(
                    transformation_results["standardized_data"], target_schema
                )

            # Step 6: Quality assessment using MCP tools
            quality_assessment = await self.quality_tools.assess_standardization_quality(
                original_data=data_input,
                standardized_data=transformation_results["standardized_data"],
                target_schema=target_schema,
                quality_requirements=quality_requirements or {},
                assessment_criteria=["data_integrity", "completeness", "consistency", "conformance"]
            )

            # Step 7: Learn from results if learning mode is enabled
            learning_insights = {}
            if learning_mode:
                learning_insights = await self._learn_from_standardization_mcp(
                    data_input, transformation_results, quality_assessment, standardization_rules
                )

            # Step 8: Performance measurement
            end_time = datetime.now().timestamp()
            performance_metrics = await self.performance_tools.measure_performance_metrics(
                operation_id=standardization_id,
                start_time=start_time,
                end_time=end_time,
                custom_metrics={
                    "input_records": transformation_results.get("input_record_count", 0),
                    "transformation_rules_applied": len(standardization_rules.get("rules", [])),
                    "quality_score": quality_assessment.get("overall_score", 0),
                    "learning_mode_enabled": learning_mode,
                    "cross_validation_enabled": cross_validation
                }
            )

            return {
                "status": "success",
                "standardization_id": standardization_id,
                "input_validation": input_validation,
                "schema_validation": schema_validation,
                "data_analysis": data_analysis,
                "standardization_rules": standardization_rules,
                "transformation_results": transformation_results,
                "cross_validation": cross_validation_results,
                "quality_assessment": quality_assessment,
                "learning_insights": learning_insights,
                "performance_metrics": performance_metrics,
                "total_duration": end_time - start_time,
                "mcp_tools_used": [
                    "validate_data_structure",
                    "validate_schema_compliance",
                    "analyze_data_patterns",
                    "generate_standardization_rules",
                    "apply_transformations",
                    "cross_agent_validation",
                    "assess_standardization_quality"
                ]
            }

        except Exception as e:
            logger.error(f"Intelligent data standardization failed: {e}")
            return {
                "status": "error",
                "standardization_id": standardization_id,
                "error": str(e)
            }

    @mcp_tool(
        name="adaptive_schema_harmonization",
        description="Harmonize multiple data schemas with intelligent conflict resolution",
        input_schema={
            "type": "object",
            "properties": {
                "source_schemas": {
                    "type": "array",
                    "description": "List of source schemas to harmonize"
                },
                "harmonization_strategy": {
                    "type": "string",
                    "enum": ["union", "intersection", "intelligent_merge", "priority_based"],
                    "default": "intelligent_merge"
                },
                "conflict_resolution": {
                    "type": "object",
                    "description": "Conflict resolution preferences"
                },
                "quality_preservation": {"type": "boolean", "default": True},
                "generate_mappings": {"type": "boolean", "default": True},
                "validate_harmonization": {"type": "boolean", "default": True}
            },
            "required": ["source_schemas"]
        }
    )
    async def adaptive_schema_harmonization(
        self,
        source_schemas: List[Dict[str, Any]],
        harmonization_strategy: str = "intelligent_merge",
        conflict_resolution: Optional[Dict[str, Any]] = None,
        quality_preservation: bool = True,
        generate_mappings: bool = True,
        validate_harmonization: bool = True
    ) -> Dict[str, Any]:
        """
        Harmonize multiple data schemas with intelligent conflict resolution
        """
        harmonization_id = f"harm_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now().timestamp()

        try:
            # Step 1: Validate all source schemas
            schema_validations = {}
            for i, schema in enumerate(source_schemas):
                validation = await self.validation_tools.validate_schema_compliance(
                    data=schema,
                    schema=self._get_schema_validation_schema(),
                    validation_level="standard"
                )
                schema_validations[f"schema_{i}"] = validation

            invalid_schemas = [k for k, v in schema_validations.items() if not v["is_valid"]]
            if invalid_schemas:
                return {
                    "status": "error",
                    "error": f"Invalid schemas: {invalid_schemas}",
                    "schema_validations": schema_validations,
                    "harmonization_id": harmonization_id
                }

            # Step 2: Analyze schema compatibility and conflicts
            compatibility_analysis = await self._analyze_schema_compatibility_mcp(source_schemas)

            # Step 3: Detect and categorize conflicts
            conflict_analysis = await self._detect_schema_conflicts_mcp(
                source_schemas, compatibility_analysis
            )

            # Step 4: Apply harmonization strategy
            harmonization_result = await self._apply_harmonization_strategy_mcp(
                source_schemas, harmonization_strategy, conflict_resolution, conflict_analysis
            )

            # Step 5: Generate field mappings if requested
            field_mappings = {}
            if generate_mappings:
                field_mappings = await self._generate_schema_mappings_mcp(
                    source_schemas, harmonization_result["harmonized_schema"]
                )

            # Step 6: Validate harmonized schema
            harmonization_validation = {}
            if validate_harmonization:
                harmonization_validation = await self._validate_harmonized_schema_mcp(
                    harmonization_result["harmonized_schema"], source_schemas
                )

            # Step 7: Quality assessment of harmonization
            quality_assessment = await self.quality_tools.assess_harmonization_quality(
                source_schemas=source_schemas,
                harmonized_schema=harmonization_result["harmonized_schema"],
                conflict_resolutions=harmonization_result.get("conflict_resolutions", {}),
                quality_preservation=quality_preservation
            )

            # Step 8: Performance measurement
            end_time = datetime.now().timestamp()
            performance_metrics = await self.performance_tools.measure_performance_metrics(
                operation_id=harmonization_id,
                start_time=start_time,
                end_time=end_time,
                operation_count=len(source_schemas),
                custom_metrics={
                    "source_schemas_count": len(source_schemas),
                    "conflicts_detected": len(conflict_analysis.get("conflicts", [])),
                    "conflicts_resolved": len(harmonization_result.get("conflict_resolutions", {})),
                    "quality_score": quality_assessment.get("overall_score", 0)
                }
            )

            return {
                "status": "success",
                "harmonization_id": harmonization_id,
                "schema_validations": schema_validations,
                "compatibility_analysis": compatibility_analysis,
                "conflict_analysis": conflict_analysis,
                "harmonization_result": harmonization_result,
                "field_mappings": field_mappings,
                "harmonization_validation": harmonization_validation,
                "quality_assessment": quality_assessment,
                "performance_metrics": performance_metrics,
                "total_duration": end_time - start_time,
                "mcp_tools_used": [
                    "validate_schema_compliance",
                    "analyze_schema_compatibility",
                    "detect_schema_conflicts",
                    "apply_harmonization_strategy",
                    "generate_schema_mappings",
                    "validate_harmonized_schema",
                    "assess_harmonization_quality"
                ]
            }

        except Exception as e:
            logger.error(f"Adaptive schema harmonization failed: {e}")
            return {
                "status": "error",
                "harmonization_id": harmonization_id,
                "error": str(e)
            }

    @mcp_tool(
        name="intelligent_data_validation",
        description="Perform comprehensive data validation with pattern recognition and anomaly detection",
        input_schema={
            "type": "object",
            "properties": {
                "data_to_validate": {
                    "type": "object",
                    "description": "Data to validate"
                },
                "validation_schema": {
                    "type": "object",
                    "description": "Schema for validation"
                },
                "validation_rules": {
                    "type": "array",
                    "description": "Custom validation rules"
                },
                "anomaly_detection": {"type": "boolean", "default": True},
                "pattern_learning": {"type": "boolean", "default": True},
                "cross_reference_validation": {"type": "boolean", "default": True},
                "remediation_suggestions": {"type": "boolean", "default": True}
            },
            "required": ["data_to_validate", "validation_schema"]
        }
    )
    async def intelligent_data_validation(
        self,
        data_to_validate: Dict[str, Any],
        validation_schema: Dict[str, Any],
        validation_rules: Optional[List[Dict[str, Any]]] = None,
        anomaly_detection: bool = True,
        pattern_learning: bool = True,
        cross_reference_validation: bool = True,
        remediation_suggestions: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive data validation with pattern recognition and anomaly detection
        """
        validation_id = f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now().timestamp()

        try:
            # Step 1: Basic schema validation using MCP tools
            basic_validation = await self.validation_tools.validate_schema_compliance(
                data=data_to_validate,
                schema=validation_schema,
                validation_level="comprehensive",
                return_details=True
            )

            # Step 2: Apply custom validation rules
            custom_validation = await self._apply_custom_validation_rules_mcp(
                data_to_validate, validation_rules or []
            )

            # Step 3: Pattern recognition and learning
            pattern_analysis = {}
            if pattern_learning:
                pattern_analysis = await self._analyze_data_patterns_for_validation_mcp(
                    data_to_validate, validation_schema
                )

            # Step 4: Anomaly detection
            anomaly_results = {}
            if anomaly_detection:
                anomaly_results = await self._detect_data_anomalies_mcp(
                    data_to_validate, validation_schema, pattern_analysis
                )

            # Step 5: Cross-reference validation with other agents
            cross_reference_results = {}
            if cross_reference_validation:
                cross_reference_results = await self._perform_cross_reference_validation_mcp(
                    data_to_validate, validation_schema
                )

            # Step 6: Comprehensive quality assessment
            quality_assessment = await self.quality_tools.assess_data_validation_quality(
                validation_results={
                    "basic": basic_validation,
                    "custom": custom_validation,
                    "pattern": pattern_analysis,
                    "anomaly": anomaly_results,
                    "cross_reference": cross_reference_results
                },
                data=data_to_validate,
                schema=validation_schema
            )

            # Step 7: Generate remediation suggestions
            remediation = {}
            if remediation_suggestions and not quality_assessment.get("is_valid", False):
                remediation = await self._generate_remediation_suggestions_mcp(
                    data_to_validate, validation_schema, quality_assessment
                )

            # Step 8: Performance measurement
            end_time = datetime.now().timestamp()
            performance_metrics = await self.performance_tools.measure_performance_metrics(
                operation_id=validation_id,
                start_time=start_time,
                end_time=end_time,
                custom_metrics={
                    "validation_rules_applied": len(validation_rules or []),
                    "anomalies_detected": len(anomaly_results.get("anomalies", [])),
                    "patterns_identified": len(pattern_analysis.get("patterns", [])),
                    "validation_quality_score": quality_assessment.get("overall_score", 0)
                }
            )

            return {
                "status": "success",
                "validation_id": validation_id,
                "basic_validation": basic_validation,
                "custom_validation": custom_validation,
                "pattern_analysis": pattern_analysis,
                "anomaly_results": anomaly_results,
                "cross_reference_results": cross_reference_results,
                "quality_assessment": quality_assessment,
                "remediation_suggestions": remediation,
                "performance_metrics": performance_metrics,
                "total_duration": end_time - start_time,
                "mcp_tools_used": [
                    "validate_schema_compliance",
                    "apply_custom_validation_rules",
                    "analyze_data_patterns",
                    "detect_data_anomalies",
                    "cross_reference_validation",
                    "assess_validation_quality",
                    "generate_remediation_suggestions"
                ]
            }

        except Exception as e:
            logger.error(f"Intelligent data validation failed: {e}")
            return {
                "status": "error",
                "validation_id": validation_id,
                "error": str(e)
            }

    @mcp_resource(
        uri="standardization://schema-registry",
        name="Schema Registry",
        description="Registry of all schemas and standardization rules"
    )
    async def get_schema_registry(self) -> Dict[str, Any]:
        """Provide access to schema registry as MCP resource"""
        return {
            "registered_schemas": self.schema_registry,
            "total_schemas": len(self.schema_registry),
            "standardization_rules": {
                rule_id: {
                    "rule_id": rule_id,
                    "name": rule.get("name", "Unknown"),
                    "type": rule.get("type", "Unknown"),
                    "created": rule.get("created_time"),
                    "usage_count": rule.get("usage_count", 0)
                }
                for rule_id, rule in self.standardization_rules.items()
            },
            "validation_patterns": len(self.validation_patterns),
            "last_updated": datetime.now().isoformat()
        }

    @mcp_resource(
        uri="standardization://transformation-cache",
        name="Transformation Cache",
        description="Cache of recent transformations and their performance"
    )
    async def get_transformation_cache(self) -> Dict[str, Any]:
        """Provide access to transformation cache as MCP resource"""
        return {
            "cached_transformations": {
                cache_id: {
                    "cache_id": cache_id,
                    "transformation_type": transform.get("type", "Unknown"),
                    "created": transform.get("created_time"),
                    "last_used": transform.get("last_used"),
                    "usage_count": transform.get("usage_count", 0),
                    "performance_score": transform.get("performance_score", 0)
                }
                for cache_id, transform in self.transformation_cache.items()
            },
            "total_cached": len(self.transformation_cache),
            "cache_statistics": self._get_cache_statistics(),
            "last_updated": datetime.now().isoformat()
        }

    @mcp_prompt(
        name="standardization_advisor",
        description="Provide intelligent advice on data standardization strategies",
        arguments=[
            {"name": "data_context", "type": "object", "description": "Context about the data to standardize"},
            {"name": "requirements", "type": "object", "description": "Standardization requirements"},
            {"name": "constraints", "type": "object", "description": "Constraints and limitations"}
        ]
    )
    async def standardization_advisor_prompt(
        self,
        data_context: Optional[Dict[str, Any]] = None,
        requirements: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Provide intelligent advice on data standardization strategies
        """
        try:
            # Analyze current standardization state
            current_state = await self._analyze_standardization_state_mcp()

            # Generate context-specific advice
            advice = await self._generate_standardization_advice_mcp(
                data_context or {}, requirements or {}, constraints or {}, current_state
            )

            return advice

        except Exception as e:
            logger.error(f"Standardization advisor failed: {e}")
            return f"I'm having trouble analyzing your standardization needs. Error: {str(e)}"

    # Private helper methods for MCP operations

    def _initialize_standardization_patterns(self):
        """Initialize built-in standardization patterns"""
        self.validation_patterns.update({
            "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            "phone": r'^\+?[\d\s\-\(\)]{10,}$',
            "date_iso": r'^\d{4}-\d{2}-\d{2}$',
            "currency": r'^\$?[\d,]+\.?\d{0,2}$',
            "alphanumeric": r'^[a-zA-Z0-9]+$'
        })

    async def _analyze_data_patterns_mcp(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data patterns using MCP tools with real implementation"""
        patterns = {
            "patterns": [],
            "statistics": {},
            "data_types": {},
            "quality_issues": [],
            "recommendations": []
        }

        try:
            # Analyze field types and patterns
            for field, value in data.items():
                field_analysis = self._analyze_field_value(field, value)
                patterns["data_types"][field] = field_analysis["type"]

                # Detect patterns
                if field_analysis.get("pattern"):
                    patterns["patterns"].append({
                        "field": field,
                        "type": field_analysis["pattern"],
                        "confidence": field_analysis.get("confidence", 0.8)
                    })

                # Detect quality issues
                if field_analysis.get("issues"):
                    patterns["quality_issues"].extend(field_analysis["issues"])

            # Generate recommendations
            if patterns["quality_issues"]:
                patterns["recommendations"].append("Address data quality issues before processing")

        except Exception as e:
            logger.warning(f"Error during pattern analysis: {e}")
            patterns["error"] = str(e)

        return patterns

    async def _generate_standardization_rules_mcp(
        self,
        data_input: Dict[str, Any],
        target_schema: Dict[str, Any],
        config: Optional[Dict[str, Any]],
        data_analysis: Dict[str, Any],
        learning_mode: bool
    ) -> Dict[str, Any]:
        """Generate standardization rules using MCP analysis"""

        rules = {
            "rules": [],
            "generated_time": datetime.now().isoformat(),
            "learning_mode": learning_mode
        }

        # Generate rules based on schema differences
        if "patterns" in data_analysis:
            for pattern in data_analysis["patterns"]:
                if pattern.get("type") == "data_type_mismatch":
                    rules["rules"].append({
                        "rule_id": str(uuid.uuid4()),
                        "type": "type_conversion",
                        "source_field": pattern.get("field"),
                        "source_type": pattern.get("detected_type"),
                        "target_type": pattern.get("expected_type"),
                        "confidence": pattern.get("confidence", 0.8)
                    })

        # Generate rules based on target schema comparison
        if target_schema and "fields" in target_schema:
            for field, field_spec in target_schema["fields"].items():
                if field in data_input:
                    current_value = data_input[field]
                    target_type = field_spec.get("type")
                    current_type = type(current_value).__name__

                    # Check for type mismatches
                    if target_type == "integer" and isinstance(current_value, str) and current_value.isdigit():
                        rules["rules"].append({
                            "rule_id": str(uuid.uuid4()),
                            "type": "type_conversion",
                            "source_field": field,
                            "source_type": "string",
                            "target_type": "integer",
                            "confidence": 0.9
                        })
                    elif target_type == "float" and isinstance(current_value, str):
                        try:
                            float(current_value)
                            rules["rules"].append({
                                "rule_id": str(uuid.uuid4()),
                                "type": "type_conversion",
                                "source_field": field,
                                "source_type": "string",
                                "target_type": "float",
                                "confidence": 0.9
                            })
                        except ValueError:
                            pass

        # Add custom rules from config
        if config and "custom_rules" in config:
            rules["rules"].extend(config["custom_rules"])

        return rules

    async def _apply_standardization_transformations_mcp(
        self,
        data_input: Dict[str, Any],
        rules: Dict[str, Any],
        target_schema: Dict[str, Any],
        optimization: bool
    ) -> Dict[str, Any]:
        """Apply standardization transformations using MCP tools"""

        transformation_start = datetime.now().timestamp()

        # Simulate transformation process
        standardized_data = data_input.copy()  # Start with copy

        transformations_applied = []

        for rule in rules.get("rules", []):
            try:
                if rule["type"] == "type_conversion":
                    field = rule["source_field"]
                    if field in standardized_data:
                        # Apply transformation based on rule
                        original_value = standardized_data[field]
                        transformed_value = self._apply_type_conversion(
                            original_value, rule["target_type"]
                        )
                        standardized_data[field] = transformed_value

                        transformations_applied.append({
                            "rule_id": rule["rule_id"],
                            "field": field,
                            "original_value": original_value,
                            "transformed_value": transformed_value,
                            "transformation_type": rule["type"]
                        })

            except Exception as e:
                logger.warning(f"Failed to apply transformation rule {rule.get('rule_id')}: {e}")

        transformation_end = datetime.now().timestamp()

        return {
            "standardized_data": standardized_data,
            "transformations_applied": transformations_applied,
            "input_record_count": len(data_input) if isinstance(data_input, list) else 1,
            "transformation_duration": transformation_end - transformation_start,
            "optimization_enabled": optimization
        }

    async def _perform_cross_agent_validation_mcp(
        self,
        standardized_data: Dict[str, Any],
        target_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform cross-agent validation using MCP"""
        validation_results = {}

        try:
            # Request validation from Data Product Agent
            data_product_validation = await self.mcp_client.call_skill_tool(
                "agent_0_data_product",
                "validate_data_product",
                {
                    "data": standardized_data,
                    "schema": target_schema,
                    "validation_level": "standard"
                }
            )
            validation_results["data_product"] = data_product_validation.get("result", {})

            # Request validation from Vector Processing Agent if applicable
            if target_schema.get("type") in ["vector_data", "embedding_data"]:
                vector_validation = await self.mcp_client.call_skill_tool(
                    "agent_3_vector_processing",
                    "validate_vector_data",
                    {
                        "data": standardized_data,
                        "schema": target_schema
                    }
                )
                validation_results["vector_processing"] = vector_validation.get("result", {})

        except Exception as e:
            validation_results["error"] = str(e)

        return validation_results

    def _apply_type_conversion(self, value: Any, target_type: str) -> Any:
        """Apply type conversion transformation"""
        try:
            if target_type == "string":
                return str(value)
            elif target_type == "integer":
                return int(float(value))
            elif target_type == "float":
                return float(value)
            elif target_type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ["true", "yes", "1", "on"]
                return bool(value)
            else:
                return value
        except (ValueError, TypeError):
            return value  # Return original if conversion fails

    def _get_schema_validation_schema(self) -> Dict[str, Any]:
        """Get schema for validating schemas"""
        return {
            "type": "object",
            "properties": {
                "fields": {"type": "object"},
                "type": {"type": "string"},
                "version": {"type": "string"},
                "metadata": {"type": "object"}
            },
            "required": ["fields", "type"]
        }

    def _get_cache_statistics(self) -> Dict[str, Any]:
        """Get transformation cache statistics"""
        if not self.transformation_cache:
            return {"total_entries": 0, "average_performance": 0}

        total_usage = sum(t.get("usage_count", 0) for t in self.transformation_cache.values())
        performance_scores = [t.get("performance_score", 0) for t in self.transformation_cache.values()]
        avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0

        return {
            "total_entries": len(self.transformation_cache),
            "total_usage": total_usage,
            "average_performance": avg_performance,
            "most_used": max(self.transformation_cache.items(),
                           key=lambda x: x[1].get("usage_count", 0))[0] if self.transformation_cache else None
        }

    def _analyze_field_value(self, field: str, value: Any) -> Dict[str, Any]:
        """Analyze individual field value"""
        analysis = {
            "field": field,
            "type": type(value).__name__,
            "issues": []
        }

        # Detect data type and patterns
        if value is None:
            analysis["issues"].append({"field": field, "issue": "null_value"})
        elif isinstance(value, str):
            analysis["type"] = "string"

            # Check for common patterns
            if "@" in value and "." in value:
                analysis["pattern"] = "email"
                analysis["confidence"] = 0.9
            elif value.startswith("http"):
                analysis["pattern"] = "url"
                analysis["confidence"] = 0.95
            elif value.replace("-", "").replace(" ", "").isdigit() and len(value) >= 10:
                analysis["pattern"] = "phone"
                analysis["confidence"] = 0.8
            elif value.count("-") == 2 and len(value) == 10:
                analysis["pattern"] = "date"
                analysis["confidence"] = 0.85

            # Check for type mismatches
            if value.isdigit():
                analysis["issues"].append({
                    "field": field,
                    "issue": "string_should_be_number",
                    "value": value
                })

        elif isinstance(value, (int, float)):
            analysis["type"] = "number"

        elif isinstance(value, bool):
            analysis["type"] = "boolean"

        elif isinstance(value, (list, dict)):
            analysis["type"] = "complex"
            analysis["nested"] = True

        return analysis

    async def _learn_from_standardization_mcp(self,
                                            data_input: Dict[str, Any],
                                            transformation_results: Dict[str, Any],
                                            quality_assessment: Dict[str, Any],
                                            standardization_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from standardization results to improve future performance"""

        learning_insights = {
            "learning_enabled": True,
            "timestamp": datetime.now().isoformat(),
            "insights": [],
            "rule_updates": [],
            "performance_feedback": {}
        }

        try:
            # Analyze quality outcomes
            overall_score = quality_assessment.get("overall_score", 0)

            if overall_score >= 0.9:
                learning_insights["insights"].append("High quality outcome - rules are effective")
                learning_insights["performance_feedback"]["effectiveness"] = "high"
            elif overall_score >= 0.7:
                learning_insights["insights"].append("Good quality outcome - minor optimizations possible")
                learning_insights["performance_feedback"]["effectiveness"] = "good"
            else:
                learning_insights["insights"].append("Quality below target - rules need improvement")
                learning_insights["performance_feedback"]["effectiveness"] = "needs_improvement"

            # Learn from transformations applied
            transformations_applied = transformation_results.get("transformations_applied", [])
            successful_transformations = [t for t in transformations_applied if t.get("success", True)]

            if len(successful_transformations) == len(transformations_applied):
                learning_insights["insights"].append("All transformations successful")
            else:
                failed_count = len(transformations_applied) - len(successful_transformations)
                learning_insights["insights"].append(f"{failed_count} transformations failed - review rules")

            # Analyze rule effectiveness
            rules_used = standardization_rules.get("rules", [])
            for rule in rules_used:
                if rule.get("confidence", 0) > 0.9:
                    learning_insights["rule_updates"].append({
                        "rule_id": rule.get("rule_id"),
                        "action": "reinforce",
                        "reason": "high_confidence_success"
                    })
                elif rule.get("confidence", 0) < 0.5:
                    learning_insights["rule_updates"].append({
                        "rule_id": rule.get("rule_id"),
                        "action": "review",
                        "reason": "low_confidence"
                    })

            # Performance metrics
            duration = transformation_results.get("transformation_duration", 0)
            learning_insights["performance_feedback"]["processing_time"] = duration
            learning_insights["performance_feedback"]["throughput"] = len(transformations_applied) / max(duration, 0.001)

            return learning_insights

        except Exception as e:
            logger.error(f"Learning from standardization failed: {e}")
            learning_insights["error"] = str(e)
            learning_insights["learning_enabled"] = False
            return learning_insights


# Factory function for creating advanced MCP standardization agent
def create_advanced_mcp_standardization_agent(base_url: str) -> AdvancedMCPStandardizationAgent:
    """Create and configure advanced MCP standardization agent"""
    return AdvancedMCPStandardizationAgent(base_url)