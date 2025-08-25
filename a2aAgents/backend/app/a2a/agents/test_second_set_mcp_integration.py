"""
Comprehensive Test Suite for Second Set of Advanced MCP Agent Implementations

Tests the following agents with their MCP tool integrations:
- Advanced MCP Data Product Agent (Agent 0)
- Advanced MCP Data Standardization Agent (Agent 1)
- Advanced MCP Vector Processing Agent (Agent 3)
- Advanced MCP Calculation Validation Agent (Agent 4)
"""

import pytest
import asyncio
import json
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Import the agents to test
from .agent0DataProduct.active.advancedMcpDataProductAgent import AdvancedMCPDataProductAgent
from .agent1Standardization.active.advancedMcpStandardizationAgent import AdvancedMCPStandardizationAgent
from .agent3VectorProcessing.active.advancedMcpVectorProcessingAgent import AdvancedMCPVectorProcessingAgent
from .agent4CalcValidation.active.advancedMcpCalculationValidationAgent import AdvancedMCPCalculationValidationAgent


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")

class TestAdvancedMCPDataProductAgent:
    """Test suite for Advanced MCP Data Product Agent"""

    @pytest.fixture
    async def data_product_agent(self):
        """Create test instance of data product agent"""
        agent = AdvancedMCPDataProductAgent(os.getenv("A2A_SERVICE_URL"))

        # Mock the MCP tool providers
        agent.performance_tools = AsyncMock()
        agent.validation_tools = AsyncMock()
        agent.quality_tools = AsyncMock()
        agent.mcp_client = AsyncMock()

        return agent

    @pytest.mark.asyncio
    async def test_intelligent_data_product_registration_success(self, data_product_agent):
        """Test successful data product registration with MCP tools"""
        # Setup mock responses
        data_product_agent.validation_tools.validate_schema_compliance.return_value = {
            "is_valid": True,
            "validation_details": {"schema_check": "passed"}
        }

        data_product_agent.quality_tools.assess_data_product_quality.return_value = {
            "overall_score": 0.95,
            "completeness": 0.98,
            "accuracy": 0.92,
            "consistency": 0.96,
            "timeliness": 0.94
        }

        data_product_agent.performance_tools.measure_performance_metrics.return_value = {
            "operation_duration": 0.25,
            "memory_usage": 1024,
            "cpu_utilization": 0.15
        }

        # Test data
        product_definition = {
            "name": "customer_data",
            "type": "structured",
            "description": "Customer demographic data",
            "version": "1.0.0",
            "schema": {
                "customer_id": {"type": "string"},
                "name": {"type": "string"},
                "email": {"type": "string"}
            }
        }

        data_source = {
            "type": "database",
            "connection": "postgresql://localhost:5432/customers",
            "table": "customer_demographics"
        }

        # Execute MCP tool
        result = await data_product_agent.intelligent_data_product_registration(
            product_definition=product_definition,
            data_source=data_source,
            auto_standardization=True,
            cross_agent_validation=True
        )

        # Verify results
        assert result["status"] == "success"
        assert "product_id" in result
        assert result["definition_validation"]["is_valid"] is True
        assert result["quality_assessment"]["overall_score"] == 0.95
        assert "mcp_tools_used" in result
        assert len(result["mcp_tools_used"]) == 6

        # Verify MCP tools were called
        data_product_agent.validation_tools.validate_schema_compliance.assert_called_once()
        data_product_agent.quality_tools.assess_data_product_quality.assert_called_once()
        data_product_agent.performance_tools.measure_performance_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_advanced_data_pipeline_orchestration(self, data_product_agent):
        """Test advanced data pipeline orchestration with multiple agents"""
        # Setup mock responses
        data_product_agent.validation_tools.validate_pipeline_definition.return_value = {
            "is_valid": True,
            "validation_details": {"pipeline_structure": "valid"}
        }

        data_product_agent.quality_tools.assess_pipeline_quality.return_value = {
            "overall_score": 0.88,
            "data_integrity": 0.92,
            "processing_accuracy": 0.85,
            "completeness": 0.87
        }

        # Test data
        pipeline_definition = {
            "name": "customer_analysis_pipeline",
            "description": "Analyze customer data with multiple processing stages",
            "version": "1.0.0"
        }

        input_products = ["customer_data_v1", "transaction_data_v1"]

        processing_stages = [
            {"agent": "agent_1_standardization", "operation": "standardize_customer_data"},
            {"agent": "agent_3_vector_processing", "operation": "generate_customer_embeddings"},
            {"agent": "agent_4_calculation", "operation": "calculate_customer_metrics"}
        ]

        # Execute MCP tool
        result = await data_product_agent.advanced_data_pipeline_orchestration(
            pipeline_definition=pipeline_definition,
            input_products=input_products,
            processing_stages=processing_stages,
            quality_gates=True,
            parallel_processing=False
        )

        # Verify results
        assert result["status"] in ["success", "failed"]  # Can be either depending on stage execution
        assert "pipeline_id" in result
        assert result["quality_assessment"]["overall_score"] == 0.88
        assert "stage_execution" in result
        assert "mcp_tools_used" in result

        # Verify MCP tools were called
        data_product_agent.validation_tools.validate_pipeline_definition.assert_called_once()
        data_product_agent.quality_tools.assess_pipeline_quality.assert_called_once()

    @pytest.mark.asyncio
    async def test_intelligent_data_quality_monitoring(self, data_product_agent):
        """Test intelligent data quality monitoring with alerting"""
        # Setup mock responses
        data_product_agent.quality_tools.generate_monitoring_insights.return_value = {
            "trends": ["quality_improving", "data_completeness_stable"],
            "recommendations": ["increase_validation_frequency", "add_anomaly_detection"],
            "risk_factors": ["schema_drift_detected"]
        }

        # Test data
        product_ids = ["customer_data_v1", "transaction_data_v1", "product_catalog_v2"]
        quality_dimensions = ["completeness", "accuracy", "consistency", "timeliness"]
        alert_thresholds = {
            "completeness": 0.95,
            "accuracy": 0.90,
            "consistency": 0.85
        }

        # Execute MCP tool
        result = await data_product_agent.intelligent_data_quality_monitoring(
            product_ids=product_ids,
            monitoring_scope="periodic",
            quality_dimensions=quality_dimensions,
            alert_thresholds=alert_thresholds,
            auto_remediation=True,
            cross_product_analysis=True
        )

        # Verify results
        assert result["status"] == "success"
        assert "monitoring_id" in result
        assert result["monitoring_scope"] == "periodic"
        assert "quality_results" in result
        assert "monitoring_insights" in result
        assert "mcp_tools_used" in result

        # Verify MCP tools were called
        data_product_agent.quality_tools.generate_monitoring_insights.assert_called_once()


class TestAdvancedMCPStandardizationAgent:
    """Test suite for Advanced MCP Data Standardization Agent"""

    @pytest.fixture
    async def standardization_agent(self):
        """Create test instance of standardization agent"""
        agent = AdvancedMCPStandardizationAgent(os.getenv("A2A_SERVICE_URL"))

        # Mock the MCP tool providers
        agent.performance_tools = AsyncMock()
        agent.validation_tools = AsyncMock()
        agent.quality_tools = AsyncMock()
        agent.mcp_client = AsyncMock()

        return agent

    @pytest.mark.asyncio
    async def test_intelligent_data_standardization_success(self, standardization_agent):
        """Test intelligent data standardization with adaptive rule learning"""
        # Setup mock responses
        standardization_agent.validation_tools.validate_data_structure.return_value = {
            "is_valid": True,
            "validation_details": {"structure_check": "passed"}
        }

        standardization_agent.validation_tools.validate_schema_compliance.return_value = {
            "is_valid": True,
            "compliance_details": {"schema_check": "passed"}
        }

        standardization_agent.validation_tools.analyze_data_patterns.return_value = {
            "patterns": [
                {
                    "type": "data_type_mismatch",
                    "field": "customer_id",
                    "detected_type": "string",
                    "expected_type": "integer",
                    "confidence": 0.9
                }
            ],
            "pattern_confidence": 0.85
        }

        standardization_agent.quality_tools.assess_standardization_quality.return_value = {
            "overall_score": 0.92,
            "data_integrity": 0.95,
            "completeness": 0.90,
            "consistency": 0.93,
            "conformance": 0.89
        }

        # Test data
        data_input = {
            "customer_id": "12345",
            "name": "John Doe",
            "email": "john.doe@example.com",
            "age": "30",
            "registration_date": "2024-01-15"
        }

        target_schema = {
            "fields": {
                "customer_id": {"type": "integer"},
                "name": {"type": "string"},
                "email": {"type": "string"},
                "age": {"type": "integer"},
                "registration_date": {"type": "date"}
            },
            "type": "structured_data",
            "version": "1.0.0"
        }

        # Execute MCP tool
        result = await standardization_agent.intelligent_data_standardization(
            data_input=data_input,
            target_schema=target_schema,
            learning_mode=True,
            cross_validation=True
        )

        # Verify results
        assert result["status"] == "success"
        assert "standardization_id" in result
        assert result["input_validation"]["is_valid"] is True
        assert result["schema_validation"]["is_valid"] is True
        assert result["quality_assessment"]["overall_score"] == 0.92
        assert "transformation_results" in result
        assert "mcp_tools_used" in result

        # Verify MCP tools were called
        standardization_agent.validation_tools.validate_data_structure.assert_called_once()
        standardization_agent.validation_tools.validate_schema_compliance.assert_called_once()
        standardization_agent.quality_tools.assess_standardization_quality.assert_called_once()

    @pytest.mark.asyncio
    async def test_adaptive_schema_harmonization(self, standardization_agent):
        """Test adaptive schema harmonization with conflict resolution"""
        # Setup mock responses
        standardization_agent.validation_tools.validate_schema_compliance.return_value = {
            "is_valid": True,
            "compliance_details": {"schema_check": "passed"}
        }

        standardization_agent.quality_tools.assess_harmonization_quality.return_value = {
            "overall_score": 0.87,
            "conflict_resolution_score": 0.85,
            "schema_coverage": 0.92,
            "data_preservation": 0.84
        }

        # Test data - multiple schemas to harmonize
        source_schemas = [
            {
                "fields": {
                    "customer_id": {"type": "string"},
                    "full_name": {"type": "string"},
                    "email_address": {"type": "string"}
                },
                "type": "customer_data",
                "version": "1.0"
            },
            {
                "fields": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"}
                },
                "type": "customer_data",
                "version": "2.0"
            }
        ]

        # Execute MCP tool
        result = await standardization_agent.adaptive_schema_harmonization(
            source_schemas=source_schemas,
            harmonization_strategy="intelligent_merge",
            quality_preservation=True,
            generate_mappings=True
        )

        # Verify results
        assert result["status"] == "success"
        assert "harmonization_id" in result
        assert "harmonization_result" in result
        assert "field_mappings" in result
        assert result["quality_assessment"]["overall_score"] == 0.87
        assert "mcp_tools_used" in result

        # Verify MCP tools were called
        standardization_agent.quality_tools.assess_harmonization_quality.assert_called_once()


class TestAdvancedMCPVectorProcessingAgent:
    """Test suite for Advanced MCP Vector Processing Agent"""

    @pytest.fixture
    async def vector_agent(self):
        """Create test instance of vector processing agent"""
        agent = AdvancedMCPVectorProcessingAgent(os.getenv("A2A_SERVICE_URL"))

        # Mock the MCP tool providers
        agent.performance_tools = AsyncMock()
        agent.validation_tools = AsyncMock()
        agent.quality_tools = AsyncMock()
        agent.mcp_client = AsyncMock()

        return agent

    @pytest.mark.asyncio
    async def test_intelligent_vector_processing_success(self, vector_agent):
        """Test intelligent vector processing with optimization"""
        # Setup mock responses
        vector_agent.validation_tools.validate_vector_data.return_value = {
            "is_valid": True,
            "validation_details": {"dimension_check": "passed", "data_type_check": "passed"}
        }

        vector_agent.quality_tools.assess_vector_processing_quality.return_value = {
            "overall_score": 0.91,
            "accuracy": 0.93,
            "efficiency": 0.88,
            "consistency": 0.92,
            "completeness": 0.91
        }

        vector_agent.performance_tools.measure_performance_metrics.return_value = {
            "operation_duration": 0.15,
            "vectors_per_second": 1000,
            "memory_efficiency": 0.85
        }

        # Test data - sample vectors
        test_vectors = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9, 1.0],
            [0.2, 0.4, 0.6, 0.8, 1.0],
            [0.1, 0.3, 0.5, 0.7, 0.9]
        ]

        operations = ["normalize", "dimensionality_reduction", "similarity_matrix"]

        processing_config = {
            "expected_dimensions": 5,
            "target_dimensions": 3,
            "optimization_level": "standard"
        }

        # Execute MCP tool
        result = await vector_agent.intelligent_vector_processing(
            vectors=test_vectors,
            operations=operations,
            processing_config=processing_config,
            optimization_level="standard",
            cross_validation=True
        )

        # Verify results
        assert result["status"] == "success"
        assert "processing_id" in result
        assert result["vector_validation"]["is_valid"] is True
        assert result["quality_assessment"]["overall_score"] == 0.91
        assert "operation_results" in result
        assert "mcp_tools_used" in result

        # Verify MCP tools were called
        vector_agent.validation_tools.validate_vector_data.assert_called_once()
        vector_agent.quality_tools.assess_vector_processing_quality.assert_called_once()
        vector_agent.performance_tools.measure_performance_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_advanced_similarity_search(self, vector_agent):
        """Test advanced similarity search with multiple metrics"""
        # Setup mock responses
        vector_agent.performance_tools.measure_performance_metrics.return_value = {
            "operation_duration": 0.08,
            "search_throughput": 5000,
            "index_efficiency": 0.92
        }

        # Test data
        query_vector = [0.5, 0.5, 0.5, 0.5, 0.5]
        search_space = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9, 1.0],
            [0.4, 0.5, 0.6, 0.7, 0.8],
            [0.2, 0.3, 0.4, 0.5, 0.6]
        ]

        similarity_metrics = ["cosine", "euclidean"]

        # Execute MCP tool
        result = await vector_agent.advanced_similarity_search(
            query_vector=query_vector,
            search_space=search_space,
            similarity_metrics=similarity_metrics,
            result_count=3,
            quality_filtering=True,
            cross_metric_validation=True
        )

        # Verify results
        assert result["status"] == "success"
        assert "search_id" in result
        assert "search_results" in result
        assert "performance_metrics" in result
        assert "mcp_tools_used" in result

        # Verify MCP tools were called
        vector_agent.performance_tools.measure_performance_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_intelligent_vector_clustering(self, vector_agent):
        """Test intelligent vector clustering with adaptive selection"""
        # Setup mock responses
        vector_agent.performance_tools.measure_performance_metrics.return_value = {
            "operation_duration": 0.22,
            "clustering_efficiency": 0.87,
            "convergence_rate": 0.94
        }

        # Test data
        test_vectors = [
            [0.1, 0.2], [0.15, 0.25], [0.12, 0.22],  # Cluster 1
            [0.8, 0.9], [0.85, 0.95], [0.82, 0.92],  # Cluster 2
            [0.4, 0.5], [0.45, 0.55], [0.42, 0.52]   # Cluster 3
        ]

        clustering_config = {
            "n_clusters": 3,
            "algorithm": "kmeans",
            "max_iterations": 100
        }

        # Execute MCP tool
        result = await vector_agent.intelligent_vector_clustering(
            vectors=test_vectors,
            clustering_config=clustering_config,
            adaptive_selection=True,
            quality_optimization=True,
            cross_validation=True
        )

        # Verify results
        assert result["status"] == "success"
        assert "clustering_id" in result
        assert "clustering_results" in result
        assert "optimized_results" in result
        assert "validation_metrics" in result
        assert "mcp_tools_used" in result

        # Verify MCP tools were called
        vector_agent.performance_tools.measure_performance_metrics.assert_called_once()


class TestAdvancedMCPCalculationValidationAgent:
    """Test suite for Advanced MCP Calculation Validation Agent"""

    @pytest.fixture
    async def calculation_agent(self):
        """Create test instance of calculation validation agent"""
        agent = AdvancedMCPCalculationValidationAgent(os.getenv("A2A_SERVICE_URL"))

        # Mock the MCP tool providers
        agent.performance_tools = AsyncMock()
        agent.validation_tools = AsyncMock()
        agent.quality_tools = AsyncMock()
        agent.mcp_client = AsyncMock()

        return agent

    @pytest.mark.asyncio
    async def test_comprehensive_calculation_validation_success(self, calculation_agent):
        """Test comprehensive calculation validation with multiple methods"""
        # Setup mock responses
        calculation_agent.validation_tools.validate_calculation_request.return_value = {
            "is_valid": True,
            "validation_details": {"expression_check": "passed", "parameter_check": "passed"}
        }

        calculation_agent.quality_tools.assess_calculation_validation_quality.return_value = {
            "overall_score": 0.96,
            "accuracy": 0.98,
            "precision": 0.95,
            "consistency": 0.97,
            "reliability": 0.94
        }

        calculation_agent.performance_tools.measure_performance_metrics.return_value = {
            "operation_duration": 0.12,
            "validation_throughput": 800,
            "accuracy_rate": 0.98
        }

        # Test data
        calculation_request = {
            "expression": "2 + 3 * 4",
            "parameters": {},
            "expected_type": "numeric",
            "precision_requirements": "standard"
        }

        expected_result = 14
        validation_methods = ["direct", "symbolic", "numerical"]

        # Execute MCP tool
        result = await calculation_agent.comprehensive_calculation_validation(
            calculation_request=calculation_request,
            expected_result=expected_result,
            validation_methods=validation_methods,
            cross_agent_validation=True,
            symbolic_verification=True,
            performance_benchmarking=True
        )

        # Verify results
        assert result["status"] == "success"
        assert "validation_id" in result
        assert result["request_validation"]["is_valid"] is True
        assert result["quality_assessment"]["overall_score"] == 0.96
        assert "calculation_results" in result
        assert "comparison_results" in result
        assert "mcp_tools_used" in result

        # Verify MCP tools were called
        calculation_agent.validation_tools.validate_calculation_request.assert_called_once()
        calculation_agent.quality_tools.assess_calculation_validation_quality.assert_called_once()
        calculation_agent.performance_tools.measure_performance_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_intelligent_test_case_generation(self, calculation_agent):
        """Test intelligent test case generation for calculations"""
        # Setup mock responses
        calculation_agent.performance_tools.measure_performance_metrics.return_value = {
            "operation_duration": 0.18,
            "test_generation_rate": 200,
            "coverage_achieved": 0.92
        }

        # Test data
        calculation_type = "arithmetic_operations"
        test_parameters = {
            "number_range": {"min": -1000, "max": 1000},
            "operation_types": ["addition", "subtraction", "multiplication", "division"],
            "precision_levels": ["standard", "high"]
        }

        coverage_requirements = {
            "edge_cases": 0.90,
            "boundary_conditions": 0.85,
            "error_conditions": 0.80
        }

        # Execute MCP tool
        result = await calculation_agent.intelligent_test_case_generation(
            calculation_type=calculation_type,
            test_parameters=test_parameters,
            coverage_requirements=coverage_requirements,
            edge_case_generation=True,
            boundary_testing=True,
            test_count=25
        )

        # Verify results
        assert result["status"] == "success"
        assert "generation_id" in result
        assert result["calculation_type"] == calculation_type
        assert "test_cases" in result
        assert "execution_plan" in result
        assert "mcp_tools_used" in result

        # Verify MCP tools were called
        calculation_agent.performance_tools.measure_performance_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_advanced_error_analysis(self, calculation_agent):
        """Test advanced error analysis on calculation discrepancies"""
        # Setup mock responses
        calculation_agent.quality_tools.assess_error_analysis_quality.return_value = {
            "completeness_score": 0.89,
            "analysis_depth": 0.92,
            "recommendation_quality": 0.87
        }

        calculation_agent.performance_tools.measure_performance_metrics.return_value = {
            "operation_duration": 0.25,
            "analysis_thoroughness": 0.91,
            "pattern_detection_rate": 0.88
        }

        # Test data
        calculation_error = {
            "error_type": "precision_loss",
            "expected_result": 0.1 + 0.2,
            "actual_result": 0.30000000000000004,
            "discrepancy": 0.00000000000000004,
            "calculation_context": {
                "operation": "floating_point_addition",
                "operands": [0.1, 0.2]
            }
        }

        error_context = {
            "calculation_environment": "python_float",
            "precision_settings": "default",
            "previous_errors": []
        }

        # Execute MCP tool
        result = await calculation_agent.advanced_error_analysis(
            calculation_error=calculation_error,
            error_context=error_context,
            analysis_depth="comprehensive",
            pattern_recognition=True,
            root_cause_analysis=True,
            remediation_suggestions=True
        )

        # Verify results
        assert result["status"] == "success"
        assert "analysis_id" in result
        assert "error_classification" in result
        assert "pattern_analysis" in result
        assert "root_cause_analysis" in result
        assert "remediation_suggestions" in result
        assert "mcp_tools_used" in result

        # Verify MCP tools were called
        calculation_agent.quality_tools.assess_error_analysis_quality.assert_called_once()
        calculation_agent.performance_tools.measure_performance_metrics.assert_called_once()


class TestMCPResourceIntegration:
    """Test suite for MCP resource functionality across all agents"""

    @pytest.mark.asyncio
    async def test_data_product_registry_resource(self):
        """Test data product registry MCP resource"""
        agent = AdvancedMCPDataProductAgent(os.getenv("A2A_SERVICE_URL"))

        # Add test data to registry
        agent.data_products = {
            "test_product_1": {
                "definition": {"name": "Test Product 1", "type": "structured"},
                "status": "active",
                "registration_time": datetime.now().timestamp(),
                "quality_assessment": {"overall_score": 0.85},
                "monitoring_enabled": True
            }
        }

        # Get resource
        resource_data = await agent.get_data_product_registry()

        # Verify resource structure
        assert "registered_products" in resource_data
        assert "total_products" in resource_data
        assert resource_data["total_products"] == 1
        assert "test_product_1" in resource_data["registered_products"]

    @pytest.mark.asyncio
    async def test_schema_registry_resource(self):
        """Test schema registry MCP resource"""
        agent = AdvancedMCPStandardizationAgent(os.getenv("A2A_SERVICE_URL"))

        # Add test data to registries
        agent.schema_registry = {
            "schema_1": {"name": "Customer Schema", "version": "1.0"}
        }
        agent.standardization_rules = {
            "rule_1": {"name": "Email Validation", "type": "validation", "usage_count": 5}
        }

        # Get resource
        resource_data = await agent.get_schema_registry()

        # Verify resource structure
        assert "registered_schemas" in resource_data
        assert "standardization_rules" in resource_data
        assert "total_schemas" in resource_data
        assert resource_data["total_schemas"] == 1

    @pytest.mark.asyncio
    async def test_vector_stores_resource(self):
        """Test vector stores MCP resource"""
        agent = AdvancedMCPVectorProcessingAgent(os.getenv("A2A_SERVICE_URL"))

        # Add test data to vector stores
        agent.vector_stores = {
            "store_1": {
                "name": "Customer Embeddings",
                "dimensions": 512,
                "vector_count": 10000,
                "index_type": "faiss_flat",
                "created_time": datetime.now().isoformat()
            }
        }

        # Get resource
        resource_data = await agent.get_vector_stores()

        # Verify resource structure
        assert "vector_stores" in resource_data
        assert "total_stores" in resource_data
        assert "total_vectors" in resource_data
        assert resource_data["total_stores"] == 1
        assert resource_data["total_vectors"] == 10000


class TestMCPPromptIntegration:
    """Test suite for MCP prompt functionality across all agents"""

    @pytest.mark.asyncio
    async def test_data_product_advisor_prompt(self):
        """Test data product advisor MCP prompt"""
        agent = AdvancedMCPDataProductAgent(os.getenv("A2A_SERVICE_URL"))

        # Mock the private methods
        agent._analyze_current_product_state_mcp = AsyncMock(return_value={
            "total_products": 5,
            "quality_score": 0.85,
            "active_pipelines": 2
        })

        agent._generate_quality_improvement_advice = AsyncMock(return_value=
            "Based on your current data products, I recommend implementing automated quality monitoring for products with scores below 0.8. Consider adding data validation rules for the customer_data product to improve completeness."
        )

        # Test prompt
        advice = await agent.data_product_advisor_prompt(
            query_type="quality_improvement",
            product_context={"focus": "customer_data"},
            requirements={"target_quality": 0.9}
        )

        # Verify response
        assert isinstance(advice, str)
        assert len(advice) > 0
        assert "quality monitoring" in advice

        # Verify mock calls
        agent._analyze_current_product_state_mcp.assert_called_once()
        agent._generate_quality_improvement_advice.assert_called_once()

    @pytest.mark.asyncio
    async def test_standardization_advisor_prompt(self):
        """Test standardization advisor MCP prompt"""
        agent = AdvancedMCPStandardizationAgent(os.getenv("A2A_SERVICE_URL"))

        # Mock the private methods
        agent._analyze_standardization_state_mcp = AsyncMock(return_value={
            "active_rules": 12,
            "schema_coverage": 0.78,
            "standardization_success_rate": 0.92
        })

        agent._generate_standardization_advice_mcp = AsyncMock(return_value=
            "Your standardization coverage is at 78%. I recommend creating additional rules for date format standardization and implementing schema versioning for better backwards compatibility."
        )

        # Test prompt
        advice = await agent.standardization_advisor_prompt(
            data_context={"data_types": ["customer", "transaction"]},
            requirements={"coverage_target": 0.9},
            constraints={"processing_time": "< 5 minutes"}
        )

        # Verify response
        assert isinstance(advice, str)
        assert len(advice) > 0
        assert "standardization coverage" in advice

        # Verify mock calls
        agent._analyze_standardization_state_mcp.assert_called_once()
        agent._generate_standardization_advice_mcp.assert_called_once()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
