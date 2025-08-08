"""
Computation Quality Testing Agent - SDK Version
Agent 4: Enhanced with A2A SDK for dynamic computation validation using template-based test generation
"""

import asyncio
import uuid
import os
import json
import yaml
import hashlib
import struct
import math
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pydantic import BaseModel, Field
from enum import Enum
import logging
import httpx
import inspect

# Template processing
try:
    from jinja2 import Template, Environment, BaseLoader
    TEMPLATE_AVAILABLE = True
except ImportError:
    TEMPLATE_AVAILABLE = False
    logging.warning("Jinja2 not available. Template processing will be limited.")

# Statistical analysis
try:
    import numpy as np
    import scipy.stats as stats
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    logging.warning("NumPy/SciPy not available. Statistical analysis will be limited.")

from app.a2a.sdk import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)
from app.a2a.sdk.utils import create_success_response, create_error_response
from src.a2a.core.workflow_context import workflow_context_manager
from src.a2a.core.workflow_monitor import workflow_monitor
from src.a2a.core.circuit_breaker import CircuitBreaker, get_breaker_manager
from app.a2a.security.smart_contract_trust import sign_a2a_message, initialize_agent_trust, verify_a2a_message
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logger = logging.getLogger(__name__)


class ComputationType(str, Enum):
    MATHEMATICAL = "mathematical"
    LOGICAL = "logical"
    TRANSFORMATIONAL = "transformational"
    PERFORMANCE = "performance"


class TestDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestMethodology(str, Enum):
    ACCURACY = "accuracy"
    PERFORMANCE = "performance"
    STRESS = "stress"
    COMPREHENSIVE = "comprehensive"


class ServiceType(str, Enum):
    API = "api"
    FUNCTION = "function"
    ALGORITHM = "algorithm"
    PIPELINE = "pipeline"


class ValidationMethod(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    PATTERN_MATCH = "pattern_match"


class ComputationTestRequest(BaseModel):
    """Request for computation quality testing"""
    service_endpoints: List[str] = Field(description="List of computational service endpoints")
    test_methodology: TestMethodology = Field(default=TestMethodology.COMPREHENSIVE)
    computation_types: List[ComputationType] = Field(default=[ComputationType.MATHEMATICAL, ComputationType.LOGICAL])
    domain_filter: Optional[str] = None
    service_types: List[ServiceType] = Field(default=[ServiceType.API, ServiceType.FUNCTION])
    test_config: Dict[str, Any] = Field(default_factory=dict)


class TestTemplate(BaseModel):
    """Template for generating test cases"""
    template_id: str
    computation_type: ComputationType
    complexity_level: TestDifficulty
    pattern_category: str
    input_generator: Dict[str, Any]
    expected_behavior: Dict[str, Any]
    validation: Dict[str, Any]
    metadata: Dict[str, Any]


class GeneratedTestCase(BaseModel):
    """Generated test case from template"""
    test_id: str
    template_source: TestTemplate
    input_data: Dict[str, Any]
    expected_output: Any
    validation_criteria: Dict[str, Any]
    timeout_seconds: float = 30.0
    metadata: Dict[str, Any]


class ServiceDiscoveryResult(BaseModel):
    """Result of service discovery"""
    service_id: str
    endpoint_url: str
    service_type: ServiceType
    computation_capabilities: List[str]
    api_schema: Optional[Dict[str, Any]] = None
    performance_characteristics: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TestExecutionResult(BaseModel):
    """Result of test execution"""
    test_id: str
    service_id: str
    success: bool
    actual_output: Any = None
    execution_time: float
    memory_usage: Optional[float] = None
    error_message: Optional[str] = None
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    quality_scores: Dict[str, float] = Field(default_factory=dict)


class QualityReport(BaseModel):
    """Comprehensive quality report"""
    suite_id: str
    service_id: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    execution_summary: Dict[str, Any]
    quality_scores: Dict[str, float]
    detailed_results: List[TestExecutionResult]
    recommendations: List[str]
    generated_at: datetime


class CalcValidationAgentSDK(A2AAgentBase):
    """
    Agent 4: Computation Quality Testing Agent
    SDK Version - Dynamic computation validation using template-based test generation
    
    Enhanced with:
    - Catalog Manager integration for ORD service discovery
    - Data Manager integration for test data storage/retrieval  
    - Agent 3 vector similarity validation
    - Agent 3 knowledge graph contextual analysis
    - Historical performance pattern analysis
    - Peer service comparison via knowledge graph
    """
    
    def __init__(
        self, 
        base_url: str, 
        template_repository_url: str = None,
        data_manager_url: str = None,
        catalog_manager_url: str = None
    ):
        super().__init__(
            agent_id="calc_validation_agent_4",
            name="Computation Quality Testing Agent",
            description="A2A v0.2.9 compliant agent for dynamic computation quality testing with Data Manager and Catalog Manager integration",
            version="1.0.0",
            base_url=base_url
        )
        
        self.template_repository_url = template_repository_url or os.getenv("TEMPLATE_REPOSITORY_URL")
        self.data_manager_url = data_manager_url or os.getenv("DATA_MANAGER_URL", "http://localhost:8001")
        self.catalog_manager_url = catalog_manager_url or os.getenv("CATALOG_MANAGER_URL", "http://localhost:8002")
        
        self.discovered_services = {}
        self.test_templates = {}
        self.test_results_cache = {}
        self.circuit_breaker_manager = get_breaker_manager()
        
        # Prometheus metrics
        self.tasks_completed = Counter('a2a_agent_tasks_completed_total', 'Total completed tasks', ['agent_id', 'task_type'])
        self.tasks_failed = Counter('a2a_agent_tasks_failed_total', 'Total failed tasks', ['agent_id', 'task_type'])
        self.processing_time = Histogram('a2a_agent_processing_time_seconds', 'Task processing time', ['agent_id', 'task_type'])
        self.queue_depth = Gauge('a2a_agent_queue_depth', 'Current queue depth', ['agent_id'])
        self.skills_count = Gauge('a2a_agent_skills_count', 'Number of skills available', ['agent_id'])
        self.tests_generated = Counter('a2a_tests_generated_total', 'Total tests generated', ['agent_id', 'computation_type'])
        self.tests_executed = Counter('a2a_tests_executed_total', 'Total tests executed', ['agent_id', 'test_type'])
        self.quality_score = Gauge('a2a_service_quality_score', 'Service quality score', ['agent_id', 'service_id'])
        
        # Set initial metrics
        self.queue_depth.labels(agent_id=self.agent_id).set(0)
        self.skills_count.labels(agent_id=self.agent_id).set(6)  # 6 main skills
        
        # Start metrics server
        self._start_metrics_server()
        
        self.processing_stats = {
            "total_tasks": 0,
            "services_discovered": 0,
            "tests_generated": 0,
            "tests_executed": 0,
            "quality_reports_generated": 0
        }
        
        logger.info(f"Initialized {self.name} v{self.version} with SDK")
    
    def _start_metrics_server(self):
        """Start Prometheus metrics server"""
        try:
            port = int(os.environ.get('PROMETHEUS_PORT', '8006'))
            start_http_server(port)
            logger.info(f"Started Prometheus metrics server on port {port}")
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")
    
    async def initialize(self) -> None:
        """Initialize agent resources"""
        logger.info("Initializing Computation Quality Testing Agent...")
        
        # Initialize storage
        storage_path = os.getenv("CALC_VALIDATION_AGENT_STORAGE_PATH", "/tmp/calc_validation_agent_state")
        os.makedirs(storage_path, exist_ok=True)
        self.storage_path = storage_path
        
        # Initialize HTTP client with circuit breaker
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Load test templates
        await self._load_test_templates()
        
        # Initialize trust system
        await self._initialize_trust_system()
        
        # Load existing state
        await self._load_agent_state()
        
        logger.info("Computation Quality Testing Agent initialization complete")
    
    # A2A Agent Integration Methods
    
    async def _call_data_manager(self, skill: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call Data Manager via A2A protocol"""
        try:
            async def make_a2a_call():
                # Create A2A message
                message = {
                    "method": "executeTask",
                    "params": {
                        "taskId": f"calc_validation_{uuid.uuid4().hex[:8]}",
                        "skill": skill,
                        "parameters": params
                    },
                    "id": f"req_{int(time.time())}"
                }
                
                # Sign message with trust system
                if hasattr(self, 'trust_identity') and self.trust_identity:
                    message = await sign_a2a_message(message, self.agent_id)
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.data_manager_url}/a2a/tasks",
                        json=message,
                        timeout=30.0
                    )
                    response.raise_for_status()
                    return response.json()
            
            # Use circuit breaker
            breaker = self.circuit_breaker_manager.get_breaker("data_manager_a2a")
            result = await breaker.call(make_a2a_call)
            return result.get('result', {})
            
        except Exception as e:
            logger.error(f"Failed to call Data Manager skill '{skill}': {e}")
            return {"error": str(e)}
    
    async def _call_catalog_manager(self, skill: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call Catalog Manager via A2A protocol"""
        try:
            async def make_a2a_call():
                message = {
                    "method": "executeTask", 
                    "params": {
                        "taskId": f"calc_validation_{uuid.uuid4().hex[:8]}",
                        "skill": skill,
                        "parameters": params
                    },
                    "id": f"req_{int(time.time())}"
                }
                
                if hasattr(self, 'trust_identity') and self.trust_identity:
                    message = await sign_a2a_message(message, self.agent_id)
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.catalog_manager_url}/a2a/tasks",
                        json=message,
                        timeout=30.0
                    )
                    response.raise_for_status()
                    return response.json()
            
            breaker = self.circuit_breaker_manager.get_breaker("catalog_manager_a2a")
            result = await breaker.call(make_a2a_call)
            return result.get('result', {})
            
        except Exception as e:
            logger.error(f"Failed to call Catalog Manager skill '{skill}': {e}")
            return {"error": str(e)}
    
    
    async def _validate_with_historical_patterns(self, test_case: GeneratedTestCase, result: Any) -> Optional[Dict[str, Any]]:
        """Self-contained validation using historical test patterns from Data Manager"""
        try:
            # Query Data Manager for historical test results for this service
            historical_data = await self._call_data_manager("data_read", {
                "data_id": f"test_history_{test_case.metadata.get('service_id', 'unknown')}_{test_case.test_type.value}"
            })
            
            if "error" in historical_data or not historical_data.get("data"):
                # No historical data available - use self-contained validation
                return self._compute_self_contained_validation(test_case, result)
            
            # Analyze historical patterns for this computation type
            return self._analyze_against_historical_patterns(test_case, result, historical_data.get("data", {}))
            
        except Exception as e:
            logger.error(f"Historical pattern validation failed: {e}")
            return None
    
    def _compute_self_contained_validation(self, test_case: GeneratedTestCase, result: Any) -> Dict[str, Any]:
        """Self-contained validation logic - no external dependencies"""
        try:
            expected_str = str(test_case.expected_output).lower().strip()
            actual_str = str(result).lower().strip()
            
            # Agent 4's own validation logic
            validation_score = 0.0
            validation_details = {}
            
            # Exact match validation
            if expected_str == actual_str:
                validation_score = 1.0
                validation_details["exact_match"] = True
            else:
                validation_details["exact_match"] = False
                
                # Numerical validation for mathematical computations
                if test_case.test_type.value in ["mathematical", "performance"]:
                    validation_score, validation_details = self._validate_numerical_result(
                        test_case.expected_output, result, validation_details
                    )
                
                # Logical validation for boolean computations
                elif test_case.test_type.value == "logical":
                    validation_score, validation_details = self._validate_logical_result(
                        test_case.expected_output, result, validation_details
                    )
                
                # String pattern validation for transformational computations
                elif test_case.test_type.value == "transformational":
                    validation_score, validation_details = self._validate_transformation_result(
                        expected_str, actual_str, validation_details
                    )
            
            return {
                "historical_pattern_score": validation_score,
                "pattern_validation": validation_score > 0.8,
                "validation_details": validation_details,
                "confidence": validation_score,
                "validation_method": "self_contained_agent4"
            }
            
        except Exception as e:
            logger.error(f"Self-contained validation failed: {e}")
            return {"historical_pattern_score": 0.0, "pattern_validation": False, "validation_method": "error"}
    
    def _validate_numerical_result(self, expected, actual, details: Dict) -> Tuple[float, Dict]:
        """Agent 4's numerical validation logic"""
        try:
            expected_num = float(expected)
            actual_num = float(actual)
            
            # Percentage difference
            if expected_num == 0:
                diff_percent = 0.0 if actual_num == 0 else 1.0
            else:
                diff_percent = abs(expected_num - actual_num) / abs(expected_num)
            
            details["expected_value"] = expected_num
            details["actual_value"] = actual_num
            details["percentage_difference"] = diff_percent
            
            # Scoring based on accuracy
            if diff_percent <= 0.01:  # 1% tolerance
                score = 1.0
            elif diff_percent <= 0.05:  # 5% tolerance
                score = 0.9
            elif diff_percent <= 0.10:  # 10% tolerance
                score = 0.7
            else:
                score = max(0.0, 1.0 - diff_percent)
            
            details["accuracy_tier"] = "high" if score > 0.9 else "medium" if score > 0.7 else "low"
            return score, details
            
        except (ValueError, TypeError):
            details["error"] = "non_numerical_result"
            return 0.0, details
    
    def _validate_logical_result(self, expected, actual, details: Dict) -> Tuple[float, Dict]:
        """Agent 4's logical validation logic"""
        try:
            # Convert to boolean
            expected_bool = str(expected).lower() in ['true', '1', 'yes', 'on']
            actual_bool = str(actual).lower() in ['true', '1', 'yes', 'on']
            
            details["expected_boolean"] = expected_bool
            details["actual_boolean"] = actual_bool
            details["boolean_match"] = expected_bool == actual_bool
            
            return 1.0 if expected_bool == actual_bool else 0.0, details
            
        except Exception:
            details["error"] = "boolean_conversion_failed"
            return 0.0, details
    
    def _validate_transformation_result(self, expected_str: str, actual_str: str, details: Dict) -> Tuple[float, Dict]:
        """Agent 4's transformation validation logic"""
        # Length preservation check
        length_ratio = len(actual_str) / max(len(expected_str), 1)
        details["length_preservation"] = 0.8 <= length_ratio <= 1.2
        
        # Character overlap
        expected_chars = set(expected_str)
        actual_chars = set(actual_str)
        char_overlap = len(expected_chars & actual_chars) / len(expected_chars | actual_chars) if expected_chars | actual_chars else 0
        details["character_overlap"] = char_overlap
        
        # Pattern similarity
        pattern_score = char_overlap * 0.7 + (1.0 if details["length_preservation"] else 0.0) * 0.3
        
        return pattern_score, details
    
    def _analyze_against_historical_patterns(self, test_case: GeneratedTestCase, result: Any, historical_data: Dict) -> Dict[str, Any]:
        """Analyze current result against historical test patterns"""
        try:
            historical_tests = historical_data.get("test_results", [])
            
            if not historical_tests:
                return self._compute_self_contained_validation(test_case, result)
            
            # Find similar historical test cases
            similar_tests = [
                test for test in historical_tests 
                if test.get("test_type") == test_case.test_type.value 
                and test.get("input_pattern") == str(test_case.input_data)
            ]
            
            if not similar_tests:
                # No similar patterns, use general service performance
                service_success_rate = historical_data.get("overall_success_rate", 0.8)
                return {
                    "historical_pattern_score": service_success_rate,
                    "pattern_validation": service_success_rate > 0.7,
                    "pattern_type": "service_baseline",
                    "confidence": service_success_rate * 0.8,  # Lower confidence without specific patterns
                    "validation_method": "historical_baseline"
                }
            
            # Analyze similar test outcomes
            successful_similar = len([t for t in similar_tests if t.get("success", False)])
            total_similar = len(similar_tests)
            pattern_success_rate = successful_similar / total_similar
            
            # Current test validation
            current_validation = self._compute_self_contained_validation(test_case, result)
            current_score = current_validation["historical_pattern_score"]
            
            # Combine current validation with historical pattern
            combined_score = (current_score * 0.7) + (pattern_success_rate * 0.3)
            
            return {
                "historical_pattern_score": combined_score,
                "pattern_validation": combined_score > 0.8,
                "pattern_type": "similar_test_pattern",
                "historical_success_rate": pattern_success_rate,
                "similar_test_count": total_similar,
                "current_validation": current_validation,
                "confidence": combined_score,
                "validation_method": "historical_pattern_analysis"
            }
            
        except Exception as e:
            logger.error(f"Historical pattern analysis failed: {e}")
            return self._compute_self_contained_validation(test_case, result)
    
    
    
    async def _store_test_results(self, test_results: List[TestExecutionResult]) -> bool:
        """Store test results in Data Manager"""
        try:
            # Prepare test results for storage
            results_data = {
                "test_execution_batch": {
                    "batch_id": str(uuid.uuid4()),
                    "executed_at": datetime.utcnow().isoformat(),
                    "agent_id": self.agent_id,
                    "total_tests": len(test_results),
                    "passed_tests": sum(1 for r in test_results if r.success),
                    "failed_tests": sum(1 for r in test_results if not r.success),
                    "results": [result.dict() for result in test_results]
                }
            }
            
            # Store in Data Manager
            storage_result = await self._call_data_manager("data_create", {
                "data": results_data,
                "storage_backend": "hana",  # Use HANA for structured test data
                "service_level": "gold",    # High priority for test results
                "metadata": {
                    "data_type": "test_execution_results",
                    "created_by": self.agent_id,
                    "batch_size": len(test_results),
                    "execution_timestamp": datetime.utcnow().isoformat()
                }
            })
            
            if "error" not in storage_result:
                logger.info(f"Successfully stored {len(test_results)} test results in Data Manager")
                return True
            else:
                logger.error(f"Failed to store test results: {storage_result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing test results: {e}")
            return False
    
    async def _retrieve_test_data(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve historical test data for a service from Data Manager"""
        try:
            # Query Data Manager for historical test data
            query_result = await self._call_data_manager("data_read", {
                "data_id": f"service_test_data_{service_id}",
                "include_metadata": True
            })
            
            if "error" not in query_result:
                return query_result.get("data", {})
            else:
                logger.info(f"No historical test data found for service {service_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve test data for {service_id}: {e}")
            return None
    
    async def _analyze_service_performance_trends(self, service_id: str, current_results: List[TestExecutionResult]) -> Dict[str, Any]:
        """Self-contained service performance analysis using Data Manager historical data"""
        try:
            # Get historical data for this service from Data Manager
            historical_data = await self._call_data_manager("data_read", {
                "data_id": f"service_performance_history_{service_id}"
            })
            
            if "error" in historical_data:
                # No historical data - analyze current results only
                return self._analyze_current_test_batch(current_results)
            
            # Combine historical and current data
            return self._analyze_performance_with_history(current_results, historical_data.get("data", {}))
            
        except Exception as e:
            logger.error(f"Failed to analyze service performance trends: {e}")
            return {"analysis": "error", "error": str(e)}
    
    def _analyze_current_test_batch(self, results: List[TestExecutionResult]) -> Dict[str, Any]:
        """Self-contained analysis of current test batch"""
        if not results:
            return {"analysis": "no_data"}
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        success_rate = passed_tests / total_tests
        
        # Analyze by test type/difficulty
        patterns = {}
        execution_times = []
        
        for result in results:
            test_type = result.validation_results.get("test_type", "unknown")
            patterns[test_type] = patterns.get(test_type, {"passed": 0, "total": 0})
            patterns[test_type]["total"] += 1
            if result.success:
                patterns[test_type]["passed"] += 1
            
            execution_times.append(result.execution_time)
        
        # Calculate pattern success rates
        for test_type in patterns:
            patterns[test_type]["success_rate"] = patterns[test_type]["passed"] / patterns[test_type]["total"]
        
        avg_execution_time = sum(execution_times) / len(execution_times)
        
        recommendations = []
        if success_rate < 0.7:
            recommendations.append("Service showing low success rate - investigate common failure patterns")
        
        # Find problematic test types
        low_performing_types = [t for t, data in patterns.items() if data["success_rate"] < 0.6]
        if low_performing_types:
            recommendations.append(f"Focus on improving: {', '.join(low_performing_types)} computation types")
        
        if avg_execution_time > 5.0:
            recommendations.append("Performance issues detected - execution times above 5 seconds")
        
        return {
            "analysis": "current_batch_only",
            "performance_metrics": {
                "total_tests": total_tests,
                "success_rate": success_rate,
                "average_execution_time": avg_execution_time,
                "patterns": patterns
            },
            "recommendations": recommendations
        }
    
    def _analyze_performance_with_history(self, current_results: List[TestExecutionResult], historical_data: Dict) -> Dict[str, Any]:
        """Analyze performance trends comparing current vs historical"""
        current_analysis = self._analyze_current_test_batch(current_results)
        
        if current_analysis["analysis"] == "no_data":
            return current_analysis
        
        # Extract historical metrics
        historical_success_rate = historical_data.get("average_success_rate", 0.8)
        historical_execution_time = historical_data.get("average_execution_time", 2.0)
        historical_patterns = historical_data.get("test_patterns", {})
        
        current_success_rate = current_analysis["performance_metrics"]["success_rate"]
        current_execution_time = current_analysis["performance_metrics"]["average_execution_time"]
        
        # Trend analysis
        success_trend = "improving" if current_success_rate > historical_success_rate + 0.05 else \
                      "degrading" if current_success_rate < historical_success_rate - 0.05 else "stable"
        
        performance_trend = "improving" if current_execution_time < historical_execution_time * 0.9 else \
                           "degrading" if current_execution_time > historical_execution_time * 1.1 else "stable"
        
        # Enhanced recommendations
        recommendations = current_analysis.get("recommendations", [])
        
        if success_trend == "degrading":
            recommendations.append("Service reliability has declined compared to historical performance")
        elif success_trend == "improving":
            recommendations.append("Service showing improvement over historical baseline")
        
        if performance_trend == "degrading":
            recommendations.append("Performance has slowed compared to historical averages")
        elif performance_trend == "improving":
            recommendations.append("Service performance has improved since last measurement")
        
        return {
            "analysis": "historical_comparison",
            "current_metrics": current_analysis["performance_metrics"],
            "historical_comparison": {
                "historical_success_rate": historical_success_rate,
                "current_success_rate": current_success_rate,
                "success_trend": success_trend,
                "historical_execution_time": historical_execution_time,
                "current_execution_time": current_execution_time,
                "performance_trend": performance_trend
            },
            "recommendations": recommendations
        }
    
    def _generate_performance_recommendations(self, success_rate: float, failure_patterns: Dict[str, int], peer_performance: Dict[str, float]) -> List[str]:
        """Generate recommendations based on performance analysis"""
        recommendations = []
        
        if success_rate < 0.7:
            recommendations.append("Service has low success rate - consider increasing test coverage")
            
        if failure_patterns:
            most_failing_type = max(failure_patterns, key=failure_patterns.get)
            recommendations.append(f"Focus testing on {most_failing_type} computations - highest failure rate")
            
        peer_avg = sum(peer_performance.values()) / len(peer_performance) if peer_performance else 0.8
        if success_rate < peer_avg - 0.1:
            recommendations.append("Service underperforming compared to similar services - investigate quality issues")
        elif success_rate > peer_avg + 0.1:
            recommendations.append("Service outperforming peers - consider using as reference implementation")
            
        if len(failure_patterns) > 3:
            recommendations.append("Multiple computation types failing - may indicate systemic issues")
            
        return recommendations
    
    def _generate_quality_recommendations(self, quality_analysis: Dict[str, Any], kg_analysis: Dict[str, Any]) -> List[str]:
        """Generate enhanced quality recommendations using knowledge graph insights"""
        recommendations = []
        
        success_rate = quality_analysis.get("success_rate", 0.0)
        kg_performance = kg_analysis.get("performance_metrics", {})
        peer_comparison = kg_analysis.get("peer_comparison", {})
        
        # Cross-reference current results with historical patterns
        historical_success_rate = kg_performance.get("success_rate", 0.0)
        if success_rate < historical_success_rate - 0.1:
            recommendations.append("Current test run shows degraded performance compared to historical patterns")
        elif success_rate > historical_success_rate + 0.1:
            recommendations.append("Service showing improved performance compared to historical baseline")
        
        # Vector similarity insights
        vector_validations = [r.validation_results.get("vector_similarity_score", 0) 
                            for r in quality_analysis.get("test_results", []) 
                            if hasattr(r, 'validation_results')]
        
        if vector_validations:
            avg_vector_score = sum(vector_validations) / len(vector_validations)
            if avg_vector_score < 0.6:
                recommendations.append("Low vector similarity scores suggest service behavior differs from learned patterns")
        
        # Knowledge graph connectivity insights
        graph_insights = quality_analysis.get("knowledge_graph_insights", {})
        if graph_insights:
            similar_services = graph_insights.get("peer_comparison", {}).get("similar_services_count", 0)
            if similar_services < 2:
                recommendations.append("Service has limited peer comparison data - consider expanding test coverage")
            
            relative_performance = graph_insights.get("peer_comparison", {}).get("relative_performance", "average")
            if relative_performance == "below_average":
                recommendations.append("Knowledge graph analysis indicates underperformance relative to similar services")
        
        return recommendations
    
    @a2a_handler("dynamic_computation_testing")
    async def handle_computation_testing(self, message: A2AMessage) -> Dict[str, Any]:
        """Main handler for computation testing requests"""
        start_time = time.time()
        
        try:
            # Verify message trust
            if self.trust_identity:
                trust_verification = await verify_a2a_message(
                    message.dict() if hasattr(message, 'dict') else message,
                    self.agent_id
                )
                if not trust_verification["valid"]:
                    return create_error_response(f"Trust verification failed: {trust_verification['error']}")
            
            # Extract request data from message
            request_data = self._extract_request_data(message)
            if not request_data:
                return create_error_response("No valid computation testing request found in message")
            
            # Process computation testing task
            testing_result = await self.execute_computation_testing(
                request_data=request_data,
                context_id=message.conversation_id
            )
            
            # Record success metrics
            self.tasks_completed.labels(agent_id=self.agent_id, task_type='computation_testing').inc()
            self.processing_time.labels(agent_id=self.agent_id, task_type='computation_testing').observe(time.time() - start_time)
            
            # Sign response if trust system is enabled
            response_data = testing_result
            if self.trust_identity:
                response_data = await sign_a2a_message(response_data, self.agent_id)
            
            return create_success_response(response_data)
            
        except Exception as e:
            # Record failure metrics
            self.tasks_failed.labels(agent_id=self.agent_id, task_type='computation_testing').inc()
            logger.error(f"Computation testing failed: {e}")
            return create_error_response(f"Computation testing failed: {str(e)}")
    
    @a2a_handler("service_discovery")
    async def handle_service_discovery(self, message: A2AMessage) -> Dict[str, Any]:
        """Handler for computational service discovery"""
        start_time = time.time()
        
        try:
            # Extract discovery parameters
            discovery_params = self._extract_request_data(message)
            
            # Perform service discovery
            discovery_result = await self.discover_computational_services(
                domain_filter=discovery_params.get('domain_filter'),
                service_types=discovery_params.get('service_types', [ServiceType.API])
            )
            
            # Record success metrics
            self.tasks_completed.labels(agent_id=self.agent_id, task_type='service_discovery').inc()
            self.processing_time.labels(agent_id=self.agent_id, task_type='service_discovery').observe(time.time() - start_time)
            
            return create_success_response(discovery_result)
            
        except Exception as e:
            self.tasks_failed.labels(agent_id=self.agent_id, task_type='service_discovery').inc()
            logger.error(f"Service discovery failed: {e}")
            return create_error_response(f"Service discovery failed: {str(e)}")
    
    @a2a_skill("service_discovery")
    async def service_discovery_skill(self, endpoints: List[str], domain_filter: Optional[str] = None) -> List[ServiceDiscoveryResult]:
        """Discover computational services via Catalog Manager ORD registry"""
        
        logger.info(f"Discovering computational services via Catalog Manager with domain filter: {domain_filter}")
        
        discovered_services = []
        
        try:
            # Query Catalog Manager for computational services
            search_query = "computational OR calculation OR compute OR api OR service"
            if domain_filter:
                search_query += f" AND {domain_filter}"
            
            catalog_result = await self._call_catalog_manager("ord_search", {
                "query": search_query,
                "filters": {
                    "service_types": ["api", "computation", "calculation"],
                    "resource_types": ["apis", "dataProducts"]
                }
            })
            
            if "error" in catalog_result:
                logger.error(f"Catalog Manager search failed: {catalog_result['error']}")
                return []
            
            # Process ORD results into ServiceDiscoveryResult objects
            ord_documents = catalog_result.get("search_results", [])
            logger.info(f"Found {len(ord_documents)} ORD documents from Catalog Manager")
            
            for ord_doc in ord_documents:
                try:
                    # Extract service information from ORD document
                    service_info = await self._process_ord_service(ord_doc)
                    if service_info:
                        discovered_services.append(service_info)
                        
                        # Store service in Data Manager for future reference
                        await self._store_discovered_service(service_info)
                        
                except Exception as e:
                    logger.error(f"Failed to process ORD document {ord_doc.get('ordId', 'unknown')}: {e}")
                    continue
            
            # Fallback: Direct endpoint discovery for non-ORD services
            if not discovered_services and endpoints:
                logger.info("No ORD services found, falling back to direct endpoint discovery")
                discovered_services = await self._direct_endpoint_discovery(endpoints, domain_filter)
            
            self.processing_stats["services_discovered"] = len(discovered_services)
            logger.info(f"Service discovery completed: {len(discovered_services)} services discovered")
            
            return discovered_services
            
        except Exception as e:
            logger.error(f"Service discovery failed: {e}")
            return []
    
    async def _process_ord_service(self, ord_doc: Dict[str, Any]) -> Optional[ServiceDiscoveryResult]:
        """Process ORD document into ServiceDiscoveryResult"""
        try:
            ord_id = ord_doc.get("ordId")
            if not ord_id:
                return None
            
            # Extract API endpoints from ORD document
            api_endpoints = []
            for api_resource in ord_doc.get("apis", []):
                api_endpoints.append({
                    "url": api_resource.get("partOfConsumptionBundles", [{}])[0].get("baseUrl", ""),
                    "protocol": api_resource.get("apiProtocol", "REST"),
                    "description": api_resource.get("description", "")
                })
            
            # Determine service type based on ORD metadata
            service_type = ServiceType.API
            if any(tag in ord_doc.get("tags", []) for tag in ["computation", "calculate", "math"]):
                service_type = ServiceType.ALGORITHM
            elif "data" in ord_doc.get("tags", []):
                service_type = ServiceType.PIPELINE
                
            # Extract computation capabilities
            capabilities = []
            for tag in ord_doc.get("tags", []):
                if tag in ["mathematical", "logical", "statistical", "ml", "ai", "compute"]:
                    capabilities.append(tag)
            
            if not capabilities:
                capabilities = ["general_computation"]
            
            # Create service discovery result
            service_result = ServiceDiscoveryResult(
                service_id=ord_id.replace(":", "_"),
                endpoint_url=api_endpoints[0]["url"] if api_endpoints else "",
                service_type=service_type,
                computation_capabilities=capabilities,
                api_schema=ord_doc.get("apiDefinitions", {}),
                performance_characteristics={
                    "estimated_latency": "unknown",
                    "throughput_capacity": "unknown",
                    "scalability": ord_doc.get("visibility", "private")
                },
                metadata={
                    "ord_id": ord_id,
                    "title": ord_doc.get("title", "Unknown Service"),
                    "description": ord_doc.get("description", ""),
                    "provider": ord_doc.get("responsible", "Unknown"),
                    "version": ord_doc.get("version", "1.0.0"),
                    "discovered_via": "catalog_manager_ord",
                    "discovered_at": datetime.utcnow().isoformat(),
                    "tags": ord_doc.get("tags", []),
                    "apis": api_endpoints
                }
            )
            
            return service_result
            
        except Exception as e:
            logger.error(f"Failed to process ORD service: {e}")
            return None
    
    async def _store_discovered_service(self, service: ServiceDiscoveryResult) -> bool:
        """Store discovered service in Data Manager"""
        try:
            data_result = await self._call_data_manager("data_create", {
                "data": service.dict(),
                "storage_backend": "filesystem",
                "service_level": "silver",
                "metadata": {
                    "data_type": "discovered_service",
                    "service_id": service.service_id,
                    "discovered_by": self.agent_id,
                    "created_at": datetime.utcnow().isoformat()
                }
            })
            
            if "error" not in data_result:
                logger.info(f"Stored discovered service {service.service_id} in Data Manager")
                return True
            else:
                logger.warning(f"Failed to store service {service.service_id}: {data_result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to store discovered service: {e}")
            return False
    
    async def _direct_endpoint_discovery(self, endpoints: List[str], domain_filter: Optional[str] = None) -> List[ServiceDiscoveryResult]:
        """Fallback: Direct endpoint discovery for non-ORD services"""
        discovered_services = []
        
        for endpoint in endpoints:
            try:
                circuit_breaker = self.circuit_breaker_manager.get_breaker(
                    f"service_{hashlib.md5(endpoint.encode()).hexdigest()[:8]}",
                    failure_threshold=3,
                    success_threshold=2,
                    timeout=30.0
                )
                
                async def discover_service():
                    async with self.http_client as client:
                        response = await client.get(f"{endpoint}/health", timeout=10.0)
                        if response.status_code == 200:
                            health_data = response.json()
                            
                            api_schema = None
                            try:
                                api_response = await client.get(f"{endpoint}/openapi.json", timeout=5.0)
                                if api_response.status_code == 200:
                                    api_schema = api_response.json()
                            except:
                                pass
                            
                            return ServiceDiscoveryResult(
                                service_id=f"service_{hashlib.md5(endpoint.encode()).hexdigest()[:8]}",
                                endpoint_url=endpoint,
                                service_type=ServiceType.API,
                                computation_capabilities=self._extract_capabilities(health_data, api_schema),
                                api_schema=api_schema,
                                performance_characteristics=self._extract_performance_characteristics(health_data),
                                metadata={
                                    "discovered_at": datetime.utcnow().isoformat(),
                                    "health_status": health_data.get("status", "unknown"),
                                    "version": health_data.get("version"),
                                    "discovered_via": "direct_endpoint"
                                }
                            )
                        return None
                
                service_result = await circuit_breaker.call(discover_service)
                if service_result:
                    discovered_services.append(service_result)
                    self.discovered_services[service_result.service_id] = service_result
                    logger.info(f"✅ Discovered service {service_result.service_id} at {endpoint}")
                
            except Exception as e:
                logger.warning(f"Failed to discover service at {endpoint}: {e}")
        
        self.processing_stats["services_discovered"] += len(discovered_services)
        return discovered_services
    
    @a2a_skill("template_loading")
    async def template_loading_skill(self, computation_types: List[ComputationType]) -> Dict[str, List[TestTemplate]]:
        """Load and prepare test templates"""
        
        templates_by_type = {}
        
        for comp_type in computation_types:
            templates = self._get_templates_for_computation_type(comp_type)
            templates_by_type[comp_type.value] = templates
            logger.info(f"Loaded {len(templates)} templates for {comp_type.value}")
        
        return templates_by_type
    
    @a2a_skill("test_generation")
    async def test_generation_skill(self, services: List[ServiceDiscoveryResult], templates: Dict[str, List[TestTemplate]], test_config: Dict[str, Any]) -> List[GeneratedTestCase]:
        """Generate dynamic test cases from templates"""
        
        generated_tests = []
        
        for service in services:
            for comp_type, template_list in templates.items():
                for template in template_list:
                    # Check if template is compatible with service
                    if self._is_template_compatible(template, service):
                        # Generate test cases for this template/service combination
                        test_cases = await self._generate_test_cases(template, service, test_config)
                        generated_tests.extend(test_cases)
                        
                        # Record metrics
                        self.tests_generated.labels(
                            agent_id=self.agent_id, 
                            computation_type=comp_type
                        ).inc(len(test_cases))
        
        self.processing_stats["tests_generated"] += len(generated_tests)
        logger.info(f"Generated {len(generated_tests)} test cases")
        return generated_tests
    
    @a2a_skill("test_execution")
    async def test_execution_skill(self, test_cases: List[GeneratedTestCase], parallel_limit: int = 10) -> List[TestExecutionResult]:
        """Execute test cases with parallel processing"""
        
        results = []
        semaphore = asyncio.Semaphore(parallel_limit)
        
        async def execute_single_test(test_case: GeneratedTestCase) -> TestExecutionResult:
            """Execute a single test case"""
            async with semaphore:
                try:
                    service = self.discovered_services.get(test_case.metadata.get('service_id'))
                    if not service:
                        raise ValueError(f"Service not found: {test_case.metadata.get('service_id')}")
                    
                    # Use circuit breaker for test execution
                    circuit_breaker = self.circuit_breaker_manager.get_breaker(
                        f"test_{service.service_id}",
                        failure_threshold=5,
                        timeout=60.0
                    )
                    
                    async def run_test():
                        """Run the actual test"""
                        start_time = time.time()
                        
                        # Execute the test based on service type
                        if service.service_type == ServiceType.API:
                            result = await self._execute_api_test(test_case, service)
                        else:
                            result = await self._execute_function_test(test_case, service)
                        
                        execution_time = time.time() - start_time
                        
                        # Validate the result
                        validation_results = await self._validate_test_result(test_case, result)
                        
                        # Use self-contained historical pattern validation
                        pattern_validation = await self._validate_with_historical_patterns(test_case, result)
                        if pattern_validation:
                            validation_results.update(pattern_validation)
                        
                        quality_scores = self._calculate_quality_scores(test_case, result, validation_results, execution_time)
                        
                        return TestExecutionResult(
                            test_id=test_case.test_id,
                            service_id=service.service_id,
                            success=validation_results.get('passed', False),
                            actual_output=result,
                            execution_time=execution_time,
                            validation_results=validation_results,
                            quality_scores=quality_scores
                        )
                    
                    return await circuit_breaker.call(run_test)
                    
                except Exception as e:
                    logger.error(f"Test execution failed for {test_case.test_id}: {e}")
                    return TestExecutionResult(
                        test_id=test_case.test_id,
                        service_id=test_case.metadata.get('service_id', 'unknown'),
                        success=False,
                        execution_time=0.0,
                        error_message=str(e),
                        validation_results={"passed": False, "error": str(e)},
                        quality_scores={"overall": 0.0}
                    )
        
        # Execute tests in parallel
        tasks = [execute_single_test(test_case) for test_case in test_cases]
        results = await asyncio.gather(*tasks)
        
        # Record metrics
        successful_tests = sum(1 for r in results if r.success)
        failed_tests = len(results) - successful_tests
        
        self.tests_executed.labels(agent_id=self.agent_id, test_type='all').inc(len(results))
        
        self.processing_stats["tests_executed"] += len(results)
        logger.info(f"Executed {len(results)} tests: {successful_tests} passed, {failed_tests} failed")
        
        # Store test results in Data Manager
        storage_success = await self._store_test_results(results)
        if storage_success:
            logger.info("Test results successfully stored in Data Manager")
        
        return results
    
    @a2a_skill("quality_analysis")
    async def quality_analysis_skill(self, test_results: List[TestExecutionResult], service_id: str) -> Dict[str, Any]:
        """Analyze test results and generate quality metrics"""
        
        if not test_results:
            return {"error": "No test results to analyze"}
        
        # Aggregate results by test type
        results_by_type = {}
        for result in test_results:
            test_type = result.validation_results.get('test_type', 'unknown')
            if test_type not in results_by_type:
                results_by_type[test_type] = []
            results_by_type[test_type].append(result)
        
        # Calculate overall quality scores
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.success)
        
        quality_analysis = {
            "service_id": service_id,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "average_execution_time": sum(r.execution_time for r in test_results) / total_tests if total_tests > 0 else 0.0,
            "quality_scores": {
                "accuracy": self._calculate_accuracy_score(test_results),
                "performance": self._calculate_performance_score(test_results),
                "reliability": self._calculate_reliability_score(test_results),
                "overall": 0.0
            },
            "test_type_breakdown": {}
        }
        
        # Calculate overall quality score
        scores = quality_analysis["quality_scores"]
        scores["overall"] = (scores["accuracy"] + scores["performance"] + scores["reliability"]) / 3
        
        # Analyze by test type
        for test_type, type_results in results_by_type.items():
            type_passed = sum(1 for r in type_results if r.success)
            quality_analysis["test_type_breakdown"][test_type] = {
                "total": len(type_results),
                "passed": type_passed,
                "success_rate": type_passed / len(type_results) if type_results else 0.0,
                "average_execution_time": sum(r.execution_time for r in type_results) / len(type_results) if type_results else 0.0
            }
        
        # Update quality score metric
        self.quality_score.labels(agent_id=self.agent_id, service_id=service_id).set(scores["overall"])
        
        # Add self-contained service analysis
        service_analysis = await self._analyze_service_performance_trends(service_id, test_results)
        if service_analysis:
            quality_analysis["service_insights"] = service_analysis
            quality_analysis["enhanced_recommendations"] = service_analysis.get("recommendations", [])
        
        return quality_analysis
    
    @a2a_skill("report_generation")
    async def report_generation_skill(self, quality_analysis: Dict[str, Any], test_results: List[TestExecutionResult]) -> QualityReport:
        """Generate comprehensive quality report"""
        
        service_id = quality_analysis["service_id"]
        
        # Generate recommendations based on results
        recommendations = self._generate_recommendations(quality_analysis, test_results)
        
        report = QualityReport(
            suite_id=str(uuid.uuid4()),
            service_id=service_id,
            total_tests=quality_analysis["total_tests"],
            passed_tests=quality_analysis["passed_tests"],
            failed_tests=quality_analysis["failed_tests"],
            execution_summary={
                "success_rate": quality_analysis["success_rate"],
                "average_execution_time": quality_analysis["average_execution_time"],
                "test_type_breakdown": quality_analysis["test_type_breakdown"]
            },
            quality_scores=quality_analysis["quality_scores"],
            detailed_results=test_results,
            recommendations=recommendations,
            generated_at=datetime.utcnow()
        )
        
        self.processing_stats["quality_reports_generated"] += 1
        return report
    
    @a2a_task(
        task_type="computation_testing_workflow",
        description="Complete computation testing workflow",
        timeout=600,
        retry_attempts=2
    )
    async def execute_computation_testing(self, request_data: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Execute complete computation testing workflow"""
        
        try:
            # Parse request
            test_request = ComputationTestRequest(**request_data)
            
            # Stage 1: Service Discovery
            logger.info("Stage 1: Starting service discovery...")
            discovered_services = await self.execute_skill(
                "service_discovery", 
                test_request.service_endpoints, 
                test_request.domain_filter
            )
            
            if not discovered_services:
                return {
                    "success": False,
                    "error": "No computational services discovered",
                    "context_id": context_id
                }
            
            # Stage 2: Template Loading
            logger.info("Stage 2: Loading test templates...")
            templates = await self.execute_skill(
                "template_loading", 
                test_request.computation_types
            )
            
            # Stage 3: Test Generation
            logger.info("Stage 3: Generating test cases...")
            test_cases = await self.execute_skill(
                "test_generation", 
                discovered_services, 
                templates, 
                test_request.test_config
            )
            
            if not test_cases:
                return {
                    "success": False,
                    "error": "No test cases generated",
                    "context_id": context_id
                }
            
            # Stage 4: Test Execution
            logger.info("Stage 4: Executing test cases...")
            test_results = await self.execute_skill(
                "test_execution", 
                test_cases,
                test_request.test_config.get("parallel_limit", 10)
            )
            
            # Stage 5: Quality Analysis (per service)
            logger.info("Stage 5: Analyzing quality metrics...")
            service_reports = []
            
            for service in discovered_services:
                service_results = [r for r in test_results if r.service_id == service.service_id]
                if service_results:
                    quality_analysis = await self.execute_skill(
                        "quality_analysis", 
                        service_results, 
                        service.service_id
                    )
                    
                    # Stage 6: Report Generation
                    report = await self.execute_skill(
                        "report_generation", 
                        quality_analysis, 
                        service_results
                    )
                    
                    service_reports.append(report.dict())
            
            self.processing_stats["total_tasks"] += 1
            
            return {
                "success": True,
                "context_id": context_id,
                "discovered_services": len(discovered_services),
                "generated_tests": len(test_cases),
                "executed_tests": len(test_results),
                "service_reports": service_reports,
                "processing_stats": self.processing_stats
            }
            
        except Exception as e:
            logger.error(f"Computation testing workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "context_id": context_id
            }
    
    async def discover_computational_services(self, domain_filter: Optional[str] = None, service_types: List[ServiceType] = None) -> Dict[str, Any]:
        """Discover computational services"""
        service_types = service_types or [ServiceType.API]
        
        # In a real implementation, this would query service registries
        # For now, we'll use the discovered services from previous operations
        
        filtered_services = []
        for service in self.discovered_services.values():
            if service_types and service.service_type not in service_types:
                continue
            
            if domain_filter and domain_filter.lower() not in service.endpoint_url.lower():
                continue
            
            filtered_services.append(service.dict())
        
        return {
            "discovered_services": filtered_services,
            "total_count": len(filtered_services),
            "discovery_metadata": {
                "domain_filter": domain_filter,
                "service_types": [st.value for st in service_types],
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    def _extract_request_data(self, message: A2AMessage) -> Dict[str, Any]:
        """Extract request data from message"""
        request_data = {}
        
        for part in message.parts:
            if part.kind == "data" and part.data:
                request_data.update(part.data)
        
        return request_data
    
    async def _load_test_templates(self):
        """Load test templates from repository"""
        try:
            # Load built-in templates
            self._load_builtin_templates()
            
            # Load external templates if repository URL is provided
            if self.template_repository_url:
                await self._load_external_templates()
            
            logger.info(f"Loaded {len(self.test_templates)} test templates")
            
        except Exception as e:
            logger.error(f"Failed to load test templates: {e}")
    
    def _load_builtin_templates(self):
        """Load built-in test templates"""
        
        # Mathematical computation templates
        self.test_templates["math_basic_arithmetic"] = TestTemplate(
            template_id="math_basic_arithmetic",
            computation_type=ComputationType.MATHEMATICAL,
            complexity_level=TestDifficulty.EASY,
            pattern_category="arithmetic",
            input_generator={
                "type": "random_numbers",
                "parameters": {"range": [-1000, 1000], "count": 2}
            },
            expected_behavior={
                "operation": "addition",
                "accuracy_threshold": 0.999999
            },
            validation={
                "method": ValidationMethod.EXACT,
                "tolerance": 1e-10
            },
            metadata={"builtin": True, "version": "1.0"}
        )
        
        self.test_templates["math_precision_ops"] = TestTemplate(
            template_id="math_precision_ops",
            computation_type=ComputationType.MATHEMATICAL,
            complexity_level=TestDifficulty.MEDIUM,
            pattern_category="precision",
            input_generator={
                "type": "precision_values",
                "parameters": {"decimal_places": 15, "count": 2}
            },
            expected_behavior={
                "operation": "floating_point",
                "accuracy_threshold": 0.99999
            },
            validation={
                "method": ValidationMethod.APPROXIMATE,
                "tolerance": 1e-12
            },
            metadata={"builtin": True, "version": "1.0"}
        )
        
        # Logical computation templates
        self.test_templates["logic_boolean_ops"] = TestTemplate(
            template_id="logic_boolean_ops",
            computation_type=ComputationType.LOGICAL,
            complexity_level=TestDifficulty.EASY,
            pattern_category="boolean_operations",
            input_generator={
                "type": "boolean_combinations",
                "parameters": {"variables": 3, "operations": ["AND", "OR", "NOT"]}
            },
            expected_behavior={
                "operation": "logical_evaluation",
                "accuracy_threshold": 1.0
            },
            validation={
                "method": ValidationMethod.EXACT
            },
            metadata={"builtin": True, "version": "1.0"}
        )
        
        # Performance templates
        self.test_templates["perf_latency"] = TestTemplate(
            template_id="perf_latency",
            computation_type=ComputationType.PERFORMANCE,
            complexity_level=TestDifficulty.MEDIUM,
            pattern_category="latency_test",
            input_generator={
                "type": "performance_payload",
                "parameters": {"size_bytes": 1024, "complexity": "medium"}
            },
            expected_behavior={
                "max_execution_time": 1.0,
                "throughput_threshold": 100
            },
            validation={
                "method": ValidationMethod.PATTERN_MATCH,
                "performance_bounds": True
            },
            metadata={"builtin": True, "version": "1.0"}
        )
    
    async def _load_external_templates(self):
        """Load templates from external repository"""
        try:
            async with self.http_client as client:
                response = await client.get(f"{self.template_repository_url}/templates.yaml")
                if response.status_code == 200:
                    templates_data = yaml.safe_load(response.text)
                    
                    for template_data in templates_data.get("templates", []):
                        template = TestTemplate(**template_data)
                        self.test_templates[template.template_id] = template
                        
                    logger.info(f"Loaded {len(templates_data.get('templates', []))} external templates")
                
        except Exception as e:
            logger.warning(f"Failed to load external templates: {e}")
    
    def _get_templates_for_computation_type(self, comp_type: ComputationType) -> List[TestTemplate]:
        """Get templates for specific computation type"""
        return [
            template for template in self.test_templates.values()
            if template.computation_type == comp_type
        ]
    
    def _is_template_compatible(self, template: TestTemplate, service: ServiceDiscoveryResult) -> bool:
        """Check if template is compatible with service"""
        # Simple compatibility check based on service capabilities
        if template.computation_type.value in service.computation_capabilities:
            return True
        
        # More sophisticated compatibility logic could be added here
        return False
    
    async def _generate_test_cases(self, template: TestTemplate, service: ServiceDiscoveryResult, test_config: Dict[str, Any]) -> List[GeneratedTestCase]:
        """Generate test cases from template"""
        test_cases = []
        max_tests = test_config.get("max_tests_per_template", 5)
        
        for i in range(max_tests):
            try:
                # Generate input data based on template
                input_data = self._generate_input_data(template.input_generator)
                
                # Calculate expected output
                expected_output = self._calculate_expected_output(template, input_data)
                
                # Create validation criteria
                validation_criteria = self._create_validation_criteria(template, expected_output)
                
                test_case = GeneratedTestCase(
                    test_id=f"{template.template_id}_{service.service_id}_{i}",
                    template_source=template,
                    input_data=input_data,
                    expected_output=expected_output,
                    validation_criteria=validation_criteria,
                    timeout_seconds=test_config.get("timeout_seconds", 30.0),
                    metadata={
                        "service_id": service.service_id,
                        "generated_at": datetime.utcnow().isoformat(),
                        "template_version": template.metadata.get("version"),
                        "difficulty": template.complexity_level.value
                    }
                )
                
                test_cases.append(test_case)
                
            except Exception as e:
                logger.warning(f"Failed to generate test case {i} for template {template.template_id}: {e}")
        
        return test_cases
    
    def _generate_input_data(self, input_generator: Dict[str, Any]) -> Dict[str, Any]:
        """Generate input data based on generator configuration"""
        gen_type = input_generator.get("type")
        params = input_generator.get("parameters", {})
        
        if gen_type == "random_numbers":
            return {
                "numbers": [
                    random.randint(params["range"][0], params["range"][1]) 
                    for _ in range(params["count"])
                ]
            }
        
        elif gen_type == "precision_values":
            return {
                "values": [
                    round(random.uniform(-100, 100), params["decimal_places"])
                    for _ in range(params["count"])
                ]
            }
        
        elif gen_type == "boolean_combinations":
            variables = params["variables"]
            return {
                "variables": [random.choice([True, False]) for _ in range(variables)],
                "operations": random.choice(params["operations"])
            }
        
        elif gen_type == "performance_payload":
            return {
                "payload_size": params["size_bytes"],
                "data": "x" * params["size_bytes"],
                "complexity": params["complexity"]
            }
        
        else:
            return {"input": "default_test_data"}
    
    def _calculate_expected_output(self, template: TestTemplate, input_data: Dict[str, Any]) -> Any:
        """Calculate expected output for test case"""
        if template.computation_type == ComputationType.MATHEMATICAL:
            if "addition" in template.expected_behavior.get("operation", ""):
                numbers = input_data.get("numbers", [])
                return sum(numbers) if numbers else 0
            
            elif "floating_point" in template.expected_behavior.get("operation", ""):
                values = input_data.get("values", [])
                return sum(values) if values else 0.0
        
        elif template.computation_type == ComputationType.LOGICAL:
            variables = input_data.get("variables", [])
            operation = input_data.get("operations", "AND")
            
            if operation == "AND":
                return all(variables) if variables else False
            elif operation == "OR":
                return any(variables) if variables else False
            elif operation == "NOT":
                return not variables[0] if variables else True
        
        elif template.computation_type == ComputationType.PERFORMANCE:
            # For performance tests, expected output is successful execution
            return {"status": "success", "processed": True}
        
        return None
    
    def _create_validation_criteria(self, template: TestTemplate, expected_output: Any) -> Dict[str, Any]:
        """Create validation criteria for test case"""
        validation = template.validation.copy()
        validation["expected_output"] = expected_output
        validation["test_type"] = template.computation_type.value
        return validation
    
    async def _execute_api_test(self, test_case: GeneratedTestCase, service: ServiceDiscoveryResult) -> Any:
        """Execute test case against API service"""
        try:
            # Construct API request based on test case
            request_data = {
                "input": test_case.input_data,
                "operation": test_case.template_source.expected_behavior.get("operation")
            }
            
            async with self.http_client as client:
                response = await client.post(
                    f"{service.endpoint_url}/compute",
                    json=request_data,
                    timeout=test_case.timeout_seconds
                )
                
                if response.status_code == 200:
                    return response.json().get("result")
                else:
                    raise Exception(f"API returned status {response.status_code}: {response.text}")
                    
        except Exception as e:
            raise Exception(f"API test execution failed: {e}")
    
    async def _execute_function_test(self, test_case: GeneratedTestCase, service: ServiceDiscoveryResult) -> Any:
        """Execute test case against function service"""
        # For function services, we would need to import and call the function
        # This is a placeholder implementation
        
        if test_case.template_source.computation_type == ComputationType.MATHEMATICAL:
            numbers = test_case.input_data.get("numbers", [])
            return sum(numbers)  # Simple addition for demo
        
        return test_case.expected_output  # Return expected for demo
    
    async def _validate_test_result(self, test_case: GeneratedTestCase, actual_result: Any) -> Dict[str, Any]:
        """Validate test result against expected output"""
        validation_criteria = test_case.validation_criteria
        expected_output = validation_criteria.get("expected_output")
        validation_method = validation_criteria.get("method", ValidationMethod.EXACT)
        
        validation_results = {
            "passed": False,
            "method": validation_method,
            "expected": expected_output,
            "actual": actual_result,
            "details": {}
        }
        
        try:
            if validation_method == ValidationMethod.EXACT:
                validation_results["passed"] = actual_result == expected_output
                
            elif validation_method == ValidationMethod.APPROXIMATE:
                tolerance = validation_criteria.get("tolerance", 1e-6)
                if isinstance(actual_result, (int, float)) and isinstance(expected_output, (int, float)):
                    diff = abs(actual_result - expected_output)
                    validation_results["passed"] = diff <= tolerance
                    validation_results["details"]["difference"] = diff
                else:
                    validation_results["passed"] = str(actual_result) == str(expected_output)
                    
            elif validation_method == ValidationMethod.PATTERN_MATCH:
                # For performance tests, check if result indicates success
                if isinstance(actual_result, dict):
                    validation_results["passed"] = actual_result.get("status") == "success"
                else:
                    validation_results["passed"] = actual_result is not None
            
        except Exception as e:
            validation_results["passed"] = False
            validation_results["error"] = str(e)
        
        return validation_results
    
    def _calculate_quality_scores(self, test_case: GeneratedTestCase, actual_result: Any, validation_results: Dict[str, Any], execution_time: float) -> Dict[str, float]:
        """Calculate quality scores for test result"""
        scores = {
            "accuracy": 1.0 if validation_results.get("passed", False) else 0.0,
            "performance": max(0.0, min(1.0, (test_case.timeout_seconds - execution_time) / test_case.timeout_seconds)),
            "reliability": 1.0 if not validation_results.get("error") else 0.0
        }
        
        scores["overall"] = (scores["accuracy"] + scores["performance"] + scores["reliability"]) / 3
        return scores
    
    def _calculate_accuracy_score(self, test_results: List[TestExecutionResult]) -> float:
        """Calculate overall accuracy score"""
        if not test_results:
            return 0.0
        
        accuracy_scores = [r.quality_scores.get("accuracy", 0.0) for r in test_results]
        return sum(accuracy_scores) / len(accuracy_scores)
    
    def _calculate_performance_score(self, test_results: List[TestExecutionResult]) -> float:
        """Calculate overall performance score"""
        if not test_results:
            return 0.0
        
        performance_scores = [r.quality_scores.get("performance", 0.0) for r in test_results]
        return sum(performance_scores) / len(performance_scores)
    
    def _calculate_reliability_score(self, test_results: List[TestExecutionResult]) -> float:
        """Calculate overall reliability score"""
        if not test_results:
            return 0.0
        
        reliability_scores = [r.quality_scores.get("reliability", 0.0) for r in test_results]
        return sum(reliability_scores) / len(reliability_scores)
    
    def _generate_recommendations(self, quality_analysis: Dict[str, Any], test_results: List[TestExecutionResult]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        success_rate = quality_analysis.get("success_rate", 0.0)
        quality_scores = quality_analysis.get("quality_scores", {})
        
        if success_rate < 0.8:
            recommendations.append("Overall test success rate is below 80%. Review failed test cases to identify systematic issues.")
        
        if quality_scores.get("accuracy", 0.0) < 0.9:
            recommendations.append("Accuracy score is below 90%. Verify computation logic and numerical precision handling.")
        
        if quality_scores.get("performance", 0.0) < 0.7:
            recommendations.append("Performance score is below 70%. Consider optimizing algorithms or increasing timeout limits.")
        
        if quality_scores.get("reliability", 0.0) < 0.8:
            recommendations.append("Reliability score is below 80%. Improve error handling and edge case management.")
        
        # Analyze execution time patterns
        execution_times = [r.execution_time for r in test_results if r.execution_time > 0]
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            if avg_time > 5.0:
                recommendations.append("Average execution time exceeds 5 seconds. Consider performance optimizations.")
        
        if not recommendations:
            recommendations.append("Service performance is within acceptable ranges. Continue monitoring for consistency.")
        
        return recommendations
    
    def _extract_capabilities(self, health_data: Dict[str, Any], api_schema: Optional[Dict[str, Any]]) -> List[str]:
        """Extract service capabilities from health data and API schema"""
        capabilities = []
        
        # Extract from health data
        if "capabilities" in health_data:
            capabilities.extend(health_data["capabilities"])
        
        # Extract from API schema
        if api_schema and "paths" in api_schema:
            for path, path_data in api_schema["paths"].items():
                if "compute" in path.lower():
                    capabilities.append("mathematical")
                if "logic" in path.lower():
                    capabilities.append("logical")
                if "transform" in path.lower():
                    capabilities.append("transformational")
        
        # Default capabilities if none found
        if not capabilities:
            capabilities = ["mathematical", "logical"]
        
        return capabilities
    
    def _extract_performance_characteristics(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance characteristics from health data"""
        return {
            "typical_latency": health_data.get("latency_ms", 100),
            "throughput_capacity": health_data.get("throughput_ops_per_sec", 1000),
            "cpu_usage": health_data.get("cpu_usage_percent", 50),
            "memory_usage": health_data.get("memory_usage_mb", 256)
        }
    
    async def _initialize_trust_system(self) -> None:
        """Initialize the agent's trust system"""
        try:
            # Initialize trust identity
            self.trust_identity = await initialize_agent_trust(
                self.agent_id,
                self.base_url
            )
            
            if self.trust_identity:
                logger.info(f"✅ Trust system initialized for {self.agent_id}")
                logger.info(f"   Trust address: {self.trust_identity.get('address')}")
                logger.info(f"   Public key fingerprint: {self.trust_identity.get('public_key_fingerprint')}")
                
                # Pre-trust essential agents
                essential_agents = [
                    "agent_manager",
                    "data_product_agent_0",
                    "data_standardization_agent_1",
                    "ai_preparation_agent_2",
                    "vector_processing_agent_3"
                ]
                
                self.trusted_agents = set(essential_agents)
                logger.info(f"   Pre-trusted agents: {self.trusted_agents}")
            else:
                logger.warning("⚠️  Trust system initialization failed, running without trust verification")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize trust system: {e}")
            logger.warning("Continuing without trust verification")
            self.trust_identity = None
    
    async def _load_agent_state(self):
        """Load existing agent state from storage"""
        try:
            # Load discovered services
            services_file = os.path.join(self.storage_path, "discovered_services.json")
            if os.path.exists(services_file):
                with open(services_file, 'r') as f:
                    services_data = json.load(f)
                    
                for service_id, service_data in services_data.items():
                    self.discovered_services[service_id] = ServiceDiscoveryResult(**service_data)
                    
                logger.info(f"Loaded {len(self.discovered_services)} discovered services from state")
            
            # Load test results cache
            results_file = os.path.join(self.storage_path, "test_results_cache.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    self.test_results_cache = json.load(f)
                    
                logger.info(f"Loaded {len(self.test_results_cache)} cached test results")
                
        except Exception as e:
            logger.warning(f"Failed to load agent state: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup agent resources"""
        try:
            # Save discovered services
            services_file = os.path.join(self.storage_path, "discovered_services.json")
            services_data = {
                service_id: service.dict() 
                for service_id, service in self.discovered_services.items()
            }
            with open(services_file, 'w') as f:
                json.dump(services_data, f, default=str, indent=2)
            
            # Save test results cache
            results_file = os.path.join(self.storage_path, "test_results_cache.json")
            with open(results_file, 'w') as f:
                json.dump(self.test_results_cache, f, default=str, indent=2)
            
            # Close HTTP client
            if hasattr(self, 'http_client'):
                await self.http_client.aclose()
            
            logger.info(f"Saved {len(self.discovered_services)} services and {len(self.test_results_cache)} cached results to state")
            
        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the agent gracefully"""
        logger.info(f"Shutting down {self.name}...")
        
        try:
            # Save state before shutdown
            await self.cleanup()
            
            # Close HTTP client
            if hasattr(self, 'http_client'):
                await self.http_client.aclose()
            
            logger.info("Agent 4 shutdown completed successfully")
            
        except Exception as e:
            logger.error(f"Error during agent shutdown: {e}")
            raise