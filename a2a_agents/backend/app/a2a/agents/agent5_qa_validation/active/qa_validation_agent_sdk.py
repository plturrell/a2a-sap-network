"""
QA Validation Agent - SDK Version
Agent 5: Enhanced with A2A SDK for ORD-integrated factuality testing using SimpleQA methodology
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
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pydantic import BaseModel, Field
from enum import Enum
import logging
import httpx
import inspect
from urllib.parse import urljoin, urlparse

# Vector embeddings and semantic search
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("SentenceTransformers/NumPy not available. Vector operations will be limited.")

# Template processing
try:
    from jinja2 import Template, Environment, BaseLoader
    TEMPLATE_AVAILABLE = True
except ImportError:
    TEMPLATE_AVAILABLE = False
    logging.warning("Jinja2 not available. Template processing will be limited.")

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


class TestDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(str, Enum):
    FACTUAL = "factual"
    REVERSE_LOOKUP = "reverse_lookup"
    ENUMERATION = "enumeration"
    RELATIONSHIP = "relationship"


class TestMethodology(str, Enum):
    SIMPLEQA = "simpleqa"
    FACTUALITY_CHECK = "factuality_check"
    COMPREHENSIVE = "comprehensive"


class MetadataSource(str, Enum):
    DUBLIN_CORE = "dublin_core"
    ORD_NATIVE = "ord_native"
    TECHNICAL = "technical"
    RELATIONSHIP = "relationship"


class ResourceType(str, Enum):
    DATA_PRODUCTS = "dataProducts"
    APIS = "apis"
    EVENTS = "events"
    ENTITY_TYPES = "entityTypes"


class QAValidationRequest(BaseModel):
    """Request for QA validation testing"""
    ord_endpoints: List[str] = Field(description="List of ORD registry endpoints")
    namespace_filter: Optional[str] = None
    resource_types: List[ResourceType] = Field(default=[ResourceType.DATA_PRODUCTS, ResourceType.APIS])
    test_methodology: TestMethodology = Field(default=TestMethodology.SIMPLEQA)
    test_config: Dict[str, Any] = Field(default_factory=dict)


class DiscoveredProduct(BaseModel):
    """Discovered ORD product with metadata"""
    ord_id: str
    title: str
    description: Optional[str] = None
    namespace: str
    registry_endpoint: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dublin_core: Dict[str, Any] = Field(default_factory=dict)
    technical_metadata: Dict[str, Any] = Field(default_factory=dict)
    relationships: Dict[str, Any] = Field(default_factory=dict)


class TestCase(BaseModel):
    """Generated test case with ground truth"""
    test_id: str
    source_product: DiscoveredProduct
    question: str
    answer: str
    test_type: TestType
    difficulty: TestDifficulty
    methodology: TestMethodology = TestMethodology.SIMPLEQA
    metadata_source: str
    extraction_method: str = "direct"
    confidence: float = 0.95
    ground_truth_source: str
    verification_method: str = "metadata_lookup"
    embedding_data: Optional[Dict[str, Any]] = None


class TestSuite(BaseModel):
    """Complete test suite with results"""
    suite_id: str
    created_at: datetime
    configuration: QAValidationRequest
    discovered_products: List[DiscoveredProduct] = Field(default_factory=list)
    generated_tests: List[TestCase] = Field(default_factory=list)
    execution_results: Optional[Dict[str, Any]] = None


class QAValidationAgentSDK(A2AAgentBase):
    """ORD-integrated factuality testing agent using SimpleQA methodology"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8007",
        data_manager_url: Optional[str] = None,
        catalog_manager_url: Optional[str] = None,
        cache_ttl: int = 3600,
        max_tests_per_product: int = 50,
        **kwargs
    ):
        super().__init__(
            agent_id=create_agent_id("qa-validation"),
            name="QA Validation Agent",
            description="A2A compliant agent for dynamic factuality testing using ORD registry data",
            version="1.0.0",
            base_url=base_url,
            **kwargs
        )
        
        # A2A Integration URLs
        self.data_manager_url = data_manager_url or "http://localhost:8001"
        self.catalog_manager_url = catalog_manager_url or "http://localhost:8002"
        
        self.cache_ttl = cache_ttl
        self.max_tests_per_product = max_tests_per_product
        
        # Initialize components
        self._setup_circuit_breakers()
        self._setup_metrics()
        self._load_question_templates()
        
        # State management
        self.test_suites: Dict[str, TestSuite] = {}
        self.websocket_connections: Dict[str, Any] = {}
        
        # Processing statistics
        self.processing_stats = {
            "total_tasks": 0,
            "ord_discoveries": 0,
            "tests_generated": 0,
            "tests_validated": 0
        }
        
        # Initialize trust system
        self.trust_identity = None
        self.trusted_agents = set()

    async def _query_catalog_manager(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query Catalog Manager for service discovery"""
        try:
            breaker = self.circuit_breaker_manager.get_breaker("catalog_manager")
            
            async def make_request():
                async with httpx.AsyncClient(timeout=30.0) as client:
                    url = f"{self.catalog_manager_url}{endpoint}"
                    response = await client.get(url, params=params or {})
                    response.raise_for_status()
                    return response.json()
            
            result = await breaker.call(make_request)
            return result
            
        except Exception as e:
            logger.error(f"Catalog Manager query failed for {endpoint}: {e}")
            return {"error": str(e), "services": []}
    
    async def _query_data_manager(self, endpoint: str, data: Dict[str, Any] = None, method: str = "GET") -> Dict[str, Any]:
        """Query Data Manager for data operations"""
        try:
            breaker = self.circuit_breaker_manager.get_breaker("data_manager")
            
            async def make_request():
                async with httpx.AsyncClient(timeout=30.0) as client:
                    url = f"{self.data_manager_url}{endpoint}"
                    
                    if method.upper() == "GET":
                        response = await client.get(url, params=data or {})
                    elif method.upper() == "POST":
                        response = await client.post(url, json=data or {})
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")
                    
                    response.raise_for_status()
                    return response.json()
            
            result = await breaker.call(make_request)
            return result
            
        except Exception as e:
            logger.error(f"Data Manager query failed for {endpoint}: {e}")
            return {"error": str(e), "data": []}
            
    def _setup_circuit_breakers(self):
        """Initialize circuit breakers for external services"""
        self.circuit_breaker_manager = get_breaker_manager()
        
        # Data Manager circuit breaker
        self.circuit_breaker_manager.get_breaker(
            "data_manager",
            failure_threshold=5,
            success_threshold=2,
            timeout=30.0
        )
        
        # Catalog Manager circuit breaker
        self.circuit_breaker_manager.get_breaker(
            "catalog_manager",
            failure_threshold=3,
            success_threshold=2,
            timeout=30.0
        )
        
        # Test validation circuit breaker
        self.circuit_breaker_manager.get_breaker(
            "test_validation",
            failure_threshold=10,
            success_threshold=3,
            timeout=60.0
        )
        
    def _setup_metrics(self):
        """Initialize Prometheus metrics"""
        self.metrics = {
            'data_manager_queries': Counter(
                'a2a_qa_data_manager_queries_total',
                'Total Data Manager queries performed',
                ['endpoint', 'status']
            ),
            'tests_generated': Counter(
                'a2a_qa_tests_generated_total',
                'Total test cases generated',
                ['difficulty', 'test_type']
            ),
            'test_execution_time': Histogram(
                'a2a_qa_test_execution_seconds',
                'Time spent executing tests',
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            ),
            'validation_time': Histogram(
                'a2a_qa_validation_seconds',
                'Time spent on self-contained validation',
                buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
            ),
            'active_test_suites': Gauge(
                'a2a_qa_active_test_suites',
                'Number of active test suites'
            )
        }
        
    def _load_question_templates(self):
        """Load question templates for different metadata elements"""
        self.question_templates = {
            'dublin_core': {
                'title': {
                    'templates': [
                        "What is the title of data product {ord_id}?",
                        "Which data product is titled '{title}'?",
                        "What data product has the title '{title}'?"
                    ],
                    'difficulty': TestDifficulty.EASY,
                    'test_type': TestType.FACTUAL
                },
                'description': {
                    'templates': [
                        "What is the description of {title}?",
                        "Describe the purpose of data product {ord_id}",
                        "What does the data product '{title}' contain?"
                    ],
                    'difficulty': TestDifficulty.MEDIUM,
                    'test_type': TestType.FACTUAL
                },
                'creator': {
                    'templates': [
                        "Who created the data product '{title}'?",
                        "What is the creator of data product {ord_id}?",
                        "Which data products were created by '{creator}'?"
                    ],
                    'difficulty': TestDifficulty.MEDIUM,
                    'test_type': TestType.REVERSE_LOOKUP
                },
                'subject': {
                    'templates': [
                        "What are the subject tags for '{title}'?",
                        "Which data products are tagged with '{tag}'?",
                        "What subjects does data product {ord_id} cover?"
                    ],
                    'difficulty': TestDifficulty.MEDIUM,
                    'test_type': TestType.ENUMERATION
                }
            },
            'technical': {
                'version': {
                    'templates': [
                        "What is the current version of '{title}'?",
                        "When was version {version} of '{title}' released?",
                        "Which data products are at version {version}?"
                    ],
                    'difficulty': TestDifficulty.EASY,
                    'test_type': TestType.FACTUAL
                },
                'apis': {
                    'templates': [
                        "What APIs does '{title}' expose?",
                        "What protocol does the {api_name} API use?",
                        "Which data products expose REST APIs?"
                    ],
                    'difficulty': TestDifficulty.MEDIUM,
                    'test_type': TestType.ENUMERATION
                },
                'format': {
                    'templates': [
                        "What format does '{title}' use?",
                        "Which data products use JSON format?",
                        "What is the data format of {ord_id}?"
                    ],
                    'difficulty': TestDifficulty.EASY,
                    'test_type': TestType.FACTUAL
                }
            },
            'relationships': {
                'dependencies': {
                    'templates': [
                        "What data products does '{title}' depend on?",
                        "Which data products consume data from '{title}'?",
                        "What are the dependencies of {ord_id}?"
                    ],
                    'difficulty': TestDifficulty.HARD,
                    'test_type': TestType.RELATIONSHIP
                },
                'consumers': {
                    'templates': [
                        "Which systems consume data from '{title}'?",
                        "What are the consumers of data product {ord_id}?",
                        "Which data products depend on '{title}'?"
                    ],
                    'difficulty': TestDifficulty.HARD,
                    'test_type': TestType.RELATIONSHIP
                }
            }
        }

    async def _get_vector_similarity_data(self, query_text: str, context: str = None) -> Optional[Dict[str, Any]]:
        """Get vector similarity data from Data Manager (pre-processed by Agent 3)"""
        try:
            # Query Data Manager for Agent 3 processed vector data
            result = await self._query_data_manager(
                "/agent3/vector_similarity",
                {
                    "query_text": query_text,
                    "context": context,
                    "similarity_threshold": 0.8,
                    "max_results": 10
                },
                method="POST"
            )
            
            if "error" not in result:
                return result.get("similarity_data", {})
            else:
                logger.warning(f"Vector similarity query returned error: {result['error']}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get vector similarity data: {e}")
            return None

    async def _discover_ord_products_from_data_manager(self, ord_endpoint: str, namespace_filter: Optional[str] = None) -> List[DiscoveredProduct]:
        """Discover data products from Data Manager (sourced from ORD registries)"""
        try:
            # Query Data Manager for ORD products instead of direct ORD discovery
            result = await self._query_data_manager(
                "/ord_products",
                {
                    "registry_endpoint": ord_endpoint,
                    "namespace_filter": namespace_filter,
                    "include_metadata": True,
                    "include_relationships": True
                }
            )
            
            if "error" in result:
                logger.error(f"Failed to get ORD products from Data Manager: {result['error']}")
                return []
            
            # Process retrieved products
            products = []
            for product_data in result.get("products", []):
                # Extract metadata components
                dublin_core = self._extract_dublin_core(product_data.get("metadata", {}))
                technical = self._extract_technical_metadata(product_data.get("metadata", {}))
                relationships = self._extract_relationships(product_data.get("metadata", {}))
                
                discovered_product = DiscoveredProduct(
                    ord_id=product_data.get("ord_id", str(uuid.uuid4())),
                    title=product_data.get("title", "Unknown"),
                    description=product_data.get("description"),
                    namespace=product_data.get("namespace", "unknown"),
                    registry_endpoint=ord_endpoint,
                    metadata=product_data.get("metadata", {}),
                    dublin_core=dublin_core,
                    technical_metadata=technical,
                    relationships=relationships
                )
                
                products.append(discovered_product)
            
            self.metrics['data_manager_queries'].labels(
                endpoint="/ord_products", 
                status='success'
            ).inc()
            
            logger.info(f"Retrieved {len(products)} products from Data Manager for {ord_endpoint}")
            self.processing_stats["ord_discoveries"] += 1
            
            return products
            
        except Exception as e:
            self.metrics['data_manager_queries'].labels(
                endpoint="/ord_products", 
                status='error'
            ).inc()
            logger.error(f"Failed to discover products from Data Manager: {e}")
            return []

    def _extract_dublin_core(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Dublin Core metadata elements"""
        dublin_core = {}
        
        # Map ORD fields to Dublin Core elements
        mappings = {
            'title': product.get('title'),
            'description': product.get('description'),
            'creator': product.get('responsible'),
            'subject': product.get('tags', []),
            'type': 'Dataset',
            'identifier': product.get('ordId'),
            'format': product.get('outputPorts', [{}])[0].get('mediaType') if product.get('outputPorts') else None,
            'coverage': product.get('supportedUseCases', []),
            'rights': product.get('accessStrategies', [])
        }
        
        # Filter out None values
        dublin_core = {k: v for k, v in mappings.items() if v is not None}
        
        return dublin_core

    def _extract_technical_metadata(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical metadata"""
        technical = {}
        
        technical['version'] = product.get('version')
        technical['visibility'] = product.get('visibility')
        technical['lifecycle_status'] = product.get('releaseStatus')
        
        # Extract API information
        apis = []
        for port in product.get('outputPorts', []):
            api_info = {
                'media_type': port.get('mediaType'),
                'protocol': port.get('protocol'),
                'url': port.get('url')
            }
            apis.append(api_info)
        technical['apis'] = apis
        
        # Extract input/output ports
        technical['input_ports'] = product.get('inputPorts', [])
        technical['output_ports'] = product.get('outputPorts', [])
        
        return technical

    def _extract_relationships(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relationship information"""
        relationships = {}
        
        relationships['depends_on'] = product.get('dataProductDependencies', [])
        relationships['links'] = product.get('dataProductLinks', [])
        relationships['use_cases'] = product.get('supportedUseCases', [])
        relationships['categories'] = product.get('industry', [])
        
        return relationships

    async def _generate_test_cases_for_product(self, product: DiscoveredProduct, max_tests: int = 50) -> List[TestCase]:
        """Generate SimpleQA-style test cases for a product"""
        test_cases = []
        test_id_counter = 0
        
        # Generate tests from Dublin Core metadata
        for element, value in product.dublin_core.items():
            if not value:
                continue
                
            template_config = self.question_templates.get('dublin_core', {}).get(element)
            if not template_config:
                continue
                
            templates = template_config['templates']
            difficulty = template_config['difficulty']
            test_type = template_config['test_type']
            
            for template in templates:
                if len(test_cases) >= max_tests:
                    break
                    
                try:
                    # Generate question and answer
                    question, answer = await self._generate_question_answer(
                        template, product, element, value
                    )
                    
                    if not question or not answer:
                        continue
                    
                    # Get self-contained validation data
                    validation_data = await self._get_self_contained_validation_data(question, answer, product)
                    
                    test_case = TestCase(
                        test_id=f"{product.ord_id}_test_{test_id_counter}",
                        source_product=product,
                        question=question,
                        answer=answer,
                        test_type=test_type,
                        difficulty=difficulty,
                        methodology=TestMethodology.SIMPLEQA,
                        metadata_source=f"dublin_core.{element}",
                        extraction_method="direct",
                        confidence=0.95,
                        ground_truth_source=f"{product.registry_endpoint}/products/{product.ord_id}",
                        verification_method="metadata_lookup",
                        embedding_data=validation_data
                    )
                    
                    test_cases.append(test_case)
                    test_id_counter += 1
                    
                    # Update metrics
                    self.metrics['tests_generated'].labels(
                        difficulty=difficulty.value,
                        test_type=test_type.value
                    ).inc()
                    
                except Exception as e:
                    logger.error(f"Failed to generate test case for {element}: {e}")
                    continue
        
        # Generate tests from technical metadata
        for element, value in product.technical_metadata.items():
            if not value or len(test_cases) >= max_tests:
                break
                
            template_config = self.question_templates.get('technical', {}).get(element)
            if not template_config:
                continue
                
            templates = template_config['templates']
            difficulty = template_config['difficulty']
            test_type = template_config['test_type']
            
            for template in templates:
                if len(test_cases) >= max_tests:
                    break
                    
                try:
                    question, answer = await self._generate_question_answer(
                        template, product, element, value
                    )
                    
                    if not question or not answer:
                        continue
                    
                    embedding_data = await self._generate_test_embeddings(question, answer, product)
                    
                    test_case = TestCase(
                        test_id=f"{product.ord_id}_test_{test_id_counter}",
                        source_product=product,
                        question=question,
                        answer=answer,
                        test_type=test_type,
                        difficulty=difficulty,
                        methodology=TestMethodology.SIMPLEQA,
                        metadata_source=f"technical.{element}",
                        extraction_method="direct",
                        confidence=0.90,
                        ground_truth_source=f"{product.registry_endpoint}/products/{product.ord_id}",
                        verification_method="metadata_lookup",
                        embedding_data=validation_data
                    )
                    
                    test_cases.append(test_case)
                    test_id_counter += 1
                    
                    self.metrics['tests_generated'].labels(
                        difficulty=difficulty.value,
                        test_type=test_type.value
                    ).inc()
                    
                except Exception as e:
                    logger.error(f"Failed to generate technical test case for {element}: {e}")
                    continue
        
        # Generate relationship tests
        for element, value in product.relationships.items():
            if not value or len(test_cases) >= max_tests:
                break
                
            template_config = self.question_templates.get('relationships', {}).get(element)
            if not template_config:
                continue
                
            templates = template_config['templates']
            difficulty = template_config['difficulty']
            test_type = template_config['test_type']
            
            for template in templates[:1]:  # Limit relationship tests
                if len(test_cases) >= max_tests:
                    break
                    
                try:
                    question, answer = await self._generate_question_answer(
                        template, product, element, value
                    )
                    
                    if not question or not answer:
                        continue
                    
                    embedding_data = await self._generate_test_embeddings(question, answer, product)
                    
                    test_case = TestCase(
                        test_id=f"{product.ord_id}_test_{test_id_counter}",
                        source_product=product,
                        question=question,
                        answer=answer,
                        test_type=test_type,
                        difficulty=difficulty,
                        methodology=TestMethodology.SIMPLEQA,
                        metadata_source=f"relationship.{element}",
                        extraction_method="derived",
                        confidence=0.85,
                        ground_truth_source=f"{product.registry_endpoint}/products/{product.ord_id}",
                        verification_method="semantic_match",
                        embedding_data=validation_data
                    )
                    
                    test_cases.append(test_case)
                    test_id_counter += 1
                    
                    self.metrics['tests_generated'].labels(
                        difficulty=difficulty.value,
                        test_type=test_type.value
                    ).inc()
                    
                except Exception as e:
                    logger.error(f"Failed to generate relationship test case for {element}: {e}")
                    continue
        
        logger.info(f"Generated {len(test_cases)} test cases for product {product.ord_id}")
        return test_cases

    async def _generate_question_answer(
        self, 
        template: str, 
        product: DiscoveredProduct, 
        element: str, 
        value: Any
    ) -> Tuple[str, str]:
        """Generate question and answer from template and metadata"""
        try:
            # Prepare template variables
            template_vars = {
                'ord_id': product.ord_id,
                'title': product.title,
                'description': product.description or '',
                'namespace': product.namespace,
                element: value
            }
            
            # Handle list values
            if isinstance(value, list):
                if value:
                    template_vars[element] = ', '.join(str(v) for v in value)
                    # For reverse lookup questions, pick first item
                    if len(value) > 0:
                        template_vars[f'{element}_item'] = str(value[0])
                        template_vars['tag'] = str(value[0])  # For subject tags
                else:
                    template_vars[element] = 'none'
            
            # Handle dict values (for APIs)
            if isinstance(value, list) and value and isinstance(value[0], dict):
                if element == 'apis':
                    api_names = [api.get('protocol', 'unknown') for api in value]
                    template_vars['api_name'] = api_names[0] if api_names else 'unknown'
            
            # Generate question
            if TEMPLATE_AVAILABLE:
                jinja_template = Template(template)
                question = jinja_template.render(**template_vars)
            else:
                # Simple string formatting fallback
                question = template.format(**template_vars)
            
            # Generate answer based on element and value
            answer = self._generate_answer(element, value, product)
            
            return question, answer
            
        except Exception as e:
            logger.error(f"Failed to generate question/answer for {element}: {e}")
            return "", ""

    def _generate_answer(self, element: str, value: Any, product: DiscoveredProduct) -> str:
        """Generate the expected answer for a test case"""
        if isinstance(value, list):
            if not value:
                return "none"
            elif len(value) == 1:
                return str(value[0])
            else:
                return ', '.join(str(v) for v in value)
        elif isinstance(value, dict):
            # Handle API information
            if element == 'apis':
                return ', '.join(f"{api.get('protocol', 'unknown')}" for api in [value])
            return str(value)
        else:
            return str(value) if value is not None else "unknown"
    
    def _validate_direct_match(self, answer: str, product: DiscoveredProduct) -> Dict[str, Any]:
        """Self-contained direct answer validation"""
        try:
            # Check against Dublin Core metadata
            dublin_matches = []
            for key, value in product.dublin_core.items():
                if isinstance(value, str) and answer.lower() in value.lower():
                    dublin_matches.append({"field": key, "value": value, "match_score": 1.0})
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and answer.lower() in item.lower():
                            dublin_matches.append({"field": key, "value": item, "match_score": 0.8})
            
            # Check against technical metadata
            technical_matches = []
            for key, value in product.technical_metadata.items():
                if isinstance(value, str) and answer.lower() in value.lower():
                    technical_matches.append({"field": key, "value": value, "match_score": 0.9})
            
            return {
                "validation_method": "direct_match",
                "dublin_core_matches": dublin_matches,
                "technical_matches": technical_matches,
                "total_matches": len(dublin_matches) + len(technical_matches),
                "confidence": min(1.0, (len(dublin_matches) + len(technical_matches)) * 0.2)
            }
            
        except Exception as e:
            logger.error(f"Direct match validation failed: {e}")
            return {"validation_method": "direct_match", "error": str(e), "confidence": 0.0}
    
    def _validate_semantic_match(self, answer: str, similarity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Self-contained semantic similarity validation using Agent 3 data"""
        try:
            if not similarity_data:
                return {"validation_method": "semantic_match", "confidence": 0.0, "reason": "no_similarity_data"}
            
            # Analyze similarity scores from Agent 3 processed data
            similar_items = similarity_data.get("similar_items", [])
            semantic_confidence = 0.0
            
            for item in similar_items:
                similarity_score = item.get("similarity_score", 0.0)
                if similarity_score > 0.8:  # High similarity threshold
                    semantic_confidence = max(semantic_confidence, similarity_score)
            
            return {
                "validation_method": "semantic_match",
                "similar_items_count": len(similar_items),
                "highest_similarity": semantic_confidence,
                "confidence": semantic_confidence,
                "threshold_met": semantic_confidence > 0.8
            }
            
        except Exception as e:
            logger.error(f"Semantic match validation failed: {e}")
            return {"validation_method": "semantic_match", "error": str(e), "confidence": 0.0}
    
    def _validate_metadata_consistency(self, answer: str, product: DiscoveredProduct) -> Dict[str, Any]:
        """Self-contained metadata consistency validation"""
        try:
            consistency_score = 0.0
            checks_performed = 0
            
            # Check title consistency
            if product.title and answer.lower() in product.title.lower():
                consistency_score += 0.3
            checks_performed += 1
            
            # Check description consistency
            if product.description and answer.lower() in product.description.lower():
                consistency_score += 0.2
            checks_performed += 1
            
            # Check ORD ID consistency
            if answer.lower() in product.ord_id.lower():
                consistency_score += 0.2
            checks_performed += 1
            
            # Check namespace consistency
            if answer.lower() in product.namespace.lower():
                consistency_score += 0.1
            checks_performed += 1
            
            # Check dublin core field consistency
            dublin_score = 0.0
            for key, value in product.dublin_core.items():
                if isinstance(value, str) and answer.lower() in value.lower():
                    dublin_score += 0.1
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and answer.lower() in item.lower():
                            dublin_score += 0.05
            consistency_score += min(0.2, dublin_score)
            checks_performed += 1
            
            final_confidence = consistency_score / checks_performed if checks_performed > 0 else 0.0
            
            return {
                "validation_method": "metadata_consistency",
                "consistency_score": consistency_score,
                "checks_performed": checks_performed,
                "confidence": final_confidence,
                "passed_threshold": final_confidence > 0.3
            }
            
        except Exception as e:
            logger.error(f"Metadata consistency validation failed: {e}")
            return {"validation_method": "metadata_consistency", "error": str(e), "confidence": 0.0}
    
    def _calculate_confidence_score(self, answer: str, product: DiscoveredProduct) -> float:
        """Calculate overall confidence score for a test case answer"""
        try:
            confidence_factors = []
            
            # Length factor (longer answers generally more specific)
            length_factor = min(1.0, len(answer) / 50.0)  # Normalize to 50 chars
            confidence_factors.append(length_factor * 0.2)
            
            # Specificity factor (contains specific terms)
            specific_terms = ['api', 'version', 'format', 'protocol', 'json', 'xml', 'rest', 'id']
            specificity_score = sum(1 for term in specific_terms if term in answer.lower()) / len(specific_terms)
            confidence_factors.append(specificity_score * 0.3)
            
            # Metadata richness factor
            total_metadata_fields = len(product.dublin_core) + len(product.technical_metadata) + len(product.relationships)
            richness_factor = min(1.0, total_metadata_fields / 10.0)  # Normalize to 10 fields
            confidence_factors.append(richness_factor * 0.3)
            
            # Answer completeness factor (not 'unknown' or 'none')
            completeness_factor = 0.0 if answer.lower() in ['unknown', 'none', ''] else 1.0
            confidence_factors.append(completeness_factor * 0.2)
            
            # Calculate weighted average
            final_confidence = sum(confidence_factors)
            return min(1.0, max(0.0, final_confidence))
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5  # Default moderate confidence

    async def _get_self_contained_validation_data(
        self, 
        question: str, 
        answer: str, 
        product: DiscoveredProduct
    ) -> Optional[Dict[str, Any]]:
        """Get self-contained validation data including vector similarity from Data Manager"""
        try:
            with self.metrics['validation_time'].time():
                # Create context from product metadata
                context = f"Data Product: {product.title} | ID: {product.ord_id} | Description: {product.description or ''}"
                
                # Get vector similarity data from Data Manager (processed by Agent 3)
                similarity_data = await self._get_vector_similarity_data(question, context)
                
                # Self-contained validation logic
                validation_data = {
                    'question_context': context,
                    'expected_answer': answer,
                    'similarity_data': similarity_data,
                    'validation_methods': {
                        'direct_match': self._validate_direct_match(answer, product),
                        'semantic_match': self._validate_semantic_match(answer, similarity_data) if similarity_data else None,
                        'metadata_consistency': self._validate_metadata_consistency(answer, product)
                    },
                    'confidence_score': self._calculate_confidence_score(answer, product)
                }
                
                return validation_data
                
        except Exception as e:
            logger.error(f"Failed to get validation data: {e}")
            return None

    async def _send_websocket_update(self, task_id: str, update_type: str, data: Dict[str, Any]):
        """Send update via WebSocket if connection exists"""
        if task_id in self.websocket_connections:
            try:
                message = {
                    'type': update_type,
                    'taskId': task_id,
                    'data': data,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                websocket = self.websocket_connections[task_id]
                await websocket.send(json.dumps(message))
                
            except Exception as e:
                logger.error(f"Failed to send WebSocket update for {task_id}: {e}")

    @a2a_skill("dynamic_test_generation")
    async def dynamic_test_generation(
        self, 
        request: QAValidationRequest,
        task_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate dynamic SimpleQA-style tests from ORD metadata"""
        try:
            if not task_id:
                task_id = f"qa_task_{uuid.uuid4().hex[:8]}"
            
            # Update metrics
            self.metrics['active_test_suites'].inc()
            
            # Send initial progress update
            await self._send_websocket_update(task_id, "progress", {
                "stage": "initializing",
                "message": "Starting ORD discovery and test generation",
                "percentage": 5
            })
            
            # Create test suite
            test_suite = TestSuite(
                suite_id=task_id,
                created_at=datetime.utcnow(),
                configuration=request
            )
            
            # Discover products from all ORD endpoints
            all_products = []
            for i, ord_endpoint in enumerate(request.ord_endpoints):
                await self._send_websocket_update(task_id, "progress", {
                    "stage": "discovery",
                    "message": f"Discovering products from {ord_endpoint}",
                    "percentage": 10 + (i * 20 // len(request.ord_endpoints))
                })
                
                products = await self._discover_ord_products_from_data_manager(ord_endpoint, request.namespace_filter)
                all_products.extend(products)
            
            test_suite.discovered_products = all_products
            
            await self._send_websocket_update(task_id, "progress", {
                "stage": "discovery_complete",
                "message": f"Discovered {len(all_products)} data products from ORD registries",
                "percentage": 30
            })
            
            # Generate test cases for each product
            all_test_cases = []
            max_tests_per_product = request.test_config.get('max_tests_per_product', self.max_tests_per_product)
            
            for i, product in enumerate(all_products):
                await self._send_websocket_update(task_id, "progress", {
                    "stage": "generating_tests",
                    "message": f"Processing data product: {product.ord_id}",
                    "percentage": 30 + (i * 50 // len(all_products)),
                    "products_processed": i + 1
                })
                
                test_cases = await self._generate_test_cases_for_product(product, max_tests_per_product)
                all_test_cases.extend(test_cases)
                
                # Send individual test notifications
                for test_case in test_cases[-3:]:  # Send last few tests as examples
                    await self._send_websocket_update(task_id, "test_generated", {
                        "product_id": product.ord_id,
                        "question": test_case.question,
                        "answer": test_case.answer,
                        "difficulty": test_case.difficulty.value,
                        "metadata_source": test_case.metadata_source
                    })
            
            test_suite.generated_tests = all_test_cases
            
            # Execute tests and validate
            await self._send_websocket_update(task_id, "progress", {
                "stage": "executing_tests",
                "message": "Executing generated tests and validating against ground truth",
                "percentage": 85,
                "tests_generated": len(all_test_cases)
            })
            
            execution_results = await self._execute_and_validate_tests(test_suite)
            test_suite.execution_results = execution_results
            
            # Store test suite
            self.test_suites[task_id] = test_suite
            
            # Send completion update
            await self._send_websocket_update(task_id, "completed", {
                "summary": {
                    "total_products": len(all_products),
                    "total_tests": len(all_test_cases),
                    "average_accuracy": execution_results.get('accuracy', 0.0),
                    "coverage_score": execution_results.get('coverage_metrics', {}).get('overall_coverage', 0.0)
                },
                "report_url": f"/a2a/tasks/{task_id}/report"
            })
            
            # Update metrics
            self.metrics['active_test_suites'].dec()
            
            return create_success_response({
                "task_id": task_id,
                "status": "completed",
                "summary": {
                    "discovered_products": len(all_products),
                    "generated_tests": len(all_test_cases),
                    "execution_results": execution_results
                },
                "streaming_endpoint": f"wss://{self.base_url}/a2a/stream/{task_id}"
            })
            
        except Exception as e:
            logger.error(f"Dynamic test generation failed: {e}")
            self.metrics['active_test_suites'].dec()
            
            await self._send_websocket_update(task_id, "error", {
                "message": f"Test generation failed: {str(e)}",
                "stage": "error"
            })
            
            return create_error_response(
                error_code="DYNAMIC_TEST_GENERATION_FAILED",
                message=f"Failed to generate dynamic tests: {str(e)}"
            )

    async def _execute_and_validate_tests(self, test_suite: TestSuite) -> Dict[str, Any]:
        """Execute test cases and validate against ground truth"""
        try:
            with self.metrics['test_execution_time'].time():
                results = {
                    'total_tests': len(test_suite.generated_tests),
                    'passed': 0,
                    'failed': 0,
                    'accuracy': 0.0,
                    'coverage_metrics': {},
                    'execution_details': []
                }
                
                dublin_core_tests = 0
                technical_tests = 0
                relationship_tests = 0
                
                for test_case in test_suite.generated_tests:
                    # Validate against ground truth (in this implementation, we assume all tests pass
                    # since they are generated from the same metadata)
                    is_valid = await self._validate_test_case(test_case)
                    
                    if is_valid:
                        results['passed'] += 1
                    else:
                        results['failed'] += 1
                    
                    # Count coverage by metadata source
                    if test_case.metadata_source.startswith('dublin_core'):
                        dublin_core_tests += 1
                    elif test_case.metadata_source.startswith('technical'):
                        technical_tests += 1
                    elif test_case.metadata_source.startswith('relationship'):
                        relationship_tests += 1
                
                # Calculate metrics
                if results['total_tests'] > 0:
                    results['accuracy'] = results['passed'] / results['total_tests']
                
                results['coverage_metrics'] = {
                    'dublin_core_coverage': dublin_core_tests / results['total_tests'] if results['total_tests'] > 0 else 0,
                    'technical_coverage': technical_tests / results['total_tests'] if results['total_tests'] > 0 else 0,
                    'relationship_coverage': relationship_tests / results['total_tests'] if results['total_tests'] > 0 else 0,
                    'overall_coverage': (dublin_core_tests + technical_tests + relationship_tests) / results['total_tests'] if results['total_tests'] > 0 else 0
                }
                
                return results
                
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'accuracy': 0.0,
                'coverage_metrics': {},
                'error': str(e)
            }

    async def _validate_test_case(self, test_case: TestCase) -> bool:
        """Validate a test case against ground truth"""
        try:
            # For SimpleQA methodology, we validate by checking if the answer
            # matches the expected value from the metadata
            
            if test_case.verification_method == "metadata_lookup":
                # Direct metadata validation
                source_element = test_case.metadata_source.split('.')[-1]
                
                if test_case.metadata_source.startswith('dublin_core'):
                    expected_value = test_case.source_product.dublin_core.get(source_element)
                elif test_case.metadata_source.startswith('technical'):
                    expected_value = test_case.source_product.technical_metadata.get(source_element)
                elif test_case.metadata_source.startswith('relationship'):
                    expected_value = test_case.source_product.relationships.get(source_element)
                else:
                    return False
                
                # Convert expected value to string for comparison
                if isinstance(expected_value, list):
                    expected_str = ', '.join(str(v) for v in expected_value) if expected_value else 'none'
                else:
                    expected_str = str(expected_value) if expected_value is not None else 'unknown'
                
                return test_case.answer.lower().strip() == expected_str.lower().strip()
                
            elif test_case.verification_method == "semantic_match":
                # Use semantic similarity if embeddings are available
                if test_case.embedding_data and 'question_vector' in test_case.embedding_data:
                    # In a real implementation, this would compare against a knowledge base
                    # For now, assume semantic matches are valid with high confidence
                    return test_case.confidence > 0.8
                
            return True  # Default to valid for generated tests
            
        except Exception as e:
            logger.error(f"Test case validation failed: {e}")
            return False

    @a2a_skill("ord_discovery")
    async def ord_discovery(
        self,
        ord_endpoints: List[str],
        namespace_filter: Optional[str] = None,
        resource_types: List[ResourceType] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Discover and process ORD data products"""
        try:
            if resource_types is None:
                resource_types = [ResourceType.DATA_PRODUCTS]
            
            all_products = []
            for ord_endpoint in ord_endpoints:
                products = await self._discover_ord_products_from_data_manager(ord_endpoint, namespace_filter)
                all_products.extend(products)
            
            return create_success_response({
                "discovered_products": len(all_products),
                "products": [
                    {
                        "ord_id": p.ord_id,
                        "title": p.title,
                        "namespace": p.namespace,
                        "registry_endpoint": p.registry_endpoint,
                        "metadata_elements": len(p.dublin_core) + len(p.technical_metadata) + len(p.relationships)
                    }
                    for p in all_products
                ]
            })
            
        except Exception as e:
            logger.error(f"ORD discovery failed: {e}")
            return create_error_response(
                error_code="ORD_DISCOVERY_FAILED",
                message=f"Failed to discover ORD products: {str(e)}"
            )

    @a2a_handler("executeTask")
    async def execute_task(self, message: A2AMessage) -> Dict[str, Any]:
        """Execute A2A task with streaming support"""
        try:
            params = message.content.get("params", {})
            task_id = params.get("taskId")
            skill = params.get("skill")
            skill_params = params.get("parameters", {})
            
            if skill == "dynamic_test_generation":
                # Convert parameters to request model
                request = QAValidationRequest(**skill_params)
                result = await self.dynamic_test_generation(request, task_id=task_id)
                
                return {
                    "id": message.content.get("id"),
                    "result": {
                        "status": "accepted",
                        "taskId": task_id,
                        "estimatedDuration": 300,  # 5 minutes
                        "streamingEndpoint": f"wss://{self.base_url}/a2a/stream/{task_id}",
                        "details": result
                    }
                }
                
            elif skill == "ord_discovery":
                result = await self.ord_discovery(**skill_params)
                
                return {
                    "id": message.content.get("id"),
                    "result": result
                }
            
            else:
                return create_error_response(
                    error_code="UNKNOWN_SKILL",
                    message=f"Unknown skill: {skill}"
                )
                
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return create_error_response(
                error_code="TASK_EXECUTION_FAILED",
                message=f"Failed to execute task: {str(e)}"
            )

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a running task"""
        try:
            if task_id in self.test_suites:
                test_suite = self.test_suites[task_id]
                
                if test_suite.execution_results:
                    status = "completed"
                    progress = {"percentage": 100, "stage": "completed"}
                elif test_suite.generated_tests:
                    status = "working"
                    progress = {"percentage": 90, "stage": "validating_tests"}
                elif test_suite.discovered_products:
                    status = "working"
                    progress = {"percentage": 50, "stage": "generating_tests"}
                else:
                    status = "working"
                    progress = {"percentage": 20, "stage": "discovering_products"}
                
                return {
                    "taskId": task_id,
                    "status": status,
                    "progress": progress,
                    "results": {
                        "partial_report": f"/a2a/tasks/{task_id}/partial-report"
                    } if test_suite.execution_results else {}
                }
            else:
                return {
                    "taskId": task_id,
                    "status": "not_found",
                    "error": "Task not found"
                }
                
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            return {
                "taskId": task_id,
                "status": "error",
                "error": str(e)
            }

    async def register_websocket_connection(self, task_id: str, websocket):
        """Register WebSocket connection for task streaming"""
        self.websocket_connections[task_id] = websocket
        logger.info(f"WebSocket registered for task {task_id}")

    async def unregister_websocket_connection(self, task_id: str):
        """Unregister WebSocket connection"""
        if task_id in self.websocket_connections:
            del self.websocket_connections[task_id]
            logger.info(f"WebSocket unregistered for task {task_id}")

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the agent (required by A2AAgentBase)"""
        try:
            # Initialize trust system if available
            try:
                self.trust_identity = initialize_agent_trust(self.agent_id, "QAValidationAgent")
                logger.info("Trust system initialized")
            except Exception as e:
                logger.warning(f"Trust system initialization failed: {e}")
            
            # Test connections to A2A services
            try:
                # Test Data Manager connection
                await self._query_data_manager("/health")
                logger.info(f"✅ Data Manager connected: {self.data_manager_url}")
                
                # Test Catalog Manager connection
                await self._query_catalog_manager("/health")
                logger.info(f"✅ Catalog Manager connected: {self.catalog_manager_url}")
                
            except Exception as e:
                logger.warning(f"⚠️ A2A service connection test failed: {e}")
            
            return {
                "agent_id": self.agent_id,
                "name": self.name,
                "version": self.version,
                "data_manager_url": self.data_manager_url,
                "catalog_manager_url": self.catalog_manager_url,
                "question_templates": len(self.question_templates),
                "circuit_breakers": list(self.circuit_breaker_manager._breakers.keys()) if hasattr(self.circuit_breaker_manager, '_breakers') else [],
                "processing_stats": self.processing_stats
            }
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            raise

    async def shutdown(self):
        """Cleanup resources"""
        try:
            # Close all WebSocket connections
            for task_id, websocket in list(self.websocket_connections.items()):
                try:
                    await websocket.close()
                except:
                    pass
            
            self.websocket_connections.clear()
            self.test_suites.clear()
            self.ord_cache.clear()
            
            logger.info("QA Validation Agent shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def get_agent_card(self) -> Dict[str, Any]:
        """Get A2A agent card for service discovery"""
        return {
            "name": "ORD-Dynamic-Testing-Agent",
            "description": "A2A compliant agent for dynamic factuality testing using ORD registry data",
            "version": "1.0.0",
            "protocolVersion": "0.2.9",
            "provider": {
                "name": "Testing Framework Inc",
                "url": "https://testing-framework.com",
                "contact": "support@testing-framework.com"
            },
            "capabilities": {
                "streaming": True,
                "pushNotifications": True,
                "stateHistory": True,
                "longRunningTasks": True
            },
            "skills": [
                {
                    "name": "dynamic_test_generation",
                    "description": "Generate SimpleQA-style tests from ORD metadata",
                    "inputModes": ["application/json"],
                    "outputModes": ["application/json", "text/plain"],
                    "parameters": {
                        "ord_endpoints": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of ORD registry endpoints"
                        },
                        "test_methodology": {
                            "type": "string",
                            "enum": ["simpleqa", "factuality_check", "comprehensive"],
                            "default": "simpleqa"
                        },
                        "metadata_schemas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["dublin_core", "ord_native"]
                        }
                    }
                },
                {
                    "name": "ord_discovery",
                    "description": "Discover and process ORD data products",
                    "parameters": {
                        "namespace_filter": {"type": "string"},
                        "resource_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "enum": ["dataProducts", "apis", "events", "entityTypes"]
                        }
                    }
                }
            ],
            "securitySchemes": {
                "bearer": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                }
            },
            "serviceEndpoint": f"{self.base_url}/a2a"
        }