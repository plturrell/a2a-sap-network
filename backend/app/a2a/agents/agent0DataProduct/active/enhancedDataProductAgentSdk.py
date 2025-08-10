#!/usr/bin/env python3
"""
Enhanced Data Product Registration Agent - SDK Version with Performance Monitoring
Agent 0: Enhanced with A2A SDK, performance monitoring, and optimization capabilities
"""

import asyncio
import json
import os
import pandas as pd
import httpx
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import hashlib
from uuid import uuid4
import logging

# Import SDK components - use local components
from app.a2a.sdk import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)
from app.a2a.core.workflowContext import WorkflowContextManager as workflowContextManager
from app.a2a.core.workflowMonitor import workflowMonitor as workflowMonitor
from app.a2a.core.asyncPatterns import (
    async_retry, async_timeout, async_concurrent_limit,
    AsyncOperationType, AsyncOperationConfig, async_manager
)

# Import performance monitoring
from app.a2a.core.performanceOptimizer import PerformanceOptimizationMixin
from app.a2a.core.performanceMonitor import AlertThresholds, monitor_performance

# Trust components
try:
    import sys
    sys.path.insert(0, '/Users/apple/projects/a2a/a2aNetwork')
    from trustSystem.smartContractTrust import sign_a2a_message
    from trustSystem.delegationContracts import DelegationAction
    print("✅ Using a2aNetwork trust components in enhanced agent0")
except ImportError:
    def sign_a2a_message(*args, **kwargs):
        return {"signature": "mock", "timestamp": datetime.utcnow().isoformat()}
    class DelegationAction:
        READ = "read"
        WRITE = "write"
        EXECUTE = "execute"
    print("⚠️  Using fallback trust components in enhanced agent0")

# Standardizers
from app.a2a.skills.accountStandardizer import AccountStandardizer
from app.a2a.skills.bookStandardizer import BookStandardizer
from app.a2a.skills.locationStandardizer import LocationStandardizer
from app.a2a.skills.measureStandardizer import MeasureStandardizer
from app.a2a.skills.productStandardizer import ProductStandardizer

logger = logging.getLogger(__name__)


class EnhancedDataProductRegistrationAgentSDK(A2AAgentBase, PerformanceOptimizationMixin):
    """
    Enhanced Agent 0: Data Product Registration Agent with Performance Monitoring
    Features:
    - High-performance ORD document registration
    - Real-time performance monitoring
    - Adaptive optimization
    - Comprehensive metrics collection
    """
    
    def __init__(self, base_url: str, ord_registry_url: str, enable_monitoring: bool = True):
        # Initialize both parent classes
        A2AAgentBase.__init__(
            self,
            agent_id="enhanced_data_product_agent_0",
            name="Enhanced Data Product Registration Agent",
            description="A2A v0.2.9 compliant agent with performance monitoring for data product registration",
            version="4.0.0",  # Enhanced version
            base_url=base_url
        )
        PerformanceOptimizationMixin.__init__(self)
        
        self.ord_registry_url = ord_registry_url
        self.enable_monitoring = enable_monitoring
        
        # Agent-specific configuration
        self.max_concurrent_registrations = 10
        self.registration_timeout = 30.0
        
        # Performance-optimized storage
        self.registration_cache = {}  # Will use optimized cache from mixin
        self.pending_registrations = set()
        
        # Standardizers with performance tracking
        self.standardizers = {
            "account": AccountStandardizer(),
            "book": BookStandardizer(),
            "location": LocationStandardizer(),
            "measure": MeasureStandardizer(),
            "product": ProductStandardizer()
        }
        
        # Statistics
        self.stats = {
            "total_registrations": 0,
            "successful_registrations": 0,
            "failed_registrations": 0,
            "cache_hits": 0,
            "standardizations_performed": 0
        }
        
        logger.info(f"Enhanced Data Product Agent initialized with monitoring: {enable_monitoring}")
        
    async def initialize(self) -> None:
        """Initialize agent with performance monitoring"""
        logger.info("Initializing Enhanced Data Product Registration Agent...")
        
        # Initialize base agent
        await super().initialize()
        
        # Enable performance monitoring if requested
        if self.enable_monitoring:
            # Custom alert thresholds for this agent
            alert_thresholds = AlertThresholds(
                cpu_threshold=75.0,  # Data processing can be CPU intensive
                memory_threshold=80.0,
                response_time_threshold=3000.0,  # 3 seconds for registration
                error_rate_threshold=0.03,  # 3% error rate
                queue_size_threshold=50
            )
            
            self.enable_performance_monitoring(
                alert_thresholds=alert_thresholds,
                metrics_port=8001  # Unique port for this agent
            )
        
        # Initialize HTTP client with connection pooling
        self.http_client = httpx.AsyncClient(
            timeout=self.registration_timeout,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5)
        )
        
        logger.info("Enhanced Data Product Agent initialization complete")
    
    @a2a_handler("data_product_registration")
    @monitor_performance("registration_handler")
    async def handle_data_product_registration(self, message: A2AMessage) -> Dict[str, Any]:
        """Enhanced handler for data product registration with performance monitoring"""
        try:
            # Extract registration data from message
            registration_data = self._extract_registration_data(message)
            if not registration_data:
                return self._create_error_response("No valid registration data found")
            
            # Check cache first
            cache_key = self._generate_cache_key(registration_data)
            cached_result = await self.cached_get(cache_key)
            
            if cached_result:
                self.stats["cache_hits"] += 1
                return self._create_success_response({
                    "registration_id": cached_result["registration_id"],
                    "status": "retrieved_from_cache",
                    "cached": True
                })
            
            # Process registration with throttling
            result = await self.throttled_operation(
                self._process_registration_with_monitoring,
                registration_data,
                message.conversation_id
            )
            
            # Cache successful result
            if result["success"]:
                await self.cached_set(cache_key, result["data"], ttl=3600)  # 1 hour cache
            
            return result
            
        except Exception as e:
            logger.error(f"Registration handler failed: {e}")
            return self._create_error_response(f"Registration failed: {str(e)}")
    
    @monitor_performance("registration_processing")
    async def _process_registration_with_monitoring(self, registration_data: Dict[str, Any], 
                                                   context_id: str) -> Dict[str, Any]:
        """Process registration with comprehensive monitoring"""
        self.stats["total_registrations"] += 1
        
        try:
            # Validate and standardize data
            standardized_data = await self._standardize_registration_data(registration_data)
            
            # Create ORD document
            ord_document = await self._create_ord_document(standardized_data)
            
            # Register with ORD registry
            registration_result = await self._register_with_ord_registry(ord_document)
            
            # Update statistics
            self.stats["successful_registrations"] += 1
            
            return self._create_success_response({
                "registration_id": registration_result["registration_id"],
                "ord_document": ord_document,
                "status": "registered",
                "context_id": context_id
            })
            
        except Exception as e:
            self.stats["failed_registrations"] += 1
            logger.error(f"Registration processing failed: {e}")
            return self._create_error_response(f"Processing failed: {str(e)}")
    
    @a2a_skill("data_standardization")
    @monitor_performance("standardization")
    async def standardize_data_product(self, data_product: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize data product with performance monitoring"""
        try:
            standardized = data_product.copy()
            standardization_count = 0
            
            # Apply standardizers based on data content
            for field_name, field_value in data_product.items():
                if field_name in ["account", "accounts"] and "account" in self.standardizers:
                    standardized[field_name] = self.standardizers["account"].standardize(field_value)
                    standardization_count += 1
                
                elif field_name in ["location", "locations"] and "location" in self.standardizers:
                    standardized[field_name] = self.standardizers["location"].standardize(field_value)
                    standardization_count += 1
                
                elif field_name in ["measure", "measures"] and "measure" in self.standardizers:
                    standardized[field_name] = self.standardizers["measure"].standardize(field_value)
                    standardization_count += 1
            
            self.stats["standardizations_performed"] += standardization_count
            
            return {
                "standardized_data": standardized,
                "standardizations_applied": standardization_count,
                "original_data": data_product
            }
            
        except Exception as e:
            logger.error(f"Standardization failed: {e}")
            raise
    
    @a2a_skill("ord_document_creation")
    @monitor_performance("ord_creation")
    async def create_ord_document_skill(self, data_product: Dict[str, Any]) -> Dict[str, Any]:
        """Create ORD document with performance optimization"""
        try:
            # Generate unique ID
            document_id = str(uuid4())
            
            # Create Dublin Core metadata
            dublin_core_metadata = self._generate_dublin_core_metadata(data_product)
            
            # Create ORD-compliant document structure
            ord_document = {
                "ord": "v1.0",
                "namespace": f"com.a2a.dataproducts.{document_id}",
                "documents": [{
                    "id": document_id,
                    "title": data_product.get("title", "Data Product"),
                    "description": data_product.get("description", "A2A registered data product"),
                    "version": data_product.get("version", "1.0.0"),
                    "created_at": datetime.utcnow().isoformat(),
                    "dublin_core": dublin_core_metadata,
                    "a2a_metadata": {
                        "agent_id": self.agent_id,
                        "standardized": True,
                        "quality_score": self._calculate_quality_score(data_product)
                    }
                }]
            }
            
            return ord_document
            
        except Exception as e:
            logger.error(f"ORD document creation failed: {e}")
            raise
    
    @a2a_task(
        task_type="performance_analysis",
        description="Analyze agent performance and apply optimizations",
        timeout=120,
        retry_attempts=1
    )
    async def analyze_and_optimize_performance(self) -> Dict[str, Any]:
        """Analyze performance and apply optimizations"""
        logger.info("Starting performance analysis and optimization...")
        
        try:
            # Run comprehensive performance analysis
            analysis_result = await self.run_performance_analysis()
            
            # Get current performance summary
            performance_summary = self.get_performance_summary()
            
            # Generate performance report
            report = {
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agent_stats": self.stats.copy(),
                "performance_analysis": analysis_result,
                "performance_summary": performance_summary,
                "cache_effectiveness": {
                    "hit_rate": self._cache_optimizer.get_hit_rate(),
                    "cache_size": len(self._cache_optimizer.cache),
                    "max_cache_size": self._cache_optimizer.max_size
                },
                "recommendations_summary": {
                    "total": len(self._optimization_recommendations),
                    "high_priority": len(self.get_optimization_recommendations(priority="high")),
                    "auto_applied": analysis_result.get("optimizations_applied", 0)
                }
            }
            
            logger.info(f"Performance analysis complete. Score: {analysis_result.get('performance_score', 0):.1f}/100")
            return report
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            raise
    
    def _extract_registration_data(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Extract registration data from A2A message"""
        for part in message.parts:
            if part.kind == "data" and part.data:
                return part.data
        return None
    
    def _generate_cache_key(self, registration_data: Dict[str, Any]) -> str:
        """Generate cache key for registration data"""
        data_hash = hashlib.md5(json.dumps(registration_data, sort_keys=True).encode()).hexdigest()
        return f"registration_{data_hash}"
    
    async def _standardize_registration_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize registration data using available standardizers"""
        standardization_result = await self.execute_skill("data_standardization", data)
        return standardization_result.get("standardized_data", data)
    
    async def _create_ord_document(self, standardized_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create ORD document from standardized data"""
        return await self.execute_skill("ord_document_creation", standardized_data)
    
    async def _register_with_ord_registry(self, ord_document: Dict[str, Any]) -> Dict[str, Any]:
        """Register ORD document with registry"""
        try:
            # Simulate registry registration (replace with actual API call)
            registration_id = str(uuid4())
            
            # In real implementation, this would POST to the ORD registry
            # response = await self.http_client.post(
            #     f"{self.ord_registry_url}/documents",
            #     json=ord_document
            # )
            
            return {
                "registration_id": registration_id,
                "status": "registered",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"ORD registry registration failed: {e}")
            raise
    
    def _generate_dublin_core_metadata(self, data_product: Dict[str, Any]) -> Dict[str, str]:
        """Generate Dublin Core metadata for data product"""
        return {
            "title": data_product.get("title", "Data Product"),
            "creator": data_product.get("creator", "A2A Agent System"),
            "subject": data_product.get("subject", "Data Product Registration"),
            "description": data_product.get("description", "A2A registered data product"),
            "publisher": data_product.get("publisher", "A2A Network"),
            "contributor": self.agent_id,
            "date": datetime.utcnow().isoformat(),
            "type": "Dataset",
            "format": data_product.get("format", "JSON"),
            "identifier": data_product.get("identifier", str(uuid4())),
            "language": data_product.get("language", "en"),
            "rights": data_product.get("rights", "A2A Network")
        }
    
    def _calculate_quality_score(self, data_product: Dict[str, Any]) -> float:
        """Calculate quality score for data product"""
        score = 0.0
        max_score = 10.0
        
        # Check for required fields
        required_fields = ["title", "description", "version"]
        for field in required_fields:
            if field in data_product and data_product[field]:
                score += 2.0
        
        # Check for optional but valuable fields
        optional_fields = ["creator", "subject", "format", "language"]
        for field in optional_fields:
            if field in data_product and data_product[field]:
                score += 1.0
        
        return min(score, max_score) / max_score
    
    def _create_success_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create success response"""
        return {
            "success": True,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": self.agent_id
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
        """Get comprehensive agent health status"""
        health = await super().get_agent_health()
        
        # Add performance-specific health metrics
        if self._performance_monitor:
            current_metrics = self._performance_monitor.get_current_metrics()
            health["performance"] = {
                "cpu_usage": current_metrics.cpu_usage,
                "memory_usage": current_metrics.memory_usage,
                "response_time_avg": current_metrics.response_time_avg,
                "error_rate": current_metrics.error_rate,
                "throughput": current_metrics.throughput
            }
        
        # Add agent-specific metrics
        health["agent_metrics"] = self.stats.copy()
        health["cache_performance"] = {
            "hit_rate": self._cache_optimizer.get_hit_rate(),
            "size": len(self._cache_optimizer.cache)
        }
        
        return health
    
    async def shutdown(self):
        """Shutdown agent with cleanup"""
        logger.info("Shutting down Enhanced Data Product Agent...")
        
        # Stop performance monitoring
        if self._performance_monitor:
            self._performance_monitor.stop_monitoring()
        
        # Close HTTP client
        if hasattr(self, 'http_client'):
            await self.http_client.aclose()
        
        # Call parent shutdown
        await super().shutdown()
        
        logger.info("Enhanced Data Product Agent shutdown complete")


# Example usage and testing function
async def demo_enhanced_agent():
    """Demonstrate enhanced agent capabilities"""
    logger.info("Demonstrating Enhanced Data Product Agent...")
    
    # Create enhanced agent
    agent = EnhancedDataProductRegistrationAgentSDK(
        base_url="http://localhost:8000",
        ord_registry_url="http://localhost:9000",
        enable_monitoring=True
    )
    
    # Initialize agent
    await agent.initialize()
    
    # Simulate some registrations
    test_data_products = [
        {
            "title": "Customer Transaction Data",
            "description": "Daily customer transaction records",
            "version": "2.1.0",
            "creator": "Data Engineering Team",
            "format": "Parquet"
        },
        {
            "title": "Product Catalog",
            "description": "Master product catalog with pricing",
            "version": "1.5.2",
            "creator": "Product Team"
        }
    ]
    
    # Process registrations
    results = []
    for data_product in test_data_products:
        # Create A2A message
        message = A2AMessage(
            conversation_id=f"demo_{uuid4()}",
            from_agent="demo_client",
            to_agent=agent.agent_id,
            parts=[
                {
                    "kind": "data",
                    "data": data_product
                }
            ],
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Process registration
        result = await agent.handle_data_product_registration(message)
        results.append(result)
        
        # Small delay to show monitoring
        await asyncio.sleep(0.5)
    
    # Run performance analysis
    performance_report = await agent.analyze_and_optimize_performance()
    
    # Get agent health
    health_status = await agent.get_agent_health()
    
    # Display results
    logger.info("\n=== DEMO RESULTS ===")
    logger.info(f"Processed {len(results)} registrations")
    logger.info(f"Performance Score: {performance_report['performance_analysis']['performance_score']:.1f}/100")
    logger.info(f"Cache Hit Rate: {health_status['cache_performance']['hit_rate']:.1%}")
    logger.info(f"Recommendations: {performance_report['recommendations_summary']['total']}")
    
    # Cleanup
    await agent.shutdown()
    
    return {
        "registration_results": results,
        "performance_report": performance_report,
        "health_status": health_status
    }


if __name__ == "__main__":
    # Run demo
    import logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        result = asyncio.run(demo_enhanced_agent())
        print("\n✅ Enhanced Agent Demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logging.error(f"Demo error: {e}", exc_info=True)
