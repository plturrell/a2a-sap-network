"""
Data Standardization Agent - SDK Version
Agent 1: Enhanced with A2A SDK for standardizing financial data
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import pandas as pd

# Import SDK components - use local components
from app.a2a.sdk import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)

# Import performance monitoring
from app.a2a.core.performanceOptimizer import PerformanceOptimizationMixin
from app.a2a.core.performanceMonitor import AlertThresholds, monitor_performance
# Import trust components from a2aNetwork
try:
    import sys
    sys.path.insert(0, '/Users/apple/projects/a2a/a2aNetwork')
    from trustSystem.smartContractTrust import sign_a2a_message, initialize_agent_trust, verify_a2a_message
except ImportError:
    # Fallback functions
    def sign_a2a_message(*args, **kwargs):
        return {"signature": "mock"}
    def initialize_agent_trust(*args, **kwargs):
        return True
    def verify_a2a_message(*args, **kwargs):
        return True

# Import standardizers
from app.a2a.skills.accountStandardizer import AccountStandardizer
from app.a2a.skills.bookStandardizer import BookStandardizer
from app.a2a.skills.locationStandardizer import LocationStandardizer
from app.a2a.skills.measureStandardizer import MeasureStandardizer
from app.a2a.skills.productStandardizer import ProductStandardizer

logger = logging.getLogger(__name__)


class DataStandardizationAgentSDK(A2AAgentBase, PerformanceOptimizationMixin):
    """
    Agent 1: Data Standardization Agent
    SDK Version - Simplified standardization with enhanced capabilities
    """
    
    def __init__(self, base_url: str, enable_monitoring: bool = True):
        # Initialize both parent classes
        A2AAgentBase.__init__(
            self,
            agent_id="data_standardization_agent_1",
            name="Enhanced Data Standardization Agent",
            description="A2A v0.2.9 compliant agent for standardizing financial data to L4 hierarchical structure with performance monitoring",
            version="4.0.0",  # Enhanced version
            base_url=base_url
        )
        PerformanceOptimizationMixin.__init__(self)
        
        self.enable_monitoring = enable_monitoring
        
        # Initialize standardizers
        self.standardizers = {
            "account": AccountStandardizer(),
            "book": BookStandardizer(),
            "location": LocationStandardizer(),
            "measure": MeasureStandardizer(),
            "product": ProductStandardizer()
        }
        
        self.standardization_stats = {
            "total_processed": 0,
            "successful_standardizations": 0,
            "records_standardized": 0,
            "data_types_processed": set()
        }
        
        # Trust system components
        self.trust_identity = None
        self.trust_contract = None
        self.trusted_agents = set()
        
        logger.info(f"Initialized {self.name} with SDK v4.0.0 and performance monitoring: {enable_monitoring}")
    
    async def initialize(self) -> None:
        """Initialize agent resources with performance monitoring"""
        logger.info("Initializing Enhanced Data Standardization Agent resources...")
        
        # Initialize base agent
        await super().initialize()
        
        # Initialize output directory
        self.output_dir = os.getenv("STANDARDIZATION_OUTPUT_DIR", "/tmp/standardized_data")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Enable performance monitoring if requested
        if self.enable_monitoring:
            alert_thresholds = AlertThresholds(
                cpu_threshold=70.0,  # Data processing can be CPU intensive
                memory_threshold=75.0,
                response_time_threshold=5000.0,  # 5 seconds for standardization
                error_rate_threshold=0.02,  # 2% error rate
                queue_size_threshold=100
            )
            
            self.enable_performance_monitoring(
                alert_thresholds=alert_thresholds,
                metrics_port=8002  # Unique port for this agent
            )
        
        # Initialize trust system
        await self._initialize_trust_system()
        
        logger.info("Enhanced Data Standardization Agent initialized successfully")
    
    async def shutdown(self) -> None:
        """Cleanup agent resources"""
        logger.info("Shutting down Data Standardization Agent...")
        logger.info("Data Standardization Agent shutdown complete")
    
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
                
                # Get trust contract reference
                self.trust_contract = get_trust_contract()
                
                # Pre-trust essential agents
                essential_agents = [
                    "agent_manager",
                    "data_product_agent_0",
                    "ai_preparation_agent_2",
                    "vector_processing_agent_3"
                ]
                
                for agent_id in essential_agents:
                    self.trusted_agents.add(agent_id)
                
                logger.info(f"   Pre-trusted agents: {self.trusted_agents}")
            else:
                logger.warning("⚠️  Trust system initialization failed, running without trust verification")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize trust system: {e}")
            logger.warning("Continuing without trust verification")
    
    @a2a_handler("standardize_data", "Standardize financial data to L4 hierarchical structure")
    async def handle_standardization_request(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Main standardization handler"""
        try:
            # Verify message trust if trust system is enabled
            if self.trust_identity:
                trust_verification = await verify_a2a_message(
                    message.dict() if hasattr(message, 'dict') else message,
                    self.agent_id
                )
                
                if not trust_verification["valid"]:
                    logger.warning(f"Trust verification failed: {trust_verification['error']}")
                    return create_error_response(403, f"Trust verification failed: {trust_verification['error']}")
                
                # Add sender to trusted agents if verification passed
                sender_id = trust_verification.get("signer_id")
                if sender_id:
                    self.trusted_agents.add(sender_id)
            
            # Extract standardization request
            standardization_request = self._extract_standardization_request(message)
            
            if not standardization_request:
                return create_error_response(400, "No standardization request found in message")
            
            # Create task for tracking
            task_id = await self.create_task("data_standardization", {
                "context_id": context_id,
                "request": standardization_request
            })
            
            # Process asynchronously
            asyncio.create_task(self._process_standardization(task_id, standardization_request, context_id))
            
            response = {
                "task_id": task_id,
                "status": "processing",
                "data_types": list(standardization_request.keys()),
                "message": "Data standardization started"
            }
            
            # Sign response if trust system is enabled
            if self.trust_identity:
                response = await sign_a2a_message(response, self.agent_id)
            
            return create_success_response(response)
            
        except Exception as e:
            logger.error(f"Standardization request failed: {e}")
            return create_error_response(500, str(e))
    
    @a2a_skill(
        name="account_standardization",
        description="Standardize account data to L4 hierarchical structure",
        capabilities=["financial-standardization", "account-hierarchy", "l4-structure"],
        domain="financial-data"
    )
    async def standardize_accounts(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize account data"""
        accounts = input_data.get("items", [])
        
        standardized_accounts = []
        for account in accounts:
            try:
                standardized = self.standardizers["account"].standardize(account)
                standardized_accounts.append(standardized)
            except Exception as e:
                logger.warning(f"Failed to standardize account {account}: {e}")
                standardized_accounts.append({
                    "original": account,
                    "standardized": None,
                    "error": str(e)
                })
        
        self.standardization_stats["records_standardized"] += len(standardized_accounts)
        self.standardization_stats["data_types_processed"].add("account")
        
        return {
            "data_type": "account",
            "total_records": len(accounts),
            "successful_records": len([a for a in standardized_accounts if a.get("standardized")]),
            "standardized_data": standardized_accounts,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @a2a_skill(
        name="location_standardization",
        description="Standardize location data to L4 hierarchical structure",
        capabilities=["financial-standardization", "location-hierarchy", "l4-structure"],
        domain="financial-data"
    )
    async def standardize_locations(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize location data"""
        locations = input_data.get("items", [])
        
        standardized_locations = []
        for location in locations:
            try:
                standardized = self.standardizers["location"].standardize(location)
                standardized_locations.append(standardized)
            except Exception as e:
                logger.warning(f"Failed to standardize location {location}: {e}")
                standardized_locations.append({
                    "original": location,
                    "standardized": None,
                    "error": str(e)
                })
        
        self.standardization_stats["records_standardized"] += len(standardized_locations)
        self.standardization_stats["data_types_processed"].add("location")
        
        return {
            "data_type": "location",
            "total_records": len(locations),
            "successful_records": len([l for l in standardized_locations if l.get("standardized")]),
            "standardized_data": standardized_locations,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @a2a_skill(
        name="product_standardization",
        description="Standardize product data to L4 hierarchical structure",
        capabilities=["financial-standardization", "product-hierarchy", "l4-structure"],
        domain="financial-data"
    )
    async def standardize_products(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize product data"""
        products = input_data.get("items", [])
        
        standardized_products = []
        for product in products:
            try:
                standardized = self.standardizers["product"].standardize(product)
                standardized_products.append(standardized)
            except Exception as e:
                logger.warning(f"Failed to standardize product {product}: {e}")
                standardized_products.append({
                    "original": product,
                    "standardized": None,
                    "error": str(e)
                })
        
        self.standardization_stats["records_standardized"] += len(standardized_products)
        self.standardization_stats["data_types_processed"].add("product")
        
        return {
            "data_type": "product",
            "total_records": len(products),
            "successful_records": len([p for p in standardized_products if p.get("standardized")]),
            "standardized_data": standardized_products,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @a2a_skill(
        name="batch_standardization",
        description="Batch standardization of multiple data types",
        capabilities=["batch-processing", "multi-type-standardization"],
        domain="financial-data"
    )
    async def standardize_batch(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Batch standardization of multiple data types"""
        results = {}
        total_records = 0
        successful_records = 0
        
        for data_type, items in input_data.items():
            if data_type in self.standardizers and isinstance(items, list):
                skill_name = f"{data_type}_standardization"
                
                if skill_name in self.skills:
                    skill_result = await self.execute_skill(skill_name, {"items": items})
                    if skill_result["success"]:
                        results[data_type] = skill_result["result"]
                        total_records += skill_result["result"]["total_records"]
                        successful_records += skill_result["result"]["successful_records"]
        
        return {
            "batch_results": results,
            "summary": {
                "data_types_processed": len(results),
                "total_records": total_records,
                "successful_records": successful_records,
                "success_rate": successful_records / total_records if total_records > 0 else 0
            }
        }
    
    @a2a_task(
        task_type="data_standardization",
        description="Complete data standardization workflow",
        timeout=600,
        retry_attempts=2
    )
    async def process_standardization_workflow(self, request: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Complete workflow for data standardization"""
        
        results = {
            "workflow_id": f"std_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "context_id": context_id,
            "standardization_results": {}
        }
        
        try:
            # Check if it's batch processing
            if len(request) > 1:
                batch_result = await self.execute_skill("batch_standardization", request)
                results["standardization_results"] = batch_result["result"] if batch_result["success"] else {"error": batch_result["error"]}
            else:
                # Single data type processing
                for data_type, items in request.items():
                    if data_type in self.standardizers:
                        skill_name = f"{data_type}_standardization"
                        skill_result = await self.execute_skill(skill_name, {"items": items})
                        results["standardization_results"][data_type] = skill_result["result"] if skill_result["success"] else {"error": skill_result["error"]}
            
            # Save results to files
            await self._save_standardization_results(results)
            
            # Update statistics
            self.standardization_stats["total_processed"] += 1
            self.standardization_stats["successful_standardizations"] += 1
            
            return {
                "workflow_successful": True,
                "results": results,
                "output_files": await self._list_output_files()
            }
            
        except Exception as e:
            logger.error(f"Standardization workflow failed: {e}")
            return {
                "workflow_successful": False,
                "error": str(e),
                "partial_results": results
            }
    
    async def _process_standardization(self, task_id: str, request: Dict[str, Any], context_id: str):
        """Process standardization asynchronously"""
        try:
            from app.a2a.sdk.types import TaskStatus
            await self.update_task(task_id, TaskStatus.RUNNING)
            
            result = await self.process_standardization_workflow(request, context_id)
            
            if result["workflow_successful"]:
                await self.update_task(task_id, TaskStatus.COMPLETED, result=result)
            else:
                await self.update_task(task_id, TaskStatus.FAILED, error=result.get("error"))
                
        except Exception as e:
            await self.update_task(task_id, TaskStatus.FAILED, error=str(e))
    
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
    
    async def _save_standardization_results(self, results: Dict[str, Any]):
        """Save standardization results to files"""
        workflow_id = results["workflow_id"]
        
        for data_type, result in results["standardization_results"].items():
            if isinstance(result, dict) and "standardized_data" in result:
                output_file = os.path.join(self.output_dir, f"standardized_{data_type}_{workflow_id}.json")
                
                output_data = {
                    "metadata": {
                        "data_type": data_type,
                        "workflow_id": workflow_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "records": result["total_records"],
                        "successful_records": result["successful_records"]
                    },
                    "data": result["standardized_data"]
                }
                
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                
                logger.info(f"Saved {data_type} standardization results to {output_file}")
    
    async def _list_output_files(self) -> List[str]:
        """List output files"""
        try:
            return [f for f in os.listdir(self.output_dir) if f.endswith('.json')]
        except:
            return []
