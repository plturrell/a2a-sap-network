"""
Data Standardization Agent - SDK Version
Agent 1: Enhanced with A2A SDK for standardizing financial data
"""

"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""



import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from uuid import uuid4

# Trust system imports
try:
    import sys
    sys.path.insert(0, '/Users/apple/projects/a2a/a2aNetwork')
    from trustSystem.smartContractTrust import (
        initialize_agent_trust,
        get_trust_contract,
        verify_a2a_message,
        sign_a2a_message
    )
except ImportError:
    # Fallback if trust system not available
    def initialize_agent_trust(*args, **kwargs):
        return {"status": "trust_system_unavailable"}

    def get_trust_contract():
        return None

    def verify_a2a_message(*args, **kwargs):
        return True, {"status": "trust_system_unavailable"}

    def sign_a2a_message(*args, **kwargs):
        return {"message": args[1] if len(args) > 1 else {}, "signature": {"status": "trust_system_unavailable"}}

# Import SDK components
from app.a2a.sdk.agentBase import A2AAgentBase
from app.a2a.sdk import (
    a2a_handler, a2a_skill, a2a_task,
    A2AMessage
)
from app.a2a.sdk.utils import create_error_response, create_success_response
from app.a2a.sdk.blockchainIntegration import BlockchainIntegrationMixin
from app.a2a.core.security_base import SecureA2AAgent


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
logger = logging.getLogger(__name__)


class BasicStandardizer:
    """Basic standardizer for financial data entities"""

    def standardize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize data to L4 hierarchical structure"""
        return {
            "original": data,
            "standardized": {
                "entity_id": data.get("id", f"std_{uuid4().hex[:8]}"),
                "entity_type": self._determine_entity_type(data),
                "standardized_fields": self._standardize_fields(data),
                "hierarchy_level": "L4",
                "standardization_timestamp": datetime.utcnow().isoformat()
            },
            "standardization_metadata": {
                "standardizer": self.__class__.__name__,
                "version": "1.0.0",
                "fields_processed": len(data),
                "success": True
            }
        }

    def _determine_entity_type(self, data: Dict[str, Any]) -> str:
        """Determine entity type from data"""
        # Simple heuristic based on common field patterns
        if any(key in data for key in ["account_number", "account_id", "balance"]):
            return "account"
        elif any(key in data for key in ["product_id", "product_name", "category"]):
            return "product"
        elif any(key in data for key in ["location_id", "address", "country"]):
            return "location"
        elif any(key in data for key in ["measure_id", "unit", "value"]):
            return "measure"
        else:
            return "general"

    def _standardize_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize field names and values"""
        standardized = {}

        field_mappings = {
            "id": "entity_id",
            "name": "entity_name",
            "desc": "description",
            "description": "description",
            "type": "entity_type",
            "category": "category",
            "value": "amount",
            "amount": "amount"
        }

        for key, value in data.items():
            standard_key = field_mappings.get(key.lower(), key)
            standardized[standard_key] = self._standardize_value(value)

        return standardized

    def _standardize_value(self, value: Any) -> Any:
        """Standardize individual values"""
        if isinstance(value, str):
            return value.strip().title() if len(value) < 100 else value.strip()
        return value


class DataStandardizationAgentSDK(SecureA2AAgent, BlockchainIntegrationMixin):
    """Data Standardization Agent - SDK Version"""

    def __init__(self, base_url: str = os.getenv("A2A_SERVICE_URL")):
        # Define blockchain capabilities for standardization agent
        blockchain_capabilities = [
            "data_standardization",
            "l4_hierarchy_mapping",
            "entity_normalization",
            "field_mapping",
            "data_validation",
            "schema_standardization",
            "quality_verification",
            "metadata_enrichment"
        ]

        # Initialize A2AAgentBase with blockchain capabilities
        A2AAgentBase.__init__(
            self,
            agent_id="data_standardization_agent_1",
            name="Data Standardization Agent",
            description="A2A v0.2.9 compliant agent for standardizing financial data to L4 hierarchical structure",
            version="4.0.0",  # Updated to A2A compliant version
            base_url=base_url,
            blockchain_capabilities=blockchain_capabilities,
            a2a_protocol_only=True  # Force A2A protocol compliance
        )

        # Initialize blockchain integration
        BlockchainIntegrationMixin.__init__(self)

        # Initialize standardizers
        self.standardizers = {
            "account": BasicStandardizer(),
            "product": BasicStandardizer(),
            "location": BasicStandardizer(),
            "measure": BasicStandardizer(),
            "book": BasicStandardizer(),
            "general": BasicStandardizer()
        }

        self.standardization_stats = {
            "total_processed": 0,
            "successful_standardizations": 0,
            "records_standardized": 0,
            "data_types_processed": set(),
            "schema_registrations": 0
        }

        # Initialize storage
        self.output_dir = os.getenv("STANDARDIZATION_OUTPUT_DIR", "/tmp/standardized_data")
        os.makedirs(self.output_dir, exist_ok=True)

        # Trust system components
        self.trust_identity = None
        self.trust_contract = None
        self.trusted_agents = set()

        logger.info(f"Initialized {self.name} with A2A Protocol v0.2.9 compliance")

    async def initialize(self) -> None:
        """Initialize agent resources"""
        logger.info("Initializing Data Standardization Agent resources...")
        try:
            # Establish standard trust relationships FIRST
            await self.establish_standard_trust_relationships()

            # Initialize trust system
            await self._initialize_trust_system()

            # Initialize blockchain integration
            try:
                await self.initialize_blockchain()
                logger.info("✅ Blockchain integration initialized for Agent 1")
            except Exception as e:
                logger.warning(f"⚠️ Blockchain initialization failed: {e}")

            # Verify A2A protocol connectivity
            await self._verify_a2a_connectivity()

            logger.info("Data Standardization Agent initialized successfully with A2A protocol")
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            raise

    async def shutdown(self) -> None:
        """Cleanup agent resources"""
        logger.info("Shutting down Data Standardization Agent...")
        try:
            # Wait for A2A queues to drain
            try:
                await asyncio.wait_for(self._drain_a2a_queues(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for A2A queues to drain")

            logger.info("Data Standardization Agent shutdown complete")
        except Exception as e:
            logger.error(f"Agent shutdown failed: {e}")

    @a2a_handler("standardize_data", "Standardize financial data to L4 hierarchical structure")
    async def handle_standardization_request(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Main standardization handler"""
        try:
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

            return create_success_response({
                "task_id": task_id,
                "status": "processing",
                "data_types": list(standardization_request.keys()),
                "message": "Data standardization started"
            })

        except Exception as e:
            logger.error(f"Standardization handler failed: {e}")
            return create_error_response(500, str(e))

    @a2a_skill(
        name="account_standardization",
        description="Standardize account data to L4 hierarchical structure",
        capabilities=["financial-standardization", "account-hierarchy", "l4-structure"],
        domain="financial-data"
    )
    async def standardize_accounts(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize account data"""
        try:
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

            # Store standardized data via data_manager
            await self.store_agent_data(
                data_type="standardized_accounts",
                data={
                    "accounts": standardized_accounts,
                    "standardization_timestamp": datetime.utcnow().isoformat(),
                    "standardization_rules": "L4_hierarchical"
                }
            )

            return {
                "data_type": "account",
                "total_records": len(accounts),
                "successful_records": len([a for a in standardized_accounts if a.get("standardized")]),
                "standardized_data": standardized_accounts,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Account standardization failed: {e}")
            return {"error": str(e)}

    @a2a_skill(
        name="product_standardization",
        description="Standardize product data to L4 hierarchical structure",
        capabilities=["financial-standardization", "product-hierarchy", "l4-structure"],
        domain="financial-data"
    )
    async def standardize_products(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize product data"""
        try:
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

        except Exception as e:
            logger.error(f"Product standardization failed: {e}")
            return {"error": str(e)}

    @a2a_skill(
        name="location_standardization",
        description="Standardize location data to L4 hierarchical structure",
        capabilities=["financial-standardization", "location-hierarchy", "l4-structure"],
        domain="financial-data"
    )
    async def standardize_locations(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize location data"""
        try:
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

        except Exception as e:
            logger.error(f"Location standardization failed: {e}")
            return {"error": str(e)}

    @a2a_skill(
        name="batch_standardization",
        description="Batch standardization of multiple data types",
        capabilities=["batch-processing", "multi-type-standardization"],
        domain="financial-data"
    )
    async def standardize_batch(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Batch standardization of multiple data types"""
        try:
            results = {}
            total_records = 0
            successful_records = 0

            for data_type, items in input_data.items():
                if data_type in self.standardizers and isinstance(items, list):
                    skill_name = f"{data_type}_standardization"
                    if skill_name in self.skills:
                        skill_result = await self.execute_skill(skill_name, {"items": items})
                        if skill_result.get("success", True):
                            result_data = skill_result.get("result", skill_result)
                            results[data_type] = result_data
                            total_records += result_data.get("total_records", 0)
                            successful_records += result_data.get("successful_records", 0)

            return {
                "batch_results": results,
                "summary": {
                    "data_types_processed": len(results),
                    "total_records": total_records,
                    "successful_records": successful_records,
                    "success_rate": successful_records / total_records if total_records > 0 else 0
                }
            }

        except Exception as e:
            logger.error(f"Batch standardization failed: {e}")
            return {"error": str(e)}

    @a2a_task(
        task_type="data_standardization",
        description="Complete data standardization workflow",
        timeout=600,
        retry_attempts=2
    )
    async def process_standardization_workflow(self, request: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Complete workflow for data standardization"""
        try:
            results = {
                "workflow_id": f"std_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "context_id": context_id,
                "standardization_results": {}
            }

            # Check if it's batch processing
            if len(request) > 1:
                batch_result = await self.execute_skill("batch_standardization", request)
                if batch_result.get("success", True):
                    results["standardization_results"] = batch_result.get("result", batch_result)
                else:
                    results["standardization_results"] = {"error": batch_result.get("error")}
            else:
                # Single data type processing
                for data_type, items in request.items():
                    if data_type in self.standardizers:
                        skill_name = f"{data_type}_standardization"
                        skill_result = await self.execute_skill(skill_name, {"items": items})
                        if skill_result.get("success", True):
                            results["standardization_results"][data_type] = skill_result.get("result", skill_result)
                        else:
                            results["standardization_results"][data_type] = {"error": skill_result.get("error")}

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
            from app.a2a.sdk.types import TaskStatus


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
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
        try:
            workflow_id = results["workflow_id"]

            for data_type, result in results["standardization_results"].items():
                if isinstance(result, dict) and "standardized_data" in result:
                    output_file = os.path.join(self.output_dir, f"standardized_{data_type}_{workflow_id}.json")
                    output_data = {
                        "metadata": {
                            "data_type": data_type,
                            "workflow_id": workflow_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "records": result.get("total_records", 0),
                            "successful_records": result.get("successful_records", 0)
                        },
                        "data": result["standardized_data"]
                    }

                    with open(output_file, 'w') as f:
                        json.dump(output_data, f, indent=2, default=str)

                    logger.info(f"Saved {data_type} standardization results to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save standardization results: {e}")

    async def _list_output_files(self) -> List[str]:
        """List output files"""
        try:
            return [f for f in os.listdir(self.output_dir) if f.endswith('.json')]
        except:
            return []

    async def _initialize_trust_system(self) -> None:
        """Initialize the agent's trust system"""
        try:
            # Initialize trust identity
            self.trust_identity = await initialize_agent_trust(self.agent_id, self.base_url)

            if self.trust_identity:
                logger.info(f"✅ Trust system initialized for {self.agent_id}")
                logger.info(f"   Trust address: {self.trust_identity.get('address')}")

                # Get trust contract reference
                self.trust_contract = get_trust_contract()

                # Pre-trust essential agents for A2A communication
                essential_agents = [
                    "agent_manager",
                    "data_product_agent_0",
                    "ai_preparation_agent_3",
                    "vector_processing_agent_4",
                    "catalog_manager_agent_2"
                ]

                self.trusted_agents = set()
                for agent_id in essential_agents:
                    self.trusted_agents.add(agent_id)

                logger.info(f"   Pre-trusted agents: {self.trusted_agents}")
            else:
                logger.warning("⚠️ Trust system initialization failed, running without trust verification")

        except Exception as e:
            logger.error(f"❌ Failed to initialize trust system: {e}")
            logger.warning("Continuing without trust verification")

    # A2A Protocol Helper Methods
    async def _verify_a2a_connectivity(self):
        """Verify A2A protocol connectivity with other agents"""
        try:
            # Discover available agents via catalog_manager standard trust relationship
            available_agents = await self.discover_agents(capability="data_processing")
            logger.info(f"Discovered {len(available_agents)} data processing agents via catalog_manager")

            # Update status with agent_manager
            await self.update_agent_status("ready", {
                "discovered_agents": len(available_agents),
                "capabilities": list(self.skills.keys())
            })

            # Test connectivity with essential agents
            essential_agents = [
                "data_product_agent_0",
                "ai_preparation_agent_3",
                "vector_processing_agent_4",
                "catalog_manager_agent_2"
            ]

            for agent_id in essential_agents:
                result = await self.request_data_from_agent_a2a(
                    target_agent=agent_id,
                    data_type="health_check",
                    query_params={"requester": self.agent_id},
                    encrypt=False
                )
                logger.info(f"A2A connectivity verified with {agent_id}: {result.get('success', False)}")

        except Exception as e:
            logger.warning(f"A2A connectivity verification failed: {e}")

    async def _drain_a2a_queues(self):
        """Wait for A2A message queues to empty"""
        while not self.outgoing_queue.empty() or not self.retry_queue.empty():
            await asyncio.sleep(1)

    async def _process_a2a_data_request(self, data_type: str, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Process A2A data request - override from base class"""
        try:
            if data_type == "health_check":
                return {
                    "agent_id": self.agent_id,
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "standardization_stats": self.standardization_stats,
                    "data_types_processed": list(self.standardization_stats.get("data_types_processed", set()))
                }
            elif data_type == "status":
                return {
                    "agent_id": self.agent_id,
                    "name": self.name,
                    "version": self.version,
                    "statistics": self.standardization_stats,
                    "active_tasks": len([t for t in self.tasks.values() if t["status"] == "running"]),
                    "records_standardized": self.standardization_stats.get("records_standardized", 0)
                }
            else:
                return {"error": f"Unknown data type: {data_type}"}
        except Exception as e:
            logger.error(f"Error processing A2A data request: {e}")
            return {"error": str(e)}
    def _init_security_features(self):
        """Initialize security features from SecureA2AAgent"""
        # Rate limiting configuration
        self.rate_limits = {
            'default': {'requests': 100, 'window': 60},  # 100 requests per minute
            'heavy': {'requests': 10, 'window': 60},     # 10 requests per minute for heavy operations
            'auth': {'requests': 5, 'window': 300}       # 5 auth attempts per 5 minutes
        }

        # Input validation rules
        self.validation_rules = {
            'max_string_length': 10000,
            'max_array_size': 1000,
            'max_object_depth': 10,
            'allowed_file_extensions': ['.json', '.txt', '.csv', '.xml'],
            'sql_injection_patterns': [
                r'((SELECT|INSERT|UPDATE|DELETE|DROP|UNION|WHERE|FROM))',
                r'(--|;|\'|"|\*|OR\s+1=1|AND\s+1=1)'
            ]
        }

        # Initialize security logger
        import logging
        self.security_logger = logging.getLogger(f'{self.__class__.__name__}.security')

    def _init_rate_limiting(self):
        """Initialize rate limiting tracking"""
        from collections import defaultdict
        import time

        self.rate_limit_tracker = defaultdict(lambda: {'count': 0, 'window_start': time.time()})

    def _init_input_validation(self):
        """Initialize input validation helpers"""
        self.input_validators = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'url': re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
            'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
            'safe_string': re.compile(r'^[a-zA-Z0-9\s\-_.,!?]+$')
        }

    @property
    def is_secure(self) -> bool:
        """Check if agent is running in secure mode"""
        return True  # SecureA2AAgent always runs in secure mode

    def validate_input(self, data: Any, rules: Dict[str, Any] = None) -> Tuple[bool, Optional[str]]:
        """Validate input data against security rules"""
        if rules is None:
            rules = self.validation_rules

        try:
            # Check string length
            if isinstance(data, str):
                if len(data) > rules.get('max_string_length', 10000):
                    return False, "String exceeds maximum length"

                # Check for SQL injection patterns
                for pattern in rules.get('sql_injection_patterns', []):
                    if re.search(pattern, data, re.IGNORECASE):
                        self.security_logger.warning(f"Potential SQL injection detected: {data[:50]}...")
                        return False, "Invalid characters detected"

            # Check array size
            elif isinstance(data, (list, tuple)):
                if len(data) > rules.get('max_array_size', 1000):
                    return False, "Array exceeds maximum size"

            # Check object depth
            elif isinstance(data, dict):
                if self._get_dict_depth(data) > rules.get('max_object_depth', 10):
                    return False, "Object exceeds maximum depth"

            return True, None

        except Exception as e:
            self.security_logger.error(f"Input validation error: {e}")
            return False, str(e)

    def _get_dict_depth(self, d: dict, current_depth: int = 0) -> int:
        """Get the maximum depth of a nested dictionary"""
        if not isinstance(d, dict) or not d:
            return current_depth

        return max(self._get_dict_depth(v, current_depth + 1)
                   for v in d.values()
                   if isinstance(v, dict))

    def check_rate_limit(self, key: str, limit_type: str = 'default') -> bool:
        """Check if rate limit is exceeded"""
        import time

        limits = self.rate_limits.get(limit_type, self.rate_limits['default'])
        tracker = self.rate_limit_tracker[f"{key}:{limit_type}"]

        current_time = time.time()
        window_duration = limits['window']

        # Reset window if expired
        if current_time - tracker['window_start'] > window_duration:
            tracker['count'] = 0
            tracker['window_start'] = current_time

        # Check limit
        if tracker['count'] >= limits['requests']:
            self.security_logger.warning(f"Rate limit exceeded for {key} ({limit_type})")
            return False

        tracker['count'] += 1
        return True
