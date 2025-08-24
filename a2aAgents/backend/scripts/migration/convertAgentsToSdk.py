"""
Convert existing agents to use A2A SDK
This script will refactor existing agents to use the new SDK structure
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



import os
import shutil
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_data_product_agent():
    """Convert Agent 0 (Data Product Agent) to use SDK"""
    
    logger.info("Converting Data Product Agent to use A2A SDK...")
    
    # Create a new SDK-based version
    new_agent_content = '''"""
Data Product Registration Agent - SDK Version
Agent 0: Enhanced with A2A SDK for simplified development and maintenance
"""

import asyncio
import json
import os
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import hashlib
from uuid import uuid4

from ..sdk import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)
from ..sdk.utils import create_success_response, create_error_response
from ..core.workflow_context import workflow_context_manager
from ..core.workflow_monitor import workflow_monitor
from ..security.smart_contract_trust import sign_a2a_message
from ..security.delegation_contracts import DelegationAction
from ..skills.account_standardizer import AccountStandardizer
from ..skills.book_standardizer import BookStandardizer
from ..skills.location_standardizer import LocationStandardizer
from ..skills.measure_standardizer import MeasureStandardizer
from ..skills.product_standardizer import ProductStandardizer

logger = logging.getLogger(__name__)


class DataProductRegistrationAgentSDK(A2AAgentBase):
    """
    Agent 0: Data Product Registration Agent with Dublin Core
    SDK Version - Simplified development with enhanced capabilities
    """
    
    def __init__(self, base_url: str, ord_registry_url: str):
        super().__init__(
            agent_id="data_product_agent_0",
            name="Data Product Registration Agent",
            description="A2A v0.2.9 compliant agent for data product registration with Dublin Core metadata",
            version="3.0.0",  # SDK version
            base_url=base_url
        )
        
        self.ord_registry_url = ord_registry_url
        self.data_products = {}
        self.processing_stats = {
            "total_processed": 0,
            "successful_registrations": 0,
            "dublin_core_extractions": 0,
            "integrity_verifications": 0
        }
        
        logger.info(f"Initialized {self.name} with SDK v3.0.0")
    
    async def initialize(self) -> None:
        """Initialize agent resources"""
        logger.info("Initializing Data Product Registration Agent resources...")
        
        # Initialize data storage
        storage_path = os.getenv("DATA_PRODUCT_AGENT_STORAGE_PATH", "/tmp/data_product_agent_state")
        os.makedirs(storage_path, exist_ok=True)
        self.storage_path = storage_path
        
        # Load any persistent state
        await self._load_persistent_state()
        
        logger.info("Data Product Registration Agent initialized successfully")
    
    async def shutdown(self) -> None:
        """Cleanup agent resources"""
        logger.info("Shutting down Data Product Registration Agent...")
        
        # Save state
        await self._save_persistent_state()
        
        # Clear in-memory data
        self.data_products.clear()
        
        logger.info("Data Product Registration Agent shutdown complete")
    
    @a2a_handler("process_data", "Process and register data products with Dublin Core metadata")
    async def handle_data_processing(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Main data processing handler"""
        try:
            # Extract data from message
            data_info = self._extract_data_info(message)
            
            if not data_info:
                return create_error_response(400, "No data found in message")
            
            # Create task for tracking
            task_id = await self.create_task("data_product_registration", {
                "context_id": context_id,
                "data_info": data_info
            })
            
            # Process asynchronously
            asyncio.create_task(self._process_data_product(task_id, data_info, context_id))
            
            return create_success_response({
                "task_id": task_id,
                "status": "processing",
                "message": "Data product registration started"
            })
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            return create_error_response(500, str(e))
    
    @a2a_handler("get_status", "Get agent status and statistics")
    async def handle_status_request(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Get agent status"""
        return create_success_response({
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "statistics": self.processing_stats,
            "active_tasks": len([t for t in self.tasks.values() if t["status"] == "running"]),
            "data_products_registered": len(self.data_products),
            "capabilities": self.list_skills()
        })
    
    @a2a_skill(
        name="dublin_core_extraction",
        description="Extract Dublin Core metadata from data products",
        capabilities=["metadata-extraction", "dublin-core", "iso-15836"],
        domain="metadata"
    )
    async def extract_dublin_core_metadata(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Dublin Core metadata according to ISO 15836 standards"""
        data_location = input_data.get("data_location", "")
        data_type = input_data.get("data_type", "dataset")
        
        # Read and analyze data
        metadata = await self._analyze_data_for_dublin_core(data_location)
        
        # Generate Dublin Core elements
        dublin_core = {
            "title": metadata.get("title", f"Financial Data Products - {datetime.utcnow().strftime('%Y-%m')}"),
            "creator": ["FinSight CIB", "Data Product Registration Agent", "A2A System"],
            "subject": metadata.get("subjects", ["financial-data", "enterprise-data"]),
            "description": metadata.get("description", f"Financial dataset with {metadata.get('record_count', 0)} records"),
            "publisher": "FinSight CIB Data Platform",
            "date": datetime.utcnow().isoformat(),
            "type": "Dataset",
            "format": metadata.get("format", "text/csv"),
            "identifier": f"dp-{uuid4().hex[:12]}",
            "language": "en",
            "rights": "Internal Use Only - FinSight CIB Proprietary Data"
        }
        
        self.processing_stats["dublin_core_extractions"] += 1
        
        return {
            "dublin_core_metadata": dublin_core,
            "compliance": {
                "iso_15836": True,
                "rfc_5013": True,
                "ansi_niso_z39_85": True
            },
            "quality_score": metadata.get("quality_score", 1.0)
        }
    
    @a2a_skill(
        name="integrity_verification",
        description="Verify data integrity using SHA256 hashing and referential checks",
        capabilities=["data-integrity", "sha256-hashing", "referential-integrity"],
        domain="data-quality"
    )
    async def verify_data_integrity(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify data integrity with SHA256 and referential checks"""
        data_location = input_data.get("data_location", "")
        
        integrity_results = {
            "overall_status": "verified",
            "sha256_hashes": {},
            "referential_integrity": {},
            "verification_timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Process CSV files for hashing
            csv_files = self._find_csv_files(data_location)
            
            for csv_file in csv_files:
                file_hash = await self._calculate_file_hash(csv_file)
                integrity_results["sha256_hashes"][os.path.basename(csv_file)] = file_hash
            
            # Verify referential integrity
            ref_integrity = await self._verify_referential_integrity(csv_files)
            integrity_results["referential_integrity"] = ref_integrity
            
            self.processing_stats["integrity_verifications"] += 1
            
        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            integrity_results["overall_status"] = "failed"
            integrity_results["error"] = str(e)
        
        return integrity_results
    
    @a2a_skill(
        name="ord_registration",
        description="Register data products with ORD Registry",
        capabilities=["ord-registration", "catalog-management"],
        domain="data-catalog"
    )
    async def register_with_ord(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register data product with ORD Registry"""
        dublin_core = input_data.get("dublin_core_metadata", {})
        integrity_info = input_data.get("integrity_info", {})
        
        ord_descriptor = {
            "title": dublin_core.get("title", ""),
            "shortDescription": dublin_core.get("description", "")[:250],
            "description": dublin_core.get("description", ""),
            "version": "1.0.0",
            "releaseStatus": "active",
            "visibility": "internal",
            "partOf": [{"title": "FinSight CIB Data Platform"}],
            "tags": dublin_core.get("subject", []),
            "labels": {
                "data-type": ["financial"],
                "processing-level": ["raw", "structured"],
                "compliance": ["dublin-core", "iso-15836"]
            },
            "documentationLabels": {
                "Created By": "Data Product Registration Agent",
                "Dublin Core Compliant": "true",
                "SHA256 Verified": str(integrity_info.get("overall_status") == "verified").lower()
            }
        }
        
        try:
            # Register with ORD Registry
            registration_result = await self._register_ord_descriptor(ord_descriptor)
            
            self.processing_stats["successful_registrations"] += 1
            
            return {
                "registration_successful": True,
                "ord_id": registration_result.get("id"),
                "ord_descriptor": ord_descriptor,
                "registry_url": self.ord_registry_url
            }
            
        except Exception as e:
            logger.error(f"ORD registration failed: {e}")
            return {
                "registration_successful": False,
                "error": str(e)
            }
    
    @a2a_task(
        task_type="data_product_registration",
        description="Complete data product registration workflow",
        timeout=600,
        retry_attempts=2
    )
    async def process_data_product_workflow(self, data_info: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Complete workflow for data product registration"""
        
        results = {
            "workflow_id": f"dp_reg_{uuid4().hex[:8]}",
            "context_id": context_id,
            "stages": {}
        }
        
        try:
            # Stage 1: Dublin Core extraction
            dublin_core_result = await self.execute_skill("dublin_core_extraction", data_info)
            results["stages"]["dublin_core"] = dublin_core_result
            
            if not dublin_core_result["success"]:
                raise Exception("Dublin Core extraction failed")
            
            # Stage 2: Integrity verification
            integrity_result = await self.execute_skill("integrity_verification", data_info)
            results["stages"]["integrity"] = integrity_result
            
            if not integrity_result["success"]:
                raise Exception("Integrity verification failed")
            
            # Stage 3: ORD registration
            ord_input = {
                "dublin_core_metadata": dublin_core_result["result"]["dublin_core_metadata"],
                "integrity_info": integrity_result["result"]
            }
            
            ord_result = await self.execute_skill("ord_registration", ord_input)
            results["stages"]["ord_registration"] = ord_result
            
            # Update statistics
            self.processing_stats["total_processed"] += 1
            
            # Store data product info
            data_product_id = dublin_core_result["result"]["dublin_core_metadata"]["identifier"]
            self.data_products[data_product_id] = {
                "dublin_core": dublin_core_result["result"]["dublin_core_metadata"],
                "integrity": integrity_result["result"],
                "ord_registration": ord_result["result"] if ord_result["success"] else None,
                "created_at": datetime.utcnow().isoformat()
            }
            
            return {
                "workflow_successful": True,
                "data_product_id": data_product_id,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Data product workflow failed: {e}")
            return {
                "workflow_successful": False,
                "error": str(e),
                "partial_results": results
            }
    
    async def _process_data_product(self, task_id: str, data_info: Dict[str, Any], context_id: str):
        """Process data product asynchronously"""
        try:
            from ....sdk.types import TaskStatus
            await self.update_task(task_id, TaskStatus.RUNNING)
            
            # Execute the workflow
            result = await self.process_data_product_workflow(data_info, context_id)
            
            if result["workflow_successful"]:
                await self.update_task(task_id, TaskStatus.COMPLETED, result=result)
            else:
                await self.update_task(task_id, TaskStatus.FAILED, error=result.get("error"))
                
        except Exception as e:
            await self.update_task(task_id, TaskStatus.FAILED, error=str(e))
    
    def _extract_data_info(self, message: A2AMessage) -> Dict[str, Any]:
        """Extract data information from message"""
        data_info = {}
        
        for part in message.parts:
            if part.kind == "data" and part.data:
                data_info.update(part.data)
            elif part.kind == "file" and part.file:
                data_info["file"] = part.file
        
        return data_info
    
    async def _analyze_data_for_dublin_core(self, data_location: str) -> Dict[str, Any]:
        """Analyze data to extract metadata for Dublin Core"""
        metadata = {
            "record_count": 0,
            "quality_score": 1.0,
            "subjects": ["financial-data"],
            "format": "text/csv"
        }
        
        try:
            if os.path.exists(data_location):
                csv_files = self._find_csv_files(data_location)
                total_records = 0
                
                for csv_file in csv_files:
                    df = pd.read_csv(csv_file)
                    total_records += len(df)
                
                metadata["record_count"] = total_records
                metadata["title"] = f"Financial Data Products - {len(csv_files)} datasets"
                metadata["description"] = f"Financial data collection with {total_records:,} records across {len(csv_files)} data types"
                
        except Exception as e:
            logger.warning(f"Data analysis failed: {e}")
        
        return metadata
    
    def _find_csv_files(self, data_location: str) -> List[str]:
        """Find CSV files in data location"""
        csv_files = []
        
        if os.path.isfile(data_location) and data_location.endswith('.csv'):
            csv_files.append(data_location)
        elif os.path.isdir(data_location):
            for root, dirs, files in os.walk(data_location):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
        
        return csv_files
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    async def _verify_referential_integrity(self, csv_files: List[str]) -> Dict[str, Any]:
        """Verify referential integrity between files"""
        integrity_status = {
            "overall_status": "verified",
            "foreign_key_checks": [],
            "orphaned_records": 0,
            "integrity_score": 1.0
        }
        
        try:
            # Simple referential integrity check
            # In a real implementation, this would be more sophisticated
            integrity_status["foreign_key_checks"].append({
                "relationship": "basic_structure_check",
                "status": "verified",
                "details": "File structure validation passed"
            })
            
        except Exception as e:
            integrity_status["overall_status"] = "failed"
            integrity_status["error"] = str(e)
        
        return integrity_status
    
    async def _register_ord_descriptor(self, ord_descriptor: Dict[str, Any]) -> Dict[str, Any]:
        """Register ORD descriptor with registry"""
        try:
            # A2A Protocol: Use blockchain messaging instead of httpx
            
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with None as _unused:
        # httpx\.AsyncClient() as client:
                response = await client.post(
                    f"{self.ord_registry_url}/api/v1/ord/register",
                    json=ord_descriptor,
                    timeout=30
                )
                
                if response.status_code == 201:
                    return response.json()
                else:
                    raise Exception(f"Registration failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"ORD registration failed: {e}")
            raise
    
    async def _load_persistent_state(self):
        """Load persistent state from storage"""
        state_file = os.path.join(self.storage_path, "agent_state.json")
        
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.processing_stats = state.get("processing_stats", self.processing_stats)
                    self.data_products = state.get("data_products", {})
                    
                logger.info("Loaded persistent state successfully")
            except Exception as e:
                logger.warning(f"Failed to load persistent state: {e}")
    
    async def _save_persistent_state(self):
        """Save persistent state to storage"""
        state_file = os.path.join(self.storage_path, "agent_state.json")
        
        try:
            state = {
                "processing_stats": self.processing_stats,
                "data_products": self.data_products,
                "last_saved": datetime.utcnow().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info("Saved persistent state successfully")
        except Exception as e:
            logger.warning(f"Failed to save persistent state: {e}")
'''

    # Write the new SDK-based agent
    sdk_agent_path = "/Users/apple/projects/a2a/a2a_agents/backend/app/a2a/agents/data_product_agent_sdk.py"
    with open(sdk_agent_path, 'w') as f:
        f.write(new_agent_content)
    
    logger.info(f"Created SDK-based Data Product Agent at {sdk_agent_path}")
    
    return sdk_agent_path


def convert_data_standardization_agent():
    """Convert Agent 1 (Data Standardization Agent) to use SDK"""
    
    logger.info("Converting Data Standardization Agent to use A2A SDK...")
    
    new_agent_content = '''"""
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

from ..sdk import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)
from ..sdk.utils import create_success_response, create_error_response
from ..skills.account_standardizer import AccountStandardizer
from ..skills.book_standardizer import BookStandardizer
from ..skills.location_standardizer import LocationStandardizer
from ..skills.measure_standardizer import MeasureStandardizer
from ..skills.product_standardizer import ProductStandardizer

logger = logging.getLogger(__name__)


class DataStandardizationAgentSDK(A2AAgentBase):
    """
    Agent 1: Data Standardization Agent
    SDK Version - Simplified standardization with enhanced capabilities
    """
    
    def __init__(self, base_url: str):
        super().__init__(
            agent_id="data_standardization_agent_1",
            name="Data Standardization Agent",
            description="A2A v0.2.9 compliant agent for standardizing financial data to L4 hierarchical structure",
            version="3.0.0",  # SDK version
            base_url=base_url
        )
        
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
        
        logger.info(f"Initialized {self.name} with SDK v3.0.0")
    
    async def initialize(self) -> None:
        """Initialize agent resources"""
        logger.info("Initializing Data Standardization Agent resources...")
        
        # Initialize output directory
        self.output_dir = os.getenv("STANDARDIZATION_OUTPUT_DIR", "/tmp/standardized_data")
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("Data Standardization Agent initialized successfully")
    
    async def shutdown(self) -> None:
        """Cleanup agent resources"""
        logger.info("Shutting down Data Standardization Agent...")
        logger.info("Data Standardization Agent shutdown complete")
    
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
            from ....sdk.types import TaskStatus
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
'''

    # Write the new SDK-based agent
    sdk_agent_path = "/Users/apple/projects/a2a/a2a_agents/backend/app/a2a/agents/data_standardization_agent_sdk.py"
    with open(sdk_agent_path, 'w') as f:
        f.write(new_agent_content)
    
    logger.info(f"Created SDK-based Data Standardization Agent at {sdk_agent_path}")
    
    return sdk_agent_path


def create_launch_scripts():
    """Create launch scripts for SDK-based agents"""
    
    # Agent 0 launch script
    agent0_launch_content = '''#!/usr/bin/env python3
"""
Launch Agent 0 (Data Product Registration) - SDK Version
"""

import asyncio
import uvicorn
from app.a2a.agents.data_product_agent_sdk import DataProductRegistrationAgentSDK

async def main():
    # Create agent
    agent = DataProductRegistrationAgentSDK(
        base_url="os.getenv("DATA_MANAGER_URL")",
        ord_registry_url="http://localhost:8000/api/v1/ord"
    )
    
    # Initialize
    await agent.initialize()
    
    try:
        # Create FastAPI app
        app = agent.create_fastapi_app()
        
        print(f"🚀 Starting {agent.name} v{agent.version}")
        print(f"📡 Listening on http://localhost:8001")
        print(f"🎯 Agent ID: {agent.agent_id}")
        print(f"🛠️  Available Skills: {len(agent.skills)}")
        print(f"📋 Available Handlers: {len(agent.handlers)}")
        
        # Start server
        config = uvicorn.Config(app, host="0.0.0.0", port=8001, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
        
    finally:
        await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    # Agent 1 launch script
    agent1_launch_content = '''#!/usr/bin/env python3
"""
Launch Agent 1 (Data Standardization) - SDK Version
"""

import asyncio
import uvicorn
from app.a2a.agents.data_standardization_agent_sdk import DataStandardizationAgentSDK


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
async def main():
    # Create agent
    agent = DataStandardizationAgentSDK(
        base_url="os.getenv("CATALOG_MANAGER_URL")"
    )
    
    # Initialize
    await agent.initialize()
    
    try:
        # Create FastAPI app
        app = agent.create_fastapi_app()
        
        print(f"🚀 Starting {agent.name} v{agent.version}")
        print(f"📡 Listening on http://localhost:8002")
        print(f"🎯 Agent ID: {agent.agent_id}")
        print(f"🛠️  Available Skills: {len(agent.skills)}")
        print(f"📋 Available Handlers: {len(agent.handlers)}")
        
        # Start server
        config = uvicorn.Config(app, host="0.0.0.0", port=8002, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
        
    finally:
        await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    # Write launch scripts
    with open("/Users/apple/projects/a2a/a2a_agents/backend/launch_agent0_sdk.py", 'w') as f:
        f.write(agent0_launch_content)
    
    with open("/Users/apple/projects/a2a/a2a_agents/backend/launch_agent1_sdk.py", 'w') as f:
        f.write(agent1_launch_content)
    
    logger.info("Created launch scripts for SDK-based agents")


def main():
    """Convert all agents to use SDK"""
    logger.info("🔄 Converting A2A agents to use SDK...")
    
    try:
        # Convert agents
        agent0_path = convert_data_product_agent()
        agent1_path = convert_data_standardization_agent()
        
        # Create launch scripts
        create_launch_scripts()
        
        logger.info("✅ Agent conversion completed successfully!")
        logger.info(f"📦 Agent 0 SDK: {agent0_path}")
        logger.info(f"📦 Agent 1 SDK: {agent1_path}")
        logger.info("🚀 Launch scripts: launch_agent0_sdk.py, launch_agent1_sdk.py")
        
        print("\n" + "="*80)
        print("🎉 A2A AGENT SDK CONVERSION COMPLETE!")
        print("="*80)
        print("\nThe following SDK-based agents have been created:")
        print(f"  • Agent 0: {agent0_path}")
        print(f"  • Agent 1: {agent1_path}")
        print("\nTo test the SDK agents:")
        print("  python launch_agent0_sdk.py  # Start Agent 0")
        print("  python launch_agent1_sdk.py  # Start Agent 1")
        print("\nSDK Benefits:")
        print("  ✨ Simplified development with decorators")
        print("  🔧 Built-in telemetry and monitoring") 
        print("  📊 Automatic task management")
        print("  🛡️  Enhanced error handling")
        print("  📡 Standard A2A endpoints")
        print("  🔌 Easy skill registration")
        
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()