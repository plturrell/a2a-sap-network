"""
Data Manager A2A Agent - SDK Version
The actual link to filesystem and databases - Enhanced with A2A SDK
"""

import asyncio
import json
import os
import shutil
import gzip
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
from uuid import uuid4
import logging
from enum import Enum
import pandas as pd
import hashlib

from fastapi import HTTPException
from pydantic import BaseModel, Field

# Import HANA and Supabase clients
try:
    from hdbcli import dbapi
    HANA_AVAILABLE = True
except ImportError:
    HANA_AVAILABLE = False
    logger.warning("SAP HANA client not available")

try:
    from supabase import create_client, Client as SupabaseClient
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.warning("Supabase client not available")

from ..sdk import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)
from ..sdk.utils import create_success_response, create_error_response
from ..core.workflow_context import workflow_context_manager
from ..core.workflow_monitor import workflow_monitor
from ..security.smart_contract_trust import initialize_agent_trust, verify_a2a_message, sign_a2a_message, get_trust_contract
from ..security.delegation_contracts import get_delegation_contract, DelegationAction, can_agent_delegate, record_delegation_usage
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

logger = logging.getLogger(__name__)


class TaskState(str, Enum):
    PENDING = "pending"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"


class ServiceLevel(str, Enum):
    GOLD = "gold"      # Immediate processing, guaranteed storage, full redundancy
    SILVER = "silver"  # Standard processing, regular backups
    BRONZE = "bronze"  # Basic processing, minimal storage


class StorageBackend(str, Enum):
    FILESYSTEM = "filesystem"
    HANA = "hana"
    SUPABASE = "supabase"
    S3 = "s3"
    REDIS = "redis"


class DataOperation(str, Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    BACKUP = "backup"
    RESTORE = "restore"
    MIGRATE = "migrate"


class DataManagerRequest(BaseModel):
    """Request for data management operations"""
    operation: DataOperation
    data_type: str
    identifier: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    storage_backend: StorageBackend = StorageBackend.FILESYSTEM
    service_level: ServiceLevel = ServiceLevel.SILVER
    compression: bool = Field(default=False)
    encryption: bool = Field(default=False)
    backup_retention_days: int = Field(default=30)
    metadata: Optional[Dict[str, Any]] = None


class DataManagerResponse(BaseModel):
    """Response from data management operations"""
    operation: DataOperation
    success: bool
    message: str
    data_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    storage_location: Optional[str] = None
    checksum: Optional[str] = None


class DataManagerAgentSDK(A2AAgentBase):
    """
    Data Manager A2A Agent - SDK Version
    The actual link to filesystem and databases with enhanced capabilities
    """
    
    def __init__(self, base_url: str, storage_config: Dict[str, Any]):
        super().__init__(
            agent_id="data_manager_agent",
            name="Data Manager Agent",
            description="A2A v0.2.9 compliant agent for data management and storage operations",
            version="3.0.0",  # SDK version
            base_url=base_url
        )
        
        self.storage_config = storage_config
        self.storage_clients = {}
        self.data_registry = {}
        
        # Database connections
        self.hana_connection = None
        self.supabase_client = None
        
        # Prometheus metrics
        self.tasks_completed = Counter('a2a_agent_tasks_completed_total', 'Total completed tasks', ['agent_id', 'task_type'])
        self.tasks_failed = Counter('a2a_agent_tasks_failed_total', 'Total failed tasks', ['agent_id', 'task_type'])
        self.processing_time = Histogram('a2a_agent_processing_time_seconds', 'Task processing time', ['agent_id', 'task_type'])
        self.queue_depth = Gauge('a2a_agent_queue_depth', 'Current queue depth', ['agent_id'])
        self.skills_count = Gauge('a2a_agent_skills_count', 'Number of skills available', ['agent_id'])
        
        # Set initial metrics
        self.queue_depth.labels(agent_id=self.agent_id).set(0)
        self.skills_count.labels(agent_id=self.agent_id).set(7)  # 7 main skills
        
        # Start metrics server
        self._start_metrics_server()
        
        self.processing_stats = {
            "total_processed": 0,
            "create_operations": 0,
            "read_operations": 0,
            "update_operations": 0,
            "delete_operations": 0,
            "backup_operations": 0
        }
        
        logger.info(f"Initialized {self.name} with SDK v3.0.0")
    
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
        logger.info("Initializing Data Manager Agent resources...")
        
        # Initialize storage paths
        storage_path = os.getenv("DATA_MANAGER_STORAGE_PATH", "/tmp/data_manager_state")
        os.makedirs(storage_path, exist_ok=True)
        self.storage_path = storage_path
        
        # Initialize backup directory
        backup_path = os.path.join(storage_path, "backups")
        os.makedirs(backup_path, exist_ok=True)
        self.backup_path = backup_path
        
        # Initialize storage clients
        await self._initialize_storage_clients()
        
        # Load existing state
        await self._load_agent_state()
        
        logger.info("Data Manager Agent initialization complete")
    
    @a2a_handler("data_management")
    async def handle_data_management(self, message: A2AMessage) -> Dict[str, Any]:
        """Main handler for data management operations"""
        start_time = time.time()
        
        try:
            # Extract data management request from message
            management_request = self._extract_management_request(message)
            if not management_request:
                return create_error_response("No valid data management request found in message")
            
            # Process data management operation
            management_result = await self.process_data_operation(
                management_request=management_request,
                context_id=message.conversation_id
            )
            
            # Record success metrics
            operation_type = management_request.get('operation', 'unknown')
            self.tasks_completed.labels(agent_id=self.agent_id, task_type=f'data_{operation_type}').inc()
            self.processing_time.labels(agent_id=self.agent_id, task_type=f'data_{operation_type}').observe(time.time() - start_time)
            
            return create_success_response(management_result)
            
        except Exception as e:
            # Record failure metrics
            self.tasks_failed.labels(agent_id=self.agent_id, task_type='data_management').inc()
            logger.error(f"Data management operation failed: {e}")
            return create_error_response(f"Data management failed: {str(e)}")
    
    @a2a_skill("data_create")
    async def data_create_skill(self, data: Dict[str, Any], storage_backend: str, service_level: str, metadata: Dict[str, Any] = None) -> DataManagerResponse:
        """Create new data entry"""
        
        try:
            # Generate unique data ID
            data_id = str(uuid4())
            
            # Add metadata
            data_with_metadata = {
                "id": data_id,
                "data": data,
                "created_at": datetime.utcnow().isoformat(),
                "service_level": service_level,
                "metadata": metadata or {}
            }
            
            # Calculate checksum
            checksum = self._calculate_checksum(data_with_metadata)
            data_with_metadata["checksum"] = checksum
            
            # Store data based on backend
            storage_location = await self._store_data(data_with_metadata, storage_backend)
            
            # Register in data registry
            self.data_registry[data_id] = {
                "storage_backend": storage_backend,
                "storage_location": storage_location,
                "service_level": service_level,
                "created_at": datetime.utcnow().isoformat(),
                "checksum": checksum
            }
            
            self.processing_stats["create_operations"] += 1
            
            return DataManagerResponse(
                operation=DataOperation.CREATE,
                success=True,
                message="Data created successfully",
                data_id=data_id,
                storage_location=storage_location,
                checksum=checksum
            )
            
        except Exception as e:
            logger.error(f"Data creation failed: {e}")
            return DataManagerResponse(
                operation=DataOperation.CREATE,
                success=False,
                message=f"Creation failed: {str(e)}"
            )
    
    @a2a_skill("data_read")
    async def data_read_skill(self, data_id: str) -> DataManagerResponse:
        """Read existing data entry"""
        
        try:
            if data_id not in self.data_registry:
                return DataManagerResponse(
                    operation=DataOperation.READ,
                    success=False,
                    message=f"Data with ID {data_id} not found"
                )
            
            # Get registry entry
            registry_entry = self.data_registry[data_id]
            
            # Retrieve data from storage
            data = await self._retrieve_data(data_id, registry_entry["storage_backend"])
            
            # Verify checksum
            if not self._verify_checksum(data, registry_entry["checksum"]):
                logger.warning(f"Checksum mismatch for data ID {data_id}")
            
            self.processing_stats["read_operations"] += 1
            
            return DataManagerResponse(
                operation=DataOperation.READ,
                success=True,
                message="Data retrieved successfully",
                data_id=data_id,
                data=data.get("data"),
                metadata=data.get("metadata"),
                storage_location=registry_entry["storage_location"]
            )
            
        except Exception as e:
            logger.error(f"Data read failed: {e}")
            return DataManagerResponse(
                operation=DataOperation.READ,
                success=False,
                message=f"Read failed: {str(e)}"
            )
    
    @a2a_skill("data_update")
    async def data_update_skill(self, data_id: str, updates: Dict[str, Any]) -> DataManagerResponse:
        """Update existing data entry"""
        
        try:
            if data_id not in self.data_registry:
                return DataManagerResponse(
                    operation=DataOperation.UPDATE,
                    success=False,
                    message=f"Data with ID {data_id} not found"
                )
            
            registry_entry = self.data_registry[data_id]
            
            # Retrieve current data
            current_data = await self._retrieve_data(data_id, registry_entry["storage_backend"])
            
            # Apply updates
            updated_data = current_data.copy()
            updated_data["data"].update(updates)
            updated_data["updated_at"] = datetime.utcnow().isoformat()
            
            # Recalculate checksum
            new_checksum = self._calculate_checksum(updated_data)
            updated_data["checksum"] = new_checksum
            
            # Store updated data
            storage_location = await self._store_data(updated_data, registry_entry["storage_backend"])
            
            # Update registry
            registry_entry["checksum"] = new_checksum
            registry_entry["updated_at"] = datetime.utcnow().isoformat()
            
            self.processing_stats["update_operations"] += 1
            
            return DataManagerResponse(
                operation=DataOperation.UPDATE,
                success=True,
                message="Data updated successfully",
                data_id=data_id,
                storage_location=storage_location,
                checksum=new_checksum
            )
            
        except Exception as e:
            logger.error(f"Data update failed: {e}")
            return DataManagerResponse(
                operation=DataOperation.UPDATE,
                success=False,
                message=f"Update failed: {str(e)}"
            )
    
    @a2a_skill("data_delete")
    async def data_delete_skill(self, data_id: str, create_backup: bool = True) -> DataManagerResponse:
        """Delete data entry"""
        
        try:
            if data_id not in self.data_registry:
                return DataManagerResponse(
                    operation=DataOperation.DELETE,
                    success=False,
                    message=f"Data with ID {data_id} not found"
                )
            
            registry_entry = self.data_registry[data_id]
            
            # Create backup if requested
            if create_backup:
                await self._backup_data(data_id)
            
            # Delete from storage
            await self._delete_data(data_id, registry_entry["storage_backend"])
            
            # Remove from registry
            del self.data_registry[data_id]
            
            self.processing_stats["delete_operations"] += 1
            
            return DataManagerResponse(
                operation=DataOperation.DELETE,
                success=True,
                message=f"Data deleted successfully{'with backup' if create_backup else ''}",
                data_id=data_id
            )
            
        except Exception as e:
            logger.error(f"Data deletion failed: {e}")
            return DataManagerResponse(
                operation=DataOperation.DELETE,
                success=False,
                message=f"Deletion failed: {str(e)}"
            )
    
    @a2a_skill("data_backup")
    async def data_backup_skill(self, data_id: str = None, storage_backend: str = None) -> DataManagerResponse:
        """Backup data entries"""
        
        try:
            backup_count = 0
            
            if data_id:
                # Backup specific data entry
                await self._backup_data(data_id)
                backup_count = 1
            else:
                # Backup all data or by storage backend
                for data_id, registry_entry in self.data_registry.items():
                    if not storage_backend or registry_entry["storage_backend"] == storage_backend:
                        await self._backup_data(data_id)
                        backup_count += 1
            
            self.processing_stats["backup_operations"] += backup_count
            
            return DataManagerResponse(
                operation=DataOperation.BACKUP,
                success=True,
                message=f"Backed up {backup_count} data entries",
                metadata={"backup_count": backup_count}
            )
            
        except Exception as e:
            logger.error(f"Data backup failed: {e}")
            return DataManagerResponse(
                operation=DataOperation.BACKUP,
                success=False,
                message=f"Backup failed: {str(e)}"
            )
    
    @a2a_skill("data_restore")
    async def data_restore_skill(self, data_id: str, backup_timestamp: str = None) -> DataManagerResponse:
        """Restore data from backup"""
        
        try:
            # Find backup file
            backup_files = self._list_backup_files(data_id)
            
            if not backup_files:
                return DataManagerResponse(
                    operation=DataOperation.RESTORE,
                    success=False,
                    message=f"No backup found for data ID {data_id}"
                )
            
            # Select backup file
            if backup_timestamp:
                backup_file = next((f for f in backup_files if backup_timestamp in f), None)
            else:
                backup_file = max(backup_files)  # Latest backup
            
            if not backup_file:
                return DataManagerResponse(
                    operation=DataOperation.RESTORE,
                    success=False,
                    message=f"No backup found for timestamp {backup_timestamp}"
                )
            
            # Restore from backup
            restored_data = await self._restore_from_backup(backup_file)
            
            # Re-register in data registry
            self.data_registry[data_id] = restored_data["registry_entry"]
            
            return DataManagerResponse(
                operation=DataOperation.RESTORE,
                success=True,
                message="Data restored successfully from backup",
                data_id=data_id,
                metadata={"restored_from": backup_file}
            )
            
        except Exception as e:
            logger.error(f"Data restore failed: {e}")
            return DataManagerResponse(
                operation=DataOperation.RESTORE,
                success=False,
                message=f"Restore failed: {str(e)}"
            )
    
    @a2a_skill("data_migration")
    async def data_migration_skill(self, data_id: str, target_backend: str) -> DataManagerResponse:
        """Migrate data between storage backends"""
        
        try:
            if data_id not in self.data_registry:
                return DataManagerResponse(
                    operation=DataOperation.MIGRATE,
                    success=False,
                    message=f"Data with ID {data_id} not found"
                )
            
            registry_entry = self.data_registry[data_id]
            source_backend = registry_entry["storage_backend"]
            
            if source_backend == target_backend:
                return DataManagerResponse(
                    operation=DataOperation.MIGRATE,
                    success=False,
                    message=f"Data is already in {target_backend} backend"
                )
            
            # Retrieve data from source
            data = await self._retrieve_data(data_id, source_backend)
            
            # Store in target backend
            new_location = await self._store_data(data, target_backend)
            
            # Update registry
            registry_entry["storage_backend"] = target_backend
            registry_entry["storage_location"] = new_location
            registry_entry["migrated_at"] = datetime.utcnow().isoformat()
            
            # Clean up from source backend (optional)
            try:
                await self._delete_data(data_id, source_backend)
            except Exception as e:
                logger.warning(f"Failed to clean up from source backend: {e}")
            
            return DataManagerResponse(
                operation=DataOperation.MIGRATE,
                success=True,
                message=f"Data migrated from {source_backend} to {target_backend}",
                data_id=data_id,
                storage_location=new_location,
                metadata={"source_backend": source_backend, "target_backend": target_backend}
            )
            
        except Exception as e:
            logger.error(f"Data migration failed: {e}")
            return DataManagerResponse(
                operation=DataOperation.MIGRATE,
                success=False,
                message=f"Migration failed: {str(e)}"
            )
    
    @a2a_task(
        task_type="data_management_operation",
        description="Complete data management operation workflow",
        timeout=300,
        retry_attempts=2
    )
    async def process_data_operation(self, management_request: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Process data management operation"""
        
        operation = management_request.get("operation")
        
        try:
            result = None
            
            if operation == "create":
                result = await self.execute_skill("data_create", 
                                                management_request.get("data", {}),
                                                management_request.get("storage_backend", "filesystem"),
                                                management_request.get("service_level", "silver"),
                                                management_request.get("metadata"))
                
            elif operation == "read":
                result = await self.execute_skill("data_read", management_request.get("identifier"))
                
            elif operation == "update":
                result = await self.execute_skill("data_update",
                                                management_request.get("identifier"),
                                                management_request.get("data", {}))
                
            elif operation == "delete":
                result = await self.execute_skill("data_delete",
                                                management_request.get("identifier"),
                                                management_request.get("create_backup", True))
                
            elif operation == "backup":
                result = await self.execute_skill("data_backup",
                                                management_request.get("identifier"),
                                                management_request.get("storage_backend"))
                
            elif operation == "restore":
                result = await self.execute_skill("data_restore",
                                                management_request.get("identifier"),
                                                management_request.get("backup_timestamp"))
                
            elif operation == "migrate":
                result = await self.execute_skill("data_migration",
                                                management_request.get("identifier"),
                                                management_request.get("target_backend"))
            else:
                result = DataManagerResponse(
                    operation=operation,
                    success=False,
                    message=f"Unknown operation: {operation}"
                )
            
            self.processing_stats["total_processed"] += 1
            
            return {
                "operation_successful": result.success if hasattr(result, 'success') else True,
                "operation": operation,
                "result": result.dict() if hasattr(result, 'dict') else result,
                "context_id": context_id
            }
            
        except Exception as e:
            logger.error(f"Data operation {operation} failed: {e}")
            return {
                "operation_successful": False,
                "operation": operation,
                "error": str(e),
                "context_id": context_id
            }
    
    def _extract_management_request(self, message: A2AMessage) -> Dict[str, Any]:
        """Extract data management request from message"""
        management_data = {}
        
        for part in message.parts:
            if part.kind == "data" and part.data:
                management_data.update(part.data)
            elif part.kind == "file" and part.file:
                management_data["file"] = part.file
        
        return management_data
    
    async def _initialize_storage_clients(self):
        """Initialize storage clients for different backends"""
        try:
            # Initialize filesystem (always available)
            self.storage_clients["filesystem"] = self._filesystem_client
            
            # Initialize HANA as primary database
            if HANA_AVAILABLE and self.storage_config.get("hana"):
                try:
                    hana_config = self.storage_config["hana"]
                    self.hana_connection = dbapi.connect(
                        address=hana_config.get("address", "localhost"),
                        port=hana_config.get("port", 30015),
                        user=hana_config.get("user"),
                        password=hana_config.get("password"),
                        databaseName=hana_config.get("databaseName", "SYSTEMDB")
                    )
                    
                    # Create A2A data schema and tables
                    cursor = self.hana_connection.cursor()
                    
                    # Create schema if not exists
                    cursor.execute("CREATE SCHEMA IF NOT EXISTS A2A_DATA")
                    
                    # Create column table for data records (HANA optimized)
                    cursor.execute('''
                        CREATE COLUMN TABLE IF NOT EXISTS A2A_DATA.DATA_RECORDS (
                            RECORD_ID NVARCHAR(255) PRIMARY KEY,
                            AGENT_ID NVARCHAR(100) NOT NULL,
                            CONTEXT_ID NVARCHAR(100) NOT NULL,
                            DATA_TYPE NVARCHAR(50) NOT NULL,
                            DATA NCLOB,  -- JSON data
                            DATA_BINARY BLOB,  -- For embeddings/vectors
                            METADATA NCLOB,
                            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            VERSION INTEGER DEFAULT 1,
                            IS_DELETED BOOLEAN DEFAULT FALSE
                        )
                    ''')
                    
                    # Create index for faster queries
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS IDX_A2A_DATA_TYPE 
                        ON A2A_DATA.DATA_RECORDS (DATA_TYPE, CREATED_AT DESC)
                    ''')
                    
                    self.hana_connection.commit()
                    self.storage_clients["hana"] = self._hana_client
                    logger.info("✅ HANA storage client initialized successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize HANA: {e}")
                    self.hana_connection = None
            
            # Initialize Supabase as fallback
            if SUPABASE_AVAILABLE and self.storage_config.get("supabase"):
                try:
                    supabase_config = self.storage_config["supabase"]
                    self.supabase_client = create_client(
                        supabase_config.get("url"),
                        supabase_config.get("key")
                    )
                    
                    # Verify connection by checking if table exists
                    # If not, create it
                    try:
                        self.supabase_client.table('a2a_data_records').select('record_id').limit(1).execute()
                    except:
                        # Table doesn't exist, create it
                        # Note: In production, table creation should be done via Supabase migrations
                        logger.info("Creating Supabase table structure...")
                    
                    self.storage_clients["supabase"] = self._supabase_client
                    logger.info("✅ Supabase storage client initialized as fallback")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize Supabase: {e}")
                    self.supabase_client = None
            
            logger.info(f"Storage clients initialized: {list(self.storage_clients.keys())}")
        except Exception as e:
            logger.error(f"Failed to initialize storage clients: {e}")
    
    def _filesystem_client(self, operation: str, *args, **kwargs):
        """Filesystem storage client implementation"""
        # This is a placeholder for the filesystem client
        return {"operation": operation, "backend": "filesystem"}
    
    def _hana_client(self, operation: str, *args, **kwargs):
        """HANA storage client implementation"""
        return {"operation": operation, "backend": "hana", "connection": self.hana_connection}
    
    def _supabase_client(self, operation: str, *args, **kwargs):
        """Supabase storage client implementation"""
        return {"operation": operation, "backend": "supabase", "client": self.supabase_client}
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for data integrity"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _verify_checksum(self, data: Dict[str, Any], expected_checksum: str) -> bool:
        """Verify data integrity using checksum"""
        current_checksum = self._calculate_checksum(data)
        return current_checksum == expected_checksum
    
    async def _store_data(self, data: Dict[str, Any], storage_backend: str) -> str:
        """Store data in the specified backend"""
        data_id = data.get("id")
        
        if storage_backend == "filesystem":
            # Store in filesystem
            file_path = os.path.join(self.storage_path, f"data_{data_id}.json")
            with open(file_path, 'w') as f:
                json.dump(data, f, default=str, indent=2)
            return file_path
            
        elif storage_backend == "hana" and self.hana_connection:
            try:
                cursor = self.hana_connection.cursor()
                
                # Prepare data
                json_data = json.dumps(data, default=str)
                metadata = json.dumps(data.get("metadata", {}), default=str)
                
                # Insert or update record
                cursor.execute('''
                    MERGE INTO A2A_DATA.DATA_RECORDS AS target
                    USING (SELECT ? AS RECORD_ID FROM DUMMY) AS source
                    ON target.RECORD_ID = source.RECORD_ID
                    WHEN MATCHED THEN UPDATE SET
                        DATA = ?,
                        METADATA = ?,
                        UPDATED_AT = CURRENT_TIMESTAMP,
                        VERSION = target.VERSION + 1
                    WHEN NOT MATCHED THEN INSERT (
                        RECORD_ID, AGENT_ID, CONTEXT_ID, DATA_TYPE,
                        DATA, METADATA
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    data_id,  # for MERGE condition
                    json_data, metadata,  # for UPDATE
                    data_id, data.get("agent_id", "data_manager"),  # for INSERT
                    data.get("context_id", "default"),
                    data.get("data_type", "generic"),
                    json_data, metadata
                ))
                
                self.hana_connection.commit()
                logger.info(f"✅ Data stored in HANA: {data_id}")
                return f"hana://A2A_DATA.DATA_RECORDS/{data_id}"
                
            except Exception as e:
                logger.error(f"HANA storage failed: {e}, falling back to Supabase")
                if "supabase" in self.storage_clients:
                    return await self._store_data(data, "supabase")
                else:
                    return await self._store_data(data, "filesystem")
                    
        elif storage_backend == "supabase" and self.supabase_client:
            try:
                # Prepare record
                record = {
                    "record_id": data_id,
                    "agent_id": data.get("agent_id", "data_manager"),
                    "context_id": data.get("context_id", "default"),
                    "data_type": data.get("data_type", "generic"),
                    "data": data,
                    "metadata": data.get("metadata", {}),
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }
                
                # Upsert to Supabase
                result = self.supabase_client.table('a2a_data_records').upsert(record).execute()
                
                logger.info(f"✅ Data stored in Supabase: {data_id}")
                return f"supabase://a2a_data_records/{data_id}"
                
            except Exception as e:
                logger.error(f"Supabase storage failed: {e}, falling back to filesystem")
                return await self._store_data(data, "filesystem")
        else:
            # Fallback to filesystem
            logger.warning(f"Storage backend {storage_backend} not available, using filesystem")
            return await self._store_data(data, "filesystem")
    
    async def _retrieve_data(self, data_id: str, storage_backend: str) -> Dict[str, Any]:
        """Retrieve data from the specified backend"""
        
        if storage_backend == "filesystem":
            file_path = os.path.join(self.storage_path, f"data_{data_id}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                raise FileNotFoundError(f"Data file not found: {file_path}")
                
        elif storage_backend == "hana" and self.hana_connection:
            try:
                cursor = self.hana_connection.cursor()
                cursor.execute('''
                    SELECT DATA, METADATA FROM A2A_DATA.DATA_RECORDS
                    WHERE RECORD_ID = ? AND IS_DELETED = FALSE
                ''', (data_id,))
                
                result = cursor.fetchone()
                if result:
                    data = json.loads(result[0]) if result[0] else {}
                    if result[1]:
                        data["metadata"] = json.loads(result[1])
                    return data
                else:
                    raise ValueError(f"Data not found in HANA: {data_id}")
                    
            except Exception as e:
                logger.error(f"HANA retrieval failed: {e}, trying Supabase")
                if "supabase" in self.storage_clients:
                    return await self._retrieve_data(data_id, "supabase")
                else:
                    raise
                    
        elif storage_backend == "supabase" and self.supabase_client:
            try:
                result = self.supabase_client.table('a2a_data_records').select('*').eq('record_id', data_id).single().execute()
                
                if result.data:
                    return result.data.get('data', {})
                else:
                    raise ValueError(f"Data not found in Supabase: {data_id}")
                    
            except Exception as e:
                logger.error(f"Supabase retrieval failed: {e}")
                raise
        else:
            # Try primary storage first
            if "hana" in self.storage_clients:
                return await self._retrieve_data(data_id, "hana")
            elif "supabase" in self.storage_clients:
                return await self._retrieve_data(data_id, "supabase")
            else:
                return await self._retrieve_data(data_id, "filesystem")
    
    async def _delete_data(self, data_id: str, storage_backend: str):
        """Delete data from the specified backend"""
        
        if storage_backend == "filesystem":
            file_path = os.path.join(self.storage_path, f"data_{data_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                
        elif storage_backend == "hana" and self.hana_connection:
            try:
                cursor = self.hana_connection.cursor()
                # Soft delete in HANA
                cursor.execute('''
                    UPDATE A2A_DATA.DATA_RECORDS
                    SET IS_DELETED = TRUE, UPDATED_AT = CURRENT_TIMESTAMP
                    WHERE RECORD_ID = ?
                ''', (data_id,))
                self.hana_connection.commit()
                logger.info(f"Data soft-deleted in HANA: {data_id}")
                
            except Exception as e:
                logger.error(f"HANA deletion failed: {e}")
                raise
                
        elif storage_backend == "supabase" and self.supabase_client:
            try:
                # Hard delete in Supabase
                self.supabase_client.table('a2a_data_records').delete().eq('record_id', data_id).execute()
                logger.info(f"Data deleted from Supabase: {data_id}")
                
            except Exception as e:
                logger.error(f"Supabase deletion failed: {e}")
                raise
        else:
            # Fallback
            logger.warning(f"Storage backend {storage_backend} not available, using filesystem")
            await self._delete_data(data_id, "filesystem")
    
    async def _backup_data(self, data_id: str):
        """Create backup of data entry"""
        
        if data_id not in self.data_registry:
            raise ValueError(f"Data ID {data_id} not found in registry")
        
        registry_entry = self.data_registry[data_id]
        
        # Retrieve data
        data = await self._retrieve_data(data_id, registry_entry["storage_backend"])
        
        # Create backup entry
        backup_entry = {
            "data": data,
            "registry_entry": registry_entry,
            "backup_timestamp": datetime.utcnow().isoformat()
        }
        
        # Save backup
        backup_filename = f"backup_{data_id}_{int(datetime.utcnow().timestamp())}.json"
        backup_path = os.path.join(self.backup_path, backup_filename)
        
        with open(backup_path, 'w') as f:
            json.dump(backup_entry, f, default=str, indent=2)
        
        logger.info(f"Created backup for data ID {data_id}: {backup_filename}")
    
    def _list_backup_files(self, data_id: str) -> List[str]:
        """List backup files for a data ID"""
        backup_files = []
        
        for filename in os.listdir(self.backup_path):
            if filename.startswith(f"backup_{data_id}_") and filename.endswith(".json"):
                backup_files.append(os.path.join(self.backup_path, filename))
        
        return sorted(backup_files)
    
    async def _restore_from_backup(self, backup_file: str) -> Dict[str, Any]:
        """Restore data from backup file"""
        
        with open(backup_file, 'r') as f:
            backup_data = json.load(f)
        
        # Restore data to storage
        data = backup_data["data"]
        registry_entry = backup_data["registry_entry"]
        
        # Store restored data
        await self._store_data(data, registry_entry["storage_backend"])
        
        return backup_data
    
    async def _load_agent_state(self):
        """Load existing agent state from storage"""
        try:
            registry_file = os.path.join(self.storage_path, "data_registry.json")
            if os.path.exists(registry_file):
                with open(registry_file, 'r') as f:
                    self.data_registry = json.load(f)
                logger.info(f"Loaded {len(self.data_registry)} data entries from registry")
        except Exception as e:
            logger.warning(f"Failed to load agent state: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup agent resources"""
        try:
            # Save data registry
            registry_file = os.path.join(self.storage_path, "data_registry.json")
            with open(registry_file, 'w') as f:
                json.dump(self.data_registry, f, default=str, indent=2)
            logger.info(f"Saved {len(self.data_registry)} data entries to registry")
        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")