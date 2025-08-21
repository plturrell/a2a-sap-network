"""
ORD Registry Dual-Database Storage Layer
HANA as Primary, SQLite as Fallback with Data Replication
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from uuid import uuid4
import asyncio

# Use absolute imports when module is in sys.path
from clients.hanaClient import get_hana_client
from clients.sqliteClient import get_sqlite_client
from registry.models import (
    ORDRegistration, ResourceIndexEntry, ORDDocument,
    RegistrationMetadata, ValidationResult, RegistrationStatus
)

logger = logging.getLogger(__name__)


class ORDDualDatabaseStorage:
    """
    Dual-database storage for ORD Registry
    - HANA as primary (enterprise performance)
    - SQLite as fallback (local backup + offline capability)
    - Bidirectional data replication
    """
    
    def __init__(self):
        self.hana_client = None
        self.sqlite_client = None
        self.replication_enabled = True
        self.fallback_mode = False
        
    async def initialize(self):
        """Initialize both database connections and create tables if needed"""
        try:
            # Try to initialize HANA client (primary) but fallback if not available
            try:
                self.hana_client = get_hana_client()
                # Test HANA connectivity to ensure it's working
                try:
                    health_check = self.hana_client.health_check()
                    if health_check.get("status") != "healthy":
                        raise Exception("HANA health check failed")
                    logger.info("HANA client initialized and healthy for ORD registry")
                except Exception as health_error:
                    logger.warning(f"HANA client initialized but connection failed: {health_error}")
                    self.hana_client = None
                    self.fallback_mode = True
            except Exception as e:
                logger.warning(f"HANA client not available, using SQLite-only mode: {e}")
                self.hana_client = None
                self.fallback_mode = True
            
            # Initialize SQLite client (fallback/primary if HANA unavailable)
            self.sqlite_client = get_sqlite_client()
            logger.info("SQLite client initialized for ORD registry")
            
            # Create tables in available systems
            if self.hana_client is not None:
                await self._create_hana_tables()
            await self._create_sqlite_tables()
            
            # Verify connectivity
            await self._verify_connectivity()
            
            logger.info("✅ Dual-database ORD storage initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize dual-database storage: {e}")
            self.fallback_mode = True
            
    async def _create_hana_tables(self):
        """Create ORD registry tables in HANA"""
        try:
            # Check if tables exist first (HANA-compatible approach)
            table_exists_sql = """
            SELECT COUNT(*) as count FROM SYS.TABLES 
            WHERE SCHEMA_NAME = CURRENT_SCHEMA AND TABLE_NAME = ?
            """
            
            # Main registrations table (HANA-compatible syntax)
            create_registrations_sql = """
            CREATE TABLE ord_registrations (
                registration_id NVARCHAR(50) PRIMARY KEY,
                ord_document NCLOB,
                registered_by NVARCHAR(100),
                registered_at TIMESTAMP,
                last_updated TIMESTAMP,
                version NVARCHAR(20),
                status NVARCHAR(20),
                validation_result NCLOB,
                governance_info NCLOB,
                analytics_info NCLOB
            )
            """
            
            # Resource index table for fast search (HANA-compatible)
            create_index_sql = """
            CREATE TABLE ord_resource_index (
                ord_id NVARCHAR(200) PRIMARY KEY,
                registration_id NVARCHAR(50),
                resource_type NVARCHAR(50),
                title NVARCHAR(500),
                description NCLOB,
                short_description NVARCHAR(1000),
                version NVARCHAR(20),
                tags NCLOB,
                labels NCLOB,
                domain NVARCHAR(100),
                category NVARCHAR(100),
                indexed_at TIMESTAMP,
                searchable_content NCLOB,
                access_strategies NCLOB,
                dublin_core NCLOB,
                dc_creator NCLOB,
                dc_subject NCLOB,
                dc_publisher NVARCHAR(200),
                dc_format NVARCHAR(100),
                FOREIGN KEY (registration_id) REFERENCES ord_registrations(registration_id)
            )
            """
            
            # Replication log table (HANA-compatible)
            create_replication_sql = """
            CREATE TABLE ord_replication_log (
                id NVARCHAR(50) PRIMARY KEY,
                table_name NVARCHAR(100),
                operation NVARCHAR(20),
                record_id NVARCHAR(200),
                timestamp TIMESTAMP,
                status NVARCHAR(20),
                error_message NCLOB
            )
            """
            
            # Execute table creation with existence checking
            await self._create_hana_table_if_not_exists('ORD_REGISTRATIONS', create_registrations_sql)
            await self._create_hana_table_if_not_exists('ORD_RESOURCE_INDEX', create_index_sql)
            await self._create_hana_table_if_not_exists('ORD_REPLICATION_LOG', create_replication_sql)
            
            logger.info("✅ HANA ORD tables created successfully")
            
            # Add missing columns if tables already exist
            await self._add_missing_columns_hana()
            
        except Exception as e:
            logger.error(f"Failed to create HANA tables: {e}")
            raise
    
    async def _add_missing_columns_hana(self):
        """Add missing columns to existing HANA tables"""
        try:
            # Check if ACCESS_STRATEGIES column exists in ORD_RESOURCE_INDEX
            check_column_sql = """
            SELECT COUNT(*) as count FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = CURRENT_SCHEMA 
            AND TABLE_NAME = 'ORD_RESOURCE_INDEX' 
            AND COLUMN_NAME = 'ACCESS_STRATEGIES'
            """
            
            result = self.hana_client.execute_query(check_column_sql)
            
            if result.data and result.data[0]['COUNT'] == 0:
                # Column doesn't exist, add it
                logger.info("Adding missing ACCESS_STRATEGIES column to ORD_RESOURCE_INDEX")
                add_column_sql = """
                ALTER TABLE ord_resource_index 
                ADD (access_strategies NCLOB)
                """
                self.hana_client.execute_query(add_column_sql)
                logger.info("✅ Added ACCESS_STRATEGIES column successfully")
            else:
                logger.debug("ACCESS_STRATEGIES column already exists")
                
        except Exception as e:
            logger.warning(f"Failed to add missing columns: {e}")
            # Don't raise - this is not critical if column already exists
            
    async def _create_hana_table_if_not_exists(self, table_name: str, create_sql: str):
        """Create HANA table only if it doesn't exist (HANA-compatible approach)"""
        try:
            # Check if table exists
            check_sql = """
            SELECT COUNT(*) as count FROM SYS.TABLES 
            WHERE SCHEMA_NAME = CURRENT_SCHEMA AND TABLE_NAME = ?
            """
            
            result = self.hana_client.execute_query(check_sql, [table_name])
            
            # If table doesn't exist, create it
            if result.data and len(result.data) > 0:
                count = result.data[0]['COUNT']
                if count == 0:
                    self.hana_client.execute_query(create_sql)
                    logger.info(f"✅ Created HANA table: {table_name}")
                else:
                    logger.info(f"📋 HANA table already exists: {table_name}")
            else:
                # Fallback: try to create (will fail if exists)
                try:
                    self.hana_client.execute_query(create_sql)
                    logger.info(f"✅ Created HANA table: {table_name}")
                except Exception as create_error:
                    if "already exists" in str(create_error).lower():
                        logger.info(f"📋 HANA table already exists: {table_name}")
                    else:
                        raise create_error
                        
        except Exception as e:
            logger.error(f"Failed to create/check HANA table {table_name}: {e}")
            raise
            
    async def _create_sqlite_tables(self):
        """Create ORD registry tables in SQLite"""
        try:
            # Create tables using SQLite client
            # Note: In production, these would be created via SQLite migrations
            
            # Create registrations table schema
            registrations_schema = {
                "table_name": "ord_registrations",
                "columns": [
                    {"name": "registration_id", "type": "text", "primary_key": True},
                    {"name": "ord_document", "type": "jsonb"},
                    {"name": "registered_by", "type": "text"},
                    {"name": "registered_at", "type": "timestamptz"},
                    {"name": "last_updated", "type": "timestamptz"},
                    {"name": "version", "type": "text"},
                    {"name": "status", "type": "text"},
                    {"name": "validation_result", "type": "jsonb"},
                    {"name": "governance_info", "type": "jsonb"},
                    {"name": "analytics_info", "type": "jsonb"}
                ]
            }
            
            # Create resource index table schema
            index_schema = {
                "table_name": "ord_resource_index",
                "columns": [
                    {"name": "ord_id", "type": "text", "primary_key": True},
                    {"name": "registration_id", "type": "text"},
                    {"name": "resource_type", "type": "text"},
                    {"name": "title", "type": "text"},
                    {"name": "description", "type": "text"},
                    {"name": "short_description", "type": "text"},
                    {"name": "version", "type": "text"},
                    {"name": "tags", "type": "jsonb"},
                    {"name": "labels", "type": "jsonb"},
                    {"name": "domain", "type": "text"},
                    {"name": "category", "type": "text"},
                    {"name": "indexed_at", "type": "timestamptz"},
                    {"name": "searchable_content", "type": "text"},
                    {"name": "access_strategies", "type": "jsonb"},
                    {"name": "dublin_core", "type": "jsonb"},
                    {"name": "dc_creator", "type": "jsonb"},
                    {"name": "dc_subject", "type": "jsonb"},
                    {"name": "dc_publisher", "type": "text"},
                    {"name": "dc_format", "type": "text"}
                ]
            }
            
            # Create replication log table schema
            replication_schema = {
                "table_name": "ord_replication_log",
                "columns": [
                    {"name": "id", "type": "text", "primary_key": True},
                    {"name": "table_name", "type": "text"},
                    {"name": "operation", "type": "text"},
                    {"name": "record_id", "type": "text"},
                    {"name": "timestamp", "type": "timestamptz"},
                    {"name": "status", "type": "text"},
                    {"name": "error_message", "type": "text"}
                ]
            }
            
            # Create tables using SQLite schema helper
            await self._create_sqlite_table(registrations_schema)
            await self._create_sqlite_table(index_schema)
            await self._create_sqlite_table(replication_schema)
            
            logger.info("✅ SQLite ORD tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create SQLite tables: {e}")
            # Don't raise - SQLite is fallback
            
    async def _create_sqlite_table(self, schema: Dict[str, Any]):
        """Helper to create SQLite table from schema using SQL DDL"""
        try:
            table_name = schema['table_name']
            columns = schema['columns']
            
            # Build CREATE TABLE SQL statement
            column_defs = []
            primary_keys = []
            
            for col in columns:
                col_def = f"{col['name']} {col['type']}"
                if col.get('primary_key'):
                    primary_keys.append(col['name'])
                column_defs.append(col_def)
            
            # Add primary key constraint if any
            if primary_keys:
                column_defs.append(f"PRIMARY KEY ({', '.join(primary_keys)})")
            
            create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(column_defs)})"
            
            # Execute table creation using SQLite
            try:
                # Execute SQL directly on SQLite
                result = self.sqlite_client.execute_query(create_sql)
                logger.info(f"✅ SQLite table created: {table_name}")
            except Exception as rpc_error:
                # Table creation handled by execute_query
                logger.info(f"🔄 SQLite table {table_name} creation attempted via SQL: {create_sql}")
                # SQLite tables are created automatically
                
        except Exception as e:
            logger.warning(f"SQLite table creation warning for {schema.get('table_name', 'unknown')}: {e}")
            
    async def _verify_connectivity(self):
        """Verify available database connections are working"""
        try:
            # Test HANA connectivity only if not in fallback mode
            if not self.fallback_mode and self.hana_client is not None:
                hana_health = self.hana_client.health_check()
                if hana_health.get("status") != "healthy":
                    logger.warning("HANA connection unhealthy, switching to fallback mode")
                    self.fallback_mode = True
                    self.hana_client = None
                else:
                    logger.info("✅ HANA connectivity verified")
                
            # Test SQLite connectivity
            try:
                sqlite_health = self.sqlite_client.health_check()
                if sqlite_health.get("status") == "healthy":
                    logger.info("✅ SQLite connectivity verified")
                else:
                    logger.warning("SQLite connection issue detected")
            except Exception as sqlite_error:
                logger.warning(f"SQLite connectivity test failed: {sqlite_error}")
                
            if self.fallback_mode:
                logger.info("✅ Database connectivity verified (SQLite-only mode)")
            else:
                logger.info("✅ Database connectivity verified (HANA primary + SQLite fallback)")
            
        except Exception as e:
            logger.error(f"Database connectivity check failed: {e}")
            # Don't raise - continue with available databases
            
    async def store_registration(self, registration: ORDRegistration) -> Dict[str, Any]:
        """Store ORD registration in both databases with replication"""
        try:
            logger.info(f"Attempting to store registration {registration.registration_id}")
            
            # Check if HANA client is available
            if self.hana_client is None:
                logger.warning("HANA client is None, using SQLite-only mode")
                return await self._store_registration_sqlite_only(registration)
            
            # Store in HANA (primary)
            logger.debug("Storing in HANA primary database")
            hana_result = await self._store_registration_hana(registration)
            
            # Replicate to SQLite (fallback)
            if self.replication_enabled:
                logger.debug("Replicating to SQLite")
                sqlite_result = await self._store_registration_sqlite(registration)
                await self._log_replication("ord_registrations", "INSERT", 
                                           registration.registration_id, sqlite_result.get("success", False))
            
            logger.info(f"Successfully stored registration {registration.registration_id}")
            return {
                "success": True,
                "registration_id": registration.registration_id,
                "primary_storage": "hana",
                "replicated": self.replication_enabled
            }
            
        except Exception as e:
            logger.error(f"Failed to store registration: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
            
    async def _store_registration_hana(self, registration: ORDRegistration) -> Dict[str, Any]:
        """Store registration in HANA"""
        try:
            insert_sql = """
            INSERT INTO ord_registrations 
            (registration_id, ord_document, registered_by, registered_at, last_updated, 
             version, status, validation_result, governance_info, analytics_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = [
                registration.registration_id,
                json.dumps(registration.ord_document.dict()),
                registration.metadata.registered_by,
                registration.metadata.registered_at,
                registration.metadata.last_updated,
                registration.metadata.version,
                registration.metadata.status.value,
                json.dumps(registration.validation.dict()),
                json.dumps(registration.governance.dict() if registration.governance else {}),
                json.dumps(registration.analytics.dict() if registration.analytics else {})
            ]
            
            result = self.hana_client.execute_query(insert_sql, params)
            return {"success": True, "hana_result": result}
            
        except Exception as e:
            logger.error(f"HANA storage failed: {e}")
            raise
            
    async def _store_registration_sqlite(self, registration: ORDRegistration) -> Dict[str, Any]:
        """Store registration in SQLite"""
        try:
            if not self.sqlite_client or not hasattr(self.sqlite_client, 'client'):
                logger.error("SQLite client not initialized or invalid")
                return {"success": False, "error": "SQLite client not available"}
            
            registration_data = {
                "registration_id": registration.registration_id,
                "ord_document": registration.ord_document.dict(),
                "registered_by": registration.metadata.registered_by,
                "registered_at": registration.metadata.registered_at.isoformat(),
                "last_updated": registration.metadata.last_updated.isoformat(),
                "version": registration.metadata.version,
                "status": registration.metadata.status.value,
                "validation_result": registration.validation.dict(),
                "governance_info": registration.governance.dict() if registration.governance else {},
                "analytics_info": registration.analytics.dict() if registration.analytics else {}
            }
            
            logger.debug(f"Attempting to store registration {registration.registration_id} in SQLite")
            
            # Use SQLite upsert
            result = self.sqlite_client.client.table("ord_registrations").upsert(registration_data).execute()
            
            logger.info(f"Successfully stored registration {registration.registration_id} in SQLite")
            return {"success": True, "sqlite_result": result.data}
            
        except Exception as e:
            logger.error(f"SQLite storage failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def _store_registration_sqlite_only(self, registration: ORDRegistration) -> Dict[str, Any]:
        """Store registration in SQLite only when HANA is unavailable"""
        try:
            sqlite_result = await self._store_registration_sqlite(registration)
            if sqlite_result.get("success"):
                return {
                    "success": True,
                    "registration_id": registration.registration_id,
                    "primary_storage": "sqlite",
                    "replicated": False
                }
            else:
                return sqlite_result
        except Exception as e:
            logger.error(f"SQLite-only storage failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def update_registration(self, registration: ORDRegistration) -> bool:
        """Update existing registration in dual-database with replication"""
        try:
            # Update in HANA (primary)
            hana_result = await self._update_registration_hana(registration)
            
            # Replicate to SQLite (fallback) if replication enabled
            if self.replication_enabled:
                logger.debug("Replicating update to SQLite")
                sqlite_result = await self._update_registration_sqlite(registration)
                await self._log_replication("ord_registrations", "UPDATE", 
                                           registration.registration_id, sqlite_result.get("success", False))
            
            logger.info(f"Successfully updated registration {registration.registration_id}")
            return hana_result.get("success", False)
            
        except Exception as e:
            logger.error(f"Failed to update registration: {e}")
            return False
            
    async def _update_registration_hana(self, registration: ORDRegistration) -> Dict[str, Any]:
        """Update registration in HANA"""
        try:
            update_sql = """
            UPDATE ord_registrations 
            SET ord_document = ?, registered_by = ?, registered_at = ?, last_updated = ?, 
                version = ?, status = ?, validation_result = ?, governance_info = ?, analytics_info = ?
            WHERE registration_id = ?
            """
            
            params = [
                json.dumps(registration.ord_document.dict()),
                registration.metadata.registered_by,
                registration.metadata.registered_at,
                registration.metadata.last_updated,
                registration.metadata.version,
                registration.metadata.status.value,
                json.dumps(registration.validation.dict()),
                json.dumps(registration.governance.dict() if registration.governance else {}),
                json.dumps(registration.analytics.dict() if registration.analytics else {}),
                registration.registration_id
            ]
            
            result = self.hana_client.execute_query(update_sql, params)
            return {"success": True, "hana_result": result}
            
        except Exception as e:
            logger.error(f"HANA update failed: {e}")
            raise
            
    async def _update_registration_sqlite(self, registration: ORDRegistration) -> Dict[str, Any]:
        """Update registration in SQLite"""
        try:
            if not self.sqlite_client or not hasattr(self.sqlite_client, 'client'):
                logger.error("SQLite client not initialized or invalid")
                return {"success": False, "error": "SQLite client not available"}
            
            registration_data = {
                "ord_document": registration.ord_document.dict(),
                "registered_by": registration.metadata.registered_by,
                "registered_at": registration.metadata.registered_at.isoformat(),
                "last_updated": registration.metadata.last_updated.isoformat(),
                "version": registration.metadata.version,
                "status": registration.metadata.status.value,
                "validation_result": registration.validation.dict(),
                "governance_info": registration.governance.dict() if registration.governance else {},
                "analytics_info": registration.analytics.dict() if registration.analytics else {}
            }
            
            logger.debug(f"Attempting to update registration {registration.registration_id} in SQLite")
            
            # Use SQLite update
            result = self.sqlite_client.client.table("ord_registrations").update(registration_data).eq("registration_id", registration.registration_id).execute()
            
            logger.info(f"Successfully updated registration {registration.registration_id} in SQLite")
            return {"success": True, "sqlite_result": result.data}
            
        except Exception as e:
            logger.error(f"SQLite update failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def _log_replication(self, table_name: str, operation: str, record_id: str, success: bool):
        """Log replication status"""
        try:
            log_entry = {
                "id": f"repl_{uuid4().hex[:8]}",
                "table_name": table_name,
                "operation": operation,
                "record_id": record_id,
                "timestamp": datetime.utcnow(),
                "status": "success" if success else "failed",
                "error_message": None if success else "Replication failed"
            }
            
            # Log to HANA
            log_sql = """
            INSERT INTO ord_replication_log (id, table_name, operation, record_id, timestamp, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            self.hana_client.execute_query(log_sql, [
                log_entry["id"], log_entry["table_name"], log_entry["operation"],
                log_entry["record_id"], log_entry["timestamp"], log_entry["status"],
                log_entry["error_message"]
            ])
            
        except Exception as e:
            logger.error(f"Failed to log replication: {e}")
            
    async def get_registration(self, registration_id: str) -> Optional[ORDRegistration]:
        """Get registration from primary database (HANA) with fallback to SQLite"""
        try:
            # Try HANA first (primary)
            result = await self._get_registration_hana(registration_id)
            if result:
                return result
                
            # Fallback to SQLite if HANA fails
            if not self.fallback_mode:
                logger.warning(f"HANA lookup failed for {registration_id}, trying SQLite fallback")
                result = await self._get_registration_sqlite(registration_id)
                return result
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to get registration {registration_id}: {e}")
            return None
            
    async def _get_registration_hana(self, registration_id: str) -> Optional[ORDRegistration]:
        """Get registration from HANA"""
        try:
            query_sql = """
            SELECT registration_id, ord_document, registered_by, registered_at, last_updated,
                   version, status, validation_result, governance_info, analytics_info
            FROM ord_registrations WHERE registration_id = ?
            """
            
            result = self.hana_client.execute_query(query_sql, [registration_id])
            
            if result.data and len(result.data) > 0:
                row = result.data[0]
                return self._convert_row_to_registration(row)
                
            return None
            
        except Exception as e:
            logger.error(f"HANA lookup failed: {e}")
            raise
            
    async def _get_registration_sqlite(self, registration_id: str) -> Optional[ORDRegistration]:
        """Get registration from SQLite"""
        try:
            result = self.sqlite_client.client.table("ord_registrations").select("*").eq("registration_id", registration_id).execute()
            
            if result.data and len(result.data) > 0:
                row = result.data[0]
                return self._convert_sqlite_to_registration(row)
                
            return None
            
        except Exception as e:
            logger.error(f"SQLite lookup failed: {e}")
            return None
            
    def _convert_row_to_registration(self, row: dict) -> ORDRegistration:
        """Convert HANA dictionary row to ORDRegistration object"""
        try:
            return ORDRegistration(
                registration_id=row['REGISTRATION_ID'],
                ord_document=ORDDocument(**json.loads(row['ORD_DOCUMENT'])),
                metadata=RegistrationMetadata(
                    registered_by=row['REGISTERED_BY'],
                    registered_at=row['REGISTERED_AT'],
                    last_updated=row['LAST_UPDATED'],
                    version=row['VERSION'],
                    status=RegistrationStatus(row['STATUS'])
                ),
                validation=ValidationResult(**json.loads(row['VALIDATION_RESULT'])),
                governance=json.loads(row['GOVERNANCE_INFO']) if row['GOVERNANCE_INFO'] else {},
                analytics=json.loads(row['ANALYTICS_INFO']) if row['ANALYTICS_INFO'] else {}
            )
        except Exception as e:
            logger.error(f"Failed to convert HANA row to registration: {e}")
            raise
            
    def _convert_sqlite_to_registration(self, row: Dict) -> ORDRegistration:
        """Convert SQLite row to ORDRegistration object"""
        try:
            return ORDRegistration(
                registration_id=row["registration_id"],
                ord_document=ORDDocument(**row["ord_document"]),
                metadata=RegistrationMetadata(
                    registered_by=row["registered_by"],
                    registered_at=datetime.fromisoformat(row["registered_at"]),
                    last_updated=datetime.fromisoformat(row["last_updated"]),
                    version=row["version"],
                    status=RegistrationStatus(row["status"])
                ),
                validation=ValidationResult(**row["validation_result"]),
                governance=row["governance_info"] or {},
                analytics=row["analytics_info"] or {}
            )
        except Exception as e:
            logger.error(f"Failed to convert SQLite row to registration: {e}")
            raise

    async def search_registrations(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[ResourceIndexEntry]:
        """Search ORD registrations with advanced filtering and Dublin Core support"""
        try:
            # Try HANA first (primary)
            if not self.fallback_mode and self.hana_client is not None:
                results = await self._search_registrations_hana(query, filters)
                if results:
                    logger.info(f"Found {len(results)} results from HANA search")
                    return results
            
            # Fallback or primary SQLite search
            results = await self._search_registrations_sqlite(query, filters)
            logger.info(f"Found {len(results)} results from SQLite search")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def _search_registrations_hana(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[ResourceIndexEntry]:
        """Search registrations in HANA using full-text search capabilities"""
        try:
            # Build dynamic search query with Dublin Core fields
            search_sql = """
            SELECT ord_id, registration_id, resource_type, title, description, short_description,
                   version, tags, labels, domain, category, indexed_at, searchable_content,
                   dublin_core, dc_creator, dc_subject, dc_publisher, dc_format, access_strategies
            FROM ord_resource_index
            WHERE 1=1
            """
            
            params = []
            
            # Add text search conditions
            if query and query.strip():
                search_sql += """
                AND (
                    LOWER(title) LIKE LOWER(?) OR
                    LOWER(description) LIKE LOWER(?) OR 
                    LOWER(short_description) LIKE LOWER(?) OR
                    LOWER(searchable_content) LIKE LOWER(?) OR
                    JSON_VALUE(dc_creator, '$[0].name') LIKE LOWER(?) OR
                    JSON_VALUE(dc_subject, '$[0]') LIKE LOWER(?)
                )
                """
                search_term = f"%{query}%"
                params.extend([search_term] * 6)
            
            # Add filters
            if filters:
                if filters.get("resource_type"):
                    search_sql += " AND resource_type = ?"
                    params.append(filters["resource_type"])
                
                if filters.get("domain"):
                    search_sql += " AND domain = ?"
                    params.append(filters["domain"])
                
                if filters.get("category"):
                    search_sql += " AND category = ?"
                    params.append(filters["category"])
                
                if filters.get("dc_publisher"):
                    search_sql += " AND dc_publisher = ?"
                    params.append(filters["dc_publisher"])
            
            # Order by relevance and date
            search_sql += " ORDER BY indexed_at DESC"
            
            result = self.hana_client.execute_query(search_sql, params)
            
            if result.data:
                return [self._convert_hana_row_to_index_entry(row) for row in result.data]
            
            return []
            
        except Exception as e:
            logger.error(f"HANA search failed: {e}")
            raise

    async def _search_registrations_sqlite(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[ResourceIndexEntry]:
        """Search registrations in SQLite using SQLite full-text search"""
        try:
            # Build SQLite query
            sqlite_query = self.sqlite_client.client.table("ord_resource_index").select("*")
            
            # Add text search conditions
            if query and query.strip():
                # Use SQLite's ilike for case-insensitive search
                search_condition = f"title.ilike.%{query}%,description.ilike.%{query}%,short_description.ilike.%{query}%"
                sqlite_query = sqlite_query.or_(search_condition)
            
            # Add filters
            if filters:
                if filters.get("resource_type"):
                    sqlite_query = sqlite_query.eq("resource_type", filters["resource_type"])
                
                if filters.get("domain"):
                    sqlite_query = sqlite_query.eq("domain", filters["domain"])
                
                if filters.get("category"):
                    sqlite_query = sqlite_query.eq("category", filters["category"])
                
                if filters.get("dc_publisher"):
                    sqlite_query = sqlite_query.eq("dc_publisher", filters["dc_publisher"])
            
            # Order and execute
            result = sqlite_query.order("indexed_at", desc=True).execute()
            
            if result.data:
                return [self._convert_sqlite_row_to_index_entry(row) for row in result.data]
            
            return []
            
        except Exception as e:
            logger.error(f"SQLite search failed: {e}")
            return []

    def _convert_hana_row_to_index_entry(self, row: Dict) -> ResourceIndexEntry:
        """Convert HANA row to ResourceIndexEntry"""
        try:
            # Fix enum casing mismatch: DB has 'DataProduct', enum expects 'dataProduct'
            raw_resource_type = row["RESOURCE_TYPE"]
            if raw_resource_type == "DataProduct":
                resource_type_value = "dataProduct"
            else:
                resource_type_value = raw_resource_type
            
            # Debug logging
            access_strategies_raw = row.get("ACCESS_STRATEGIES")
            logger.debug(f"HANA row ACCESS_STRATEGIES: type={type(access_strategies_raw)}, value={access_strategies_raw}")
            access_strategies = json.loads(access_strategies_raw) if access_strategies_raw else []
            logger.debug(f"Parsed access_strategies: {access_strategies}")
            
            return ResourceIndexEntry(
                ord_id=row["ORD_ID"],
                registration_id=row["REGISTRATION_ID"],
                resource_type=resource_type_value,
                title=row["TITLE"],
                description=row["DESCRIPTION"],
                short_description=row["SHORT_DESCRIPTION"],
                version=row["VERSION"],
                tags=json.loads(row["TAGS"]) if row["TAGS"] else [],
                labels=json.loads(row["LABELS"]) if row["LABELS"] else {},
                domain=row["DOMAIN"],
                category=row["CATEGORY"],
                indexed_at=row["INDEXED_AT"],
                searchable_content=row["SEARCHABLE_CONTENT"],
                access_strategies=access_strategies,
                dublin_core=json.loads(row["DUBLIN_CORE"]) if row["DUBLIN_CORE"] else {},
                dc_creator=json.loads(row["DC_CREATOR"]) if row["DC_CREATOR"] else [],
                dc_subject=json.loads(row["DC_SUBJECT"]) if row["DC_SUBJECT"] else [],
                dc_publisher=row["DC_PUBLISHER"],
                dc_format=row["DC_FORMAT"]
            )
        except Exception as e:
            logger.error(f"Failed to convert HANA row to index entry: {e}")
            raise

    def _convert_sqlite_row_to_index_entry(self, row: Dict) -> ResourceIndexEntry:
        """Convert SQLite row to ResourceIndexEntry"""
        try:
            # Debug logging
            access_strategies = row.get("access_strategies", [])
            logger.debug(f"Converting SQLite row: ord_id={row.get('ord_id')}, access_strategies type={type(access_strategies)}, value={access_strategies}")
            
            return ResourceIndexEntry(
                ord_id=row["ord_id"],
                registration_id=row["registration_id"],
                resource_type=row["resource_type"],
                title=row["title"],
                description=row["description"],
                short_description=row["short_description"],
                version=row["version"],
                tags=row["tags"] if row["tags"] else [],
                labels=row["labels"] if row["labels"] else {},
                domain=row["domain"],
                category=row["category"],
                indexed_at=datetime.fromisoformat(row["indexed_at"]),
                searchable_content=row["searchable_content"],
                access_strategies=access_strategies,
                dublin_core=row["dublin_core"] if row["dublin_core"] else {},
                dc_creator=row["dc_creator"] if row["dc_creator"] else [],
                dc_subject=row["dc_subject"] if row["dc_subject"] else [],
                dc_publisher=row["dc_publisher"],
                dc_format=row["dc_format"]
            )
        except Exception as e:
            logger.error(f"Failed to convert SQLite row to index entry: {e}")
            raise

    async def index_registration(self, registration: ORDRegistration):
        """Index registration for search (both HANA and SQLite)"""
        try:
            # Extract searchable content from ORD document
            ord_doc = registration.ord_document
            
            # Build searchable content from Dublin Core and ORD fields
            searchable_parts = []
            
            # Add Dublin Core fields
            if hasattr(ord_doc, 'dublin_core') and ord_doc.dublin_core:
                dc = ord_doc.dublin_core
                if dc.get('title'):
                    searchable_parts.append(str(dc['title']))
                if dc.get('description'):
                    searchable_parts.append(str(dc['description']))
                if dc.get('subject'):
                    searchable_parts.extend([str(s) for s in dc['subject']])
                if dc.get('creator'):
                    searchable_parts.extend([str(c.get('name', c)) for c in dc['creator']])
            
            # Add ORD document fields
            if hasattr(ord_doc, 'title'):
                searchable_parts.append(ord_doc.title)
            if hasattr(ord_doc, 'shortDescription'):
                searchable_parts.append(ord_doc.shortDescription)
            if hasattr(ord_doc, 'description'):
                searchable_parts.append(ord_doc.description)
            
            searchable_content = " ".join(filter(None, searchable_parts))
            
            # Create index entry for each data product in the ORD document
            if hasattr(ord_doc, 'dataProducts') and ord_doc.dataProducts:
                logger.info(f"Indexing {len(ord_doc.dataProducts)} data products")
                for dp in ord_doc.dataProducts:
                    # Debug logging
                    logger.info(f"Data product keys: {list(dp.keys())}")
                    access_strategies = dp.get('accessStrategies', [])
                    logger.info(f"Processing data product: {dp.get('ordId')}, accessStrategies: {access_strategies}")
                    
                    index_entry = {
                        "ord_id": dp.get('ordId', f"{registration.registration_id}_{dp.get('title', 'unknown')}"),
                        "registration_id": registration.registration_id,
                        "resource_type": dp.get('type', 'dataProduct'),
                        "title": dp.get('title', ''),
                        "description": dp.get('description', ''),
                        "short_description": dp.get('shortDescription', ''),
                        "version": dp.get('version', ''),
                        "tags": json.dumps(dp.get('tags', [])),
                        "labels": json.dumps(dp.get('labels', {})),
                        "domain": dp.get('partOfPackage', '').split(':')[0] if dp.get('partOfPackage') else '',
                        "category": dp.get('category', ''),
                        "indexed_at": datetime.utcnow(),
                        "searchable_content": searchable_content,
                        "access_strategies": json.dumps(access_strategies),
                        "dublin_core": json.dumps(ord_doc.dublin_core if hasattr(ord_doc, 'dublin_core') else {}),
                        "dc_creator": json.dumps(ord_doc.dublin_core.get('creator', []) if hasattr(ord_doc, 'dublin_core') and ord_doc.dublin_core else []),
                        "dc_subject": json.dumps(ord_doc.dublin_core.get('subject', []) if hasattr(ord_doc, 'dublin_core') and ord_doc.dublin_core else []),
                        "dc_publisher": ord_doc.dublin_core.get('publisher', '') if hasattr(ord_doc, 'dublin_core') and ord_doc.dublin_core else '',
                        "dc_format": ord_doc.dublin_core.get('format', '') if hasattr(ord_doc, 'dublin_core') and ord_doc.dublin_core else ''
                    }
                    
                    # Store in HANA
                    if not self.fallback_mode and self.hana_client is not None:
                        await self._index_entry_hana(index_entry)
                    
                    # Store in SQLite
                    await self._index_entry_sqlite(index_entry)
            
            logger.info(f"Successfully indexed registration {registration.registration_id}")
            
        except Exception as e:
            logger.error(f"Failed to index registration {registration.registration_id}: {e}")
            raise

    async def _index_entry_hana(self, entry: Dict[str, Any]):
        """Store index entry in HANA with UPSERT logic"""
        try:
            # First try to update existing entry
            update_sql = """
            UPDATE ord_resource_index 
            SET registration_id = ?, resource_type = ?, title = ?, description = ?, 
                short_description = ?, version = ?, tags = ?, labels = ?, domain = ?, 
                category = ?, indexed_at = ?, searchable_content = ?, access_strategies = ?,
                dublin_core = ?, dc_creator = ?, dc_subject = ?, dc_publisher = ?, dc_format = ?
            WHERE ord_id = ?
            """
            
            update_params = [
                entry["registration_id"], entry["resource_type"],
                entry["title"], entry["description"], entry["short_description"],
                entry["version"], entry["tags"], entry["labels"],
                entry["domain"], entry["category"], entry["indexed_at"],
                entry["searchable_content"], entry["access_strategies"],
                entry["dublin_core"], entry["dc_creator"], entry["dc_subject"],
                entry["dc_publisher"], entry["dc_format"],
                entry["ord_id"]  # WHERE clause
            ]
            
            result = self.hana_client.execute_query(update_sql, update_params)
            
            # Check if update affected any rows
            if not result or (hasattr(result, 'rowcount') and result.rowcount == 0):
                # No rows updated, so insert new entry
                insert_sql = """
                INSERT INTO ord_resource_index 
                (ord_id, registration_id, resource_type, title, description, short_description,
                 version, tags, labels, domain, category, indexed_at, searchable_content,
                 access_strategies, dublin_core, dc_creator, dc_subject, dc_publisher, dc_format)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                insert_params = [
                    entry["ord_id"], entry["registration_id"], entry["resource_type"],
                    entry["title"], entry["description"], entry["short_description"],
                    entry["version"], entry["tags"], entry["labels"],
                    entry["domain"], entry["category"], entry["indexed_at"],
                    entry["searchable_content"], entry["access_strategies"],
                    entry["dublin_core"], entry["dc_creator"], entry["dc_subject"],
                    entry["dc_publisher"], entry["dc_format"]
                ]
                
                self.hana_client.execute_query(insert_sql, insert_params)
                logger.debug(f"Inserted new index entry for ORD ID: {entry['ord_id']}")
            else:
                logger.debug(f"Updated existing index entry for ORD ID: {entry['ord_id']}")
            
        except Exception as e:
            logger.error(f"Failed to index in HANA: {e}")
            raise

    async def _index_entry_sqlite(self, entry: Dict[str, Any]):
        """Store index entry in SQLite"""
        try:
            # Convert datetime to ISO string for SQLite
            sqlite_entry = entry.copy()
            sqlite_entry["indexed_at"] = entry["indexed_at"].isoformat()
            
            # Parse JSON strings back to objects for SQLite JSONB columns
            sqlite_entry["tags"] = json.loads(entry["tags"])
            sqlite_entry["labels"] = json.loads(entry["labels"])
            sqlite_entry["access_strategies"] = json.loads(entry["access_strategies"])
            sqlite_entry["dublin_core"] = json.loads(entry["dublin_core"])
            sqlite_entry["dc_creator"] = json.loads(entry["dc_creator"])
            sqlite_entry["dc_subject"] = json.loads(entry["dc_subject"])
            
            result = self.sqlite_client.client.table("ord_resource_index").upsert(sqlite_entry).execute()
            
        except Exception as e:
            logger.error(f"Failed to index in SQLite: {e}")
            # Don't raise - SQLite is fallback

    async def list_all_registrations(self, limit: int = 100) -> List[ResourceIndexEntry]:
        """List all registrations with pagination"""
        try:
            # Try HANA first
            if not self.fallback_mode and self.hana_client is not None:
                results = await self._list_registrations_hana(limit)
                if results:
                    return results
            
            # Fallback to SQLite
            return await self._list_registrations_sqlite(limit)
            
        except Exception as e:
            logger.error(f"Failed to list registrations: {e}")
            return []

    async def _list_registrations_hana(self, limit: int) -> List[ResourceIndexEntry]:
        """List registrations from HANA"""
        try:
            list_sql = """
            SELECT ord_id, registration_id, resource_type, title, description, short_description,
                   version, tags, labels, domain, category, indexed_at, searchable_content,
                   dublin_core, dc_creator, dc_subject, dc_publisher, dc_format
            FROM ord_resource_index
            ORDER BY indexed_at DESC
            LIMIT ?
            """
            
            result = self.hana_client.execute_query(list_sql, [limit])
            
            if result.data:
                return [self._convert_hana_row_to_index_entry(row) for row in result.data]
            
            return []
            
        except Exception as e:
            logger.error(f"HANA list failed: {e}")
            raise

    async def _list_registrations_sqlite(self, limit: int) -> List[ResourceIndexEntry]:
        """List registrations from SQLite"""
        try:
            result = (self.sqlite_client.client.table("ord_resource_index")
                     .select("*")
                     .order("indexed_at", desc=True)
                     .limit(limit)
                     .execute())
            
            if result.data:
                return [self._convert_sqlite_row_to_index_entry(row) for row in result.data]
            
            return []
            
        except Exception as e:
            logger.error(f"SQLite list failed: {e}")
            return []

    async def get_registration_count(self, active_only: bool = True) -> int:
        """Get count of registrations"""
        try:
            # Try HANA first
            if not self.fallback_mode and self.hana_client is not None:
                count = await self._get_registration_count_hana(active_only)
                if count >= 0:
                    return count
            
            # Fallback to SQLite
            return await self._get_registration_count_sqlite(active_only)
            
        except Exception as e:
            logger.error(f"Failed to get registration count: {e}")
            return 0

    async def _get_registration_count_hana(self, active_only: bool) -> int:
        """Get registration count from HANA"""
        try:
            if active_only:
                count_sql = "SELECT COUNT(*) as count FROM ord_registrations WHERE status = 'ACTIVE'"
            else:
                count_sql = "SELECT COUNT(*) as count FROM ord_registrations"
            
            result = self.hana_client.execute_query(count_sql)
            
            if result.data and len(result.data) > 0:
                return result.data[0]['COUNT']
            return 0
            
        except Exception as e:
            logger.error(f"HANA count failed: {e}")
            raise

    async def _get_registration_count_sqlite(self, active_only: bool) -> int:
        """Get registration count from SQLite"""
        try:
            query = self.sqlite_client.client.table("ord_registrations").select("registration_id", count="exact")
            
            if active_only:
                query = query.eq("status", "active")
            
            result = query.execute()
            return result.count if result.count is not None else 0
            
        except Exception as e:
            logger.error(f"SQLite count failed: {e}")
            return 0
    
    async def get_resource_by_ord_id(self, ord_id: str) -> Optional[Dict[str, Any]]:
        """Get a resource by its ORD ID from dual-database storage"""
        try:
            # Try HANA first (primary)
            if not self.fallback_mode and self.hana_client:
                try:
                    result = await self._get_resource_by_ord_id_hana(ord_id)
                    if result:
                        await self._log_replication("ord_resource_index", "read", ord_id, True)
                        return result
                except Exception as e:
                    logger.error(f"HANA resource lookup failed for {ord_id}: {e}")
                    await self._log_replication("ord_resource_index", "read", ord_id, False)
            
            # Fallback to SQLite
            if self.sqlite_client:
                try:
                    result = await self._get_resource_by_ord_id_sqlite(ord_id)
                    await self._log_replication("ord_resource_index", "read_fallback", ord_id, result is not None)
                    return result
                except Exception as e:
                    logger.error(f"SQLite resource lookup failed for {ord_id}: {e}")
                    await self._log_replication("ord_resource_index", "read_fallback", ord_id, False)
            
            return None
            
        except Exception as e:
            logger.error(f"Resource lookup failed for {ord_id}: {e}")
            return None
    
    async def _get_resource_by_ord_id_hana(self, ord_id: str) -> Optional[Dict[str, Any]]:
        """Get resource by ORD ID from HANA"""
        try:
            query = """SELECT * FROM ord_resource_index WHERE ord_id = ?"""
            result = self.hana_client.execute_query(query, [ord_id])
            
            if result and result.rows:
                row = result.rows[0] if hasattr(result.rows[0], '__dict__') else dict(zip(result.columns, result.rows[0]))
                return {
                    "ord_id": row.get("ord_id"),
                    "title": row.get("title"),
                    "description": row.get("description"),
                    "resource_type": row.get("resource_type"),
                    "registration_id": row.get("registration_id"),
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at")
                }
            
            return None
            
        except Exception as e:
            logger.error(f"HANA resource lookup error for {ord_id}: {e}")
            return None
    
    async def _get_resource_by_ord_id_sqlite(self, ord_id: str) -> Optional[Dict[str, Any]]:
        """Get resource by ORD ID from SQLite"""
        try:
            query = self.sqlite_client.client.table("ord_resource_index").select("*").eq("ord_id", ord_id)
            result = query.execute()
            
            if result.data:
                data = result.data[0]
                return {
                    "ord_id": data.get("ord_id"),
                    "title": data.get("title"),
                    "description": data.get("description"),
                    "resource_type": data.get("resource_type"),
                    "registration_id": data.get("registration_id"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at")
                }
            
            return None
            
        except Exception as e:
            logger.error(f"SQLite resource lookup error for {ord_id}: {e}")
            return None
    
    async def search_resources(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """Search resources with advanced filtering and pagination"""
        try:
            query = search_params.get("query", "")
            page = search_params.get("page", 1)
            page_size = search_params.get("page_size", 10)
            
            # Use existing search_registrations method
            results = await self.search_registrations(query, search_params)
            
            return {
                "results": results,
                "total_count": len(results),
                "page": page,
                "page_size": page_size
            }
            
        except Exception as e:
            logger.error(f"Resource search failed: {e}")
            return {
                "results": [],
                "total_count": 0,
                "page": 1,
                "page_size": 10
            }
    
    async def delete_registration(self, registration_id: str) -> bool:
        """Delete a registration from both databases (hard delete)"""
        try:
            success = True
            
            # Delete from HANA (primary)
            if not self.fallback_mode and self.hana_client:
                try:
                    self.hana_client.execute_query(
                        "DELETE FROM ord_registrations WHERE registration_id = ?",
                        [registration_id]
                    )
                    self.hana_client.execute_query(
                        "DELETE FROM ord_resource_index WHERE registration_id = ?",
                        [registration_id]
                    )
                    logger.info(f"HANA delete successful for {registration_id}")
                except Exception as e:
                    logger.error(f"HANA delete failed for {registration_id}: {e}")
                    success = False
            
            # Delete from SQLite (fallback)
            if self.sqlite_client:
                try:
                    self.sqlite_client.client.table("ord_registrations").delete().eq("registration_id", registration_id).execute()
                    self.sqlite_client.client.table("ord_resource_index").delete().eq("registration_id", registration_id).execute()
                    logger.info(f"SQLite delete successful for {registration_id}")
                except Exception as e:
                    logger.error(f"SQLite delete failed for {registration_id}: {e}")
                    success = False
            
            await self._log_replication("ord_registrations", "delete", registration_id, success)
            return success
            
        except Exception as e:
            logger.error(f"Registration deletion failed for {registration_id}: {e}")
            return False


# Global storage instance
_ord_storage = None

async def get_ord_storage() -> ORDDualDatabaseStorage:
    """Get or create the ORD dual-database storage instance"""
    global _ord_storage
    if _ord_storage is None:
        _ord_storage = ORDDualDatabaseStorage()
        await _ord_storage.initialize()
    return _ord_storage
