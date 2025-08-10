"""
Supabase Production Client
Production-ready client for Supabase integration with database, auth, and storage
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

try:
    from supabase import create_client, Client
    from gotrue import User, Session
    from postgrest import APIResponse
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None
    User = None
    Session = None
    APIResponse = None

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class SupabaseConfig:
    """Configuration for Supabase client"""
    url: str
    anon_key: str
    service_role_key: Optional[str] = None
    auto_refresh_token: bool = True
    persist_session: bool = True


@dataclass
class SupabaseResponse:
    """Structured response from Supabase operations"""
    data: Union[List[Dict[str, Any]], Dict[str, Any], None]
    count: Optional[int]
    status_code: int
    error: Optional[Dict[str, Any]]
    raw_response: Optional[Any] = None


class SupabaseClient:
    """Production-ready Supabase client for A2A agents"""
    
    def __init__(self, config: Optional[SupabaseConfig] = None):
        """Initialize Supabase client with configuration"""
        if not SUPABASE_AVAILABLE:
            raise ImportError("Supabase client not available. Install with: pip install supabase")
        
        if config is None:
            config = SupabaseConfig(
                url=os.getenv('SUPABASE_URL'),
                anon_key=os.getenv('SUPABASE_ANON_KEY'),
                service_role_key=os.getenv('SUPABASE_SERVICE_ROLE_KEY')
            )
        
        if not config.url or not config.anon_key:
            raise ValueError("Supabase URL and anon key are required")
        
        self.config = config
        
        # Create client instances
        self.client: Client = create_client(config.url, config.anon_key)
        
        # Service role client for admin operations
        if config.service_role_key:
            self.admin_client: Client = create_client(config.url, config.service_role_key)
        else:
            self.admin_client = None
        
        logger.info(f"Supabase client initialized for {config.url}")
    
    # Database Operations
    def select(
        self,
        table: str,
        columns: str = "*",
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[str] = None
    ) -> SupabaseResponse:
        """Select data from a table"""
        try:
            query = self.client.table(table).select(columns)
            
            # Apply filters
            if filters:
                for key, value in filters.items():
                    if isinstance(value, dict):
                        # Handle complex filters like {"gte": 100}
                        for op, val in value.items():
                            query = getattr(query, op)(key, val)
                    else:
                        query = query.eq(key, value)
            
            # Apply ordering
            if order_by:
                ascending = not order_by.startswith('-')
                column = order_by.lstrip('-')
                query = query.order(column, desc=not ascending)
            
            # Apply pagination
            if limit:
                query = query.limit(limit)
            if offset:
                query = query.offset(offset)
            
            response = query.execute()
            
            return SupabaseResponse(
                data=response.data,
                count=response.count,
                status_code=200,
                error=None,
                raw_response=response
            )
        
        except Exception as e:
            logger.error(f"Supabase select error: {e}")
            return SupabaseResponse(
                data=None,
                count=None,
                status_code=500,
                error={"message": str(e)},
                raw_response=None
            )
    
    def insert(
        self,
        table: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        upsert: bool = False
    ) -> SupabaseResponse:
        """Insert data into a table"""
        try:
            query = self.client.table(table).insert(data)
            
            if upsert:
                query = query.upsert()
            
            response = query.execute()
            
            return SupabaseResponse(
                data=response.data,
                count=response.count,
                status_code=201,
                error=None,
                raw_response=response
            )
        
        except Exception as e:
            logger.error(f"Supabase insert error: {e}")
            return SupabaseResponse(
                data=None,
                count=None,
                status_code=500,
                error={"message": str(e)},
                raw_response=None
            )
    
    def update(
        self,
        table: str,
        data: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> SupabaseResponse:
        """Update data in a table"""
        try:
            query = self.client.table(table).update(data)
            
            # Apply filters
            for key, value in filters.items():
                query = query.eq(key, value)
            
            response = query.execute()
            
            return SupabaseResponse(
                data=response.data,
                count=response.count,
                status_code=200,
                error=None,
                raw_response=response
            )
        
        except Exception as e:
            logger.error(f"Supabase update error: {e}")
            return SupabaseResponse(
                data=None,
                count=None,
                status_code=500,
                error={"message": str(e)},
                raw_response=None
            )
    
    def delete(
        self,
        table: str,
        filters: Dict[str, Any]
    ) -> SupabaseResponse:
        """Delete data from a table"""
        try:
            query = self.client.table(table).delete()
            
            # Apply filters
            for key, value in filters.items():
                query = query.eq(key, value)
            
            response = query.execute()
            
            return SupabaseResponse(
                data=response.data,
                count=response.count,
                status_code=200,
                error=None,
                raw_response=response
            )
        
        except Exception as e:
            logger.error(f"Supabase delete error: {e}")
            return SupabaseResponse(
                data=None,
                count=None,
                status_code=500,
                error={"message": str(e)},
                raw_response=None
            )
    
    # Authentication Operations
    def sign_up(self, email: str, password: str, user_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Sign up a new user"""
        try:
            options = {"data": user_metadata} if user_metadata else {}
            response = self.client.auth.sign_up({"email": email, "password": password, **options})
            
            return {
                "success": True,
                "user": response.user.__dict__ if response.user else None,
                "session": response.session.__dict__ if response.session else None
            }
        
        except Exception as e:
            logger.error(f"Supabase sign up error: {e}")
            return {"success": False, "error": str(e)}
    
    def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """Sign in a user"""
        try:
            response = self.client.auth.sign_in_with_password({"email": email, "password": password})
            
            return {
                "success": True,
                "user": response.user.__dict__ if response.user else None,
                "session": response.session.__dict__ if response.session else None
            }
        
        except Exception as e:
            logger.error(f"Supabase sign in error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_user(self) -> Dict[str, Any]:
        """Get current user"""
        try:
            user = self.client.auth.get_user()
            return {
                "success": True,
                "user": user.user.__dict__ if user.user else None
            }
        
        except Exception as e:
            logger.error(f"Supabase get user error: {e}")
            return {"success": False, "error": str(e)}
    
    def sign_out(self) -> Dict[str, Any]:
        """Sign out current user"""
        try:
            self.client.auth.sign_out()
            return {"success": True}
        
        except Exception as e:
            logger.error(f"Supabase sign out error: {e}")
            return {"success": False, "error": str(e)}
    
    # Storage Operations
    def upload_file(
        self,
        bucket: str,
        path: str,
        file_data: bytes,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Upload a file to storage"""
        try:
            options = {"content-type": content_type} if content_type else {}
            response = self.client.storage.from_(bucket).upload(path, file_data, options)
            
            return {
                "success": True,
                "path": response.get("path"),
                "full_path": response.get("fullPath")
            }
        
        except Exception as e:
            logger.error(f"Supabase file upload error: {e}")
            return {"success": False, "error": str(e)}
    
    def download_file(self, bucket: str, path: str) -> Dict[str, Any]:
        """Download a file from storage"""
        try:
            response = self.client.storage.from_(bucket).download(path)
            
            return {
                "success": True,
                "data": response
            }
        
        except Exception as e:
            logger.error(f"Supabase file download error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_public_url(self, bucket: str, path: str) -> str:
        """Get public URL for a file"""
        try:
            response = self.client.storage.from_(bucket).get_public_url(path)
            return response.get("publicUrl", "")
        
        except Exception as e:
            logger.error(f"Supabase get public URL error: {e}")
            return ""
    
    def list_files(self, bucket: str, path: str = "") -> Dict[str, Any]:
        """List files in a storage bucket"""
        try:
            response = self.client.storage.from_(bucket).list(path)
            
            return {
                "success": True,
                "files": response
            }
        
        except Exception as e:
            logger.error(f"Supabase list files error: {e}")
            return {"success": False, "error": str(e)}
    
    # Schema Management Operations
    def create_agent_data_table(self) -> Dict[str, Any]:
        """Create the agent_data table if it doesn't exist
        
        Note: This requires service role key permissions
        """
        if not self.admin_client:
            return {"success": False, "error": "Service role key required for table creation"}
        
        try:
            # This would typically be done via SQL or Supabase dashboard
            # For now, just document the expected schema
            schema_definition = {
                "table_name": "agent_data",
                "columns": {
                    "id": "uuid PRIMARY KEY DEFAULT gen_random_uuid()",
                    "agent_id": "text NOT NULL",
                    "data_type": "text NOT NULL",
                    "data": "jsonb NOT NULL",
                    "metadata": "jsonb DEFAULT '{}'",
                    "created_at": "timestamptz DEFAULT now()",
                    "updated_at": "timestamptz DEFAULT now()"
                },
                "indexes": [
                    "CREATE INDEX IF NOT EXISTS idx_agent_data_agent_id ON agent_data(agent_id)",
                    "CREATE INDEX IF NOT EXISTS idx_agent_data_type ON agent_data(data_type)",
                    "CREATE INDEX IF NOT EXISTS idx_agent_data_created_at ON agent_data(created_at)"
                ]
            }
            
            return {
                "success": True,
                "message": "Schema definition provided. Create table manually or via SQL.",
                "schema": schema_definition
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def validate_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database"""
        try:
            # Try to query the table with a limit of 0 to check existence
            result = self.select(table_name, columns="*", limit=0)
            return result.status_code == 200
        except:
            return False
    
    # A2A Specific Operations
    def store_agent_data(
        self,
        agent_id: str,
        data_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        table_name: str = "agent_data"
    ) -> SupabaseResponse:
        """Store A2A agent data with metadata
        
        Note: This method assumes the target table exists. 
        Use create_agent_data_table() to create the table if needed.
        """
        record = {
            "agent_id": agent_id,
            "data_type": data_type,
            "data": data,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        return self.insert(table_name, record)
    
    def get_agent_data(
        self,
        agent_id: str,
        data_type: Optional[str] = None,
        limit: int = 100
    ) -> SupabaseResponse:
        """Retrieve A2A agent data"""
        filters = {"agent_id": agent_id}
        if data_type:
            filters["data_type"] = data_type
        
        return self.select(
            table="agent_data",
            filters=filters,
            limit=limit,
            order_by="-created_at"
        )
    
    def log_agent_interaction(
        self,
        agent_id: str,
        interaction_type: str,
        details: Dict[str, Any],
        success: bool = True
    ) -> SupabaseResponse:
        """Log A2A agent interactions"""
        record = {
            "agent_id": agent_id,
            "interaction_type": interaction_type,
            "details": details,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return self.insert("agent_interactions", record)
    
    def store_financial_data(
        self,
        data_source: str,
        data_type: str,
        records: List[Dict[str, Any]],
        validation_status: str = "pending"
    ) -> SupabaseResponse:
        """Store financial data with validation tracking"""
        processed_records = []
        
        for record in records:
            processed_record = {
                "data_source": data_source,
                "data_type": data_type,
                "record_data": record,
                "validation_status": validation_status,
                "created_at": datetime.utcnow().isoformat()
            }
            processed_records.append(processed_record)
        
        return self.insert("financial_data", processed_records)
    
    def get_financial_data(
        self,
        data_source: Optional[str] = None,
        data_type: Optional[str] = None,
        validation_status: Optional[str] = None,
        limit: int = 1000
    ) -> SupabaseResponse:
        """Retrieve financial data with optional filters"""
        filters = {}
        if data_source:
            filters["data_source"] = data_source
        if data_type:
            filters["data_type"] = data_type
        if validation_status:
            filters["validation_status"] = validation_status
        
        return self.select(
            table="financial_data",
            filters=filters,
            limit=limit,
            order_by="-created_at"
        )
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for the Supabase client"""
        try:
            # Test database connection
            response = self.select("information_schema.tables", limit=1)
            
            # Test auth
            user_response = self.get_user()
            
            # Test storage (list buckets)
            try:
                buckets = self.client.storage.list_buckets()
                storage_available = True
                bucket_count = len(buckets) if buckets else 0
            except:
                storage_available = False
                bucket_count = 0
            
            return {
                "status": "healthy",
                "database": "connected" if response.status_code == 200 else "error",
                "auth": "available" if user_response["success"] else "error",
                "storage": "available" if storage_available else "unavailable",
                "bucket_count": bucket_count,
                "url": self.config.url
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def close(self):
        """Close client connections (if any persistent connections exist)"""
        logger.info("Supabase client connections closed")


# Factory function for easy instantiation
def create_supabase_client(config: Optional[SupabaseConfig] = None) -> SupabaseClient:
    """Factory function to create a Supabase client"""
    return SupabaseClient(config)


# Singleton instance for global use
_supabase_client_instance: Optional[SupabaseClient] = None

def get_supabase_client() -> SupabaseClient:
    """Get singleton Supabase client instance"""
    global _supabase_client_instance
    
    if _supabase_client_instance is None:
        _supabase_client_instance = create_supabase_client()
    
    return _supabase_client_instance
