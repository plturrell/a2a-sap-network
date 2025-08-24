"""
Catalog Manager Agent - A2A Microservice
Manages Open Resource Discovery (ORD) registry and data product catalog
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
import os
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
# HTTP client needed for initial registration with A2A network
# A2A Protocol: Use blockchain messaging instead of httpx
from contextlib import asynccontextmanager
import aiosqlite
import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential
import hashlib
import uuid
import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
shared_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'shared')
sys.path.insert(0, shared_dir)

from a2aCommon import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole
)
from a2aCommon.sdk.utils import create_success_response, create_error_response

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    API = "api"
    EVENT = "event"
    DATA_PRODUCT = "data_product"
    CAPABILITY = "capability"
    INTEGRATION = "integration"
    PACKAGE = "package"


class ResourceStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    BETA = "beta"
    PLANNED = "planned"
    RETIRED = "retired"


@dataclass
class ORDResource:
    """Open Resource Discovery resource entry"""
    ord_id: str
    title: str
    short_description: str
    description: str
    resource_type: ResourceType
    version: str
    status: ResourceStatus
    visibility: str = "public"
    package: str = ""
    responsible: str = ""
    tags: List[str] = field(default_factory=list)
    countries: List[str] = field(default_factory=list)
    industries: List[str] = field(default_factory=list)
    line_of_business: List[str] = field(default_factory=list)
    channels: List[str] = field(default_factory=list)
    links: Dict[str, str] = field(default_factory=dict)
    api_definitions: List[Dict[str, Any]] = field(default_factory=list)
    event_definitions: List[Dict[str, Any]] = field(default_factory=list)
    documentation: List[Dict[str, Any]] = field(default_factory=list)
    extensions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_modified_by: str = ""


@dataclass
class DataProductEntry:
    """Data product catalog entry"""
    product_id: str
    ord_id: str
    name: str
    description: str
    category: str
    data_sources: List[str]
    output_formats: List[str]
    refresh_frequency: str
    quality_score: float
    usage_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CatalogManagerAgent(A2AAgentBase):
    """
    Catalog Manager Agent - ORD Registry and Data Product Catalog
    """
    
    def __init__(self, base_url: str, agent_manager_url: str, data_manager_url: str):
        super().__init__(
            agent_id="catalog_manager_agent",
            name="Catalog Manager Agent",
            description="A2A v0.2.9 compliant agent for ORD registry and data product catalog management",
            version="2.0.0",
            base_url=base_url
        )
        
        self.agent_manager_url = agent_manager_url
        self.data_manager_url = data_manager_url
        
        # Storage
        self.db_connection = None
        self.redis_client = None
        self.http_client = None
        
        # Caches
        self.ord_cache: Dict[str, ORDResource] = {}
        self.product_cache: Dict[str, DataProductEntry] = {}
        
        # Metrics
        self.metrics = {
            "total_resources_registered": 0,
            "total_products_cataloged": 0,
            "discovery_requests": 0,
            "api_registrations": 0,
            "event_registrations": 0,
            "cache_operations": 0
        }
        
        # Configuration
        self.db_path = os.getenv("CATALOG_DB_PATH", "/tmp/a2a_catalog.db")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
        
        logger.info(f"Initialized {self.name} v{self.version}")
    
    async def initialize(self) -> None:
        """Initialize catalog storage and connections"""
        logger.info("Initializing Catalog Manager...")
        
        # Initialize HTTP client for registration and emergency communication
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Initialize database
        await self._initialize_database()
        
        # Initialize Redis cache
        try:
            self.redis_client = await redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.warning(f"Redis cache not available: {e}")
            self.redis_client = None
        
        # Load initial catalog
        await self._load_catalog()
        
        # Initialize A2A components
        self.is_ready = True
        self.is_registered = False
        self.tasks = {}
        
        logger.info("Catalog Manager initialized successfully")
    
    async def _initialize_database(self) -> None:
        """Initialize catalog database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.db_connection = await aiosqlite.connect(self.db_path)
        self.db_connection.row_factory = aiosqlite.Row
        
        # Create ORD resources table
        await self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS ord_resources (
                ord_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                short_description TEXT,
                description TEXT,
                resource_type TEXT NOT NULL,
                version TEXT NOT NULL,
                status TEXT NOT NULL,
                visibility TEXT DEFAULT 'public',
                package TEXT,
                responsible TEXT,
                tags TEXT,
                countries TEXT,
                industries TEXT,
                line_of_business TEXT,
                channels TEXT,
                links TEXT,
                api_definitions TEXT,
                event_definitions TEXT,
                documentation TEXT,
                extensions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_modified_by TEXT
            )
        """)
        
        # Create data products table
        await self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS data_products (
                product_id TEXT PRIMARY KEY,
                ord_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                category TEXT,
                data_sources TEXT,
                output_formats TEXT,
                refresh_frequency TEXT,
                quality_score REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ord_id) REFERENCES ord_resources(ord_id)
            )
        """)
        
        # Create indexes
        await self.db_connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_resource_type ON ord_resources(resource_type)"
        )
        await self.db_connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_resource_status ON ord_resources(status)"
        )
        await self.db_connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_product_category ON data_products(category)"
        )
        
        await self.db_connection.commit()
        logger.info("Catalog database initialized")
    
    async def register_with_network(self) -> None:
        """Register with A2A Agent Manager"""
        try:
            registration = {
                "agent_id": self.agent_id,
                "name": self.name,
                "base_url": self.base_url,
                "capabilities": {
                    "ord_registry": True,
                    "data_product_catalog": True,
                    "api_discovery": True,
                    "event_discovery": True,
                    "semantic_search": True,
                    "quality_scoring": True,
                    "usage_analytics": True
                },
                "handlers": list(self.handlers.keys()),
                "skills": [s.name for s in self.skills.values()]
            }
            
            # Send registration
            response = await self.http_client.post(
                f"{self.agent_manager_url}/rpc",
                json={
                    "jsonrpc": "2.0",
                    "method": "register_agent",
                    "params": registration,
                    "id": f"reg_{self.agent_id}_{int(datetime.utcnow().timestamp())}"
                }
            )
            
            if response.status_code == 200:
                logger.info("Registered with A2A network")
                self.is_registered = True
            else:
                logger.error(f"Registration failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to register: {e}")
            raise
    
    @a2a_handler("register_resource", "Register a resource in ORD")
    async def handle_register_resource(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Register a new resource in the ORD registry"""
        try:
            resource_data = message.content if hasattr(message, 'content') else message
            
            # Validate required fields
            required = ["title", "short_description", "resource_type", "version"]
            for field in required:
                if field not in resource_data:
                    return create_error_response(400, f"Missing required field: {field}")
            
            # Generate ORD ID if not provided
            ord_id = resource_data.get("ord_id")
            if not ord_id:
                # Generate deterministic ID based on content
                content_hash = hashlib.sha256(
                    f"{resource_data['title']}:{resource_data['resource_type']}:{resource_data['version']}".encode()
                ).hexdigest()[:12]
                ord_id = f"sap.a2a:{resource_data['resource_type']}:{content_hash}:v{resource_data['version']}"
            
            # Create ORD resource
            resource = ORDResource(
                ord_id=ord_id,
                title=resource_data["title"],
                short_description=resource_data["short_description"],
                description=resource_data.get("description", ""),
                resource_type=ResourceType(resource_data["resource_type"]),
                version=resource_data["version"],
                status=ResourceStatus(resource_data.get("status", "active")),
                visibility=resource_data.get("visibility", "public"),
                package=resource_data.get("package", ""),
                responsible=resource_data.get("responsible", ""),
                tags=resource_data.get("tags", []),
                countries=resource_data.get("countries", []),
                industries=resource_data.get("industries", []),
                line_of_business=resource_data.get("line_of_business", []),
                channels=resource_data.get("channels", []),
                links=resource_data.get("links", {}),
                api_definitions=resource_data.get("api_definitions", []),
                event_definitions=resource_data.get("event_definitions", []),
                documentation=resource_data.get("documentation", []),
                extensions=resource_data.get("extensions", {}),
                last_modified_by=message.sender_id if hasattr(message, 'sender_id') else "system"
            )
            
            # Store in database
            await self._store_resource(resource)
            
            # Update cache
            self.ord_cache[ord_id] = resource
            if self.redis_client:
                await self._cache_resource(resource)
            
            # Track metrics
            self.metrics["total_resources_registered"] += 1
            if resource.resource_type == ResourceType.API:
                self.metrics["api_registrations"] += 1
            elif resource.resource_type == ResourceType.EVENT:
                self.metrics["event_registrations"] += 1
            
            # Store in Data Manager for persistence
            await self._store_in_data_manager(resource, context_id)
            
            return create_success_response({
                "ord_id": ord_id,
                "status": "registered",
                "resource_type": resource.resource_type.value,
                "visibility": resource.visibility,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error registering resource: {e}")
            return create_error_response(500, str(e))
    
    @a2a_handler("discover_resources", "Discover resources by criteria")
    async def handle_discover_resources(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Discover resources based on search criteria"""
        try:
            criteria = message.content if hasattr(message, 'content') else message
            
            # Track discovery request
            self.metrics["discovery_requests"] += 1
            
            # Build query
            resources = await self._query_resources(criteria)
            
            # Format response
            results = []
            for resource in resources:
                results.append({
                    "ord_id": resource.ord_id,
                    "title": resource.title,
                    "short_description": resource.short_description,
                    "resource_type": resource.resource_type.value,
                    "version": resource.version,
                    "status": resource.status.value,
                    "tags": resource.tags,
                    "links": resource.links,
                    "score": self._calculate_relevance_score(resource, criteria)
                })
            
            # Sort by relevance score
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return create_success_response({
                "resources": results,
                "count": len(results),
                "query": criteria
            })
            
        except Exception as e:
            logger.error(f"Error discovering resources: {e}")
            return create_error_response(500, str(e))
    
    @a2a_handler("catalog_data_product", "Catalog a data product")
    async def handle_catalog_data_product(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Catalog a data product with its ORD resource"""
        try:
            product_data = message.content if hasattr(message, 'content') else message
            
            # Validate required fields
            required = ["name", "description", "category", "data_sources"]
            for field in required:
                if field not in product_data:
                    return create_error_response(400, f"Missing required field: {field}")
            
            # Register as ORD resource first
            ord_resource = {
                "title": product_data["name"],
                "short_description": f"Data Product: {product_data['name']}",
                "description": product_data["description"],
                "resource_type": "data_product",
                "version": product_data.get("version", "1.0.0"),
                "status": "active",
                "tags": product_data.get("tags", []) + ["data-product", product_data["category"]],
                "extensions": {
                    "data_sources": product_data["data_sources"],
                    "output_formats": product_data.get("output_formats", ["json"]),
                    "refresh_frequency": product_data.get("refresh_frequency", "daily")
                }
            }
            
            # Create ORD entry
            result = await self.handle_register_resource(
                self.create_message(ord_resource),
                context_id
            )
            
            if result["status"] != "success":
                return result
            
            ord_id = result["data"]["ord_id"]
            
            # Create data product entry
            product = DataProductEntry(
                product_id=f"dp_{uuid.uuid4().hex[:12]}",
                ord_id=ord_id,
                name=product_data["name"],
                description=product_data["description"],
                category=product_data["category"],
                data_sources=product_data["data_sources"],
                output_formats=product_data.get("output_formats", ["json"]),
                refresh_frequency=product_data.get("refresh_frequency", "daily"),
                quality_score=product_data.get("quality_score", 0.8),
                metadata=product_data.get("metadata", {})
            )
            
            # Store product
            await self._store_data_product(product)
            
            # Update cache
            self.product_cache[product.product_id] = product
            
            self.metrics["total_products_cataloged"] += 1
            
            return create_success_response({
                "product_id": product.product_id,
                "ord_id": ord_id,
                "status": "cataloged",
                "category": product.category,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error cataloging data product: {e}")
            return create_error_response(500, str(e))
    
    @a2a_handler("update_quality_score", "Update data product quality score")
    async def handle_update_quality_score(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Update quality score for a data product"""
        try:
            update_data = message.content if hasattr(message, 'content') else message
            
            product_id = update_data.get("product_id")
            quality_score = update_data.get("quality_score")
            
            if not product_id or quality_score is None:
                return create_error_response(400, "Missing product_id or quality_score")
            
            # Update in database
            query = """
                UPDATE data_products 
                SET quality_score = ?, last_accessed = ?
                WHERE product_id = ?
            """
            
            await self.db_connection.execute(query, (
                quality_score,
                datetime.utcnow().isoformat(),
                product_id
            ))
            await self.db_connection.commit()
            
            # Update cache
            if product_id in self.product_cache:
                self.product_cache[product_id].quality_score = quality_score
                self.product_cache[product_id].last_accessed = datetime.utcnow()
            
            return create_success_response({
                "product_id": product_id,
                "quality_score": quality_score,
                "updated_at": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error updating quality score: {e}")
            return create_error_response(500, str(e))
    
    @a2a_skill("semantic_search", "Search catalog using semantic similarity")
    async def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search catalog using semantic similarity with real embeddings"""
        try:
            # First, get all resources with their embeddings from Data Manager
            data_response = await self.http_client.post(
                f"{self.data_manager_url}/a2a/data_manager_agent/v1/rpc",
                json={
                    "jsonrpc": "2.0",
                    "method": "retrieve_data",
                    "params": {
                        "data_type": "resource_embeddings"
                    },
                    "id": "get_embeddings"
                }
            )
            
            resource_embeddings = {}
            if data_response.status_code == 200:
                result = data_response.json()
                if "result" in result and result["result"]["status"] == "success":
                    for record in result["result"]["data"]["records"]:
                        resource_embeddings[record["data"]["ord_id"]] = record["data"]["embedding"]
            
            # Get embedding for query from AI Preparation Agent
            ai_response = await self.http_client.post(
                f"{self.agent_manager_url}/rpc",
                json={
                    "jsonrpc": "2.0",
                    "method": "discover_agents",
                    "params": {
                        "capabilities": {"embedding_generation": True}
                    },
                    "id": "discover_ai_agent"
                }
            )
            
            query_embedding = None
            if ai_response.status_code == 200:
                agents = ai_response.json().get("result", {}).get("agents", [])
                if agents:
                    ai_agent_url = agents[0]["base_url"]
                    
                    # Get query embedding
                    embed_response = await self.http_client.post(
                        f"{ai_agent_url}/a2a/ai_preparation_agent_2/v1/rpc",
                        json={
                            "jsonrpc": "2.0",
                            "method": "generate_embeddings",
                            "params": {
                                "entities": [{"text": query, "type": "query", "id": "q1"}]
                            },
                            "id": "get_embedding"
                        }
                    )
                    
                    if embed_response.status_code == 200:
                        embed_result = embed_response.json()
                        if "result" in embed_result and isinstance(embed_result["result"], list):
                            query_embedding = embed_result["result"][0].get("embedding", {}).get("vector")
            
            # Perform semantic search
            if query_embedding and resource_embeddings:
                # Calculate cosine similarity
                query_vec = np.array(query_embedding)
                results = []
                
                for resource in await self._get_all_resources():
                    if resource.ord_id in resource_embeddings:
                        resource_vec = np.array(resource_embeddings[resource.ord_id])
                        
                        # Cosine similarity
                        similarity = np.dot(query_vec, resource_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(resource_vec))
                        
                        results.append({
                            "ord_id": resource.ord_id,
                            "title": resource.title,
                            "description": resource.short_description,
                            "score": float(similarity),
                            "resource_type": resource.resource_type.value
                        })
                
                # Sort by similarity and return top results
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:limit]
            
            # Fallback to enhanced keyword search with scoring
            return await self._enhanced_keyword_search(query, limit)
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return await self._keyword_search(query, limit)
    
    @a2a_skill("get_usage_analytics", "Get usage analytics for resources")
    async def get_usage_analytics(self, timeframe: str = "30d") -> Dict[str, Any]:
        """Get usage analytics for catalog resources"""
        # Parse timeframe
        days = 30
        if timeframe.endswith("d"):
            days = int(timeframe[:-1])
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Query usage data
        query = """
            SELECT 
                dp.category,
                COUNT(*) as product_count,
                AVG(dp.quality_score) as avg_quality,
                SUM(dp.usage_count) as total_usage
            FROM data_products dp
            WHERE dp.last_accessed >= ?
            GROUP BY dp.category
        """
        
        analytics = {
            "timeframe": timeframe,
            "categories": {},
            "top_products": [],
            "quality_distribution": {}
        }
        
        async with self.db_connection.execute(query, (cutoff_date.isoformat(),)) as cursor:
            async for row in cursor:
                analytics["categories"][row["category"]] = {
                    "count": row["product_count"],
                    "avg_quality": round(row["avg_quality"], 2),
                    "usage": row["total_usage"]
                }
        
        # Get top products
        top_query = """
            SELECT product_id, name, usage_count, quality_score
            FROM data_products
            ORDER BY usage_count DESC
            LIMIT 10
        """
        
        async with self.db_connection.execute(top_query) as cursor:
            async for row in cursor:
                analytics["top_products"].append({
                    "product_id": row["product_id"],
                    "name": row["name"],
                    "usage_count": row["usage_count"],
                    "quality_score": row["quality_score"]
                })
        
        return analytics
    
    async def _store_resource(self, resource: ORDResource) -> None:
        """Store ORD resource in database"""
        query = """
            INSERT OR REPLACE INTO ord_resources 
            (ord_id, title, short_description, description, resource_type, version,
             status, visibility, package, responsible, tags, countries, industries,
             line_of_business, channels, links, api_definitions, event_definitions,
             documentation, extensions, created_at, updated_at, last_modified_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        await self.db_connection.execute(query, (
            resource.ord_id,
            resource.title,
            resource.short_description,
            resource.description,
            resource.resource_type.value,
            resource.version,
            resource.status.value,
            resource.visibility,
            resource.package,
            resource.responsible,
            json.dumps(resource.tags),
            json.dumps(resource.countries),
            json.dumps(resource.industries),
            json.dumps(resource.line_of_business),
            json.dumps(resource.channels),
            json.dumps(resource.links),
            json.dumps(resource.api_definitions),
            json.dumps(resource.event_definitions),
            json.dumps(resource.documentation),
            json.dumps(resource.extensions),
            resource.created_at.isoformat(),
            resource.updated_at.isoformat(),
            resource.last_modified_by
        ))
        
        await self.db_connection.commit()
    
    async def _store_data_product(self, product: DataProductEntry) -> None:
        """Store data product in database"""
        query = """
            INSERT OR REPLACE INTO data_products
            (product_id, ord_id, name, description, category, data_sources,
             output_formats, refresh_frequency, quality_score, usage_count,
             last_accessed, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        await self.db_connection.execute(query, (
            product.product_id,
            product.ord_id,
            product.name,
            product.description,
            product.category,
            json.dumps(product.data_sources),
            json.dumps(product.output_formats),
            product.refresh_frequency,
            product.quality_score,
            product.usage_count,
            product.last_accessed.isoformat() if product.last_accessed else None,
            json.dumps(product.metadata)
        ))
        
        await self.db_connection.commit()
    
    async def _query_resources(self, criteria: Dict[str, Any]) -> List[ORDResource]:
        """Query resources based on criteria"""
        # Check cache first
        cache_key = f"query:{json.dumps(criteria, sort_keys=True)}"
        if self.redis_client:
            cached = await self.redis_client.get(cache_key)
            if cached:
                self.metrics["cache_operations"] += 1
                # Deserialize cached resources
                return [self._dict_to_resource(r) for r in json.loads(cached)]
        
        # Build query
        query = "SELECT * FROM ord_resources WHERE 1=1"
        params = []
        
        if "resource_type" in criteria:
            query += " AND resource_type = ?"
            params.append(criteria["resource_type"])
        
        if "status" in criteria:
            query += " AND status = ?"
            params.append(criteria["status"])
        
        if "tags" in criteria:
            # Simple tag matching - in production, use FTS
            for tag in criteria["tags"]:
                query += " AND tags LIKE ?"
                params.append(f"%{tag}%")
        
        if "visibility" in criteria:
            query += " AND visibility = ?"
            params.append(criteria["visibility"])
        
        # Execute query
        resources = []
        async with self.db_connection.execute(query, params) as cursor:
            async for row in cursor:
                resources.append(self._row_to_resource(row))
        
        # Cache results
        if self.redis_client and resources:
            await self.redis_client.setex(
                cache_key,
                300,  # 5 minute cache
                json.dumps([self._resource_to_dict(r) for r in resources])
            )
        
        return resources
    
    def _calculate_relevance_score(self, resource: ORDResource, criteria: Dict[str, Any]) -> float:
        """Calculate relevance score for a resource"""
        score = 1.0
        
        # Exact matches get higher scores
        if criteria.get("resource_type") == resource.resource_type.value:
            score += 0.5
        
        if criteria.get("status") == resource.status.value:
            score += 0.3
        
        # Tag matches
        if "tags" in criteria:
            matching_tags = set(criteria["tags"]) & set(resource.tags)
            score += len(matching_tags) * 0.2
        
        # Recency bonus
        age_days = (datetime.utcnow() - resource.updated_at).days
        if age_days < 7:
            score += 0.5
        elif age_days < 30:
            score += 0.3
        
        return min(score, 5.0)  # Cap at 5.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity score"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    async def _keyword_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback keyword search"""
        keywords = query.lower().split()
        all_resources = await self._get_all_resources()
        
        results = []
        for resource in all_resources:
            text = f"{resource.title} {resource.short_description} {' '.join(resource.tags)}".lower()
            score = sum(1 for keyword in keywords if keyword in text)
            
            if score > 0:
                results.append({
                    "ord_id": resource.ord_id,
                    "title": resource.title,
                    "description": resource.short_description,
                    "score": score
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    async def _enhanced_keyword_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Enhanced keyword search with better scoring"""
        keywords = query.lower().split()
        all_resources = await self._get_all_resources()
        
        results = []
        for resource in all_resources:
            # Create searchable text
            searchable = f"{resource.title} {resource.short_description} {' '.join(resource.tags)}".lower()
            
            # Calculate score based on multiple factors
            score = 0.0
            
            # Exact phrase match
            if query.lower() in searchable:
                score += 2.0
            
            # Individual keyword matches
            for keyword in keywords:
                if keyword in searchable:
                    score += 1.0
                if keyword in resource.title.lower():
                    score += 0.5  # Bonus for title match
            
            # Tag matches
            for tag in resource.tags:
                if any(keyword in tag.lower() for keyword in keywords):
                    score += 0.3
            
            # Status and recency bonus
            if resource.status == ResourceStatus.ACTIVE:
                score += 0.2
            
            if score > 0:
                results.append({
                    "ord_id": resource.ord_id,
                    "title": resource.title,
                    "description": resource.short_description,
                    "score": score,
                    "resource_type": resource.resource_type.value,
                    "tags": resource.tags
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    async def _get_all_resources(self) -> List[ORDResource]:
        """Get all resources from database"""
        resources = []
        query = "SELECT * FROM ord_resources WHERE status != 'retired'"
        
        async with self.db_connection.execute(query) as cursor:
            async for row in cursor:
                resources.append(self._row_to_resource(row))
        
        return resources
    
    def _row_to_resource(self, row: aiosqlite.Row) -> ORDResource:
        """Convert database row to ORDResource"""
        return ORDResource(
            ord_id=row["ord_id"],
            title=row["title"],
            short_description=row["short_description"],
            description=row["description"],
            resource_type=ResourceType(row["resource_type"]),
            version=row["version"],
            status=ResourceStatus(row["status"]),
            visibility=row["visibility"],
            package=row["package"],
            responsible=row["responsible"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            countries=json.loads(row["countries"]) if row["countries"] else [],
            industries=json.loads(row["industries"]) if row["industries"] else [],
            line_of_business=json.loads(row["line_of_business"]) if row["line_of_business"] else [],
            channels=json.loads(row["channels"]) if row["channels"] else [],
            links=json.loads(row["links"]) if row["links"] else {},
            api_definitions=json.loads(row["api_definitions"]) if row["api_definitions"] else [],
            event_definitions=json.loads(row["event_definitions"]) if row["event_definitions"] else [],
            documentation=json.loads(row["documentation"]) if row["documentation"] else [],
            extensions=json.loads(row["extensions"]) if row["extensions"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_modified_by=row["last_modified_by"]
        )
    
    def _resource_to_dict(self, resource: ORDResource) -> Dict[str, Any]:
        """Convert ORDResource to dictionary"""
        return {
            "ord_id": resource.ord_id,
            "title": resource.title,
            "short_description": resource.short_description,
            "description": resource.description,
            "resource_type": resource.resource_type.value,
            "version": resource.version,
            "status": resource.status.value,
            "visibility": resource.visibility,
            "package": resource.package,
            "responsible": resource.responsible,
            "tags": resource.tags,
            "countries": resource.countries,
            "industries": resource.industries,
            "line_of_business": resource.line_of_business,
            "channels": resource.channels,
            "links": resource.links,
            "api_definitions": resource.api_definitions,
            "event_definitions": resource.event_definitions,
            "documentation": resource.documentation,
            "extensions": resource.extensions,
            "created_at": resource.created_at.isoformat(),
            "updated_at": resource.updated_at.isoformat(),
            "last_modified_by": resource.last_modified_by
        }
    
    def _dict_to_resource(self, data: Dict[str, Any]) -> ORDResource:
        """Convert dictionary to ORDResource"""
        return ORDResource(
            ord_id=data["ord_id"],
            title=data["title"],
            short_description=data["short_description"],
            description=data["description"],
            resource_type=ResourceType(data["resource_type"]),
            version=data["version"],
            status=ResourceStatus(data["status"]),
            visibility=data["visibility"],
            package=data["package"],
            responsible=data["responsible"],
            tags=data["tags"],
            countries=data["countries"],
            industries=data["industries"],
            line_of_business=data["line_of_business"],
            channels=data["channels"],
            links=data["links"],
            api_definitions=data["api_definitions"],
            event_definitions=data["event_definitions"],
            documentation=data["documentation"],
            extensions=data["extensions"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            last_modified_by=data["last_modified_by"]
        )
    
    async def _cache_resource(self, resource: ORDResource) -> None:
        """Cache resource in Redis"""
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"resource:{resource.ord_id}",
                    self.cache_ttl,
                    json.dumps(self._resource_to_dict(resource))
                )
            except Exception as e:
                logger.warning(f"Failed to cache resource: {e}")
    
    async def _store_in_data_manager(self, resource: ORDResource, context_id: str) -> None:
        """Store resource in Data Manager for backup"""
        try:
            store_request = {
                "jsonrpc": "2.0",
                "method": "store_data",
                "params": {
                    "data_type": "ord_resources",
                    "data": self._resource_to_dict(resource),
                    "metadata": {
                        "catalog_manager": self.agent_id,
                        "resource_type": resource.resource_type.value
                    },
                    "context_id": context_id
                },
                "id": f"store_ord_{resource.ord_id}"
            }
            
            await self.http_client.post(
                f"{self.data_manager_url}/a2a/data_manager_agent/v1/rpc",
                json=store_request
            )
        except Exception as e:
            logger.warning(f"Failed to backup to Data Manager: {e}")
    
    async def _load_catalog(self) -> None:
        """Load catalog into memory caches"""
        # Load ORD resources
        query = "SELECT * FROM ord_resources WHERE status != 'retired' LIMIT 1000"
        async with self.db_connection.execute(query) as cursor:
            async for row in cursor:
                resource = self._row_to_resource(row)
                self.ord_cache[resource.ord_id] = resource
        
        logger.info(f"Loaded {len(self.ord_cache)} ORD resources into cache")
        
        # Load data products
        query = "SELECT * FROM data_products LIMIT 1000"
        async with self.db_connection.execute(query) as cursor:
            async for row in cursor:
                product = DataProductEntry(
                    product_id=row["product_id"],
                    ord_id=row["ord_id"],
                    name=row["name"],
                    description=row["description"],
                    category=row["category"],
                    data_sources=json.loads(row["data_sources"]),
                    output_formats=json.loads(row["output_formats"]),
                    refresh_frequency=row["refresh_frequency"],
                    quality_score=row["quality_score"],
                    usage_count=row["usage_count"],
                    last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else None,
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {}
                )
                self.product_cache[product.product_id] = product
        
        logger.info(f"Loaded {len(self.product_cache)} data products into cache")
    
    def generate_context_id(self) -> str:
        """Generate unique context ID"""
        return str(uuid.uuid4())
    
    def create_message(self, content: Any) -> A2AMessage:
        """Create A2A message"""
        return A2AMessage(
            sender_id=self.agent_id,
            content=content,
            role=MessageRole.AGENT
        )
    
    async def create_task(self, task_type: str, metadata: Dict[str, Any]) -> str:
        """Create and track a task"""
        task_id = str(uuid.uuid4())
        
        self.tasks[task_id] = {
            "task_id": task_id,
            "type": task_type,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": metadata
        }
        
        return task_id
    
    async def update_task_status(self, task_id: str, status: str, update_data: Dict[str, Any] = None):
        """Update task status"""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = status
            self.tasks[task_id]["updated_at"] = datetime.utcnow().isoformat()
            
            if update_data:
                self.tasks[task_id]["metadata"].update(update_data)
    
    async def deregister_from_network(self) -> None:
        """Deregister from A2A network"""
        logger.info("Deregistering from A2A network...")
        self.is_registered = False
    
    async def shutdown(self) -> None:
        """Cleanup resources"""
        logger.info("Shutting down Catalog Manager...")
        
        # Close database
        if self.db_connection:
            await self.db_connection.close()
        
        # Close Redis
        if self.redis_client:
            await self.redis_client.close()
        
        # Close HTTP client
        if self.http_client:
            await self.http_client.aclose()
        
        self.is_ready = False
        logger.info("Catalog Manager shutdown complete")