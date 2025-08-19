"""
A2A Protocol Router for Data Manager Agent
Implements JSON-RPC 2.0 endpoints for data storage operations
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import json
import asyncio
from datetime import datetime
from typing import Dict, Any

def create_a2a_router(agent):
    """Create A2A compliant router for Data Manager"""
    router = APIRouter(tags=["A2A Data Manager"])
    
    @router.get("/.well-known/agent.json")
    async def get_agent_card():
        """
        A2A Agent Card endpoint
        Returns agent capabilities for data management
        """
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "description": agent.description,
            "version": agent.version,
            "protocol_version": "0.2.9",
            "capabilities": {
                "handlers": [
                    {"name": "store_data", "description": "Store data in persistent storage"},
                    {"name": "retrieve_data", "description": "Retrieve data from storage"},
                    {"name": "update_data", "description": "Update existing data"},
                    {"name": "delete_data", "description": "Soft delete data"}
                ],
                "skills": [
                    {"name": "bulk_operations", "description": "Perform bulk data operations"},
                    {"name": "query_builder", "description": "Build complex queries"}
                ],
                "storage_backend": agent.storage_backend.value,
                "supports_versioning": True,
                "supports_caching": agent.redis_client is not None,
                "supports_bulk_operations": True,
                "supports_transactions": True,
                "supports_blockchain": agent.blockchain_enabled,
                "max_record_size_mb": 10,
                "data_types": ["accounts", "books", "locations", "measures", "products", "embeddings", "relationships", "quality_assessment"],
                "blockchain_address": agent.agent_identity.address if agent.blockchain_enabled and agent.agent_identity else None,
                "blockchain_capabilities": ["data_storage", "caching", "persistence", "reputation_tracking"] if agent.blockchain_enabled else []
            },
            "endpoints": {
                "rpc": f"/a2a/{agent.agent_id}/v1/rpc",
                "status": f"/a2a/{agent.agent_id}/v1/status",
                "health": "/health",
                "metrics": "/metrics"
            }
        }
    
    @router.post(f"/a2a/{agent.agent_id}/v1/rpc")
    async def json_rpc_handler(request: Request):
        """
        JSON-RPC 2.0 endpoint for data operations
        """
        try:
            body = await request.json()
            
            # Validate JSON-RPC 2.0 format
            if "jsonrpc" not in body or body["jsonrpc"] != "2.0":
                return JSONResponse(
                    status_code=400,
                    content={
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request - Not JSON-RPC 2.0"
                        },
                        "id": body.get("id")
                    }
                )
            
            method = body.get("method")
            params = body.get("params", {})
            request_id = body.get("id")
            
            # Route to appropriate handler
            if method == "store_data":
                context_id = params.get("context_id", str(agent.generate_context_id()))
                message = agent.create_message(params)
                result = await agent.handle_store_data(message, context_id)
            
            elif method == "retrieve_data":
                context_id = params.get("context_id", str(agent.generate_context_id()))
                message = agent.create_message(params)
                result = await agent.handle_retrieve_data(message, context_id)
            
            elif method == "update_data":
                context_id = params.get("context_id", str(agent.generate_context_id()))
                message = agent.create_message(params)
                result = await agent.handle_update_data(message, context_id)
            
            elif method == "delete_data":
                context_id = params.get("context_id", str(agent.generate_context_id()))
                message = agent.create_message(params)
                result = await agent.handle_delete_data(message, context_id)
            
            elif method == "bulk_operations":
                result = await agent.bulk_operations(params.get("operations", []))
            
            elif method == "query":
                result = await agent.query_builder(
                    params.get("filters", {}),
                    params.get("options", {})
                )
                # Convert QueryResult to dict
                if hasattr(result, "records"):
                    result = {
                        "records": [agent._record_to_dict(r) for r in result.records],
                        "total_count": result.total_count,
                        "page": result.page,
                        "page_size": result.page_size,
                        "has_next": result.has_next
                    }
            
            elif method in agent.handlers:
                context_id = params.get("context_id", str(agent.generate_context_id()))
                message = agent.create_message(params)
                result = await agent.handlers[method](message, context_id)
            
            else:
                return JSONResponse(
                    status_code=404,
                    content={
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}"
                        },
                        "id": request_id
                    }
                )
            
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            })
            
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    },
                    "id": None
                }
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    },
                    "id": body.get("id") if 'body' in locals() else None
                }
            )
    
    @router.get(f"/a2a/{agent.agent_id}/v1/status")
    async def get_agent_status():
        """Get current agent status and statistics"""
        db_status = "connected"
        try:
            # Check database connection
            if agent.db_connection:
                await agent.db_connection.execute("SELECT 1")
            else:
                db_status = "disconnected"
        except:
            db_status = "error"
        
        cache_status = "connected"
        try:
            # Check Redis connection
            if agent.redis_client:
                await agent.redis_client.ping()
            else:
                cache_status = "not_configured"
        except:
            cache_status = "error"
        
        return {
            "agent_id": agent.agent_id,
            "status": "active" if agent.is_ready else "initializing",
            "registered": agent.is_registered,
            "storage": {
                "backend": agent.storage_backend.value,
                "database_status": db_status,
                "cache_status": cache_status,
                "database_path": agent.sqlite_db_path if agent.storage_backend.value == "sqlite" else None
            },
            "blockchain": {
                "enabled": agent.blockchain_enabled,
                "address": agent.agent_identity.address if agent.blockchain_enabled and agent.agent_identity else None,
                "capabilities": ["data_storage", "caching", "persistence", "reputation_tracking"] if agent.blockchain_enabled else [],
                "registered_on_chain": agent.blockchain_enabled and agent.agent_identity and agent.agent_identity.address is not None
            },
            "statistics": agent.metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @router.post(f"/a2a/{agent.agent_id}/v1/export")
    async def export_data(request: Request):
        """Export data based on query"""
        try:
            body = await request.json()
            filters = body.get("filters", {})
            format_type = body.get("format", "json")
            
            # Query data
            result = await agent.query_builder(filters, {"page_size": 10000})
            
            if format_type == "json":
                return JSONResponse(content={
                    "records": [agent._record_to_dict(r) for r in result.records],
                    "count": result.total_count,
                    "exported_at": datetime.utcnow().isoformat()
                })
            elif format_type == "csv":
                # Generate CSV
                import csv
                import io
                
                output = io.StringIO()
                if result.records:
                    writer = csv.DictWriter(output, fieldnames=[
                        "record_id", "agent_id", "context_id", "data_type",
                        "created_at", "updated_at", "version"
                    ])
                    writer.writeheader()
                    
                    for record in result.records:
                        writer.writerow({
                            "record_id": record.record_id,
                            "agent_id": record.agent_id,
                            "context_id": record.context_id,
                            "data_type": record.data_type,
                            "created_at": record.created_at.isoformat(),
                            "updated_at": record.updated_at.isoformat(),
                            "version": record.version
                        })
                
                return StreamingResponse(
                    io.StringIO(output.getvalue()),
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": f"attachment; filename=a2a_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
                    }
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {format_type}")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get(f"/a2a/{agent.agent_id}/v1/schema")
    async def get_data_schema():
        """Get data schema information"""
        return {
            "version": "1.0",
            "data_types": {
                "accounts": {
                    "fields": ["id", "name", "currency", "balance", "type"],
                    "indexed_fields": ["id", "currency", "type"]
                },
                "books": {
                    "fields": ["id", "name", "type", "period", "entries"],
                    "indexed_fields": ["id", "type", "period"]
                },
                "locations": {
                    "fields": ["id", "name", "address", "country", "coordinates"],
                    "indexed_fields": ["id", "country"]
                },
                "measures": {
                    "fields": ["id", "name", "unit", "value", "timestamp"],
                    "indexed_fields": ["id", "unit", "timestamp"]
                },
                "products": {
                    "fields": ["id", "name", "category", "price", "currency"],
                    "indexed_fields": ["id", "category", "currency"]
                },
                "embeddings": {
                    "fields": ["id", "vector", "dimension", "model", "entity_type", "entity_id"],
                    "indexed_fields": ["id", "entity_type", "entity_id"]
                },
                "relationships": {
                    "fields": ["source_id", "target_id", "relationship_type", "confidence", "attributes"],
                    "indexed_fields": ["source_id", "target_id", "relationship_type"]
                }
            }
        }
    
    @router.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint"""
        metrics_text = f"""# HELP a2a_records_stored_total Total number of records stored
# TYPE a2a_records_stored_total counter
a2a_records_stored_total {agent.metrics['total_records_stored']}

# HELP a2a_queries_processed_total Total number of queries processed
# TYPE a2a_queries_processed_total counter
a2a_queries_processed_total {agent.metrics['total_queries_processed']}

# HELP a2a_cache_hits_total Total number of cache hits
# TYPE a2a_cache_hits_total counter
a2a_cache_hits_total {agent.metrics['cache_hits']}

# HELP a2a_cache_misses_total Total number of cache misses
# TYPE a2a_cache_misses_total counter
a2a_cache_misses_total {agent.metrics['cache_misses']}

# HELP a2a_storage_errors_total Total number of storage errors
# TYPE a2a_storage_errors_total counter
a2a_storage_errors_total {agent.metrics['storage_errors']}

# HELP a2a_storage_backend Storage backend type
# TYPE a2a_storage_backend info
a2a_storage_backend{{backend="{agent.storage_backend.value}"}} 1

# HELP a2a_cache_enabled Cache enabled status
# TYPE a2a_cache_enabled gauge
a2a_cache_enabled {1 if agent.redis_client else 0}

# HELP a2a_blockchain_enabled Blockchain integration enabled status
# TYPE a2a_blockchain_enabled gauge
a2a_blockchain_enabled {1 if agent.blockchain_enabled else 0}
"""
        return metrics_text
    
    return router