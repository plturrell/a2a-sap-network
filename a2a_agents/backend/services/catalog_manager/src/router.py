"""
A2A Protocol Router for Catalog Manager Agent
Implements JSON-RPC 2.0 endpoints for ORD registry operations
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import json
from datetime import datetime
from typing import Dict, Any

def create_a2a_router(agent):
    """Create A2A compliant router for Catalog Manager"""
    router = APIRouter(tags=["A2A Catalog Manager"])
    
    @router.get("/.well-known/agent.json")
    async def get_agent_card():
        """
        A2A Agent Card endpoint
        Returns agent capabilities for catalog management
        """
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "description": agent.description,
            "version": agent.version,
            "protocol_version": "0.2.9",
            "capabilities": {
                "handlers": [
                    {
                        "name": handler.name,
                        "description": handler.description
                    } for handler in agent.handlers.values()
                ],
                "skills": [
                    {
                        "name": skill.name,
                        "description": skill.description
                    } for skill in agent.skills.values()
                ],
                "ord_registry": True,
                "data_product_catalog": True,
                "api_discovery": True,
                "event_discovery": True,
                "semantic_search": True,
                "quality_scoring": True,
                "usage_analytics": True,
                "resource_types": ["api", "event", "data_product", "capability", "integration", "package"]
            },
            "endpoints": {
                "rpc": f"/a2a/{agent.agent_id}/v1/rpc",
                "status": f"/a2a/{agent.agent_id}/v1/status",
                "ord": f"/a2a/{agent.agent_id}/v1/ord",
                "catalog": f"/a2a/{agent.agent_id}/v1/catalog",
                "search": f"/a2a/{agent.agent_id}/v1/search"
            }
        }
    
    @router.post(f"/a2a/{agent.agent_id}/v1/rpc")
    async def json_rpc_handler(request: Request):
        """
        JSON-RPC 2.0 endpoint for catalog operations
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
            if method == "register_resource":
                context_id = params.get("context_id", str(agent.generate_context_id()))
                message = agent.create_message(params)
                result = await agent.handle_register_resource(message, context_id)
            
            elif method == "discover_resources":
                context_id = params.get("context_id", str(agent.generate_context_id()))
                message = agent.create_message(params)
                result = await agent.handle_discover_resources(message, context_id)
            
            elif method == "catalog_data_product":
                context_id = params.get("context_id", str(agent.generate_context_id()))
                message = agent.create_message(params)
                result = await agent.handle_catalog_data_product(message, context_id)
            
            elif method == "update_quality_score":
                context_id = params.get("context_id", str(agent.generate_context_id()))
                message = agent.create_message(params)
                result = await agent.handle_update_quality_score(message, context_id)
            
            elif method == "semantic_search":
                result = await agent.semantic_search(
                    params.get("query", ""),
                    params.get("limit", 10)
                )
            
            elif method == "get_usage_analytics":
                result = await agent.get_usage_analytics(
                    params.get("timeframe", "30d")
                )
            
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
        return {
            "agent_id": agent.agent_id,
            "status": "active" if agent.is_ready else "initializing",
            "registered": agent.is_registered,
            "statistics": agent.metrics,
            "cache": {
                "ord_resources": len(agent.ord_cache),
                "data_products": len(agent.product_cache),
                "redis_available": agent.redis_client is not None
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @router.get(f"/a2a/{agent.agent_id}/v1/ord")
    async def get_ord_document():
        """
        Get ORD document in SAP standard format
        https://sap.github.io/open-resource-discovery/
        """
        # Get all active resources
        resources = []
        for resource in agent.ord_cache.values():
            if resource.status.value != "retired":
                resources.append(agent._resource_to_dict(resource))
        
        ord_document = {
            "$schema": "https://sap.github.io/open-resource-discovery/spec-v1/interfaces/Document.schema.json",
            "openResourceDiscovery": "1.9",
            "products": [
                {
                    "ordId": "sap.a2a:product:a2a-network:v1",
                    "title": "A2A Network",
                    "shortDescription": "Agent-to-Agent Network for Enterprise Integration",
                    "vendor": "SAP"
                }
            ],
            "packages": [
                {
                    "ordId": "sap.a2a:package:a2a-core:v1",
                    "title": "A2A Core Package",
                    "shortDescription": "Core A2A network components",
                    "version": "1.0.0"
                }
            ],
            "consumptionBundles": [],
            "apis": [r for r in resources if r["resource_type"] == "api"],
            "events": [r for r in resources if r["resource_type"] == "event"],
            "capabilities": [r for r in resources if r["resource_type"] == "capability"],
            "dataProducts": [r for r in resources if r["resource_type"] == "data_product"],
            "integrationDependencies": []
        }
        
        return ord_document
    
    @router.get(f"/a2a/{agent.agent_id}/v1/catalog")
    async def get_catalog_summary():
        """Get catalog summary with statistics"""
        summary = {
            "total_resources": len(agent.ord_cache),
            "total_products": len(agent.product_cache),
            "resources_by_type": {},
            "resources_by_status": {},
            "top_categories": {},
            "recent_additions": []
        }
        
        # Count by type and status
        for resource in agent.ord_cache.values():
            # By type
            resource_type = resource.resource_type.value
            summary["resources_by_type"][resource_type] = \
                summary["resources_by_type"].get(resource_type, 0) + 1
            
            # By status
            status = resource.status.value
            summary["resources_by_status"][status] = \
                summary["resources_by_status"].get(status, 0) + 1
        
        # Top categories for data products
        category_counts = {}
        for product in agent.product_cache.values():
            category_counts[product.category] = category_counts.get(product.category, 0) + 1
        
        summary["top_categories"] = dict(sorted(
            category_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])
        
        # Recent additions (last 10)
        recent = sorted(
            agent.ord_cache.values(),
            key=lambda x: x.created_at,
            reverse=True
        )[:10]
        
        summary["recent_additions"] = [
            {
                "ord_id": r.ord_id,
                "title": r.title,
                "type": r.resource_type.value,
                "created_at": r.created_at.isoformat()
            }
            for r in recent
        ]
        
        return summary
    
    @router.post(f"/a2a/{agent.agent_id}/v1/search")
    async def search_catalog(request: Request):
        """Advanced search endpoint with multiple search strategies"""
        try:
            body = await request.json()
            
            search_type = body.get("type", "keyword")
            query = body.get("query", "")
            filters = body.get("filters", {})
            limit = body.get("limit", 20)
            
            if search_type == "semantic":
                # Use semantic search
                results = await agent.semantic_search(query, limit)
            
            elif search_type == "structured":
                # Use structured query
                message = agent.create_message(filters)
                response = await agent.handle_discover_resources(
                    message,
                    str(agent.generate_context_id())
                )
                
                if response["status"] == "success":
                    results = response["data"]["resources"]
                else:
                    results = []
            
            else:
                # Default keyword search
                results = await agent._keyword_search(query, limit)
            
            return {
                "results": results,
                "count": len(results),
                "search_type": search_type,
                "query": query,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get(f"/a2a/{agent.agent_id}/v1/resources/{{ord_id}}")
    async def get_resource_details(ord_id: str):
        """Get detailed information about a specific resource"""
        # Check cache first
        if ord_id in agent.ord_cache:
            resource = agent.ord_cache[ord_id]
            return agent._resource_to_dict(resource)
        
        # Query database
        query = "SELECT * FROM ord_resources WHERE ord_id = ?"
        async with agent.db_connection.execute(query, (ord_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                resource = agent._row_to_resource(row)
                return agent._resource_to_dict(resource)
        
        raise HTTPException(status_code=404, detail=f"Resource {ord_id} not found")
    
    @router.get(f"/a2a/{agent.agent_id}/v1/products/{{product_id}}")
    async def get_product_details(product_id: str):
        """Get detailed information about a data product"""
        # Check cache first
        if product_id in agent.product_cache:
            product = agent.product_cache[product_id]
            
            # Increment usage count
            product.usage_count += 1
            product.last_accessed = datetime.utcnow()
            
            # Update in database (async)
            asyncio.create_task(agent._update_product_usage(product_id))
            
            return {
                "product_id": product.product_id,
                "ord_id": product.ord_id,
                "name": product.name,
                "description": product.description,
                "category": product.category,
                "data_sources": product.data_sources,
                "output_formats": product.output_formats,
                "refresh_frequency": product.refresh_frequency,
                "quality_score": product.quality_score,
                "usage_count": product.usage_count,
                "last_accessed": product.last_accessed.isoformat() if product.last_accessed else None,
                "metadata": product.metadata
            }
        
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
    
    @router.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint"""
        metrics_text = f"""# HELP a2a_resources_registered_total Total number of resources registered
# TYPE a2a_resources_registered_total counter
a2a_resources_registered_total {agent.metrics['total_resources_registered']}

# HELP a2a_products_cataloged_total Total number of products cataloged  
# TYPE a2a_products_cataloged_total counter
a2a_products_cataloged_total {agent.metrics['total_products_cataloged']}

# HELP a2a_discovery_requests_total Total number of discovery requests
# TYPE a2a_discovery_requests_total counter
a2a_discovery_requests_total {agent.metrics['discovery_requests']}

# HELP a2a_api_registrations_total Total number of API registrations
# TYPE a2a_api_registrations_total counter
a2a_api_registrations_total {agent.metrics['api_registrations']}

# HELP a2a_event_registrations_total Total number of event registrations
# TYPE a2a_event_registrations_total counter
a2a_event_registrations_total {agent.metrics['event_registrations']}

# HELP a2a_cache_operations_total Total number of cache operations
# TYPE a2a_cache_operations_total counter
a2a_cache_operations_total {agent.metrics['cache_operations']}

# HELP a2a_ord_resources_cached Number of ORD resources in cache
# TYPE a2a_ord_resources_cached gauge
a2a_ord_resources_cached {len(agent.ord_cache)}

# HELP a2a_data_products_cached Number of data products in cache
# TYPE a2a_data_products_cached gauge
a2a_data_products_cached {len(agent.product_cache)}
"""
        return metrics_text
    
    @router.get("/")
    async def catalog_ui():
        """Simple web UI for browsing the catalog"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>A2A Catalog Manager</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }
                .stat-card { background: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                .stat-value { font-size: 24px; font-weight: bold; color: #333; }
                .stat-label { color: #666; font-size: 14px; }
                .search-box { width: 100%; padding: 10px; font-size: 16px; margin: 20px 0; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background: #f5f5f5; }
                .tag { background: #e0e0e0; padding: 2px 8px; border-radius: 3px; margin: 0 2px; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>A2A Catalog Manager</h1>
                    <p>Open Resource Discovery (ORD) Registry & Data Product Catalog</p>
                </div>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-value" id="total-resources">0</div>
                        <div class="stat-label">Total Resources</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="total-products">0</div>
                        <div class="stat-label">Data Products</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="total-apis">0</div>
                        <div class="stat-label">APIs</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="total-events">0</div>
                        <div class="stat-label">Events</div>
                    </div>
                </div>
                
                <input type="text" class="search-box" id="search" placeholder="Search catalog..." />
                
                <h2>Recent Resources</h2>
                <table id="resources-table">
                    <thead>
                        <tr>
                            <th>ORD ID</th>
                            <th>Title</th>
                            <th>Type</th>
                            <th>Status</th>
                            <th>Tags</th>
                        </tr>
                    </thead>
                    <tbody id="resources-body">
                    </tbody>
                </table>
            </div>
            
            <script>
                async function loadCatalog() {
                    const response = await fetch('/a2a/catalog_manager_agent/v1/catalog');
                    const data = await response.json();
                    
                    document.getElementById('total-resources').textContent = data.total_resources;
                    document.getElementById('total-products').textContent = data.total_products;
                    document.getElementById('total-apis').textContent = data.resources_by_type.api || 0;
                    document.getElementById('total-events').textContent = data.resources_by_type.event || 0;
                    
                    const tbody = document.getElementById('resources-body');
                    tbody.innerHTML = '';
                    
                    data.recent_additions.forEach(resource => {
                        const row = tbody.insertRow();
                        row.innerHTML = `
                            <td>${resource.ord_id}</td>
                            <td>${resource.title}</td>
                            <td>${resource.type}</td>
                            <td><span class="tag">${resource.status || 'active'}</span></td>
                            <td>${resource.tags ? resource.tags.map(t => `<span class="tag">${t}</span>`).join('') : ''}</td>
                        `;
                    });
                }
                
                document.getElementById('search').addEventListener('keypress', async (e) => {
                    if (e.key === 'Enter') {
                        const query = e.target.value;
                        const response = await fetch('/a2a/catalog_manager_agent/v1/search', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ query, type: 'keyword' })
                        });
                        const results = await response.json();
                        console.log('Search results:', results);
                    }
                });
                
                loadCatalog();
                setInterval(loadCatalog, 30000); // Refresh every 30 seconds
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    return router