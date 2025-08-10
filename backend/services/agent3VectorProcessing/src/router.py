"""
A2A Protocol Router for Vector Processing Agent
Implements A2A v0.2.9 compliant endpoints
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import json
import asyncio
from datetime import datetime
from typing import Dict, Any

def create_a2a_router(agent):
    """Create A2A compliant router for the agent"""
    router = APIRouter(prefix="/a2a/agent3/v1", tags=["A2A Agent 3 - Vector Processing"])
    
    @router.get("/.well-known/agent.json")
    async def get_agent_card():
        """
        A2A Agent Card endpoint
        Returns agent capabilities and metadata
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
                "vector_storage": True,
                "similarity_search": True,
                "knowledge_graph": True,
                "supported_dimensions": [384, 768, 1536],
                "max_vectors": 1000000,
                "index_types": ["flat", "hnsw"],
                "vector_db_type": "mock" if agent.vector_db_config.get("use_mock") else "sap_hana"
            },
            "endpoints": {
                "rpc": "/a2a/agent3/v1/rpc",
                "stream": "/a2a/agent3/v1/stream",
                "status": "/a2a/agent3/v1/status",
                "search": "/a2a/agent3/v1/search"
            },
            "trust_contract": {
                "type": "smart_contract",
                "address": agent.trust_contract_address if hasattr(agent, 'trust_contract_address') else None
            }
        }
    
    @router.post("/rpc")
    async def json_rpc_handler(request: Request):
        """
        A2A JSON-RPC 2.0 endpoint
        Handle vector storage requests via RPC
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
            if method == "store_vectors":
                # Create A2A message from RPC params
                message = agent.create_message(params)
                context_id = params.get("context_id", str(agent.generate_context_id()))
                
                result = await agent.handle_vector_storage_request(message, context_id)
                
                return JSONResponse(content={
                    "jsonrpc": "2.0",
                    "result": result,
                    "id": request_id
                })
            
            elif method == "search_similar":
                query_vector = params.get("query_vector")
                top_k = params.get("top_k", 10)
                filters = params.get("filters")
                
                if not query_vector:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32602,
                                "message": "Invalid params - query_vector required"
                            },
                            "id": request_id
                        }
                    )
                
                results = await agent.search_similar_vectors(query_vector, top_k, filters)
                
                return JSONResponse(content={
                    "jsonrpc": "2.0",
                    "result": {
                        "results": results,
                        "count": len(results)
                    },
                    "id": request_id
                })
            
            elif method == "get_status":
                task_id = params.get("task_id")
                if not task_id:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32602,
                                "message": "Invalid params - task_id required"
                            },
                            "id": request_id
                        }
                    )
                
                status = await agent.get_task_status(task_id)
                return JSONResponse(content={
                    "jsonrpc": "2.0",
                    "result": status,
                    "id": request_id
                })
            
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
    
    @router.get("/stream")
    async def event_stream():
        """
        A2A Server-Sent Events endpoint
        Stream vector processing updates
        """
        async def generate():
            while True:
                # Get next event from agent's event queue
                event = await agent.get_next_event()
                if event:
                    yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(0.5)
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
    
    @router.get("/status")
    async def get_agent_status():
        """Get current agent status and statistics"""
        return {
            "agent_id": agent.agent_id,
            "status": "active" if agent.is_ready else "initializing",
            "registered": agent.is_registered,
            "vector_db_connected": agent.vector_db_connected,
            "statistics": agent.processing_stats,
            "active_tasks": len([t for t in agent.tasks.values() if t.status == "processing"]),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @router.post("/search")
    async def vector_search(request: Request):
        """Direct vector search endpoint"""
        try:
            data = await request.json()
            query_vector = data.get("query_vector")
            top_k = data.get("top_k", 10)
            filters = data.get("filters")
            
            if not query_vector:
                raise HTTPException(status_code=400, detail="query_vector required")
            
            results = await agent.search_similar_vectors(query_vector, top_k, filters)
            
            return {
                "results": results,
                "count": len(results),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/invoke/{skill_name}")
    async def invoke_skill(skill_name: str, request: Request):
        """Direct skill invocation endpoint"""
        try:
            if skill_name not in agent.skills:
                raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")
            
            data = await request.json()
            
            # Execute skill
            if skill_name == "store_embedding":
                result = await agent.store_embedding(
                    data["entity_id"],
                    data["embedding"],
                    data.get("metadata", {})
                )
            elif skill_name == "search_similar":
                result = await agent.search_similar_vectors(
                    data["query_vector"],
                    data.get("top_k", 10),
                    data.get("filters")
                )
            elif skill_name == "build_knowledge_graph":
                result = await agent.build_knowledge_graph(
                    data["entities"],
                    data["relationships"]
                )
            else:
                skill = agent.skills[skill_name]
                result = await skill.handler(agent, data)
            
            return {
                "skill": skill_name,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return router