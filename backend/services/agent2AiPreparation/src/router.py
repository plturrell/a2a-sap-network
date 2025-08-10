"""
A2A Protocol Router for AI Preparation Agent
Implements JSON-RPC 2.0 endpoints for Agent 2
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import json
import asyncio
from datetime import datetime
from typing import Dict, Any

def create_a2a_router(agent):
    """Create A2A compliant router for AI Preparation Agent"""
    router = APIRouter(tags=["A2A AI Preparation"])
    
    @router.get("/.well-known/agent.json")
    async def get_agent_card():
        """
        A2A Agent Card endpoint
        Returns agent capabilities for AI preparation
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
                "semantic_enrichment": True,
                "embedding_generation": True,
                "embedding_model": agent.embedding_model_name if hasattr(agent, 'embedding_model_name') else "all-MiniLM-L6-v2",
                "embedding_dimension": agent.embedding_dim,
                "relationship_extraction": True,
                "supported_entities": ["account", "book", "location", "measure", "product"]
            },
            "endpoints": {
                "rpc": f"/a2a/{agent.agent_id}/v1/rpc",
                "status": f"/a2a/{agent.agent_id}/v1/status",
                "tasks": f"/a2a/{agent.agent_id}/v1/tasks"
            }
        }
    
    @router.post(f"/a2a/{agent.agent_id}/v1/rpc")
    async def json_rpc_handler(request: Request):
        """
        JSON-RPC 2.0 endpoint for AI preparation operations
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
            if method == "prepare_for_ai":
                context_id = params.get("context_id", str(agent.generate_context_id()))
                message = agent.create_message(params)
                result = await agent.handle_ai_preparation_request(message, context_id)
            
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
            "statistics": agent.processing_stats,
            "embedding_model": {
                "name": agent.embedding_model_name if hasattr(agent, 'embedding_model_name') else "unknown",
                "dimension": agent.embedding_dim,
                "loaded": agent.embedding_model is not None
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @router.get(f"/a2a/{agent.agent_id}/v1/tasks")
    async def list_tasks(status: str = None):
        """List active tasks with optional status filter"""
        tasks = []
        
        for task_id, task in agent.tasks.items():
            if status and task.get("status") != status:
                continue
            
            tasks.append({
                "task_id": task_id,
                "type": task.get("type"),
                "status": task.get("status"),
                "created_at": task.get("created_at"),
                "updated_at": task.get("updated_at"),
                "metadata": task.get("metadata", {})
            })
        
        return {
            "tasks": tasks,
            "count": len(tasks),
            "filter": {"status": status} if status else None
        }
    
    @router.get(f"/a2a/{agent.agent_id}/v1/tasks/{task_id}")
    async def get_task_details(task_id: str):
        """Get detailed information about a specific task"""
        task = agent.tasks.get(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        return task
    
    @router.post(f"/a2a/{agent.agent_id}/v1/sse")
    async def sse_endpoint(request: Request):
        """
        Server-Sent Events endpoint for real-time task updates
        """
        async def event_generator():
            """Generate SSE events for task updates"""
            try:
                # Send initial connection event
                yield f"event: connected\ndata: {{\"agent_id\": \"{agent.agent_id}\", \"timestamp\": \"{datetime.utcnow().isoformat()}\"}}\n\n"
                
                # Monitor task updates (simplified for now)
                while True:
                    await asyncio.sleep(5)  # Check every 5 seconds
                    
                    # Send heartbeat
                    yield f"event: heartbeat\ndata: {{\"timestamp\": \"{datetime.utcnow().isoformat()}\"}}\n\n"
                    
            except asyncio.CancelledError:
                yield f"event: disconnect\ndata: {{\"agent_id\": \"{agent.agent_id}\"}}\n\n"
                raise
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    @router.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint"""
        metrics_text = f"""# HELP a2a_entities_enriched_total Total number of entities enriched
# TYPE a2a_entities_enriched_total counter
a2a_entities_enriched_total {agent.processing_stats['entities_enriched']}

# HELP a2a_embeddings_generated_total Total number of embeddings generated
# TYPE a2a_embeddings_generated_total counter
a2a_embeddings_generated_total {agent.processing_stats['embeddings_generated']}

# HELP a2a_relationships_extracted_total Total number of relationships extracted
# TYPE a2a_relationships_extracted_total counter
a2a_relationships_extracted_total {agent.processing_stats['relationships_extracted']}

# HELP a2a_preparation_tasks_total Total number of AI preparation tasks processed
# TYPE a2a_preparation_tasks_total counter
a2a_preparation_tasks_total {agent.processing_stats['total_processed']}
"""
        return metrics_text
    
    return router