"""
A2A Protocol Router for Data Standardization Agent
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
    router = APIRouter(prefix="/a2a/agent1/v1", tags=["A2A Agent 1 - Data Standardization"])
    
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
                "standardization_types": list(agent.standardizers.keys()),
                "input_formats": ["csv", "json", "excel"],
                "output_formats": ["json", "parquet"],
                "batch_processing": True
            },
            "endpoints": {
                "rpc": "/a2a/agent1/v1/rpc",
                "stream": "/a2a/agent1/v1/stream",
                "status": "/a2a/agent1/v1/status"
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
        Handle standardization requests via RPC
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
            if method == "standardize_data":
                # Create A2A message from RPC params
                message = agent.create_message(params)
                context_id = params.get("context_id", str(agent.generate_context_id()))
                
                result = await agent.handle_standardization_request(message, context_id)
                
                return JSONResponse(content={
                    "jsonrpc": "2.0",
                    "result": result,
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
        Stream standardization progress and updates
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
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
    
    @router.get("/status")
    async def get_agent_status():
        """Get current agent status and statistics"""
        return {
            "agent_id": agent.agent_id,
            "status": "active" if agent.is_ready else "initializing",
            "registered": agent.is_registered,
            "statistics": {
                "total_processed": agent.standardization_stats["total_processed"],
                "successful_standardizations": agent.standardization_stats["successful_standardizations"],
                "records_standardized": agent.standardization_stats["records_standardized"],
                "data_types_processed": list(agent.standardization_stats["data_types_processed"])
            },
            "active_tasks": len([t for t in agent.tasks.values() if t.status == "processing"]),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @router.post("/invoke/{skill_name}")
    async def invoke_skill(skill_name: str, request: Request):
        """Direct skill invocation endpoint"""
        try:
            if skill_name not in agent.skills:
                raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")
            
            data = await request.json()
            skill = agent.skills[skill_name]
            
            # Execute skill
            result = await skill.handler(agent, data)
            
            return {
                "skill": skill_name,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return router