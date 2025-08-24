from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import json
import asyncio
from datetime import datetime

from .aiPreparationAgentSdk import AIPreparationAgentSDK, A2AMessage

from app.a2a.core.security_base import SecureA2AAgent
router = APIRouter(prefix="/a2a/agent2/v1", tags=["Agent 2 - AI Preparation"])

# Initialize Agent 2
# Note: agent2 will be set dynamically from launch_agent2.py
agent2 = None


@router.get("/.well-known/agent.json")
async def get_agent_card():
    """Get the agent card for Agent 2"""
    if agent2:
        return agent2.get_agent_card()
    else:
        raise HTTPException(status_code=503, detail="Agent 2 not initialized")


@router.post("/rpc")
async def json_rpc_handler(request: Request):
    """Handle JSON-RPC 2.0 requests for Agent 2"""
    try:
        body = await request.json()
        
        if "jsonrpc" not in body or body["jsonrpc"] != "2.0":
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request"
                    },
                    "id": body.get("id")
                }
            )
        
        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id")
        
        if not agent2:
            return JSONResponse(
                status_code=503,
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32000,
                        "message": "Agent 2 not initialized"
                    },
                    "id": request_id
                }
            )
        
        if method == "agent.getCard":
            result = agent2.get_agent_card()
        
        elif method == "agent.processMessage":
            message = A2AMessage(**params.get("message", {}))
            context_id = params.get("contextId", str(datetime.utcnow().timestamp()))
            priority = params.get("priority", "medium")
            result = await agent2.process_message(message, context_id, priority)
        
        elif method == "agent.getTaskStatus":
            task_id = params.get("taskId")
            result = agent2.get_task(task_id)
            if not result:
                return JSONResponse(
                    status_code=404,
                    content={
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32602,
                            "message": f"Task {task_id} not found"
                        },
                        "id": request_id
                    }
                )
        
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": "Method not found"
                    },
                    "id": request_id
                }
            )
        
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                },
                "id": body.get("id") if "body" in locals() else None
            }
        )


@router.post("/messages")
async def rest_message_handler(request: Request):
    """REST-style message endpoint for Agent 2"""
    try:
        if not agent2:
            raise HTTPException(status_code=503, detail="Agent 2 not initialized")
        
        body = await request.json()
        message = A2AMessage(**body.get("message", {}))
        context_id = body.get("contextId", str(datetime.utcnow().timestamp()))
        priority = body.get("priority", "medium")
        
        result = await agent2.process_message(message, context_id, priority)
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status for Agent 2"""
    try:
        if not agent2:
            raise HTTPException(status_code=503, detail="Agent 2 not initialized")
        
        task = agent2.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task
    except Exception as e:
        return JSONResponse(
            status_code=404,
            content={"error": str(e)}
        )


@router.get("/queue/status")
async def get_queue_status():
    """Get message queue status for Agent 2"""
    if agent2 and agent2.message_queue:
        return agent2.message_queue.get_queue_status()
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "Message queue not available"}
        )


@router.get("/queue/messages/{message_id}")
async def get_message_status(message_id: str):
    """Get status of a specific message"""
    if agent2 and agent2.message_queue:
        status = agent2.message_queue.get_message_status(message_id)
        if status:
            return status
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Message not found"}
            )
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "Message queue not available"}
        )


@router.delete("/queue/messages/{message_id}")
async def cancel_message(message_id: str):
    """Cancel a queued or processing message"""
    if agent2 and agent2.message_queue:
        cancelled = await agent2.message_queue.cancel_message(message_id)
        if cancelled:
            return {"message": "Message cancelled successfully"}
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Message not found or cannot be cancelled"}
            )
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "Message queue not available"}
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for Agent 2"""
    queue_info = {}
    if agent2 and agent2.message_queue:
        queue_status = agent2.message_queue.get_queue_status()
        queue_info = {
            "queue_depth": queue_status["queue_status"]["queue_depth"],
            "processing_count": queue_status["queue_status"]["processing_count"],
            "streaming_enabled": queue_status["capabilities"]["streaming_enabled"],
            "batch_processing_enabled": queue_status["capabilities"]["batch_processing_enabled"]
        }
    
    return {
        "status": "healthy",
        "agent": "AI Preparation Agent",
        "version": "1.0.0",
        "protocol_version": "0.2.9",
        "timestamp": datetime.utcnow().isoformat(),
        "message_queue": queue_info
    }


@router.get("/skills")
async def get_skills():
    """Get available skills for Agent 2"""
    if agent2:
        return {"skills": agent2.skills}
    else:
        raise HTTPException(status_code=503, detail="Agent 2 not initialized")


@router.get("/capabilities")
async def get_capabilities():
    """Get capabilities for Agent 2"""
    if agent2:
        return {"capabilities": agent2.capabilities}
    else:
        raise HTTPException(status_code=503, detail="Agent 2 not initialized")