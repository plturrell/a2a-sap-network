from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from datetime import datetime

from ....core.a2aTypes import A2AMessage
router = APIRouter(prefix="/a2a/agent0/v1", tags=["Agent 0 - Data Product Registration"])

# Initialize Agent 0
# Note: downstream_agent_url will be set dynamically from launch_agent0.py
agent0 = None


@router.get("/.well-known/agent.json")
async def get_agent_card():
    """Get the agent card for Agent 0"""
    if agent0 is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Agent 0 not initialized yet"}
        )
    return await agent0.get_agent_card()


@router.post("/rpc")
async def json_rpc_handler(request: Request):
    """Handle JSON-RPC 2.0 requests for Agent 0"""
    if agent0 is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Agent 0 not initialized yet"}
        )
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

        if method == "agent.getCard":
            result = await agent0.get_agent_card()

        elif method == "agent.processMessage":
            message = A2AMessage(**params.get("message", {}))
            context_id = params.get("contextId", str(datetime.utcnow().timestamp()))
            result = await agent0.process_message(message, context_id)

        elif method == "agent.getTaskStatus":
            task_id = params.get("taskId")
            result = await agent0.get_task_status(task_id)

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
    """REST-style message endpoint for Agent 0"""
    try:
        body = await request.json()
        message = A2AMessage(**body.get("message", {}))
        context_id = body.get("contextId", str(datetime.utcnow().timestamp()))

        result = await agent0.process_message(message, context_id)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status for Agent 0"""
    try:
        status = await agent0.get_task_status(task_id)
        return status
    except Exception as e:
        return JSONResponse(
            status_code=404,
            content={"error": str(e)}
        )


@router.get("/queue/status")
async def get_queue_status():
    """Get message queue status for Agent 0"""
    if agent0 and agent0.message_queue:
        return agent0.message_queue.get_queue_status()
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "Message queue not available"}
        )


@router.get("/queue/messages/{message_id}")
async def get_message_status(message_id: str):
    """Get status of a specific message"""
    if agent0 and agent0.message_queue:
        status = agent0.message_queue.get_message_status(message_id)
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
    if agent0 and agent0.message_queue:
        cancelled = await agent0.message_queue.cancel_message(message_id)
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
    """Health check endpoint for Agent 0"""
    queue_info = {}
    if agent0 and agent0.message_queue:
        queue_status = agent0.message_queue.get_queue_status()
        queue_info = {
            "queue_depth": queue_status["queue_status"]["queue_depth"],
            "processing_count": queue_status["queue_status"]["processing_count"],
            "streaming_enabled": queue_status["capabilities"]["streaming_enabled"],
            "batch_processing_enabled": queue_status["capabilities"]["batch_processing_enabled"]
        }

    return {
        "status": "healthy",
        "agent": "Data Product Registration Agent",
        "version": "2.0.0",
        "protocol_version": "0.2.9",
        "timestamp": datetime.utcnow().isoformat(),
        "message_queue": queue_info
    }
