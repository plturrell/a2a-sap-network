from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import json
import asyncio
import os
from datetime import datetime

from ..agents.agent1Standardization.active.dataStandardizationAgentSdk import (
    DataStandardizationAgentSDK,
    A2AMessage,
)

router = APIRouter(prefix="/a2a/v1", tags=["A2A Protocol"])

# Initialize agent
# Agent base URL must be configured via environment variable
agent_base_url = os.getenv("A2A_AGENT_BASE_URL")
if not agent_base_url:
    raise ValueError("A2A_AGENT_BASE_URL environment variable is required")

agent = DataStandardizationAgentSDK(base_url=agent_base_url)


@router.get("/.well-known/agent.json")
async def get_agent_card():
    """Get the agent card according to A2A protocol"""
    return await agent.get_agent_card()


@router.post("/rpc")
async def json_rpc_handler(request: Request):
    """Handle JSON-RPC 2.0 requests"""
    try:
        body = await request.json()

        # Validate JSON-RPC structure
        if "jsonrpc" not in body or body["jsonrpc"] != "2.0":
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "error": {"code": -32600, "message": "Invalid Request"},
                    "id": body.get("id"),
                },
            )

        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id")

        # Route to appropriate method
        if method == "agent.getCard":
            result = await agent.get_agent_card()

        elif method == "agent.processMessage":
            message = A2AMessage(**params.get("message", {}))
            context_id = params.get("contextId", str(datetime.utcnow().timestamp()))
            result = await agent.process_message(message, context_id)

        elif method == "agent.getTaskStatus":
            task_id = params.get("taskId")
            result = await agent.get_task_status(task_id)

        elif method == "agent.cancelTask":
            task_id = params.get("taskId")
            await agent.cancel_task(task_id)
            result = {"status": "cancelled"}

        else:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": "Method not found"},
                    "id": request_id,
                },
            )

        return JSONResponse(content={"jsonrpc": "2.0", "result": result, "id": request_id})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": "Internal error", "data": str(e)},
                "id": body.get("id") if "body" in locals() else None,
            },
        )


@router.post("/messages")
async def rest_message_handler(request: Request):
    """REST-style message endpoint"""
    try:
        body = await request.json()
        message = A2AMessage(**body.get("message", {}))
        context_id = body.get("contextId", str(datetime.utcnow().timestamp()))

        result = await agent.process_message(message, context_id)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status"""
    try:
        status = await agent.get_task_status(task_id)
        return status
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a task"""
    try:
        await agent.cancel_task(task_id)
        return {"status": "cancelled", "taskId": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}/integrity")
async def get_integrity_report(task_id: str):
    """Get comprehensive data integrity report for a task"""
    try:
        report = await agent.get_integrity_report(task_id)
        return report
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}/stream")
async def stream_task_updates(task_id: str):
    """Stream task updates using Server-Sent Events"""

    async def event_generator():
        last_event_count = 0

        while True:
            try:
                # Get current task status
                task = agent.tasks.get(task_id)
                if not task:
                    yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
                    break

                # Send new events
                events = task.get("events", [])
                if len(events) > last_event_count:
                    for event in events[last_event_count:]:
                        yield f"data: {json.dumps(event)}\n\n"
                    last_event_count = len(events)

                # Check if task is complete
                status = task["status"]
                if status.state in ["completed", "failed", "canceled"]:
                    yield f"data: {json.dumps({'type': 'complete', 'status': status.dict()})}\n\n"
                    break

                await asyncio.sleep(1)  # Poll every second

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "Financial Data Standardization Agent",
        "version": "1.0.0",
        "protocol_version": "0.2.9",
        "timestamp": datetime.utcnow().isoformat(),
    }
