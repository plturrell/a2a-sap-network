from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
import json
import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
from datetime import datetime

from .catalogManagerAgentSdk import CatalogManagerAgentSDK as CatalogManagerAgent
from app.a2a.core.a2aTypes import A2AMessage, MessagePart

from app.a2a.core.security_base import SecureA2AAgent
router = APIRouter(prefix="/a2a/catalog_manager/v1", tags=["Catalog Manager - ORD Repository Management"])

# Initialize Catalog Manager Agent
# Note: downstream_agent_url will be set dynamically from launcher
catalog_manager = None


@router.get("/.well-known/agent.json")
async def get_agent_card():
    """Get the agent card for Catalog Manager"""
    return await catalog_manager.get_agent_card()


@router.post("/rpc")
async def json_rpc_handler(request: Request):
    """Handle JSON-RPC 2.0 requests for Catalog Manager"""
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
            result = await catalog_manager.get_agent_card()

        elif method == "agent.processMessage":
            message = A2AMessage(**params.get("message", {}))
            context_id = params.get("contextId", str(datetime.utcnow().timestamp()))
            result = await catalog_manager.process_message(message, context_id)

        elif method == "ord.register":
            ord_document = params.get("ord_document")
            enhancement_type = params.get("enhancement_type", "metadata_enrichment")
            ai_powered = params.get("ai_powered", True)

            message = A2AMessage(
                role="user",
                parts=[
                    MessagePart(
                        kind="text",
                        text=json.dumps({
                            "operation": "register",
                            "ord_document": ord_document,
                            "enhancement_type": enhancement_type,
                            "ai_powered": ai_powered
                        })
                    )
                ]
            )
            result = await catalog_manager.process_message(message)

        elif method == "ord.enhance":
            registration_id = params.get("registration_id")
            enhancement_type = params.get("enhancement_type", "metadata_enrichment")

            message = A2AMessage(
                role="user",
                parts=[
                    MessagePart(
                        kind="text",
                        text=json.dumps({
                            "operation": "enhance",
                            "registration_id": registration_id,
                            "enhancement_type": enhancement_type
                        })
                    )
                ]
            )
            result = await catalog_manager.process_message(message)

        elif method == "ord.search":
            query = params.get("query")
            filters = params.get("filters", {})

            message = A2AMessage(
                role="user",
                parts=[
                    MessagePart(
                        kind="text",
                        text=json.dumps({
                            "operation": "search",
                            "query": query,
                            "context": filters
                        })
                    )
                ]
            )
            result = await catalog_manager.process_message(message)

        elif method == "ord.qualityCheck":
            registration_id = params.get("registration_id")
            assessment_type = params.get("assessment_type", "comprehensive")

            message = A2AMessage(
                role="user",
                parts=[
                    MessagePart(
                        kind="text",
                        text=json.dumps({
                            "operation": "quality_check",
                            "registration_id": registration_id,
                            "context": {"assessment_type": assessment_type}
                        })
                    )
                ]
            )
            result = await catalog_manager.process_message(message)

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

        return {
            "jsonrpc": "2.0",
            "result": result.model_dump() if hasattr(result, 'model_dump') else result,
            "id": request_id
        }

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
                "id": body.get("id") if 'body' in locals() else None
            }
        )


@router.post("/message")
async def rest_message_handler(request: Request):
    """REST-style message endpoint for Catalog Manager"""
    try:
        body = await request.json()
        message_data = body.get("message", {})

        # DEBUG: Log incoming message structure
        print(f"🔍 DEBUG: Incoming request body: {body}")
        print(f"🔍 DEBUG: Message data: {message_data}")
        print(f"🔍 DEBUG: Message data keys: {list(message_data.keys())}")

        # Transform old 'content' format to new 'parts' format if needed
        if "content" in message_data and "parts" not in message_data:
            content = message_data.pop("content")
            message_data["parts"] = [
                MessagePart(
                    kind="text",
                    text=content
                ).model_dump()
            ]
            print(f"🔍 DEBUG: Applied transformation, new message_data: {message_data}")
        else:
            print(f"🔍 DEBUG: No transformation needed or parts already present")

        print(f"🔍 DEBUG: Creating A2AMessage with: {message_data}")
        message = A2AMessage(**message_data)
        print(f"🔍 DEBUG: A2AMessage created successfully: {message.messageId}")
        context_id = body.get("contextId", str(datetime.utcnow().timestamp()))

        result = await catalog_manager.process_message(message, context_id)

        return {
            "status": "success",
            "result": result.model_dump() if hasattr(result, 'model_dump') else result
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status for Catalog Manager"""
    try:
        status = await catalog_manager.get_task_status(task_id)
        if status:
            return status.model_dump()
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Task not found"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.get("/queue/status")
async def get_queue_status():
    """Get message queue status for Catalog Manager"""
    if catalog_manager and catalog_manager.message_queue:
        return catalog_manager.message_queue.get_queue_status()
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "Message queue not available"}
        )


@router.get("/queue/messages/{message_id}")
async def get_message_status(message_id: str):
    """Get status of a specific message"""
    if catalog_manager and catalog_manager.message_queue:
        status = catalog_manager.message_queue.get_message_status(message_id)
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
    if catalog_manager and catalog_manager.message_queue:
        cancelled = await catalog_manager.message_queue.cancel_message(message_id)
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
    """Health check endpoint for Catalog Manager"""
    try:
        queue_info = {}
        if catalog_manager and catalog_manager.message_queue:
            queue_status = catalog_manager.message_queue.get_queue_status()
            queue_info = {
                "queue_depth": queue_status["queue_status"]["queue_depth"],
                "processing_count": queue_status["queue_status"]["processing_count"],
                "streaming_enabled": queue_status["capabilities"]["streaming_enabled"],
                "batch_processing_enabled": queue_status["capabilities"]["batch_processing_enabled"]
            }

        return {
            "status": "healthy",
            "agent": "Catalog Manager Agent",
            "version": "2.0.0",
            "protocol_version": "0.2.9",
            "timestamp": datetime.utcnow().isoformat(),
            "message_queue": queue_info
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


# ORD-specific endpoints
@router.post("/ord/register")
async def register_ord_document(request: Request):
    """Register ORD document with AI enhancement"""
    try:
        body = await request.json()

        # DEBUG: Log incoming request to /ord/register
        print(f"🔍 DEBUG [/ord/register]: Incoming request body: {body}")

        message_data = {
            "role": "user",
            "parts": [
                MessagePart(
                    kind="text",
                    text=json.dumps({
                        "operation": "register",
                        "ord_document": body.get("ord_document"),
                        "enhancement_type": body.get("enhancement_type", "metadata_enrichment"),
                        "ai_powered": body.get("ai_powered", True)
                    })
                ).model_dump()
            ]
        }

        print(f"🔍 DEBUG [/ord/register]: Creating A2AMessage with: {message_data}")
        message = A2AMessage(**message_data)
        print(f"🔍 DEBUG [/ord/register]: A2AMessage created successfully: {message.messageId}")

        result = await catalog_manager.process_message(message)
        return result.model_dump() if hasattr(result, 'model_dump') else result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.post("/ord/enhance/{registration_id}")
async def enhance_ord_document(registration_id: str, request: Request):
    """Enhance existing ORD document with AI"""
    try:
        body = await request.json()

        message = A2AMessage(
            role="user",
            parts=[
                MessagePart(
                    kind="text",
                    text=json.dumps({
                        "operation": "enhance",
                        "registration_id": registration_id,
                        "enhancement_type": body.get("enhancement_type", "metadata_enrichment")
                    })
                )
            ]
        )

        result = await catalog_manager.process_message(message)
        return result.model_dump() if hasattr(result, 'model_dump') else result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.get("/ord/search")
async def search_ord_repository(query: str, semantic: bool = True):
    """Search ORD repository with AI-powered semantic search"""
    try:
        message = A2AMessage(
            role="user",
            parts=[
                MessagePart(
                    kind="text",
                    text=json.dumps({
                        "operation": "search",
                        "query": query,
                        "context": {"semantic": semantic}
                    })
                )
            ]
        )

        result = await catalog_manager.process_message(message)
        return result.model_dump() if hasattr(result, 'model_dump') else result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.get("/ord/quality/{registration_id}")
async def assess_ord_quality(registration_id: str, assessment_type: str = "comprehensive"):
    """Assess quality of ORD document with AI"""
    try:
        message = A2AMessage(
            role="user",
            parts=[
                MessagePart(
                    kind="text",
                    text=json.dumps({
                        "operation": "quality_check",
                        "registration_id": registration_id,
                        "context": {"assessment_type": assessment_type}
                    })
                )
            ]
        )

        result = await catalog_manager.process_message(message)
        return result.model_dump() if hasattr(result, 'model_dump') else result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
