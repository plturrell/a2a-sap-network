from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
# Performance: Consider using asyncio.gather for concurrent operations
from datetime import datetime

from .vectorProcessingAgentSdk import A2AMessage

router = APIRouter(prefix="/a2a/agent3/v1", tags=["Agent 3 - SAP HANA Vector Engine Ingestion"])

# Initialize Agent 3
# Note: agent will be set from launch_agent3.py
agent3 = None


@router.get("/.well-known/agent.json")
async def get_agent_card():
    """Get the agent card for Agent 3"""
    if not agent3:
        return JSONResponse(
            status_code=503,
            content={"error": "Agent 3 not initialized"}
        )
    return await agent3.get_agent_card()


@router.post("/rpc")
async def json_rpc_handler(request: Request):
    """Handle JSON-RPC 2.0 requests for Agent 3"""
    if not agent3:
        return JSONResponse(
            status_code=503,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32000,
                    "message": "Agent 3 not initialized"
                },
                "id": None
            }
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
            result = await agent3.get_agent_card()

        elif method == "agent.processMessage":
            message = A2AMessage(**params.get("message", {}))
            context_id = params.get("contextId", str(datetime.utcnow().timestamp()))
            result = await agent3.process_message(message, context_id)

        elif method == "agent.getTaskStatus":
            task_id = params.get("taskId")
            result = await agent3.get_task_status(task_id)

        elif method == "agent.searchVectors":
            # Vector search specific method
            query = params.get("query", "")
            entity_types = params.get("entityTypes", [])
            options = params.get("options", {})
            result = await agent3.search_vectors(query, entity_types, options)

        elif method == "agent.queryKnowledgeGraph":
            # SPARQL query method
            sparql_query = params.get("sparqlQuery", "")
            result = await agent3.query_knowledge_graph(sparql_query)

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
    """REST-style message endpoint for Agent 3"""
    if not agent3:
        return JSONResponse(
            status_code=503,
            content={"error": "Agent 3 not initialized"}
        )

    try:
        body = await request.json()
        message = A2AMessage(**body.get("message", {}))
        context_id = body.get("contextId", str(datetime.utcnow().timestamp()))

        result = await agent3.process_message(message, context_id)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status for Agent 3"""
    if not agent3:
        return JSONResponse(
            status_code=503,
            content={"error": "Agent 3 not initialized"}
        )

    try:
        status = await agent3.get_task_status(task_id)
        return status
    except Exception as e:
        return JSONResponse(
            status_code=404,
            content={"error": str(e)}
        )


@router.post("/vector/search")
async def vector_search(request: Request):
    """Vector similarity search endpoint"""
    if not agent3:
        return JSONResponse(
            status_code=503,
            content={"error": "Agent 3 not initialized"}
        )

    try:
        body = await request.json()
        query = body.get("query", "")
        entity_types = body.get("entityTypes", [])
        options = body.get("options", {})

        # Check if agent3 has search_vectors method, if not create a basic implementation
        if hasattr(agent3, 'search_vectors'):
            result = await agent3.search_vectors(query, entity_types, options)
        else:
            result = {
                "error": "Vector search not implemented",
                "message": "Vector search functionality requires SAP HANA Cloud connection"
            }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.post("/knowledge-graph/query")
async def sparql_query(request: Request):
    """SPARQL query endpoint for knowledge graph"""
    if not agent3:
        return JSONResponse(
            status_code=503,
            content={"error": "Agent 3 not initialized"}
        )

    try:
        body = await request.json()
        sparql_query = body.get("query", "")

        # Check if agent3 has query_knowledge_graph method
        if hasattr(agent3, 'query_knowledge_graph'):
            result = await agent3.query_knowledge_graph(sparql_query)
        else:
            result = {
                "error": "Knowledge graph query not implemented",
                "message": "SPARQL query functionality requires knowledge graph setup"
            }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.get("/vector/stores")
async def get_vector_stores():
    """Get information about available vector stores"""
    if not agent3:
        return JSONResponse(
            status_code=503,
            content={"error": "Agent 3 not initialized"}
        )

    try:
        if hasattr(agent3, 'vector_stores') and agent3.vector_stores:
            stores_info = {}
            for entity_type, store in agent3.vector_stores.items():
                stores_info[entity_type] = {
                    "entity_type": entity_type,
                    "table_name": getattr(store, 'table_name', f"FINANCIAL_VECTORS_{entity_type.upper()}"),
                    "status": "active" if store else "inactive"
                }
            return JSONResponse(content={"vector_stores": stores_info})
        else:
            return JSONResponse(content={
                "vector_stores": {},
                "message": "No vector stores available. Ensure SAP HANA Cloud is configured."
            })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.get("/knowledge-graph/info")
async def get_knowledge_graph_info():
    """Get information about the knowledge graph"""
    if not agent3:
        return JSONResponse(
            status_code=503,
            content={"error": "Agent 3 not initialized"}
        )

    try:
        if hasattr(agent3, 'knowledge_graph_store') and agent3.knowledge_graph_store:
            return JSONResponse(content={
                "knowledge_graph": {
                    "status": "active",
                    "sparql_endpoint": "/sparql/financial-knowledge-graph",
                    "supported_formats": ["application/sparql-results+json", "text/csv"],
                    "features": ["SELECT", "CONSTRUCT", "ASK", "DESCRIBE"]
                }
            })
        else:
            return JSONResponse(content={
                "knowledge_graph": {
                    "status": "inactive",
                    "message": "Knowledge graph not initialized"
                }
            })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.get("/queue/status")
async def get_queue_status():
    """Get message queue status for Agent 3"""
    if agent3 and agent3.message_queue:
        return agent3.message_queue.get_queue_status()
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "Message queue not available"}
        )


@router.get("/queue/messages/{message_id}")
async def get_message_status(message_id: str):
    """Get status of a specific message"""
    if agent3 and agent3.message_queue:
        status = agent3.message_queue.get_message_status(message_id)
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
    if agent3 and agent3.message_queue:
        cancelled = await agent3.message_queue.cancel_message(message_id)
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
    """Health check endpoint for Agent 3"""
    # Check SAP HANA availability
    hana_status = "unknown"
    try:


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
        hana_status = "available"
    except ImportError:
        hana_status = "not_available"

    if not agent3:
        return {
            "status": "unhealthy",
            "agent": "SAP HANA Vector Engine Ingestion & Knowledge Graph Agent",
            "version": "1.0.0",
            "protocol_version": "0.2.9",
            "timestamp": datetime.utcnow().isoformat(),
            "error": "Agent 3 not initialized",
            "message_queue": {},
            "capabilities": {
                "sap_hana_integration": hana_status,
                "hana_connection": "unknown",
                "vector_stores": 0,
                "knowledge_graph": "inactive"
            }
        }

    queue_info = {}
    if agent3.message_queue:
        queue_status = agent3.message_queue.get_queue_status()
        queue_info = {
            "queue_depth": queue_status["queue_status"]["queue_depth"],
            "processing_count": queue_status["queue_status"]["processing_count"],
            "streaming_enabled": queue_status["capabilities"]["streaming_enabled"],
            "batch_processing_enabled": queue_status["capabilities"]["batch_processing_enabled"]
        }

    # Check HANA connection
    connection_status = "connected" if agent3.hana_connection else "disconnected"

    return {
        "status": "healthy",
        "agent": "SAP HANA Vector Engine Ingestion & Knowledge Graph Agent",
        "version": "1.0.0",
        "protocol_version": "0.2.9",
        "timestamp": datetime.utcnow().isoformat(),
        "message_queue": queue_info,
        "capabilities": {
            "sap_hana_integration": hana_status,
            "hana_connection": connection_status,
            "vector_stores": len(agent3.vector_stores) if hasattr(agent3, 'vector_stores') else 0,
            "knowledge_graph": "active" if hasattr(agent3, 'knowledge_graph_store') and agent3.knowledge_graph_store else "inactive"
        }
    }
