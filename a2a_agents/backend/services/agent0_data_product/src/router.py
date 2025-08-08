"""
A2A Router for Agent 0 - Data Product Registration
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import json
import asyncio
from datetime import datetime

def create_router(agent):
    """Create router with agent instance"""
    router = APIRouter(prefix="/a2a/agent0/v1", tags=["Agent 0 - Data Product Registration"])
    
    @router.get("/.well-known/agent.json")
    async def get_agent_card():
        """Get the agent card for Agent 0"""
        return await agent.get_agent_card()
    
    @router.post("/rpc")
    async def json_rpc_handler(request: Request):
        """Handle JSON-RPC 2.0 requests"""
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
            
            # Process the RPC request
            result = await agent.handle_rpc(body)
            return JSONResponse(content=result)
            
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
        """SSE endpoint for real-time updates"""
        async def generate():
            while True:
                # Get next event from agent
                event = await agent.get_next_event()
                if event:
                    yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(1)
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    return router