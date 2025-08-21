"""
A2A Router for Embedding Fine-Tuner Agent
Handles HTTP routing and A2A protocol compliance
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from typing import Any, Dict
import logging
import uuid
from datetime import datetime

from a2aCommon import A2AMessage, MessageRole

logger = logging.getLogger(__name__)


def create_a2a_router(agent):
    """Create FastAPI router for A2A protocol compliance"""
    router = APIRouter(prefix="/a2a", tags=["A2A Protocol"])
    
    @router.post("/message")
    async def handle_a2a_message(request: Request) -> JSONResponse:
        """Handle incoming A2A protocol messages"""
        try:
            # Parse request body
            body = await request.json()
            
            # Create A2A message
            message = A2AMessage(
                message_id=str(uuid.uuid4()),
                sender_id=body.get("sender_id", "unknown"),
                recipient_id=agent.agent_id,
                role=MessageRole(body.get("role", "user")),
                content=body.get("content", {}),
                timestamp=datetime.utcnow(),
                context_id=body.get("context_id", str(uuid.uuid4()))
            )
            
            # Process message
            result = await agent.process_message(message, message.context_id)
            
            return JSONResponse(content=result)
            
        except Exception as e:
            logger.error(f"Error processing A2A message: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "success": False}
            )
    
    @router.get("/capabilities")
    async def get_capabilities():
        """Get agent capabilities and skills"""
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "description": agent.description,
            "version": agent.version,
            "capabilities": {
                "model_fine_tuning": True,
                "embedding_training": True,
                "contrastive_learning": True,
                "model_evaluation": True,
                "batch_processing": True,
                "available_models": list(getattr(agent, 'available_models', {}).keys())
            },
            "skills": [
                {
                    "name": skill.name,
                    "description": skill.description,
                    "input_schema": skill.input_schema,
                    "output_schema": skill.output_schema
                }
                for skill in agent.skills.values()
            ],
            "handlers": list(agent.handlers.keys()),
            "active_training_jobs": len([
                job for job in getattr(agent, 'training_jobs', {}).values()
                if job.status in ["pending", "running"]
            ])
        }
    
    @router.post("/rpc")
    async def handle_rpc(request: Request) -> JSONResponse:
        """Handle JSON-RPC calls for direct skill execution"""
        try:
            body = await request.json()
            method = body.get("method")
            params = body.get("params", {})
            rpc_id = body.get("id", 1)
            
            # Create A2A message for skill execution
            message = A2AMessage(
                message_id=str(uuid.uuid4()),
                sender_id=params.get("sender_id", "rpc_client"),
                recipient_id=agent.agent_id,
                role=MessageRole.USER,
                content={
                    "skill": method,
                    "parameters": params
                },
                timestamp=datetime.utcnow(),
                context_id=params.get("context_id", str(uuid.uuid4()))
            )
            
            # Execute skill
            if method in [skill.name for skill in agent.skills.values()]:
                result = await agent.process_message(message, message.context_id)
                
                return JSONResponse(content={
                    "jsonrpc": "2.0",
                    "result": result,
                    "id": rpc_id
                })
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32601,
                            "message": f"Method '{method}' not found"
                        },
                        "id": rpc_id
                    }
                )
                
        except Exception as e:
            logger.error(f"RPC error: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": str(e)
                    },
                    "id": body.get("id", 1) if 'body' in locals() else 1
                }
            )
    
    return router