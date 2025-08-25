from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import uuid4

from .comprehensiveReasoningAgentSdk import ComprehensiveReasoningAgentSdk

from app.a2a.core.security_base import SecureA2AAgent
"""
Agent 9 (Reasoning Agent) FastAPI Router
Provides HTTP endpoints for reasoning and decision-making operations
"""
# Initialize router with prefix for Agent 9
router = APIRouter(prefix="/a2a/agent9/v1", tags=["Agent 9 - Reasoning"])

# Initialize Agent 9
agent9 = None

@router.on_event("startup")
async def startup_event():
    """Initialize Agent 9 on startup"""
    global agent9
    try:
        agent9 = ComprehensiveReasoningAgentSdk()
        await agent9.initialize()
        print("Agent 9 (Reasoning Agent) initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Agent 9: {str(e)}")
        raise

@router.get("/.well-known/agent.json")
async def get_agent_card():
    """Get the agent card for Agent 9"""
    if agent9:
        return agent9.get_agent_card()
    else:
        raise HTTPException(status_code=503, detail="Agent 9 not initialized")

@router.post("/rpc")
async def json_rpc_handler(request: Request):
    """Handle JSON-RPC 2.0 requests for Agent 9"""
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

        if not agent9:
            return JSONResponse(
                status_code=503,
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": "Agent not initialized"
                    },
                    "id": request_id
                }
            )

        # Route to appropriate handler based on method
        result = await handle_rpc_method(method, params)

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
                    "message": f"Internal error: {str(e)}"
                },
                "id": body.get("id") if 'body' in locals() else None
            }
        )

async def handle_rpc_method(method: str, params: Dict[str, Any]) -> Any:
    """Handle individual RPC methods"""

    # MCP tool methods
    if method == "tools/reason":
        return await agent9.reason(
            query=params.get("query", ""),
            context=params.get("context", {}),
            reasoning_type=params.get("reasoning_type", "deductive")
        )

    elif method == "tools/solve_problem":
        return await agent9.solve_problem(
            problem=params.get("problem", ""),
            constraints=params.get("constraints", []),
            approach=params.get("approach", "analytical")
        )

    elif method == "tools/make_decision":
        return await agent9.make_decision(
            options=params.get("options", []),
            criteria=params.get("criteria", {}),
            context=params.get("context", {})
        )

    elif method == "tools/analyze":
        return await agent9.analyze(
            data=params.get("data", {}),
            analysis_type=params.get("analysis_type", "comprehensive")
        )

    elif method == "tools/validate_logic":
        return await agent9.validate_logic(
            statements=params.get("statements", []),
            rules=params.get("rules", [])
        )

    else:
        raise ValueError(f"Unknown method: {method}")

# REST API Endpoints

@router.post("/reasoning-tasks")
async def create_reasoning_task(task_data: Dict[str, Any]):
    """Create a new reasoning task"""
    try:
        task_id = str(uuid4())
        result = await agent9.create_reasoning_task(
            task_id=task_id,
            task_name=task_data.get("taskName", ""),
            description=task_data.get("description", ""),
            reasoning_type=task_data.get("reasoningType", "deductive"),
            problem_domain=task_data.get("problemDomain", "general"),
            priority=task_data.get("priority", "medium")
        )
        return {"id": task_id, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reasoning-tasks")
async def list_reasoning_tasks(
    status: Optional[str] = None,
    reasoning_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """List reasoning tasks with optional filters"""
    try:
        tasks = await agent9.list_reasoning_tasks(
            status=status,
            reasoning_type=reasoning_type,
            limit=limit,
            offset=offset
        )
        return {"tasks": tasks, "total": len(tasks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reasoning-tasks/{task_id}/start")
async def start_reasoning(task_id: str, configuration: Dict[str, Any]):
    """Start reasoning process for a task"""
    try:
        result = await agent9.start_reasoning(
            task_id=task_id,
            configuration=configuration
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reasoning-tasks/{task_id}/validate")
async def validate_conclusion(task_id: str, validation_params: Dict[str, Any]):
    """Validate reasoning conclusion"""
    try:
        result = await agent9.validate_conclusion(
            task_id=task_id,
            validation_method=validation_params.get("validation_method", "logical")
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reasoning-tasks/{task_id}/explain")
async def explain_reasoning(task_id: str, explanation_params: Dict[str, Any]):
    """Generate explanation for reasoning process"""
    try:
        result = await agent9.explain_reasoning(
            task_id=task_id,
            detail_level=explanation_params.get("detail_level", 3)
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledge-base/add")
async def add_knowledge(knowledge_data: Dict[str, Any]):
    """Add knowledge to the knowledge base"""
    try:
        result = await agent9.add_knowledge(
            element_type=knowledge_data.get("element_type", "fact"),
            content=knowledge_data.get("content", ""),
            domain=knowledge_data.get("domain", "general"),
            confidence_level=knowledge_data.get("confidence_level", 0.8)
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledge-base/validate")
async def validate_knowledge_base(validation_params: Dict[str, Any]):
    """Validate knowledge base consistency"""
    try:
        result = await agent9.validate_knowledge_base(
            domain=validation_params.get("domain", "all")
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reasoning-tasks/{task_id}/generate-inferences")
async def generate_inferences(task_id: str, inference_params: Dict[str, Any]):
    """Generate inferences for a reasoning task"""
    try:
        result = await agent9.generate_inferences(
            task_id=task_id,
            inference_types=inference_params.get("inference_types", ["deductive"]),
            max_inferences=inference_params.get("max_inferences", 10)
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reasoning-tasks/{task_id}/make-decision")
async def make_decision(task_id: str, decision_params: Dict[str, Any]):
    """Make a decision based on reasoning"""
    try:
        result = await agent9.make_decision(
            task_id=task_id,
            decision_criteria=decision_params.get("decision_criteria", ""),
            alternatives=decision_params.get("alternatives", "")
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/problems/solve")
async def solve_problem(problem_data: Dict[str, Any]):
    """Solve a problem using reasoning"""
    try:
        result = await agent9.solve_problem(
            problem_description=problem_data.get("problem_description", ""),
            problem_type=problem_data.get("problem_type", "general"),
            solving_strategy=problem_data.get("solving_strategy", "analytical"),
            constraints=problem_data.get("constraints", "")
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard")
async def get_dashboard_data(time_range: str = "24h"):
    """Get reasoning dashboard data"""
    try:
        result = await agent9.get_dashboard_data(time_range=time_range)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reasoning-options")
async def get_reasoning_options():
    """Get available reasoning options"""
    try:
        result = await agent9.get_reasoning_options()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if agent9 else "unhealthy",
        "agent": "reasoning-agent",
        "agent_id": 9,
        "timestamp": datetime.utcnow().isoformat()
    }

# Export router
__all__ = ['router']
