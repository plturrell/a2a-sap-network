from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from uuid import uuid4

from app.dependencies import get_current_user
from .enhancedCalculationAgentSdk import EnhancedCalculationAgentSDK
from app.a2a.sdk import A2AMessage, MessagePart, MessageRole


from app.a2a.core.security_base import SecureA2AAgent
"""
Calculation Agent Router
Provides REST API endpoints for the Calculation Agent
"""

# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(
    prefix="/api/v1/calculation",
    tags=["calculation"],
    responses={404: {"description": "Not found"}}
)

# Initialize agent (in production, this would be dependency injected)
calculation_agent = None

async def get_calculation_agent() -> EnhancedCalculationAgentSDK:
    """Get or initialize the calculation agent"""
    global calculation_agent
    if calculation_agent is None:
        calculation_agent = EnhancedCalculationAgentSDK(
            base_url=os.getenv("A2A_SERVICE_URL"),
            enable_monitoring=True,
            enable_ray=True
        )
        await calculation_agent.initialize()
    return calculation_agent


@router.get("/")
async def calculation_agent_info():
    """Get calculation agent information"""
    agent = await get_calculation_agent()
    return {
        "agent_id": agent.agent_id,
        "name": agent.name,
        "description": agent.description,
        "version": agent.version,
        "capabilities": {
            "sympy": agent.skills.get("evaluate_calculation") is not None,
            "quantlib": agent.skills.get("quantlib_bond_pricing") is not None,
            "networkx": agent.skills.get("networkx_graph_analysis") is not None,
            "distributed": agent.enable_ray,
            "ai_assisted": agent.grok_client is not None
        },
        "available_skills": agent.list_skills()
    }


@router.post("/calculate")
async def perform_calculation(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """Perform a calculation with intelligent routing support"""
    try:
        agent = await get_calculation_agent()
        
        # Create A2A message
        message = A2AMessage(
            messageId=str(uuid4()),
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data=request
                )
            ],
            metadata={
                "user_id": current_user.get("id"),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Process calculation
        context_id = str(uuid4())
        result = await agent.handle_calculation_request(message, context_id)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/natural-language")
async def natural_language_calculation(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    current_user: Dict = Depends(get_current_user)
):
    """Process natural language calculation requests"""
    try:
        agent = await get_calculation_agent()
        
        result = await agent.natural_language_calculation({
            "query": query,
            "context": context or {}
        })
        
        return {
            "success": True,
            "result": result,
            "user_id": current_user.get("id")
        }
        
    except Exception as e:
        logger.error(f"Natural language calculation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/intelligent-dispatch")
async def intelligent_dispatch(
    request: str,
    context: Optional[Dict[str, Any]] = None,
    execute_skill: bool = True,
    current_user: Dict = Depends(get_current_user)
):
    """Intelligently analyze and dispatch calculation requests"""
    try:
        agent = await get_calculation_agent()
        
        dispatch_data = {
            "request": request,
            "context": context or {}
        }
        
        if execute_skill:
            # Full dispatch with skill execution
            result = await agent.intelligent_dispatch(dispatch_data)
        else:
            # Just analyze without execution
            if agent.intelligent_dispatcher:
                result = await agent.intelligent_dispatcher.analyze_and_dispatch(request, context)
            else:
                raise HTTPException(status_code=503, detail="Intelligent dispatcher not available")
        
        return {
            "success": True,
            "result": result,
            "executed": execute_skill,
            "user_id": current_user.get("id")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intelligent dispatch failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/evaluate")
async def evaluate_expression(
    expression: str,
    variables: Optional[Dict[str, float]] = None,
    current_user: Dict = Depends(get_current_user)
):
    """Evaluate a mathematical expression"""
    try:
        agent = await get_calculation_agent()
        
        result = await agent.evaluate_expression({
            "expression": expression,
            "variables": variables or {}
        })
        
        return {
            "success": True,
            "result": result,
            "user_id": current_user.get("id")
        }
        
    except Exception as e:
        logger.error(f"Expression evaluation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/differentiate")
async def differentiate_expression(
    expression: str,
    variable: str = "x",
    order: int = 1,
    current_user: Dict = Depends(get_current_user)
):
    """Differentiate a mathematical expression"""
    try:
        agent = await get_calculation_agent()
        
        result = await agent.differentiate_expression({
            "expression": expression,
            "variable": variable,
            "order": order
        })
        
        return {
            "success": True,
            "result": result,
            "user_id": current_user.get("id")
        }
        
    except Exception as e:
        logger.error(f"Differentiation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/integrate")
async def integrate_expression(
    expression: str,
    variable: str = "x",
    limits: Optional[List[float]] = None,
    current_user: Dict = Depends(get_current_user)
):
    """Integrate a mathematical expression"""
    try:
        agent = await get_calculation_agent()
        
        result = await agent.integrate_expression({
            "expression": expression,
            "variable": variable,
            "limits": limits
        })
        
        return {
            "success": True,
            "result": result,
            "user_id": current_user.get("id")
        }
        
    except Exception as e:
        logger.error(f"Integration failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/solve")
async def solve_equations(
    equations: List[str],
    variables: Optional[List[str]] = None,
    current_user: Dict = Depends(get_current_user)
):
    """Solve equations"""
    try:
        agent = await get_calculation_agent()
        
        result = await agent.solve_equation({
            "equations": equations,
            "variables": variables or ["x"]
        })
        
        return {
            "success": True,
            "result": result,
            "user_id": current_user.get("id")
        }
        
    except Exception as e:
        logger.error(f"Equation solving failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/financial/bond-pricing")
async def price_bond(
    bond_params: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """Price a bond using QuantLib"""
    try:
        agent = await get_calculation_agent()
        
        result = await agent.price_bond(bond_params)
        
        return {
            "success": True,
            "result": result,
            "user_id": current_user.get("id")
        }
        
    except Exception as e:
        logger.error(f"Bond pricing failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/financial/option-pricing")
async def price_option(
    option_params: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """Price an option using QuantLib"""
    try:
        agent = await get_calculation_agent()
        
        result = await agent.price_option(option_params)
        
        return {
            "success": True,
            "result": result,
            "user_id": current_user.get("id")
        }
        
    except Exception as e:
        logger.error(f"Option pricing failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/graph/analyze")
async def analyze_graph(
    graph_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """Analyze a graph using NetworkX"""
    try:
        agent = await get_calculation_agent()
        
        result = await agent.analyze_graph(graph_data)
        
        return {
            "success": True,
            "result": result,
            "user_id": current_user.get("id")
        }
        
    except Exception as e:
        logger.error(f"Graph analysis failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/graph/shortest-path")
async def find_shortest_path(
    graph_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """Find shortest path in a graph"""
    try:
        agent = await get_calculation_agent()
        
        result = await agent.find_shortest_path(graph_data)
        
        return {
            "success": True,
            "result": result,
            "user_id": current_user.get("id")
        }
        
    except Exception as e:
        logger.error(f"Pathfinding failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/distributed")
async def distributed_calculation(
    calculation_params: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """Perform distributed calculation using Ray"""
    try:
        agent = await get_calculation_agent()
        
        result = await agent.distributed_calculation(calculation_params)
        
        return {
            "success": True,
            "result": result,
            "user_id": current_user.get("id")
        }
        
    except Exception as e:
        logger.error(f"Distributed calculation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ai-assist")
async def ai_assisted_calculation(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    calculation_hint: Optional[str] = None,
    current_user: Dict = Depends(get_current_user)
):
    """Get AI assistance for calculations"""
    try:
        agent = await get_calculation_agent()
        
        result = await agent.ai_assisted_calculation({
            "query": query,
            "context": context or {},
            "calculation_hint": calculation_hint
        })
        
        return {
            "success": True,
            "result": result,
            "user_id": current_user.get("id")
        }
        
    except Exception as e:
        logger.error(f"AI-assisted calculation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/intelligent-dispatch")
async def intelligent_dispatch(
    request: str,
    context: Optional[Dict[str, Any]] = None,
    auto_execute: bool = True,
    current_user: Dict = Depends(get_current_user)
):
    """Intelligently analyze and dispatch calculation request to appropriate skill"""
    try:
        agent = await get_calculation_agent()
        
        result = await agent.intelligent_dispatch_calculation({
            "request": request,
            "context": context or {},
            "auto_execute": auto_execute
        })
        
        return {
            "success": result.get("success", False),
            "result": result,
            "user_id": current_user.get("id")
        }
        
    except Exception as e:
        logger.error(f"Intelligent dispatch failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/multi-step")
async def multi_step_calculation(
    instructions: List[str],
    context: Optional[Dict[str, Any]] = None,
    share_results: bool = True,
    current_user: Dict = Depends(get_current_user)
):
    """Execute multiple calculation steps from natural language instructions"""
    try:
        agent = await get_calculation_agent()
        
        result = await agent.multi_step_calculation({
            "instructions": instructions,
            "context": context or {},
            "share_results": share_results
        })
        
        return {
            "success": result.get("success", False),
            "result": result,
            "user_id": current_user.get("id")
        }
        
    except Exception as e:
        logger.error(f"Multi-step calculation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/workflow/create")
async def create_workflow(
    workflow_definition: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """Create a computation workflow"""
    try:
        agent = await get_calculation_agent()
        
        workflow_id = await agent.orchestrator.create_workflow(workflow_definition)
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "user_id": current_user.get("id")
        }
        
    except Exception as e:
        logger.error(f"Workflow creation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/workflow/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    input_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """Execute a computation workflow"""
    try:
        agent = await get_calculation_agent()
        
        # Execute workflow in background
        background_tasks.add_task(
            agent.orchestrator.execute_workflow,
            workflow_id,
            input_data
        )
        
        return {
            "success": True,
            "message": "Workflow execution started",
            "workflow_id": workflow_id,
            "user_id": current_user.get("id")
        }
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/workflow/{workflow_id}/status")
async def get_workflow_status(
    workflow_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get workflow execution status"""
    try:
        agent = await get_calculation_agent()
        
        if workflow_id not in agent.orchestrator.workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow = agent.orchestrator.workflows[workflow_id]
        
        return {
            "success": True,
            "workflow": workflow,
            "user_id": current_user.get("id")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_calculation_stats(current_user: Dict = Depends(get_current_user)):
    """Get calculation statistics"""
    try:
        agent = await get_calculation_agent()
        
        return {
            "success": True,
            "stats": agent.calculation_stats,
            "health": await agent.get_agent_health(),
            "user_id": current_user.get("id")
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        agent = await get_calculation_agent()
        health = await agent.get_agent_health()
        
        return {
            "status": "healthy",
            "agent_health": health,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )