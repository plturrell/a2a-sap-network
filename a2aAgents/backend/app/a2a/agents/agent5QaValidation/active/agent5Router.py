import os
from app.a2a.core.security_base import SecureA2AAgent
"""
FastAPI Router for Agent 5 (QA Validation Agent)
Handles HTTP endpoints and WebSocket streaming for ORD-integrated factuality testing
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .qaValidationAgentSdk import (
    QaValidationAgentSDK as QAValidationAgentSDK,
    QAValidationResult
)

# Define missing classes locally
class QAValidationRequest(BaseModel):
    data_product_id: str
    test_type: str
    parameters: Dict[str, Any] = {}

from enum import Enum


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
class TestMethodology(str, Enum):
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    STATISTICAL = "statistical"
    SIMPLEQA = "simpleqa"

class ResourceType(str, Enum):
    FILE = "file"
    DATABASE = "database"
    API = "api"
    DATA_PRODUCTS = "data_products"
    APIS = "apis"

class MetadataSource(str, Enum):
    MANUAL = "manual"
    EXTRACTED = "extracted"
    GENERATED = "generated"

logger = logging.getLogger(__name__)

# Global agent instance
agent: Optional[QAValidationAgentSDK] = None

# API Models
class QATaskRequest(BaseModel):
    """Request model for QA validation task"""
    ord_endpoints: List[str] = Field(description="List of ORD registry endpoints")
    namespace_filter: Optional[str] = Field(None, description="Namespace filter pattern")
    resource_types: List[ResourceType] = Field(
        default=[ResourceType.DATA_PRODUCTS, ResourceType.APIS],
        description="Types of resources to process"
    )
    test_methodology: TestMethodology = Field(
        default=TestMethodology.SIMPLEQA,
        description="Test generation methodology"
    )
    test_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_tests_per_product": 50,
            "difficulty_distribution": {
                "easy": 0.4,
                "medium": 0.4,
                "hard": 0.2
            },
            "coverage_threshold": 0.8
        },
        description="Test configuration parameters"
    )


class ORDDiscoveryRequest(BaseModel):
    """Request model for ORD discovery"""
    ord_endpoints: List[str] = Field(description="ORD registry endpoints")
    namespace_filter: Optional[str] = Field(None, description="Namespace filter")
    resource_types: List[ResourceType] = Field(
        default=[ResourceType.DATA_PRODUCTS],
        description="Resource types to discover"
    )


class TaskStatusResponse(BaseModel):
    """Task status response"""
    taskId: str
    status: str
    progress: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None


class A2ATaskExecutionRequest(BaseModel):
    """A2A JSON-RPC 2.0 task execution request"""
    method: str = Field(default="executeTask")
    params: Dict[str, Any]
    id: Optional[str] = None


def get_agent() -> QAValidationAgentSDK:
    """Dependency to get agent instance"""
    global agent
    if agent is None:
        raise HTTPException(status_code=503, detail="QA Validation Agent not initialized")
    return agent


async def initialize_agent(**kwargs) -> QAValidationAgentSDK:
    """Initialize the agent instance"""
    global agent
    if agent is None:
        agent = QAValidationAgentSDK(**kwargs)
        await agent.initialize()
    return agent


# Create router
router = APIRouter(prefix="/agent5", tags=["Agent 5 - QA Validation"])


@router.post("/initialize")
async def initialize_agent_endpoint(
    base_url: str = os.getenv("A2A_SERVICE_URL"),
    data_manager_url: str = os.getenv("A2A_SERVICE_URL"),
    catalog_manager_url: str = os.getenv("A2A_SERVICE_URL"),
    cache_ttl: int = 3600,
    max_tests_per_product: int = 50
):
    """Initialize the QA Validation Agent with A2A integration"""
    try:
        global agent
        agent = QAValidationAgentSDK(
            base_url=base_url,
            data_manager_url=data_manager_url,
            catalog_manager_url=catalog_manager_url,
            cache_ttl=cache_ttl,
            max_tests_per_product=max_tests_per_product
        )
        
        # Initialize agent
        initialization_result = await agent.initialize()
        
        return {
            "status": "initialized",
            "agent_id": agent.agent_id,
            "name": agent.name,
            "version": agent.version,
            "data_manager_url": data_manager_url,
            "catalog_manager_url": catalog_manager_url,
            "base_url": base_url,
            "initialization_result": initialization_result
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize Agent 5: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    try:
        agent_instance = get_agent()
        
        return {
            "status": "healthy",
            "services": {
                "a2a_protocol": "healthy",
                "data_manager": "healthy",
                "catalog_manager": "healthy", 
                "validation_engine": "healthy"
            },
            "metrics": {
                "active_tasks": len(agent_instance.test_suites),
                "total_tests_generated": sum(
                    len(suite.generated_tests) 
                    for suite in agent_instance.test_suites.values()
                ),
                "websocket_connections": len(agent_instance.websocket_connections),
                "processing_stats": agent_instance.processing_stats
            },
            "a2a_integration": {
                "data_manager_url": agent_instance.data_manager_url,
                "catalog_manager_url": agent_instance.catalog_manager_url,
                "trust_system_enabled": agent_instance.trust_identity is not None
            },
            "a2a_compliance": {
                "protocol_version": "0.2.9",
                "capabilities": ["streaming", "longRunningTasks", "pushNotifications"],
                "agent_card_published": True
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/agent-card")
async def get_agent_card(agent_instance: QAValidationAgentSDK = Depends(get_agent)):
    """Get A2A agent card for service discovery"""
    return agent_instance.get_agent_card()


@router.post("/ord/discover")
async def discover_ord_products(
    request: ORDDiscoveryRequest,
    agent_instance: QAValidationAgentSDK = Depends(get_agent)
):
    """Discover ORD data products from registries"""
    try:
        result = await agent_instance.ord_discovery(
            ord_endpoints=request.ord_endpoints,
            namespace_filter=request.namespace_filter,
            resource_types=request.resource_types
        )
        
        return result
        
    except Exception as e:
        logger.error(f"ORD discovery failed: {e}")
        raise HTTPException(status_code=500, detail=f"ORD discovery failed: {str(e)}")


@router.post("/tests/generate")
async def generate_dynamic_tests(
    request: QATaskRequest,
    agent_instance: QAValidationAgentSDK = Depends(get_agent)
):
    """Generate dynamic SimpleQA-style tests from ORD metadata"""
    try:
        # Convert to internal request model
        qa_request = QAValidationRequest(
            ord_endpoints=request.ord_endpoints,
            namespace_filter=request.namespace_filter,
            resource_types=request.resource_types,
            test_methodology=request.test_methodology,
            test_config=request.test_config
        )
        
        result = await agent_instance.dynamic_test_generation(qa_request)
        
        return result
        
    except Exception as e:
        logger.error(f"Dynamic test generation failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Dynamic test generation failed: {str(e)}"
        )


# A2A JSON-RPC 2.0 Endpoints

@router.post("/a2a/tasks")
async def execute_a2a_task(
    request: A2ATaskExecutionRequest,
    agent_instance: QAValidationAgentSDK = Depends(get_agent)
):
    """Execute A2A task with JSON-RPC 2.0 protocol"""
    try:
        # Create A2A message
        from app.a2a.sdk import A2AMessage, MessageRole
        
        a2a_message = A2AMessage(
            role=MessageRole.USER,
            content={
                "method": request.method,
                "params": request.params,
                "id": request.id or f"req_{datetime.utcnow().timestamp()}"
            }
        )
        
        # Execute task
        result = await agent_instance.execute_task(a2a_message)
        
        return result
        
    except Exception as e:
        logger.error(f"A2A task execution failed: {e}")
        return {
            "id": request.id,
            "error": {
                "code": -32603,
                "message": "Internal error",
                "data": str(e)
            }
        }


@router.get("/a2a/tasks/{task_id}/status")
async def get_task_status(
    task_id: str,
    agent_instance: QAValidationAgentSDK = Depends(get_agent)
) -> TaskStatusResponse:
    """Get A2A task status"""
    try:
        status_data = await agent_instance.get_task_status(task_id)
        return TaskStatusResponse(**status_data)
        
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")


@router.get("/a2a/tasks/{task_id}/report")
async def get_task_report(
    task_id: str,
    format: str = Query(default="json", description="Report format: json, html, csv"),
    agent_instance: QAValidationAgentSDK = Depends(get_agent)
):
    """Get comprehensive test report"""
    try:
        if task_id not in agent_instance.test_suites:
            raise HTTPException(status_code=404, detail="Task not found")
        
        test_suite = agent_instance.test_suites[task_id]
        
        if format.lower() == "json":
            return {
                "suite_id": test_suite.suite_id,
                "created_at": test_suite.created_at.isoformat(),
                "configuration": test_suite.configuration.dict(),
                "discovered_products": [
                    {
                        "ord_id": p.ord_id,
                        "title": p.title,
                        "namespace": p.namespace,
                        "dublin_core_elements": len(p.dublin_core),
                        "technical_elements": len(p.technical_metadata),
                        "relationship_elements": len(p.relationships)
                    }
                    for p in test_suite.discovered_products
                ],
                "test_summary": {
                    "total_tests": len(test_suite.generated_tests),
                    "by_difficulty": {
                        "easy": sum(1 for t in test_suite.generated_tests if t.difficulty.value == "easy"),
                        "medium": sum(1 for t in test_suite.generated_tests if t.difficulty.value == "medium"),
                        "hard": sum(1 for t in test_suite.generated_tests if t.difficulty.value == "hard")
                    },
                    "by_type": {
                        "factual": sum(1 for t in test_suite.generated_tests if t.test_type.value == "factual"),
                        "reverse_lookup": sum(1 for t in test_suite.generated_tests if t.test_type.value == "reverse_lookup"),
                        "enumeration": sum(1 for t in test_suite.generated_tests if t.test_type.value == "enumeration"),
                        "relationship": sum(1 for t in test_suite.generated_tests if t.test_type.value == "relationship")
                    },
                    "by_metadata_source": {
                        "dublin_core": sum(1 for t in test_suite.generated_tests if t.metadata_source.startswith("dublin_core")),
                        "technical": sum(1 for t in test_suite.generated_tests if t.metadata_source.startswith("technical")),
                        "relationship": sum(1 for t in test_suite.generated_tests if t.metadata_source.startswith("relationship"))
                    }
                },
                "execution_results": test_suite.execution_results,
                "sample_tests": [
                    {
                        "test_id": t.test_id,
                        "question": t.question,
                        "answer": t.answer,
                        "difficulty": t.difficulty.value,
                        "test_type": t.test_type.value,
                        "metadata_source": t.metadata_source,
                        "confidence": t.confidence
                    }
                    for t in test_suite.generated_tests[:10]  # First 10 tests as samples
                ]
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task report: {str(e)}")


@router.get("/a2a/tasks/{task_id}/partial-report")
async def get_partial_task_report(
    task_id: str,
    agent_instance: QAValidationAgentSDK = Depends(get_agent)
):
    """Get partial test report for ongoing tasks"""
    try:
        if task_id not in agent_instance.test_suites:
            raise HTTPException(status_code=404, detail="Task not found")
        
        test_suite = agent_instance.test_suites[task_id]
        
        return {
            "suite_id": test_suite.suite_id,
            "created_at": test_suite.created_at.isoformat(),
            "status": "in_progress",
            "progress": {
                "discovered_products": len(test_suite.discovered_products),
                "generated_tests": len(test_suite.generated_tests),
                "completed": test_suite.execution_results is not None
            },
            "partial_results": {
                "tests_generated": len(test_suite.generated_tests),
                "latest_tests": [
                    {
                        "question": t.question,
                        "difficulty": t.difficulty.value,
                        "metadata_source": t.metadata_source
                    }
                    for t in test_suite.generated_tests[-5:]  # Last 5 tests
                ]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get partial report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get partial report: {str(e)}")


# WebSocket Streaming Endpoint

@router.websocket("/a2a/stream/{task_id}")
async def websocket_stream_task(
    websocket: WebSocket,
    task_id: str
):
    """WebSocket endpoint for streaming task progress and results"""
    agent_instance = None
    
    try:
        # Get agent instance
        global agent
        if agent is None:
            await websocket.close(code=1003, reason="Agent not initialized")
            return
        
        agent_instance = agent
        await websocket.accept()
        
        # Register WebSocket connection
        await agent_instance.register_websocket_connection(task_id, websocket)
        
        # Send initial connection acknowledgment
        await websocket.send_text(json.dumps({
            "type": "connection",
            "taskId": task_id,
            "data": {
                "message": "WebSocket connection established",
                "timestamp": datetime.utcnow().isoformat()
            }
        }))
        
        # Keep connection alive and handle incoming messages
        try:
            while True:
                # Wait for messages from client (like ping/pong)
                try:
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    
                    # Handle client messages
                    try:
                        data = json.loads(message)
                        
                        if data.get("type") == "ping":
                            await websocket.send_text(json.dumps({
                                "type": "pong",
                                "timestamp": datetime.utcnow().isoformat()
                            }))
                        elif data.get("type") == "status_request":
                            status = await agent_instance.get_task_status(task_id)
                            await websocket.send_text(json.dumps({
                                "type": "status_response",
                                "taskId": task_id,
                                "data": status
                            }))
                            
                    except json.JSONDecodeError:
                        # Ignore invalid JSON
                        pass
                        
                except asyncio.TimeoutError:
                    # Send keep-alive ping
                    await websocket.send_text(json.dumps({
                        "type": "keep_alive",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for task {task_id}")
        except Exception as e:
            logger.error(f"WebSocket error for task {task_id}: {e}")
            await websocket.close(code=1011, reason=f"Internal error: {str(e)}")
            
    except Exception as e:
        logger.error(f"WebSocket setup failed for task {task_id}: {e}")
        if websocket.client_state.name != "DISCONNECTED":
            await websocket.close(code=1011, reason=f"Setup failed: {str(e)}")
    
    finally:
        # Cleanup
        if agent_instance and task_id:
            await agent_instance.unregister_websocket_connection(task_id)


# Additional utility endpoints

@router.get("/templates/question-templates")
async def get_question_templates(
    agent_instance: QAValidationAgentSDK = Depends(get_agent)
):
    """Get available question templates"""
    return {
        "templates": agent_instance.question_templates,
        "categories": list(agent_instance.question_templates.keys()),
        "total_templates": sum(
            len(category.keys()) 
            for category in agent_instance.question_templates.values()
        )
    }


@router.get("/metrics")
async def get_metrics(
    agent_instance: QAValidationAgentSDK = Depends(get_agent)
):
    """Get agent performance metrics"""
    return {
        "active_test_suites": len(agent_instance.test_suites),
        "websocket_connections": len(agent_instance.websocket_connections),
        "processing_stats": agent_instance.processing_stats,
        "total_tests_generated": sum(
            len(suite.generated_tests) 
            for suite in agent_instance.test_suites.values()
        ),
        "circuit_breakers": agent_instance.circuit_breaker_manager.get_all_stats() if hasattr(agent_instance, 'circuit_breaker_manager') else {}
    }


@router.post("/validation/reset")
async def reset_validation_state(
    agent_instance: QAValidationAgentSDK = Depends(get_agent)
):
    """Reset validation state and circuit breakers"""
    
    # Reset circuit breakers
    if hasattr(agent_instance, 'circuit_breaker_manager'):
        agent_instance.circuit_breaker_manager.reset_all()
    
    # Clear completed test suites (keep active ones)
    completed_suites = [
        suite_id for suite_id, suite in agent_instance.test_suites.items()
        if suite.execution_results is not None
    ]
    
    for suite_id in completed_suites:
        del agent_instance.test_suites[suite_id]
    
    return {
        "message": f"Reset validation state and cleared {len(completed_suites)} completed test suites",
        "reset_at": datetime.utcnow().isoformat(),
        "remaining_active_suites": len(agent_instance.test_suites)
    }


@router.post("/shutdown")
async def shutdown_agent(
    agent_instance: QAValidationAgentSDK = Depends(get_agent)
):
    """Gracefully shutdown the agent"""
    try:
        await agent_instance.shutdown()
        
        global agent
        agent = None
        
        return {
            "message": "Agent shutdown successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Agent shutdown failed: {e}")
        raise HTTPException(status_code=500, detail=f"Shutdown failed: {str(e)}")


# Export router for use in main application
__all__ = ["router", "initialize_agent"]