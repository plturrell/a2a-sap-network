"""
Agent 4 Router - Computation Quality Testing Agent
FastAPI router for HTTP endpoints and A2A protocol integration
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .calc_validation_agent_sdk import (
    CalcValidationAgentSDK, 
    ComputationType, 
    TestMethodology, 
    ServiceType,
    ComputationTestRequest,
    ServiceDiscoveryResult,
    TestExecutionResult,
    QualityReport
)
from app.a2a.core.a2aTypes import A2AMessage, MessagePart, MessageRole
from app.a2a.core.responseMapper import ResponseMapperRegistry as ResponseMapper
# Import trust components from a2aNetwork
try:
    import sys
    sys.path.insert(0, '/Users/apple/projects/a2a/a2aNetwork')
    from trustSystem.smartContractTrust import sign_a2a_message, initialize_agent_trust, verify_a2a_message
except ImportError:
    # Fallback functions
    def sign_a2a_message(*args, **kwargs):
        return {"signature": "mock"}
    def initialize_agent_trust(*args, **kwargs):
        return True
    def verify_a2a_message(*args, **kwargs):
        return True
logger = logging.getLogger(__name__)


class Agent4Card(BaseModel):
    """Agent 4 card for A2A discovery"""
    name: str = "Computation-Quality-Testing-Agent"
    description: str = "A2A compliant agent for dynamic computation quality testing using template-based test generation"
    version: str = "1.0.0"
    protocolVersion: str = "0.2.9"
    provider: Dict[str, str] = {
        "name": "Computation Testing Framework Inc",
        "url": "https://comp-testing-framework.com",
        "contact": "support@comp-testing-framework.com"
    }
    capabilities: Dict[str, bool] = {
        "streaming": True,
        "pushNotifications": True,
        "stateHistory": True,
        "longRunningTasks": True,
        "dynamicTestGeneration": True,
        "templateBasedTesting": True,
        "serviceDiscovery": True,
        "qualityMetrics": True
    }
    defaultInputModes: List[str] = ["application/json"]
    defaultOutputModes: List[str] = ["application/json", "text/plain"]
    skills: List[Dict[str, Any]] = [
        {
            "id": "dynamic_computation_testing",
            "name": "Dynamic Computation Testing",
            "description": "Generate template-based computation quality tests",
            "inputModes": ["application/json"],
            "outputModes": ["application/json", "text/plain"],
            "parameters": {
                "service_endpoints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of computational service endpoints"
                },
                "test_methodology": {
                    "type": "string",
                    "enum": ["accuracy", "performance", "stress", "comprehensive"],
                    "default": "comprehensive"
                },
                "computation_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["mathematical", "logical", "transformational"]
                }
            }
        },
        {
            "id": "service_discovery",
            "name": "Computational Service Discovery",
            "description": "Discover and analyze computational services",
            "parameters": {
                "domain_filter": {"type": "string"},
                "service_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "enum": ["api", "function", "algorithm", "pipeline"]
                }
            }
        }
    ]


class TestingTaskRequest(BaseModel):
    """HTTP request for testing task"""
    service_endpoints: List[str] = Field(description="Service endpoints to test")
    test_methodology: TestMethodology = TestMethodology.COMPREHENSIVE
    computation_types: List[ComputationType] = [ComputationType.MATHEMATICAL, ComputationType.LOGICAL]
    domain_filter: Optional[str] = None
    service_types: List[ServiceType] = [ServiceType.API]
    test_config: Dict[str, Any] = Field(default_factory=dict)
    max_tests_per_service: int = Field(default=20, ge=1, le=100)
    parallel_limit: int = Field(default=10, ge=1, le=50)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    quality_thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "accuracy": 0.99,
        "performance": 0.95,
        "reliability": 0.98
    })


class ServiceDiscoveryRequest(BaseModel):
    """HTTP request for service discovery"""
    domain_filter: Optional[str] = None
    service_types: List[ServiceType] = [ServiceType.API]
    endpoint_hints: List[str] = Field(default_factory=list)


class TaskStatusResponse(BaseModel):
    """Task status response"""
    task_id: str
    status: str
    progress: float
    message: str
    estimated_completion: Optional[datetime] = None
    current_stage: str
    results: Optional[Dict[str, Any]] = None


# Global agent instance
agent_instance: Optional[CalcValidationAgentSDK] = None


def get_agent() -> CalcValidationAgentSDK:
    """Get the global agent instance"""
    global agent_instance
    if agent_instance is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    return agent_instance


async def initialize_agent(
    base_url: str, 
    template_repository_url: Optional[str] = None,
    data_manager_url: Optional[str] = None,
    catalog_manager_url: Optional[str] = None
) -> CalcValidationAgentSDK:
    """Initialize the agent instance with A2A integration"""
    global agent_instance
    try:
        agent_instance = CalcValidationAgentSDK(
            base_url=base_url,
            template_repository_url=template_repository_url,
            data_manager_url=data_manager_url,
            catalog_manager_url=catalog_manager_url
        )
        await agent_instance.initialize()
        logger.info("✅ Computation Quality Testing Agent initialized successfully with A2A integration")
        logger.info(f"   Data Manager: {data_manager_url}")
        logger.info(f"   Catalog Manager: {catalog_manager_url}")
        return agent_instance
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {e}")
        raise


# Create the router
router = APIRouter(prefix="/agent4", tags=["Agent 4: Computation Quality Testing"])


@router.get("/", response_model=Agent4Card)
async def get_agent_card():
    """Get the A2A agent card for service discovery"""
    return Agent4Card()


@router.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    try:
        agent = get_agent()
        
        return {
            "status": "healthy",
            "agent": {
                "id": agent.agent_id,
                "name": agent.name,
                "version": agent.version,
                "uptime": str(datetime.utcnow() - agent.startup_time) if hasattr(agent, 'startup_time') else "unknown"
            },
            "services": {
                "a2a_protocol": "healthy",
                "service_discovery": "healthy",
                "template_engine": "healthy",
                "test_executor": "healthy"
            },
            "metrics": {
                "total_tasks": agent.processing_stats.get("total_tasks", 0),
                "services_discovered": agent.processing_stats.get("services_discovered", 0),
                "tests_generated": agent.processing_stats.get("tests_generated", 0),
                "tests_executed": agent.processing_stats.get("tests_executed", 0),
                "templates_loaded": len(agent.test_templates),
                "discovered_services": len(agent.discovered_services)
            },
            "service_registry": {
                "total_discovered": len(agent.discovered_services),
                "available": len([s for s in agent.discovered_services.values() if s.metadata.get("health_status") == "healthy"]),
                "last_discovery": max([s.metadata.get("discovered_at", "1970-01-01T00:00:00Z") 
                                     for s in agent.discovered_services.values()], default="never")
            },
            "a2a_compliance": {
                "protocol_version": "0.2.9",
                "capabilities": ["streaming", "longRunningTasks", "dynamicTestGeneration"],
                "agent_card_published": True,
                "trust_system_enabled": agent.trust_identity is not None
            },
            "circuit_breakers": agent.circuit_breaker_manager.get_all_stats() if hasattr(agent, 'circuit_breaker_manager') else {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.post("/a2a/tasks", response_model=Dict[str, Any])
async def execute_a2a_task(
    background_tasks: BackgroundTasks,
    method: str = Body(...),
    params: Dict[str, Any] = Body(...)
):
    """Execute A2A task using JSON-RPC 2.0 format"""
    try:
        agent = get_agent()
        task_id = params.get("taskId", str(uuid.uuid4()))
        context_id = params.get("contextId", str(uuid.uuid4()))
        skill = params.get("skill")
        parameters = params.get("parameters", {})
        
        logger.info(f"Executing A2A task {task_id} with skill {skill}")
        
        # Create A2A message
        message = A2AMessage(
            role=MessageRole.USER,
            taskId=task_id,
            contextId=context_id,
            parts=[
                MessagePart(
                    kind="data",
                    data=parameters
                )
            ]
        )
        
        if skill == "dynamic_computation_testing":
            result = await agent.handle_computation_testing(message)
        elif skill == "service_discovery":
            result = await agent.handle_service_discovery(message)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown skill: {skill}")
        
        return {
            "id": task_id,
            "result": {
                "status": "completed" if result.get("success", False) else "failed",
                "taskId": task_id,
                "data": result,
                "executionTime": parameters.get("executionTime", "unknown")
            }
        }
        
    except Exception as e:
        logger.error(f"A2A task execution failed: {e}")
        return {
            "id": task_id,
            "error": {
                "code": -32603,
                "message": "Internal error",
                "data": str(e)
            }
        }


@router.post("/testing/execute", response_model=Dict[str, Any])
async def execute_computation_testing(
    request: TestingTaskRequest,
    background_tasks: BackgroundTasks
):
    """Execute computation quality testing workflow"""
    try:
        agent = get_agent()
        
        # Create test request
        test_request = ComputationTestRequest(
            service_endpoints=request.service_endpoints,
            test_methodology=request.test_methodology,
            computation_types=request.computation_types,
            domain_filter=request.domain_filter,
            service_types=request.service_types,
            test_config={
                "max_tests_per_service": request.max_tests_per_service,
                "parallel_limit": request.parallel_limit,
                "timeout_seconds": request.timeout_seconds,
                "quality_thresholds": request.quality_thresholds,
                **request.test_config
            }
        )
        
        # Execute testing workflow
        context_id = str(uuid.uuid4())
        result = await agent.execute_computation_testing(
            request_data=test_request.dict(),
            context_id=context_id
        )
        
        return {
            "task_id": context_id,
            "success": result.get("success", False),
            "context_id": context_id,
            "summary": {
                "discovered_services": result.get("discovered_services", 0),
                "generated_tests": result.get("generated_tests", 0),
                "executed_tests": result.get("executed_tests", 0),
                "service_reports": len(result.get("service_reports", []))
            },
            "service_reports": result.get("service_reports", []),
            "processing_stats": result.get("processing_stats", {}),
            "error": result.get("error") if not result.get("success") else None
        }
        
    except Exception as e:
        logger.error(f"Computation testing execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/discovery/services", response_model=Dict[str, Any])
async def discover_services(request: ServiceDiscoveryRequest):
    """Discover computational services"""
    try:
        agent = get_agent()
        
        # Perform service discovery
        discovery_result = await agent.discover_computational_services(
            domain_filter=request.domain_filter,
            service_types=request.service_types
        )
        
        # If endpoint hints provided, also discover those
        if request.endpoint_hints:
            discovered_services = await agent.execute_skill(
                "service_discovery",
                request.endpoint_hints,
                request.domain_filter
            )
            
            discovery_result["newly_discovered"] = [s.dict() for s in discovered_services]
        
        return discovery_result
        
    except Exception as e:
        logger.error(f"Service discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services", response_model=List[Dict[str, Any]])
async def list_discovered_services(
    service_type: Optional[ServiceType] = Query(None),
    domain_filter: Optional[str] = Query(None)
):
    """List discovered computational services"""
    try:
        agent = get_agent()
        
        services = []
        for service in agent.discovered_services.values():
            # Apply filters
            if service_type and service.service_type != service_type:
                continue
            if domain_filter and domain_filter.lower() not in service.endpoint_url.lower():
                continue
            
            services.append(service.dict())
        
        return services
        
    except Exception as e:
        logger.error(f"Failed to list services: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates", response_model=Dict[str, Any])
async def list_test_templates(
    computation_type: Optional[ComputationType] = Query(None),
    difficulty: Optional[str] = Query(None)
):
    """List available test templates"""
    try:
        agent = get_agent()
        
        templates = {}
        for template_id, template in agent.test_templates.items():
            # Apply filters
            if computation_type and template.computation_type != computation_type:
                continue
            if difficulty and template.complexity_level.value != difficulty:
                continue
            
            templates[template_id] = template.dict()
        
        return {
            "templates": templates,
            "total_count": len(templates),
            "by_computation_type": {
                comp_type.value: len([t for t in agent.test_templates.values() 
                                    if t.computation_type == comp_type])
                for comp_type in ComputationType
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=Dict[str, Any])
async def get_agent_metrics():
    """Get agent performance metrics"""
    try:
        agent = get_agent()
        
        return {
            "processing_stats": agent.processing_stats,
            "discovered_services": len(agent.discovered_services),
            "loaded_templates": len(agent.test_templates),
            "circuit_breakers": agent.circuit_breaker_manager.get_all_stats() if hasattr(agent, 'circuit_breaker_manager') else {},
            "trust_system": {
                "enabled": agent.trust_identity is not None,
                "trusted_agents": list(agent.trusted_agents) if hasattr(agent, 'trusted_agents') else []
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/services/{service_id}/test", response_model=Dict[str, Any])
async def test_single_service(
    service_id: str,
    computation_types: List[ComputationType] = Body(...),
    max_tests: int = Body(10, ge=1, le=50),
    timeout_seconds: float = Body(30.0, ge=1.0, le=300.0)
):
    """Test a single discovered service"""
    try:
        agent = get_agent()
        
        # Get the service
        if service_id not in agent.discovered_services:
            raise HTTPException(status_code=404, detail=f"Service {service_id} not found")
        
        service = agent.discovered_services[service_id]
        
        # Load templates for requested computation types
        templates = await agent.execute_skill("template_loading", computation_types)
        
        # Generate test cases
        test_cases = await agent.execute_skill(
            "test_generation",
            [service],
            templates,
            {
                "max_tests_per_template": max_tests,
                "timeout_seconds": timeout_seconds
            }
        )
        
        if not test_cases:
            return {
                "success": False,
                "error": "No test cases generated for service",
                "service_id": service_id
            }
        
        # Execute tests
        test_results = await agent.execute_skill("test_execution", test_cases)
        
        # Analyze quality
        quality_analysis = await agent.execute_skill("quality_analysis", test_results, service_id)
        
        # Generate report
        report = await agent.execute_skill("report_generation", quality_analysis, test_results)
        
        return {
            "success": True,
            "service_id": service_id,
            "test_summary": {
                "total_tests": len(test_cases),
                "executed_tests": len(test_results),
                "passed_tests": sum(1 for r in test_results if r.success)
            },
            "quality_report": report.dict(),
            "test_results": [r.dict() for r in test_results[:10]]  # Limit response size
        }
        
    except Exception as e:
        logger.error(f"Single service testing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Response mapping for different output modes
response_mapper = ResponseMapper()

@response_mapper.register("application/json")
def json_response(data: Any) -> JSONResponse:
    """Return JSON response"""
    return JSONResponse(content=data)

@response_mapper.register("text/plain")
def text_response(data: Any) -> str:
    """Return plain text response"""
    if isinstance(data, dict) and "service_reports" in data:
        # Format quality report as text
        reports = data["service_reports"]
        text_parts = []
        
        for report in reports:
            text_parts.append(f"Service: {report.get('service_id', 'Unknown')}")
            text_parts.append(f"Quality Score: {report.get('quality_scores', {}).get('overall', 0.0):.2f}")
            text_parts.append(f"Tests: {report.get('passed_tests', 0)}/{report.get('total_tests', 0)} passed")
            text_parts.append("---")
        
        return "\n".join(text_parts)
    
    return str(data)


@router.get("/formats/supported")
async def get_supported_formats():
    """Get supported input/output formats"""
    return {
        "input_modes": ["application/json"],
        "output_modes": ["application/json", "text/plain"],
        "streaming": True,
        "websocket": True
    }