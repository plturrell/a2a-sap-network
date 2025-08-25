"""
A2A Registry Web Service
Provides HTTP endpoints for the A2A Registry
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a2aRegistry.service import A2ARegistryService
from a2aRegistry.models import (
    AgentRegistrationRequest, AgentRegistrationResponse,
    AgentSearchRequest, AgentSearchResponse
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="A2A Registry Service",
    description="Agent registration and discovery service for A2A network",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global registry instance
registry_service = None

@app.on_event("startup")
async def startup_event():
    global registry_service
    registry_service = A2ARegistryService()
    logger.info("A2A Registry Service initialized")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "a2a-registry",
        "version": "1.0.0"
    }

@app.post("/register", response_model=AgentRegistrationResponse)
async def register_agent(request: AgentRegistrationRequest):
    try:
        response = await registry_service.register_agent(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=AgentSearchResponse)
async def search_agents(request: AgentSearchRequest):
    try:
        response = await registry_service.search_agents(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    try:
        agents = []
        for agent_id, record in registry_service.agents.items():
            agents.append({
                "agent_id": agent_id,
                "name": record.agent_card.name,
                "status": record.registration_metadata.status.value
            })
        return {"agents": agents, "total": len(agents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8090))
    uvicorn.run(app, host="0.0.0.0", port=port)
