"""
Health check endpoint for A2A Platform
"""
from fastapi import APIRouter
from datetime import datetime
import os
import psutil

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    try:
        # Get basic system info
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "a2a-platform",
            "environment": os.getenv("A2A_ENVIRONMENT", "unknown"),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available // (1024 * 1024)
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "A2A Platform",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "agents": "/api/agents"
        }
    }