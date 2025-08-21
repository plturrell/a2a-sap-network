#!/usr/bin/env python3
"""
Trust System Service Runner
Standalone entry point for the Trust System Service
"""

import asyncio
import logging
import uvicorn
from fastapi import FastAPI
from .service import TrustSystemService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="A2A Trust System Service", version="1.0.0")

# Initialize trust service
trust_service = TrustSystemService()

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "trust-system"}

@app.get("/trust/metrics")
async def get_trust_metrics():
    """Get trust system metrics"""
    return await trust_service.get_trust_metrics()

@app.get("/trust/health")  
async def get_system_health():
    """Get trust system health"""
    return await trust_service.get_system_health()

@app.get("/trust/leaderboard")
async def get_trust_leaderboard():
    """Get trust leaderboard"""
    return await trust_service.get_trust_leaderboard()

def main():
    """Main entry point"""
    logger.info("Starting A2A Trust System Service on port 8020")
    uvicorn.run(app, host="0.0.0.0", port=8020)

if __name__ == "__main__":
    main()