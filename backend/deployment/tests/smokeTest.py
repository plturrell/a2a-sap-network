#!/usr/bin/env python3
"""
Smoke tests for A2A network deployment
Quick validation that all services are responding
"""

import asyncio
import httpx
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get port offset from environment (set by deployment manager)
PORT_OFFSET = int(os.environ.get('PORT_OFFSET', '0'))

# Service endpoints to test
SERVICES = [
    {"name": "Data Product Agent", "port": 8001 + PORT_OFFSET, "endpoint": "/health"},
    {"name": "Data Standardization Agent", "port": 8002 + PORT_OFFSET, "endpoint": "/health"},
    {"name": "AI Preparation Agent", "port": 8003 + PORT_OFFSET, "endpoint": "/health"},
    {"name": "Vector Processing Agent", "port": 8004 + PORT_OFFSET, "endpoint": "/health"},
    {"name": "Catalog Manager Agent", "port": 8005 + PORT_OFFSET, "endpoint": "/health"},
    {"name": "Data Manager Agent", "port": 8006 + PORT_OFFSET, "endpoint": "/health"},
    {"name": "API Gateway", "port": 8080 + PORT_OFFSET, "endpoint": "/health"},
]


async def test_service_health(service):
    """Test if a service is responding to health checks"""
    url = f"http://localhost:{service['port']}{service['endpoint']}"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            
            if response.status_code == 200:
                logger.info(f"✅ {service['name']} is healthy")
                return True
            else:
                logger.error(f"❌ {service['name']} returned status {response.status_code}")
                return False
                
    except Exception as e:
        logger.error(f"❌ {service['name']} failed: {e}")
        return False


async def run_smoke_tests():
    """Run all smoke tests"""
    logger.info(f"🔥 Running smoke tests (PORT_OFFSET={PORT_OFFSET})...")
    
    results = []
    for service in SERVICES:
        result = await test_service_health(service)
        results.append(result)
    
    passed = sum(results)
    total = len(results)
    
    logger.info(f"📊 Smoke test results: {passed}/{total} services healthy")
    
    if passed == total:
        logger.info("🎉 All smoke tests passed!")
        return 0
    else:
        logger.error(f"💥 {total - passed} smoke tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_smoke_tests())
    sys.exit(exit_code)