#!/usr/bin/env python3
"""
Simple test script to verify Agent 0 and Agent 1 are working
"""

import httpx
import json
from datetime import datetime
import asyncio

# Agent URLs
AGENT_0_URL = "http://localhost:8002"
AGENT_1_URL = "http://localhost:8001"
ORD_REGISTRY_URL = "http://localhost:8000/api/v1/ord"

# A2A endpoints
AGENT_0_MESSAGES = f"{AGENT_0_URL}/a2a/agent0/v1/messages"
AGENT_0_STATUS = f"{AGENT_0_URL}/a2a/agent0/v1/tasks"
AGENT_1_MESSAGES = f"{AGENT_1_URL}/a2a/v1/messages"
AGENT_1_STATUS = f"{AGENT_1_URL}/status"

async def test_agent_health():
    """Test that both agents are healthy"""
    print("\n🔍 Testing Agent Health...")
    
    async with httpx.AsyncClient() as client:
        # Test Agent 0
        try:
            response = await client.get(f"{AGENT_0_URL}/health")
            if response.status_code == 200:
                print(f"✅ Agent 0 is healthy: {response.json()}")
            else:
                print(f"❌ Agent 0 health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Agent 0 is not reachable: {e}")
        
        # Test Agent 1
        try:
            response = await client.get(f"{AGENT_1_URL}/health")
            if response.status_code == 200:
                print(f"✅ Agent 1 is healthy: {response.json()}")
            else:
                print(f"❌ Agent 1 health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Agent 1 is not reachable: {e}")

async def test_agent_0_processing():
    """Test Agent 0 data product registration"""
    print("\n📊 Testing Agent 0 - Data Product Registration...")
    
    message = {
        "message": {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "Register financial data products from raw data"
                },
                {
                    "kind": "data",
                    "data": {
                        "data_location": "/Users/apple/projects/finsight_cib/data/raw",
                        "processing_instructions": {
                            "action": "register_data_products",
                            "data_types": ["account", "location", "product", "book", "measure"]
                        },
                        "create_workflow": True,
                        "workflow_metadata": {
                            "name": "Test Data Registration",
                            "plan_id": "test_registration_plan"
                        }
                    }
                }
            ]
        },
        "contextId": f"test_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(AGENT_0_MESSAGES, json=message)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Agent 0 task created: {result['taskId']}")
                
                # Wait for processing
                print("⏳ Waiting for Agent 0 to process...")
                await asyncio.sleep(10)
                
                # Check status
                status_response = await client.get(f"{AGENT_0_STATUS}/{result['taskId']}")
                if status_response.status_code == 200:
                    status = status_response.json()
                    print(f"📋 Task status: {status['status']['state']}")
                    if status.get('artifacts'):
                        print(f"📦 Artifacts created: {len(status['artifacts'])}")
            else:
                print(f"❌ Agent 0 processing failed: {response.text}")
        except Exception as e:
            print(f"❌ Agent 0 processing error: {e}")

async def test_agent_1_processing():
    """Test Agent 1 data standardization"""
    print("\n🔧 Testing Agent 1 - Data Standardization...")
    
    # First, check if there are data products in ORD
    async with httpx.AsyncClient() as client:
        try:
            # Search for data products
            search_response = await client.post(
                f"{ORD_REGISTRY_URL}/search",
                json={
                    "resource_type": "dataProduct",
                    "tags": ["crd", "raw-data"]
                }
            )
            
            if search_response.status_code == 200:
                products = search_response.json().get("results", [])
                print(f"📚 Found {len(products)} data products in ORD Registry")
                
                if products:
                    # Create message for Agent 1
                    message = {
                        "message": {
                            "role": "user",
                            "parts": [
                                {
                                    "kind": "text",
                                    "text": "Standardize financial data products registered in ORD catalog"
                                },
                                {
                                    "kind": "data",
                                    "data": {
                                        "ord_reference": {
                                            "registry_url": ORD_REGISTRY_URL,
                                            "query_params": {
                                                "tags": ["crd", "raw-data"],
                                                "registered_by": "data_product_agent"
                                            }
                                        }
                                    }
                                }
                            ]
                        },
                        "contextId": f"test_standardization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    }
                    
                    # Send to Agent 1
                    response = await client.post(AGENT_1_MESSAGES, json=message)
                    if response.status_code == 200:
                        result = response.json()
                        print(f"✅ Agent 1 task created: {result['taskId']}")
                        
                        # Wait for processing
                        print("⏳ Waiting for Agent 1 to process...")
                        await asyncio.sleep(10)
                        
                        # Check status
                        status_response = await client.get(f"{AGENT_1_STATUS}/{result['taskId']}")
                        if status_response.status_code == 200:
                            status = status_response.json()
                            print(f"📋 Task status: {status['status']['state']}")
                            if status.get('artifacts'):
                                print(f"📦 Artifacts created: {len(status['artifacts'])}")
                    else:
                        print(f"❌ Agent 1 processing failed: {response.text}")
                else:
                    print("⚠️  No data products found. Run Agent 0 test first.")
            else:
                print(f"❌ Failed to search ORD Registry: {search_response.text}")
        except Exception as e:
            print(f"❌ Agent 1 processing error: {e}")

async def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING AGENT 0 AND AGENT 1")
    print("=" * 60)
    
    # Test health
    await test_agent_health()
    
    # Test Agent 0
    await test_agent_0_processing()
    
    # Test Agent 1
    await test_agent_1_processing()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())