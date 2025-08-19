#!/usr/bin/env python3
"""
Test Data Manager RPC directly
"""

import asyncio
import json
import httpx

async def test_data_manager_rpc():
    """Test Data Manager RPC endpoints directly"""
    print("🧪 Testing Data Manager RPC Endpoints")
    print("=" * 40)
    
    # Test 1: Store data (exact format QC Manager sends)
    print("\n1️⃣ Testing store_data with QC Manager format...")
    store_request = {
        "jsonrpc": "2.0",
        "method": "store_data",
        "params": {
            "data_type": "quality_assessment",
            "data": {
                "assessment_id": "test_rpc_001",
                "timestamp": "2025-08-19T01:00:00",
                "decision": "reject_retry",
                "quality_scores": {"accuracy": 0.0, "precision": 0.0, "performance": 0.0, "reliability": 0.0, "completeness": 0.0, "consistency": 0.67},
                "confidence_level": 0.09,
                "workflow_type": "final_integration_test"
            },
            "agent_id": "quality_control_manager_6",
            "metadata": {
                "source": "quality_control_manager",
                "version": "1.0"
            }
        },
        "id": "store_qa_75d7197d"
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(
                "http://localhost:8001/a2a/data_manager_agent/v1/rpc",
                json=store_request
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Store successful: {json.dumps(result, indent=2)}")
            else:
                print(f"   ❌ Store failed: {response.text}")
        except Exception as e:
            print(f"   ❌ Store error: {e}")
    
    # Test 2: Retrieve data
    print("\n2️⃣ Testing retrieve_data...")
    retrieve_request = {
        "jsonrpc": "2.0",
        "method": "retrieve_data",
        "params": {
            "agent_id": "quality_control_manager_6",
            "data_type": "quality_assessment"
        },
        "id": "test_retrieve_001"
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(
                "http://localhost:8001/a2a/data_manager_agent/v1/rpc",
                json=retrieve_request
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Retrieve successful: {json.dumps(result, indent=2)}")
            else:
                print(f"   ❌ Retrieve failed: {response.text}")
        except Exception as e:
            print(f"   ❌ Retrieve error: {e}")
    
    # Test 3: Check what handlers are available
    print("\n3️⃣ Checking available handlers...")
    try:
        response = await client.get("http://localhost:8001/ready")
        if response.status_code == 200:
            result = response.json()
            print(f"   Capabilities: {result.get('capabilities', 0)}")
            print(f"   Skills: {result.get('skills', 0)}")
    except Exception as e:
        print(f"   ❌ Ready check failed: {e}")

async def main():
    await test_data_manager_rpc()

if __name__ == "__main__":
    asyncio.run(main())