#!/usr/bin/env python3
"""
Test complete Agent 0 → Agent 1 → ORD → Data Manager workflow
"""

import asyncio
import httpx
import json
import time
from datetime import datetime
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.a2a.agents.dataStandardizationAgent import A2AMessage, MessageRole


async def test_complete_workflow():
    """Test the complete data pipeline workflow"""
    print("🔍 TESTING COMPLETE A2A WORKFLOW WITH DATA MANAGER")
    print("=" * 70)
    print("Agent 0 → Agent 1 → ORD Registry → Data Manager")
    print("=" * 70)
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # 1. Check all services
        print("\n1. CHECKING SERVICE AVAILABILITY")
        print("=" * 50)
        
        services = [
            ("Registry", "http://localhost:8000/health"),
            ("Agent 0", "http://localhost:8002/health"),
            ("Agent 1", "http://localhost:8001/health"),
            ("Data Manager", "http://localhost:8003/health")
        ]
        
        all_healthy = True
        for name, url in services:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    print(f"✅ {name} is healthy")
                else:
                    print(f"❌ {name} returned {response.status_code}")
                    all_healthy = False
            except Exception as e:
                print(f"❌ {name} is not accessible: {e}")
                all_healthy = False
        
        if not all_healthy:
            print("\n⚠️ Not all services are running!")
            return
        
        # 2. Send data registration request to Agent 0
        print("\n2. SENDING DATA REGISTRATION REQUEST TO AGENT 0")
        print("=" * 50)
        
        context_id = f"test-workflow-{int(time.time())}"
        message = {
            "role": "user",
            "contextId": context_id,
            "parts": [{
                "kind": "text",
                "text": "register data products"
            }]
        }
        
        response = await client.post(
            "http://localhost:8002/a2a/agent0/v1/messages",
            json=message
        )
        
        if response.status_code == 200:
            result = response.json()
            task_id = result.get("taskId")
            print(f"✅ Agent 0 accepted request, task ID: {task_id}")
        else:
            print(f"❌ Agent 0 error: {response.status_code}")
            return
        
        # 3. Wait for Agent 0 to complete
        print("\n3. WAITING FOR AGENT 0 TO COMPLETE")
        print("=" * 50)
        
        for i in range(30):
            response = await client.get(
                f"http://localhost:8002/a2a/agent0/v1/tasks/{task_id}/status"
            )
            if response.status_code == 200:
                status_data = response.json()
                if status_data["status"]:
                    latest_status = status_data["status"][-1]
                    state = latest_status.get("state", "unknown")
                    print(f"   Status: {state}")
                    
                    if state == "completed":
                        print("✅ Agent 0 completed processing")
                        break
                    elif state == "failed":
                        print(f"❌ Agent 0 failed: {latest_status.get('error')}")
                        return
            await asyncio.sleep(2)
        
        # 4. Get Agent 0 artifacts to find ORD registration
        print("\n4. CHECKING AGENT 0 ORD REGISTRATION")
        print("=" * 50)
        
        response = await client.get(
            f"http://localhost:8002/a2a/agent0/v1/tasks/{task_id}/artifacts"
        )
        
        ord_registration_id = None
        if response.status_code == 200:
            artifacts = response.json().get("artifacts", [])
            for artifact in artifacts:
                for part in artifact.get("parts", []):
                    if part.get("kind") == "data":
                        data = part.get("data", {})
                        ord_info = data.get("ord_registration", {})
                        if ord_info.get("registration_id"):
                            ord_registration_id = ord_info["registration_id"]
                            print(f"✅ ORD Registration ID: {ord_registration_id}")
                            break
        
        if not ord_registration_id:
            print("❌ No ORD registration found in Agent 0 artifacts")
            return
        
        # 5. Search ORD Registry for downstream data
        print("\n5. SEARCHING ORD REGISTRY FOR STANDARDIZED DATA")
        print("=" * 50)
        
        # Wait a bit for Agent 1 to process and register
        await asyncio.sleep(5)
        
        response = await client.post(
            "http://localhost:8000/api/v1/ord/search",
            json={
                "tags": ["standardized"],
                "labels": {"upstream_registration": ord_registration_id}
            }
        )
        
        standardized_registration_id = None
        if response.status_code == 200:
            results = response.json()
            if results.get("dataProducts"):
                print(f"✅ Found {len(results['dataProducts'])} standardized data products")
                # Get the registration ID from the first result
                if results.get("registrations"):
                    standardized_registration_id = results["registrations"][0].get("registration_id")
                    print(f"✅ Standardized data registration: {standardized_registration_id}")
            else:
                print("⚠️ No standardized data found yet")
        
        # 6. Check Data Manager processing
        print("\n6. CHECKING DATA MANAGER STORAGE")
        print("=" * 50)
        
        # Wait for Data Manager to process
        await asyncio.sleep(3)
        
        # Search for stored data in ORD
        response = await client.post(
            "http://localhost:8000/api/v1/ord/search",
            json={
                "tags": ["stored", "database"],
                "labels": {"agent": "data_manager"}
            }
        )
        
        if response.status_code == 200:
            results = response.json()
            if results.get("dataProducts"):
                print(f"✅ Found {len(results['dataProducts'])} stored data products")
                for product in results["dataProducts"]:
                    strategies = product.get("accessStrategies", [])
                    for strategy in strategies:
                        if strategy.get("type") == "database":
                            db_type = strategy.get("database")
                            table = strategy.get("table")
                            print(f"   - {table} stored in {db_type}")
            else:
                print("⚠️ No stored data products found")
        
        # 7. Final summary
        print("\n7. WORKFLOW SUMMARY")
        print("=" * 50)
        print(f"✅ Context ID: {context_id}")
        print(f"✅ Agent 0 Task: {task_id}")
        print(f"✅ Raw Data Registration: {ord_registration_id}")
        print(f"✅ Standardized Data Registration: {standardized_registration_id or 'pending'}")
        print(f"✅ Data Storage: {'completed' if results.get('dataProducts') else 'pending'}")
        
        print("\n🎉 COMPLETE WORKFLOW TEST FINISHED!")


if __name__ == "__main__":
    asyncio.run(test_complete_workflow())