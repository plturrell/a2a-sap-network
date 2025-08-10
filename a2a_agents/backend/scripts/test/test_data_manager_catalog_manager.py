#!/usr/bin/env python3
"""
Test script to verify Data Manager Agent and Catalog Manager Agent functionality
"""

import httpx
import json
from datetime import datetime
import asyncio

# Agent URLs
DATA_MANAGER_URL = "http://localhost:8003"
CATALOG_MANAGER_URL = "http://localhost:8005"

async def test_data_manager_agent():
    """Test Data Manager Agent CRUD operations"""
    print("\n📊 Testing Data Manager Agent...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test 1: Create a test file
        print("  📝 Test 1: Creating test data...")
        create_request = {
            "operation": "create",
            "storage_type": "file",
            "service_level": "silver",
            "path": "test/test_data.json",
            "data": {
                "test_id": "dm_test_001",
                "timestamp": datetime.now().isoformat(),
                "message": "Test data from Data Manager Agent test"
            }
        }
        
        try:
            response = await client.post(f"{DATA_MANAGER_URL}/data/crud", json=create_request)
            if response.status_code == 200:
                result = response.json()
                print(f"    ✅ Create operation successful: {result.get('created', {}).get('file', {}).get('path')}")
            else:
                print(f"    ❌ Create operation failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"    ❌ Create operation error: {e}")
        
        # Test 2: Read the test file
        print("  📖 Test 2: Reading test data...")
        read_request = {
            "operation": "read",
            "storage_type": "file",
            "service_level": "silver",
            "path": "test/test_data.json"
        }
        
        try:
            response = await client.post(f"{DATA_MANAGER_URL}/data/crud", json=read_request)
            if response.status_code == 200:
                result = response.json()
                if result.get('data'):
                    print(f"    ✅ Read operation successful: Found test_id = {result['data'].get('test_id')}")
                else:
                    print("    ⚠️  Read operation returned no data")
            else:
                print(f"    ❌ Read operation failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"    ❌ Read operation error: {e}")
        
        # Test 3: List files
        print("  📋 Test 3: Listing files...")
        list_request = {
            "operation": "list",
            "storage_type": "file",
            "service_level": "bronze",
            "path": "test"
        }
        
        try:
            response = await client.post(f"{DATA_MANAGER_URL}/data/crud", json=list_request)
            if response.status_code == 200:
                result = response.json()
                items = result.get('items', [])
                print(f"    ✅ List operation successful: Found {len(items)} items")
                for item in items[:3]:  # Show first 3 items
                    print(f"      - {item.get('path')} ({item.get('size')} bytes)")
            else:
                print(f"    ❌ List operation failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"    ❌ List operation error: {e}")
        
        # Test 4: Check if file exists
        print("  🔍 Test 4: Checking file existence...")
        exists_request = {
            "operation": "exists",
            "storage_type": "file",
            "path": "test/test_data.json"
        }
        
        try:
            response = await client.post(f"{DATA_MANAGER_URL}/data/crud", json=exists_request)
            if response.status_code == 200:
                result = response.json()
                exists = result.get('exists', False)
                locations = result.get('locations', [])
                print(f"    ✅ Exists check successful: File exists = {exists}, Locations = {len(locations)}")
            else:
                print(f"    ❌ Exists operation failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"    ❌ Exists operation error: {e}")

async def test_catalog_manager_agent():
    """Test Catalog Manager Agent ORD operations"""
    print("\n🗂️  Testing Catalog Manager Agent...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test 1: Search ORD repository
        print("  🔍 Test 1: Searching ORD repository...")
        search_message = {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "search for financial data products"
                }
            ],
            "ai_powered": True
        }
        
        try:
            response = await client.post(f"{CATALOG_MANAGER_URL}/a2a/v1/messages", json=search_message)
            if response.status_code == 200:
                result = response.json()
                print(f"    ✅ Search request accepted: Task ID = {result.get('taskId')}")
                
                # Wait a bit and check status
                await asyncio.sleep(3)
                task_id = result.get('taskId')
                if task_id:
                    status_response = await client.get(f"{CATALOG_MANAGER_URL}/a2a/v1/tasks/{task_id}/status")
                    if status_response.status_code == 200:
                        status = status_response.json()
                        print(f"    📋 Search status: {status.get('state', 'unknown')}")
            else:
                print(f"    ❌ Search request failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"    ❌ Search request error: {e}")
        
        # Test 2: Quality assessment
        print("  📊 Test 2: Requesting quality assessment...")
        quality_message = {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "perform quality assessment on ORD documents"
                }
            ],
            "ai_powered": True
        }
        
        try:
            response = await client.post(f"{CATALOG_MANAGER_URL}/a2a/v1/messages", json=quality_message)
            if response.status_code == 200:
                result = response.json()
                print(f"    ✅ Quality assessment request accepted: Task ID = {result.get('taskId')}")
            else:
                print(f"    ❌ Quality assessment failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"    ❌ Quality assessment error: {e}")
        
        # Test 3: Get agent capabilities
        print("  🔧 Test 3: Getting agent capabilities...")
        try:
            response = await client.get(f"{CATALOG_MANAGER_URL}/.well-known/agent.json")
            if response.status_code == 200:
                agent_card = response.json()
                capabilities = agent_card.get('capabilities', {})
                skills = agent_card.get('skills', [])
                print(f"    ✅ Agent card retrieved:")
                print(f"      - AI Enhancement: {capabilities.get('aiEnhancement', False)}")
                print(f"      - Dublin Core: {capabilities.get('dublinCoreEnrichment', False)}")
                print(f"      - Skills: {len(skills)} available")
            else:
                print(f"    ❌ Agent card failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"    ❌ Agent card error: {e}")

async def test_integration():
    """Test integration between Data Manager and Catalog Manager"""
    print("\n🔄 Testing Integration Between Agents...")
    
    # This would test scenarios where:
    # 1. Data Manager stores data
    # 2. Catalog Manager registers metadata about that data
    # 3. Agents communicate through A2A protocol
    
    print("  📋 Integration tests would include:")
    print("    - Data Manager storing ORD documents")
    print("    - Catalog Manager retrieving stored data")
    print("    - A2A message passing between agents")
    print("    - Workflow coordination")
    print("  ⚠️  Full integration tests require workflow orchestration")

async def main():
    """Run all tests"""
    print("=" * 70)
    print("TESTING DATA MANAGER AND CATALOG MANAGER AGENTS")
    print("=" * 70)
    
    # Test Data Manager
    await test_data_manager_agent()
    
    # Test Catalog Manager
    await test_catalog_manager_agent()
    
    # Test Integration
    await test_integration()
    
    print("\n" + "=" * 70)
    print("AGENT TESTING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())