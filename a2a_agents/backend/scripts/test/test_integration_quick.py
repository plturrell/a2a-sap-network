#!/usr/bin/env python3
"""
Quick Integration Tests for A2A System
Focused tests for verifying key integration points
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any

async def test_data_flow():
    """Test data flow from Agent 0 through Agent 5"""
    print("🔄 Testing complete data flow...")
    
    # Test data
    test_data = {
        "title": "Integration Test Dataset",
        "description": "Testing complete A2A pipeline",
        "creator": "Test Suite",
        "type": "Dataset",
        "format": "JSON",
        "data": {
            "records": [
                {"id": 1, "value": 100, "category": "A"},
                {"id": 2, "value": 200, "category": "B"}
            ]
        }
    }
    
    async with aiohttp.ClientSession() as session:
        # Agent 0: Register
        print("  1️⃣ Agent 0 - Registering data product...")
        async with session.post(
            "http://localhost:8003/api/register",
            json=test_data
        ) as response:
            if response.status == 200:
                result = await response.json()
                product_id = result.get("product_id")
                print(f"  ✅ Registered with ID: {product_id}")
            else:
                print(f"  ❌ Registration failed: {response.status}")
                return False
        
        # Agent 1: Standardize
        print("  2️⃣ Agent 1 - Standardizing data...")
        async with session.post(
            "http://localhost:8004/api/standardize",
            json={"product_id": product_id, "target_format": "standard"}
        ) as response:
            if response.status == 200:
                print("  ✅ Data standardized")
            else:
                print(f"  ❌ Standardization failed: {response.status}")
        
        # Agent 2: AI Preparation
        print("  3️⃣ Agent 2 - AI preparation...")
        async with session.post(
            "http://localhost:8005/api/prepare",
            json={"product_id": product_id, "enrichment_type": "semantic"}
        ) as response:
            if response.status == 200:
                print("  ✅ AI preparation complete")
            else:
                print(f"  ❌ AI preparation failed: {response.status}")
        
        # Check Data Manager
        print("  📦 Checking Data Manager storage...")
        async with session.get(
            f"http://localhost:8001/api/data/{product_id}"
        ) as response:
            if response.status == 200:
                print("  ✅ Data stored in Data Manager")
            else:
                print(f"  ❌ Data not found in Data Manager")
        
        # Check Catalog Manager
        print("  📚 Checking Catalog Manager...")
        async with session.post(
            "http://localhost:8002/api/search",
            json={"query": "Integration Test Dataset"}
        ) as response:
            if response.status == 200:
                results = await response.json()
                if results.get("results"):
                    print("  ✅ Product indexed in Catalog Manager")
                else:
                    print("  ⚠️ Product not found in search")
    
    return True


async def test_trust_relationships():
    """Test trust verification between agents"""
    print("\n🔐 Testing trust relationships...")
    
    trust_pairs = [
        ("http://localhost:8003", "http://localhost:8004", "Agent 0 -> Agent 1"),
        ("http://localhost:8004", "http://localhost:8005", "Agent 1 -> Agent 2"),
        ("http://localhost:8001", "http://localhost:8003", "Data Manager -> Agent 0"),
    ]
    
    async with aiohttp.ClientSession() as session:
        for source_url, target_url, description in trust_pairs:
            print(f"  Testing {description}...")
            
            # Get public key from source
            async with session.get(f"{source_url}/trust/public-key") as response:
                if response.status == 200:
                    key_data = await response.json()
                    public_key = key_data.get("public_key")
                    
                    # Verify at target
                    verify_data = {
                        "source_agent": source_url.split(":")[-1],
                        "public_key": public_key
                    }
                    
                    async with session.post(
                        f"{target_url}/trust/verify",
                        json=verify_data
                    ) as verify_response:
                        if verify_response.status == 200:
                            print(f"  ✅ {description}: Trust verified")
                        else:
                            print(f"  ❌ {description}: Trust failed")
                else:
                    print(f"  ❌ {description}: Could not get public key")


async def test_smart_contract_config():
    """Test smart contract configuration"""
    print("\n📜 Testing smart contract configuration...")
    
    expected_bdc_address = "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0"
    agents = [
        ("Agent 0", "http://localhost:8003"),
        ("Agent 1", "http://localhost:8004"),
        ("Agent 2", "http://localhost:8005"),
        ("Agent 3", "http://localhost:8008"),
        ("Agent 4", "http://localhost:8006"),
        ("Agent 5", "http://localhost:8007"),
    ]
    
    async with aiohttp.ClientSession() as session:
        for agent_name, url in agents:
            try:
                async with session.get(f"{url}/api/blockchain/config") as response:
                    if response.status == 200:
                        config = await response.json()
                        bdc_address = config.get("business_data_cloud_address")
                        if bdc_address == expected_bdc_address:
                            print(f"  ✅ {agent_name}: Correct BDC address")
                        else:
                            print(f"  ❌ {agent_name}: Wrong BDC address: {bdc_address}")
                    else:
                        print(f"  ⚠️ {agent_name}: No blockchain config endpoint")
            except:
                print(f"  ⚠️ {agent_name}: Could not check blockchain config")


async def test_circuit_breakers():
    """Test circuit breaker functionality"""
    print("\n🛡️ Testing circuit breakers...")
    
    # Test with invalid data to trigger circuit breaker
    async with aiohttp.ClientSession() as session:
        # Send multiple bad requests to trigger circuit breaker
        for i in range(5):
            try:
                async with session.post(
                    "http://localhost:8003/api/register",
                    json={"invalid": "data"},
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as response:
                    if response.status >= 500:
                        print(f"  Request {i+1}: Service error (expected)")
            except:
                print(f"  Request {i+1}: Timeout/error (circuit breaker may be open)")
        
        # Wait a moment
        await asyncio.sleep(2)
        
        # Try valid request
        async with session.post(
            "http://localhost:8003/api/register",
            json={
                "title": "Valid Product",
                "description": "Testing after circuit breaker",
                "creator": "Test",
                "type": "Dataset"
            }
        ) as response:
            if response.status == 200:
                print("  ✅ Circuit breaker recovered, accepting valid requests")
            else:
                print("  ⚠️ Circuit breaker may still be open")


async def main():
    """Run quick integration tests"""
    print("🧪 A2A Quick Integration Tests")
    print("=" * 50)
    
    # Check if services are running
    print("\n📋 Checking service health...")
    services = {
        "Data Manager": "http://localhost:8001/health",
        "Catalog Manager": "http://localhost:8002/health",
        "Agent 0": "http://localhost:8003/health",
        "Agent 1": "http://localhost:8004/health",
        "Agent 2": "http://localhost:8005/health",
    }
    
    all_healthy = True
    async with aiohttp.ClientSession() as session:
        for name, url in services.items():
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                    if response.status == 200:
                        print(f"  ✅ {name}: Healthy")
                    else:
                        print(f"  ❌ {name}: Unhealthy")
                        all_healthy = False
            except:
                print(f"  ❌ {name}: Not responding")
                all_healthy = False
    
    if not all_healthy:
        print("\n❌ Not all services are healthy. Please start all services first.")
        return False
    
    # Run tests
    await test_data_flow()
    await test_trust_relationships()
    await test_smart_contract_config()
    await test_circuit_breakers()
    
    print("\n✅ Quick integration tests complete!")
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted")
        exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)