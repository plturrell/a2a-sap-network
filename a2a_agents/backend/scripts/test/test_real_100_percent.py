#!/usr/bin/env python3
"""
Test REAL 100% functionality - no false claims
Tests what actually works, reports what doesn't
"""

import asyncio
import httpx
import json
import os
from datetime import datetime

TESTS_PASSED = 0
TESTS_FAILED = 0

def test_result(name, passed, details=""):
    global TESTS_PASSED, TESTS_FAILED
    if passed:
        TESTS_PASSED += 1
        print(f"✅ {name}")
        if details:
            print(f"   {details}")
    else:
        TESTS_FAILED += 1
        print(f"❌ {name}")
        if details:
            print(f"   {details}")


async def test_services_running():
    """Test if services are actually running"""
    print("\n1. TESTING SERVICE AVAILABILITY")
    print("=" * 50)
    
    services = [
        ("Registry", "http://localhost:8000/health"),
        ("Agent 0", "http://localhost:8002/health"),
        ("Agent 1", "http://localhost:8001/health")
    ]
    
    all_running = True
    for name, url in services:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    test_result(f"{name} running", True, f"Response: {response.status_code}")
                else:
                    test_result(f"{name} running", False, f"Bad status: {response.status_code}")
                    all_running = False
        except Exception as e:
            test_result(f"{name} running", False, f"Error: {type(e).__name__}")
            all_running = False
    
    return all_running


async def test_agent_registration():
    """Test if agents are registered in registry"""
    print("\n2. TESTING AGENT REGISTRATION")
    print("=" * 50)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/api/v1/a2a/agents/search")
            if response.status_code == 200:
                data = response.json()
                agents = data.get("results", []) or data.get("agents", [])
                
                test_result("Registry query works", True, f"Found {len(agents)} agents")
                
                # Check for specific agents by name
                agent0_found = any(a for a in agents if "data product" in a.get("name", "").lower())
                agent1_found = any(a for a in agents if "standardization" in a.get("name", "").lower())
                
                test_result("Agent 0 registered", agent0_found, 
                          f"ID: {agents[0]['agent_id']}" if agents else "Not found")
                test_result("Agent 1 registered", agent1_found,
                          "Found" if agent1_found else "Not found")
                
                return agent0_found and agent1_found
            else:
                test_result("Registry query works", False, f"Status: {response.status_code}")
                return False
    except Exception as e:
        test_result("Registry query works", False, str(e))
        return False


async def test_agent0_message_endpoint():
    """Test if Agent 0 can receive messages"""
    print("\n3. TESTING AGENT 0 MESSAGE ENDPOINT")
    print("=" * 50)
    
    test_message = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "Test message"}]
        },
        "contextId": "test_100_percent"
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "http://localhost:8002/a2a/agent0/v1/messages",
                json=test_message
            )
            
            if response.status_code == 200:
                result = response.json()
                task_id = result.get("taskId")
                test_result("Agent 0 accepts messages", True, f"Task ID: {task_id}")
                
                # Check task status
                await asyncio.sleep(2)
                status_response = await client.get(
                    f"http://localhost:8002/a2a/agent0/v1/tasks/{task_id}"
                )
                
                if status_response.status_code == 200:
                    status = status_response.json()
                    state = status["status"]["state"]
                    test_result("Agent 0 task tracking works", True, f"State: {state}")
                    return True
                else:
                    test_result("Agent 0 task tracking works", False, 
                              f"Status code: {status_response.status_code}")
                    return False
            else:
                test_result("Agent 0 accepts messages", False, 
                          f"Status: {response.status_code}, Body: {response.text[:100]}")
                return False
    except Exception as e:
        test_result("Agent 0 accepts messages", False, str(e))
        return False


async def test_real_data_exists():
    """Test if real CSV data exists"""
    print("\n4. TESTING REAL DATA FILES")
    print("=" * 50)
    
    data_path = "/Users/apple/projects/finsight_cib/data/raw"
    
    expected_files = [
        "CRD_Extraction_Indexed.csv",
        "CRD_Extraction_v1_account_sorted.csv",
        "CRD_Extraction_v1_book_sorted.csv",
        "CRD_Extraction_v1_location_sorted.csv",
        "CRD_Extraction_v1_measure_sorted.csv",
        "CRD_Extraction_v1_product_sorted.csv"
    ]
    
    all_exist = True
    total_size = 0
    
    for filename in expected_files:
        filepath = os.path.join(data_path, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            total_size += size
            size_mb = size / (1024 * 1024)
            test_result(f"File: {filename}", True, f"Size: {size_mb:.2f} MB")
        else:
            test_result(f"File: {filename}", False, "Not found")
            all_exist = False
    
    test_result("Total data size", True, f"{total_size / (1024 * 1024):.2f} MB")
    return all_exist


async def test_ord_registry():
    """Test ORD Registry functionality"""
    print("\n5. TESTING ORD REGISTRY")
    print("=" * 50)
    
    try:
        async with httpx.AsyncClient() as client:
            # Test health
            response = await client.get("http://localhost:8000/api/v1/ord/health", timeout=60.0)
            if response.status_code == 200:
                health = response.json()
                test_result("ORD Registry healthy", 
                          health["status"] == "healthy",
                          f"Status: {health['status']}")
                
                # Test registration
                test_doc = {
                    "ord_document": {
                        "openResourceDiscovery": "1.5.0",
                        "description": "Test document",
                        "dataProducts": []
                    },
                    "registered_by": "test_script"
                }
                
                reg_response = await client.post(
                    "http://localhost:8000/api/v1/ord/register",
                    json=test_doc,
                    timeout=30.0
                )
                
                if reg_response.status_code == 200:
                    reg_result = reg_response.json()
                    test_result("ORD Registration works", True, 
                              f"ID: {reg_result.get('registration_id')}")
                    return True
                else:
                    test_result("ORD Registration works", False,
                              f"Status: {reg_response.status_code}")
                    return False
            else:
                test_result("ORD Registry healthy", False, 
                          f"Status: {response.status_code}")
                return False
    except Exception as e:
        test_result("ORD Registry test", False, str(e))
        return False


async def test_trust_system():
    """Test trust system functionality"""
    print("\n6. TESTING TRUST SYSTEM")
    print("=" * 50)
    
    try:
        from app.a2a.security.smart_contract_trust import (
            initialize_agent_trust, sign_a2a_message, verify_a2a_message
        )
        
        # Initialize test identity
        identity = initialize_agent_trust("test_real_100", "TestAgent")
        test_result("Trust identity creation", True, f"Agent ID: {identity.agent_id}")
        
        # Sign message
        test_msg = {"test": "data", "timestamp": datetime.utcnow().isoformat()}
        signed = sign_a2a_message("test_real_100", test_msg)
        test_result("Message signing", True, "Signature created")
        
        # Verify message
        verified, info = verify_a2a_message(signed)
        test_result("Message verification", verified, 
                   f"Trust score: {info.get('trust_score', 0)}")
        
        return verified
    except Exception as e:
        test_result("Trust system test", False, str(e))
        return False


async def test_real_data_processing():
    """Test processing real financial data through Agent 0"""
    print("\n7. TESTING REAL DATA PROCESSING")
    print("=" * 50)
    
    message = {
        "message": {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "Process financial data"
                },
                {
                    "kind": "data",
                    "data": {
                        "raw_data_location": "/Users/apple/projects/finsight_cib/data/raw",
                        "processing_stage": "raw_to_ord"
                    }
                }
            ]
        },
        "contextId": f"real_data_test_{datetime.utcnow().isoformat()}"
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Send to Agent 0
            response = await client.post(
                "http://localhost:8002/a2a/agent0/v1/messages",
                json=message
            )
            
            if response.status_code == 200:
                result = response.json()
                task_id = result.get("taskId")
                test_result("Agent 0 accepts data request", True, f"Task: {task_id}")
                
                # Monitor task
                for i in range(30):  # 30 seconds max
                    await asyncio.sleep(1)
                    
                    status_response = await client.get(
                        f"http://localhost:8002/a2a/agent0/v1/tasks/{task_id}"
                    )
                    
                    if status_response.status_code == 200:
                        status = status_response.json()
                        state = status["status"]["state"]
                        
                        if state == "completed":
                            test_result("Data processing completed", True)
                            
                            # Check artifacts
                            artifacts = status.get("artifacts", [])
                            test_result("Artifacts generated", len(artifacts) > 0,
                                      f"Count: {len(artifacts)}")
                            return True
                            
                        elif state == "failed":
                            error = status["status"].get("error", {})
                            test_result("Data processing completed", False,
                                      f"Error: {error.get('message', 'Unknown')}")
                            return False
                
                test_result("Data processing completed", False, "Timeout")
                return False
            else:
                test_result("Agent 0 accepts data request", False,
                          f"Status: {response.status_code}")
                return False
    except Exception as e:
        test_result("Real data processing", False, str(e))
        return False


async def main():
    """Run all tests"""
    print("🔍 TESTING REAL 100% FUNCTIONALITY")
    print("=" * 70)
    print("Testing what actually works, no false claims")
    print("=" * 70)
    
    # Run tests
    services_ok = await test_services_running()
    
    if services_ok:
        await test_agent_registration()
        await test_agent0_message_endpoint()
        await test_real_data_exists()
        await test_ord_registry()
        await test_trust_system()
        await test_real_data_processing()
    else:
        print("\n⚠️  Skipping remaining tests - services not running")
        print("Run: ./start_a2a_services.sh")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Passed: {TESTS_PASSED}")
    print(f"❌ Failed: {TESTS_FAILED}")
    
    success_rate = (TESTS_PASSED / (TESTS_PASSED + TESTS_FAILED) * 100) if (TESTS_PASSED + TESTS_FAILED) > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("\n🎉 100% FUNCTIONALITY VERIFIED!")
    else:
        print(f"\n⚠️  System is {success_rate:.1f}% functional")
        print("Areas needing fixes:")
        if TESTS_FAILED > 0:
            print("- Check failed tests above for details")


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    asyncio.run(main())