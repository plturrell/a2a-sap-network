#!/usr/bin/env python3
"""
Test Agent 0 with Realistic Data Path
"""

import asyncio
import httpx
import json
from datetime import datetime

async def test_agent0_realistic():
    """Test Agent 0 with realistic data processing"""
    
    print("=== TESTING AGENT 0 WITH REALISTIC DATA ===")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    
    # Test with actual data path that exists
    agent0_message = {
        "message": {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "Register financial data products"
                },
                {
                    "kind": "data",
                    "data": {
                        "raw_data_location": "/Users/apple/projects/finsight_cib/data/raw",
                        "processing_stage": "raw_to_ord",
                        "target_agents": ["financial-data-standardization-agent"]
                    }
                }
            ]
        },
        "contextId": "realistic_test"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            print("🚀 Triggering Agent 0 (Data Product Registration)...")
            response = await client.post(
                "http://localhost:8000/a2a/agent0/v1/messages",
                json=agent0_message,
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                task_id = result.get("taskId")
                print(f"✅ Agent 0 task started: {task_id}")
                
                # Monitor task status
                print("⏳ Monitoring Agent 0 task progress...")
                for i in range(60):  # Wait up to 60 seconds
                    await asyncio.sleep(1)
                    status_response = await client.get(f"http://localhost:8000/a2a/agent0/v1/tasks/{task_id}")
                    
                    if status_response.status_code == 200:
                        status = status_response.json()
                        state = status["status"]["state"]
                        
                        print(f"⏳ Agent 0 status: {state}")
                        
                        if state == "completed":
                            print(f"✅ Agent 0 completed successfully!")
                            
                            # Check artifacts
                            artifacts = status.get("artifacts", [])
                            print(f"📦 Generated {len(artifacts)} artifacts:")
                            for artifact in artifacts:
                                print(f"   - {artifact.get('name', 'Unknown')}: {artifact.get('type', 'Unknown')}")
                            
                            # Check for downstream trigger
                            for artifact in artifacts:
                                if "downstream_trigger" in str(artifact):
                                    print(f"✅ Agent 1 was triggered via trusted A2A communication")
                            
                            return True
                            
                        elif state == "failed":
                            error = status.get("status", {}).get("error", {})
                            print(f"❌ Agent 0 failed:")
                            print(f"   Code: {error.get('code', 'Unknown')}")
                            print(f"   Message: {error.get('message', 'Unknown error')}")
                            if error.get('traceback'):
                                print(f"   Traceback: {error['traceback']}")
                            return False
                    
                    if i == 59:
                        print("⚠️ Agent 0 timed out")
                        return False
                
            else:
                print(f"❌ Failed to start Agent 0: {response.status_code} - {response.text}")
                return False
    
    except Exception as e:
        print(f"❌ Error testing Agent 0: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_agent0_realistic())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")