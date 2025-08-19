#!/usr/bin/env python3
"""
Quick test script for the A2A Financial Data Standardization Agent
"""

import requests
import json
import time
import base64
from datetime import datetime


def test_agent():
    """Run basic tests against the A2A agent"""
    base_url = "http://localhost:8000"
    
    print("Testing A2A Financial Data Standardization Agent")
    print("=" * 50)
    
    # 1. Test Agent Card
    print("\n1. Testing Agent Card...")
    response = requests.get(f"{base_url}/a2a/v1/.well-known/agent.json")
    if response.status_code == 200:
        agent_card = response.json()
        print(f"✓ Agent: {agent_card['name']}")
        print(f"✓ Version: {agent_card['version']}")
        print(f"✓ Skills: {len(agent_card['skills'])} available")
    else:
        print(f"✗ Failed to get agent card: {response.status_code}")
    
    # 2. Test Location Standardization
    print("\n2. Testing Location Standardization...")
    response = requests.post(f"{base_url}/a2a/v1/rpc", json={
        "jsonrpc": "2.0",
        "method": "agent.processMessage",
        "params": {
            "message": {
                "role": "user",
                "parts": [{
                    "kind": "text",
                    "text": "standardize location: NYC, New York, United States"
                }]
            },
            "contextId": f"test-{datetime.now().isoformat()}"
        },
        "id": 1
    })
    
    if response.status_code == 200:
        result = response.json()
        task_id = result["result"]["taskId"]
        print(f"✓ Task created: {task_id}")
        
        # Wait for processing
        time.sleep(1)
        
        # Check status
        status_response = requests.post(f"{base_url}/a2a/v1/rpc", json={
            "jsonrpc": "2.0",
            "method": "agent.getTaskStatus",
            "params": {"taskId": task_id},
            "id": 2
        })
        
        if status_response.status_code == 200:
            status = status_response.json()["result"]
            print(f"✓ Task status: {status['status']['state']}")
            if status.get("artifacts"):
                print(f"✓ Results available: {len(status['artifacts'])} artifacts")
    else:
        print(f"✗ Failed to process message: {response.status_code}")
    
    # 3. Test Account Standardization with Sample Data
    print("\n3. Testing Account Standardization...")
    csv_data = """Account (L0),Account (L1),Account (L2),Account (L3),_row_number
Impairments,Impairments,Credit Impairments,ECL P/L Provision,1
Income,DVA,Fee Income,Fee Income,2
Income,Fee Income,Fee Income,Portfolio Fee,3"""
    
    csv_bytes = base64.b64encode(csv_data.encode()).decode()
    
    response = requests.post(f"{base_url}/a2a/v1/messages", json={
        "message": {
            "role": "user",
            "parts": [{
                "kind": "file",
                "file": {
                    "name": "test_accounts.csv",
                    "mimeType": "text/csv",
                    "bytes": csv_bytes
                }
            }]
        },
        "contextId": f"test-csv-{datetime.now().isoformat()}"
    })
    
    if response.status_code == 200:
        task_id = response.json()["taskId"]
        print(f"✓ CSV processing task created: {task_id}")
        
        # Wait and check
        time.sleep(1)
        status_response = requests.get(f"{base_url}/a2a/v1/tasks/{task_id}")
        
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"✓ CSV processing status: {status['status']['state']}")
    else:
        print(f"✗ Failed to process CSV: {response.status_code}")
    
    # 4. Test Batch Processing
    print("\n4. Testing Batch Multi-Type Processing...")
    response = requests.post(f"{base_url}/a2a/v1/messages", json={
        "message": {
            "role": "user",
            "parts": [{
                "kind": "data",
                "data": {
                    "type": "batch",
                    "location": [
                        {"raw_value": "London, UK"},
                        {"raw_value": "Singapore"}
                    ],
                    "account": [
                        {"raw_value": "Income → Fee Income → Trading Fee"}
                    ],
                    "product": [
                        {"raw_value": "Banking → Corporate → Loans"}
                    ]
                }
            }]
        }
    })
    
    if response.status_code == 200:
        print(f"✓ Batch processing task created")
    else:
        print(f"✗ Failed to process batch: {response.status_code}")
    
    print("\n" + "=" * 50)
    print("Testing complete!")


if __name__ == "__main__":
    test_agent()