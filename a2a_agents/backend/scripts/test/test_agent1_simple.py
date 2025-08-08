#!/usr/bin/env python3
"""
Simple test for Agent 1 standardization
"""

import requests
import json
import time

# Test simple location standardization
location_message = {
    "messageId": "simple-test-001",
    "role": "user",
    "parts": [{
        "kind": "data",
        "data": {
            "type": "location",
            "items": [
                {"Location (L0)": "Europe", "Location (L1)": "United Kingdom", "Location (L2)": "London", "Location (L3)": "City of London", "Location (L4)": "Bank", "_row_number": 1}
            ]
        }
    }]
}

print("Sending location standardization request...")
response = requests.post(
    "http://localhost:8001/process",
    json={"message": location_message, "contextId": "simple-test"}
)

if response.status_code == 200:
    task = response.json()
    task_id = task['taskId']
    print(f"Task created: {task_id}")
    
    # Wait a bit
    time.sleep(2)
    
    # Check status
    status_response = requests.get(f"http://localhost:8001/status/{task_id}")
    if status_response.status_code == 200:
        status = status_response.json()
        print(f"Status: {status['status']['state']}")
        
        if 'artifacts' in status and status['artifacts']:
            print("Artifacts created:")
            for artifact in status['artifacts']:
                print(f"  - {artifact['name']}")
                if 'parts' in artifact:
                    for part in artifact['parts']:
                        if part['kind'] == 'data' and 'data' in part:
                            data = part['data']
                            print(f"    Results: {data.get('standardized_count')} items")
                            if 'results' in data and data['results']:
                                print(f"    First result: {json.dumps(data['results'][0], indent=2)}")
        else:
            print("No artifacts created")
    else:
        print(f"Status check failed: {status_response.status_code}")
else:
    print(f"Request failed: {response.status_code}")
    print(response.text)