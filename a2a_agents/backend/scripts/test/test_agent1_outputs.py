#!/usr/bin/env python3
"""
Test Agent 1 (Data Standardization Agent) to verify it produces actual outputs
"""

import requests
import json
import time
import os

def test_agent1_outputs():
    """Test Agent 1 directly with sample data"""
    
    print("Testing Agent 1 - Data Standardization Agent")
    print("="*60)
    
    # 1. Get Agent 1 card
    print("\n1. Getting Agent 1 card...")
    response = requests.get("http://localhost:8001")
    if response.status_code == 200:
        agent_card = response.json()
        print(f"✓ Agent: {agent_card['name']}")
        print(f"  Version: {agent_card['version']}")
        print(f"  Skills: {len(agent_card['skills'])} available")
        for skill in agent_card['skills']:
            print(f"    - {skill['name']}")
    
    # 2. Test location standardization
    print("\n2. Testing location standardization...")
    
    location_message = {
        "messageId": "test-location-001",
        "role": "user",
        "parts": [{
            "kind": "data",
            "data": {
                "type": "location",
                "items": [
                    {"Location (L0)": "Americas", "Location (L1)": "United States", "Location (L2)": "New York", "Location (L3)": "New York City", "Location (L4)": "Manhattan", "_row_number": 1},
                    {"Location (L0)": "Africa", "Location (L1)": "Kenya", "Location (L2)": "Nairobi", "Location (L3)": "Nairobi City", "Location (L4)": "CBD", "_row_number": 2},
                    {"Location (L0)": "Americas", "Location (L1)": "Brazil", "Location (L2)": "Sao Paulo", "Location (L3)": "Sao Paulo City", "Location (L4)": "Centro", "_row_number": 3}
                ]
            }
        }]
    }
    
    response = requests.post(
        "http://localhost:8001/process",
        json={"message": location_message, "contextId": "test-location-context"}
    )
    
    if response.status_code == 200:
        task = response.json()
        print(f"✓ Location standardization task created: {task['taskId']}")
        
        # Monitor task status
        for i in range(10):
            time.sleep(1)
            status_response = requests.get(f"http://localhost:8001/status/{task['taskId']}")
            
            if status_response.status_code == 200:
                status = status_response.json()
                print(f"  Status: {status['status']['state']}")
                
                if status['status']['state'] == 'completed':
                    print("✓ Location standardization completed!")
                    
                    # Check outputs
                    output_dir = "/Users/apple/projects/finsight_cib/data/interim/1/dataStandardization"
                    if os.path.exists(output_dir):
                        files = os.listdir(output_dir)
                        print(f"  Output files in {output_dir}:")
                        for file in files:
                            if file.startswith("standardized_location"):
                                print(f"    - {file}")
                                
                                # Read and display sample
                                file_path = os.path.join(output_dir, file)
                                with open(file_path, 'r') as f:
                                    data = json.load(f)
                                    print(f"      Records: {data['metadata']['records']}")
                                    print(f"      Timestamp: {data['metadata']['timestamp']}")
                                    
                                    if data['data'] and len(data['data']) > 0:
                                        print(f"      Sample standardized location:")
                                        sample = data['data'][0]
                                        if 'standardized' in sample and sample['standardized']:
                                            std = sample['standardized']
                                            print(f"        Name: {std.get('name')}")
                                            print(f"        Country: {std.get('country')}")
                                            print(f"        ISO2: {std.get('iso2')}")
                                            coords = std.get('coordinates', {})
                                            if coords:
                                                print(f"        Coordinates: ({coords.get('latitude')}, {coords.get('longitude')})")
                                        else:
                                            print(f"        Error: {sample.get('error', 'Unknown error')}")
                                        
                    break
                elif status['status']['state'] == 'failed':
                    print(f"✗ Task failed: {status['status'].get('error')}")
                    break
    
    # 3. Test account standardization
    print("\n3. Testing account standardization...")
    
    account_message = {
        "messageId": "test-account-001",
        "role": "user",
        "parts": [{
            "kind": "data",
            "data": {
                "type": "account",
                "items": [
                    {"Account (L0)": "Income", "Account (L1)": "Fee Income", "Account (L2)": "Banking Fees", "Account (L3)": "Transaction Fees (0L)", "_row_number": 1},
                    {"Account (L0)": "Total Cost", "Account (L1)": "Staff Costs", "Account (L2)": "Salaries", "Account (L3)": "Base Salaries", "_row_number": 2},
                    {"Account (L0)": "Risk Weighted Assets", "Account (L1)": "Credit RWA", "Account (L2)": "Corporate Lending", "Account (L3)": "Large Corporate", "_row_number": 3}
                ]
            }
        }]
    }
    
    response = requests.post(
        "http://localhost:8001/process",
        json={"message": account_message, "contextId": "test-account-context"}
    )
    
    if response.status_code == 200:
        task = response.json()
        print(f"✓ Account standardization task created: {task['taskId']}")
        
        # Monitor task status
        for i in range(10):
            time.sleep(1)
            status_response = requests.get(f"http://localhost:8001/status/{task['taskId']}")
            
            if status_response.status_code == 200:
                status = status_response.json()
                print(f"  Status: {status['status']['state']}")
                
                if status['status']['state'] == 'completed':
                    print("✓ Account standardization completed!")
                    
                    # Check outputs
                    output_dir = "/Users/apple/projects/finsight_cib/data/interim/1/dataStandardization"
                    if os.path.exists(output_dir):
                        files = os.listdir(output_dir)
                        for file in files:
                            if file.startswith("standardized_account"):
                                print(f"    - {file}")
                                
                                # Read and display sample
                                file_path = os.path.join(output_dir, file)
                                with open(file_path, 'r') as f:
                                    data = json.load(f)
                                    print(f"      Records: {data['metadata']['records']}")
                                    
                                    if data['data'] and len(data['data']) > 0:
                                        print(f"      Sample standardized account:")
                                        sample = data['data'][0]
                                        if 'standardized' in sample:
                                            std = sample['standardized']
                                            print(f"        Hierarchy: {std.get('hierarchy_path')}")
                                            print(f"        Type: {std.get('account_type')}")
                                            print(f"        IFRS9: {std.get('ifrs9_classification')}")
                                            print(f"        Basel: {std.get('basel_classification')}")
                                        
                    break
                elif status['status']['state'] == 'failed':
                    print(f"✗ Task failed: {status['status'].get('error')}")
                    break
    
    # 4. Check overall output directory
    print("\n4. Checking output directory contents...")
    output_dir = "/Users/apple/projects/finsight_cib/data/interim/1/dataStandardization"
    
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"✓ Found {len(files)} files in output directory:")
        for file in files:
            file_path = os.path.join(output_dir, file)
            size = os.path.getsize(file_path)
            print(f"  - {file} ({size} bytes)")
    else:
        print(f"✗ Output directory not found: {output_dir}")
        print(f"  Creating directory...")
        os.makedirs(output_dir, exist_ok=True)
        
    return True


if __name__ == "__main__":
    try:
        test_agent1_outputs()
        print(f"\n{'='*60}")
        print("✓ Agent 1 testing completed!")
    except Exception as e:
        print(f"\n✗ Error testing Agent 1: {str(e)}")