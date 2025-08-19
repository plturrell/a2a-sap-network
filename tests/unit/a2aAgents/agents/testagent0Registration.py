#!/usr/bin/env python3
"""
Simple test script to directly call Agent 0 registration endpoint
and capture detailed error information.
"""

import requests
import json
import traceback
from datetime import datetime

def test_agent0_registration():
    """Test Agent 0 registration with detailed error capture."""
    
    # Test data for registration
    test_data = {
        "message": {
            "role": "user",
            "parts": [
                {
                    "kind": "a2a",
                    "data": {
                        "type": "agent0_register",
                        "agent": {
                            "name": "test-agent",
                            "type": "general",
                            "capabilities": ["text_processing"],
                            "description": "Test agent for debugging"
                        }
                    }
                }
            ]
        }
    }
    
    # Endpoint URL
    url = "http://localhost:8000/a2a/agent0/v1/messages"
    
    print(f"\n{'='*60}")
    print(f"Testing Agent 0 Registration - {datetime.now()}")
    print(f"{'='*60}\n")
    
    print("Request URL:", url)
    print("\nRequest Data:")
    print(json.dumps(test_data, indent=2))
    
    try:
        # Make the request with detailed error capture
        print("\nSending request...")
        response = requests.post(
            url,
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"\nResponse Status Code: {response.status_code}")
        print(f"Response Headers:")
        for header, value in response.headers.items():
            print(f"  {header}: {value}")
        
        # Try to parse JSON response
        try:
            response_data = response.json()
            print("\nResponse Body (JSON):")
            print(json.dumps(response_data, indent=2))
        except json.JSONDecodeError:
            print("\nResponse Body (Text):")
            print(response.text)
        
        # Check if request was successful
        if response.status_code == 200:
            print("\n✓ Registration request successful!")
            
            # Check task status after a short delay
            if 'taskId' in response_data:
                task_id = response_data['taskId']
                print(f"\nTask ID: {task_id}")
                print("\nWaiting 2 seconds before checking task status...")
                import time
                time.sleep(2)
                
                # Check task status
                status_url = f"http://localhost:8000/a2a/agent0/v1/tasks/{task_id}"
                print(f"\nChecking task status at: {status_url}")
                
                try:
                    status_response = requests.get(status_url)
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        print("\nTask Status:")
                        print(json.dumps(status_data, indent=2))
                    else:
                        print(f"\nFailed to get task status: {status_response.status_code}")
                        print(status_response.text)
                except Exception as e:
                    print(f"\nError checking task status: {str(e)}")
        else:
            print(f"\n✗ Registration failed with status code: {response.status_code}")
            
            # Try to extract error details from response
            if response.headers.get('content-type', '').startswith('application/json'):
                try:
                    error_data = response.json()
                    if 'detail' in error_data:
                        print(f"\nError Detail: {error_data['detail']}")
                    if 'error' in error_data:
                        print(f"\nError: {error_data['error']}")
                    if 'message' in error_data:
                        print(f"\nMessage: {error_data['message']}")
                except:
                    pass
        
    except requests.exceptions.ConnectionError as e:
        print(f"\n✗ Connection Error: Could not connect to {url}")
        print(f"   Make sure the server is running on port 8000")
        print(f"\nDetailed error: {str(e)}")
        
    except requests.exceptions.Timeout as e:
        print(f"\n✗ Timeout Error: Request timed out after 30 seconds")
        print(f"\nDetailed error: {str(e)}")
        
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Request Error: {type(e).__name__}")
        print(f"\nDetailed error: {str(e)}")
        print(f"\nTraceback:")
        traceback.print_exc()
        
    except Exception as e:
        print(f"\n✗ Unexpected Error: {type(e).__name__}")
        print(f"\nDetailed error: {str(e)}")
        print(f"\nTraceback:")
        traceback.print_exc()
    
    print(f"\n{'='*60}\n")

def test_server_health():
    """Test if the server is running and accessible."""
    print("\nTesting server health...")
    
    # Try common health endpoints
    health_endpoints = [
        "http://localhost:8000/",
        "http://localhost:8000/health",
        "http://localhost:8000/api/health",
        "http://localhost:8000/docs"
    ]
    
    for endpoint in health_endpoints:
        try:
            response = requests.get(endpoint, timeout=5)
            if response.status_code < 500:
                print(f"✓ {endpoint} - Status: {response.status_code}")
                return True
        except:
            print(f"✗ {endpoint} - Not accessible")
    
    return False

if __name__ == "__main__":
    print("Agent 0 Registration Test Script")
    print("================================\n")
    
    # First check if server is running
    if test_server_health():
        print("\nServer appears to be running. Testing registration...")
        test_agent0_registration()
    else:
        print("\n✗ Server does not appear to be running on port 8000")
        print("  Please start the server first.")