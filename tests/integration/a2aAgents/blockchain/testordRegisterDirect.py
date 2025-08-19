#!/usr/bin/env python3
"""
Test the ORD register endpoint directly with a minimal payload
"""
import asyncio
import json
import httpx

async def test_ord_register_direct():
    """Test ORD register endpoint with minimal payload"""
    print("=== TESTING ORD REGISTER ENDPOINT DIRECTLY ===")
    
    # Create minimal ORD document
    minimal_ord_document = {
        "openResourceDiscovery": "1.5.0",
        "description": "Test ORD Document",
        "dataProducts": [
            {
                "ordId": "com.test:dataProduct:test_data",
                "title": "Test Data Product",
                "shortDescription": "Test data - 10 records",
                "description": "Test financial data for registration test",
                "version": "1.0.0",
                "visibility": "internal",
                "tags": ["test", "registration"],
                "labels": {
                    "source": "test",
                    "format": "csv",
                    "records": "10",
                    "columns": "3"
                },
                "accessStrategies": [{
                    "type": "file",
                    "path": "/tmp/test_data.csv"
                }]
            }
        ]
    }
    
    registration_payload = {
        "ord_document": minimal_ord_document,
        "registered_by": "test_client",
        "tags": ["test"],
        "labels": {
            "test": "direct_registration"
        }
    }
    
    print(f"Payload size: {len(json.dumps(registration_payload))} characters")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8000/api/v1/ord/register",
                json=registration_payload
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Registration successful!")
                print(f"Registration ID: {result.get('registration_id')}")
                
                # Test search to see if it was registered
                search_response = await client.post(
                    "http://localhost:8000/api/v1/ord/search",
                    json={
                        "resource_type": "dataProduct",
                        "registered_by": "test_client"
                    }
                )
                
                if search_response.status_code == 200:
                    search_results = search_response.json()
                    print(f"Search found {len(search_results.get('results', []))} products")
                else:
                    print(f"Search failed: {search_response.status_code}")
                    
            else:
                print(f"❌ Registration failed: {response.text}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_ord_register_direct())