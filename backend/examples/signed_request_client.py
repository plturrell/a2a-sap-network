#!/usr/bin/env python3
"""
Example: Signed Request Client
Demonstrates how to make signed API requests
"""

import httpx
import asyncio
import json
from typing import Dict, Any, Optional

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.requestSigning import get_signing_service


class SignedAPIClient:
    """Example client that signs API requests"""
    
    def __init__(self, base_url: str, api_key_id: str = "default"):
        self.base_url = base_url.rstrip('/')
        self.api_key_id = api_key_id
        self.signing_service = get_signing_service()
        self.client = httpx.AsyncClient()
    
    async def request(self, 
                     method: str,
                     path: str,
                     json_data: Optional[Dict[str, Any]] = None,
                     **kwargs) -> httpx.Response:
        """Make a signed API request"""
        
        # Prepare URL
        url = f"{self.base_url}{path}"
        
        # Prepare body
        body = None
        headers = kwargs.get("headers", {})
        
        if json_data:
            body = json.dumps(json_data).encode('utf-8')
            headers["Content-Type"] = "application/json"
        
        # Sign request
        signing_headers = self.signing_service.sign_request(
            method=method,
            url=url,
            body=body,
            api_key_id=self.api_key_id,
            headers=headers
        )
        
        # Add signing headers
        headers.update(signing_headers)
        kwargs["headers"] = headers
        
        # Add body if present
        if body:
            kwargs["content"] = body
        
        # Make request
        response = await self.client.request(method, url, **kwargs)
        return response
    
    async def close(self):
        """Close the client"""
        await self.client.aclose()


async def main():
    """Example usage of signed API client"""
    
    # Create client
    client = SignedAPIClient("http://localhost:8000")
    
    try:
        print("🔐 Signed API Request Example\n")
        
        # Example 1: GET request
        print("1. Making signed GET request to /health")
        response = await client.request("GET", "/health")
        print(f"   Status: {response.status_code}")
        print(f"   Signature Verified: {response.headers.get('x-signature-verified', 'not checked')}")
        print(f"   Response: {response.json()}\n")
        
        # Example 2: GET with authentication
        print("2. Making signed GET request to /api/v1/users/me")
        # Note: This would also need a Bearer token for user authentication
        response = await client.request(
            "GET", 
            "/api/v1/users/me",
            headers={"Authorization": "Bearer YOUR_JWT_TOKEN_HERE"}
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 401:
            print("   Note: This endpoint also requires user authentication (Bearer token)\n")
        
        # Example 3: POST request with body
        print("3. Making signed POST request with body")
        data = {
            "name": "Test Item",
            "value": 42,
            "tags": ["example", "signed"]
        }
        
        # This is just an example - the endpoint might not exist
        try:
            response = await client.request(
                "POST",
                "/api/v1/data/items",
                json_data=data,
                headers={"Authorization": "Bearer YOUR_JWT_TOKEN_HERE"}
            )
            print(f"   Status: {response.status_code}")
            print(f"   Body hash was included for integrity verification\n")
        except Exception as e:
            print(f"   Example endpoint might not exist: {e}\n")
        
        # Example 4: Show signing headers
        print("4. Example of signing headers:")
        signing_headers = client.signing_service.sign_request(
            method="GET",
            url="http://localhost:8000/api/v1/data",
            api_key_id="default"
        )
        
        for header, value in signing_headers.items():
            print(f"   {header}: {value}")
        
        print("\n✅ Request signing examples completed!")
        print("\n💡 Tips:")
        print("   - Signatures include timestamp to prevent replay attacks")
        print("   - Nonces ensure each request is unique")
        print("   - Body hash verifies request integrity")
        print("   - API keys have specific permissions (read, write, admin, a2a)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())