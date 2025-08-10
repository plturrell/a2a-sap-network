#!/usr/bin/env python3
"""
Test script to check ORD registration and discovery
"""
import asyncio
import json
import httpx

async def test_ord_discovery():
    """Test the ORD registration and discovery system"""
    
    print("=== ORD REGISTRY TEST ===")
    
    # Test ORD Registry connection
    ordRegistry_url = "http://localhost:8000/api/v1/ord"
    
    try:
        async with httpx.AsyncClient() as client:
            # Test registry health
            response = await client.get(f"{ordRegistry_url}/health")
            if response.status_code == 200:
                print("✅ ORD Registry is healthy")
            else:
                print(f"❌ ORD Registry health check failed: {response.status_code}")
                return
    except Exception as e:
        print(f"❌ Cannot reach ORD Registry: {e}")
        return
    
    # Search for registered data products
    print("\n=== SEARCHING FOR REGISTERED DATA PRODUCTS ===")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ordRegistry_url}/search",
                json={
                    "resource_type": "dataProduct",
                    "tags": ["crd", "raw-data"],
                    "registered_by": "data_product_agent"
                }
            )
            
            if response.status_code == 200:
                search_results = response.json()
                products = search_results.get("results", [])
                print(f"Found {len(products)} registered data products")
                
                for product in products:
                    ord_id = product.get("ord_id", "")
                    title = product.get("title", "")
                    labels = product.get("labels", {})
                    records = labels.get("records", "0")
                    integrity_hash = labels.get("integrity_hash", "")
                    
                    print(f"  - {title}")
                    print(f"    ORD ID: {ord_id}")
                    print(f"    Records: {records}")
                    print(f"    Integrity Hash: {integrity_hash[:16]}..." if integrity_hash else "    No integrity hash")
                    
                    # Check access strategies
                    access_strategies = product.get("accessStrategies", [])
                    for strategy in access_strategies:
                        if strategy.get("type") == "file":
                            file_path = strategy.get("path", "")
                            print(f"    File Path: {file_path}")
                            
                            # Check if file exists and count rows
                            try:
                                import pandas as pd
                                df = pd.read_csv(file_path)
                                actual_rows = len(df)
                                expected_rows = int(records) if records.isdigit() else 0
                                if actual_rows != expected_rows:
                                    print(f"    ⚠️  Row count mismatch: file has {actual_rows}, ORD says {expected_rows}")
                                else:
                                    print(f"    ✅ Row count matches: {actual_rows}")
                            except Exception as e:
                                print(f"    ❌ Cannot read file: {e}")
                    
                    print()
                
                # Test Agent 1's discovery query
                print("\n=== TESTING AGENT 1 DISCOVERY QUERY ===")
                print("Query that Agent 1 should use:")
                query = {
                    "resource_type": "dataProduct",
                    "tags": ["crd", "raw-data"],
                    "registered_by": "data_product_agent"
                }
                print(json.dumps(query, indent=2))
                
                if products:
                    print(f"\n✅ Agent 1 should discover {len(products)} data products")
                    total_expected_rows = 0
                    for product in products:
                        labels = product.get("labels", {})
                        records = labels.get("records", "0")
                        if records.isdigit():
                            total_expected_rows += int(records)
                    print(f"Total rows Agent 1 should process: {total_expected_rows}")
                else:
                    print("❌ No data products found - Agent 1 will have nothing to process")
                    
            else:
                print(f"❌ ORD search failed: {response.status_code} - {response.text}")
                
    except Exception as e:
        print(f"❌ ORD search error: {e}")
    
    # Test Agent 0 registration endpoint
    print("\n=== TESTING AGENT 0 REGISTRATION ===")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health")
            if response.status_code == 200:
                print("✅ Agent 0 is healthy")
            else:
                print(f"❌ Agent 0 health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot reach Agent 0: {e}")

if __name__ == "__main__":
    asyncio.run(test_ord_discovery())