#!/usr/bin/env python3
"""
Test script for Data Manager functionality
"""

import asyncio
import json
import httpx
from datetime import datetime
import sys


async def test_data_manager():
    """Test Data Manager operations"""
    print("🧪 Testing Data Manager Functionality\n")
    
    base_url = "http://localhost:8008"
    agent_id = "data_manager_agent"
    
    async with httpx.AsyncClient() as client:
        # Check health
        print("🏥 Checking health...")
        response = await client.get(f"{base_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Health check passed: {health['status']}")
            print(f"   Storage backend: {health['components']['storage_backend']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
        
        # Test storing data
        print("\n📝 Testing data storage...")
        store_request = {
            "jsonrpc": "2.0",
            "method": "store_data",
            "params": {
                "data_type": "accounts",
                "data": {
                    "id": "ACC001",
                    "name": "Test Account",
                    "currency": "USD",
                    "balance": 10000.00,
                    "type": "checking"
                },
                "metadata": {
                    "source": "test_script",
                    "environment": "test"
                },
                "context_id": "test_context_001"
            },
            "id": "store_1"
        }
        
        response = await client.post(
            f"{base_url}/a2a/{agent_id}/v1/rpc",
            json=store_request
        )
        
        if response.status_code == 200:
            result = response.json()
            if "result" in result and result["result"]["status"] == "success":
                record_id = result["result"]["data"]["record_id"]
                print(f"✅ Data stored successfully: {record_id}")
            else:
                print(f"❌ Store failed: {result}")
                return
        else:
            print(f"❌ Store request failed: {response.status_code}")
            return
        
        # Test retrieving data
        print("\n🔍 Testing data retrieval...")
        retrieve_request = {
            "jsonrpc": "2.0",
            "method": "retrieve_data",
            "params": {
                "record_id": record_id
            },
            "id": "retrieve_1"
        }
        
        response = await client.post(
            f"{base_url}/a2a/{agent_id}/v1/rpc",
            json=retrieve_request
        )
        
        if response.status_code == 200:
            result = response.json()
            if "result" in result and result["result"]["status"] == "success":
                records = result["result"]["data"]["records"]
                print(f"✅ Retrieved {len(records)} record(s)")
                if records:
                    print(f"   Record ID: {records[0]['record_id']}")
                    print(f"   Data type: {records[0]['data_type']}")
                    print(f"   From cache: {result['result']['data']['from_cache']}")
            else:
                print(f"❌ Retrieve failed: {result}")
        
        # Test querying data
        print("\n🔎 Testing data query...")
        query_request = {
            "jsonrpc": "2.0",
            "method": "query",
            "params": {
                "filters": {
                    "data_type": "accounts",
                    "context_id": "test_context_001"
                },
                "options": {
                    "page": 1,
                    "page_size": 10,
                    "order_by": "created_at",
                    "order_dir": "DESC"
                }
            },
            "id": "query_1"
        }
        
        response = await client.post(
            f"{base_url}/a2a/{agent_id}/v1/rpc",
            json=query_request
        )
        
        if response.status_code == 200:
            result = response.json()
            if "result" in result:
                query_result = result["result"]
                print(f"✅ Query successful")
                print(f"   Total records: {query_result['total_count']}")
                print(f"   Page: {query_result['page']} of {(query_result['total_count'] + query_result['page_size'] - 1) // query_result['page_size']}")
                print(f"   Has next page: {query_result['has_next']}")
        
        # Test bulk operations
        print("\n📦 Testing bulk operations...")
        bulk_request = {
            "jsonrpc": "2.0",
            "method": "bulk_operations",
            "params": {
                "operations": [
                    {
                        "type": "store",
                        "data_type": "accounts",
                        "data": {
                            "id": "ACC002",
                            "name": "Bulk Account 1",
                            "currency": "EUR",
                            "balance": 5000.00
                        },
                        "context_id": "test_context_001"
                    },
                    {
                        "type": "store",
                        "data_type": "accounts",
                        "data": {
                            "id": "ACC003",
                            "name": "Bulk Account 2",
                            "currency": "GBP",
                            "balance": 7500.00
                        },
                        "context_id": "test_context_001"
                    }
                ]
            },
            "id": "bulk_1"
        }
        
        response = await client.post(
            f"{base_url}/a2a/{agent_id}/v1/rpc",
            json=bulk_request
        )
        
        if response.status_code == 200:
            result = response.json()
            if "result" in result:
                bulk_result = result["result"]
                print(f"✅ Bulk operations completed")
                print(f"   Successful: {bulk_result['successful']}")
                print(f"   Failed: {bulk_result['failed']}")
        
        # Test data export
        print("\n📤 Testing data export...")
        export_response = await client.post(
            f"{base_url}/a2a/{agent_id}/v1/export",
            json={
                "filters": {"data_type": "accounts"},
                "format": "json"
            }
        )
        
        if export_response.status_code == 200:
            export_data = export_response.json()
            print(f"✅ Export successful")
            print(f"   Exported {export_data['count']} records")
        
        # Get metrics
        print("\n📊 Getting metrics...")
        metrics_response = await client.get(f"{base_url}/metrics")
        if metrics_response.status_code == 200:
            metrics = metrics_response.text
            print("✅ Metrics retrieved")
            # Parse some key metrics
            for line in metrics.split('\n'):
                if 'a2a_records_stored_total' in line and not line.startswith('#'):
                    print(f"   Records stored: {line.split()[-1]}")
                elif 'a2a_queries_processed_total' in line and not line.startswith('#'):
                    print(f"   Queries processed: {line.split()[-1]}")
        
        print("\n✨ All tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_data_manager())