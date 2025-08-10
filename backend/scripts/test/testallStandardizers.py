#!/usr/bin/env python3
"""
Test all standardizers with Grok enrichment
"""

import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def test_all_standardizers():
    """Test all 5 standardizers with enrichment"""
    
    print("Testing All Standardizers with Grok Enrichment")
    print("="*60)
    
    # Check if Grok is available
    has_grok = os.getenv('XAI_API_KEY') is not None
    print(f"Grok API Available: {has_grok}")
    print()
    
    # Base URL for agent
    base_url = "http://localhost:8001"
    
    # Test data for each standardizer
    test_data = {
        "location": {
            "type": "location",
            "items": [
                {"Location (L0)": "Europe", "Location (L1)": "Germany", "Location (L2)": "Frankfurt", "_row_number": 1},
                {"Location (L0)": "Americas", "Location (L1)": "Brazil", "Location (L2)": "São Paulo", "_row_number": 2}
            ]
        },
        "account": {
            "type": "account", 
            "items": [
                {"accountNumber": "1001", "accountDescription": "Cash", "costCenter": "CC100", "_row_number": 1},
                {"accountNumber": "2001", "accountDescription": "Accounts Payable", "costCenter": "CC200", "_row_number": 2}
            ]
        },
        "product": {
            "type": "product",
            "items": [
                {"Product (L0)": "Options", "Product (L1)": "Equity Options", "_row_number": 1},
                {"Product (L0)": "Bonds", "Product (L1)": "Corporate Bonds", "_row_number": 2}
            ]
        },
        "book": {
            "type": "book",
            "items": [
                {"Books": "NEWF Group Adjustment", "_row_number": 1},
                {"Books": "Trading Book - Asia", "_row_number": 2}
            ]
        },
        "measure": {
            "type": "measure",
            "items": [
                {"measureType": "Actual", "Version": "YTD", "Currency": "CFX", "_row_number": 1},
                {"measureType": "Budget", "Version": "Q1", "Currency": "LOC", "_row_number": 2}
            ]
        }
    }
    
    results = {}
    
    # Test each standardizer
    for data_type, test_info in test_data.items():
        print(f"\n{data_type.upper()} Standardization:")
        print("-" * 40)
        
        message = {
            "messageId": f"test-all-{data_type}",
            "role": "user",
            "parts": [{
                "kind": "data",
                "data": test_info
            }]
        }
        
        try:
            # Send request
            response = requests.post(
                f"{base_url}/process",
                json={"message": message, "contextId": f"test-all-{data_type}"}
            )
            
            if response.status_code == 200:
                task = response.json()
                print(f"✓ Task created: {task['taskId']}")
                
                # Wait for processing
                time.sleep(2)
                
                # Check status
                status_response = requests.get(f"{base_url}/status/{task['taskId']}")
                if status_response.status_code == 200:
                    status = status_response.json()
                    
                    if 'artifacts' in status and status['artifacts']:
                        for artifact in status['artifacts']:
                            if 'parts' in artifact:
                                for part in artifact['parts']:
                                    if part['kind'] == 'data' and 'data' in part:
                                        data = part['data']
                                        if 'results' in data and data['results']:
                                            result = data['results'][0]
                                            std = result.get('standardized', {})
                                            completeness = result.get('completeness', 0)
                                            enriched = result.get('metadata', {}).get('enriched_with_ai', False)
                                            
                                            print(f"  Completeness: {completeness*100:.1f}%")
                                            print(f"  Enriched with AI: {enriched}")
                                            
                                            # Show some standardized fields
                                            if data_type == "location":
                                                print(f"  ISO codes: {std.get('iso2')}/{std.get('iso3')}")
                                            elif data_type == "account":
                                                print(f"  GL Code: {std.get('gl_account_code')}")
                                            elif data_type == "product":
                                                print(f"  Basel Category: {std.get('basel_category')}")
                                            elif data_type == "book":
                                                print(f"  Entity Type: {std.get('entity_type')}")
                                            elif data_type == "measure":
                                                print(f"  Measure Type: {std.get('measure_type')}")
                                            
                                            results[data_type] = {
                                                "success": True,
                                                "completeness": completeness,
                                                "enriched": enriched
                                            }
                else:
                    print(f"✗ Failed to get status: {status_response.status_code}")
                    results[data_type] = {"success": False, "error": "Status check failed"}
            else:
                print(f"✗ Failed to create task: {response.status_code}")
                results[data_type] = {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            results[data_type] = {"success": False, "error": str(e)}
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print("-" * 40)
    
    successful = sum(1 for r in results.values() if r.get("success", False))
    enriched = sum(1 for r in results.values() if r.get("enriched", False))
    avg_completeness = sum(r.get("completeness", 0) for r in results.values() if r.get("success", False)) / max(successful, 1)
    
    print(f"Successful standardizations: {successful}/5")
    print(f"Enriched with AI: {enriched}")
    print(f"Average completeness: {avg_completeness*100:.1f}%")
    
    # Check output files
    print("\nOutput Files:")
    output_dir = "/Users/apple/projects/finsight_cib/data/interim/1/dataStandardization"
    if os.path.exists(output_dir):
        files = [f for f in os.listdir(output_dir) if f.startswith('standardized_')]
        for f in sorted(files):
            file_path = os.path.join(output_dir, f)
            size = os.path.getsize(file_path)
            print(f"  {f}: {size:,} bytes")


if __name__ == "__main__":
    try:
        test_all_standardizers()
        print(f"\n{'='*60}")
        print("✓ All standardizers tested!")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")