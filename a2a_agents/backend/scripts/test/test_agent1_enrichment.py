#!/usr/bin/env python3
"""
Test Agent 1 with Grok enrichment for completeness
"""

import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def test_agent1_enrichment():
    """Test Agent 1 with data that needs enrichment"""
    
    print("Testing Agent 1 - Data Standardization with Enrichment")
    print("="*60)
    
    # Check if Grok is available
    has_grok = os.getenv('XAI_API_KEY') is not None
    print(f"Grok API Available: {has_grok}")
    
    # 1. Test location needing enrichment (missing ISO codes and coordinates)
    print("\n1. Testing location standardization with enrichment...")
    
    location_message = {
        "messageId": "test-enrich-001",
        "role": "user",
        "parts": [{
            "kind": "data",
            "data": {
                "type": "location",
                "items": [
                    {"Location (L0)": "Europe", "Location (L1)": "France", "Location (L2)": "Paris", "Location (L3)": "8th Arrondissement", "Location (L4)": "Champs-Élysées", "_row_number": 1},
                    {"Location (L0)": "Asia", "Location (L1)": "Singapore", "Location (L2)": "Central", "Location (L3)": "Marina Bay", "Location (L4)": "Financial District", "_row_number": 2},
                    {"Location (L0)": "Africa", "Location (L1)": "South Africa", "Location (L2)": "Gauteng", "Location (L3)": "Johannesburg", "Location (L4)": "Sandton", "_row_number": 3}
                ]
            }
        }]
    }
    
    response = requests.post(
        "http://localhost:8001/process",
        json={"message": location_message, "contextId": "test-enrichment"}
    )
    
    if response.status_code == 200:
        task = response.json()
        print(f"✓ Location task created: {task['taskId']}")
        
        # Wait for processing
        time.sleep(3)
        
        # Check status
        status_response = requests.get(f"http://localhost:8001/status/{task['taskId']}")
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"  Status: {status['status']['state']}")
            
            if 'artifacts' in status and status['artifacts']:
                for artifact in status['artifacts']:
                    if 'parts' in artifact:
                        for part in artifact['parts']:
                            if part['kind'] == 'data' and 'data' in part:
                                data = part['data']
                                if 'results' in data and data['results']:
                                    print(f"\n  Sample location result:")
                                    result = data['results'][0]
                                    std = result.get('standardized', {})
                                    print(f"    Name: {std.get('name')}")
                                    print(f"    Country: {std.get('country')}")
                                    print(f"    ISO2: {std.get('iso2')}")
                                    print(f"    ISO3: {std.get('iso3')}")
                                    print(f"    Coordinates: {std.get('coordinates')}")
                                    print(f"    Completeness: {result.get('completeness', 0)*100:.1f}%")
                                    print(f"    Enriched with AI: {result.get('metadata', {}).get('enriched_with_ai', False)}")
    
    # 2. Test product standardization with enrichment
    print("\n2. Testing product standardization with enrichment...")
    
    product_message = {
        "messageId": "test-enrich-002",
        "role": "user",
        "parts": [{
            "kind": "data",
            "data": {
                "type": "product",
                "items": [
                    {"Product (L0)": "Derivatives", "Product (L1)": "Interest Rate Derivatives", "Product (L2)": "Swaps", "Product (L3)": "Vanilla IRS", "_row_number": 1},
                    {"Product (L0)": "Lending", "Product (L1)": "Corporate Lending", "Product (L2)": "Term Loans", "Product (L3)": "Syndicated Loans", "_row_number": 2},
                    {"Product (L0)": "Trading", "Product (L1)": "FX Trading", "Product (L2)": "Spot FX", "Product (L3)": "Major Currency Pairs", "_row_number": 3}
                ]
            }
        }]
    }
    
    response = requests.post(
        "http://localhost:8001/process",
        json={"message": product_message, "contextId": "test-enrichment"}
    )
    
    if response.status_code == 200:
        task = response.json()
        print(f"✓ Product task created: {task['taskId']}")
        
        # Wait for processing
        time.sleep(3)
        
        # Check status
        status_response = requests.get(f"http://localhost:8001/status/{task['taskId']}")
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"  Status: {status['status']['state']}")
            
            if 'artifacts' in status and status['artifacts']:
                for artifact in status['artifacts']:
                    if 'parts' in artifact:
                        for part in artifact['parts']:
                            if part['kind'] == 'data' and 'data' in part:
                                data = part['data']
                                if 'results' in data and data['results']:
                                    print(f"\n  Sample product result:")
                                    result = data['results'][0]
                                    std = result.get('standardized', {})
                                    print(f"    Hierarchy: {std.get('hierarchy_path')}")
                                    print(f"    Category: {std.get('product_category')}")
                                    print(f"    Basel Category: {std.get('basel_category')}")
                                    print(f"    Regulatory Treatment: {std.get('regulatory_treatment')}")
                                    print(f"    Completeness: {result.get('completeness', 0)*100:.1f}%")
                                    print(f"    Enriched with AI: {result.get('metadata', {}).get('enriched_with_ai', False)}")
    
    # 3. Check output files
    print("\n3. Checking output files...")
    output_dir = "/Users/apple/projects/finsight_cib/data/interim/1/dataStandardization"
    
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"✓ Found {len(files)} files in output directory")
        
        # Check location file
        location_file = os.path.join(output_dir, "standardized_location.json")
        if os.path.exists(location_file):
            with open(location_file, 'r') as f:
                data = json.load(f)
                total_records = len(data.get('data', []))
                enriched_count = sum(1 for record in data.get('data', []) 
                                   if record.get('metadata', {}).get('enriched_with_ai', False))
                print(f"  Location records: {total_records} total, {enriched_count} enriched")
        
        # Check product file
        product_file = os.path.join(output_dir, "standardized_product.json")
        if os.path.exists(product_file):
            with open(product_file, 'r') as f:
                data = json.load(f)
                total_records = len(data.get('data', []))
                enriched_count = sum(1 for record in data.get('data', []) 
                                   if record.get('metadata', {}).get('enriched_with_ai', False))
                print(f"  Product records: {total_records} total, {enriched_count} enriched")


if __name__ == "__main__":
    try:
        test_agent1_enrichment()
        print(f"\n{'='*60}")
        print("✓ Agent 1 enrichment testing completed!")
    except Exception as e:
        print(f"\n✗ Error testing Agent 1: {str(e)}")