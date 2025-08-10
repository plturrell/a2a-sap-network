#!/usr/bin/env python3
"""
Test Agent 1 with data that requires multi-pass enrichment
"""

import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def test_agent1_multipass():
    """Test Agent 1 with incomplete data"""
    
    print("Testing Agent 1 - Multi-Pass Enrichment")
    print("="*60)
    
    # Check if Grok is available
    has_grok = os.getenv('XAI_API_KEY') is not None
    print(f"Grok API Available: {has_grok}")
    
    # Test with very incomplete data
    print("\n1. Testing location with minimal data...")
    
    location_message = {
        "messageId": "test-multipass-001",
        "role": "user",
        "parts": [{
            "kind": "data",
            "data": {
                "type": "location",
                "items": [
                    {"Location (L0)": "Europe", "Location (L1)": "Malta", "_row_number": 1},
                    {"Location (L0)": "Asia", "Location (L1)": "Cambodia", "_row_number": 2},
                    {"Location (L0)": "Americas", "Location (L1)": "Uruguay", "_row_number": 3}
                ]
            }
        }]
    }
    
    response = requests.post(
        "http://localhost:8001/process",
        json={"message": location_message, "contextId": "test-multipass"}
    )
    
    if response.status_code == 200:
        task = response.json()
        print(f"✓ Location task created: {task['taskId']}")
        
        # Wait for processing
        time.sleep(5)
        
        # Check status
        status_response = requests.get(f"http://localhost:8001/status/{task['taskId']}")
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"  Status: {status['status']['state']}")
    
    # Test with vague product data
    print("\n2. Testing product with incomplete hierarchy...")
    
    product_message = {
        "messageId": "test-multipass-002",
        "role": "user",
        "parts": [{
            "kind": "data",
            "data": {
                "type": "product",
                "items": [
                    {"Product (L0)": "Exotic Options", "_row_number": 1},
                    {"Product (L0)": "Structured Notes", "_row_number": 2},
                    {"Product (L0)": "Hybrid Instruments", "_row_number": 3}
                ]
            }
        }]
    }
    
    response = requests.post(
        "http://localhost:8001/process",
        json={"message": product_message, "contextId": "test-multipass"}
    )
    
    if response.status_code == 200:
        task = response.json()
        print(f"✓ Product task created: {task['taskId']}")
        
        # Wait for processing
        time.sleep(5)
        
        # Check status
        status_response = requests.get(f"http://localhost:8001/status/{task['taskId']}")
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"  Status: {status['status']['state']}")
    
    # Check output files
    print("\n3. Analyzing enrichment results...")
    output_dir = "/Users/apple/projects/finsight_cib/data/interim/1/dataStandardization"
    
    # Check location file
    location_file = os.path.join(output_dir, "standardized_location.json")
    if os.path.exists(location_file):
        with open(location_file, 'r') as f:
            data = json.load(f)
            print("\n  Location Results:")
            for record in data.get('data', [])[:3]:  # Show first 3
                std = record.get('standardized', {})
                meta = record.get('metadata', {})
                name = std.get('name', '')
                country = std.get('country', '')
                iso2 = std.get('iso2', 'N/A')
                iso3 = std.get('iso3', 'N/A')
                completeness = record.get('completeness', 0)
                passes = meta.get('enrichment_passes', 0)
                
                print(f"    - {country or name}: ISO={iso2}/{iso3}, Completeness={completeness*100:.1f}%, Passes={passes}")
    
    # Check product file
    product_file = os.path.join(output_dir, "standardized_product.json")
    if os.path.exists(product_file):
        with open(product_file, 'r') as f:
            data = json.load(f)
            print("\n  Product Results:")
            for record in data.get('data', [])[:3]:  # Show first 3
                std = record.get('standardized', {})
                meta = record.get('metadata', {})
                hierarchy = std.get('hierarchy_path', '')
                category = std.get('product_category', 'Unknown')
                basel = std.get('basel_category', 'N/A')
                completeness = record.get('completeness', 0)
                passes = meta.get('enrichment_passes', 0)
                
                print(f"    - {hierarchy}: Category={category}, Basel={basel}, Completeness={completeness*100:.1f}%, Passes={passes}")


if __name__ == "__main__":
    try:
        test_agent1_multipass()
        print(f"\n{'='*60}")
        print("✓ Multi-pass enrichment test completed!")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")