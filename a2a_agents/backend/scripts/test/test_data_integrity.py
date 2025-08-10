#!/usr/bin/env python3
"""
Test script to check data integrity and row counts throughout the pipeline
"""
import asyncio
import json
import pandas as pd
import os
from datetime import datetime
import httpx

async def test_data_integrity():
    """Test the data integrity tracking system"""
    
    # Check raw data first
    raw_data_path = "/Users/apple/projects/finsight_cib/data/raw"
    
    print("=== RAW DATA ANALYSIS ===")
    raw_counts = {}
    
    for filename in os.listdir(raw_data_path):
        if filename.endswith('.csv') and filename.startswith('CRD_'):
            file_path = os.path.join(raw_data_path, filename)
            try:
                df = pd.read_csv(file_path)
                data_type = filename.replace('CRD_Extraction_v1_', '').replace('_sorted.csv', '')
                raw_counts[data_type] = len(df)
                print(f"Raw {data_type}: {len(df)} rows")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    print(f"\nTotal raw records: {sum(raw_counts.values())}")
    
    # Check Agent 1 output files
    print("\n=== AGENT 1 OUTPUT ANALYSIS ===")
    output_path = "/Users/apple/projects/finsight_cib/data/interim/1/dataStandardization"
    standardized_counts = {}
    
    if os.path.exists(output_path):
        for filename in os.listdir(output_path):
            if filename.startswith('standardized_') and filename.endswith('.json'):
                file_path = os.path.join(output_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    data_type = filename.replace('standardized_', '').replace('.json', '')
                    record_count = len(data.get('data', []))
                    metadata_count = data.get('metadata', {}).get('records', 0)
                    
                    standardized_counts[data_type] = record_count
                    print(f"Standardized {data_type}: {record_count} rows (metadata says {metadata_count})")
                    
                    # Check integrity info if available
                    if 'integrity' in data.get('metadata', {}):
                        integrity = data['metadata']['integrity']
                        print(f"  - Integrity hash: {integrity.get('dataset_hash', 'N/A')[:16]}...")
                        print(f"  - Row count check: {integrity.get('row_count', 'N/A')}")
                    
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        
        print(f"\nTotal standardized records: {sum(standardized_counts.values())}")
    else:
        print("No Agent 1 output directory found")
    
    # Compare counts
    print("\n=== COMPARISON ===")
    for data_type in raw_counts:
        raw = raw_counts.get(data_type, 0)
        standardized = standardized_counts.get(data_type, 0)
        if raw != standardized:
            print(f"❌ {data_type}: Raw({raw}) vs Standardized({standardized}) - LOSS: {raw - standardized}")
        else:
            print(f"✅ {data_type}: {raw} rows (no loss)")
    
    # Test Agent 1 health
    print("\n=== AGENT 1 HEALTH CHECK ===")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8001/health")
            if response.status_code == 200:
                health = response.json()
                print(f"Agent 1 Status: {health.get('status', 'unknown')}")
                print(f"Agent Version: {health.get('version', 'unknown')}")
            else:
                print(f"Agent 1 health check failed: {response.status_code}")
    except Exception as e:
        print(f"Cannot reach Agent 1: {e}")

if __name__ == "__main__":
    asyncio.run(test_data_integrity())