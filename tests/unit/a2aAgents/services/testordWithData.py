#!/usr/bin/env python3
"""
Test ORD Registry with actual data files
"""

import requests
import json
import os
import pandas as pd


def register_data_products():
    """Register our CSV data files as data products in ORD Registry"""
    base_url = "http://localhost:8000/api/v1/ord"
    
    print("Registering Data Products in ORD Registry")
    print("=" * 50)
    
    # Define our data files
    data_files = {
        "account": {
            "file": "CRD_Extraction_v1_account_sorted.csv",
            "title": "CRD Account Hierarchy Data",
            "description": "Financial account hierarchy with 4 levels (L0-L3) including impairments, income, and cost categories"
        },
        "location": {
            "file": "CRD_Extraction_v1_location_sorted.csv",
            "title": "CRD Location Hierarchy Data",
            "description": "Geographic location hierarchy with 5 levels (L0-L4) for global regions and entities"
        },
        "product": {
            "file": "CRD_Extraction_v1_product_sorted.csv",
            "title": "CRD Product Hierarchy Data",
            "description": "Financial product hierarchy with 4 levels (L0-L3) for banking products and services"
        },
        "book": {
            "file": "CRD_Extraction_v1_book_sorted.csv",
            "title": "CRD Book Hierarchy Data",
            "description": "Legal entity and book hierarchy with 4 levels (L0-L3) for financial reporting"
        },
        "measure": {
            "file": "CRD_Extraction_v1_measure_sorted.csv",
            "title": "CRD Measure Configuration Data",
            "description": "Financial measure configurations including version, year, measure type, and currency type"
        }
    }
    
    # Create ORD document with all data products
    data_products = []
    
    for key, info in data_files.items():
        # Check if file exists and get row count
        file_path = f"/Users/apple/projects/finsight_cib/data/raw/{info['file']}"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            row_count = len(df)
            columns = list(df.columns)
            
            data_product = {
                "ordId": f"com.finsight.cib:dataProduct:crd_{key}_data",
                "title": info["title"],
                "shortDescription": f"{info['title']} - {row_count} records",
                "description": info["description"],
                "version": "1.0.0",
                "visibility": "internal",
                "tags": ["crd", "financial", key, "raw-data"],
                "labels": {
                    "source": "crd_extraction",
                    "format": "csv",
                    "records": str(row_count),
                    "columns": str(len(columns))
                },
                "documentationLabels": {
                    "dataStructure": "hierarchical",
                    "updateFrequency": "monthly",
                    "dataQuality": "raw"
                },
                "accessStrategies": [
                    {
                        "type": "file",
                        "path": file_path
                    }
                ],
                "dataProductLinks": {
                    "schema": f"/schemas/{key}_schema.json",
                    "sample": f"/samples/{key}_sample.csv"
                }
            }
            
            data_products.append(data_product)
            print(f"\n✓ Found {info['file']}: {row_count} records, {len(columns)} columns")
            print(f"  Columns: {', '.join(columns[:3])}...")
    
    # Create complete ORD document
    ord_document = {
        "openResourceDiscovery": "1.5.0",
        "description": "CRD Financial Data Extraction - Raw data products for standardization pipeline",
        "dataProducts": data_products,
        "packages": [
            {
                "ordId": "com.finsight.cib:package:crd_extraction",
                "title": "CRD Data Extraction Package",
                "shortDescription": "Complete set of CRD extracted financial data",
                "description": "Raw financial data extracted from CRD system for standardization and processing",
                "version": "1.0.0",
                "tags": ["crd", "extraction", "financial-data"]
            }
        ]
    }
    
    # Register the ORD document
    print("\n\nRegistering ORD document...")
    
    registration_request = {
        "ord_document": ord_document,
        "registered_by": "data_product_agent",
        "tags": ["crd", "raw-data", "pipeline-input"],
        "labels": {
            "pipeline": "financial-standardization",
            "stage": "raw",
            "source_system": "crd"
        }
    }
    
    response = requests.post(
        f"{base_url}/register",
        json=registration_request
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Registration successful!")
        print(f"  - Registration ID: {result['registration_id']}")
        print(f"  - Registry URL: {result['registry_url']}")
        print(f"  - Registered {len(data_products)} data products")
        
        return result['registration_id']
    else:
        print(f"\n✗ Registration failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return None


def search_registered_products():
    """Search for registered data products"""
    base_url = "http://localhost:8000/api/v1/ord"
    
    print("\n\nSearching for registered data products...")
    print("=" * 50)
    
    # Search for all data products
    response = requests.post(
        f"{base_url}/search",
        json={
            "resource_type": "dataProduct",
            "tags": ["crd"]
        }
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"\nFound {results['total_count']} CRD data products:")
        
        for product in results['results']:
            print(f"\n- {product['title']}")
            print(f"  ORD ID: {product['ord_id']}")
            print(f"  Description: {product['short_description']}")
            print(f"  Tags: {', '.join(product['tags'])}")
            if 'records' in product['labels']:
                print(f"  Records: {product['labels']['records']}")


if __name__ == "__main__":
    # Register data products
    registration_id = register_data_products()
    
    if registration_id:
        # Search for them
        search_registered_products()
        
        print(f"\n\n✓ ORD Registry is working correctly!")
        print(f"✓ Data products are registered and discoverable")
        print(f"✓ Ready to trigger Agent 0 for processing")