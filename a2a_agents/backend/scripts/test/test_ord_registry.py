#!/usr/bin/env python3
"""
Test script for ORD Registry
"""

import requests
import json
from datetime import datetime


def test_ord_registry():
    """Test ORD Registry functionality"""
    base_url = "http://localhost:8000/api/v1/ord"
    
    print("Testing ORD Registry")
    print("=" * 50)
    
    # 1. Test Health Check
    print("\n1. Testing Health Check...")
    response = requests.get(f"{base_url}/health")
    if response.status_code == 200:
        health = response.json()
        print(f"✓ Registry Status: {health['status']}")
        print(f"✓ Total Registrations: {health['metrics']['total_registrations']}")
        print(f"✓ Active Resources: {health['metrics']['active_resources']}")
    else:
        print(f"✗ Health check failed: {response.status_code}")
    
    # 2. Register a Sample ORD Document
    print("\n2. Testing ORD Document Registration...")
    
    # Create a sample ORD document for financial data
    ord_document = {
        "openResourceDiscovery": "1.5.0",
        "description": "Financial data products from CRD extraction",
        "dataProducts": [
            {
                "ordId": "com.finsight.cib:dataProduct:account_data",
                "title": "Account Hierarchy Data",
                "shortDescription": "Financial account hierarchy and classification data",
                "description": "Contains account hierarchies (L0-L3) with financial classifications including impairments, income, and costs",
                "version": "1.0.0",
                "visibility": "public",
                "partOfPackage": "com.finsight.cib:package:financial_data",
                "tags": ["financial", "accounts", "hierarchy", "crd"],
                "labels": {
                    "domain": "finance",
                    "source": "crd_extraction"
                },
                "documentationLabels": {
                    "dataStructure": "hierarchical",
                    "updateFrequency": "monthly"
                },
                "accessStrategies": [
                    {
                        "type": "open",
                        "supportedEnvironments": ["production", "staging"]
                    }
                ]
            },
            {
                "ordId": "com.finsight.cib:dataProduct:location_data",
                "title": "Location Master Data",
                "shortDescription": "Geographic location hierarchy data",
                "description": "Contains location hierarchies (L0-L4) for geographic regions and entities",
                "version": "1.0.0",
                "visibility": "public",
                "tags": ["location", "geography", "master-data"],
                "labels": {
                    "domain": "reference-data",
                    "source": "crd_extraction"
                }
            }
        ],
        "apiResources": [
            {
                "ordId": "com.finsight.cib:api:standardization_service",
                "title": "Data Standardization API",
                "shortDescription": "A2A agent API for data standardization",
                "description": "RESTful API exposed by the standardization agent",
                "version": "1.0.0",
                "visibility": "public",
                "tags": ["api", "standardization", "a2a"],
                "entryPoints": [
                    "http://localhost:8000/a2a/v1"
                ],
                "apiProtocol": "rest"
            }
        ],
        "entityTypes": [
            {
                "ordId": "com.finsight.cib:entityType:Account",
                "title": "Account Entity",
                "shortDescription": "Financial account entity",
                "description": "Represents a financial account in the hierarchy",
                "version": "1.0.0",
                "visibility": "public",
                "tags": ["entity", "account", "financial"]
            },
            {
                "ordId": "com.finsight.cib:entityType:Location",
                "title": "Location Entity",
                "shortDescription": "Geographic location entity",
                "description": "Represents a geographic location in the hierarchy",
                "version": "1.0.0",
                "visibility": "public",
                "tags": ["entity", "location", "geography"]
            }
        ]
    }
    
    registration_request = {
        "ord_document": ord_document,
        "registered_by": "test_script",
        "tags": ["test", "financial"],
        "labels": {
            "environment": "development",
            "project": "finsight_cib"
        }
    }
    
    response = requests.post(
        f"{base_url}/register",
        json=registration_request,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Registration successful!")
        print(f"  - Registration ID: {result['registration_id']}")
        print(f"  - Status: {result['status']}")
        print(f"  - Registry URL: {result['registry_url']}")
        print(f"  - Validation: {result['validation_results']['valid']}")
        
        registration_id = result['registration_id']
    else:
        print(f"✗ Registration failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return
    
    # 3. Search for Resources
    print("\n3. Testing Resource Search...")
    
    # Search by type
    search_request = {
        "resource_type": "dataProduct",
        "page": 1,
        "page_size": 10
    }
    
    response = requests.post(
        f"{base_url}/search",
        json=search_request
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"✓ Search completed")
        print(f"  - Total results: {results['total_count']}")
        print(f"  - Results on page: {len(results['results'])}")
        
        for resource in results['results']:
            print(f"  - {resource['title']} ({resource['ord_id']})")
    else:
        print(f"✗ Search failed: {response.status_code}")
    
    # 4. Get Specific Resource by ORD ID
    print("\n4. Testing Get Resource by ORD ID...")
    
    ord_id = "com.finsight.cib:dataProduct:account_data"
    response = requests.get(f"{base_url}/resources/{ord_id}")
    
    if response.status_code == 200:
        resource = response.json()
        print(f"✓ Resource found: {resource['title']}")
        print(f"  - Type: {resource['resource_type']}")
        print(f"  - Description: {resource['short_description']}")
        print(f"  - Tags: {', '.join(resource['tags'])}")
    else:
        print(f"✗ Resource not found: {response.status_code}")
    
    # 5. Get Registration Status
    print("\n5. Testing Registration Status...")
    
    response = requests.get(f"{base_url}/register/{registration_id}/status")
    
    if response.status_code == 200:
        status = response.json()
        print(f"✓ Registration Status: {status['status']}")
        print(f"  - Last Updated: {status['last_updated']}")
        print(f"  - Validation Valid: {status['validation']['valid']}")
    else:
        print(f"✗ Status check failed: {response.status_code}")
    
    # 6. Browse by Domain
    print("\n6. Testing Browse by Domain...")
    
    response = requests.get(f"{base_url}/browse?domain=finance")
    
    if response.status_code == 200:
        browse_results = response.json()
        print(f"✓ Browse results for domain 'finance': {browse_results['total_count']} items")
    else:
        print(f"✗ Browse failed: {response.status_code}")
    
    # 7. Test Validation with Invalid ORD
    print("\n7. Testing Invalid ORD Document...")
    
    invalid_ord = {
        "openResourceDiscovery": "1.5.0",
        "dataProducts": [
            {
                "ordId": "invalid-ord-id-format",  # Invalid format
                "title": "Test Product"
            }
        ]
    }
    
    invalid_request = {
        "ord_document": invalid_ord,
        "registered_by": "test_script"
    }
    
    response = requests.post(
        f"{base_url}/register",
        json=invalid_request
    )
    
    if response.status_code == 400:
        print("✓ Invalid ORD correctly rejected")
    else:
        print(f"✗ Expected 400 error, got: {response.status_code}")
    
    print("\n" + "=" * 50)
    print("ORD Registry testing complete!")


if __name__ == "__main__":
    test_ord_registry()