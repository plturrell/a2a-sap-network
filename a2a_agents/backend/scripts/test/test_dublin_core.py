#!/usr/bin/env python3
"""
Test Dublin Core integration in ORD Registry
"""

import requests
import json
from datetime import datetime


def test_dublin_core_validation():
    """Test Dublin Core validation endpoint"""
    base_url = "http://localhost:8000/api/v1/ord"
    
    print("Testing Dublin Core Validation")
    print("=" * 50)
    
    # Test with complete Dublin Core metadata
    complete_dc = {
        "dublin_core": {
            "title": "Financial Data Processing API",
            "creator": ["FinSight CIB", "API Development Team"],
            "subject": ["financial-api", "data-processing", "rest-service"],
            "description": "Comprehensive API for processing financial transactions and data standardization",
            "publisher": "FinSight CIB Platform",
            "contributor": ["Architecture Team", "Security Team"],
            "date": datetime.utcnow().isoformat(),
            "type": "Service",
            "format": "REST",
            "identifier": "api-financial-processing-v1",
            "source": "Internal Development",
            "language": "en",
            "relation": ["part-of:financial-platform", "uses:ord-registry"],
            "coverage": "Global Financial Markets",
            "rights": "Internal Use Only - Proprietary"
        }
    }
    
    response = requests.post(
        f"{base_url}/dublincore/validate",
        json=complete_dc
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Validation successful!")
        print(f"  Valid: {result['valid']}")
        print(f"  Completeness: {result['metadata_completeness']:.2%}")
        print(f"\n  Quality Metrics:")
        metrics = result['quality_metrics']
        print(f"    - Overall Score: {metrics['overall_score']:.2f}")
        print(f"    - Completeness: {metrics['completeness']:.2f}")
        print(f"    - Accuracy: {metrics['accuracy']:.2f}")
        print(f"    - Consistency: {metrics['consistency']:.2f}")
        print(f"    - ISO 15836 Compliant: {metrics['iso15836_compliant']}")
        print(f"    - RFC 5013 Compliant: {metrics['rfc5013_compliant']}")
        
        if result.get('recommendations'):
            print(f"\n  Recommendations:")
            for rec in result['recommendations']:
                print(f"    - {rec}")
    else:
        print(f"\n✗ Validation failed: {response.status_code}")
        print(f"  Error: {response.text}")
    
    # Test with minimal Dublin Core
    print("\n\nTesting with Minimal Dublin Core")
    print("-" * 30)
    
    minimal_dc = {
        "dublin_core": {
            "title": "Test Resource",
            "creator": ["Test Creator"]
        }
    }
    
    response = requests.post(
        f"{base_url}/dublincore/validate",
        json=minimal_dc
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"  Completeness: {result['metadata_completeness']:.2%}")
        print(f"  Valid: {result['valid']}")
        if result.get('recommendations'):
            print(f"\n  Recommendations:")
            for rec in result['recommendations']:
                print(f"    - {rec}")


def test_ord_registration_with_dublin_core():
    """Test ORD registration with Dublin Core metadata"""
    base_url = "http://localhost:8000/api/v1/ord"
    
    print("\n\nTesting ORD Registration with Dublin Core")
    print("=" * 50)
    
    ord_document = {
        "ord_document": {
            "openResourceDiscovery": "1.5.0",
            "description": "Test ORD Document with Dublin Core",
            "dublinCore": {
                "title": "Test Financial API Collection",
                "creator": ["Test Team", "Development Department"],
                "subject": ["test", "financial", "api"],
                "description": "Test collection of financial APIs with Dublin Core metadata",
                "publisher": "Test Organization",
                "date": datetime.utcnow().isoformat(),
                "type": "Service",
                "format": "REST",
                "language": "en",
                "rights": "Test Rights Statement"
            },
            "apiResources": [
                {
                    "ordId": "com.test:api:financial-processing",
                    "title": "Financial Processing API",
                    "shortDescription": "API for financial data processing",
                    "description": "Complete API for processing financial transactions",
                    "version": "1.0.0",
                    "visibility": "internal",
                    "tags": ["financial", "processing", "api"]
                }
            ]
        },
        "registered_by": "test_user",
        "tags": ["test", "dublin-core"],
        "labels": {
            "environment": "test",
            "dublin_core_enabled": "true"
        }
    }
    
    response = requests.post(
        f"{base_url}/register",
        json=ord_document
    )
    
    if response.status_code == 200 or response.status_code == 201:
        result = response.json()
        print(f"\n✓ Registration successful!")
        print(f"  Registration ID: {result['registration_id']}")
        print(f"  Registry URL: {result['registry_url']}")
        
        if 'validation_results' in result:
            val = result['validation_results']
            print(f"\n  Validation Results:")
            print(f"    - Valid: {val['valid']}")
            print(f"    - Compliance Score: {val['compliance_score']:.2f}")
            
            if 'dublincore_validation' in val:
                dc_val = val['dublincore_validation']
                print(f"\n  Dublin Core Validation:")
                print(f"    - Overall Score: {dc_val['overall_score']:.2f}")
                print(f"    - ISO 15836 Compliant: {dc_val['iso15836_compliant']}")
        
        return result.get('registration_id')
    else:
        print(f"\n✗ Registration failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return None


def test_search_with_dublin_core_facets():
    """Test search with Dublin Core facets"""
    base_url = "http://localhost:8000/api/v1/ord"
    
    print("\n\nTesting Search with Dublin Core Facets")
    print("=" * 50)
    
    # First register some test data
    test_ord_registration_with_dublin_core()
    
    # Search with Dublin Core facets
    search_params = {
        "query": "financial",
        "includeDublinCore": True
    }
    
    response = requests.post(
        f"{base_url}/search",
        json=search_params
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Search successful!")
        print(f"  Total results: {result['total_count']}")
        
        if result.get('facets'):
            print(f"\n  Dublin Core Facets:")
            for facet_type, facet_values in result['facets'].items():
                if facet_values:
                    print(f"\n    {facet_type.capitalize()}:")
                    for facet in facet_values[:5]:
                        print(f"      - {facet['value']}: {facet['count']} occurrences")
    else:
        print(f"\n✗ Search failed: {response.status_code}")
        print(f"  Error: {response.text}")


def test_health_with_dublin_core_metrics():
    """Test health endpoint with Dublin Core metrics"""
    base_url = "http://localhost:8000/api/v1/ord"
    
    print("\n\nTesting Health Check with Dublin Core Metrics")
    print("=" * 50)
    
    response = requests.get(f"{base_url}/health")
    
    if response.status_code == 200:
        health = response.json()
        print(f"\n✓ Health check successful!")
        print(f"  Status: {health['status']}")
        
        print(f"\n  Services:")
        for service, status in health['services'].items():
            print(f"    - {service}: {status}")
        
        print(f"\n  Metrics:")
        metrics = health['metrics']
        print(f"    - Total Registrations: {metrics['total_registrations']}")
        print(f"    - Dublin Core Enabled: {metrics['dublin_core_enabled']}")
        print(f"    - Average Quality Score: {metrics['average_quality_score']:.2f}")
        
        print(f"\n  Standards Compliance:")
        compliance = health['standards_compliance']
        print(f"    - ISO 15836: {compliance['iso15836_compliance_rate']:.2%}")
        print(f"    - RFC 5013: {compliance['rfc5013_compliance_rate']:.2%}")
        print(f"    - ANSI/NISO: {compliance['ansi_niso_compliance_rate']:.2%}")
    else:
        print(f"\n✗ Health check failed: {response.status_code}")
        print(f"  Error: {response.text}")


if __name__ == "__main__":
    print("Dublin Core Integration Test Suite")
    print("==================================\n")
    
    # Run all tests
    test_dublin_core_validation()
    test_ord_registration_with_dublin_core()
    test_search_with_dublin_core_facets()
    test_health_with_dublin_core_metrics()
    
    print("\n\n✓ All Dublin Core tests completed!")