#!/usr/bin/env python3
"""
Complete Dublin Core Integration Test
Tests the full workflow from Agent 0 to ORD Registry
"""

import requests
import json
import time
from datetime import datetime


def test_complete_dublin_core_workflow():
    """Test the complete Dublin Core workflow"""
    
    print("Complete Dublin Core Workflow Test")
    print("=" * 70)
    
    # Step 1: Test ORD Registry Dublin Core support
    print("\n1. Testing ORD Registry Dublin Core Support...")
    
    # Test validation endpoint
    validation_request = {
        "dublin_core": {
            "title": "Test Financial Dataset",
            "creator": ["Test Team"],
            "subject": ["test", "financial"],
            "description": "Test dataset for Dublin Core integration",
            "publisher": "Test Publisher",
            "date": datetime.utcnow().isoformat(),
            "type": "Dataset",
            "format": "CSV",
            "identifier": f"test-{int(time.time())}",
            "language": "en",
            "rights": "Test Rights"
        }
    }
    
    response = requests.post(
        "http://localhost:8000/api/v1/ord/dublincore/validate",
        json=validation_request
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"  ✓ Dublin Core Validation: Valid={result['valid']}")
        print(f"    Completeness: {result['metadata_completeness']:.2%}")
        print(f"    Quality Score: {result['quality_metrics']['overall_score']:.2f}")
    else:
        print(f"  ✗ Validation failed: {response.status_code}")
    
    # Step 2: Test Agent 0 Dublin Core extraction
    print("\n2. Testing Agent 0 Dublin Core Extraction...")
    
    agent0_message = {
        "message": {
            "role": "user",
            "parts": [{
                "kind": "text",
                "text": "Process financial data with Dublin Core metadata"
            }]
        },
        "contextId": f"test_complete_{int(time.time())}"
    }
    
    response = requests.post(
        "http://localhost:8000/a2a/agent0/v1/messages",
        json=agent0_message
    )
    
    if response.status_code == 200:
        result = response.json()
        task_id = result['taskId']
        print(f"  ✓ Agent 0 Task Created: {task_id}")
        
        # Wait for completion
        print("    Waiting for processing", end='', flush=True)
        registration_id = None
        for i in range(20):
            time.sleep(1)
            print(".", end='', flush=True)
            
            status_response = requests.get(
                f"http://localhost:8000/a2a/agent0/v1/tasks/{task_id}"
            )
            
            if status_response.status_code == 200:
                task_status = status_response.json()
                if task_status['status']['state'] == 'completed':
                    print("\n  ✓ Agent 0 Processing Complete")
                    
                    # Extract registration ID
                    if task_status.get('artifacts'):
                        data = task_status['artifacts'][0]['parts'][0]['data']
                        if 'catalog_registration' in data:
                            registration_id = data['catalog_registration'].get('registration_id')
                            print(f"    Registration ID: {registration_id}")
                        
                        # Show Dublin Core quality
                        if 'dublin_core_quality' in data:
                            quality = data['dublin_core_quality']
                            print(f"    Dublin Core Quality: {quality['overall_score']:.2f}")
                            print(f"    Standards Compliant: ISO={quality['standards_compliance']['iso15836_compliant']}")
                    break
                elif task_status['status']['state'] == 'failed':
                    print("\n  ✗ Agent 0 Processing Failed")
                    break
    else:
        print(f"  ✗ Failed to create Agent 0 task: {response.status_code}")
    
    # Step 3: Test ORD search with Dublin Core facets
    print("\n3. Testing ORD Search with Dublin Core Facets...")
    
    search_request = {
        "query": "financial",
        "includeDublinCore": True,
        "filters": {
            "tags": ["crd", "financial"]
        }
    }
    
    response = requests.post(
        "http://localhost:8000/api/v1/ord/search",
        json=search_request
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"  ✓ Search Results: {result['total_count']} items found")
        
        if result.get('facets'):
            print("    Dublin Core Facets:")
            for facet_type in ['creators', 'subjects', 'types', 'formats']:
                if facet_type in result['facets'] and result['facets'][facet_type]:
                    print(f"      {facet_type.capitalize()}:")
                    for facet in result['facets'][facet_type][:3]:
                        print(f"        - {facet['value']}: {facet['count']}")
        
        # Check if results have Dublin Core
        if result.get('results') and result['results']:
            first_result = result['results'][0]
            if 'dublinCore' in first_result.get('document', {}):
                print("  ✓ Search results include Dublin Core metadata")
    else:
        print(f"  ✗ Search failed: {response.status_code}")
    
    # Step 4: Test ORD health metrics
    print("\n4. Testing ORD Health with Dublin Core Metrics...")
    
    response = requests.get("http://localhost:8000/api/v1/ord/health")
    
    if response.status_code == 200:
        health = response.json()
        print(f"  ✓ Health Status: {health['status']}")
        
        metrics = health.get('metrics', {})
        if 'dublin_core_enabled' in metrics:
            print(f"    Dublin Core Enabled: {metrics['dublin_core_enabled']}")
            print(f"    Average Quality Score: {metrics['average_quality_score']:.2f}")
        
        compliance = health.get('standards_compliance', {})
        if compliance:
            print("    Standards Compliance:")
            print(f"      ISO 15836: {compliance.get('iso15836_compliance_rate', 0):.2%}")
            print(f"      RFC 5013: {compliance.get('rfc5013_compliance_rate', 0):.2%}")
    else:
        print(f"  ✗ Health check failed: {response.status_code}")
    
    # Step 5: Test metrics endpoint
    print("\n5. Testing Metrics Endpoint...")
    
    response = requests.get("http://localhost:8000/api/v1/ord/metrics")
    
    if response.status_code == 200:
        metrics = response.json()
        print(f"  ✓ Metrics Retrieved")
        
        if 'dublin_core_metrics' in metrics:
            dc_metrics = metrics['dublin_core_metrics']
            print("    Dublin Core Metrics:")
            print(f"      Resources with metadata: {dc_metrics.get('resources_with_dublin_core', 0)}")
            print(f"      Average completeness: {dc_metrics.get('average_completeness', 0):.2%}")
            print(f"      Quality distribution:")
            for range_key, count in dc_metrics.get('quality_distribution', {}).items():
                print(f"        {range_key}: {count}")
    else:
        print(f"  ✗ Metrics retrieval failed: {response.status_code}")


if __name__ == "__main__":
    test_complete_dublin_core_workflow()
    print("\n\n✓ Complete Dublin Core workflow test finished!")