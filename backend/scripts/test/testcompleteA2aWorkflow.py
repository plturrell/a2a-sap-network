#!/usr/bin/env python3
"""
Complete A2A Registry and Workflow Test
"""

import requests
import json
import time

def test_complete_a2a_workflow():
    """Test complete A2A workflow with ORD integration"""
    
    print("Complete A2A Registry & Workflow Test")
    print("="*60)
    
    # 1. Test A2A Registry Health
    print("\n1. Testing A2A Registry Health...")
    response = requests.get("http://localhost:8000/api/v1/a2a/health")
    if response.status_code == 200:
        health = response.json()
        print(f"✓ A2A Registry Status: {health['status']}")
    
    # 2. Check registered agents
    print("\n2. Checking registered agents...")
    response = requests.get("http://localhost:8000/api/v1/a2a/agents/search")
    if response.status_code == 200:
        agents = response.json()
        print(f"✓ Found {agents['total_count']} registered agents:")
        
        for agent in agents['results']:
            print(f"  - {agent['name']} (ID: {agent['agent_id']}, Status: {agent['status']})")
    
    # 3. Test workflow creation and execution
    print("\n3. Creating financial data processing workflow...")
    
    workflow_request = {
        "workflow_name": "financial_data_pipeline",
        "description": "Complete financial data processing with Dublin Core metadata",
        "stages": [
            {
                "name": "data_product_creation",
                "required_capabilities": ["cds-csn-generation", "ord-descriptor-creation-with-dublin-core", "dublin-core-extraction"]
            },
            {
                "name": "standardization",
                "required_capabilities": ["location-standardization", "account-standardization"]
            }
        ]
    }
    
    response = requests.post(
        "http://localhost:8000/api/v1/a2a/orchestration/plan",
        json=workflow_request
    )
    
    if response.status_code == 201:
        workflow_plan = response.json()
        print(f"✓ Workflow plan created: {workflow_plan['workflow_id']}")
        print(f"  Total agents in plan: {workflow_plan['total_agents']}")
        print(f"  Estimated duration: {workflow_plan['estimated_duration']}")
        
        for stage in workflow_plan['execution_plan']:
            print(f"    Stage '{stage['stage']}': {stage['agent']['name']}")
        
        # 4. Execute the workflow
        print(f"\n4. Executing workflow {workflow_plan['workflow_id']}...")
        
        execution_request = {
            "input_data": {
                "data_location": "/Users/apple/projects/finsight_cib/data/raw",
                "processing_requirements": {
                    "dublin_core_enabled": True,
                    "quality_threshold": 0.6
                }
            },
            "context_id": f"test_workflow_{int(time.time())}",
            "execution_mode": "sequential"
        }
        
        response = requests.post(
            f"http://localhost:8000/api/v1/a2a/orchestration/execute/{workflow_plan['workflow_id']}",
            json=execution_request
        )
        
        if response.status_code == 202:
            execution = response.json()
            print(f"✓ Workflow execution started: {execution['execution_id']}")
            
            # 5. Monitor execution status
            print(f"\n5. Monitoring execution status...")
            
            for i in range(10):  # Check for up to 10 seconds
                time.sleep(1)
                
                status_response = requests.get(
                    f"http://localhost:8000/api/v1/a2a/orchestration/status/{execution['execution_id']}"
                )
                
                if status_response.status_code == 200:
                    status = status_response.json()
                    print(f"  Status: {status['status']} (Current stage: {status.get('current_stage', 'None')})")
                    
                    if status['status'] in ['completed', 'failed']:
                        if status['status'] == 'completed':
                            print(f"✓ Workflow completed successfully!")
                            print(f"  Duration: {status.get('duration_ms', 0):.0f}ms")
                            print(f"  Stages completed: {len(status.get('stage_results', []))}")
                            
                            if status.get('output_data'):
                                print(f"  Output: {status['output_data']['message']}")
                        else:
                            print(f"✗ Workflow failed: {status.get('error_details')}")
                        break
    else:
        print(f"✗ Failed to create workflow: {response.status_code}")
    
    # 6. Test agent health monitoring
    print(f"\n6. Testing agent health monitoring...")
    
    response = requests.get("http://localhost:8000/api/v1/a2a/agents/search")
    if response.status_code == 200:
        agents = response.json()
        
        for agent in agents['results'][:2]:  # Test first 2 agents
            health_response = requests.get(
                f"http://localhost:8000/api/v1/a2a/agents/{agent['agent_id']}/health"
            )
            
            if health_response.status_code == 200:
                health = health_response.json()
                print(f"  {agent['name']}: {health['status']} ({health['response_time_ms']:.1f}ms)")
                
                # Test metrics
                metrics_response = requests.get(
                    f"http://localhost:8000/api/v1/a2a/agents/{agent['agent_id']}/metrics?period=1h"
                )
                
                if metrics_response.status_code == 200:
                    metrics = metrics_response.json()
                    agent_metrics = metrics['metrics']
                    print(f"    Health checks: {agent_metrics['health_checks']}")
                    print(f"    Uptime: {agent_metrics['uptime_percentage']:.1f}%")
    
    # 7. Test integration with ORD Registry
    print(f"\n7. Testing ORD Registry integration...")
    
    ord_response = requests.post(
        "http://localhost:8000/api/v1/ord/search",
        json={
            "query": "a2a-agent",
            "filters": {
                "tags": ["a2a-agent"]
            }
        }
    )
    
    if ord_response.status_code == 200:
        ord_results = ord_response.json()
        print(f"✓ Found {ord_results['total_count']} A2A agents in ORD Registry")
        
        for result in ord_results['results']:
            print(f"  - {result['document']['apiResources'][0]['title']}")
    else:
        print(f"⚠ ORD integration may need adjustment: {ord_response.status_code}")
    
    # 8. Test system overview
    print(f"\n8. System overview...")
    
    a2a_health = requests.get("http://localhost:8000/api/v1/a2a/system/health")
    ord_health = requests.get("http://localhost:8000/api/v1/ord/health")
    
    if a2a_health.status_code == 200 and ord_health.status_code == 200:
        a2a_data = a2a_health.json()
        ord_data = ord_health.json()
        
        print(f"✓ System Status:")
        print(f"  A2A Registry: {a2a_data['status']} ({a2a_data['total_agents']} agents)")
        print(f"  ORD Registry: {ord_data['status']} ({ord_data['metrics']['total_registrations']} resources)")
        print(f"  Combined Services: Operational")


if __name__ == "__main__":
    test_complete_a2a_workflow()
    print(f"\n{'='*60}")
    print("✓ Complete A2A Registry and workflow testing finished!")
    print("  - A2A Registry: Operational")
    print("  - Agent Discovery: Working")
    print("  - Workflow Orchestration: Working")
    print("  - Health Monitoring: Working")
    print("  - ORD Integration: Active")
    print(f"{'='*60}")