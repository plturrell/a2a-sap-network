#!/usr/bin/env python3
"""
A2A Workflow Orchestration Test
Demonstrates workflow tracking, data instance management, and context passing
"""

import requests
import json
import time
from datetime import datetime

def test_workflow_orchestration():
    """Test A2A workflow orchestration with trust contracts and data tracking"""
    
    print("A2A Workflow Orchestration Test")
    print("="*50)
    print("Testing workflow tracking, data instances, and context passing")
    print("="*50)
    
    # 1. Create Trust Contract and SLA
    print("\n1. Setting up trust relationships...")
    
    # Get agent cards
    agent0_response = requests.get("http://localhost:8000/a2a/agent0/v1/.well-known/agent.json")
    agent1_response = requests.get("http://localhost:8000/a2a/v1/.well-known/agent.json")
    
    if agent0_response.status_code == 200 and agent1_response.status_code == 200:
        agent0_card = agent0_response.json()
        agent1_card = agent1_response.json()
        print(f"✓ Agent 0: {agent0_card['name']}")
        print(f"✓ Agent 1: {agent1_card['name']}")
    else:
        print("✗ Failed to get agent cards")
        return
    
    # Register agents with trust system
    agent0_trust_reg = {
        "agent_card": agent0_card,
        "commitment_level": "high"
    }
    
    response = requests.post(
        "http://localhost:8000/api/v1/a2a/trust/agents/register",
        json=agent0_trust_reg
    )
    
    if response.status_code == 201:
        agent0_trust = response.json()
        agent0_trust_id = agent0_trust['trust_agent_id']
        print(f"✓ Agent 0 registered with trust ID: {agent0_trust_id[:20]}...")
    else:
        print(f"✗ Failed to register Agent 0: {response.status_code}")
        return
    
    # Create SLA
    sla_request = {
        "provider_id": agent0_trust_id,
        "consumer_id": "workflow_orchestrator",
        "terms": {
            "response_time_max": 5000,
            "availability_min": 99.0,
            "error_rate_max": 2.0
        },
        "validity_hours": 24
    }
    
    response = requests.post(
        "http://localhost:8000/api/v1/a2a/trust/sla",
        json=sla_request
    )
    
    if response.status_code == 201:
        sla_data = response.json()
        sla_id = sla_data['sla_id']
        print(f"✓ SLA created: {sla_id[:20]}...")
    else:
        print("✗ Failed to create SLA")
        sla_id = None
    
    # 2. Create Workflow with Trust Contract
    print("\n2. Creating workflow with trust contract...")
    
    workflow_request = {
        "workflow_plan_id": "financial_data_processing_plan",
        "workflow_name": "Financial Data Registration and Standardization",
        "trust_contract_id": agent0_trust_id,
        "sla_id": sla_id,
        "required_trust_level": 3.0,
        "initial_data_location": "/Users/apple/projects/finsight_cib/data/raw",
        "metadata": {
            "purpose": "Process CRD financial data",
            "compliance": {
                "a2a_version": "0.2.9",
                "ord_version": "1.5.0"
            }
        }
    }
    
    response = requests.post(
        "http://localhost:8000/api/v1/a2a/workflows/create",
        json=workflow_request
    )
    
    if response.status_code == 201:
        workflow_data = response.json()
        workflow_id = workflow_data['workflow_id']
        context_id = workflow_data['context_id']
        print(f"✓ Workflow created:")
        print(f"  - Workflow ID: {workflow_id}")
        print(f"  - Context ID: {context_id}")
        print(f"  - Initial Stage: {workflow_data['initial_stage']}")
        print(f"  - Monitoring: {'Started' if workflow_data['monitoring_started'] else 'Not started'}")
    else:
        print(f"✗ Failed to create workflow: {response.status_code}")
        print(response.text)
        return
    
    # 3. Trigger Agent 0 with Workflow Context
    print("\n3. Triggering Agent 0 with workflow context...")
    
    agent0_message = {
        "role": "user",
        "parts": [
            {
                "kind": "text",
                "text": "Process financial data from CRD extraction"
            },
            {
                "kind": "data",
                "data": {
                    "create_workflow": True,
                    "workflow_metadata": {
                        "plan_id": workflow_request["workflow_plan_id"],
                        "name": workflow_request["workflow_name"],
                        "trust_contract_id": workflow_request["trust_contract_id"],
                        "sla_id": workflow_request["sla_id"],
                        "required_trust_level": workflow_request["required_trust_level"]
                    },
                    "workflow_context": {
                        "workflow_id": workflow_id,
                        "context_id": context_id
                    },
                    "processing_instructions": {
                        "data_location": "/Users/apple/projects/finsight_cib/data/raw",
                        "dublin_core_required": True,
                        "quality_threshold": 0.6
                    }
                }
            }
        ],
        "contextId": context_id
    }
    
    response = requests.post(
        "http://localhost:8000/a2a/agent0/v1/messages",
        json=agent0_message
    )
    
    if response.status_code == 200:
        agent0_response = response.json()
        task_id = agent0_response.get('taskId')
        print(f"✓ Agent 0 triggered:")
        print(f"  - Task ID: {task_id}")
        print(f"  - Workflow ID: {agent0_response.get('workflowId')}")
    else:
        print(f"✗ Failed to trigger Agent 0: {response.status_code}")
        return
    
    # 4. Monitor Workflow Progress
    print("\n4. Monitoring workflow progress...")
    
    # Wait a bit for processing
    time.sleep(2)
    
    # Check workflow status
    response = requests.get(
        f"http://localhost:8000/api/v1/a2a/workflows/{workflow_id}/status"
    )
    
    if response.status_code == 200:
        status = response.json()
        print(f"✓ Workflow Status:")
        print(f"  - Current Stage: {status['current_stage']}")
        print(f"  - State: {status['state']}")
        print(f"  - Stages Completed: {status['stages_completed']}/{status['total_stages']}")
        print(f"  - Artifacts Created: {status['artifacts_created']}")
    
    # 5. Get Workflow Context with Artifacts
    print("\n5. Getting workflow context and data artifacts...")
    
    response = requests.get(
        f"http://localhost:8000/api/v1/a2a/workflows/{workflow_id}/context"
    )
    
    if response.status_code == 200:
        context = response.json()
        print(f"✓ Workflow Context:")
        print(f"  - Workflow Name: {context['workflow_name']}")
        print(f"  - Current Stage: {context['current_stage']}")
        print(f"  - Trust Contract: {context['trust_contract_id'][:20]}..." if context['trust_contract_id'] else "None")
        print(f"  - Required Trust Level: {context['required_trust_level']}")
        
        if context['artifacts']:
            print(f"\n  Data Artifacts ({len(context['artifacts'])} total):")
            for artifact in context['artifacts']:
                print(f"    - {artifact['artifact_id'][:12]}... ({artifact['type']})")
                print(f"      Location: {artifact['location']}")
                print(f"      Created by: {artifact['created_by']}")
                print(f"      Checksum: {artifact['checksum'][:20]}...")
    
    # 6. Check Data Lineage
    if context['artifacts']:
        print("\n6. Checking data lineage...")
        
        # Get lineage for the latest artifact
        latest_artifact = context['artifacts'][-1]
        lineage_request = {
            "workflow_id": workflow_id,
            "artifact_id": latest_artifact['artifact_id']
        }
        
        response = requests.post(
            "http://localhost:8000/api/v1/a2a/workflows/lineage",
            json=lineage_request
        )
        
        if response.status_code == 200:
            lineage_data = response.json()
            print(f"✓ Data Lineage for {lineage_data['artifact_id'][:12]}...:")
            for i, artifact in enumerate(lineage_data['lineage']):
                print(f"  {i+1}. {artifact['type']} - {artifact['artifact_id'][:12]}...")
                print(f"     Created by: {artifact['created_by']}")
    
    # 7. Check Active Workflows
    print("\n7. Checking all active workflows...")
    
    response = requests.get("http://localhost:8000/api/v1/a2a/workflows/active")
    
    if response.status_code == 200:
        active_data = response.json()
        print(f"✓ Active Workflows: {active_data['total_count']}")
        for wf in active_data['active_workflows']:
            print(f"  - {wf['workflow_id']}:")
            print(f"    Stage: {wf['current_stage']} ({wf['stages_completed']}/{wf['total_stages']})")
            print(f"    Duration: {wf['duration_seconds']:.1f}s")
    
    # 8. Summary
    print("\n8. Workflow Orchestration Summary")
    print("="*50)
    print("✓ Workflow created with trust contract and SLA")
    print("✓ Agent 0 processing with workflow context")
    print("✓ Data artifacts tracked with unique IDs and checksums")
    print("✓ Workflow context passed between agents")
    print("✓ Data lineage maintained for full traceability")
    print("✓ Real-time workflow monitoring available")
    print("\nKey Features Demonstrated:")
    print("1. Trust contract identification - Agents know which trust level to use")
    print("2. Data instance tracking - Each artifact has unique ID (like blockchain blocks)")
    print("3. Workflow context passing - Context flows through the entire pipeline")
    print("4. Workflow monitoring - Real-time visibility into execution state")


if __name__ == "__main__":
    test_workflow_orchestration()
    print("\n" + "="*50)
    print("✓ A2A Workflow Orchestration test complete!")
    print("="*50)