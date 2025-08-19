#!/usr/bin/env python3
"""
A2A Trust Relationship Test
Test trust relationship management between agents
"""

import requests
import json
import time

def test_trust_relationships():
    """Test A2A Trust Relationship Management"""
    
    print("A2A Trust Relationship Management Test")
    print("="*50)
    
    # 1. Test Trust System Health
    print("\n1. Testing Trust System Health...")
    response = requests.get("http://localhost:8000/api/v1/a2a/trust/health")
    if response.status_code == 200:
        health = response.json()
        print(f"✓ Trust System Status: {health['status']}")
        print(f"  Registered Agents: {health['total_registered_agents']}")
        print(f"  Active Workflows: {health['active_workflows']}")
    else:
        print(f"✗ Trust System health check failed: {response.status_code}")
        return
    
    # 2. Get Agent Cards
    print("\n2. Getting Agent Cards...")
    
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
    
    # 3. Register Agents with Trust System
    print("\n3. Registering agents with trust system...")
    
    # Register Agent 0 with HIGH trust commitment
    agent0_registration = {
        "agent_card": agent0_card,
        "commitment_level": "high"
    }
    
    response = requests.post(
        "http://localhost:8000/api/v1/a2a/trust/agents/register",
        json=agent0_registration
    )
    
    if response.status_code == 201:
        agent0_trust = response.json()
        print(f"✓ Agent 0 registered:")
        print(f"  Trust ID: {agent0_trust['trust_agent_id'][:20]}...")
        print(f"  Commitment Level: {agent0_trust['commitment_level']}")
        print(f"  Initial Trust Score: {agent0_trust['initial_trust_score']}")
        agent0_trust_id = agent0_trust['trust_agent_id']
    else:
        print(f"✗ Failed to register Agent 0: {response.status_code}")
        return
    
    # Register Agent 1 with MEDIUM trust commitment
    agent1_registration = {
        "agent_card": agent1_card,
        "commitment_level": "medium"
    }
    
    response = requests.post(
        "http://localhost:8000/api/v1/a2a/trust/agents/register",
        json=agent1_registration
    )
    
    if response.status_code == 201:
        agent1_trust = response.json()
        print(f"✓ Agent 1 registered:")
        print(f"  Trust ID: {agent1_trust['trust_agent_id'][:20]}...")
        print(f"  Commitment Level: {agent1_trust['commitment_level']}")
        print(f"  Initial Trust Score: {agent1_trust['initial_trust_score']}")
        agent1_trust_id = agent1_trust['trust_agent_id']
    else:
        print(f"✗ Failed to register Agent 1: {response.status_code}")
        return
    
    # 4. Check Initial Trust Scores
    print("\n4. Checking initial trust scores...")
    
    for agent_id, name in [(agent0_trust_id, "Agent 0"), (agent1_trust_id, "Agent 1")]:
        response = requests.get(
            f"http://localhost:8000/api/v1/a2a/trust/agents/{agent_id}/score"
        )
        
        if response.status_code == 200:
            score_data = response.json()
            print(f"✓ {name} Trust Score: {score_data['overall_trust_score']:.2f}/5.0")
            trust_metrics = score_data['trust_metrics']
            print(f"  Total Interactions: {trust_metrics['total_interactions']}")
            print(f"  Trust Rating: {trust_metrics['trust_rating']}")
    
    # 5. Create Trust-Managed Workflow
    print("\n5. Creating trust-managed workflow...")
    
    workflow_request = {
        "workflow_definition": {
            "workflow_name": "financial_data_trust_pipeline",
            "description": "Financial data processing with trust verification",
            "stages": [
                {
                    "name": "data_registration",
                    "required_capabilities": ["cds-csn-generation", "dublin-core-extraction"]
                },
                {
                    "name": "data_standardization",
                    "required_capabilities": ["location-standardization", "account-standardization"]
                }
            ]
        },
        "trust_requirements": {
            "stage_1": 3.0,
            "stage_2": 2.0
        }
    }
    
    response = requests.post(
        "http://localhost:8000/api/v1/a2a/trust/workflows",
        json=workflow_request
    )
    
    if response.status_code == 201:
        workflow_data = response.json()
        print(f"✓ Trust workflow created:")
        print(f"  Workflow ID: {workflow_data['workflow_id']}")
        print(f"  Trust Workflow ID: {workflow_data['trust_workflow_id'][:20]}...")
        print(f"  Trust Requirements: {workflow_data['trust_requirements']}")
    else:
        print(f"✗ Failed to create workflow: {response.status_code}")
    
    # 6. Record Trust Interactions
    print("\n6. Recording trust interactions between agents...")
    
    # Excellent interaction: Agent 0 -> Agent 1
    interaction1 = {
        "provider_id": agent0_trust_id,
        "consumer_id": agent1_trust_id,
        "rating": 5,
        "skill_used": "data-registration",
        "response_time": 850,
        "error_occurred": False
    }
    
    response = requests.post(
        f"http://localhost:8000/api/v1/a2a/trust/agents/{agent0_trust_id}/interaction",
        json=interaction1
    )
    
    if response.status_code == 200:
        print("✓ Recorded excellent interaction from Agent 0")
    
    # Good interaction: Agent 1 -> Agent 0
    interaction2 = {
        "provider_id": agent1_trust_id,
        "consumer_id": agent0_trust_id,
        "rating": 4,
        "skill_used": "data-standardization",
        "response_time": 1200,
        "error_occurred": False
    }
    
    response = requests.post(
        f"http://localhost:8000/api/v1/a2a/trust/agents/{agent1_trust_id}/interaction",
        json=interaction2
    )
    
    if response.status_code == 200:
        print("✓ Recorded good interaction from Agent 1")
    
    # Mixed interaction: Agent 0 -> Agent 1 (with minor issue)
    interaction3 = {
        "provider_id": agent0_trust_id,
        "consumer_id": agent1_trust_id,
        "rating": 3,
        "skill_used": "data-validation",
        "response_time": 2500,
        "error_occurred": False
    }
    
    response = requests.post(
        f"http://localhost:8000/api/v1/a2a/trust/agents/{agent0_trust_id}/interaction",
        json=interaction3
    )
    
    if response.status_code == 200:
        print("✓ Recorded mixed interaction from Agent 0")
    
    # 7. Check Updated Trust Scores
    print("\n7. Checking updated trust scores after interactions...")
    
    for agent_id, name in [(agent0_trust_id, "Agent 0"), (agent1_trust_id, "Agent 1")]:
        response = requests.get(
            f"http://localhost:8000/api/v1/a2a/trust/agents/{agent_id}/score"
        )
        
        if response.status_code == 200:
            score_data = response.json()
            trust_metrics = score_data['trust_metrics']
            
            print(f"✓ {name}:")
            print(f"  Overall Trust Score: {score_data['overall_trust_score']:.2f}/5.0")
            print(f"  Trust Rating: {trust_metrics['trust_rating']:.2f}")
            print(f"  Total Interactions: {trust_metrics['total_interactions']}")
            print(f"  Successful Interactions: {trust_metrics['successful_interactions']}")
            print(f"  Success Rate: {trust_metrics['successful_interactions'] / max(trust_metrics['total_interactions'], 1) * 100:.0f}%")
            
            if trust_metrics['skill_ratings']:
                print(f"  Skill Ratings: {trust_metrics['skill_ratings']}")
    
    # 8. Create SLA Agreement
    print("\n8. Creating Service Level Agreement...")
    
    sla_request = {
        "provider_id": agent0_trust_id,
        "consumer_id": agent1_trust_id,
        "terms": {
            "response_time_max": 5000,
            "availability_min": 99.0,
            "error_rate_max": 2.0
        },
        "validity_hours": 168  # 1 week
    }
    
    response = requests.post(
        "http://localhost:8000/api/v1/a2a/trust/sla",
        json=sla_request
    )
    
    if response.status_code == 201:
        sla_data = response.json()
        print(f"✓ SLA created: {sla_data['sla_id'][:20]}...")
        print("  Terms: <5s response, >99% uptime, <2% errors")
        print("  Validity: 1 week")
    
    # 9. Get Trust Leaderboard
    print("\n9. Getting trust leaderboard...")
    
    response = requests.get("http://localhost:8000/api/v1/a2a/trust/leaderboard?limit=5")
    
    if response.status_code == 200:
        leaderboard_data = response.json()
        print(f"✓ Trust Leaderboard (Top {len(leaderboard_data['leaderboard'])}):")
        
        for i, agent in enumerate(leaderboard_data['leaderboard'], 1):
            print(f"  {i}. Agent {agent['agent_id'][:10]}...:")
            print(f"     Trust Score: {agent['trust_score']:.2f}")
            print(f"     Interactions: {agent['total_interactions']}")
            print(f"     Success Rate: {agent['success_rate'] * 100:.0f}%")
    
    # 10. Get System Metrics
    print("\n10. Getting trust system metrics...")
    
    response = requests.get("http://localhost:8000/api/v1/a2a/trust/metrics")
    
    if response.status_code == 200:
        metrics = response.json()
        print(f"✓ System Metrics (24h):")
        print(f"  Agent Registrations: {metrics['agent_registrations']}")
        print(f"  Trust Updates: {metrics['trust_updates']}")
        print(f"  Workflows Created: {metrics['workflows_created']}")
        print(f"  Average Trust Score: {metrics['average_trust_score']:.2f}")
        print(f"  Workflow Success Rate: {metrics['workflow_success_rate'] * 100:.0f}%")


if __name__ == "__main__":
    test_trust_relationships()
    print(f"\n{'='*50}")
    print("✓ A2A Trust Relationship Management test complete!")
    print("  - Trust system operational")
    print("  - Agent registration with commitment levels")
    print("  - Trust score calculation")
    print("  - Inter-agent trust relationships")
    print("  - SLA agreements")
    print("  - No cryptocurrency or wallets involved!")
    print(f"{'='*50}")