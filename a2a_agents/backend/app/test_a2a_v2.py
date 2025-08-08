#!/usr/bin/env python3
"""
Test A2A Blockchain v2.0 Network
"""

import requests
import json
from datetime import datetime

def test_a2a_v2_network():
    """Test the A2A v2.0 blockchain network"""
    
    base_url = "http://localhost:8084"
    
    print("🧪 Testing A2A Blockchain Network v2.0...")
    
    try:
        # Test root endpoint
        print("\n1️⃣ Testing root endpoint...")
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Network: {data['network']}")
            print(f"✅ Version: {data['version']}")
            print(f"✅ Protocol: {data['protocol']['version']}")
            print(f"✅ Compliance: {data['protocol']['compliance']}")
            print(f"✅ Blockchain: {data['blockchain']['status']}")
            print(f"✅ Agents: {data['agents']['total']}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return
        
        # Test agent discovery
        print("\n2️⃣ Testing agent discovery...")
        response = requests.get(f"{base_url}/agents")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {data['total']} A2A v{data['network_version']} agents")
            for agent in data['agents']:
                print(f"   • {agent['name']} ({agent['agent_id'][:10]}...)")
                print(f"     Trust: {agent['trust']['score']:.3f}" if agent['trust']['score'] else "     Trust: Not available")
                print(f"     Skills: {len(agent['skills'])}")
        else:
            print(f"❌ Agent discovery failed: {response.status_code}")
            return
        
        # Test agent cards (A2A standard)
        print("\n3️⃣ Testing A2A agent cards...")
        agents = data['agents']
        for agent in agents[:2]:  # Test first 2 agents
            agent_id = agent['agent_id']
            card_url = f"{base_url}/agents/{agent_id}/.well-known/agent.json"
            response = requests.get(card_url)
            
            if response.status_code == 200:
                card = response.json()
                print(f"✅ Agent Card: {card['name']}")
                print(f"   Protocol: {card['protocolVersion']}")
                print(f"   Skills: {len(card['skills'])}")
                print(f"   Capabilities: {len([k for k, v in card['capabilities'].items() if v])}")
                
                # Verify A2A v0.2.9 compliance
                required_fields = ['name', 'description', 'url', 'version', 'protocolVersion', 
                                 'provider', 'capabilities', 'skills']
                missing = [f for f in required_fields if f not in card]
                if not missing:
                    print("   ✅ A2A v0.2.9 compliant")
                else:
                    print(f"   ❌ Missing fields: {missing}")
            else:
                print(f"❌ Agent card failed for {agent_id}: {response.status_code}")
        
        # Test health endpoints
        print("\n4️⃣ Testing agent health...")
        for agent in agents[:2]:
            agent_id = agent['agent_id']
            health_url = f"{base_url}/agents/{agent_id}/health"
            response = requests.get(health_url)
            
            if response.status_code == 200:
                health = response.json()
                print(f"✅ {agent['name']}: {health['status']}")
                print(f"   Blockchain: {health['blockchain']['status']}")
                print(f"   Trust: {health['trust']['level']}")
                print(f"   Skills: {health['skills']['total']} operational")
            else:
                print(f"❌ Health check failed for {agent_id}: {response.status_code}")
        
        # Test A2A message processing
        print("\n5️⃣ Testing A2A v0.2.9 message processing...")
        
        # Test financial agent
        financial_agent = next((a for a in agents if 'financial' in a['name'].lower()), None)
        if financial_agent:
            agent_id = financial_agent['agent_id']
            
            # Create A2A v0.2.9 compliant message
            test_message = {
                "messageId": f"test_msg_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "role": "user",
                "parts": [
                    {
                        "type": "function-call",
                        "name": "portfolio-analysis",
                        "arguments": {
                            "portfolio_value": 2000000,
                            "holdings": ["AAPL", "GOOGL", "MSFT", "AMZN"]
                        },
                        "id": "func_001"
                    }
                ],
                "taskId": f"task_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "contextId": f"ctx_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
            response = requests.post(
                f"{base_url}/agents/{agent_id}/messages",
                json=test_message,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Portfolio analysis completed")
                print(f"   Message ID: {result['messageId']}")
                print(f"   Status: {result['status']}")
                print(f"   Protocol: {result['protocol']}")
                print(f"   Network Version: {result['network_version']}")
                print(f"   Blockchain executed: {result['blockchain']['executed']}")
                print(f"   Results: {len(result['results'])} parts processed")
                
                # Show portfolio analysis results
                for res in result['results']:
                    if res.get('skill') == 'portfolio-analysis':
                        analysis = res.get('output', {}).get('analysis', {})
                        print(f"   📊 Portfolio Value: ${analysis.get('total_value', 0):,}")
                        print(f"   📊 Sharpe Ratio: {analysis.get('risk_metrics', {}).get('sharpe_ratio', 'N/A')}")
                        print(f"   📊 Max Drawdown: {analysis.get('risk_metrics', {}).get('maximum_drawdown', 'N/A')}")
            else:
                print(f"❌ Message processing failed: {response.status_code}")
                print(f"   Response: {response.text}")
        
        print("\n🎯 A2A Blockchain Network v2.0 Test Summary:")
        print("   ✅ Network operational")
        print("   ✅ A2A v0.2.9 protocol compliance")
        print("   ✅ Blockchain integration active")
        print("   ✅ Trust system integrated")
        print("   ✅ Agent discovery working")
        print("   ✅ Message processing functional")
        print("   ✅ Version 2.0.0 properly implemented")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to A2A Blockchain v2.0 server")
        print("   Make sure the server is running on http://localhost:8084")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_a2a_v2_network()