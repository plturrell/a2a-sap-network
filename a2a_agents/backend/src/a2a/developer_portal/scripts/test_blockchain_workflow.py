#!/usr/bin/env python3
"""
Test script for blockchain workflow integration
Demonstrates real production usage of A2A blockchain with BPMN workflows
"""

import asyncio
import httpx
import json
import os
from datetime import datetime
from eth_account import Account

# Configuration
PORTAL_URL = os.environ.get("PORTAL_URL", "http://localhost:3001")
BLOCKCHAIN_NETWORK = os.environ.get("BLOCKCHAIN_NETWORK", "local")
PRIVATE_KEY = os.environ.get("WORKFLOW_PRIVATE_KEY")

# Generate a private key if not provided
if not PRIVATE_KEY:
    account = Account.create()
    PRIVATE_KEY = account.key.hex()
    print(f"Generated new account: {account.address}")
    print(f"Private key: {PRIVATE_KEY}")
    print("Note: Save this private key for future use")


async def main():
    """Test blockchain workflow integration"""
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # 1. Check health
            print("\n1. Checking portal health...")
            response = await client.get(f"{PORTAL_URL}/api/health")
            health_data = response.json()
            print(f"Portal status: {health_data['status']}")
            print(f"Blockchain enabled: {health_data.get('blockchain_enabled', False)}")
            
            # 2. Get contract addresses
            print("\n2. Getting contract addresses...")
            response = await client.get(f"{PORTAL_URL}/api/blockchain/contracts?network={BLOCKCHAIN_NETWORK}")
            contracts = response.json()
            print(f"Network: {contracts['network']}")
            print(f"Contracts: {json.dumps(contracts['contracts'], indent=2)}")
            
            # 3. Register workflow agent on blockchain
            print("\n3. Registering workflow agent on blockchain...")
            agent_data = {
                "name": f"Workflow Agent {datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                "endpoint": f"{PORTAL_URL}/api/agent",
                "capabilities": ["workflow", "bpmn", "automation", "data_processing"],
                "network": BLOCKCHAIN_NETWORK,
                "privateKey": PRIVATE_KEY
            }
            
            response = await client.post(
                f"{PORTAL_URL}/api/blockchain/register-agent",
                json=agent_data
            )
            
            if response.status_code == 200:
                registration_result = response.json()
                print(f"Registration successful!")
                print(f"Transaction hash: {registration_result.get('transactionHash')}")
                print(f"Agent address: {registration_result.get('agentAddress')}")
                print(f"Gas used: {registration_result.get('gasUsed')}")
                
                agent_address = registration_result.get('agentAddress')
            else:
                print(f"Registration failed: {response.text}")
                return
            
            # 4. Discover agents with data_processing capability
            print("\n4. Discovering agents with data_processing capability...")
            response = await client.get(
                f"{PORTAL_URL}/api/blockchain/agents?network={BLOCKCHAIN_NETWORK}&capability=data_processing"
            )
            
            if response.status_code == 200:
                discovery_result = response.json()
                print(f"Found {discovery_result['count']} agents with data_processing capability")
                
                for agent in discovery_result.get('agents', []):
                    print(f"  - {agent['name']} ({agent['address']})")
                    print(f"    Endpoint: {agent['endpoint']}")
                    print(f"    Reputation: {agent['reputation']}")
                    print(f"    Capabilities: {', '.join(agent['capabilities'])}")
            
            # 5. Execute blockchain workflow
            print("\n5. Executing blockchain workflow...")
            
            # First, load the workflow
            workflow_id = "blockchain-agent-workflow"
            
            # Execute workflow with variables
            execution_data = {
                "variables": {
                    "agentName": f"Test Agent {datetime.utcnow().strftime('%H%M%S')}",
                    "agentEndpoint": f"{PORTAL_URL}/api/test-agent",
                    "agentCapabilities": ["test", "automation"],
                    "requireDiscovery": True,
                    "requiredCapability": "data_processing",
                    "targetAgent": agent_address if 'agent_address' in locals() else "0x0000000000000000000000000000000000000000",
                    "messageContent": "Test message from workflow execution"
                },
                "privateKey": PRIVATE_KEY
            }
            
            response = await client.post(
                f"{PORTAL_URL}/api/workflows/{workflow_id}/execute",
                json=execution_data
            )
            
            if response.status_code == 200:
                execution_result = response.json()
                print(f"Workflow execution started!")
                print(f"Execution ID: {execution_result['execution_id']}")
                print(f"Status: {execution_result['status']}")
                
                execution_id = execution_result['execution_id']
                
                # 6. Monitor execution status
                print("\n6. Monitoring workflow execution...")
                for i in range(30):  # Check for 30 seconds
                    await asyncio.sleep(2)
                    
                    response = await client.get(
                        f"{PORTAL_URL}/api/workflow-executions/{execution_id}"
                    )
                    
                    if response.status_code == 200:
                        status_data = response.json()
                        print(f"  Status: {status_data['status']}")
                        
                        if status_data['status'] in ['completed', 'failed', 'terminated']:
                            print(f"\nWorkflow execution finished with status: {status_data['status']}")
                            if status_data.get('result'):
                                print(f"Result: {json.dumps(status_data['result'], indent=2)}")
                            if status_data.get('error_message'):
                                print(f"Error: {status_data['error_message']}")
                            break
                    else:
                        print(f"  Failed to get status: {response.status_code}")
            else:
                print(f"Workflow execution failed: {response.text}")
            
            # 7. Send a direct blockchain message
            print("\n7. Sending direct blockchain message...")
            message_data = {
                "toAgent": agent_address if 'agent_address' in locals() else "0x0000000000000000000000000000000000000000",
                "content": "Direct test message via blockchain",
                "messageType": "test",
                "network": BLOCKCHAIN_NETWORK,
                "privateKey": PRIVATE_KEY
            }
            
            response = await client.post(
                f"{PORTAL_URL}/api/blockchain/send-message",
                json=message_data
            )
            
            if response.status_code == 200:
                message_result = response.json()
                print(f"Message sent successfully!")
                print(f"Transaction hash: {message_result.get('transactionHash')}")
                print(f"Message ID: {message_result.get('messageId')}")
                print(f"Gas used: {message_result.get('gasUsed')}")
            else:
                print(f"Message sending failed: {response.text}")
            
            print("\n✅ Blockchain workflow integration test completed!")
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())