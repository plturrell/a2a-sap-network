#!/usr/bin/env python3
"""
Simple Trust System Test
"""

import asyncio
from app.a2a.security.smart_contract_trust import (
    initialize_agent_trust, 
    sign_a2a_message, 
    verify_a2a_message,
    get_trust_contract
)

def test_trust_system():
    """Test the smart contract trust system"""
    print("=== TESTING SMART CONTRACT TRUST SYSTEM ===")
    
    try:
        # Initialize two agents
        print("🔧 Initializing agent identities...")
        agent0_identity = initialize_agent_trust("agent0", "DataProductRegistrationAgent")
        agent1_identity = initialize_agent_trust("agent1", "FinancialStandardizationAgent")
        
        print(f"✅ Agent 0 initialized: {agent0_identity.agent_id}")
        print(f"✅ Agent 1 initialized: {agent1_identity.agent_id}")
        
        # Create a test message
        test_message = {
            "message": {
                "role": "user", 
                "parts": [{"kind": "text", "text": "Test message"}]
            },
            "contextId": "test_context"
        }
        
        # Sign the message with Agent 0
        print("\n📝 Signing message with Agent 0...")
        signed_message = sign_a2a_message("agent0", test_message)
        print(f"✅ Message signed successfully")
        
        # Verify the message
        print("\n🔍 Verifying signed message...")
        verified, verification_info = verify_a2a_message(signed_message)
        
        if verified:
            print(f"✅ Message verification successful!")
            print(f"   Agent: {verification_info.get('agent_id')}")
            print(f"   Trust Score: {verification_info.get('trust_score')}")
            print(f"   Contract ID: {verification_info.get('contract_id')}")
        else:
            print(f"❌ Message verification failed: {verification_info}")
            return False
        
        # Get contract status
        print("\n📊 Contract status:")
        contract = get_trust_contract()
        status = contract.get_contract_status()
        print(f"   Contract ID: {status['contract_id']}")
        print(f"   Total Agents: {status['statistics']['total_agents']}")
        print(f"   Average Trust Score: {status['statistics']['average_trust_score']}")
        
        print("\n🎉 TRUST SYSTEM TEST SUCCESSFUL!")
        return True
        
    except Exception as e:
        print(f"❌ Trust system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_trust_system()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")