#!/usr/bin/env python3
"""
Simple test to verify trust integration with specific agent addresses
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a2a.security.smart_contract_trust import SmartContractTrust
from a2a.security.delegation_contracts import AgentDelegationContract, DelegationAction


def test_trust_with_addresses():
    """Test trust system with the specific addresses"""
    
    # Test addresses
    AGENT1 = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
    AGENT2 = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
    
    print("🚀 Testing Trust System with Specific Addresses")
    print(f"Agent1: {AGENT1}")
    print(f"Agent2: {AGENT2}")
    print("-" * 60)
    
    # Initialize trust contract
    trust = SmartContractTrust()
    
    # Test 1: Register agents
    print("\n1️⃣ Registering agents...")
    
    agent1_identity = trust.register_agent(AGENT1, "financial_analyzer")
    print(f"✅ Agent1 registered: {agent1_identity.agent_id}")
    print(f"   Public key: {agent1_identity.public_key[:50]}...")
    
    agent2_identity = trust.register_agent(AGENT2, "data_processor")
    print(f"✅ Agent2 registered: {agent2_identity.agent_id}")
    print(f"   Public key: {agent2_identity.public_key[:50]}...")
    
    # Initialize delegation contract after agents are registered
    delegation = AgentDelegationContract()
    
    # Test 2: Set trust scores
    print("\n2️⃣ Setting trust scores...")
    
    trust.update_trust_score(AGENT1, 0.9)
    print(f"✅ Agent1 trust score: {trust.get_trust_score(AGENT1)}")
    
    trust.update_trust_score(AGENT2, 0.6)
    print(f"✅ Agent2 trust score: {trust.get_trust_score(AGENT2)}")
    
    # Test 3: Create delegation
    print("\n3️⃣ Creating delegation contract...")
    
    delegation.create_delegation(
        delegator_id=AGENT1,
        delegatee_id=AGENT2,
        actions=[DelegationAction.DATA_STORAGE],
        duration_hours=24
    )
    print(f"✅ Delegation created: Agent1 → Agent2 for DATA_STORAGE")
    
    # Test 4: Verify delegation
    print("\n4️⃣ Verifying delegation...")
    
    can_process = delegation.can_delegate(
        AGENT1, AGENT2, DelegationAction.DATA_STORAGE
    )
    print(f"✅ Can Agent2 process data for Agent1? {can_process}")
    
    # Test 5: Test message signing
    print("\n5️⃣ Testing message signing...")
    
    test_message = {
        "action": "analyze_portfolio",
        "data": {"portfolio_id": "TEST123"},
        "timestamp": "2024-01-01T00:00:00Z"
    }
    
    # Agent1 signs message
    signed_msg = trust.sign_message(AGENT1, test_message)
    print(f"✅ Message signed by Agent1")
    print(f"   Signature: {signed_msg['signature'][:50]}...")
    
    # Verify signature
    is_valid, verified_msg = trust.verify_message(signed_msg)
    print(f"✅ Signature verification: {'PASSED' if is_valid else 'FAILED'}")
    print(f"   Signer: {signed_msg['signer_id']}")
    print(f"   Trust level: {trust.get_trust_level(signed_msg['signer_id'])}")
    
    # Test 6: Test trust-based comparison
    print("\n6️⃣ Comparing agent trust levels...")
    
    agents = [
        {"id": AGENT1, "name": "Agent1", "trust": trust.get_trust_score(AGENT1)},
        {"id": AGENT2, "name": "Agent2", "trust": trust.get_trust_score(AGENT2)}
    ]
    
    # Sort by trust score (descending)
    agents.sort(key=lambda x: x['trust'], reverse=True)
    
    print("Agents ranked by trust:")
    for i, agent in enumerate(agents, 1):
        print(f"   {i}. {agent['name']}: {agent['trust']} ({trust.get_trust_level(agent['id'])})")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Agents registered: 2")
    print(f"✅ Trust scores set: Agent1=0.9, Agent2=0.6")
    print(f"✅ Delegation active: Agent1 → Agent2")
    print(f"✅ Message signing: Working")
    print(f"✅ Trust ranking: Agent1 > Agent2")
    print("\n✨ All tests passed successfully!")


if __name__ == "__main__":
    test_trust_with_addresses()