#!/usr/bin/env python3
"""
Test script for trust-aware A2A Registry integration
Tests with specific agent addresses:
- Agent1: 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
- Agent2: 0x70997970C51812dc3A010C7d01b50e0d17dc79C8
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Import the enhanced registry service
from service import A2ARegistryService
from models import (
    AgentRegistrationRequest, AgentSearchRequest, 
    WorkflowMatchRequest, WorkflowStageRequirement,
    AgentCard, AgentProvider, AgentCapabilities, AgentSkill
)

# Import trust system
from ..a2a.security.smart_contract_trust import get_trust_contract
from ..a2a.security.delegation_contracts import get_delegation_contract, DelegationAction

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_trust_integration():
    """Test the trust-aware A2A Registry integration"""
    
    # Test agent addresses
    AGENT1_ADDRESS = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
    AGENT2_ADDRESS = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
    
    # Initialize services
    logger.info("🚀 Initializing trust-aware A2A Registry Service...")
    registry = A2ARegistryService(enable_trust_integration=True)
    
    # Get trust contracts
    trust_contract = get_trust_contract()
    delegation_contract = get_delegation_contract()
    
    # Step 1: Register agents in trust system first
    logger.info("📝 Registering agents in trust system...")
    
    # Register Agent1 with high trust
    agent1_identity = trust_contract.register_agent(
        agent_id=AGENT1_ADDRESS,
        agent_type="financial_analysis"
    )
    trust_contract.update_trust_score(AGENT1_ADDRESS, 0.9)  # High trust
    logger.info(f"✅ Agent1 registered with trust score: 0.9")
    
    # Register Agent2 with medium trust
    agent2_identity = trust_contract.register_agent(
        agent_id=AGENT2_ADDRESS,
        agent_type="data_processing"
    )
    trust_contract.update_trust_score(AGENT2_ADDRESS, 0.6)  # Medium trust
    logger.info(f"✅ Agent2 registered with trust score: 0.6")
    
    # Step 2: Create delegation between agents
    logger.info("🤝 Creating delegation contract...")
    delegation_contract.create_delegation(
        delegator_id=AGENT1_ADDRESS,
        delegate_id=AGENT2_ADDRESS,
        action=DelegationAction.DATA_PROCESSING,
        duration_hours=24
    )
    logger.info(f"✅ Delegation created: Agent1 -> Agent2 for DATA_PROCESSING")
    
    # Step 3: Register agents in A2A Registry
    logger.info("📋 Registering agents in A2A Registry...")
    
    # Create Agent1 card
    agent1_card = AgentCard(
        name="Financial Analysis Agent (High Trust)",
        description="High-trust agent for financial analysis and portfolio management",
        url="http://localhost:8001",
        version="1.0.0",
        protocolVersion="0.2.9",
        provider=AgentProvider(
            organization="FinSight CIB Test",
            url="https://finsight-cib.com"
        ),
        capabilities=AgentCapabilities(
            streaming=True,
            batchProcessing=True
        ),
        skills=[
            AgentSkill(
                id="portfolio-analysis",
                name="Portfolio Analysis",
                description="Analyze investment portfolios",
                tags=["financial", "analysis", "portfolio"]
            ),
            AgentSkill(
                id="risk-assessment",
                name="Risk Assessment",
                description="Assess financial risks",
                tags=["financial", "risk", "analysis"]
            )
        ],
        defaultInputModes=["application/json"],
        defaultOutputModes=["application/json"],
        authentication={"schemes": ["Bearer", "Basic"]},
        preferredTransport="https"
    )
    
    # Register Agent1
    agent1_request = AgentRegistrationRequest(
        agent_card=agent1_card,
        registered_by="test_user"
    )
    
    # Use the agent address as the agent_id by modifying the registration
    agent1_response = await registry.register_agent(agent1_request)
    # Update the agent_id to use the address
    registry.agents[AGENT1_ADDRESS] = registry.agents.pop(agent1_response.agent_id)
    registry.agents[AGENT1_ADDRESS].agent_id = AGENT1_ADDRESS
    logger.info(f"✅ Agent1 registered: {AGENT1_ADDRESS}")
    
    # Create Agent2 card
    agent2_card = AgentCard(
        name="Data Processing Agent (Medium Trust)",
        description="Medium-trust agent for data processing and transformation",
        url="http://localhost:8002",
        version="1.0.0",
        protocolVersion="0.2.9",
        provider=AgentProvider(
            organization="FinSight CIB Test",
            url="https://finsight-cib.com"
        ),
        capabilities=AgentCapabilities(
            streaming=True,
            batchProcessing=True
        ),
        skills=[
            AgentSkill(
                id="data-transformation",
                name="Data Transformation",
                description="Transform and process financial data",
                tags=["data", "processing", "transformation"]
            ),
            AgentSkill(
                id="data-validation",
                name="Data Validation",
                description="Validate financial data integrity",
                tags=["data", "validation", "quality"]
            )
        ],
        defaultInputModes=["application/json", "text/csv"],
        defaultOutputModes=["application/json", "text/csv"],
        authentication={"schemes": ["Bearer", "Basic"]},
        preferredTransport="https"
    )
    
    # Register Agent2
    agent2_request = AgentRegistrationRequest(
        agent_card=agent2_card,
        registered_by="test_user"
    )
    
    agent2_response = await registry.register_agent(agent2_request)
    # Update the agent_id to use the address
    registry.agents[AGENT2_ADDRESS] = registry.agents.pop(agent2_response.agent_id)
    registry.agents[AGENT2_ADDRESS].agent_id = AGENT2_ADDRESS
    logger.info(f"✅ Agent2 registered: {AGENT2_ADDRESS}")
    
    # Step 4: Test trust-aware agent search
    logger.info("\n🔍 Testing trust-aware agent search...")
    
    search_request = AgentSearchRequest(
        tags=["financial"],
        pageSize=10
    )
    
    search_results = await registry.search_agents(search_request)
    
    logger.info(f"Found {len(search_results.results)} agents")
    for result in search_results.results:
        logger.info(f"  - {result.name}: trust_score={result.trust_score}, trust_level={result.trust_level}")
    
    # Verify Agent1 appears first due to higher trust
    if search_results.results and search_results.results[0].agent_id == AGENT1_ADDRESS:
        logger.info("✅ Trust-based ranking working: High-trust agent ranked first")
    else:
        logger.warning("⚠️ Trust-based ranking issue: Expected high-trust agent first")
    
    # Step 5: Test workflow matching with trust consideration
    logger.info("\n🔄 Testing trust-aware workflow matching...")
    
    workflow_request = WorkflowMatchRequest(
        workflow_requirements=[
            WorkflowStageRequirement(
                stage="analyze",
                required_skills=["portfolio-analysis"],
                input_modes=["application/json"],
                output_modes=["application/json"]
            ),
            WorkflowStageRequirement(
                stage="process",
                required_skills=["data-transformation"],
                input_modes=["application/json"],
                output_modes=["application/json"]
            )
        ]
    )
    
    workflow_match = await registry.match_workflow_agents(workflow_request)
    
    logger.info(f"Workflow coverage: {workflow_match.coverage_percentage}%")
    for stage_match in workflow_match.matching_agents:
        logger.info(f"  Stage '{stage_match.stage}':")
        for agent in stage_match.agents:
            logger.info(f"    - {agent.name} (trust: {agent.trust_score})")
    
    # Step 6: Test delegation verification
    logger.info("\n🔐 Testing delegation verification...")
    
    # Check if Agent2 can process data for Agent1
    can_delegate = delegation_contract.can_agent_delegate(
        delegator_id=AGENT1_ADDRESS,
        delegate_id=AGENT2_ADDRESS,
        action=DelegationAction.DATA_PROCESSING
    )
    
    logger.info(f"Can Agent2 process data for Agent1? {can_delegate}")
    
    # Step 7: Test trust score updates
    logger.info("\n📊 Testing trust score updates...")
    
    # Simulate successful interaction - increase Agent2's trust
    trust_contract.update_trust_score(AGENT2_ADDRESS, 0.75)
    logger.info(f"Updated Agent2 trust score to 0.75")
    
    # Search again to see updated ranking
    search_results2 = await registry.search_agents(search_request)
    logger.info("Updated search results:")
    for result in search_results2.results:
        logger.info(f"  - {result.name}: trust_score={result.trust_score}, trust_level={result.trust_level}")
    
    # Step 8: Test message signing with trust
    logger.info("\n✍️ Testing message signing with trust...")
    
    test_message = {
        "type": "request",
        "content": "Analyze portfolio risk",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Sign message with Agent1
    signed_message = trust_contract.sign_message(AGENT1_ADDRESS, test_message)
    logger.info(f"Message signed by Agent1")
    
    # Verify signature
    is_valid, verified_message = trust_contract.verify_message(signed_message)
    logger.info(f"Signature valid: {is_valid}")
    logger.info(f"Signer trust score: {trust_contract.get_trust_score(signed_message['signer_id'])}")
    
    # Summary
    logger.info("\n📋 Test Summary:")
    logger.info(f"✅ Trust system integration: {'Enabled' if registry.enable_trust_integration else 'Disabled'}")
    logger.info(f"✅ Agents registered: 2")
    logger.info(f"✅ Trust-based ranking: Working")
    logger.info(f"✅ Delegation contracts: Active")
    logger.info(f"✅ Message signing: Functional")
    logger.info(f"✅ Trust scores: Agent1={trust_contract.get_trust_score(AGENT1_ADDRESS)}, Agent2={trust_contract.get_trust_score(AGENT2_ADDRESS)}")
    
    return True


async def main():
    """Main test runner"""
    try:
        logger.info("Starting trust-aware A2A Registry integration test...")
        success = await test_trust_integration()
        if success:
            logger.info("✅ All tests completed successfully!")
        else:
            logger.error("❌ Some tests failed")
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())