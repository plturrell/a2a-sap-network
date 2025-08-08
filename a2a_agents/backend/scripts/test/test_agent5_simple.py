#!/usr/bin/env python3
"""
Simple test for Agent 5 imports and basic functionality
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def test_imports():
    """Test that Agent 5 can be imported"""
    try:
        from app.a2a.agents.agent5_qa_validation.active.qa_validation_agent_sdk import (
            QAValidationAgentSDK,
            TestDifficulty,
            TestMethodology,
            ResourceType,
            QAValidationRequest
        )
        print("✅ Agent 5 imports successful")
        
        # Test basic instantiation
        agent = QAValidationAgentSDK(
            base_url="http://localhost:8007",
            embedding_model="all-mpnet-base-v2",
            cache_ttl=3600
        )
        
        print(f"✅ Agent 5 instantiation successful")
        print(f"   Agent ID: {agent.agent_id}")
        print(f"   Name: {agent.name}")
        print(f"   Version: {agent.version}")
        print(f"   Embedding Model: {agent.embedding_model}")
        
        # Test question templates loading
        print(f"✅ Question templates loaded:")
        for category, templates in agent.question_templates.items():
            print(f"   {category}: {len(templates)} template types")
        
        # Test agent card generation
        agent_card = agent.get_agent_card()
        print(f"✅ Agent card generated: {agent_card['name']}")
        print(f"   Skills: {len(agent_card['skills'])}")
        print(f"   Capabilities: {agent_card['capabilities']}")
        
        # Test request model
        request = QAValidationRequest(
            ord_endpoints=["http://localhost:8080"],
            test_methodology=TestMethodology.SIMPLEQA,
            resource_types=[ResourceType.DATA_PRODUCTS]
        )
        print(f"✅ Request model validation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent 5 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_router_imports():
    """Test that Agent 5 router can be imported"""
    try:
        from app.a2a.agents.agent5_qa_validation.active.agent5_router import (
            router, initialize_agent, QATaskRequest, ORDDiscoveryRequest
        )
        from app.a2a.agents.agent5_qa_validation.active.qa_validation_agent_sdk import (
            TestMethodology, ResourceType
        )
        print("✅ Agent 5 router imports successful")
        
        # Test request models
        qa_request = QATaskRequest(
            ord_endpoints=["http://localhost:8080"],
            test_methodology=TestMethodology.SIMPLEQA
        )
        print(f"✅ QA task request model validation successful")
        
        ord_request = ORDDiscoveryRequest(
            ord_endpoints=["http://localhost:8080"],
            resource_types=[ResourceType.DATA_PRODUCTS]
        )
        print(f"✅ ORD discovery request model validation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent 5 router test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Agent 5 Simple Test")
    
    success = True
    
    # Test core agent
    if not test_imports():
        success = False
    
    # Test router
    if not test_router_imports():
        success = False
    
    if success:
        print("🎉 All Agent 5 tests passed!")
    else:
        print("💥 Some Agent 5 tests failed!")
    
    sys.exit(0 if success else 1)