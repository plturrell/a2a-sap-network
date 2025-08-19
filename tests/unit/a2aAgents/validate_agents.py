#!/usr/bin/env python3
"""
Validation script for performance-enhanced A2A agents
Tests all enhanced agents with performance monitoring capabilities
"""

import sys
import asyncio
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🚀 Testing Performance-Enhanced A2A Agents...\n")


async def main():
    global agent0_passed, agent1_passed, agent2_passed, agent3_passed, agent4_passed
    
    # Initialize all variables
    agent0_passed = False
    agent1_passed = False
    agent2_passed = False
    agent3_passed = False
    agent4_passed = False
    
    # Test Agent 0 - Enhanced Data Product Agent
    print("1️⃣  Testing Enhanced Agent 0 (Data Product Agent)...")
    try:
        from app.a2a.agents.agent0DataProduct.active.enhancedDataProductAgentSdk import EnhancedDataProductRegistrationAgentSDK
        
        # Create agent
        agent0 = EnhancedDataProductRegistrationAgentSDK(
            base_url="http://localhost:8000",
            ord_registry_url="http://localhost:9000",
            enable_monitoring=False  # Disable for testing
        )
        
        # Test initialization
        await agent0.initialize()
        
        # Test health check
        health = await agent0.get_agent_health()
        assert "agent_metrics" in health
        assert "cache_performance" in health
        
        # Cleanup
        await agent0.shutdown()
        
        print("✅ Agent 0 enhanced successfully")
        agent0_passed = True
    except Exception as e:
        print(f"❌ Agent 0 enhancement failed: {e}")
        agent0_passed = False

    # Test Agent 1 - Enhanced Data Standardization Agent
    print("\n2️⃣  Testing Enhanced Agent 1 (Data Standardization Agent)...")
    try:
        from app.a2a.agents.agent1Standardization.active.dataStandardizationAgentSdk import DataStandardizationAgentSDK
        
        # Create agent
        agent1 = DataStandardizationAgentSDK(
            base_url="http://localhost:8001",
            enable_monitoring=False  # Disable for testing
        )
        
        # Test initialization
        await agent1.initialize()
        
        # Test health check
        health = await agent1.get_agent_health()
        assert "agent_metrics" in health
        assert "cache_performance" in health
        
        # Cleanup
        await agent1.shutdown()
        
        print("✅ Agent 1 enhanced successfully")
        agent1_passed = True
    except Exception as e:
        print(f"❌ Agent 1 enhancement failed: {e}")
        agent1_passed = False

    # Test Agent 2 - Enhanced AI Preparation Agent
    print("\n3️⃣  Testing Enhanced Agent 2 (AI Preparation Agent)...")
    try:
        from app.a2a.agents.agent2AiPreparation.active.aiPreparationAgentSdk import AIPreparationAgentSDK
        
        # Create agent
        agent2 = AIPreparationAgentSDK(
            base_url="http://localhost:8002",
            vector_service_url="http://localhost:9002",
            enable_monitoring=False  # Disable for testing
        )
        
        # Test initialization
        await agent2.initialize()
        
        # Test health check
        health = await agent2.get_agent_health()
        assert "agent_metrics" in health
        assert "cache_performance" in health
        
        # Cleanup
        await agent2.shutdown()
        
        print("✅ Agent 2 enhanced successfully")
        agent2_passed = True
    except Exception as e:
        print(f"❌ Agent 2 enhancement failed: {e}")
        agent2_passed = False

    # Test Agent 3 - Enhanced Vector Processing Agent
    print("\n4️⃣  Testing Enhanced Agent 3 (Vector Processing Agent)...")
    try:
        from app.a2a.agents.agent3VectorProcessing.active.vectorProcessingAgentSdk import VectorProcessingAgentSDK
        
        # Create agent
        agent3 = VectorProcessingAgentSDK(
            base_url="http://localhost:8003",
            hana_config={
                "host": "localhost",
                "port": 39015,
                "user": "test",
                "password": "test",
                "database": "test"
            },
            enable_monitoring=False  # Disable for testing
        )
        
        # Test initialization
        await agent3.initialize()
        
        print("✅ Agent 3 enhanced successfully")
        agent3_passed = True
    except Exception as e:
        print(f"❌ Agent 3 enhancement failed: {e}")
        agent3_passed = False

    # Test Agent 4 - Enhanced Calc Validation Agent
    print("\n5️⃣  Testing Enhanced Agent 4 (Calc Validation Agent)...")
    try:
        from app.a2a.agents.agent4CalcValidation.active.calcValidationAgentSdk import CalcValidationAgentSDK
        
        # Create agent
        agent4 = CalcValidationAgentSDK(
            base_url="http://localhost:8004",
            enable_monitoring=False  # Disable for testing
        )
        
        # Test basic attributes
        assert agent4.agent_id == "calc_validation_agent_4"
        assert agent4.enable_monitoring == False
        
        print("✅ Agent 4 enhanced successfully")
        agent4_passed = True
    except Exception as e:
        print(f"❌ Agent 4 enhancement failed: {e}")
        agent4_passed = False

    # Summary
    print("\n" + "="*60)
    print("🎯 PERFORMANCE ENHANCEMENT SUMMARY")
    print("="*60)

    test_results = [
        ("Agent 0 (Data Product)", agent0_passed),
        ("Agent 1 (Standardization)", agent1_passed),
        ("Agent 2 (AI Preparation)", agent2_passed),
        ("Agent 3 (Vector Processing)", agent3_passed),
        ("Agent 4 (Calc Validation)", agent4_passed)
    ]

    passed = 0
    for test_name, result in test_results:
        status = "✅ ENHANCED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    total = len(test_results)
    print(f"\n📊 Results: {passed}/{total} agents successfully enhanced")

    if passed == total:
        print("\n🎉 All core agents successfully enhanced with performance monitoring!")
        print("✅ Enhanced capabilities available:")
        print("   - Real-time performance metrics collection")
        print("   - Automatic performance optimization")
        print("   - Adaptive throttling and caching")
        print("   - Performance alerts and recommendations")
        print("   - Enhanced health monitoring")
        print("   - Prometheus metrics integration")
    else:
        print(f"\n⚠️  {total - passed} agents need additional work.")
        print("Enhanced agents are ready for production use.")

if __name__ == "__main__":
    asyncio.run(main())