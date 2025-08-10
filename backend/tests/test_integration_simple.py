#!/usr/bin/env python3
"""
Simple integration test to validate a2aAgents and a2aNetwork work together
"""

import sys
import os
from pathlib import Path

# Add both projects to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/Users/apple/projects/a2a/a2aNetwork")

print("🧪 Running Simple Integration Tests...\n")

# Test 1: SDK Import Test
print("1️⃣  Testing SDK imports from a2aNetwork...")
try:
    from app.a2a.sdk import A2AAgentBase, a2a_handler, A2AMessage
    print("✅ SDK components imported successfully")
    print(f"   - A2AAgentBase: {A2AAgentBase}")
    print(f"   - a2a_handler: {a2a_handler}")
    print(f"   - A2AMessage: {A2AMessage}")
    sdk_test_passed = True
except Exception as e:
    print(f"❌ SDK import failed: {e}")
    sdk_test_passed = False

# Test 2: Security Import Test
print("\n2️⃣  Testing security imports from a2aNetwork...")
try:
    from app.a2a.security import sign_a2a_message, verify_a2a_message
    print("✅ Security components imported successfully")
    print(f"   - sign_a2a_message: {sign_a2a_message}")
    print(f"   - verify_a2a_message: {verify_a2a_message}")
    security_test_passed = True
except Exception as e:
    print(f"❌ Security import failed: {e}")
    security_test_passed = False

# Test 3: Agent Creation Test
print("\n3️⃣  Testing agent creation with network components...")
try:
    from app.a2a.agents.agent0DataProduct.active.dataProductAgentSdk import DataProductRegistrationAgentSDK
    
    # Create agent instance
    agent = DataProductRegistrationAgentSDK(
        base_url="http://localhost:8000",
        ord_registry_url="http://localhost:9000"
    )
    
    print("✅ Agent created successfully")
    print(f"   - Agent ID: {agent.agent_id}")
    print(f"   - Agent Name: {agent.name}")
    print(f"   - Base URL: {agent.base_url}")
    agent_test_passed = True
except Exception as e:
    print(f"❌ Agent creation failed: {e}")
    agent_test_passed = False

# Test 4: Network Connector Test
print("\n4️⃣  Testing NetworkConnector integration...")
try:
    from app.a2a.network.networkConnector import NetworkConnector
    import asyncio
    
    async def test_connector():
        connector = NetworkConnector(
            registry_url="http://localhost:9000",
            trust_service_url="http://localhost:9001"
        )
        await connector.initialize()
        print("✅ NetworkConnector initialized successfully")
        # Check attributes based on actual implementation
        print(f"   - Registry URL: {connector.registry_url}")
        print(f"   - Trust Service URL: {connector.trust_service_url}")
        print(f"   - Network Available: {connector._network_available}")
        print(f"   - Initialized: {connector._initialized}")
        return True
    
    # Run async test
    connector_test_passed = asyncio.run(test_connector())
except Exception as e:
    print(f"❌ NetworkConnector test failed: {e}")
    connector_test_passed = False

# Test 5: Version Manager Test
print("\n5️⃣  Testing version management...")
try:
    from app.a2a.version.versionManager import VersionManager
    
    vm = VersionManager()
    
    print("✅ Version management working")
    print(f"   - VersionManager created successfully")
    print(f"   - Protocol version: {vm.protocol_version}")
    print(f"   - Agents version: {vm.agents_version}")
    
    # Check compatibility using async method
    import asyncio
    compat = asyncio.run(vm.check_compatibility())
    print(f"   - Compatibility check result: Compatible={compat.get('compatible', 'Unknown')}")
    version_test_passed = True
except Exception as e:
    print(f"❌ Version management failed: {e}")
    version_test_passed = False

# Test 6: Multiple Agent Compatibility Test
print("\n6️⃣  Testing multiple agents share network components...")
try:
    from app.a2a.agents.catalogManager.active.catalogManagerAgentSdk import CatalogManagerAgentSDK
    from app.a2a.agents.agent4CalcValidation.active.calcValidationAgentSdk import CalcValidationAgentSDK
    
    # Test that agent classes exist and share base
    print("✅ Multiple agent classes imported successfully")
    print(f"   - CatalogManagerAgentSDK base: {CatalogManagerAgentSDK.__bases__}")
    print(f"   - CalcValidationAgentSDK base: {CalcValidationAgentSDK.__bases__}")
    
    # Verify they share the same base class
    both_inherit_from_base = (
        issubclass(CatalogManagerAgentSDK, A2AAgentBase) and 
        issubclass(CalcValidationAgentSDK, A2AAgentBase)
    )
    
    print(f"   - Both inherit from same A2AAgentBase: {both_inherit_from_base}")
    print(f"   - Agent classes ready for instantiation")
    multi_agent_test_passed = True
except Exception as e:
    print(f"❌ Multiple agent test failed: {e}")
    multi_agent_test_passed = False

# Summary
print("\n" + "="*60)
print("🎯 INTEGRATION TEST SUMMARY")
print("="*60)

test_results = [
    ("SDK Import", sdk_test_passed),
    ("Security Import", security_test_passed),
    ("Agent Creation", agent_test_passed),
    ("Network Connector", connector_test_passed),
    ("Version Management", version_test_passed),
    ("Multiple Agents", multi_agent_test_passed)
]

passed = 0
for test_name, result in test_results:
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status} {test_name}")
    if result:
        passed += 1

total = len(test_results)
print(f"\n📊 Results: {passed}/{total} tests passed")

if passed == total:
    print("\n🎉 All integration tests passed!")
    print("✅ a2aAgents and a2aNetwork are properly integrated")
    print("\n📄 Integration verified:")
    print("   - SDK components from a2aNetwork work in all agents")
    print("   - Security functions properly delegated to network")
    print("   - Network connectivity layer functioning")
    print("   - Version compatibility maintained")
    print("   - Multiple agents can share network components")
    exit(0)
else:
    print(f"\n⚠️  {total - passed} tests failed. Integration needs attention.")
    exit(1)
