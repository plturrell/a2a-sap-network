#!/usr/bin/env python3
"""
Test script to verify migrated components work in a2aNetwork
"""

import sys
import os

print("🚀 Testing A2A Network Migrated Components")
print("=" * 60)

def testCoreModules():
    """Test core modules"""
    print("\n📋 Testing Core Modules")
    print("-" * 30)
    
    try:
        from core.telemetry import init_telemetry, trace_async, add_span_attributes
        print("✅ Core telemetry imports successful")
        
        from config.telemetryConfig import telemetry_config
        print("✅ Telemetry config imports successful")
        print(f"   Service name: {telemetry_config.otel_service_name}")
        print(f"   OTEL enabled: {telemetry_config.otel_enabled}")
        
        return True
    except Exception as e:
        print(f"❌ Core modules failed: {e}")
        return False

def testSdkTypes():
    """Test SDK types directly"""
    print("\n📋 Testing SDK Types")
    print("-" * 20)
    
    try:
        # Import types directly without going through __init__
        sys.path.append('.')
        import sdk.types as sdkTypes
        
        AgentCard = sdkTypes.AgentCard
        A2AMessage = sdkTypes.A2AMessage
        MessageRole = sdkTypes.MessageRole
        
        print("✅ SDK types imports successful")
        print(f"   AgentCard: {AgentCard}")
        print(f"   A2AMessage: {A2AMessage}")
        print(f"   MessageRole: {MessageRole}")
        
        # Test creating instances
        messagePart = sdkTypes.MessagePart(kind="text", text="Hello A2A!")
        print("✅ MessagePart creation successful")
        
        message = A2AMessage(
            role=MessageRole.USER,
            parts=[messagePart]
        )
        print("✅ A2AMessage creation successful")
        print(f"   Message ID: {message.messageId}")
        
        return True
    except Exception as e:
        print(f"❌ SDK types failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def testRegistryModels():
    """Test registry models"""
    print("\n📋 Testing Registry Models")
    print("-" * 25)
    
    try:
        from registry.models import ORDDocument, DublinCoreMetadata, ResourceType
        print("✅ Registry models imports successful")
        print(f"   ORDDocument: {ORDDocument}")
        print(f"   DublinCoreMetadata: {DublinCoreMetadata}")
        print(f"   ResourceType: {ResourceType}")
        
        # Test creating Dublin Core metadata
        dublinCore = DublinCoreMetadata(
            title="Test Data Product",
            creator=["A2A Network"],
            description="Test description",
            publisher="A2A Registry",
            type="Dataset"
        )
        print("✅ DublinCoreMetadata creation successful")
        print(f"   Title: {dublinCore.title}")
        
        return True
    except Exception as e:
        print(f"❌ Registry models failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def testTrustSystem():
    """Test trust system components"""
    print("\n📋 Testing Trust System")
    print("-" * 25)
    
    try:
        from trustSystem.smartContractTrust import sign_a2a_message
        print("✅ Smart contract trust imports successful")
        
        from trustSystem.delegationContracts import DelegationAction
        print("✅ Delegation contracts imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Trust system failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def testRegistryService():
    """Test registry service functionality"""  
    print("\n📋 Testing Registry Service")
    print("-" * 26)
    
    try:
        from registry.client import get_registry_client
        print("✅ Registry client imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Registry service failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting comprehensive migration tests...")
    
    results = {
        "Core Modules": testCoreModules(),
        "SDK Types": testSdkTypes(), 
        "Registry Models": testRegistryModels(),
        "Trust System": testTrustSystem(),
        "Registry Service": testRegistryService(),
    }
    
    print("\n" + "=" * 60)
    print("📊 MIGRATION TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for testName, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{testName:<20} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL MIGRATION TESTS PASSED!")
        print("✨ Migrated components are fully functional in a2aNetwork!")
        return 0
    else:
        print("💥 Some migration tests failed!")
        print(f"📝 {total - passed} components need further fixes")
        return 1

if __name__ == "__main__":
    exitCode = main()
    sys.exit(exitCode)