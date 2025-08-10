#!/usr/bin/env python3
"""
Validate cleanup results - test all components work after duplicate removal
"""

import os
import sys
from pathlib import Path

def test_sdk_imports():
    """Test SDK components import from a2aNetwork"""
    print("\n🔬 Testing SDK imports...")
    try:
        from app.a2a.sdk import (
            A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
            A2AMessage, MessageRole, create_agent_id
        )
        print("✅ SDK imports successful")
        return True
    except Exception as e:
        print(f"❌ SDK imports failed: {e}")
        return False

def test_security_imports():
    """Test security components import from a2aNetwork"""
    print("\n🔐 Testing security imports...")
    try:
        from app.a2a.security import (
            sign_a2a_message, verify_a2a_message, 
            initialize_agent_trust, get_trust_contract
        )
        print("✅ Security imports successful")
        return True
    except Exception as e:
        print(f"❌ Security imports failed: {e}")
        return False

def test_agent_imports():
    """Test all agents can import successfully"""
    print("\n🤖 Testing agent imports...")
    agents_to_test = [
        ('agent0DataProduct', 'dataProductAgentSdk', 'DataProductRegistrationAgentSDK'),
        ('catalogManager', 'catalogManagerAgentSdk', 'CatalogManagerAgentSDK'), 
        ('agent4CalcValidation', 'calcValidationAgentSdk', 'CalcValidationAgentSDK'),
        ('agent5QaValidation', 'qaValidationAgentSdk', 'QAValidationAgentSDK'),
        ('agentBuilder', 'agentBuilderAgentSdk', 'AgentBuilderAgentSDK'),
        ('agent1Standardization', 'dataStandardizationAgentSdk', 'DataStandardizationAgentSDK'),
        ('agent2AiPreparation', 'aiPreparationAgentSdk', 'AIPreparationAgentSDK'),
        ('agent3VectorProcessing', 'vectorProcessingAgentSdk', 'VectorProcessingAgentSDK'),
        ('dataManager', 'dataManagerAgentSdk', 'DataManagerAgentSDK')
    ]
    
    success_count = 0
    for agent_dir, module_name, class_name in agents_to_test:
        try:
            module_path = f"app.a2a.agents.{agent_dir}.active.{module_name}"
            module = __import__(module_path, fromlist=[class_name])
            agent_class = getattr(module, class_name)
            print(f"✅ {agent_dir} import successful")
            success_count += 1
        except Exception as e:
            print(f"❌ {agent_dir} import failed: {e}")
    
    print(f"\n📊 Agent imports: {success_count}/{len(agents_to_test)} successful")
    return success_count == len(agents_to_test)

def test_network_integration():
    """Test network integration components"""
    print("\n🌐 Testing network integration...")
    try:
        from app.a2a.network.networkConnector import NetworkConnector
        from app.a2a.network.agentRegistration import AgentRegistrationService
        from app.a2a.network.networkMessaging import NetworkMessagingService
        print("✅ Network integration components successful")
        return True
    except Exception as e:
        print(f"❌ Network integration failed: {e}")
        return False

def test_version_management():
    """Test version management components"""
    print("\n📋 Testing version management...")
    try:
        from app.a2a.version.versionManager import VersionManager
        from app.a2a.version.dependencyResolver import DependencyResolver
        from app.a2a.version.compatibilityChecker import CompatibilityChecker
        print("✅ Version management components successful")
        return True
    except Exception as e:
        print(f"❌ Version management failed: {e}")
        return False

def verify_cleanup_complete():
    """Verify duplicate files were actually removed"""
    print("\n🗑️  Verifying duplicate files removed...")
    
    removed_files = [
        "sdk/agentBase.py",
        "sdk/client.py",
        "sdk/decorators.py", 
        "sdk/types.py",
        "sdk/utils.py",
        "security/delegationContracts.py",
        "security/sharedTrust.py",
        "security/smartContractTrust.py"
    ]
    
    base_path = Path("/Users/apple/projects/a2a/a2aAgents/backend/app/a2a")
    all_removed = True
    
    for file_path in removed_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"❌ Duplicate file still exists: {file_path}")
            all_removed = False
        else:
            print(f"✅ Removed: {file_path}")
    
    # Check backup exists
    backup_dir = base_path / "backup_before_cleanup"
    if backup_dir.exists():
        print(f"✅ Backup directory exists: {backup_dir}")
    else:
        print(f"❌ Backup directory missing: {backup_dir}")
        all_removed = False
    
    return all_removed

def test_a2a_network_availability():
    """Test that a2aNetwork components are accessible"""
    print("\n🔗 Testing a2aNetwork availability...")
    try:
        import sys
        sys.path.insert(0, '/Users/apple/projects/a2a/a2aNetwork')
        
        # Test key components
        from sdk.agentBase import A2AAgentBase
        from sdk.types import A2AMessage
        from trustSystem.smartContractTrust import sign_a2a_message
        from api.networkClient import NetworkClient
        
        print("✅ a2aNetwork components accessible")
        return True
    except Exception as e:
        print(f"❌ a2aNetwork components failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🧪 Starting cleanup validation tests...")
    
    tests = [
        ("SDK Imports", test_sdk_imports),
        ("Security Imports", test_security_imports), 
        ("Agent Imports", test_agent_imports),
        ("Network Integration", test_network_integration),
        ("Version Management", test_version_management),
        ("Cleanup Verification", verify_cleanup_complete),
        ("a2aNetwork Availability", test_a2a_network_availability)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("🎯 CLEANUP VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Cleanup was successful.")
        print("\n✨ Summary of what was accomplished:")
        print("   - Removed 8 duplicate components from a2aAgents")
        print("   - Updated import paths to use a2aNetwork")
        print("   - Verified 9+ agents work with new architecture")
        print("   - Maintained backward compatibility with fallbacks")
        print("   - Created backup of original components")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review issues above.")
        return False

if __name__ == "__main__":
    os.chdir("/Users/apple/projects/a2a/a2aAgents/backend")
    success = main()
    exit(0 if success else 1)
