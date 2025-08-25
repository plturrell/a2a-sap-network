#!/usr/bin/env python3
"""
Test script for dynamic contract configuration
Validates that the configuration system can load contracts from deployment artifacts
"""

import sys
import os
from pathlib import Path

# Add app directory to path
current_dir = Path(__file__).parent
app_dir = current_dir.parent.parent
sys.path.insert(0, str(app_dir))

def test_contract_configuration():
    """Test the dynamic contract configuration system"""

    print("🧪 TESTING DYNAMIC CONTRACT CONFIGURATION")
    print("=" * 50)

    try:
        # Import and test configuration
        from a2a.config.contract_config import (
            ContractConfigManager,
            get_contract_config,
            validate_contracts,
            get_agent_registry_address,
            get_message_router_address
        )

        print("\n1️⃣ INITIALIZING CONFIGURATION MANAGER")
        print("-" * 40)

        config = ContractConfigManager()
        print(f"✅ Configuration initialized")
        print(f"   Network: {config.network}")
        print(f"   Artifacts Path: {config.artifacts_path or 'Not found'}")
        print(f"   Loaded Contracts: {len(config.contracts)}")

        print("\n2️⃣ VALIDATING CONFIGURATION")
        print("-" * 35)

        validation = config.validate_configuration()
        print(f"✅ Validation Status: {'VALID' if validation['valid'] else 'INVALID'}")

        if validation['errors']:
            print("   Errors:")
            for error in validation['errors']:
                print(f"     - {error}")

        if validation['warnings']:
            print("   Warnings:")
            for warning in validation['warnings']:
                print(f"     - {warning}")

        if validation['contracts']:
            print("   Loaded Contracts:")
            for name, info in validation['contracts'].items():
                print(f"     - {name}: {info['address'][:20]}...")

        print("\n3️⃣ TESTING CONVENIENCE FUNCTIONS")
        print("-" * 40)

        registry_addr = get_agent_registry_address()
        router_addr = get_message_router_address()

        print(f"✅ AgentRegistry Address: {registry_addr[:20] + '...' if registry_addr else 'Not configured'}")
        print(f"✅ MessageRouter Address: {router_addr[:20] + '...' if router_addr else 'Not configured'}")

        contracts_valid = validate_contracts()
        print(f"✅ Contract Validation: {'PASS' if contracts_valid else 'FAIL'}")

        print("\n4️⃣ TESTING GLOBAL CONFIGURATION")
        print("-" * 40)

        global_config = get_contract_config()
        print(f"✅ Global config same instance: {global_config is config}")
        print(f"✅ Global config network: {global_config.network}")

        print("\n5️⃣ TESTING CONTRACT DETAILS")
        print("-" * 35)

        for contract_name in ['AgentRegistry', 'MessageRouter', 'ORDRegistry']:
            contract_info = config.get_contract(contract_name)
            if contract_info:
                print(f"✅ {contract_name}:")
                print(f"   Address: {contract_info.address}")
                print(f"   Network: {contract_info.network}")
                print(f"   ABI Loaded: {len(contract_info.abi) > 0}")
                print(f"   Available: {config.is_contract_available(contract_name)}")
            else:
                print(f"⚠️  {contract_name}: Not configured")

        print("\n" + "=" * 50)
        print("🎯 CONFIGURATION TEST RESULTS")
        print("=" * 50)

        results = [
            f"✅ Configuration Manager: {'WORKING' if config else 'FAILED'}",
            f"✅ Validation: {'PASS' if validation['valid'] else 'FAIL'}",
            f"✅ Contract Loading: {'SUCCESS' if len(config.contracts) > 0 else 'NO_CONTRACTS'}",
            f"✅ Global Access: {'WORKING' if global_config else 'FAILED'}",
            f"✅ Ready for Integration: {'YES' if contracts_valid else 'NEEDS_SETUP'}"
        ]

        for result in results:
            print(result)

        if validation['valid'] and len(config.contracts) > 0:
            print(f"\n🚀 DYNAMIC CONFIGURATION IS WORKING!")
            print(f"   • Found {len(config.contracts)} contracts")
            print(f"   • Network: {config.network}")
            print(f"   • Source: {'Deployment Artifacts' if config.artifacts_path else 'Environment Variables'}")
            return True
        else:
            print(f"\n⚠️  CONFIGURATION NEEDS SETUP")
            print(f"   • Deploy a2a_network contracts")
            print(f"   • Set environment variables")
            print(f"   • Check artifacts path")
            return validation['valid']

    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting dynamic contract configuration test...")
    success = test_contract_configuration()
    if success:
        print("\n🎉 CONFIGURATION TEST PASSED!")
        print("Dynamic contract configuration is ready for use! 🚀")
    else:
        print("\n❌ Configuration test failed - check the errors above")
