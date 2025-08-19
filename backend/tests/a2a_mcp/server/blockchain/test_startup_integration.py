#!/usr/bin/env python3
"""
Test A2A Startup Integration with Blockchain

This script tests the end-to-end integration when started via ./start.sh blockchain
"""

import os
import sys
import requests
import json
import time
from typing import Dict, Any

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

def test_blockchain_rpc_connection(rpc_url: str = "http://localhost:8545") -> bool:
    """Test blockchain RPC connection"""
    try:
        response = requests.post(
            rpc_url,
            json={"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            if "result" in result:
                print(f"✅ Blockchain RPC connected, block number: {int(result['result'], 16)}")
                return True
        
        print(f"❌ Blockchain RPC response invalid: {response.text}")
        return False
        
    except Exception as e:
        print(f"❌ Blockchain RPC connection failed: {e}")
        return False

def test_contract_deployment(rpc_url: str = "http://localhost:8545") -> Dict[str, Any]:
    """Test if contracts are deployed"""
    try:
        # Check for deployed contracts config
        config_files = [
            "/Users/apple/projects/a2a/a2aNetwork/deployed-contracts.json",
            "/Users/apple/projects/a2a/logs/contract-deploy.log"
        ]
        
        results = {"contracts_found": False, "addresses": {}}
        
        for config_file in config_files:
            if os.path.exists(config_file):
                if config_file.endswith(".json"):
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                            if "contracts" in config:
                                results["contracts_found"] = True
                                results["addresses"] = config["contracts"]
                                print(f"✅ Contract deployment config found: {config_file}")
                                break
                    except:
                        pass
                elif config_file.endswith(".log"):
                    # Check deployment log
                    try:
                        with open(config_file, 'r') as f:
                            log_content = f.read()
                            if "deployed to:" in log_content.lower():
                                results["contracts_found"] = True
                                print(f"✅ Contract deployment log found: {config_file}")
                    except:
                        pass
        
        if not results["contracts_found"]:
            print("⚠️  No contract deployment evidence found")
        
        return results
        
    except Exception as e:
        print(f"❌ Contract deployment check failed: {e}")
        return {"contracts_found": False, "addresses": {}}

def test_agent_blockchain_readiness() -> Dict[str, Any]:
    """Test that agents are ready for blockchain"""
    try:
        # Test environment variables
        required_env_vars = [
            "BLOCKCHAIN_ENABLED",
            "A2A_RPC_URL",
            "AGENT_MANAGER_PRIVATE_KEY"
        ]
        
        results = {"env_vars": {}, "all_set": True}
        
        for var in required_env_vars:
            value = os.environ.get(var)
            results["env_vars"][var] = value is not None
            if value is None:
                results["all_set"] = False
                print(f"❌ Missing environment variable: {var}")
            else:
                print(f"✅ Environment variable set: {var}")
        
        return results
        
    except Exception as e:
        print(f"❌ Agent blockchain readiness check failed: {e}")
        return {"env_vars": {}, "all_set": False}

def test_agent_communication_basic() -> bool:
    """Test basic agent communication capabilities"""
    try:
        # Try to import and test the blockchain integration
        sys.path.append('/Users/apple/projects/a2a/a2aAgents/backend/app/a2a')
        
        from sdk.blockchainIntegration import BlockchainIntegrationMixin
        
        # Create test instance
        mixin = BlockchainIntegrationMixin()
        
        if mixin.blockchain_enabled:
            print("✅ BlockchainIntegrationMixin enabled by default")
            return True
        else:
            print("❌ BlockchainIntegrationMixin not enabled")
            return False
            
    except ImportError as e:
        print(f"⚠️  Cannot import blockchain integration: {e}")
        return False
    except Exception as e:
        print(f"❌ Agent communication test failed: {e}")
        return False

def run_startup_integration_tests():
    """Run all startup integration tests"""
    print("🧪 Running A2A Startup Integration Tests")
    print("=" * 50)
    
    results = {
        "blockchain_rpc": False,
        "contract_deployment": False,
        "agent_readiness": False,
        "agent_communication": False,
        "overall_success": False
    }
    
    # Test 1: Blockchain RPC Connection
    print("\n1. Testing Blockchain RPC Connection")
    results["blockchain_rpc"] = test_blockchain_rpc_connection()
    
    # Test 2: Contract Deployment
    print("\n2. Testing Contract Deployment")
    deployment_result = test_contract_deployment()
    results["contract_deployment"] = deployment_result["contracts_found"]
    
    # Test 3: Agent Blockchain Readiness  
    print("\n3. Testing Agent Blockchain Readiness")
    readiness_result = test_agent_blockchain_readiness()
    results["agent_readiness"] = readiness_result["all_set"]
    
    # Test 4: Agent Communication
    print("\n4. Testing Agent Communication")
    results["agent_communication"] = test_agent_communication_basic()
    
    # Overall Results
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results.items():
        if test_name != "overall_success":
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    # Calculate overall success
    test_results = [v for k, v in results.items() if k != "overall_success"]
    results["overall_success"] = all(test_results)
    
    print(f"\n🎯 Overall Result: {'✅ SUCCESS' if results['overall_success'] else '❌ PARTIAL SUCCESS'}")
    
    if results["overall_success"]:
        print("\n🎉 All tests passed! A2A blockchain integration is working properly.")
        print("   Ready for end-to-end agent communication testing.")
    else:
        print("\n⚠️  Some tests failed, but basic functionality may still work.")
        print("   Check individual test results above for specific issues.")
    
    return results["overall_success"]

if __name__ == "__main__":
    # Check if we're being called from start.sh
    if len(sys.argv) > 1 and sys.argv[1] == "--from-startup":
        # Brief output for startup integration
        success = run_startup_integration_tests()
        sys.exit(0 if success else 1)
    else:
        # Full interactive testing
        success = run_startup_integration_tests()
        sys.exit(0 if success else 1)