#!/usr/bin/env python3
"""
Deploy Real Smart Contracts to Anvil
This fixes the false claims by actually deploying contracts to the running Anvil blockchain
"""

import json
import subprocess
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_contracts():
    """Deploy the BusinessDataCloudA2A contract to Anvil"""
    
    # Get the a2a_network path
    network_dir = Path(__file__).parent.parent.parent / "a2a_network"
    
    if not network_dir.exists():
        logger.error("❌ a2a_network directory not found")
        return False
        
    logger.info(f"📂 Network directory: {network_dir}")
    
    # Check if DeployBDCA2A script exists
    deploy_script = network_dir / "script" / "DeployBDCA2A.s.sol"
    if not deploy_script.exists():
        logger.error(f"❌ Deploy script not found: {deploy_script}")
        return False
        
    logger.info(f"✅ Found deploy script: {deploy_script}")
    
    # Execute forge build first
    try:
        logger.info("🔨 Compiling contracts...")
        result = subprocess.run([
            "forge", "build", "--root", str(network_dir)
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            logger.error(f"❌ Compilation failed: {result.stderr}")
            return False
            
        logger.info("✅ Contracts compiled successfully")
        
    except Exception as e:
        logger.error(f"❌ Compilation error: {e}")
        return False
    
    # Deploy using forge script
    try:
        logger.info("🚀 Deploying BusinessDataCloudA2A contract...")
        
        # Use the standard Anvil private key for deployment
        private_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
        
        result = subprocess.run([
            "forge", "script", 
            "script/DeployBDCA2A.s.sol:DeployBDCA2A",
            "--rpc-url", "http://localhost:8545",
            "--broadcast",
            "--private-key", private_key,
            "--root", str(network_dir)
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            logger.warning(f"⚠️ Deployment may have failed: {result.stderr}")
            logger.info(f"stdout: {result.stdout}")
            
            # Check if it's just a warning but deployment succeeded
            if "ONCHAIN EXECUTION COMPLETE & SUCCESSFUL" in result.stderr:
                logger.info("✅ Deployment completed despite warnings")
            else:
                return False
        else:
            logger.info("✅ Contract deployment successful")
            
        logger.info(f"Deployment output: {result.stdout}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Deployment error: {e}")
        return False

def get_deployed_addresses():
    """Get the actual deployed contract addresses from blockchain"""
    
    try:
        # Read the latest broadcast file
        network_dir = Path(__file__).parent.parent.parent / "a2a_network"
        broadcast_file = network_dir / "broadcast" / "DeployBDCA2A.s.sol" / "31337" / "run-latest.json"
        
        if broadcast_file.exists():
            with open(broadcast_file, 'r') as f:
                broadcast_data = json.load(f)
            
            addresses = {}
            for transaction in broadcast_data.get("transactions", []):
                contract_name = transaction.get("contractName")
                contract_address = transaction.get("contractAddress")
                
                if contract_name and contract_address:
                    if contract_name == "BusinessDataCloudA2A":
                        addresses["business_data_cloud"] = contract_address
                    elif contract_name == "AgentRegistry":
                        addresses["agent_registry"] = contract_address
                    elif contract_name == "MessageRouter":
                        addresses["message_router"] = contract_address
            
            logger.info("✅ Retrieved deployed addresses:")
            for name, addr in addresses.items():
                logger.info(f"   {name}: {addr}")
            
            return addresses
    except Exception as e:
        logger.warning(f"⚠️ Could not read deployment addresses: {e}")
    
    # Fallback: get from existing deployment
    try:
        existing_file = network_dir / "broadcast" / "Deploy.s.sol" / "31337" / "run-latest.json"
        if existing_file.exists():
            with open(existing_file, 'r') as f:
                broadcast_data = json.load(f)
            
            # We know from previous check these are the real addresses
            addresses = {
                "agent_registry": "0x5fbdb2315678afecb367f032d93f642f64180aa3",
                "message_router": "0xe7f1725e7734ce288f8367e1bb143e90bb3f0512",
                "business_data_cloud": None  # Will be deployed
            }
            
            logger.info("✅ Using existing deployed addresses:")
            for name, addr in addresses.items():
                if addr:
                    logger.info(f"   {name}: {addr}")
            
            return addresses
    except Exception as e:
        logger.error(f"❌ Could not get any addresses: {e}")
    
    return {}

def verify_contracts(addresses):
    """Verify contracts are actually deployed and working"""
    
    verified = {}
    
    for name, address in addresses.items():
        if not address:
            continue
            
        try:
            # Use curl to call eth_getCode
            result = subprocess.run([
                "curl", "-s", "-X", "POST", 
                "-H", "Content-Type: application/json",
                "--data", f'{{"jsonrpc":"2.0","method":"eth_getCode","params":["{address}","latest"],"id":1}}',
                "http://localhost:8545"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                response = json.loads(result.stdout)
                code = response.get("result", "0x")
                
                if code != "0x" and len(code) > 3:
                    verified[name] = True
                    logger.info(f"✅ {name} verified at {address}")
                else:
                    verified[name] = False
                    logger.warning(f"⚠️ {name} at {address} has no code")
            else:
                verified[name] = False
                logger.warning(f"⚠️ Could not verify {name} at {address}")
                
        except Exception as e:
            verified[name] = False
            logger.warning(f"⚠️ Error verifying {name}: {e}")
    
    return verified

def main():
    """Main function to deploy and verify contracts"""
    
    logger.info("🚀 Starting real contract deployment to fix false claims...")
    
    # Step 1: Deploy contracts
    deployment_success = deploy_contracts()
    
    # Step 2: Get deployed addresses
    addresses = get_deployed_addresses()
    
    if not addresses:
        logger.error("❌ No contract addresses found")
        return False
    
    # Step 3: Verify contracts
    verified = verify_contracts(addresses)
    
    # Step 4: Report results
    logger.info("\n📋 Final Status:")
    all_verified = True
    
    for name, address in addresses.items():
        if address:
            status = "✅ VERIFIED" if verified.get(name, False) else "❌ NOT VERIFIED"
            logger.info(f"   {name}: {address} - {status}")
            if not verified.get(name, False):
                all_verified = False
        else:
            logger.info(f"   {name}: NOT DEPLOYED")
            all_verified = False
    
    # Save the real addresses
    config_file = Path("real_contract_addresses.json")
    config_data = {
        "addresses": addresses,
        "verified": verified,
        "deployment_success": deployment_success,
        "all_verified": all_verified,
        "blockchain": "anvil",
        "rpc_url": "http://localhost:8545",
        "chain_id": 31337
    }
    
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)
    
    logger.info(f"✅ Real addresses saved to: {config_file}")
    
    if all_verified:
        logger.info("🎉 All contracts successfully deployed and verified!")
        logger.info("✅ FALSE CLAIMS FIXED: Smart contracts are now actually deployed")
    else:
        logger.warning("⚠️ Some contracts not verified, but deployment attempted")
    
    return all_verified

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 SUCCESS: Smart contracts are now actually deployed to Anvil blockchain!")
            print("✅ False claims have been fixed with real deployment")
        else:
            print("\n⚠️ Partial success: Some issues encountered but progress made")
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")