#!/usr/bin/env python3
"""
A2A Network Deployment CLI
Command-line interface for managing A2A Network deployments
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
import json

# Add app directory to path
current_dir = Path(__file__).parent
app_dir = current_dir.parent.parent
sys.path.insert(0, str(app_dir))

from a2a.deployment import (
    create_deployment_coordinator,
    NetworkConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def cmd_deploy_dev(args):
    """Deploy contracts for development"""
    print("🚀 DEPLOYING A2A NETWORK FOR DEVELOPMENT")
    print("=" * 50)
    
    try:
        result = await deploy_for_development(
            port=args.port,
            a2a_network_path=args.a2a_path
        )
        
        if result.success:
            print(f"✅ Development deployment successful!")
            print(f"   Network: {result.network}")
            print(f"   Deployment time: {result.deployment_time:.2f}s")
            print(f"   Contracts deployed:")
            
            for name, address in result.contracts.items():
                print(f"     - {name}: {address}")
            
            print(f"\n🔧 Configuration updated for finsight_cib")
            print(f"   Agents can now connect to blockchain at http://localhost:{args.port}")
            
        else:
            print(f"❌ Development deployment failed!")
            if result.error:
                print(f"   Error: {result.error}")
            return 1
            
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        return 1
    
    return 0

async def cmd_deploy_testnet(args):
    """Deploy contracts to testnet"""
    print(f"🚀 DEPLOYING A2A NETWORK TO {args.network.upper()} TESTNET")
    print("=" * 60)
    
    try:
        # Get private key
        private_key = args.private_key or os.getenv("PRIVATE_KEY")
        if not private_key:
            print("❌ Private key required for testnet deployment")
            print("   Use --private-key or set PRIVATE_KEY environment variable")
            return 1
        
        result = await deploy_for_testing(
            network=args.network,
            private_key=private_key,
            a2a_network_path=args.a2a_path
        )
        
        if result.success:
            print(f"✅ {args.network} deployment successful!")
            print(f"   Network: {result.network}")
            print(f"   Deployment time: {result.deployment_time:.2f}s")
            print(f"   Contracts deployed:")
            
            for name, address in result.contracts.items():
                print(f"     - {name}: {address}")
            
            if result.transaction_hashes:
                print(f"   Transaction hashes:")
                for name, tx_hash in result.transaction_hashes.items():
                    print(f"     - {name}: {tx_hash}")
            
            print(f"\n🔧 Configuration updated for finsight_cib")
            print(f"   Agents configured for {args.network} testnet")
            
        else:
            print(f"❌ {args.network} deployment failed!")
            if result.error:
                print(f"   Error: {result.error}")
            return 1
            
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        return 1
    
    return 0

def cmd_status(args):
    """Show deployment status"""
    print("📊 A2A NETWORK DEPLOYMENT STATUS")
    print("=" * 40)
    
    try:
        coordinator = create_deployment_coordinator(args.a2a_path)
        status = coordinator.get_deployment_status()
        
        if status.get("deployed"):
            print("✅ Contracts are deployed")
            print(f"   Network: {status.get('network')}")
            print(f"   Contract count: {status.get('contract_count')}")
            
            contracts = status.get("contracts", {})
            if contracts:
                print("   Deployed contracts:")
                for name, address in contracts.items():
                    print(f"     - {name}: {address}")
            
            deployment_time = status.get("deployment_time")
            if deployment_time:
                print(f"   Deployment time: {deployment_time:.2f}s")
                
        else:
            print("⚠️  No deployment found")
            if "error" in status:
                print(f"   Error: {status['error']}")
            else:
                print("   Run 'deploy dev' or 'deploy testnet' to deploy contracts")
        
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return 1
    
    return 0

def cmd_config(args):
    """Show current configuration"""
    print("⚙️  A2A NETWORK CONFIGURATION")
    print("=" * 35)
    
    try:
        # Show deployment config
        config_file = Path(__file__).parent.parent / "config" / "contracts.json"
        
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            
            print("📄 Contract Configuration:")
            print(f"   Network: {config.get('network')}")
            print(f"   Contracts: {len(config.get('contracts', {}))}")
            
            for name, address in config.get('contracts', {}).items():
                print(f"     - {name}: {address}")
        else:
            print("⚠️  No contract configuration found")
        
        # Show environment config
        env_file = Path(__file__).parent.parent / "config" / ".env.contracts"
        
        if env_file.exists():
            print(f"\n🔧 Environment Configuration:")
            print(f"   File: {env_file}")
            
            with open(env_file) as f:
                lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
                print(f"   Variables: {len(lines)}")
        else:
            print(f"\n⚠️  No environment configuration found")
        
        # Show a2a_network path
        try:
            coordinator = create_deployment_coordinator(args.a2a_path)
            print(f"\n📁 A2A Network Path: {coordinator.a2a_network_path}")
        except Exception as e:
            print(f"\n❌ A2A Network path error: {e}")
        
    except Exception as e:
        print(f"❌ Config check failed: {e}")
        return 1
    
    return 0

async def cmd_test_connection(args):
    """Test blockchain connection"""
    print("🔌 TESTING BLOCKCHAIN CONNECTION")
    print("=" * 35)
    
    try:
        from a2a.config.contractConfig import get_contractConfig, validate_contracts
        from a2a.blockchain.web3Client import get_blockchain_client
        
        # Test configuration
        print("1️⃣ Testing contract configuration...")
        config = get_contractConfig()
        validation = config.validate_configuration()
        
        if validation['valid']:
            print("   ✅ Contract configuration valid")
            print(f"   📊 Network: {config.network}")
            print(f"   📊 Contracts loaded: {len(config.contracts)}")
        else:
            print("   ❌ Contract configuration invalid")
            for error in validation['errors']:
                print(f"     - {error}")
            return 1
        
        # Test blockchain connection
        print("\n2️⃣ Testing blockchain connection...")
        client = get_blockchain_client()
        
        if client.web3.is_connected():
            print("   ✅ Blockchain connection successful")
            print(f"   📊 RPC URL: {client.rpc_url}")
            print(f"   📊 Chain ID: {client.web3.eth.chain_id}")
            print(f"   📊 Latest block: {client.web3.eth.block_number}")
            print(f"   📊 Agent address: {client.agent_identity.address}")
            print(f"   📊 Agent balance: {client.get_balance():.4f} ETH")
        else:
            print("   ❌ Blockchain connection failed")
            return 1
        
        # Test contract interaction
        print("\n3️⃣ Testing contract interaction...")
        
        if config.is_contract_available("AgentRegistry"):
            registry_addr = config.get_contract_address("AgentRegistry")
            print(f"   ✅ AgentRegistry available: {registry_addr}")
        else:
            print("   ⚠️  AgentRegistry not available")
        
        if config.is_contract_available("MessageRouter"):
            router_addr = config.get_contract_address("MessageRouter")
            print(f"   ✅ MessageRouter available: {router_addr}")
        else:
            print("   ⚠️  MessageRouter not available")
        
        print("\n🎉 Connection test completed successfully!")
        print("   All systems ready for A2A Network integration")
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="A2A Network Deployment CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy for development
  python cli.py deploy dev
  
  # Deploy to Sepolia testnet
  python cli.py deploy testnet --network sepolia --private-key 0x...
  
  # Check deployment status
  python cli.py status
  
  # Test blockchain connection
  python cli.py test
        """
    )
    
    parser.add_argument(
        "--a2a-path",
        help="Path to a2a_network project directory"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy contracts")
    deploy_subparsers = deploy_parser.add_subparsers(dest="deploy_type")
    
    # Deploy dev
    dev_parser = deploy_subparsers.add_parser("dev", help="Deploy for development")
    dev_parser.add_argument("--port", type=int, default=8545, help="Anvil port (default: 8545)")
    
    # Deploy testnet
    testnet_parser = deploy_subparsers.add_parser("testnet", help="Deploy to testnet")
    testnet_parser.add_argument("--network", choices=["sepolia", "goerli"], default="sepolia", help="Testnet network")
    testnet_parser.add_argument("--private-key", help="Private key for deployment")
    
    # Status command
    subparsers.add_parser("status", help="Show deployment status")
    
    # Config command
    subparsers.add_parser("config", help="Show configuration")
    
    # Test command
    subparsers.add_parser("test", help="Test blockchain connection")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == "deploy":
        if args.deploy_type == "dev":
            return asyncio.run(cmd_deploy_dev(args))
        elif args.deploy_type == "testnet":
            return asyncio.run(cmd_deploy_testnet(args))
        else:
            deploy_parser.print_help()
            return 1
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "config":
        return cmd_config(args)
    elif args.command == "test":
        return asyncio.run(cmd_test_connection(args))
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())