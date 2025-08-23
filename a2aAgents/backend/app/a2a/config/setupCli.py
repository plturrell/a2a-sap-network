#!/usr/bin/env python3
"""
A2A Network Configuration Setup CLI
Interactive setup and validation tool for production configurations
"""

import argparse
import sys
import os
from pathlib import Path
import json
import shutil
from typing import Dict, Any
import logging

# Add app directory to path
current_dir = Path(__file__).parent
app_dir = current_dir.parent.parent
sys.path.insert(0, str(app_dir))

from a2a.config.deploymentConfig import (
    DeploymentConfigManager as ProductionConfigManager,
    Environment,
    get_deployment_config as get_production_config,
    validate_deployment_setup as validate_production_setup
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def cmd_init(args):
    """Initialize configuration for a specific environment"""
    print(f"🚀 INITIALIZING A2A NETWORK CONFIGURATION")
    print(f"Environment: {args.environment.upper()}")
    print("=" * 50)
    
    try:
        # Create configuration manager
        config = ProductionConfigManager(environment=args.environment)
        
        # Create configuration directory
        config_dir = Path(args.config_dir or ".")
        config_dir.mkdir(exist_ok=True)
        
        # Copy appropriate template
        template_file = current_dir / "templates" / f"env.{args.environment}.template"
        target_file = config_dir / f".env.{args.environment}"
        
        if template_file.exists():
            shutil.copy2(template_file, target_file)
            print(f"✅ Created environment file: {target_file}")
        else:
            # Generate template
            template_content = config.create_environment_template(Environment(args.environment))
            with open(target_file, 'w') as f:
                f.write(template_content)
            print(f"✅ Generated environment file: {target_file}")
        
        # Copy Docker Compose for production
        if args.environment == "production":
            docker_template = current_dir / "templates" / "docker-compose.production.yml"
            docker_target = config_dir / "docker-compose.yml"
            
            if docker_template.exists():
                shutil.copy2(docker_template, docker_target)
                print(f"✅ Created Docker Compose: {docker_target}")
        
        # Save configuration
        config.save_configuration()
        print(f"✅ Saved configuration file")
        
        # Show next steps
        print(f"\n📋 NEXT STEPS:")
        print(f"1. Edit {target_file} with your specific values")
        print(f"2. Replace placeholder addresses with actual contract addresses")
        print(f"3. Set secure private keys (encrypted for production)")
        print(f"4. Run 'python setup_cli.py validate --environment {args.environment}' to check configuration")
        
        if args.environment == "production":
            print(f"5. Review and customize docker-compose.yml for your deployment")
            print(f"6. Set up SSL certificates in nginx/ssl/ directory")
        
        return 0
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return 1

def cmd_validate(args):
    """Validate configuration for a specific environment"""
    print(f"🔍 VALIDATING A2A NETWORK CONFIGURATION")
    print(f"Environment: {args.environment.upper()}")
    print("=" * 50)
    
    try:
        # Set environment variables from file if exists
        env_file = Path(args.config_dir or ".") / f".env.{args.environment}"
        if env_file.exists():
            print(f"📄 Loading environment from: {env_file}")
            _load_env_file(env_file)
        
        # Create configuration manager
        config = ProductionConfigManager(environment=args.environment, config_dir=args.config_dir)
        
        # Validate configuration
        validation = config.validate_configuration()
        
        print(f"\n🔧 CONFIGURATION SUMMARY")
        print(f"Environment: {validation['environment']}")
        print(f"Config Directory: {config.config_dir}")
        
        # Network configuration
        print(f"\n🌐 NETWORK CONFIGURATION")
        print(f"Network: {config.network.name}")
        print(f"RPC URL: {config.network.rpc_url}")
        print(f"Chain ID: {config.network.chain_id}")
        print(f"Block Confirmations: {config.network.block_confirmation_count}")
        
        # Security configuration
        print(f"\n🔒 SECURITY CONFIGURATION")
        print(f"HTTPS Required: {config.security.require_https}")
        print(f"Rate Limiting: {config.security.enable_rate_limiting}")
        print(f"Private Key Encrypted: {config.security.private_key_encrypted}")
        
        # Contract configuration
        print(f"\n📋 CONTRACT CONFIGURATION")
        print(f"Agent Registry: {config.contracts.agent_registry_address or 'Not configured'}")
        print(f"Message Router: {config.contracts.message_router_address or 'Not configured'}")
        print(f"ORD Registry: {config.contracts.ord_registry_address or 'Not configured'}")
        
        # Validation results
        print(f"\n🎯 VALIDATION RESULTS")
        print(f"Status: {'✅ VALID' if validation['valid'] else '❌ INVALID'}")
        
        if validation['errors']:
            print(f"\n❌ ERRORS ({len(validation['errors'])}):")
            for error in validation['errors']:
                print(f"   • {error}")
        
        if validation['warnings']:
            print(f"\n⚠️  WARNINGS ({len(validation['warnings'])}):")
            for warning in validation['warnings']:
                print(f"   • {warning}")
        
        if validation['valid']:
            print(f"\n🎉 Configuration is valid and ready for deployment!")
        else:
            print(f"\n💡 Please fix the errors above before deployment")
        
        return 0 if validation['valid'] else 1
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1

def cmd_show(args):
    """Show current configuration"""
    print(f"📊 A2A NETWORK CONFIGURATION")
    print(f"Environment: {args.environment.upper()}")
    print("=" * 40)
    
    try:
        # Load environment file
        env_file = Path(args.config_dir or ".") / f".env.{args.environment}"
        if env_file.exists():
            _load_env_file(env_file)
        
        # Create configuration manager
        config = ProductionConfigManager(environment=args.environment, config_dir=args.config_dir)
        
        # Get environment info
        info = config.get_environment_info()
        
        # Display configuration
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            _display_config_human_readable(info)
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to show configuration: {e}")
        return 1

def cmd_generate_keys(args):
    """Generate secure private keys"""
    print(f"🔐 GENERATING SECURE PRIVATE KEYS")
    print("=" * 40)
    
    try:
        from eth_account import Account
        
        # Generate new account
        account = Account.create()
        
        print(f"✅ Generated new agent identity:")
        print(f"Address: {account.address}")
        print(f"Private Key: {account.key.hex()}")
        
        print(f"\n⚠️  SECURITY WARNING:")
        print(f"• Store the private key securely")
        print(f"• Never commit private keys to version control")
        print(f"• Use encrypted storage for production")
        print(f"• Consider hardware wallets for high-value accounts")
        
        if args.environment == "production":
            print(f"\n🔒 PRODUCTION RECOMMENDATIONS:")
            print(f"• Use encrypted private key storage")
            print(f"• Set A2A_PRIVATE_KEY_ENCRYPTED=true")
            print(f"• Consider multi-signature wallets")
            print(f"• Implement key rotation policies")
        
        return 0
        
    except Exception as e:
        print(f"❌ Key generation failed: {e}")
        return 1

def cmd_test_connection(args):
    """Test blockchain connection"""
    print(f"🔌 TESTING BLOCKCHAIN CONNECTION")
    print(f"Environment: {args.environment.upper()}")
    print("=" * 40)
    
    try:
        # Load environment file
        env_file = Path(args.config_dir or ".") / f".env.{args.environment}"
        if env_file.exists():
            _load_env_file(env_file)
        
        # Test connection using blockchain client
        from a2a_network.python_sdk.blockchain.web3_client import get_blockchain_client
        from a2a.config.contract_config import validate_contracts
        
        print("1️⃣ Testing blockchain connection...")
        client = get_blockchain_client()
        
        if client.web3.is_connected():
            print("   ✅ Connected to blockchain")
            print(f"   Chain ID: {client.web3.eth.chain_id}")
            print(f"   Latest block: {client.web3.eth.block_number}")
            print(f"   Agent address: {client.agent_identity.address}")
            print(f"   Agent balance: {client.get_balance():.4f} ETH")
        else:
            print("   ❌ Failed to connect to blockchain")
            return 1
        
        print("\n2️⃣ Testing contract configuration...")
        if validate_contracts():
            print("   ✅ Contract configuration valid")
        else:
            print("   ❌ Contract configuration invalid")
            return 1
        
        print("\n🎉 All connection tests passed!")
        return 0
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return 1

def _load_env_file(env_file: Path):
    """Load environment variables from file"""
    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    except Exception as e:
        logger.warning(f"Failed to load environment file {env_file}: {e}")

def _display_config_human_readable(info: Dict[str, Any]):
    """Display configuration in human-readable format"""
    print(f"Environment: {info['environment']}")
    print(f"Config Directory: {info['config_dir']}")
    
    print(f"\n🔒 Security:")
    for key, value in info['security'].items():
        print(f"   {key}: {value}")
    
    print(f"\n🌐 Network:")
    for key, value in info['network'].items():
        print(f"   {key}: {value}")
    
    print(f"\n📊 Monitoring:")
    for key, value in info['monitoring'].items():
        print(f"   {key}: {value}")
    
    print(f"\n📋 Contracts:")
    for key, value in info['contracts'].items():
        print(f"   {key}: {value or 'Not configured'}")
    
    validation = info['validation']
    print(f"\n🎯 Validation: {'✅ VALID' if validation['valid'] else '❌ INVALID'}")
    
    if validation['errors']:
        print(f"   Errors: {len(validation['errors'])}")
    if validation['warnings']:
        print(f"   Warnings: {len(validation['warnings'])}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="A2A Network Configuration Setup CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config-dir",
        help="Configuration directory path"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize configuration")
    init_parser.add_argument(
        "--environment", "-e",
        choices=["development", "testing", "staging", "production"],
        required=True,
        help="Target environment"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument(
        "--environment", "-e",
        choices=["development", "testing", "staging", "production"],
        required=True,
        help="Target environment"
    )
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show configuration")
    show_parser.add_argument(
        "--environment", "-e",
        choices=["development", "testing", "staging", "production"],
        required=True,
        help="Target environment"
    )
    show_parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    
    # Generate keys command
    keys_parser = subparsers.add_parser("generate-keys", help="Generate secure private keys")
    keys_parser.add_argument(
        "--environment", "-e",
        choices=["development", "testing", "staging", "production"],
        default="development",
        help="Target environment (affects security recommendations)"
    )
    
    # Test connection command
    test_parser = subparsers.add_parser("test", help="Test blockchain connection")
    test_parser.add_argument(
        "--environment", "-e",
        choices=["development", "testing", "staging", "production"],
        required=True,
        help="Target environment"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == "init":
        return cmd_init(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "show":
        return cmd_show(args)
    elif args.command == "generate-keys":
        return cmd_generate_keys(args)
    elif args.command == "test":
        return cmd_test_connection(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())