#!/usr/bin/env python3
"""
Blue/Green Deployment CLI for A2A Network
Command-line interface for managing zero-downtime deployments
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
import yaml

from blue_green_deployment import create_blue_green_manager, DeploymentConfig, ServiceConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def deploy_command(args):
    """Execute deployment command"""
    config_path = Path(args.config)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    # Load deployment configuration
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Convert to DeploymentConfig model
    services = [ServiceConfig(**service) for service in config_data['services']]
    deployment_config = DeploymentConfig(
        version=config_data['version'],
        services=services,
        pre_deployment_checks=config_data.get('pre_deployment_checks', []),
        post_deployment_tests=config_data.get('post_deployment_tests', []),
        rollback_on_failure=config_data.get('rollback_on_failure', True),
        health_check_timeout=config_data.get('health_check_timeout', 300),
        traffic_switch_delay=config_data.get('traffic_switch_delay', 30)
    )
    
    # Create deployment manager
    manager = create_blue_green_manager(str(config_path))
    
    logger.info(f"🚀 Starting deployment of version {deployment_config.version}")
    
    # Execute deployment
    result = await manager.deploy(deployment_config)
    
    if result['success']:
        logger.info("✅ Deployment completed successfully!")
        logger.info(f"Environment: {result['environment']}")
        logger.info(f"Version: {result['version']}")
        logger.info("ℹ️ Call 'deploy.py switch' to activate the new deployment")
        return 0
    else:
        logger.error(f"❌ Deployment failed: {result['error']}")
        return 1


async def switch_command(args):
    """Execute traffic switch command"""
    config_path = Path(args.config)
    manager = create_blue_green_manager(str(config_path))
    
    logger.info("🔄 Switching traffic to new deployment...")
    
    result = await manager.switch_traffic()
    
    if result['success']:
        logger.info("✅ Traffic switched successfully!")
        logger.info(f"New active: {result['new_active']}")
        logger.info(f"Previous active: {result.get('previous_active', 'None')}")
        return 0
    else:
        logger.error(f"❌ Traffic switch failed: {result['error']}")
        return 1


async def rollback_command(args):
    """Execute rollback command"""
    config_path = Path(args.config)
    manager = create_blue_green_manager(str(config_path))
    
    logger.info("⏪ Rolling back deployment...")
    
    result = await manager.rollback()
    
    if result['success']:
        logger.info("✅ Rollback completed successfully!")
        logger.info(f"Rolled back to: {result['rollback_environment']}")
        return 0
    else:
        logger.error(f"❌ Rollback failed: {result['error']}")
        return 1


async def status_command(args):
    """Show deployment status"""
    config_path = Path(args.config)
    manager = create_blue_green_manager(str(config_path))
    
    status = await manager.get_status()
    
    print("📊 A2A Network Deployment Status")
    print("=" * 40)
    print(f"Current Active Environment: {status['current_active']}")
    print(f"Last Updated: {status['last_updated']}")
    print()
    
    for env_name, env_state in status['environments'].items():
        print(f"Environment: {env_name.upper()}")
        if env_state:
            print(f"  Status: {env_state['status']}")
            print(f"  Version: {env_state['version']}")
            print(f"  Deployed: {env_state['deployed_at']}")
            print(f"  Services: {len(env_state['services'])}")
            
            if env_state.get('health_checks'):
                print("  Health Checks:")
                for service, healthy in env_state['health_checks'].items():
                    status_icon = "✅" if healthy else "❌"
                    print(f"    {status_icon} {service}")
            
            if env_state.get('test_results'):
                print("  Test Results:")
                for test, passed in env_state['test_results'].items():
                    status_icon = "✅" if passed else "❌"
                    print(f"    {status_icon} {test}")
        else:
            print("  Status: Not deployed")
        print()
    
    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Blue/Green Deployment Manager for A2A Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deploy.py deploy                 # Deploy new version
  python deploy.py switch                 # Switch traffic to new deployment
  python deploy.py status                 # Show deployment status
  python deploy.py rollback               # Rollback to previous version
  
  python deploy.py deploy --config deployment_config.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        default='deployment_config.yaml',
        help='Path to deployment configuration file (default: deployment_config.yaml)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy new version')
    deploy_parser.set_defaults(func=deploy_command)
    
    # Switch command  
    switch_parser = subparsers.add_parser('switch', help='Switch traffic to new deployment')
    switch_parser.set_defaults(func=switch_command)
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback to previous version')
    rollback_parser.set_defaults(func=rollback_command)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show deployment status')
    status_parser.set_defaults(func=status_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        exit_code = asyncio.run(args.func(args))
        return exit_code
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())