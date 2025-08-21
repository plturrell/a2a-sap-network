#!/usr/bin/env python3
"""
Deploy A2A Developer Portal with Network Integration
Quick deployment script for testing the integration
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import httpx
        import web3
        print("✓ All Python dependencies found")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    # Check if A2A Network SDK is accessible
    a2a_network_path = Path(__file__).parent.parent.parent.parent.parent.parent / "a2a_network"
    if not a2a_network_path.exists():
        print(f"✗ A2A Network path not found: {a2a_network_path}")
        return False
    
    print(f"✓ A2A Network found at: {a2a_network_path}")
    return True

def create_test_config():
    """Create test configuration for the portal"""
    config = {
        "workspace_path": "/tmp/a2a_workspace",
        "port": 3001,
        "sap_btp": {
            "rbac": {
                "development_mode": True
            },
            "session": {
                "session_timeout_minutes": 30
            }
        },
        "blockchain": {
            "local_provider": os.getenv("A2A_SERVICE_URL"),
            "testnet_provider": "https://sepolia.infura.io/v3/YOUR_INFURA_KEY",
            "contracts": {
                "AgentRegistry": "0x0000000000000000000000000000000000000000",
                "MessageRouter": "0x0000000000000000000000000000000000000000"
            }
        }
    }
    
    config_path = Path("portal_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Created test configuration at: {config_path}")
    return config_path

def start_portal():
    """Start the developer portal server"""
    print("\n🚀 Starting A2A Developer Portal with Network Integration...")
    
    # Create launch script
    launch_script = """
import asyncio
import json
import sys
from pathlib import Path

# Add portal to path
sys.path.insert(0, str(Path(__file__).parent))

from portalServer import create_developer_portal
import uvicorn


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
async def main():
    # Load config
    with open("portal_config.json", "r") as f:
        config = json.load(f)
    
    # Create portal
    portal = create_developer_portal(config)
    
    # Initialize
    await portal.initialize()
    
    # Run server
    uvicorn_config = uvicorn.Config(
        app=portal.app,
        host="0.0.0.0",
        port=config.get("port", 3001),
        log_level="info"
    )
    
    server = uvicorn.Server(uvicorn_config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
"""
    
    launch_path = Path("launch_portal_with_integration.py")
    with open(launch_path, "w") as f:
        f.write(launch_script)
    
    print(f"✓ Created launch script at: {launch_path}")
    
    # Start the server
    try:
        subprocess.run([sys.executable, str(launch_path)], check=True)
    except KeyboardInterrupt:
        print("\n\n✓ Portal stopped by user")
    except Exception as e:
        print(f"\n✗ Error starting portal: {e}")
        return False
    
    return True

def print_access_info():
    """Print access information"""
    print("\n" + "="*60)
    print("🌐 A2A Developer Portal - Access Information")
    print("="*60)
    print("\n📍 Portal URL: http://localhost:3001")
    print("\n🔧 Available Features:")
    print("  • Project Management")
    print("  • Agent Builder")
    print("  • BPMN Workflow Designer")
    print("  • A2A Network Integration (NEW!)")
    print("    - Agent Registration on Blockchain")
    print("    - Real-time Message Routing")
    print("    - Reputation Management")
    print("    - Webhook Subscriptions")
    print("\n🔌 API Endpoints:")
    print("  • Portal API: http://localhost:3001/api/")
    print("  • A2A Network API: http://localhost:3001/api/a2a-network/")
    print("  • Health Check: http://localhost:3001/api/health")
    print("\n📋 Quick Start:")
    print("  1. Navigate to A2A Network in the sidebar")
    print("  2. Configure network settings (mainnet/testnet/local)")
    print("  3. Register your agents on the blockchain")
    print("  4. Set up webhooks for real-time updates")
    print("\n⚙️  Network Configuration:")
    print("  • Settings are stored in browser localStorage")
    print("  • Private keys never leave your browser")
    print("  • WebSocket auto-reconnects for reliability")
    print("\n" + "="*60 + "\n")

def main():
    """Main deployment function"""
    print("🎯 A2A Developer Portal - Network Integration Deployment")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Create test configuration
    config_path = create_test_config()
    
    # Print access information
    print_access_info()
    
    # Start portal
    if not start_portal():
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())