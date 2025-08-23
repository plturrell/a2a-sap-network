#!/usr/bin/env python3
"""
Start Real A2A System - Launch all components with real blockchain integration
No mocks, no simulations - everything real
"""

import asyncio
import json
import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

class A2ASystemStarter:
    """Manages startup of real A2A system components"""
    
    def __init__(self):
        self.processes = {}
        self.project_root = Path(__file__).parent.parent.parent.parent.parent / "a2aNetwork"
        logger.info(f"Project root: {self.project_root}")
        
    async def check_prerequisites(self):
        """Check if prerequisites are met"""
        logger.info("🔍 Checking prerequisites...")
        
        # Check Node.js
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True)
            logger.info(f"   Node.js: {result.stdout.strip()}")
        except:
            logger.error("❌ Node.js not found! Please install Node.js")
            return False
            
        # Check Python
        logger.info(f"   Python: {sys.version.split()[0]}")
        
        # Check npm packages
        try:
            hardhat_path = self.project_root / "node_modules" / ".bin" / "hardhat"
            if not hardhat_path.exists():
                logger.error("❌ Hardhat not installed! Run: npm install")
                return False
            logger.info("   Hardhat: Installed")
        except:
            logger.error("❌ Cannot check Hardhat installation")
            return False
            
        return True
        
    async def start_blockchain(self):
        """Start local blockchain (Hardhat)"""
        logger.info("\n🔗 Starting Local Blockchain...")
        
        try:
            # Start Hardhat node
            cmd = ["npx", "hardhat", "node"]
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes['blockchain'] = process
            
            # Wait for blockchain to start
            logger.info("   Waiting for blockchain to start...")
            time.sleep(5)
            
            # Check if running
            if process.poll() is None:
                logger.info("✅ Blockchain started on http://localhost:8545")
                return True
            else:
                logger.error("❌ Blockchain failed to start")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start blockchain: {e}")
            return False
            
    async def deploy_contracts(self):
        """Deploy smart contracts"""
        logger.info("\n📜 Deploying Smart Contracts...")
        
        try:
            # Deploy contracts
            cmd = ["npx", "hardhat", "run", "scripts/deploy.js", "--network", "localhost"]
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("✅ Contracts deployed successfully")
                
                # Parse deployment output for addresses
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if "AgentRegistry deployed to:" in line:
                        address = line.split(":")[-1].strip()
                        os.environ["A2A_AGENT_REGISTRY_ADDRESS"] = address
                        logger.info(f"   AgentRegistry: {address}")
                    elif "MessageRouter deployed to:" in line:
                        address = line.split(":")[-1].strip()
                        os.environ["A2A_MESSAGE_ROUTER_ADDRESS"] = address
                        logger.info(f"   MessageRouter: {address}")
                        
                return True
            else:
                logger.error(f"❌ Contract deployment failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to deploy contracts: {e}")
            return False
            
    async def start_agent_manager(self):
        """Start AgentManager service"""
        logger.info("\n🤖 Starting AgentManager Service...")
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env.update({
                "A2A_SERVICE_URL": "http://localhost:8010",
                "A2A_SERVICE_HOST": "localhost",
                "A2A_BASE_URL": "http://localhost:8010",
                "A2A_RPC_URL": "http://localhost:8545",
                "AI_ENABLED": "true",
                "BLOCKCHAIN_ENABLED": "true"
            })
            
            # Start AgentManager
            agent_manager_path = Path(__file__).parent.parent.parent / "agentManager" / "agentManager.py"
            
            cmd = [sys.executable, str(agent_manager_path)]
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes['agent_manager'] = process
            
            # Wait for startup
            time.sleep(3)
            
            if process.poll() is None:
                logger.info("✅ AgentManager started on http://localhost:8010")
                return True
            else:
                logger.error("❌ AgentManager failed to start")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start AgentManager: {e}")
            return False
            
    async def start_data_manager(self):
        """Start DataManager agent"""
        logger.info("\n💾 Starting DataManager Agent...")
        
        try:
            env = os.environ.copy()
            
            # Start DataManager
            data_manager_path = Path(__file__).parent.parent.parent / "dataManager" / "dataManager.py"
            
            cmd = [sys.executable, str(data_manager_path)]
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes['data_manager'] = process
            
            # Wait for startup
            time.sleep(3)
            
            if process.poll() is None:
                logger.info("✅ DataManager started")
                return True
            else:
                logger.error("❌ DataManager failed to start")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start DataManager: {e}")
            return False
            
    async def register_initial_agents(self):
        """Register initial agents on blockchain"""
        logger.info("\n📝 Registering Initial Agents...")
        
        try:
            from blockchain_integration import BlockchainIntegration
            
            blockchain = BlockchainIntegration()
            
            # Define initial agents
            initial_agents = [
                {
                    "id": "agent_manager",
                    "name": "Agent Manager",
                    "endpoint": "http://localhost:8010",
                    "capabilities": ["agent_management", "message_tracking", "reputation_calculation", "network_analysis"]
                },
                {
                    "id": "data_manager",
                    "name": "Data Manager",
                    "endpoint": "http://localhost:8011",
                    "capabilities": ["data_storage", "data_retrieval", "data_analysis", "persistence"]
                },
                {
                    "id": "calc_agent",
                    "name": "Calculation Agent",
                    "endpoint": "http://localhost:8012",
                    "capabilities": ["mathematical_computation", "financial_analysis", "calculations"]
                },
                {
                    "id": "chat_agent",
                    "name": "Chat Agent",
                    "endpoint": "http://localhost:8000",
                    "capabilities": ["chat", "ai_reasoning", "message_conversion", "skills_matching"]
                }
            ]
            
            for agent in initial_agents:
                logger.info(f"   Registering {agent['name']}...")
                
                # Check if already registered
                is_registered = await blockchain.is_agent_registered(agent['id'])
                
                if not is_registered:
                    result = await blockchain.register_agent(
                        agent['id'],
                        agent['name'],
                        agent['endpoint'],
                        agent['capabilities']
                    )
                    
                    if result.get('success'):
                        logger.info(f"   ✅ {agent['name']} registered")
                    else:
                        logger.error(f"   ❌ Failed to register {agent['name']}: {result.get('error')}")
                else:
                    logger.info(f"   ℹ️ {agent['name']} already registered")
                    
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register agents: {e}")
            return False
            
    async def verify_system_health(self):
        """Verify all components are healthy"""
        logger.info("\n🏥 Verifying System Health...")
        
        try:
            from blockchain_integration import BlockchainIntegration
            
            blockchain = BlockchainIntegration()
            
            # Check blockchain connection
            if blockchain.web3.is_connected():
                logger.info("   ✅ Blockchain connected")
                
                # Get agent count
                agent_count = await blockchain.get_registered_agent_count()
                logger.info(f"   ✅ Agents registered: {agent_count}")
            else:
                logger.error("   ❌ Blockchain not connected")
                return False
                
            # Check running processes
            for name, process in self.processes.items():
                if process and process.poll() is None:
                    logger.info(f"   ✅ {name} running")
                else:
                    logger.error(f"   ❌ {name} not running")
                    return False
                    
            logger.info("\n✅ System health check passed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False
            
    async def start_system(self):
        """Start the complete A2A system"""
        logger.info("🚀 Starting Real A2A System (No Mocks)")
        
        # Check prerequisites
        if not await self.check_prerequisites():
            return False
            
        # Start blockchain
        if not await self.start_blockchain():
            return False
            
        # Deploy contracts
        if not await self.deploy_contracts():
            return False
            
        # Start AgentManager
        if not await self.start_agent_manager():
            return False
            
        # Start DataManager
        if not await self.start_data_manager():
            return False
            
        # Register agents
        if not await self.register_initial_agents():
            return False
            
        # Verify health
        if not await self.verify_system_health():
            return False
            
        logger.info("\n🎉 Real A2A System Started Successfully!")
        logger.info("\nSystem Components:")
        logger.info("  - Blockchain: http://localhost:8545")
        logger.info("  - AgentManager: http://localhost:8010")
        logger.info("  - DataManager: http://localhost:8011")
        logger.info("\nEnvironment Variables Set:")
        logger.info(f"  - A2A_AGENT_REGISTRY_ADDRESS: {os.getenv('A2A_AGENT_REGISTRY_ADDRESS')}")
        logger.info(f"  - A2A_MESSAGE_ROUTER_ADDRESS: {os.getenv('A2A_MESSAGE_ROUTER_ADDRESS')}")
        
        return True
        
    def shutdown(self):
        """Shutdown all components"""
        logger.info("\n🛑 Shutting down A2A System...")
        
        for name, process in self.processes.items():
            if process and process.poll() is None:
                logger.info(f"   Stopping {name}...")
                process.terminate()
                process.wait(timeout=5)
                
        logger.info("✅ System shutdown complete")

async def main():
    """Main startup function"""
    starter = A2ASystemStarter()
    
    try:
        # Start the system
        success = await starter.start_system()
        
        if success:
            logger.info("\n✅ System is ready for real A2A messaging tests!")
            logger.info("Press Ctrl+C to shutdown the system")
            
            # Keep running
            while True:
                await asyncio.sleep(1)
        else:
            logger.error("\n❌ System startup failed!")
            starter.shutdown()
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Shutdown requested")
        starter.shutdown()
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        starter.shutdown()

if __name__ == "__main__":
    asyncio.run(main())