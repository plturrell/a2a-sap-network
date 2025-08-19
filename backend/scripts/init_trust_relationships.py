#!/usr/bin/env python3
"""
A2A Trust System Initialization Script
Initialize trust relationships between agents at startup
"""

import asyncio
import logging
import os
import sys
from typing import Dict, List

# Add the backend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.a2a.core.trustManager import TrustManager
    from app.a2a.core.blockchainQueueManager import get_blockchain_queue_manager
except ImportError as e:
    print(f"Warning: Could not import trust system components: {e}")
    print("Trust system will run in mock mode")
    TrustManager = None
    get_blockchain_queue_manager = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# A2A Agent Registry - Known agents in the system
A2A_AGENTS = [
    {"agent_id": "agent0", "name": "Data Product Agent", "port": 8001, "initial_trust": 0.7},
    {"agent_id": "agent1", "name": "Standardization Agent", "port": 8002, "initial_trust": 0.7},
    {"agent_id": "agent2", "name": "Validation Agent", "port": 8003, "initial_trust": 0.7},
    {"agent_id": "agent3", "name": "Pipeline Agent", "port": 8004, "initial_trust": 0.7},
    {"agent_id": "agent4", "name": "Processing Agent", "port": 8005, "initial_trust": 0.6},
    {"agent_id": "agent5", "name": "Analytics Agent", "port": 8006, "initial_trust": 0.6},
    {"agent_id": "agent6", "name": "Monitoring Agent", "port": 8007, "initial_trust": 0.6},
    {"agent_id": "registry_agent", "name": "Registry Agent", "port": 8000, "initial_trust": 0.9},
    {"agent_id": "reasoning_agent", "name": "Reasoning Agent", "port": 8008, "initial_trust": 0.8},
    {"agent_id": "sql_agent", "name": "SQL Agent", "port": 8009, "initial_trust": 0.8},
    {"agent_id": "agent_manager", "name": "Agent Manager", "port": 8010, "initial_trust": 0.9},
    {"agent_id": "data_manager", "name": "Data Manager", "port": 8011, "initial_trust": 0.8},
    {"agent_id": "catalog_manager", "name": "Catalog Manager", "port": 8012, "initial_trust": 0.8},
    {"agent_id": "calculation_agent", "name": "Calculation Agent", "port": 8013, "initial_trust": 0.7},
    {"agent_id": "agent_builder", "name": "Agent Builder", "port": 8014, "initial_trust": 0.8},
    {"agent_id": "embedding_tuner", "name": "Embedding Fine-Tuner", "port": 8015, "initial_trust": 0.7},
]

# Trust relationships between agents
TRUST_RELATIONSHIPS = [
    # High-trust core system agents
    {"from": "registry_agent", "to": "agent_manager", "trust_level": 0.9},
    {"from": "registry_agent", "to": "data_manager", "trust_level": 0.9},
    {"from": "agent_manager", "to": "registry_agent", "trust_level": 0.9},
    
    # Data processing pipeline trust relationships
    {"from": "agent0", "to": "agent1", "trust_level": 0.8},  # Data Product -> Standardization
    {"from": "agent1", "to": "agent2", "trust_level": 0.8},  # Standardization -> Validation
    {"from": "agent2", "to": "agent3", "trust_level": 0.8},  # Validation -> Pipeline
    {"from": "agent3", "to": "agent4", "trust_level": 0.7},  # Pipeline -> Processing
    {"from": "agent4", "to": "agent5", "trust_level": 0.7},  # Processing -> Analytics
    
    # Management and monitoring relationships
    {"from": "agent_manager", "to": "agent6", "trust_level": 0.8},  # Agent Manager -> Monitoring
    {"from": "agent6", "to": "agent_manager", "trust_level": 0.7},  # Monitoring -> Agent Manager
    
    # AI/ML specialized agents
    {"from": "reasoning_agent", "to": "sql_agent", "trust_level": 0.8},
    {"from": "sql_agent", "to": "data_manager", "trust_level": 0.8},
    {"from": "agent_builder", "to": "embedding_tuner", "trust_level": 0.8},
    
    # Cross-functional trust relationships
    {"from": "catalog_manager", "to": "data_manager", "trust_level": 0.8},
    {"from": "calculation_agent", "to": "reasoning_agent", "trust_level": 0.7},
]

class TrustInitializer:
    """Initialize trust relationships for A2A agents"""
    
    def __init__(self):
        self.trust_manager = None
        self.blockchain_available = False
        self.initialized_agents = set()
        self.established_relationships = set()
        
    async def initialize(self):
        """Initialize the trust system and establish relationships"""
        logger.info("Starting A2A Trust System initialization...")
        
        # Check if trust system components are available
        if TrustManager is None:
            logger.warning("Trust system components not available, running in mock mode")
            return await self._mock_initialization()
            
        try:
            # Initialize trust manager for system
            self.trust_manager = TrustManager()
            await self.trust_manager.initialize()
            
            # Check blockchain availability
            blockchain_url = os.getenv("A2A_BLOCKCHAIN_URL", "http://localhost:8545")
            self.blockchain_available = await self._check_blockchain_availability(blockchain_url)
            
            if self.blockchain_available:
                logger.info("✅ Blockchain available for trust system")
            else:
                logger.warning("⚠️  Blockchain not available, using local trust store")
            
            # Initialize agent trust scores
            await self._initialize_agent_trust_scores()
            
            # Establish trust relationships
            await self._establish_trust_relationships()
            
            # Verify trust system is working
            await self._verify_trust_system()
            
            logger.info("🎉 A2A Trust System initialization completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Trust system initialization failed: {e}")
            return False
            
        return True
    
    async def _check_blockchain_availability(self, blockchain_url: str) -> bool:
        """Check if blockchain is available for trust operations"""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    blockchain_url,
                    json={
                        "jsonrpc": "2.0",
                        "method": "eth_blockNumber",
                        "params": [],
                        "id": 1
                    }
                )
                return response.status_code == 200
                
        except Exception as e:
            logger.debug(f"Blockchain availability check failed: {e}")
            return False
    
    async def _initialize_agent_trust_scores(self):
        """Initialize trust scores for all known agents"""
        logger.info("Initializing agent trust scores...")
        
        for agent_info in A2A_AGENTS:
            agent_id = agent_info["agent_id"]
            initial_trust = agent_info["initial_trust"]
            
            try:
                # Set initial trust score
                await self.trust_manager.set_trust_score(
                    agent_id=agent_id,
                    trust_score=initial_trust,
                    reason=f"Initial trust setup for {agent_info['name']}"
                )
                
                self.initialized_agents.add(agent_id)
                logger.info(f"✅ {agent_info['name']} ({agent_id}) trust initialized: {initial_trust}")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize trust for {agent_id}: {e}")
    
    async def _establish_trust_relationships(self):
        """Establish trust relationships between agents"""
        logger.info("Establishing inter-agent trust relationships...")
        
        for relationship in TRUST_RELATIONSHIPS:
            from_agent = relationship["from"]
            to_agent = relationship["to"]
            trust_level = relationship["trust_level"]
            
            try:
                # Establish bidirectional trust relationship
                await self.trust_manager.establish_trust_relationship(
                    from_agent=from_agent,
                    to_agent=to_agent,
                    trust_level=trust_level
                )
                
                relationship_key = f"{from_agent}->{to_agent}"
                self.established_relationships.add(relationship_key)
                
                logger.info(f"✅ Trust relationship established: {from_agent} -> {to_agent} ({trust_level})")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to establish trust relationship {from_agent}->{to_agent}: {e}")
    
    async def _verify_trust_system(self):
        """Verify that trust system is working correctly"""
        logger.info("Verifying trust system functionality...")
        
        # Test trust score retrieval
        test_agent = "agent0"
        try:
            trust_score = await self.trust_manager.get_trust_score(test_agent)
            logger.info(f"✅ Trust score retrieval test passed: {test_agent} = {trust_score}")
        except Exception as e:
            logger.warning(f"⚠️  Trust score retrieval test failed: {e}")
        
        # Test trust relationship query
        try:
            relationships = await self.trust_manager.get_trust_relationships(test_agent)
            logger.info(f"✅ Trust relationships query test passed: {len(relationships)} relationships found")
        except Exception as e:
            logger.warning(f"⚠️  Trust relationships query test failed: {e}")
    
    async def _mock_initialization(self) -> bool:
        """Mock initialization when trust system is not available"""
        logger.info("Running mock trust initialization...")
        
        # Simulate initialization process
        for agent_info in A2A_AGENTS:
            logger.info(f"✅ Mock trust initialized for {agent_info['name']} ({agent_info['initial_trust']})")
        
        for relationship in TRUST_RELATIONSHIPS:
            logger.info(f"✅ Mock trust relationship: {relationship['from']} -> {relationship['to']}")
        
        logger.info("🎉 Mock trust system initialization completed")
        return True

async def main():
    """Main initialization function"""
    initializer = TrustInitializer()
    
    try:
        success = await initializer.initialize()
        
        if success:
            print("✅ A2A Trust System initialization successful")
            
            # Print summary
            print(f"\n📊 Initialization Summary:")
            print(f"   • Agents initialized: {len(initializer.initialized_agents)}/16")
            print(f"   • Trust relationships: {len(initializer.established_relationships)}")
            print(f"   • Blockchain integration: {'✅ Enabled' if initializer.blockchain_available else '⚠️  Local mode'}")
            
            return 0
        else:
            print("❌ A2A Trust System initialization failed")
            return 1
            
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        print(f"❌ Trust system initialization error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)