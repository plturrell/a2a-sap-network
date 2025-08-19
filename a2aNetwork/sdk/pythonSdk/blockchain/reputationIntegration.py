#!/usr/bin/env python3
"""
Reputation Integration for A2A Network
Handles blockchain integration for the reputation system
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from web3 import Web3
from web3.contract import Contract
from web3.exceptions import ContractLogicError, TransactionNotFound
from dataclasses import dataclass
from enum import Enum
import hashlib
import time

from .web3Client import Web3Client
from .eventListener import BlockchainEventListener

logger = logging.getLogger(__name__)


class EndorsementReason(str, Enum):
    EXCELLENT_COLLABORATION = "EXCELLENT_COLLABORATION"
    TIMELY_ASSISTANCE = "TIMELY_ASSISTANCE"
    HIGH_QUALITY_WORK = "HIGH_QUALITY_WORK"
    KNOWLEDGE_SHARING = "KNOWLEDGE_SHARING"
    PROBLEM_SOLVING = "PROBLEM_SOLVING"
    INNOVATION = "INNOVATION"
    MENTORING = "MENTORING"
    RELIABILITY = "RELIABILITY"


@dataclass
class ReputationTransaction:
    transaction_hash: str
    from_agent: str
    to_agent: str
    amount: int
    reason: str
    context_hash: str
    block_number: int
    timestamp: datetime
    gas_used: int
    
    @classmethod
    def from_blockchain_event(cls, event: Dict[str, Any]) -> 'ReputationTransaction':
        return cls(
            transaction_hash=event['transactionHash'].hex(),
            from_agent=event['args']['fromAgent'],
            to_agent=event['args']['toAgent'],
            amount=event['args']['amount'],
            reason=event['args']['reason'],
            context_hash=event['args']['contextHash'].hex(),
            block_number=event['blockNumber'],
            timestamp=datetime.fromtimestamp(event['timestamp']),
            gas_used=event.get('gasUsed', 0)
        )


@dataclass
class AgentReputation:
    agent_address: str
    reputation: int
    status: str
    last_updated: datetime
    total_endorsements_received: int
    total_endorsements_given: int
    trust_id: str


class ReputationIntegration:
    """
    Integration layer for the ReputationExchange smart contract
    """
    
    def __init__(self, web3_client: Web3Client, contract_address: str, contract_abi: List[Dict]):
        self.web3_client = web3_client
        self.contract_address = contract_address
        self.contract_abi = contract_abi
        self.contract: Optional[Contract] = None
        self.event_listener: Optional[BlockchainEventListener] = None
        self.reputation_cache: Dict[str, AgentReputation] = {}
        self._initialize_contract()
    
    def _initialize_contract(self):
        """Initialize the smart contract instance"""
        try:
            self.contract = self.web3_client.get_contract(self.contract_address, self.contract_abi)
            logger.info(f"ReputationExchange contract initialized at {self.contract_address}")
            
            # Initialize event listener
            self.event_listener = BlockchainEventListener(
                self.web3_client.web3,
                self.contract,
                self._handle_reputation_event
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize reputation contract: {e}")
            raise
    
    async def register_agent(self, agent_address: str, name: str, endpoint: str) -> Dict[str, Any]:
        """
        Register an agent in the reputation system
        """
        try:
            logger.info(f"Registering agent {name} ({agent_address}) in reputation system")
            
            # Check if agent is already registered
            agent_info = await self.get_agent_info(agent_address)
            if agent_info and agent_info.get('reputation', 0) > 0:
                logger.info(f"Agent {agent_address} already registered")
                return {
                    'success': True,
                    'message': 'Agent already registered',
                    'agent_address': agent_address,
                    'reputation': agent_info['reputation']
                }
            
            # Register agent on blockchain
            tx_hash = await self.web3_client.send_transaction(
                self.contract.functions.registerAgent(
                    agent_address,
                    name,
                    endpoint
                ),
                gas_limit=200000
            )
            
            # Wait for transaction confirmation
            receipt = await self.web3_client.wait_for_transaction(tx_hash)
            
            if receipt['status'] == 1:
                logger.info(f"Agent {name} registered successfully. TX: {tx_hash.hex()}")
                
                # Cache the new agent
                self.reputation_cache[agent_address] = AgentReputation(
                    agent_address=agent_address,
                    reputation=100,  # Default reputation
                    status='ACTIVE',
                    last_updated=datetime.now(),
                    total_endorsements_received=0,
                    total_endorsements_given=0,
                    trust_id=f"trust_{agent_address[:16]}"
                )
                
                return {
                    'success': True,
                    'transaction_hash': tx_hash.hex(),
                    'agent_address': agent_address,
                    'reputation': 100,
                    'gas_used': receipt['gasUsed']
                }
            else:
                raise Exception("Transaction failed")
                
        except ContractLogicError as e:
            logger.error(f"Contract error registering agent: {e}")
            return {'success': False, 'error': str(e)}
        except Exception as e:
            logger.error(f"Error registering agent: {e}")
            raise
    
    async def endorse_agent(
        self, 
        from_agent: str, 
        to_agent: str, 
        amount: int, 
        reason: EndorsementReason, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Record a peer endorsement on the blockchain
        """
        try:
            logger.info(f"Processing endorsement: {from_agent} -> {to_agent} ({amount} points)")
            
            # Validate inputs
            if amount < 1 or amount > 10:
                raise ValueError("Endorsement amount must be between 1 and 10")
            
            if reason not in EndorsementReason:
                raise ValueError(f"Invalid endorsement reason: {reason}")
            
            # Generate context hash
            context_hash = self._hash_context(context)
            
            # Check if endorsement is allowed (call contract view function)
            try:
                can_endorse = await self._check_endorsement_limits(from_agent, to_agent, amount)
                if not can_endorse:
                    return {
                        'success': False,
                        'error': 'Endorsement limits exceeded or cooldown active'
                    }
            except Exception as e:
                logger.warning(f"Could not check endorsement limits: {e}")
                # Continue anyway - let the contract enforce limits
            
            # Send endorsement transaction
            tx_hash = await self.web3_client.send_transaction(
                self.contract.functions.endorsePeer(
                    to_agent,
                    amount,
                    reason.value,
                    Web3.keccak(text=context_hash)
                ),
                sender=from_agent,
                gas_limit=300000
            )
            
            # Wait for confirmation
            receipt = await self.web3_client.wait_for_transaction(tx_hash)
            
            if receipt['status'] == 1:
                logger.info(f"Endorsement recorded successfully. TX: {tx_hash.hex()}")
                
                # Clear cache for updated agents
                self.reputation_cache.pop(to_agent, None)
                self.reputation_cache.pop(from_agent, None)
                
                return {
                    'success': True,
                    'transaction_hash': tx_hash.hex(),
                    'from_agent': from_agent,
                    'to_agent': to_agent,
                    'amount': amount,
                    'reason': reason.value,
                    'gas_used': receipt['gasUsed']
                }
            else:
                raise Exception("Transaction failed")
                
        except ContractLogicError as e:
            error_msg = str(e)
            logger.error(f"Contract error in endorsement: {error_msg}")
            return {'success': False, 'error': error_msg}
        except Exception as e:
            logger.error(f"Error processing endorsement: {e}")
            raise
    
    async def get_agent_info(self, agent_address: str) -> Optional[Dict[str, Any]]:
        """
        Get agent information from the blockchain
        """
        try:
            # Check cache first
            if agent_address in self.reputation_cache:
                cached = self.reputation_cache[agent_address]
                if datetime.now() - cached.last_updated < timedelta(minutes=5):
                    return {
                        'agent_address': cached.agent_address,
                        'reputation': cached.reputation,
                        'status': cached.status,
                        'last_updated': cached.last_updated.isoformat(),
                        'trust_id': cached.trust_id
                    }
            
            # Fetch from blockchain
            agent_data = await self.contract.functions.getAgent(agent_address).call()
            
            if agent_data[0] == '0x0000000000000000000000000000000000000000':
                return None
            
            agent_info = {
                'agent_address': agent_data[0],
                'reputation': agent_data[1],
                'last_updated': datetime.fromtimestamp(agent_data[2]).isoformat(),
                'is_active': agent_data[3],
                'name': agent_data[4],
                'endpoint': agent_data[5],
                'status': 'ACTIVE' if agent_data[3] else 'INACTIVE'
            }
            
            # Update cache
            self.reputation_cache[agent_address] = AgentReputation(
                agent_address=agent_data[0],
                reputation=agent_data[1],
                status='ACTIVE' if agent_data[3] else 'INACTIVE',
                last_updated=datetime.fromtimestamp(agent_data[2]),
                total_endorsements_received=0,  # Would need additional contract calls
                total_endorsements_given=0,
                trust_id=f"trust_{agent_address[:16]}"
            )
            
            return agent_info
            
        except Exception as e:
            logger.error(f"Error getting agent info: {e}")
            return None
    
    async def get_reputation_history(
        self, 
        agent_address: str, 
        from_block: int = 0,
        to_block: str = 'latest'
    ) -> List[ReputationTransaction]:
        """
        Get reputation change history for an agent
        """
        try:
            # Get endorsement events
            endorsement_filter = self.contract.events.EndorsementCreated.createFilter(
                fromBlock=from_block,
                toBlock=to_block,
                argument_filters={'toAgent': agent_address}
            )
            
            reputation_filter = self.contract.events.ReputationChanged.createFilter(
                fromBlock=from_block,
                toBlock=to_block,
                argument_filters={'agentAddress': agent_address}
            )
            
            # Fetch events
            endorsement_events = await endorsement_filter.get_all_entries()
            reputation_events = await reputation_filter.get_all_entries()
            
            # Convert to ReputationTransaction objects
            transactions = []
            
            for event in endorsement_events:
                # Add timestamp from block
                block = await self.web3_client.web3.eth.get_block(event['blockNumber'])
                event['timestamp'] = block['timestamp']
                
                transaction = ReputationTransaction.from_blockchain_event(event)
                transactions.append(transaction)
            
            # Sort by timestamp
            transactions.sort(key=lambda x: x.timestamp, reverse=True)
            
            return transactions
            
        except Exception as e:
            logger.error(f"Error getting reputation history: {e}")
            return []
    
    async def get_endorsement_limits(self, agent_address: str) -> Dict[str, Any]:
        """
        Get current endorsement limits for an agent
        """
        try:
            agent_info = await self.get_agent_info(agent_address)
            if not agent_info:
                return {'error': 'Agent not found'}
            
            reputation = agent_info['reputation']
            max_endorsement = await self.contract.functions.getMaxEndorsementAmount(agent_address).call()
            
            # Calculate remaining daily limit (this would require tracking daily usage)
            daily_limit = await self.contract.functions.DAILY_ENDORSEMENT_LIMIT().call()
            
            return {
                'max_endorsement_amount': max_endorsement,
                'daily_limit': daily_limit,
                'weekly_peer_limit': await self.contract.functions.WEEKLY_PEER_LIMIT().call(),
                'reciprocal_cooldown_hours': (await self.contract.functions.RECIPROCAL_COOLDOWN().call()) / 3600,
                'current_reputation': reputation
            }
            
        except Exception as e:
            logger.error(f"Error getting endorsement limits: {e}")
            return {'error': str(e)}
    
    async def start_event_monitoring(self):
        """
        Start monitoring blockchain events for reputation changes
        """
        if self.event_listener:
            await self.event_listener.start_monitoring()
            logger.info("Started reputation event monitoring")
    
    async def stop_event_monitoring(self):
        """
        Stop monitoring blockchain events
        """
        if self.event_listener:
            await self.event_listener.stop_monitoring()
            logger.info("Stopped reputation event monitoring")
    
    def _handle_reputation_event(self, event: Dict[str, Any]):
        """
        Handle reputation-related blockchain events
        """
        try:
            event_name = event['event']
            
            if event_name == 'ReputationChanged':
                agent_address = event['args']['agentAddress']
                old_rep = event['args']['oldReputation']
                new_rep = event['args']['newReputation']
                reason = event['args']['reason']
                
                logger.info(f"Reputation changed: {agent_address} {old_rep} -> {new_rep} ({reason})")
                
                # Clear cache to force refresh
                self.reputation_cache.pop(agent_address, None)
                
            elif event_name == 'EndorsementCreated':
                from_agent = event['args']['fromAgent']
                to_agent = event['args']['toAgent']
                amount = event['args']['amount']
                reason = event['args']['reason']
                
                logger.info(f"Endorsement created: {from_agent} -> {to_agent} ({amount} points, {reason})")
                
                # Clear cache for both agents
                self.reputation_cache.pop(from_agent, None)
                self.reputation_cache.pop(to_agent, None)
                
            elif event_name == 'ReputationMilestone':
                agent_address = event['args']['agentAddress']
                milestone = event['args']['milestone']
                badge = event['args']['badge']
                
                logger.info(f"Reputation milestone reached: {agent_address} achieved {badge} badge ({milestone} points)")
                
        except Exception as e:
            logger.error(f"Error handling reputation event: {e}")
    
    async def _check_endorsement_limits(self, from_agent: str, to_agent: str, amount: int) -> bool:
        """
        Check if an endorsement is allowed based on contract limits
        """
        try:
            # Check daily limit
            daily_ok = await self.contract.functions.checkDailyLimit(from_agent, amount).call()
            if not daily_ok:
                return False
            
            # Check weekly peer limit
            weekly_ok = await self.contract.functions.checkWeeklyPeerLimit(from_agent, to_agent, amount).call()
            if not weekly_ok:
                return False
            
            # Check reciprocal cooldown
            reciprocal_ok = not await self.contract.functions.hasRecentReciprocal(from_agent, to_agent).call()
            if not reciprocal_ok:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking endorsement limits: {e}")
            return False
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """
        Generate a hash for endorsement context
        """
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.sha256(context_str.encode()).hexdigest()


async def main():
    """
    Example usage of the ReputationIntegration
    """
    # Load configuration
    with open('../config/blockchain-reputation.json', 'r') as f:
        config = json.load(f)
    
    # Load contract ABI
    with open('../artifacts/contracts/ReputationExchange.sol/ReputationExchange.json', 'r') as f:
        contract_artifact = json.load(f)
        contract_abi = contract_artifact['abi']
    
    # Initialize Web3 client
    web3_client = Web3Client("http://localhost:8545")
    await web3_client.initialize()
    
    # Initialize reputation integration
    reputation = ReputationIntegration(
        web3_client,
        config['blockchain']['reputationExchange']['address'],
        contract_abi
    )
    
    # Example: Register agents
    agent1 = "0x1234567890123456789012345678901234567890"
    agent2 = "0x0987654321098765432109876543210987654321"
    
    result1 = await reputation.register_agent(agent1, "Agent Alpha", "http://localhost:8001")
    result2 = await reputation.register_agent(agent2, "Agent Beta", "http://localhost:8002")
    
    print(f"Agent registration results: {result1}, {result2}")
    
    # Example: Endorse an agent
    endorsement_result = await reputation.endorse_agent(
        from_agent=agent1,
        to_agent=agent2,
        amount=5,
        reason=EndorsementReason.EXCELLENT_COLLABORATION,
        context={
            "task_id": "task_123",
            "description": "Great collaboration on data processing task",
            "timestamp": datetime.now().isoformat()
        }
    )
    
    print(f"Endorsement result: {endorsement_result}")
    
    # Example: Get agent info
    agent_info = await reputation.get_agent_info(agent2)
    print(f"Agent info: {agent_info}")
    
    # Example: Get reputation history
    history = await reputation.get_reputation_history(agent2)
    print(f"Reputation history: {[t.__dict__ for t in history]}")


if __name__ == "__main__":
    asyncio.run(main())