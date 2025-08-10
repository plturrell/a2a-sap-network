"""
A2A Network Service Managers

Service manager modules for handling different aspects of A2A Network functionality.
"""

from .agentManager import AgentManager

# Service manager implementations
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class MessageManager:
    """Message management service for A2A Network"""
    
    def __init__(self, client):
        self.client = client
        self.message_cache = {}
        self.pending_messages = []
    
    async def send_message(self, recipient_id: str, message_type: str, content: Any) -> Dict[str, Any]:
        """Send a message to another agent"""
        try:
            message = {
                "id": f"msg_{datetime.now().timestamp()}",
                "sender_id": self.client.agent_id if hasattr(self.client, 'agent_id') else 'unknown',
                "recipient_id": recipient_id,
                "message_type": message_type,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "status": "pending"
            }
            
            # Call contract method if available
            if hasattr(self.client, 'contracts') and 'MessageRegistry' in self.client.contracts:
                tx_receipt = await self.client.contracts['MessageRegistry'].send_message(
                    recipient_id, message_type, content
                )
                message["tx_hash"] = tx_receipt.transactionHash.hex()
                message["status"] = "sent"
            
            self.message_cache[message["id"]] = message
            return message
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise
    
    async def get_messages(self, filter_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve messages based on filter parameters"""
        try:
            messages = []
            
            # Get from contract if available
            if hasattr(self.client, 'contracts') and 'MessageRegistry' in self.client.contracts:
                raw_messages = await self.client.contracts['MessageRegistry'].get_messages(
                    self.client.agent_id if hasattr(self.client, 'agent_id') else None
                )
                messages = [self._format_message(msg) for msg in raw_messages]
            else:
                # Return cached messages
                messages = list(self.message_cache.values())
            
            # Apply filters
            if filter_params:
                if 'message_type' in filter_params:
                    messages = [m for m in messages if m.get('message_type') == filter_params['message_type']]
                if 'status' in filter_params:
                    messages = [m for m in messages if m.get('status') == filter_params['status']]
                if 'sender_id' in filter_params:
                    messages = [m for m in messages if m.get('sender_id') == filter_params['sender_id']]
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get messages: {e}")
            return []
    
    def _format_message(self, raw_message: Any) -> Dict[str, Any]:
        """Format raw message data from contract"""
        return {
            "id": raw_message.get('id', 'unknown'),
            "sender_id": raw_message.get('sender', ''),
            "recipient_id": raw_message.get('recipient', ''),
            "message_type": raw_message.get('messageType', ''),
            "content": raw_message.get('content', ''),
            "timestamp": raw_message.get('timestamp', ''),
            "status": raw_message.get('status', 'unknown')
        }


class TokenManager:
    """Token management service for A2A Network"""
    
    def __init__(self, client):
        self.client = client
        self.token_balances = {}
        self.transaction_history = []
    
    async def get_balance(self, address: Optional[str] = None) -> Dict[str, Any]:
        """Get token balance for an address"""
        try:
            target_address = address or (self.client.account.address if hasattr(self.client, 'account') else None)
            if not target_address:
                raise ValueError("No address provided")
            
            balance = 0
            
            # Get from contract if available
            if hasattr(self.client, 'contracts') and 'A2AToken' in self.client.contracts:
                balance = await self.client.contracts['A2AToken'].balance_of(target_address)
            else:
                balance = self.token_balances.get(target_address, 0)
            
            return {
                "address": target_address,
                "balance": balance,
                "symbol": "A2A",
                "decimals": 18,
                "formatted_balance": balance / (10 ** 18)
            }
            
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            raise
    
    async def transfer(self, recipient: str, amount: int) -> Dict[str, Any]:
        """Transfer tokens to another address"""
        try:
            tx_data = {
                "from": self.client.account.address if hasattr(self.client, 'account') else 'unknown',
                "to": recipient,
                "amount": amount,
                "timestamp": datetime.now().isoformat(),
                "status": "pending"
            }
            
            # Execute transfer if contract available
            if hasattr(self.client, 'contracts') and 'A2AToken' in self.client.contracts:
                tx_receipt = await self.client.contracts['A2AToken'].transfer(recipient, amount)
                tx_data["tx_hash"] = tx_receipt.transactionHash.hex()
                tx_data["status"] = "confirmed"
                tx_data["block_number"] = tx_receipt.blockNumber
            
            self.transaction_history.append(tx_data)
            return tx_data
            
        except Exception as e:
            logger.error(f"Failed to transfer tokens: {e}")
            raise
    
    async def get_transaction_history(self, address: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get transaction history for an address"""
        target_address = address or (self.client.account.address if hasattr(self.client, 'account') else None)
        
        if not target_address:
            return self.transaction_history
        
        return [
            tx for tx in self.transaction_history 
            if tx.get('from') == target_address or tx.get('to') == target_address
        ]


class GovernanceManager:
    """Governance management service for A2A Network"""
    
    def __init__(self, client):
        self.client = client
        self.proposals = {}
        self.votes = {}
    
    async def create_proposal(self, title: str, description: str, proposal_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new governance proposal"""
        try:
            proposal = {
                "id": f"prop_{datetime.now().timestamp()}",
                "title": title,
                "description": description,
                "type": proposal_type,
                "params": params,
                "proposer": self.client.account.address if hasattr(self.client, 'account') else 'unknown',
                "created_at": datetime.now().isoformat(),
                "status": "active",
                "votes_for": 0,
                "votes_against": 0,
                "votes_abstain": 0
            }
            
            # Submit to contract if available
            if hasattr(self.client, 'contracts') and 'Governance' in self.client.contracts:
                tx_receipt = await self.client.contracts['Governance'].create_proposal(
                    title, description, proposal_type, params
                )
                proposal["tx_hash"] = tx_receipt.transactionHash.hex()
                proposal["on_chain"] = True
            else:
                proposal["on_chain"] = False
            
            self.proposals[proposal["id"]] = proposal
            return proposal
            
        except Exception as e:
            logger.error(f"Failed to create proposal: {e}")
            raise
    
    async def vote(self, proposal_id: str, vote_type: str) -> Dict[str, Any]:
        """Vote on a proposal"""
        if vote_type not in ['for', 'against', 'abstain']:
            raise ValueError("Invalid vote type. Must be 'for', 'against', or 'abstain'")
        
        try:
            vote_data = {
                "proposal_id": proposal_id,
                "voter": self.client.account.address if hasattr(self.client, 'account') else 'unknown',
                "vote_type": vote_type,
                "timestamp": datetime.now().isoformat()
            }
            
            # Submit vote to contract if available
            if hasattr(self.client, 'contracts') and 'Governance' in self.client.contracts:
                tx_receipt = await self.client.contracts['Governance'].vote(proposal_id, vote_type)
                vote_data["tx_hash"] = tx_receipt.transactionHash.hex()
            
            # Update local proposal counts
            if proposal_id in self.proposals:
                if vote_type == 'for':
                    self.proposals[proposal_id]['votes_for'] += 1
                elif vote_type == 'against':
                    self.proposals[proposal_id]['votes_against'] += 1
                else:
                    self.proposals[proposal_id]['votes_abstain'] += 1
            
            self.votes[f"{proposal_id}_{vote_data['voter']}"] = vote_data
            return vote_data
            
        except Exception as e:
            logger.error(f"Failed to vote: {e}")
            raise
    
    async def get_proposals(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get governance proposals"""
        proposals = list(self.proposals.values())
        
        if status:
            proposals = [p for p in proposals if p.get('status') == status]
        
        return sorted(proposals, key=lambda x: x.get('created_at', ''), reverse=True)


class ScalabilityManager:
    """Scalability management service for A2A Network"""
    
    def __init__(self, client):
        self.client = client
        self.shard_info = {}
        self.layer2_bridges = {}
        self.performance_metrics = {
            "transactions_per_second": 0,
            "average_block_time": 0,
            "pending_transactions": 0
        }
    
    async def get_shard_info(self) -> Dict[str, Any]:
        """Get information about network shards"""
        try:
            shard_data = {
                "total_shards": 1,
                "active_shards": 1,
                "shard_distribution": {},
                "current_shard": 0
            }
            
            # Get from contract if available
            if hasattr(self.client, 'contracts') and 'ShardManager' in self.client.contracts:
                shard_data = await self.client.contracts['ShardManager'].get_shard_info()
            
            self.shard_info = shard_data
            return shard_data
            
        except Exception as e:
            logger.error(f"Failed to get shard info: {e}")
            return self.shard_info
    
    async def bridge_to_layer2(self, layer2_name: str, amount: int) -> Dict[str, Any]:
        """Bridge tokens to a Layer 2 solution"""
        try:
            bridge_tx = {
                "id": f"bridge_{datetime.now().timestamp()}",
                "layer2": layer2_name,
                "amount": amount,
                "from_address": self.client.account.address if hasattr(self.client, 'account') else 'unknown',
                "timestamp": datetime.now().isoformat(),
                "status": "pending"
            }
            
            # Execute bridge if contract available
            if hasattr(self.client, 'contracts') and f'{layer2_name}Bridge' in self.client.contracts:
                tx_receipt = await self.client.contracts[f'{layer2_name}Bridge'].deposit(amount)
                bridge_tx["tx_hash"] = tx_receipt.transactionHash.hex()
                bridge_tx["status"] = "confirmed"
            
            self.layer2_bridges[bridge_tx["id"]] = bridge_tx
            return bridge_tx
            
        except Exception as e:
            logger.error(f"Failed to bridge to layer2: {e}")
            raise
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get network performance metrics"""
        try:
            # Update metrics from network if possible
            if hasattr(self.client, 'web3'):
                latest_block = await self.client.web3.eth.get_block('latest')
                pending_tx_count = await self.client.web3.eth.get_block_transaction_count('pending')
                
                self.performance_metrics.update({
                    "latest_block": latest_block.number,
                    "gas_price": await self.client.web3.eth.gas_price,
                    "pending_transactions": pending_tx_count
                })
            
            return self.performance_metrics
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return self.performance_metrics
    
    async def optimize_gas_usage(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize gas usage for a transaction"""
        optimized_data = transaction_data.copy()
        
        try:
            if hasattr(self.client, 'web3'):
                # Estimate gas
                estimated_gas = await self.client.web3.eth.estimate_gas(transaction_data)
                
                # Get current gas prices
                gas_price = await self.client.web3.eth.gas_price
                
                optimized_data.update({
                    "gas": int(estimated_gas * 1.1),  # Add 10% buffer
                    "gasPrice": gas_price,
                    "maxFeePerGas": int(gas_price * 1.5),
                    "maxPriorityFeePerGas": int(gas_price * 0.1)
                })
            
            return optimized_data
            
        except Exception as e:
            logger.error(f"Failed to optimize gas usage: {e}")
            return optimized_data


class ReputationManager:
    """Reputation management service for A2A Network"""
    
    def __init__(self, client):
        self.client = client
        self.reputation_scores = {}
        self.reputation_history = []
        self.endorsements = {}
    
    async def get_reputation(self, agent_id: str) -> Dict[str, Any]:
        """Get reputation score for an agent"""
        try:
            reputation_data = {
                "agent_id": agent_id,
                "score": 0,
                "level": "neutral",
                "total_interactions": 0,
                "positive_feedback": 0,
                "negative_feedback": 0,
                "endorsements": []
            }
            
            # Get from contract if available
            if hasattr(self.client, 'contracts') and 'ReputationRegistry' in self.client.contracts:
                raw_data = await self.client.contracts['ReputationRegistry'].get_reputation(agent_id)
                reputation_data.update(self._format_reputation_data(raw_data))
            else:
                # Use cached data
                reputation_data.update(self.reputation_scores.get(agent_id, {}))
            
            # Calculate level based on score
            score = reputation_data.get('score', 0)
            if score >= 90:
                reputation_data['level'] = 'excellent'
            elif score >= 70:
                reputation_data['level'] = 'good'
            elif score >= 50:
                reputation_data['level'] = 'neutral'
            elif score >= 30:
                reputation_data['level'] = 'poor'
            else:
                reputation_data['level'] = 'very_poor'
            
            return reputation_data
            
        except Exception as e:
            logger.error(f"Failed to get reputation: {e}")
            return {"agent_id": agent_id, "score": 0, "level": "unknown"}
    
    async def submit_feedback(self, agent_id: str, rating: int, comment: Optional[str] = None) -> Dict[str, Any]:
        """Submit feedback for an agent"""
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")
        
        try:
            feedback = {
                "id": f"feedback_{datetime.now().timestamp()}",
                "agent_id": agent_id,
                "rater": self.client.account.address if hasattr(self.client, 'account') else 'unknown',
                "rating": rating,
                "comment": comment,
                "timestamp": datetime.now().isoformat()
            }
            
            # Submit to contract if available
            if hasattr(self.client, 'contracts') and 'ReputationRegistry' in self.client.contracts:
                tx_receipt = await self.client.contracts['ReputationRegistry'].submit_feedback(
                    agent_id, rating, comment or ""
                )
                feedback["tx_hash"] = tx_receipt.transactionHash.hex()
            
            # Update local reputation
            if agent_id not in self.reputation_scores:
                self.reputation_scores[agent_id] = {
                    "score": 50,
                    "total_interactions": 0,
                    "positive_feedback": 0,
                    "negative_feedback": 0
                }
            
            self.reputation_scores[agent_id]["total_interactions"] += 1
            if rating >= 4:
                self.reputation_scores[agent_id]["positive_feedback"] += 1
            elif rating <= 2:
                self.reputation_scores[agent_id]["negative_feedback"] += 1
            
            # Recalculate score
            rep_data = self.reputation_scores[agent_id]
            if rep_data["total_interactions"] > 0:
                positive_ratio = rep_data["positive_feedback"] / rep_data["total_interactions"]
                rep_data["score"] = int(positive_ratio * 100)
            
            self.reputation_history.append(feedback)
            return feedback
            
        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            raise
    
    async def endorse_agent(self, agent_id: str, skill: str) -> Dict[str, Any]:
        """Endorse an agent for a specific skill"""
        try:
            endorsement = {
                "id": f"endorse_{datetime.now().timestamp()}",
                "agent_id": agent_id,
                "endorser": self.client.account.address if hasattr(self.client, 'account') else 'unknown',
                "skill": skill,
                "timestamp": datetime.now().isoformat()
            }
            
            # Submit to contract if available
            if hasattr(self.client, 'contracts') and 'ReputationRegistry' in self.client.contracts:
                tx_receipt = await self.client.contracts['ReputationRegistry'].endorse(agent_id, skill)
                endorsement["tx_hash"] = tx_receipt.transactionHash.hex()
            
            # Store endorsement
            if agent_id not in self.endorsements:
                self.endorsements[agent_id] = []
            self.endorsements[agent_id].append(endorsement)
            
            return endorsement
            
        except Exception as e:
            logger.error(f"Failed to endorse agent: {e}")
            raise
    
    def _format_reputation_data(self, raw_data: Any) -> Dict[str, Any]:
        """Format raw reputation data from contract"""
        return {
            "score": raw_data.get('score', 0),
            "total_interactions": raw_data.get('totalInteractions', 0),
            "positive_feedback": raw_data.get('positiveFeedback', 0),
            "negative_feedback": raw_data.get('negativeFeedback', 0),
            "endorsements": raw_data.get('endorsements', [])
        }

__all__ = [
    'AgentManager',
    'MessageManager',
    'TokenManager', 
    'GovernanceManager',
    'ScalabilityManager',
    'ReputationManager'
]