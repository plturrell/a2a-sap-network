"""
Agent Management Service

Provides agent registration, management, and discovery functionality for A2A Network.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from web3 import Web3
from eth_account.datastructures import SignedTransaction

from ..utils.errors import A2AError, ErrorCode
from ..utils.validation import validate_address, validate_agent_params
from ..utils.formatting import format_agent_data, parse_agent_capabilities

logger = logging.getLogger(__name__)

class AgentManager:
    """
    Agent management service for A2A Network.
    
    Handles agent registration, updates, searches, and status management.
    """
    
    def __init__(self, client):
        """
        Initialize AgentManager.
        
        Args:
            client: A2A client instance
        """
        self.client = client

    async def register(self, params: Dict[str, Any]) -> Dict[str, str]:
        """
        Register a new agent on the network.
        
        Args:
            params (Dict[str, Any]): Agent registration parameters
                - name (str): Agent name
                - description (str): Agent description  
                - endpoint (str): Agent endpoint URL
                - capabilities (Dict[str, bool]): Agent capabilities
                - metadata (str, optional): Additional metadata
                
        Returns:
            Dict[str, str]: Registration result with transaction hash and agent ID
            
        Raises:
            A2AError: If registration fails
        """
        # Validate parameters
        validation = validate_agent_params(params)
        if not validation['is_valid']:
            raise A2AError(
                ErrorCode.INVALID_PARAMS, 
                ', '.join(validation.get('errors', []))
            )

        try:
            contract = self.client.get_contract('AgentRegistry')
            account = self.client.get_account()
            
            if not account:
                raise A2AError(ErrorCode.NO_SIGNER, 'Account required for registration')

            # Prepare capabilities array
            capabilities_array = [
                {'name': key, 'enabled': value, 'metadata': ''}
                for key, value in params['capabilities'].items()
            ]

            # Get registration fee
            registration_fee = await self._call_contract_method(
                contract.functions.getRegistrationFee()
            )

            # Build transaction
            tx_data = contract.functions.registerAgent(
                params['name'],
                params['description'],
                params['endpoint'],
                capabilities_array,
                params.get('metadata', '{}')
            ).buildTransaction({
                'from': account.address,
                'value': registration_fee,
                'gas': 500000,  # Conservative gas limit
                'gasPrice': self.client.get_web3().eth.gas_price,
                'nonce': self.client.get_web3().eth.get_transaction_count(account.address)
            })

            # Sign and send transaction
            signed_tx = self.client.get_web3().eth.account.sign_transaction(
                tx_data, 
                account.privateKey
            )
            tx_hash = self.client.get_web3().eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for receipt
            receipt = self.client.get_web3().eth.wait_for_transaction_receipt(tx_hash)
            
            # Extract agent ID from logs
            agent_id = self._extract_agent_id_from_receipt(receipt)
            if not agent_id:
                raise A2AError(ErrorCode.REGISTRATION_FAILED, 'Failed to get agent ID from registration')

            logger.info(f"Agent registered successfully with ID: {agent_id}")

            return {
                'transaction_hash': tx_hash.hex(),
                'agent_id': str(agent_id)
            }

        except Exception as e:
            if isinstance(e, A2AError):
                raise e
            raise A2AError(ErrorCode.REGISTRATION_FAILED, str(e))

    async def update(self, agent_id: str, params: Dict[str, Any]) -> Dict[str, str]:
        """
        Update agent information.
        
        Args:
            agent_id (str): Agent ID to update
            params (Dict[str, Any]): Update parameters
            
        Returns:
            Dict[str, str]: Transaction result
            
        Raises:
            A2AError: If update fails
        """
        try:
            contract = self.client.get_contract('AgentRegistry')
            account = self.client.get_account()
            
            if not account:
                raise A2AError(ErrorCode.NO_SIGNER, 'Account required for update')

            # Verify agent ownership
            agent = await self.get_agent(agent_id)
            if agent['owner'].lower() != account.address.lower():
                raise A2AError(ErrorCode.UNAUTHORIZED, 'Not authorized to update this agent')

            # Build transaction
            tx_data = contract.functions.updateAgent(
                agent_id,
                params.get('name', agent['name']),
                params.get('description', agent['description']),
                params.get('endpoint', agent['endpoint']),
                params.get('metadata', agent['metadata'])
            ).buildTransaction({
                'from': account.address,
                'gas': 200000,
                'gasPrice': self.client.get_web3().eth.gas_price,
                'nonce': self.client.get_web3().eth.get_transaction_count(account.address)
            })

            # Sign and send transaction
            signed_tx = self.client.get_web3().eth.account.sign_transaction(
                tx_data,
                account.privateKey
            )
            tx_hash = self.client.get_web3().eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for confirmation
            self.client.get_web3().eth.wait_for_transaction_receipt(tx_hash)
            
            logger.info(f"Agent {agent_id} updated successfully")

            return {'transaction_hash': tx_hash.hex()}

        except Exception as e:
            if isinstance(e, A2AError):
                raise e
            raise A2AError(ErrorCode.UPDATE_FAILED, str(e))

    async def get_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Get agent information by ID.
        
        Args:
            agent_id (str): Agent ID
            
        Returns:
            Dict[str, Any]: Agent information
            
        Raises:
            A2AError: If fetch fails
        """
        try:
            contract = self.client.get_contract('AgentRegistry')
            result = await self._call_contract_method(
                contract.functions.getAgent(agent_id)
            )

            return format_agent_data(agent_id, result)

        except Exception as e:
            raise A2AError(ErrorCode.FETCH_FAILED, f"Failed to fetch agent: {e}")

    async def get_agent_profile(self, agent_id: str) -> Dict[str, Any]:
        """
        Get agent profile with reputation and performance metrics.
        
        Args:
            agent_id (str): Agent ID
            
        Returns:
            Dict[str, Any]: Agent profile with additional metrics
            
        Raises:
            A2AError: If fetch fails
        """
        try:
            # Get basic agent info and reputation in parallel
            agent, reputation = await asyncio.gather(
                self.get_agent(agent_id),
                self.client.reputation.get_reputation(agent_id)
            )

            # Get detailed profile from AI agent matcher
            matcher_contract = self.client.get_contract('AIAgentMatcher')
            profile_data = await self._call_contract_method(
                matcher_contract.functions.getAIAgentProfile(agent['owner'])
            )

            # Combine all data
            profile = {
                **agent,
                'reputation': {
                    'score': reputation['score'],
                    'rank': reputation['rank'],
                    'total_tasks': profile_data[4],  # totalTasksCompleted
                    'success_rate': profile_data[2] / 100,  # taskSuccessRate
                    'avg_response_time': profile_data[3],  # avgResponseTime
                    'total_earnings': Web3.fromWei(profile_data[5], 'ether')  # totalEarnings
                },
                'performance': {
                    'quality': profile_data[7][0],
                    'speed': profile_data[7][1],
                    'reliability': profile_data[7][2],
                    'innovation': profile_data[7][3]
                },
                'skills': [
                    Web3.toText(tag).replace('\x00', '') 
                    for tag in profile_data[6]  # skillTags
                ]
            }

            return profile

        except Exception as e:
            raise A2AError(ErrorCode.FETCH_FAILED, f"Failed to fetch agent profile: {e}")

    async def search_agents(self, criteria: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Search agents by criteria.
        
        Args:
            criteria (Dict[str, Any], optional): Search criteria
                - skills (List[str]): Required skills
                - min_reputation (int): Minimum reputation
                - max_response_time (int): Maximum response time
                - region (str): Geographic region
                - limit (int): Result limit
                - offset (int): Result offset
                
        Returns:
            Dict[str, Any]: Search results with agents and total count
            
        Raises:
            A2AError: If search fails
        """
        if criteria is None:
            criteria = {}

        try:
            matcher_contract = self.client.get_contract('AIAgentMatcher')
            
            if criteria.get('skills'):
                # Convert skills to bytes32
                skill_bytes = [
                    Web3.keccak(text=skill)[:32] 
                    for skill in criteria['skills']
                ]

                result = await self._call_contract_method(
                    matcher_contract.functions.getTopAgentsBySkills(
                        skill_bytes,
                        criteria.get('limit', 10)
                    )
                )

                agents = []
                registry_contract = self.client.get_contract('AgentRegistry')
                
                for i, address in enumerate(result[0]):  # agents array
                    try:
                        # Find agent ID by owner address
                        agent_ids = await self._call_contract_method(
                            registry_contract.functions.getAgentsByOwner(address)
                        )
                        
                        if not agent_ids:
                            continue
                            
                        agent = await self.get_agent(str(agent_ids[0]))
                        agent['match_score'] = result[1][i]  # scores array
                        agents.append(agent)
                        
                    except Exception:
                        continue

                return {
                    'agents': agents,
                    'total': len(agents)
                }

            # Fallback to general search
            return await self.get_all_agents(
                criteria.get('limit', 50), 
                criteria.get('offset', 0)
            )

        except Exception as e:
            raise A2AError(ErrorCode.SEARCH_FAILED, f"Failed to search agents: {e}")

    async def get_all_agents(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """
        Get all registered agents (paginated).
        
        Args:
            limit (int): Maximum number of agents to return
            offset (int): Number of agents to skip
            
        Returns:
            Dict[str, Any]: Paginated agents list
            
        Raises:
            A2AError: If fetch fails
        """
        try:
            contract = self.client.get_contract('AgentRegistry')
            total_agents = await self._call_contract_method(
                contract.functions.getTotalAgents()
            )
            
            agent_ids = list(range(offset, min(offset + limit, total_agents)))
            
            # Fetch agents in parallel
            agents = await asyncio.gather(*[
                self._get_agent_safe(str(agent_id)) 
                for agent_id in agent_ids
            ])
            
            # Filter out None results
            valid_agents = [agent for agent in agents if agent is not None]

            return {
                'agents': valid_agents,
                'total': total_agents
            }

        except Exception as e:
            raise A2AError(ErrorCode.FETCH_FAILED, f"Failed to fetch agents: {e}")

    async def get_agents_by_owner(self, owner_address: str) -> List[Dict[str, Any]]:
        """
        Get agents owned by address.
        
        Args:
            owner_address (str): Owner address
            
        Returns:
            List[Dict[str, Any]]: List of owned agents
            
        Raises:
            A2AError: If address invalid or fetch fails
        """
        if not validate_address(owner_address):
            raise A2AError(ErrorCode.INVALID_ADDRESS, 'Invalid owner address')

        try:
            contract = self.client.get_contract('AgentRegistry')
            agent_ids = await self._call_contract_method(
                contract.functions.getAgentsByOwner(owner_address)
            )

            # Fetch agents in parallel
            agents = await asyncio.gather(*[
                self._get_agent_safe(str(agent_id)) 
                for agent_id in agent_ids
            ])
            
            # Filter out None results
            return [agent for agent in agents if agent is not None]

        except Exception as e:
            raise A2AError(ErrorCode.FETCH_FAILED, f"Failed to fetch owner agents: {e}")

    async def set_status(self, agent_id: str, is_active: bool) -> Dict[str, str]:
        """
        Update agent status (active/inactive).
        
        Args:
            agent_id (str): Agent ID
            is_active (bool): Active status
            
        Returns:
            Dict[str, str]: Transaction result
            
        Raises:
            A2AError: If status update fails
        """
        try:
            contract = self.client.get_contract('AgentRegistry')
            account = self.client.get_account()
            
            if not account:
                raise A2AError(ErrorCode.NO_SIGNER, 'Account required')

            # Build transaction
            tx_data = contract.functions.setAgentStatus(
                agent_id, 
                is_active
            ).buildTransaction({
                'from': account.address,
                'gas': 100000,
                'gasPrice': self.client.get_web3().eth.gas_price,
                'nonce': self.client.get_web3().eth.get_transaction_count(account.address)
            })

            # Sign and send transaction
            signed_tx = self.client.get_web3().eth.account.sign_transaction(
                tx_data,
                account.privateKey
            )
            tx_hash = self.client.get_web3().eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for confirmation
            self.client.get_web3().eth.wait_for_transaction_receipt(tx_hash)

            return {'transaction_hash': tx_hash.hex()}

        except Exception as e:
            raise A2AError(ErrorCode.STATUS_UPDATE_FAILED, str(e))

    async def deregister(self, agent_id: str) -> Dict[str, str]:
        """
        Deregister agent.
        
        Args:
            agent_id (str): Agent ID to deregister
            
        Returns:
            Dict[str, str]: Transaction result
            
        Raises:
            A2AError: If deregistration fails
        """
        try:
            contract = self.client.get_contract('AgentRegistry')
            account = self.client.get_account()
            
            if not account:
                raise A2AError(ErrorCode.NO_SIGNER, 'Account required')

            # Verify ownership
            agent = await self.get_agent(agent_id)
            if agent['owner'].lower() != account.address.lower():
                raise A2AError(ErrorCode.UNAUTHORIZED, 'Not authorized to deregister this agent')

            # Build transaction
            tx_data = contract.functions.deregisterAgent(
                agent_id
            ).buildTransaction({
                'from': account.address,
                'gas': 150000,
                'gasPrice': self.client.get_web3().eth.gas_price,
                'nonce': self.client.get_web3().eth.get_transaction_count(account.address)
            })

            # Sign and send transaction
            signed_tx = self.client.get_web3().eth.account.sign_transaction(
                tx_data,
                account.privateKey
            )
            tx_hash = self.client.get_web3().eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for confirmation
            self.client.get_web3().eth.wait_for_transaction_receipt(tx_hash)

            logger.info(f"Agent {agent_id} deregistered successfully")

            return {'transaction_hash': tx_hash.hex()}

        except Exception as e:
            if isinstance(e, A2AError):
                raise e
            raise A2AError(ErrorCode.DEREGISTRATION_FAILED, str(e))

    async def get_statistics(self, agent_id: str) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Args:
            agent_id (str): Agent ID
            
        Returns:
            Dict[str, Any]: Agent statistics
            
        Raises:
            A2AError: If fetch fails
        """
        try:
            agent = await self.get_agent(agent_id)
            
            matcher_contract = self.client.get_contract('AIAgentMatcher')
            profile_data = await self._call_contract_method(
                matcher_contract.functions.getAIAgentProfile(agent['owner'])
            )

            return {
                'total_messages': agent['message_count'],
                'successful_tasks': profile_data[4],  # totalTasksCompleted
                'failed_tasks': 0,  # Calculate from success rate if needed
                'avg_response_time': profile_data[3],  # avgResponseTime
                'uptime': 99.5,  # Calculate from last active
                'earnings': Web3.fromWei(profile_data[5], 'ether')  # totalEarnings
            }

        except Exception as e:
            raise A2AError(ErrorCode.FETCH_FAILED, f"Failed to fetch statistics: {e}")

    async def subscribe_to_events(self, callback: callable) -> str:
        """
        Subscribe to agent events.
        
        Args:
            callback (callable): Event callback function
            
        Returns:
            str: Subscription ID
        """
        return await self.client.subscribe_to_events('AgentRegistry', '*', callback)

    async def subscribe_to_agent(self, agent_id: str, callback: callable) -> str:
        """
        Subscribe to specific agent events.
        
        Args:
            agent_id (str): Agent ID to monitor
            callback (callable): Event callback function
            
        Returns:
            str: Subscription ID
        """
        async def filtered_callback(event):
            if hasattr(event, 'args') and hasattr(event.args, 'agentId'):
                if str(event.args.agentId) == agent_id:
                    await callback({
                        'type': 'AgentUpdated',
                        'agent_id': agent_id,
                        'data': event
                    })

        return await self.client.subscribe_to_events(
            'AgentRegistry', 
            'AgentUpdated', 
            filtered_callback
        )

    async def estimate_registration_gas(self, params: Dict[str, Any]) -> int:
        """
        Estimate gas for agent registration.
        
        Args:
            params (Dict[str, Any]): Registration parameters
            
        Returns:
            int: Estimated gas amount
            
        Raises:
            A2AError: If estimation fails
        """
        try:
            contract = self.client.get_contract('AgentRegistry')
            account = self.client.get_account()
            
            if not account:
                raise A2AError(ErrorCode.NO_SIGNER, 'Account required for estimation')

            # Prepare capabilities array
            capabilities_array = [
                {'name': key, 'enabled': value, 'metadata': ''}
                for key, value in params['capabilities'].items()
            ]

            # Get registration fee
            registration_fee = await self._call_contract_method(
                contract.functions.getRegistrationFee()
            )

            # Estimate gas
            gas_estimate = contract.functions.registerAgent(
                params['name'],
                params['description'],
                params['endpoint'],
                capabilities_array,
                params.get('metadata', '{}')
            ).estimateGas({
                'from': account.address,
                'value': registration_fee
            })

            return gas_estimate

        except Exception as e:
            raise A2AError(ErrorCode.ESTIMATION_FAILED, f"Failed to estimate gas: {e}")

    # Private helper methods

    async def _call_contract_method(self, method):
        """Call a contract method asynchronously."""
        return method.call()

    async def _get_agent_safe(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Safely get agent, returning None if not found."""
        try:
            return await self.get_agent(agent_id)
        except Exception:
            return None

    def _extract_agent_id_from_receipt(self, receipt) -> Optional[str]:
        """Extract agent ID from transaction receipt."""
        try:
            contract = self.client.get_contract('AgentRegistry')
            logs = contract.events.AgentRegistered().processReceipt(receipt)
            if logs:
                return str(logs[0]['args']['agentId'])
        except Exception as e:
            logger.warning(f"Failed to extract agent ID from receipt: {e}")
        return None