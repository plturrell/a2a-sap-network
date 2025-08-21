#!/usr/bin/env python3
"""
Dynamic Contract Configuration System
Loads contract addresses from deployment artifacts and environment
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ContractInfo:
    """Contract information with address and ABI"""
    address: str
    abi: list
    name: str
    network: str
    deployed_at: Optional[str] = None
    
class ContractConfigManager:
    """Manages contract configuration from deployment artifacts"""
    
    def __init__(self, network: str = None, artifacts_path: str = None):
        self.network = network or os.getenv('A2A_NETWORK', 'localhost')
        self.artifacts_path = artifacts_path or self._find_artifacts_path()
        self.contracts: Dict[str, ContractInfo] = {}
        self._load_contracts()
    
    def _find_artifacts_path(self) -> str:
        """Find the deployment artifacts path"""
        # Look for a2a_network deployment artifacts
        possible_paths = [
            # Relative to current project
            "../a2a_network/broadcast/DeployUpgradeable.s.sol",
            "../a2a_network/broadcast/Deploy.s.sol", 
            "../../a2a_network/broadcast/DeployUpgradeable.s.sol",
            "../../a2a_network/broadcast/Deploy.s.sol",
            # Environment variable
            os.getenv('A2A_ARTIFACTS_PATH', ''),
            # Default foundry paths
            "../artifacts/broadcast",
            "./artifacts/broadcast"
        ]
        
        for path in possible_paths:
            if path and Path(path).exists():
                logger.info(f"Found artifacts path: {path}")
                return str(Path(path).resolve())
        
        logger.warning("No deployment artifacts found, using fallback configuration")
        return ""
    
    def _load_contracts(self):
        """Load contract information from deployment artifacts"""
        if not self.artifacts_path:
            logger.error("No artifacts path configured. Loading from environment variables.")
            self._load_required_contracts()
            return
            
        try:
            # Load from deployment artifacts
            self._load_from_artifacts()
        except Exception as e:
            logger.error(f"Failed to load from artifacts: {e}. Loading from environment variables as fallback.")
            self._load_required_contracts()
    
    def _load_from_artifacts(self):
        """Load contracts from Foundry deployment artifacts"""
        artifacts_dir = Path(self.artifacts_path)
        
        # Look for network-specific deployments
        network_dirs = [
            artifacts_dir / self.network,
            artifacts_dir / "31337",  # Anvil default
            artifacts_dir / "1337",   # Ganache default
        ]
        
        for network_dir in network_dirs:
            if network_dir.exists():
                self._load_network_contracts(network_dir)
                break
        else:
            # Try to find any recent deployment
            for item in artifacts_dir.iterdir():
                if item.is_dir() and item.name.isdigit():
                    self._load_network_contracts(item)
                    break
    
    def _load_network_contracts(self, network_dir: Path):
        """Load contracts from a specific network directory"""
        run_latest = network_dir / "run-latest.json"
        if not run_latest.exists():
            return
            
        with open(run_latest) as f:
            deployment_data = json.load(f)
        
        # Extract contract addresses from deployment
        for tx in deployment_data.get('transactions', []):
            if tx.get('transactionType') == 'CREATE':
                contract_name = tx.get('contractName', '')
                if contract_name in ['AgentRegistryUpgradeable', 'AgentRegistry']:
                    self.contracts['AgentRegistry'] = ContractInfo(
                        address=tx.get('contractAddress', ''),
                        abi=self._load_contract_abi('AgentRegistry'),
                        name='AgentRegistry',
                        network=self.network,
                        deployed_at=tx.get('timestamp')
                    )
                elif contract_name in ['MessageRouterUpgradeable', 'MessageRouter']:
                    self.contracts['MessageRouter'] = ContractInfo(
                        address=tx.get('contractAddress', ''),
                        abi=self._load_contract_abi('MessageRouter'),
                        name='MessageRouter',
                        network=self.network,
                        deployed_at=tx.get('timestamp')
                    )
        
        logger.info(f"Loaded {len(self.contracts)} contracts from deployment artifacts")
    
    def _load_contract_abi(self, contract_name: str) -> list:
        """Load contract ABI from artifacts"""
        try:
            # Check environment variable first
            abi_base_path = os.getenv('A2A_ABI_PATH')
            
            # Look for ABI in various locations
            abi_paths = []
            
            if abi_base_path:
                abi_paths.extend([
                    f"{abi_base_path}/{contract_name}.sol/{contract_name}.json",
                    f"{abi_base_path}/{contract_name}Upgradeable.sol/{contract_name}Upgradeable.json",
                ])
            
            # Fallback paths
            abi_paths.extend([
                f"../a2aNetwork/out/{contract_name}.sol/{contract_name}.json",
                f"../../a2aNetwork/out/{contract_name}.sol/{contract_name}.json",
                f"../a2a_network/out/{contract_name}.sol/{contract_name}.json",
                f"../../a2a_network/out/{contract_name}.sol/{contract_name}.json",
                f"../a2a_network/out/{contract_name}Upgradeable.sol/{contract_name}Upgradeable.json",
                f"../../a2a_network/out/{contract_name}Upgradeable.sol/{contract_name}Upgradeable.json",
            ])
            
            for abi_path in abi_paths:
                if Path(abi_path).exists():
                    with open(abi_path) as f:
                        artifact = json.load(f)
                        return artifact.get('abi', [])
            
            error_msg = f"ABI not found for {contract_name} in any expected location. Contract artifacts must be available for production deployment."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except Exception as e:
            logger.error(f"Failed to load ABI for {contract_name}: {e}")
            raise
    
    def _load_required_contracts(self):
        """Load contract configuration from environment variables - all contracts required for production"""
        logger.info("Loading production contract configuration")
        
        # AgentRegistry - REQUIRED
        registry_address = os.getenv('A2A_AGENT_REGISTRY_ADDRESS')
        if not registry_address:
            raise ValueError("A2A_AGENT_REGISTRY_ADDRESS environment variable is required for production deployment")
            
        self.contracts['AgentRegistry'] = ContractInfo(
            address=registry_address,
            abi=self._load_contract_abi('AgentRegistry'),
            name='AgentRegistry',
            network=self.network
        )
        
        # MessageRouter - REQUIRED
        router_address = os.getenv('A2A_MESSAGE_ROUTER_ADDRESS')
        if not router_address:
            raise ValueError("A2A_MESSAGE_ROUTER_ADDRESS environment variable is required for production deployment")
            
        self.contracts['MessageRouter'] = ContractInfo(
            address=router_address,
            abi=self._load_contract_abi('MessageRouter'),
            name='MessageRouter',
            network=self.network
        )
        
        # ORDRegistry - REQUIRED
        ord_address = os.getenv('A2A_ORD_REGISTRY_ADDRESS')
        if not ord_address:
            raise ValueError("A2A_ORD_REGISTRY_ADDRESS environment variable is required for production deployment")
            
        # In development, use AgentRegistry ABI as fallback for ORDRegistry
        try:
            ord_abi = self._load_contract_abi('ORDRegistry')
        except FileNotFoundError:
            logger.warning("ORDRegistry ABI not found, using AgentRegistry ABI as fallback")
            ord_abi = self._load_contract_abi('AgentRegistry')
            
        self.contracts['ORDRegistry'] = ContractInfo(
            address=ord_address,
            abi=ord_abi,
            name='ORDRegistry',
            network=self.network
        )
    
    def get_contract(self, name: str) -> Optional[ContractInfo]:
        """Get contract information by name"""
        return self.contracts.get(name)
    
    def get_contract_address(self, name: str) -> Optional[str]:
        """Get contract address by name"""
        contract = self.get_contract(name)
        return contract.address if contract else None
    
    def get_contract_abi(self, name: str) -> Optional[list]:
        """Get contract ABI by name"""
        contract = self.get_contract(name)
        return contract.abi if contract else None
    
    def is_contract_available(self, name: str) -> bool:
        """Check if contract is available and has valid address"""
        contract = self.get_contract(name)
        return contract is not None and bool(contract.address) and contract.address != '0x'
    
    def get_all_contracts(self) -> Dict[str, ContractInfo]:
        """Get all loaded contracts"""
        return self.contracts.copy()
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the contract configuration"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'contracts': {}
        }
        
        required_contracts = ['AgentRegistry', 'MessageRouter']
        
        for contract_name in required_contracts:
            contract = self.get_contract(contract_name)
            if not contract:
                validation_result['errors'].append(f"Missing required contract: {contract_name}")
                validation_result['valid'] = False
            elif not contract.address or contract.address == '0x':
                validation_result['errors'].append(f"Invalid address for {contract_name}: {contract.address}")
                validation_result['valid'] = False
            else:
                validation_result['contracts'][contract_name] = {
                    'address': contract.address,
                    'network': contract.network,
                    'abi_loaded': len(contract.abi) > 0
                }
                
                if len(contract.abi) == 0:
                    validation_result['warnings'].append(f"No ABI loaded for {contract_name}")
        
        return validation_result

# Global configuration manager instance
_config_manager: Optional[ContractConfigManager] = None

def get_contract_config() -> ContractConfigManager:
    """Get the global contract configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ContractConfigManager()
    return _config_manager

def reload_contract_config(network: str = None, artifacts_path: str = None):
    """Reload the contract configuration"""
    global _config_manager
    _config_manager = ContractConfigManager(network=network, artifacts_path=artifacts_path)
    return _config_manager

# Convenience functions
def get_agent_registry_address() -> Optional[str]:
    """Get AgentRegistry contract address"""
    return get_contract_config().get_contract_address('AgentRegistry')

def get_message_router_address() -> Optional[str]:
    """Get MessageRouter contract address"""
    return get_contract_config().get_contract_address('MessageRouter')

def get_ord_registry_address() -> Optional[str]:
    """Get ORDRegistry contract address"""
    return get_contract_config().get_contract_address('ORDRegistry')

def validate_contracts() -> bool:
    """Validate that all required contracts are configured"""
    validation = get_contract_config().validate_configuration()
    if not validation['valid']:
        for error in validation['errors']:
            logger.error(f"Contract validation error: {error}")
    for warning in validation['warnings']:
        logger.warning(f"Contract validation warning: {warning}")
    return validation['valid']