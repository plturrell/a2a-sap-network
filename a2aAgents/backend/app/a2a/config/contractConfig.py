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
            self._load_fallback_contracts()
            return
            
        try:
            # Load from deployment artifacts
            self._load_from_artifacts()
        except Exception as e:
            logger.error(f"Failed to load from artifacts: {e}")
            self._load_fallback_contracts()
    
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
            # Get base path for a2aNetwork directory
            current_file = Path(__file__).resolve()
            # Navigate up to find a2aNetwork directory
            project_root = current_file
            for _ in range(10):  # Try up to 10 levels
                if (project_root / 'a2aNetwork').exists():
                    break
                project_root = project_root.parent
            
            # Look for ABI in a2a_network out directory
            abi_paths = [
                # Absolute path from project root
                project_root / 'a2aNetwork' / 'out' / f'{contract_name}.sol' / f'{contract_name}.json',
                project_root / 'a2aNetwork' / 'out' / f'{contract_name}Upgradeable.sol' / f'{contract_name}Upgradeable.json',
                # Fallback absolute path
                Path('/Users/apple/projects/a2a/a2aNetwork/out') / f'{contract_name}.sol' / f'{contract_name}.json',
                # Relative paths as fallback
                Path(f"../a2a_network/out/{contract_name}.sol/{contract_name}.json"),
                Path(f"../../a2a_network/out/{contract_name}.sol/{contract_name}.json"),
                Path(f"../a2a_network/out/{contract_name}Upgradeable.sol/{contract_name}Upgradeable.json"),
                Path(f"../../a2a_network/out/{contract_name}Upgradeable.sol/{contract_name}Upgradeable.json"),
            ]
            
            for abi_path in abi_paths:
                if abi_path.exists():
                    with open(abi_path) as f:
                        artifact = json.load(f)
                        abi = artifact.get('abi', [])
                        if abi:
                            logger.info(f"Loaded ABI for {contract_name} from {abi_path}")
                            return abi
            
            logger.warning(f"ABI not found for {contract_name}, using empty ABI")
            return []
        except Exception as e:
            logger.error(f"Failed to load ABI for {contract_name}: {e}")
            return []
    
    def _load_fallback_contracts(self):
        """Load fallback contract configuration from environment variables"""
        logger.info("Loading fallback contract configuration")
        
        # AgentRegistry
        registry_address = os.getenv('A2A_AGENT_REGISTRY_ADDRESS')
        if registry_address:
            self.contracts['AgentRegistry'] = ContractInfo(
                address=registry_address,
                abi=self._load_contract_abi('AgentRegistry'),
                name='AgentRegistry',
                network=self.network
            )
        
        # MessageRouter
        router_address = os.getenv('A2A_MESSAGE_ROUTER_ADDRESS')
        if router_address:
            self.contracts['MessageRouter'] = ContractInfo(
                address=router_address,
                abi=self._load_contract_abi('MessageRouter'),
                name='MessageRouter',
                network=self.network
            )
        
        # ORDRegistry
        ord_address = os.getenv('A2A_ORD_REGISTRY_ADDRESS')
        if ord_address:
            self.contracts['ORDRegistry'] = ContractInfo(
                address=ord_address,
                abi=self._load_contract_abi('ORDRegistry'),
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
    
    def load_config(self) -> Dict[str, Any]:
        """Load and return the complete contract configuration"""
        config = {
            'network': self.network,
            'artifacts_path': self.artifacts_path,
            'contracts': {}
        }
        
        for name, contract_info in self.contracts.items():
            config['contracts'][name] = {
                'address': contract_info.address,
                'abi': contract_info.abi,
                'name': contract_info.name,
                'network': contract_info.network,
                'deployed_at': contract_info.deployed_at
            }
        
        # Ensure we have the required contracts with fallback values
        if 'AgentRegistry' not in config['contracts']:
            registry_address = os.getenv('A2A_AGENT_REGISTRY_ADDRESS', '0x5FbDB2315678afecb367f032d93F642f64180aa3')
            config['contracts']['AgentRegistry'] = {
                'address': registry_address,
                'abi': [],
                'name': 'AgentRegistry',
                'network': self.network,
                'deployed_at': None
            }
            
        if 'MessageRouter' not in config['contracts']:
            router_address = os.getenv('A2A_MESSAGE_ROUTER_ADDRESS', '0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512')
            config['contracts']['MessageRouter'] = {
                'address': router_address,
                'abi': [],
                'name': 'MessageRouter',
                'network': self.network,
                'deployed_at': None
            }
            
        if 'ORDRegistry' not in config['contracts']:
            ord_address = os.getenv('A2A_ORD_REGISTRY_ADDRESS', '0x5FbDB2315678afecb367f032d93F642f64180aa3')
            config['contracts']['ORDRegistry'] = {
                'address': ord_address,
                'abi': [],
                'name': 'ORDRegistry',
                'network': self.network,
                'deployed_at': None
            }
        
        return config

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