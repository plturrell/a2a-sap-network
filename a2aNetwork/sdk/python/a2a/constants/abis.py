"""
A2A Network Constants - ABIs

Contract ABI definitions.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Standard ERC20 ABI for token contracts
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "totalSupply",
        "outputs": [{"name": "", "type": "uint256"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "payable": False,
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "from", "type": "address"},
            {"indexed": True, "name": "to", "type": "address"},
            {"indexed": False, "name": "value", "type": "uint256"}
        ],
        "name": "Transfer",
        "type": "event"
    }
]

# A2A Network specific contract ABIs
AGENT_REGISTRY_ABI = [
    {
        "inputs": [
            {"name": "_name", "type": "string"},
            {"name": "_description", "type": "string"},
            {"name": "_endpoint", "type": "string"},
            {"name": "_capabilities", "type": "string[]"},
            {"name": "_metadata", "type": "string"}
        ],
        "name": "registerAgent",
        "outputs": [{"name": "", "type": "bytes32"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "_agentId", "type": "bytes32"},
            {"name": "_endpoint", "type": "string"}
        ],
        "name": "updateAgentEndpoint",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"name": "_agentId", "type": "bytes32"}],
        "name": "getAgent",
        "outputs": [
            {"name": "name", "type": "string"},
            {"name": "description", "type": "string"},
            {"name": "endpoint", "type": "string"},
            {"name": "owner", "type": "address"},
            {"name": "messageCount", "type": "uint256"},
            {"name": "metadata", "type": "string"},
            {"name": "isActive", "type": "bool"},
            {"name": "registrationTime", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getAllAgents",
        "outputs": [{"name": "", "type": "bytes32[]"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "agentId", "type": "bytes32"},
            {"indexed": True, "name": "owner", "type": "address"},
            {"indexed": False, "name": "name", "type": "string"}
        ],
        "name": "AgentRegistered",
        "type": "event"
    }
]

MESSAGE_REGISTRY_ABI = [
    {
        "inputs": [
            {"name": "_recipient", "type": "bytes32"},
            {"name": "_messageType", "type": "string"},
            {"name": "_content", "type": "string"}
        ],
        "name": "sendMessage",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"name": "_agentId", "type": "bytes32"}],
        "name": "getMessages",
        "outputs": [
            {
                "components": [
                    {"name": "id", "type": "uint256"},
                    {"name": "sender", "type": "bytes32"},
                    {"name": "recipient", "type": "bytes32"},
                    {"name": "messageType", "type": "string"},
                    {"name": "content", "type": "string"},
                    {"name": "timestamp", "type": "uint256"},
                    {"name": "status", "type": "uint8"}
                ],
                "name": "",
                "type": "tuple[]"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# Contract name to ABI mapping
CONTRACT_ABIS = {
    "AgentRegistry": AGENT_REGISTRY_ABI,
    "MessageRegistry": MESSAGE_REGISTRY_ABI,
    "A2AToken": ERC20_ABI,
    "ERC20": ERC20_ABI
}

# Cache for loaded ABIs
_abi_cache: Dict[str, List[Dict[str, Any]]] = {}

def get_contract_abi(contract_name: str) -> List[Dict[str, Any]]:
    """
    Get contract ABI by name.
    
    First checks predefined ABIs, then attempts to load from file system.
    
    Args:
        contract_name: Name of the contract
        
    Returns:
        List of ABI entries for the contract
    """
    # Check cache first
    if contract_name in _abi_cache:
        return _abi_cache[contract_name]
    
    # Check predefined ABIs
    if contract_name in CONTRACT_ABIS:
        _abi_cache[contract_name] = CONTRACT_ABIS[contract_name]
        return CONTRACT_ABIS[contract_name]
    
    # Try to load from file system
    abi = _load_abi_from_file(contract_name)
    if abi:
        _abi_cache[contract_name] = abi
        return abi
    
    # Return minimal fallback ABI
    fallback_abi = _get_fallback_abi()
    _abi_cache[contract_name] = fallback_abi
    return fallback_abi

def _load_abi_from_file(contract_name: str) -> Optional[List[Dict[str, Any]]]:
    """
    Attempt to load ABI from file system.
    
    Looks for ABI files in common locations:
    - ./abis/{contract_name}.json
    - ./contracts/abis/{contract_name}.json
    - ../contracts/out/{contract_name}.sol/{contract_name}.json (Foundry format)
    """
    search_paths = [
        Path(f"./abis/{contract_name}.json"),
        Path(f"./contracts/abis/{contract_name}.json"),
        Path(f"../contracts/out/{contract_name}.sol/{contract_name}.json"),
        Path(f"./build/contracts/{contract_name}.json"),  # Truffle format
        Path(f"./artifacts/contracts/{contract_name}.sol/{contract_name}.json")  # Hardhat format
    ]
    
    for path in search_paths:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    
                # Handle different formats
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    # Foundry/Hardhat format
                    if 'abi' in data:
                        return data['abi']
                    # Truffle format
                    elif 'abi' in data:
                        return data['abi']
                        
            except (json.JSONDecodeError, IOError) as e:
                # Log error but continue searching
                print(f"Warning: Failed to load ABI from {path}: {e}")
                continue
    
    return None

def _get_fallback_abi() -> List[Dict[str, Any]]:
    """Get minimal fallback ABI with basic functions"""
    return [
        {
            "stateMutability": "nonpayable",
            "type": "fallback"
        },
        {
            "inputs": [],
            "name": "owner",
            "outputs": [{"name": "", "type": "address"}],
            "stateMutability": "view",
            "type": "function"
        }
    ]

def add_contract_abi(contract_name: str, abi: List[Dict[str, Any]]) -> None:
    """
    Add or update a contract ABI in the registry.
    
    Args:
        contract_name: Name of the contract
        abi: ABI definition
    """
    CONTRACT_ABIS[contract_name] = abi
    _abi_cache[contract_name] = abi

def clear_abi_cache() -> None:
    """Clear the ABI cache"""
    _abi_cache.clear()

__all__ = [
    'get_contract_abi',
    'add_contract_abi',
    'clear_abi_cache',
    'ERC20_ABI',
    'AGENT_REGISTRY_ABI',
    'MESSAGE_REGISTRY_ABI'
]