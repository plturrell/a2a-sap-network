#!/usr/bin/env python3
"""
Deployment Coordinator for A2A Network Integration
Automates deployment and configuration of a2a_network contracts for finsight_cib
"""

import os
import json
import subprocess
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import time
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class NetworkConfig:
    """Network configuration for deployment"""
    name: str
    rpc_url: str
    chain_id: int
    private_key: str
    gas_price: Optional[str] = None
    gas_limit: Optional[int] = None

@dataclass
class DeploymentResult:
    """Result of contract deployment"""
    success: bool
    contracts: Dict[str, str]  # contract_name -> address
    transaction_hashes: Dict[str, str]  # contract_name -> tx_hash
    gas_used: Dict[str, int]  # contract_name -> gas_used
    deployment_time: float
    network: str
    block_number: Optional[int] = None
    error: Optional[str] = None

class A2ADeploymentCoordinator:
    """
    Coordinates deployment of a2a_network contracts for finsight_cib integration
    """
    
    def __init__(self, a2a_network_path: str = None, anvil_accounts: List[str] = None):
        # Find a2a_network project
        self.a2a_network_path = Path(a2a_network_path or self._find_a2a_network_path())
        
        if not self.a2a_network_path.exists():
            raise ValueError(f"A2A Network project not found at: {self.a2a_network_path}")
        
        # Default Anvil accounts - ONLY for local testing
        # In production, use environment variables or secure key management
        if anvil_accounts:
            self.anvil_accounts = anvil_accounts
        else:
            # Try to get from environment first
            env_keys = os.getenv("ANVIL_PRIVATE_KEYS", "").split(",")
            if env_keys and env_keys[0]:
                self.anvil_accounts = [key.strip() for key in env_keys]
                logger.info("Using private keys from ANVIL_PRIVATE_KEYS environment variable")
            else:
                # Default Anvil test accounts - NEVER use in production
                self.anvil_accounts = [
                    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",  # Account 0
                    "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",  # Account 1
                    "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",  # Account 2
                ]
                logger.warning("⚠️ Using default Anvil test accounts - DO NOT use in production!")
        
        logger.info(f"Deployment coordinator initialized for: {self.a2a_network_path}")
    
    def _find_a2a_network_path(self) -> str:
        """Find the a2a_network project path"""
        current_dir = Path(__file__).parent
        
        # Search upward for a2a_network
        search_paths = [
            current_dir / "../../../a2a_network",
            current_dir / "../../../../a2a_network", 
            current_dir / "../../../../../a2a_network",
            Path("/Users/apple/projects/a2a_network"),
            Path.home() / "projects/a2a_network"
        ]
        
        for path in search_paths:
            if path.exists() and (path / "foundry.toml").exists():
                logger.info(f"Found a2a_network at: {path}")
                return str(path.resolve())
        
        raise ValueError("Could not find a2a_network project directory")
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        try:
            # Check if contracts.json exists
            json_file = Path(__file__).parent.parent / "config" / "contracts.json"
            
            if json_file.exists():
                with open(json_file) as f:
                    config = json.load(f)
                
                return {
                    "deployed": True,
                    "network": config.get("network"),
                    "contracts": config.get("contracts", {}),
                    "deployment_time": config.get("deployment_time"),
                    "contract_count": len(config.get("contracts", {}))
                }
            else:
                return {
                    "deployed": False,
                    "message": "No deployment found"
                }
                
        except Exception as e:
            return {
                "deployed": False,
                "error": str(e)
            }

# Factory functions for easy use
def create_deployment_coordinator(a2a_network_path: str = None) -> A2ADeploymentCoordinator:
    """Create a deployment coordinator"""
    return A2ADeploymentCoordinator(a2a_network_path)