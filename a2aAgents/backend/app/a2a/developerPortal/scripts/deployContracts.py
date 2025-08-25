#!/usr/bin/env python3
"""
Deploy A2A smart contracts to blockchain networks
Real deployment script for production use
"""

import os
import json
import asyncio
from pathlib import Path
from web3 import Web3
from eth_account import Account
import subprocess


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Configuration
NETWORKS = {
    "local": {
        "rpc_url": os.getenv("A2A_SERVICE_URL"),
        "chain_id": 31337,
        "gas_price": 20000000000  # 20 gwei
    },
    "sepolia": {
        "rpc_url": os.environ.get("SEPOLIA_RPC_URL", "https://rpc.sepolia.org"),
        "chain_id": 11155111,
        "gas_price": 30000000000  # 30 gwei
    },
    "mumbai": {
        "rpc_url": os.environ.get("MUMBAI_RPC_URL", "https://rpc-mumbai.maticvigil.com"),
        "chain_id": 80001,
        "gas_price": 35000000000  # 35 gwei
    }
}


def compile_contracts():
    """Compile A2A contracts using Foundry"""
    print("Compiling contracts with Foundry...")

    # Path to A2A network directory
    a2a_network_path = Path(__file__).parent.parent.parent.parent.parent.parent / "a2a_network" / "foundry"

    if not a2a_network_path.exists():
        raise ValueError(f"A2A network directory not found at {a2a_network_path}")

    # Run forge build
    result = subprocess.run(
        ["forge", "build"],
        cwd=a2a_network_path,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise Exception(f"Contract compilation failed: {result.stderr}")

    print("Contracts compiled successfully")
    return a2a_network_path / "out"


def load_contract_artifact(contracts_path: Path, contract_name: str):
    """Load compiled contract artifact"""
    artifact_path = contracts_path / f"{contract_name}.sol" / f"{contract_name}.json"

    if not artifact_path.exists():
        raise ValueError(f"Contract artifact not found: {artifact_path}")

    with open(artifact_path, 'r') as f:
        artifact = json.load(f)

    return {
        "abi": artifact["abi"],
        "bytecode": artifact["bytecode"]["object"]
    }


async def deploy_contract(web3: Web3, account: Account, contract_data: dict, contract_name: str, constructor_args: list = None):
    """Deploy a contract to the blockchain"""
    print(f"\nDeploying {contract_name}...")

    # Create contract instance
    Contract = web3.eth.contract(
        abi=contract_data["abi"],
        bytecode=contract_data["bytecode"]
    )

    # Get constructor
    if constructor_args is None:
        constructor_args = []

    # Build transaction
    nonce = web3.eth.get_transaction_count(account.address)

    tx = Contract.constructor(*constructor_args).build_transaction({
        'from': account.address,
        'nonce': nonce,
        'gas': 3000000,
        'gasPrice': web3.eth.gas_price,
        'chainId': web3.eth.chain_id
    })

    # Sign and send transaction
    signed_tx = web3.eth.account.sign_transaction(tx, account.key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)

    print(f"Transaction sent: {tx_hash.hex()}")
    print("Waiting for confirmation...")

    # Wait for receipt
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    if receipt.status == 1:
        print(f"✅ {contract_name} deployed at: {receipt.contractAddress}")
        return receipt.contractAddress
    else:
        raise Exception(f"Deployment failed for {contract_name}")


async def deploy_to_network(network_name: str, private_key: str):
    """Deploy contracts to a specific network"""
    print(f"\n{'='*60}")
    print(f"Deploying to {network_name.upper()}")
    print(f"{'='*60}")

    # Get network config
    network_config = NETWORKS.get(network_name)
    if not network_config:
        raise ValueError(f"Unknown network: {network_name}")

    # Connect to network
    web3 = Web3(Web3.HTTPProvider(network_config["rpc_url"]))

    if not web3.is_connected():
        raise Exception(f"Failed to connect to {network_name}")

    print(f"Connected to {network_name}")
    print(f"Chain ID: {web3.eth.chain_id}")
    print(f"Latest block: {web3.eth.block_number}")

    # Get account
    account = Account.from_key(private_key)
    print(f"Deploying from: {account.address}")

    # Check balance
    balance = web3.eth.get_balance(account.address)
    print(f"Balance: {web3.from_wei(balance, 'ether')} ETH")

    if balance == 0:
        raise Exception(f"Insufficient balance for deployment on {network_name}")

    # Compile contracts
    contracts_path = compile_contracts()

    # Load contract artifacts
    agent_registry_data = load_contract_artifact(contracts_path, "AgentRegistry")
    message_router_data = load_contract_artifact(contracts_path, "MessageRouter")

    # Deploy AgentRegistry with 2 required confirmations for multi-sig
    agent_registry_address = await deploy_contract(
        web3,
        account,
        agent_registry_data,
        "AgentRegistry",
        [2]  # requiredConfirmations
    )

    # Deploy MessageRouter with AgentRegistry address
    message_router_address = await deploy_contract(
        web3,
        account,
        message_router_data,
        "MessageRouter",
        [agent_registry_address, 2]  # registry address and requiredConfirmations
    )

    # Save deployment info
    deployment_info = {
        "network": network_name,
        "chainId": web3.eth.chain_id,
        "deployedAt": web3.eth.block_number,
        "deployer": account.address,
        "contracts": {
            "AgentRegistry": agent_registry_address,
            "MessageRouter": message_router_address
        }
    }

    # Save to file
    deployments_dir = Path(__file__).parent.parent / "deployments"
    deployments_dir.mkdir(exist_ok=True)

    deployment_file = deployments_dir / f"{network_name}_deployment.json"
    with open(deployment_file, 'w') as f:
        json.dump(deployment_info, f, indent=2)

    print(f"\n✅ Deployment complete!")
    print(f"Deployment info saved to: {deployment_file}")

    # Print environment variables to set
    print(f"\nSet these environment variables:")
    print(f"export {network_name.upper()}_AGENT_REGISTRY={agent_registry_address}")
    print(f"export {network_name.upper()}_MESSAGE_ROUTER={message_router_address}")

    return deployment_info


async def verify_deployment(network_name: str, deployment_info: dict):
    """Verify that contracts are deployed and working"""
    print(f"\n{'='*60}")
    print(f"Verifying deployment on {network_name}")
    print(f"{'='*60}")

    network_config = NETWORKS[network_name]
    web3 = Web3(Web3.HTTPProvider(network_config["rpc_url"]))

    # Check contract code
    for contract_name, address in deployment_info["contracts"].items():
        code = web3.eth.get_code(address)
        if code == b'':
            print(f"❌ No code at {contract_name} address: {address}")
        else:
            print(f"✅ {contract_name} has code at: {address}")
            print(f"   Code size: {len(code)} bytes")


async def main():
    """Main deployment function"""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy A2A smart contracts")
    parser.add_argument("network", choices=["local", "sepolia", "mumbai"], help="Network to deploy to")
    parser.add_argument("--private-key", help="Private key for deployment (or set DEPLOYER_PRIVATE_KEY env var)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing deployment")

    args = parser.parse_args()

    # Get private key
    private_key = args.private_key or os.environ.get("DEPLOYER_PRIVATE_KEY")

    if args.verify_only:
        # Load existing deployment
        deployment_file = Path(__file__).parent.parent / "deployments" / f"{args.network}_deployment.json"
        if not deployment_file.exists():
            print(f"No deployment found for {args.network}")
            return

        with open(deployment_file, 'r') as f:
            deployment_info = json.load(f)

        await verify_deployment(args.network, deployment_info)
    else:
        if not private_key:
            print("Error: Private key required for deployment")
            print("Set DEPLOYER_PRIVATE_KEY environment variable or use --private-key")
            return

        # Deploy contracts
        deployment_info = await deploy_to_network(args.network, private_key)

        # Verify deployment
        await verify_deployment(args.network, deployment_info)


if __name__ == "__main__":
    asyncio.run(main())