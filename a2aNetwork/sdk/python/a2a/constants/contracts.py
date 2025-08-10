"""
A2A Network Constants - Contracts

Contract address and ABI constants.
"""

CONTRACT_ADDRESSES = {
    'localhost': {
        'AGENT_REGISTRY': '0x5FbDB2315678afecb367f032d93F642f64180aa3',
        'MESSAGE_ROUTER': '0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512',
        'A2A_TOKEN': '0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0',
        'TIMELOCK_GOVERNOR': '0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9',
        'LOAD_BALANCER': '0xDc64a140Aa3E981100a9becA4E685f962f0cF6C9',
        'AI_AGENT_MATCHER': '0x5FC8d32690cc91D4c39d9d3abcBD16989F875707'
    }
}

def get_contract_abi(contract_name):
    """Get contract ABI by name"""
    # Placeholder empty ABI - in production this would load from files
    return []

__all__ = ['CONTRACT_ADDRESSES', 'get_contract_abi']