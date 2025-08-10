export const CONTRACT_ADDRESSES: Record<string, Record<string, string>> = {
  localhost: {
    AGENT_REGISTRY: '0x5FbDB2315678afecb367f032d93F642f64180aa3',
    MESSAGE_ROUTER: '0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512',
    A2A_TOKEN: '0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0',
    TIMELOCK_GOVERNOR: '0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9',
    LOAD_BALANCER: '0x68B1D87F95878fE05B998F19b66F4baba5De1aed',
    AI_AGENT_MATCHER: '0x3Aa5ebB10DC797CAC828524e59A333d0A371443c'
  },
  mainnet: {
    AGENT_REGISTRY: '',
    MESSAGE_ROUTER: '',
    A2A_TOKEN: '',
    TIMELOCK_GOVERNOR: '',
    LOAD_BALANCER: '',
    AI_AGENT_MATCHER: ''
  },
  sepolia: {
    AGENT_REGISTRY: '',
    MESSAGE_ROUTER: '',
    A2A_TOKEN: '',
    TIMELOCK_GOVERNOR: '',
    LOAD_BALANCER: '',
    AI_AGENT_MATCHER: ''
  },
  polygon: {
    AGENT_REGISTRY: '',
    MESSAGE_ROUTER: '',
    A2A_TOKEN: '',
    TIMELOCK_GOVERNOR: '',
    LOAD_BALANCER: '',
    AI_AGENT_MATCHER: ''
  }
};

export const CONTRACT_NAMES = [
  'AgentRegistry',
  'MessageRouter',
  'TokenManager',
  'Governance'
] as const;

export type ContractName = typeof CONTRACT_NAMES[number];