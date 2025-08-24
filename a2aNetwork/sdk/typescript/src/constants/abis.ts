export const AGENT_REGISTRY_ABI = [
  'function registerAgent(string memory _name, string[] memory _capabilities) external returns (uint256)',
  'function getAgent(address _address) external view returns (tuple(string name, address owner, string[] capabilities, bool isActive, uint256 reputation))',
  'function updateCapabilities(string[] memory _capabilities) external',
  'function deactivateAgent() external',
  'function activateAgent() external',
  'event AgentRegistered(address indexed agent, uint256 agentId, string name)',
  'event AgentUpdated(address indexed agent, string[] capabilities)',
  'event AgentDeactivated(address indexed agent)',
  'event AgentActivated(address indexed agent)'
];

export const MESSAGE_ROUTER_ABI = [
  'function sendMessage(address _to, bytes memory _content, bool _encrypted) external returns (uint256)',
  'function getMessage(uint256 _messageId) external view returns (tuple(address from, address to, bytes content, uint256 timestamp, bool encrypted, bool read))',
  'function markAsRead(uint256 _messageId) external',
  'function getInbox(address _agent) external view returns (uint256[] memory)',
  'function getOutbox(address _agent) external view returns (uint256[] memory)',
  'event MessageSent(uint256 indexed messageId, address indexed from, address indexed to)',
  'event MessageRead(uint256 indexed messageId, address indexed reader)'
];

export const TOKEN_MANAGER_ABI = [
  'function mint(address to, uint256 amount) external',
  'function burn(address from, uint256 amount) external',
  'function transfer(address to, uint256 amount) external returns (bool)',
  'function balanceOf(address account) external view returns (uint256)',
  'function totalSupply() external view returns (uint256)',
  'event Transfer(address indexed from, address indexed to, uint256 value)',
  'event Mint(address indexed to, uint256 value)',
  'event Burn(address indexed from, uint256 value)'
];

export const GOVERNANCE_ABI = [
  'function propose(string memory description, bytes memory data) external returns (uint256)',
  'function vote(uint256 proposalId, bool support) external',
  'function execute(uint256 proposalId) external',
  'function getProposal(uint256 proposalId) external view returns (tuple(string description, uint256 forVotes, uint256 againstVotes, uint256 startTime, uint256 endTime, bool executed, address proposer))',
  'event ProposalCreated(uint256 indexed proposalId, address indexed proposer, string description)',
  'event VoteCast(uint256 indexed proposalId, address indexed voter, bool support, uint256 weight)',
  'event ProposalExecuted(uint256 indexed proposalId)'
];

export const A2A_TOKEN_ABI = [
  'function transfer(address to, uint256 amount) external returns (bool)',
  'function balanceOf(address account) external view returns (uint256)',
  'function totalSupply() external view returns (uint256)',
  'function mint(address to, uint256 amount) external',
  'function burn(address from, uint256 amount) external',
  'event Transfer(address indexed from, address indexed to, uint256 value)'
];

export const LOAD_BALANCER_ABI = [
  'function distributeLoad(address[] memory agents, uint256[] memory loads) external',
  'function getCurrentLoad(address agent) external view returns (uint256)',
  'function getOptimalAgent() external view returns (address)',
  'event LoadDistributed(address[] agents, uint256[] loads)'
];

export const AI_AGENT_MATCHER_ABI = [
  'function matchAgent(string[] memory requirements) external view returns (address)',
  'function getTopAgentsBySkills(bytes32[] memory skills, uint256 limit) external view returns (address[], uint256[])',
  'function getAIAgentProfile(address agent) external view returns (tuple(uint256 totalTasksCompleted, uint256 taskSuccessRate, uint256 avgResponseTime, uint256 totalEarnings, uint256[] performanceMetrics, bytes32[] skillTags))',
  'event AgentMatched(address indexed requester, address indexed agent, string[] requirements)'
];

export const REPUTATION_MANAGER_ABI = [
  'function getReputationScore(address agent) external view returns (uint256)',
  'function updateReputation(address agent, int256 change, string memory reason) external',
  'function getReputationHistory(address agent, uint256 limit) external view returns (tuple(uint256 timestamp, uint256 score, int256 change, string reason)[])',
  'event ReputationUpdated(address indexed agent, uint256 newScore, int256 change, string reason)'
];

export const SCALABILITY_MANAGER_ABI = [
  'function executeBatch(tuple(address to, bytes data, uint256 value)[] transactions) external returns (bytes[])',
  'function optimizeGas(tuple(address to, bytes data, uint256 value) transaction) external view returns (tuple(uint256 gasLimit, uint256 gasPrice, uint256 maxFeePerGas, uint256 maxPriorityFeePerGas))',
  'function getNetworkMetrics() external view returns (tuple(uint256 tps, uint256 avgBlockTime, uint256 pendingTxs, uint256 gasPrice, uint8 congestion))',
  'event BatchExecuted(address indexed sender, uint256 transactionCount, bytes[] results)'
];