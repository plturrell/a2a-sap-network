// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";
import "./ZKMessageProof.sol";

/**
 * @title EncryptedAgentRegistry  
 * @notice Privacy-preserving agent registry with encrypted capabilities and ZK proofs
 * @dev Allows agents to register with hidden capabilities while maintaining verifiability
 */
contract EncryptedAgentRegistry is AccessControl, ReentrancyGuard, Pausable {
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    bytes32 public constant VERIFIER_ROLE = keccak256("VERIFIER_ROLE");
    
    ZKMessageProof public immutable zkProofContract;

    // Encrypted agent information
    struct EncryptedAgent {
        address owner;
        bytes encryptedName;           // Encrypted agent name
        bytes encryptedEndpoint;       // Encrypted endpoint URL
        bytes32[] capabilityCommitments; // Commitments to capabilities
        bytes encryptedMetadata;       // Additional encrypted data
        uint256 reputation;
        uint256 registeredAt;
        bool active;
        bytes32 zkIdentityCommitment;  // ZK commitment for identity proofs
        uint256 stakingDeposit;       // Required stake for privacy features
    }

    // Public discovery hints (encrypted with different keys)
    struct DiscoveryHint {
        bytes32 capabilityCategory;    // Coarse-grained category
        bytes encryptedHint;          // Encrypted detailed capability info
        bytes32 geolocationHash;      // Optional location commitment
        uint256 priceRange;           // Price range category (1-10)
        bool acceptingWork;           // Whether accepting new work
    }

    // Capability proof structure
    struct CapabilityProof {
        bytes32 capabilityCommitment;
        bytes32 proofHash;            // Hash of ZK proof
        uint256 timestamp;
        bool verified;
    }

    // State variables
    mapping(address => EncryptedAgent) public encryptedAgents;
    mapping(address => DiscoveryHint) public discoveryHints;
    mapping(bytes32 => CapabilityProof) public capabilityProofs;
    mapping(bytes32 => address[]) public categoryAgents; // category -> agents
    mapping(address => bytes32[]) public agentCapabilityProofs;
    
    address[] public allAgents;
    uint256 public activeAgentsCount;
    uint256 public constant MIN_STAKING_DEPOSIT = 0.1 ether;
    uint256 public constant REPUTATION_DECAY_PERIOD = 90 days;

    // Events
    event EncryptedAgentRegistered(
        address indexed agent,
        bytes32 indexed zkIdentityCommitment,
        bytes32 indexed category,
        uint256 stakingDeposit
    );
    event CapabilityProofSubmitted(
        address indexed agent,
        bytes32 indexed capabilityCommitment,
        bytes32 proofHash
    );
    event DiscoveryHintUpdated(address indexed agent, bytes32 indexed category);
    event ReputationUpdated(address indexed agent, uint256 newReputation);
    event StakeSlashed(address indexed agent, uint256 amount, string reason);

    // Custom errors
    error InsufficientStake();
    error AgentNotRegistered();
    error AgentAlreadyRegistered();
    error InvalidZKCommitment();
    error InvalidCapabilityProof();
    error UnauthorizedReputationUpdate();

    constructor(
        address admin,
        address pauser,
        address zkProofContract_
    ) {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(PAUSER_ROLE, pauser);
        _grantRole(VERIFIER_ROLE, admin);
        zkProofContract = ZKMessageProof(zkProofContract_);
    }

    /**
     * @notice Register agent with encrypted information and ZK identity commitment
     * @param encryptedName Encrypted agent name
     * @param encryptedEndpoint Encrypted endpoint
     * @param capabilityCommitments Array of capability commitments
     * @param encryptedMetadata Additional encrypted metadata
     * @param zkIdentityCommitment ZK commitment for identity proofs
     * @param discoveryCategory Coarse category for discovery
     * @param discoveryHint Encrypted discovery hint
     */
    function registerEncryptedAgent(
        bytes calldata encryptedName,
        bytes calldata encryptedEndpoint,
        bytes32[] calldata capabilityCommitments,
        bytes calldata encryptedMetadata,
        bytes32 zkIdentityCommitment,
        bytes32 discoveryCategory,
        bytes calldata discoveryHint
    ) external payable whenNotPaused nonReentrant {
        if (msg.value < MIN_STAKING_DEPOSIT) revert InsufficientStake();
        if (encryptedAgents[msg.sender].owner != address(0)) revert AgentAlreadyRegistered();
        if (zkIdentityCommitment == bytes32(0)) revert InvalidZKCommitment();

        // Register encrypted agent
        encryptedAgents[msg.sender] = EncryptedAgent({
            owner: msg.sender,
            encryptedName: encryptedName,
            encryptedEndpoint: encryptedEndpoint,
            capabilityCommitments: capabilityCommitments,
            encryptedMetadata: encryptedMetadata,
            reputation: 100, // Starting reputation
            registeredAt: block.timestamp,
            active: true,
            zkIdentityCommitment: zkIdentityCommitment,
            stakingDeposit: msg.value
        });

        // Set discovery hints
        discoveryHints[msg.sender] = DiscoveryHint({
            capabilityCategory: discoveryCategory,
            encryptedHint: discoveryHint,
            geolocationHash: bytes32(0), // Can be set later
            priceRange: 5, // Default mid-range
            acceptingWork: true
        });

        // Add to indexes
        allAgents.push(msg.sender);
        categoryAgents[discoveryCategory].push(msg.sender);
        activeAgentsCount++;

        emit EncryptedAgentRegistered(
            msg.sender,
            zkIdentityCommitment,
            discoveryCategory,
            msg.value
        );
    }

    /**
     * @notice Submit capability proof using zero-knowledge
     * @param capabilityCommitment The capability being proven
     * @param proof ZK proof of capability
     * @param publicInputs Public inputs for proof verification
     * @param nullifier Unique nullifier for the proof
     */
    function submitCapabilityProof(
        bytes32 capabilityCommitment,
        ZKMessageProof.Proof calldata proof,
        uint256[] calldata publicInputs,
        bytes32 nullifier
    ) external whenNotPaused nonReentrant {
        if (encryptedAgents[msg.sender].owner == address(0)) revert AgentNotRegistered();
        
        // Verify the agent has committed to this capability
        EncryptedAgent storage agent = encryptedAgents[msg.sender];
        bool hasCommitment = false;
        for (uint256 i = 0; i < agent.capabilityCommitments.length; i++) {
            if (agent.capabilityCommitments[i] == capabilityCommitment) {
                hasCommitment = true;
                break;
            }
        }
        if (!hasCommitment) revert InvalidCapabilityProof();

        // Create commitment for proof verification
        bytes32 proofCommitmentId = zkProofContract.createMessageCommitment(
            keccak256(abi.encode(capabilityCommitment)),
            agent.zkIdentityCommitment,
            capabilityCommitment, // Recipient is the capability itself
            zkProofContract.getCurrentNonce(msg.sender)
        );

        // Verify ZK proof
        try zkProofContract.verifyMessageProof(
            proofCommitmentId,
            proof,
            publicInputs,
            nullifier
        ) {
            bytes32 proofHash = keccak256(abi.encode(proof));
            
            // Store capability proof
            capabilityProofs[capabilityCommitment] = CapabilityProof({
                capabilityCommitment: capabilityCommitment,
                proofHash: proofHash,
                timestamp: block.timestamp,
                verified: true
            });
            
            agentCapabilityProofs[msg.sender].push(capabilityCommitment);
            
            // Increase reputation for successful proof
            agent.reputation = _min(agent.reputation + 10, 200);
            
            emit CapabilityProofSubmitted(msg.sender, capabilityCommitment, proofHash);
            emit ReputationUpdated(msg.sender, agent.reputation);
        } catch {
            revert InvalidCapabilityProof();
        }
    }

    /**
     * @notice Update discovery hints for better matching
     * @param newCategory Updated category
     * @param newHint Updated encrypted hint
     * @param geolocationHash Optional location hash
     * @param priceRange Price range (1-10)
     * @param acceptingWork Whether accepting work
     */
    function updateDiscoveryHints(
        bytes32 newCategory,
        bytes calldata newHint,
        bytes32 geolocationHash,
        uint256 priceRange,
        bool acceptingWork
    ) external whenNotPaused {
        if (encryptedAgents[msg.sender].owner == address(0)) revert AgentNotRegistered();
        require(priceRange >= 1 && priceRange <= 10, "Invalid price range");

        DiscoveryHint storage hint = discoveryHints[msg.sender];
        
        // Update category index if changed
        if (hint.capabilityCategory != newCategory) {
            _removeFromCategory(msg.sender, hint.capabilityCategory);
            categoryAgents[newCategory].push(msg.sender);
        }
        
        hint.capabilityCategory = newCategory;
        hint.encryptedHint = newHint;
        hint.geolocationHash = geolocationHash;
        hint.priceRange = priceRange;
        hint.acceptingWork = acceptingWork;

        emit DiscoveryHintUpdated(msg.sender, newCategory);
    }

    /**
     * @notice Discover agents by category with privacy preservation
     * @param category The category to search
     * @param priceRangeMin Minimum price range
     * @param priceRangeMax Maximum price range
     * @param acceptingWorkOnly Only return agents accepting work
     * @param offset Pagination offset
     * @param limit Maximum results
     * @return agents Array of agent addresses
     * @return hints Array of discovery hints
     * @return total Total matching agents
     */
    function discoverAgentsByCategory(
        bytes32 category,
        uint256 priceRangeMin,
        uint256 priceRangeMax,
        bool acceptingWorkOnly,
        uint256 offset,
        uint256 limit
    ) external view returns (
        address[] memory agents,
        DiscoveryHint[] memory hints,
        uint256 total
    ) {
        return _discoverAgentsByCategory(
            category,
            priceRangeMin,
            priceRangeMax,
            acceptingWorkOnly,
            offset,
            limit
        );
    }

    function _discoverAgentsByCategory(
        bytes32 category,
        uint256 priceMin,
        uint256 priceMax,
        bool acceptingOnly,
        uint256 offset,
        uint256 limit
    ) internal view returns (
        address[] memory agents,
        DiscoveryHint[] memory hints,
        uint256 total
    ) {
        address[] storage categoryList = categoryAgents[category];
        
        // First pass: count matches
        total = 0;
        for (uint256 i = 0; i < categoryList.length; i++) {
            address agent = categoryList[i];
            if (_isAgentMatch(agent, priceMin, priceMax, acceptingOnly)) {
                total++;
            }
        }
        
        if (offset >= total) return (new address[](0), new DiscoveryHint[](0), total);
        
        uint256 end = offset + limit;
        if (end > total) end = total;
        uint256 resultCount = end - offset;
        
        agents = new address[](resultCount);
        hints = new DiscoveryHint[](resultCount);
        
        // Second pass: collect results
        uint256 currentIndex = 0;
        uint256 resultIndex = 0;
        
        for (uint256 i = 0; i < categoryList.length && resultIndex < resultCount; i++) {
            address agent = categoryList[i];
            if (_isAgentMatch(agent, priceMin, priceMax, acceptingOnly)) {
                if (currentIndex >= offset) {
                    agents[resultIndex] = agent;
                    hints[resultIndex] = discoveryHints[agent];
                    resultIndex++;
                }
                currentIndex++;
            }
        }
    }

    function _isAgentMatch(
        address agent,
        uint256 priceMin,
        uint256 priceMax,
        bool acceptingOnly
    ) internal view returns (bool) {
        DiscoveryHint storage hint = discoveryHints[agent];
        return hint.priceRange >= priceMin &&
               hint.priceRange <= priceMax &&
               (!acceptingOnly || hint.acceptingWork) &&
               encryptedAgents[agent].active;
    }

    /**
     * @notice Get encrypted agent information
     * @param agent The agent address
     * @return Encrypted agent data
     */
    function getEncryptedAgent(address agent) external view returns (EncryptedAgent memory) {
        if (encryptedAgents[agent].owner == address(0)) revert AgentNotRegistered();
        return encryptedAgents[agent];
    }

    /**
     * @notice Slash stake for misbehavior (admin/verifier only)
     * @param agent Agent to slash
     * @param amount Amount to slash
     * @param reason Reason for slashing
     */
    function slashStake(
        address agent,
        uint256 amount,
        string calldata reason
    ) external onlyRole(VERIFIER_ROLE) {
        EncryptedAgent storage agentData = encryptedAgents[agent];
        if (agentData.owner == address(0)) revert AgentNotRegistered();
        
        if (amount > agentData.stakingDeposit) {
            amount = agentData.stakingDeposit;
        }
        
        agentData.stakingDeposit -= amount;
        
        // Reduce reputation for slashing
        if (agentData.reputation > 20) {
            agentData.reputation -= 20;
        } else {
            agentData.reputation = 0;
        }
        
        // Send slashed funds to admin (could be treasury)
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Slash transfer failed");
        
        emit StakeSlashed(agent, amount, reason);
        emit ReputationUpdated(agent, agentData.reputation);
    }

    /**
     * @notice Increase agent reputation (verifier only)
     * @param agent Agent to update
     * @param amount Amount to increase
     */
    function increaseReputation(
        address agent,
        uint256 amount
    ) external onlyRole(VERIFIER_ROLE) {
        EncryptedAgent storage agentData = encryptedAgents[agent];
        if (agentData.owner == address(0)) revert AgentNotRegistered();
        
        agentData.reputation = _min(agentData.reputation + amount, 200);
        emit ReputationUpdated(agent, agentData.reputation);
    }


    /**
     * @notice Remove agent from category index
     */
    function _removeFromCategory(address agent, bytes32 category) internal {
        address[] storage categoryList = categoryAgents[category];
        for (uint256 i = 0; i < categoryList.length; i++) {
            if (categoryList[i] == agent) {
                categoryList[i] = categoryList[categoryList.length - 1];
                categoryList.pop();
                break;
            }
        }
    }

    /**
     * @notice Helper function for minimum of two numbers
     */
    function _min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }

    /**
     * @notice Pause contract (emergency use)
     */
    function pause() external onlyRole(PAUSER_ROLE) {
        _pause();
    }

    /**
     * @notice Unpause contract
     */
    function unpause() external onlyRole(PAUSER_ROLE) {
        _unpause();
    }

    /**
     * @notice Withdraw slashed funds (admin only)
     */
    function withdraw() external onlyRole(DEFAULT_ADMIN_ROLE) {
        uint256 balance = address(this).balance;
        require(balance > 0, "No funds to withdraw");
        
        (bool success, ) = msg.sender.call{value: balance}("");
        require(success, "Withdrawal failed");
    }

    // Receive function for staking deposits
    receive() external payable {}
}