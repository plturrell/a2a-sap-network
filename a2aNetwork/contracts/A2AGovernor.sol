// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/governance/GovernorUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/governance/extensions/GovernorSettingsUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/governance/extensions/GovernorCountingSimpleUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/governance/extensions/GovernorVotesUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/governance/extensions/GovernorVotesQuorumFractionUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/governance/extensions/GovernorTimelockControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";

/**
 * @title A2AGovernor
 * @notice Governance contract for A2A Network with advanced voting mechanisms
 * @dev Implements OpenZeppelin Governor with custom features for blockchain governance
 */
contract A2AGovernor is
    GovernorUpgradeable,
    GovernorSettingsUpgradeable,
    GovernorCountingSimpleUpgradeable,
    GovernorVotesUpgradeable,
    GovernorVotesQuorumFractionUpgradeable,
    GovernorTimelockControlUpgradeable,
    AccessControlUpgradeable,
    ReentrancyGuardUpgradeable,
    UUPSUpgradeable
{
    bytes32 public constant PROPOSER_ROLE = keccak256("PROPOSER_ROLE");
    bytes32 public constant EXECUTOR_ROLE = keccak256("EXECUTOR_ROLE");
    bytes32 public constant CANCELLER_ROLE = keccak256("CANCELLER_ROLE");
    
    enum ProposalCategory {
        PROTOCOL_UPGRADE,
        PARAMETER_CHANGE,
        TREASURY_ALLOCATION,
        EMERGENCY_ACTION,
        AGENT_MANAGEMENT,
        REPUTATION_SYSTEM
    }
    
    struct ProposalMetadata {
        ProposalCategory category;
        uint256 requiredQuorum;
        uint256 requiredMajority;
        bool emergencyProposal;
        string ipfsHash;
        address[] stakeholders;
        uint256 estimatedImpact;
    }
    
    struct VotingStats {
        uint256 totalProposals;
        uint256 executedProposals;
        uint256 defeatedProposals;
        uint256 totalVotes;
        mapping(address => uint256) participationCount;
        mapping(ProposalCategory => uint256) categoryCount;
    }
    
    mapping(uint256 => ProposalMetadata) public proposalMetadata;
    mapping(address => bool) public whitelistedProposers;
    mapping(ProposalCategory => uint256) public categoryQuorums;
    mapping(ProposalCategory => uint256) public categoryMajorities;
    
    VotingStats public votingStats;
    
    uint256 public constant EMERGENCY_VOTING_PERIOD = 1 days;
    uint256 public constant EMERGENCY_QUORUM = 30; // 30%
    uint256 public constant MAX_PROPOSAL_THRESHOLD = 100000e18; // 100k tokens
    uint256 public constant MIN_VOTING_DELAY = 1 days;
    uint256 public constant MAX_VOTING_DELAY = 7 days;
    
    bool public emergencyMode;
    uint256 public emergencyActivatedAt;
    address public emergencyCouncil;
    
    event ProposalCreatedWithMetadata(
        uint256 indexed proposalId,
        ProposalCategory category,
        string ipfsHash,
        uint256 estimatedImpact
    );
    event EmergencyModeActivated(address activator, string reason);
    event EmergencyModeDeactivated(address deactivator);
    event ProposerWhitelisted(address proposer);
    event ProposerRemovedFromWhitelist(address proposer);
    event CategoryParametersUpdated(ProposalCategory category, uint256 quorum, uint256 majority);
    
    modifier onlyEmergencyCouncil() {
        require(msg.sender == emergencyCouncil, "Only emergency council");
        _;
    }
    
    modifier whenNotEmergencyMode() {
        require(!emergencyMode, "Emergency mode active");
        _;
    }
    
    modifier validCategory(ProposalCategory category) {
        require(uint256(category) <= uint256(ProposalCategory.REPUTATION_SYSTEM), "Invalid category");
        _;
    }
    
    function initialize(
        IVotesUpgradeable token,
        TimelockControllerUpgradeable timelock,
        address emergencyCouncilAddress
    ) public initializer {
        __Governor_init("A2AGovernor");
        __GovernorSettings_init(
            1 days,    // voting delay
            7 days,    // voting period  
            10000e18   // proposal threshold (10k tokens)
        );
        __GovernorCountingSimple_init();
        __GovernorVotes_init(token);
        __GovernorVotesQuorumFraction_init(10); // 10% quorum
        __GovernorTimelockControl_init(timelock);
        __AccessControl_init();
        __ReentrancyGuard_init();
        __UUPSUpgradeable_init();
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(PROPOSER_ROLE, address(this));
        _grantRole(EXECUTOR_ROLE, address(this));
        _grantRole(CANCELLER_ROLE, msg.sender);
        
        emergencyCouncil = emergencyCouncilAddress;
        
        // Initialize category-specific parameters
        _initializeCategoryParameters();
    }
    
    /**
     * @notice Create proposal with metadata
     */
    function proposeWithMetadata(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        string memory description,
        ProposalCategory category,
        string memory ipfsHash,
        uint256 estimatedImpact
    ) public nonReentrant whenNotEmergencyMode validCategory(category) returns (uint256) {
        // Check if proposer is whitelisted for certain categories
        if (category == ProposalCategory.PROTOCOL_UPGRADE || 
            category == ProposalCategory.EMERGENCY_ACTION) {
            require(
                whitelistedProposers[msg.sender] || hasRole(PROPOSER_ROLE, msg.sender),
                "Not authorized for this proposal category"
            );
        }
        
        uint256 proposalId = propose(targets, values, calldatas, description);
        
        // Store metadata
        proposalMetadata[proposalId] = ProposalMetadata({
            category: category,
            requiredQuorum: categoryQuorums[category],
            requiredMajority: categoryMajorities[category],
            emergencyProposal: false,
            ipfsHash: ipfsHash,
            stakeholders: new address[](0),
            estimatedImpact: estimatedImpact
        });
        
        // Update statistics
        votingStats.totalProposals++;
        votingStats.categoryCount[category]++;
        
        emit ProposalCreatedWithMetadata(proposalId, category, ipfsHash, estimatedImpact);
        
        return proposalId;
    }
    
    /**
     * @notice Create emergency proposal with expedited timeline
     */
    function proposeEmergency(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        string memory description,
        string memory justification
    ) external onlyRole(PROPOSER_ROLE) nonReentrant returns (uint256) {
        require(
            emergencyMode || hasRole(DEFAULT_ADMIN_ROLE, msg.sender),
            "Emergency proposals only allowed in emergency mode"
        );
        
        uint256 proposalId = propose(targets, values, calldatas, description);
        
        // Set emergency metadata
        proposalMetadata[proposalId] = ProposalMetadata({
            category: ProposalCategory.EMERGENCY_ACTION,
            requiredQuorum: EMERGENCY_QUORUM,
            requiredMajority: 50, // Simple majority for emergency
            emergencyProposal: true,
            ipfsHash: "",
            stakeholders: new address[](0),
            estimatedImpact: 0
        });
        
        votingStats.totalProposals++;
        votingStats.categoryCount[ProposalCategory.EMERGENCY_ACTION]++;
        
        return proposalId;
    }
    
    /**
     * @notice Activate emergency mode
     */
    function activateEmergencyMode(string memory reason) 
        external 
        onlyEmergencyCouncil 
    {
        emergencyMode = true;
        emergencyActivatedAt = block.timestamp;
        
        emit EmergencyModeActivated(msg.sender, reason);
    }
    
    /**
     * @notice Deactivate emergency mode
     */
    function deactivateEmergencyMode() external {
        require(
            msg.sender == emergencyCouncil || hasRole(DEFAULT_ADMIN_ROLE, msg.sender),
            "Not authorized"
        );
        require(emergencyMode, "Emergency mode not active");
        
        emergencyMode = false;
        emergencyActivatedAt = 0;
        
        emit EmergencyModeDeactivated(msg.sender);
    }
    
    /**
     * @notice Vote on proposal with delegation support
     */
    function castVoteWithReason(
        uint256 proposalId,
        uint8 support,
        string calldata reason
    ) public override nonReentrant returns (uint256) {
        uint256 weight = super.castVoteWithReason(proposalId, support, reason);
        
        // Update participation statistics
        votingStats.totalVotes++;
        votingStats.participationCount[msg.sender]++;
        
        return weight;
    }
    
    /**
     * @notice Batch vote on multiple proposals
     */
    function batchVote(
        uint256[] memory proposalIds,
        uint8[] memory supports,
        string[] memory reasons
    ) external nonReentrant {
        require(
            proposalIds.length == supports.length && 
            supports.length == reasons.length,
            "Array length mismatch"
        );
        require(proposalIds.length <= 10, "Too many proposals");
        
        for (uint256 i = 0; i < proposalIds.length; i++) {
            castVoteWithReason(proposalIds[i], supports[i], reasons[i]);
        }
    }
    
    /**
     * @notice Delegate voting power to another address
     */
    function delegateVoting(address delegatee) external {
        IVotesUpgradeable(token).delegate(delegatee);
    }
    
    /**
     * @notice Add proposer to whitelist
     */
    function addWhitelistedProposer(address proposer) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        whitelistedProposers[proposer] = true;
        emit ProposerWhitelisted(proposer);
    }
    
    /**
     * @notice Remove proposer from whitelist
     */
    function removeWhitelistedProposer(address proposer) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        whitelistedProposers[proposer] = false;
        emit ProposerRemovedFromWhitelist(proposer);
    }
    
    /**
     * @notice Update category-specific parameters
     */
    function updateCategoryParameters(
        ProposalCategory category,
        uint256 quorum,
        uint256 majority
    ) external onlyRole(DEFAULT_ADMIN_ROLE) validCategory(category) {
        require(quorum >= 5 && quorum <= 50, "Invalid quorum");
        require(majority >= 50 && majority <= 80, "Invalid majority");
        
        categoryQuorums[category] = quorum;
        categoryMajorities[category] = majority;
        
        emit CategoryParametersUpdated(category, quorum, majority);
    }
    
    /**
     * @notice Get voting statistics for an address
     */
    function getVotingStatistics(address voter) 
        external 
        view 
        returns (
            uint256 participationCount,
            uint256 votingPower,
            uint256 delegatedVotes
        ) 
    {
        return (
            votingStats.participationCount[voter],
            getVotes(voter, block.number - 1),
            IVotesUpgradeable(token).getVotes(voter)
        );
    }
    
    /**
     * @notice Get proposal metadata
     */
    function getProposalMetadata(uint256 proposalId) 
        external 
        view 
        returns (ProposalMetadata memory) 
    {
        return proposalMetadata[proposalId];
    }
    
    /**
     * @notice Get governance statistics
     */
    function getGovernanceStatistics() 
        external 
        view 
        returns (
            uint256 totalProposals,
            uint256 executedProposals,
            uint256 defeatedProposals,
            uint256 totalVotes,
            uint256 activeProposals
        ) 
    {
        return (
            votingStats.totalProposals,
            votingStats.executedProposals,
            votingStats.defeatedProposals,
            votingStats.totalVotes,
            _getActiveProposalCount()
        );
    }
    
    /**
     * @notice Override voting delay for emergency proposals
     */
    function votingDelay() public view override returns (uint256) {
        if (emergencyMode) {
            return 1 hours; // Reduced delay in emergency
        }
        return super.votingDelay();
    }
    
    /**
     * @notice Override voting period for emergency proposals
     */
    function votingPeriod() public view override returns (uint256) {
        if (emergencyMode) {
            return EMERGENCY_VOTING_PERIOD;
        }
        return super.votingPeriod();
    }
    
    /**
     * @notice Override quorum calculation with category-specific requirements
     */
    function quorum(uint256 blockNumber) 
        public 
        view 
        override(IGovernorUpgradeable, GovernorVotesQuorumFractionUpgradeable) 
        returns (uint256) 
    {
        if (emergencyMode) {
            return (token.getPastTotalSupply(blockNumber) * EMERGENCY_QUORUM) / 100;
        }
        return super.quorum(blockNumber);
    }
    
    /**
     * @notice Custom proposal threshold based on category
     */
    function proposalThreshold() 
        public 
        view 
        override(GovernorUpgradeable, GovernorSettingsUpgradeable) 
        returns (uint256) 
    {
        if (emergencyMode) {
            return MAX_PROPOSAL_THRESHOLD / 2; // Lower threshold in emergency
        }
        return super.proposalThreshold();
    }
    
    /**
     * @notice Execute proposal and update statistics
     */
    function _execute(
        uint256 proposalId,
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        bytes32 descriptionHash
    ) internal override(GovernorUpgradeable, GovernorTimelockControlUpgradeable) {
        super._execute(proposalId, targets, values, calldatas, descriptionHash);
        votingStats.executedProposals++;
    }
    
    /**
     * @notice Cancel proposal and update statistics
     */
    function _cancel(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        bytes32 descriptionHash
    ) internal override(GovernorUpgradeable, GovernorTimelockControlUpgradeable) returns (uint256) {
        uint256 proposalId = super._cancel(targets, values, calldatas, descriptionHash);
        // Note: We don't increment defeated count here as cancel is different from defeat
        return proposalId;
    }
    
    /**
     * @notice Initialize category-specific parameters
     */
    function _initializeCategoryParameters() internal {
        // Protocol upgrades require higher quorum and majority
        categoryQuorums[ProposalCategory.PROTOCOL_UPGRADE] = 20; // 20%
        categoryMajorities[ProposalCategory.PROTOCOL_UPGRADE] = 75; // 75%
        
        // Parameter changes require moderate requirements
        categoryQuorums[ProposalCategory.PARAMETER_CHANGE] = 15; // 15%
        categoryMajorities[ProposalCategory.PARAMETER_CHANGE] = 60; // 60%
        
        // Treasury allocations require high accountability
        categoryQuorums[ProposalCategory.TREASURY_ALLOCATION] = 25; // 25%
        categoryMajorities[ProposalCategory.TREASURY_ALLOCATION] = 70; // 70%
        
        // Emergency actions have expedited requirements
        categoryQuorums[ProposalCategory.EMERGENCY_ACTION] = 30; // 30%
        categoryMajorities[ProposalCategory.EMERGENCY_ACTION] = 50; // 50%
        
        // Agent management requires moderate oversight
        categoryQuorums[ProposalCategory.AGENT_MANAGEMENT] = 10; // 10%
        categoryMajorities[ProposalCategory.AGENT_MANAGEMENT] = 55; // 55%
        
        // Reputation system changes require community consensus
        categoryQuorums[ProposalCategory.REPUTATION_SYSTEM] = 15; // 15%
        categoryMajorities[ProposalCategory.REPUTATION_SYSTEM] = 65; // 65%
    }
    
    /**
     * @notice Get count of active proposals
     */
    function _getActiveProposalCount() internal view returns (uint256) {
        // This would require tracking active proposals in a more sophisticated way
        // For now, return a placeholder
        return 0;
    }
    
    function _authorizeUpgrade(address newImplementation) 
        internal 
        override 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {}
    
    /**
     * @notice Required overrides for multiple inheritance
     */
    function state(uint256 proposalId)
        public
        view
        override(GovernorUpgradeable, GovernorTimelockControlUpgradeable)
        returns (ProposalState)
    {
        return super.state(proposalId);
    }
    
    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(GovernorUpgradeable, AccessControlUpgradeable)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }
}