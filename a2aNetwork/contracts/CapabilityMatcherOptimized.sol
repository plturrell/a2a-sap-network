// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "./interfaces/IAgentRegistry.sol";

/**
 * @title CapabilityMatcher - Optimized
 * @notice Semantic capability matching for agent discovery and task routing
 * @dev Optimized version focusing on core functionality
 */
contract CapabilityMatcherOptimized is AccessControlUpgradeable, UUPSUpgradeable {
    bytes32 public constant CAPABILITY_ADMIN_ROLE = keccak256("CAPABILITY_ADMIN_ROLE");
    
    IAgentRegistry public agentRegistry;
    
    // Optimized structs with packed data types
    struct Capability {
        string name;
        string description;
        bytes32[] tags; // Use bytes32 instead of string[] for efficiency
        uint16 category;
        bool verified;
        uint8 status; // 0=Active, 1=Deprecated, 2=Disabled
    }
    
    struct AgentProfile {
        uint256[] capabilityIds;
        uint8 currentWorkload;
        uint32 lastUpdated;
    }
    
    // Core storage
    mapping(uint256 => Capability) public capabilities;
    mapping(address => AgentProfile) public agentProfiles;
    mapping(string => uint256) public capabilityNameToId;
    mapping(uint256 => address[]) public capabilityProviders;
    mapping(address => mapping(uint256 => uint8)) public proficiencyScores; // Flattened mapping
    
    uint256 public capabilityCounter;
    uint16 public constant MAX_CAPABILITIES_PER_AGENT = 50;
    uint8 public constant PROFICIENCY_THRESHOLD = 70;
    uint8 public constant MAX_WORKLOAD_PER_AGENT = 10;
    
    // Events
    event CapabilityRegistered(uint256 indexed capabilityId, string name);
    event AgentCapabilityAdded(address indexed agent, uint256 indexed capabilityId);
    event ProficiencyUpdated(address indexed agent, uint256 indexed capabilityId, uint8 score);
    event CapabilityMatched(bytes32 indexed taskHash, address[] agents);
    
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }
    
    function initialize(address _agentRegistry) public initializer {
        __AccessControl_init();
        __UUPSUpgradeable_init();
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(CAPABILITY_ADMIN_ROLE, msg.sender);
        
        agentRegistry = IAgentRegistry(_agentRegistry);
    }
    
    /**
     * @notice Register a new capability
     */
    function registerCapability(
        string calldata name,
        string calldata description,
        bytes32[] calldata tags,
        uint16 category
    ) external onlyRole(CAPABILITY_ADMIN_ROLE) returns (uint256) {
        require(bytes(name).length > 0, "Empty name");
        require(capabilityNameToId[name] == 0, "Capability exists");
        
        capabilityCounter++;
        uint256 capabilityId = capabilityCounter;
        
        capabilities[capabilityId] = Capability({
            name: name,
            description: description,
            tags: tags,
            category: category,
            verified: true,
            status: 0 // Active
        });
        
        capabilityNameToId[name] = capabilityId;
        
        emit CapabilityRegistered(capabilityId, name);
        return capabilityId;
    }
    
    /**
     * @notice Add capability to agent profile
     */
    function addAgentCapability(
        address agent,
        uint256 capabilityId,
        uint8 proficiency
    ) external {
        require(msg.sender == agent || hasRole(CAPABILITY_ADMIN_ROLE, msg.sender), "Unauthorized");
        require(capabilities[capabilityId].status == 0, "Capability not active");
        require(proficiency <= 100, "Invalid proficiency");
        
        AgentProfile storage profile = agentProfiles[agent];
        require(profile.capabilityIds.length < MAX_CAPABILITIES_PER_AGENT, "Too many capabilities");
        
        // Check if capability already exists
        for (uint256 i = 0; i < profile.capabilityIds.length; i++) {
            require(profile.capabilityIds[i] != capabilityId, "Capability exists");
        }
        
        profile.capabilityIds.push(capabilityId);
        profile.lastUpdated = uint32(block.timestamp);
        proficiencyScores[agent][capabilityId] = proficiency;
        
        capabilityProviders[capabilityId].push(agent);
        
        emit AgentCapabilityAdded(agent, capabilityId);
        emit ProficiencyUpdated(agent, capabilityId, proficiency);
    }
    
    /**
     * @notice Update agent proficiency for a capability
     */
    function updateProficiency(
        address agent,
        uint256 capabilityId,
        uint8 proficiency
    ) external {
        require(msg.sender == agent || hasRole(CAPABILITY_ADMIN_ROLE, msg.sender), "Unauthorized");
        require(proficiency <= 100, "Invalid proficiency");
        require(proficiencyScores[agent][capabilityId] > 0, "Capability not found");
        
        proficiencyScores[agent][capabilityId] = proficiency;
        agentProfiles[agent].lastUpdated = uint32(block.timestamp);
        
        emit ProficiencyUpdated(agent, capabilityId, proficiency);
    }
    
    /**
     * @notice Find agents with specific capability
     */
    function findAgentsWithCapability(
        uint256 capabilityId,
        uint8 minProficiency
    ) external view returns (address[] memory qualifiedAgents) {
        address[] memory providers = capabilityProviders[capabilityId];
        address[] memory temp = new address[](providers.length);
        uint256 count = 0;
        
        for (uint256 i = 0; i < providers.length; i++) {
            address agent = providers[i];
            if (proficiencyScores[agent][capabilityId] >= minProficiency &&
                agentProfiles[agent].currentWorkload < MAX_WORKLOAD_PER_AGENT) {
                temp[count] = agent;
                count++;
            }
        }
        
        qualifiedAgents = new address[](count);
        for (uint256 i = 0; i < count; i++) {
            qualifiedAgents[i] = temp[i];
        }
    }
    
    /**
     * @notice Match agents by capability name
     */
    function matchAgentsByCapabilityName(
        string calldata capabilityName,
        uint8 minProficiency
    ) external view returns (address[] memory) {
        uint256 capabilityId = capabilityNameToId[capabilityName];
        require(capabilityId > 0, "Capability not found");
        return this.findAgentsWithCapability(capabilityId, minProficiency);
    }
    
    /**
     * @notice Get agent capability profile
     */
    function getAgentProfile(address agent) external view returns (
        uint256[] memory capabilityIds,
        uint8[] memory proficiencies,
        uint8 currentWorkload
    ) {
        AgentProfile storage profile = agentProfiles[agent];
        capabilityIds = profile.capabilityIds;
        
        proficiencies = new uint8[](capabilityIds.length);
        for (uint256 i = 0; i < capabilityIds.length; i++) {
            proficiencies[i] = proficiencyScores[agent][capabilityIds[i]];
        }
        
        currentWorkload = profile.currentWorkload;
    }
    
    /**
     * @notice Update agent workload
     */
    function updateWorkload(address agent, uint8 workload) external onlyRole(CAPABILITY_ADMIN_ROLE) {
        require(workload <= MAX_WORKLOAD_PER_AGENT, "Workload too high");
        agentProfiles[agent].currentWorkload = workload;
    }
    
    /**
     * @notice Deprecate a capability
     */
    function deprecateCapability(uint256 capabilityId) external onlyRole(CAPABILITY_ADMIN_ROLE) {
        require(capabilities[capabilityId].status == 0, "Not active");
        capabilities[capabilityId].status = 1; // Deprecated
    }
    
    /**
     * @notice Get capability details
     */
    function getCapability(uint256 capabilityId) external view returns (
        string memory name,
        string memory description,
        bytes32[] memory tags,
        uint16 category,
        bool verified,
        uint8 status
    ) {
        Capability storage cap = capabilities[capabilityId];
        return (cap.name, cap.description, cap.tags, cap.category, cap.verified, cap.status);
    }
    
    /**
     * @notice Get total number of capabilities
     */
    function getTotalCapabilities() external view returns (uint256) {
        return capabilityCounter;
    }
    
    /**
     * @notice Get capability providers count
     */
    function getCapabilityProvidersCount(uint256 capabilityId) external view returns (uint256) {
        return capabilityProviders[capabilityId].length;
    }
    
    function _authorizeUpgrade(address newImplementation) internal override onlyRole(DEFAULT_ADMIN_ROLE) {}
}