// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "./interfaces/IAgentRegistry.sol";

/**
 * @title CapabilityMatcher
 * @notice Semantic capability matching for agent discovery and task routing
 * @dev Implements capability registration, matching, and intent resolution
 */
contract CapabilityMatcher is AccessControlUpgradeable, UUPSUpgradeable {
    bytes32 public constant CAPABILITY_ADMIN_ROLE = keccak256("CAPABILITY_ADMIN_ROLE");
    
    IAgentRegistry public agentRegistry;
    
    struct Capability {
        string name;
        string description;
        string[] tags;
        string[] inputTypes;
        string[] outputTypes;
        uint256 category;
        bool verified;
        // Versioning and lifecycle
        uint256 version;
        CapabilityStatus status;
        uint256 deprecationDate;
        uint256 sunsetDate;
        uint256 replacedBy; // ID of replacement capability
        // Dependencies
        uint256[] dependencies;
        uint256[] conflicts;
    }
    
    enum CapabilityStatus {
        Active,
        Deprecated,
        Sunset,
        Disabled
    }
    
    struct AgentCapabilityProfile {
        uint256[] capabilityIds;
        mapping(uint256 => uint256) proficiencyScores; // 0-100
        mapping(uint256 => uint256) completedTasks;
        uint256 lastUpdated;
        // Workload management
        uint256 currentWorkload;
        uint256 maxWorkload;
        uint256[] activeTaskIds;
        // Pricing
        mapping(uint256 => uint256) capabilityPrices; // capability ID => price per task
    }
    
    struct CompositeTask {
        uint256 taskId;
        string description;
        uint256[] requiredCapabilities;
        uint256[] subtaskIds;
        TaskComplexity complexity;
        uint256 estimatedDuration;
        address requester;
        TaskStatus status;
    }
    
    struct Subtask {
        uint256 subtaskId;
        uint256 parentTaskId;
        uint256 requiredCapability;
        address assignedAgent;
        uint256 priority;
        uint256 deadline;
        TaskStatus status;
        uint256 price;
    }
    
    enum TaskComplexity { Simple, Moderate, Complex, ExtraComplex }
    enum TaskStatus { Created, Assigned, InProgress, Completed, Failed }
    
    struct TaskRequirement {
        string description;
        uint256[] requiredCapabilities;
        uint256[] preferredCapabilities;
        uint256 minReputation;
        uint256 urgency; // 1-10
    }
    
    // Categories for capability organization
    enum CapabilityCategory {
        DataProcessing,
        Communication,
        Analysis,
        Storage,
        Computation,
        Integration,
        Security,
        Monitoring
    }
    
    mapping(uint256 => Capability) public capabilities;
    mapping(address => AgentCapabilityProfile) public agentProfiles;
    mapping(string => uint256) public capabilityNameToId;
    mapping(uint256 => address[]) public capabilityProviders;
    mapping(bytes32 => uint256[]) public intentToCapabilities;
    
    // Task management
    mapping(uint256 => CompositeTask) public compositeTasks;
    mapping(uint256 => Subtask) public subtasks;
    mapping(address => uint256[]) public agentActiveTasks;
    
    // Versioning
    mapping(string => uint256[]) public capabilityVersions; // name => version IDs
    mapping(uint256 => uint256[]) public dependencyGraph; // capability => dependencies
    
    uint256 public capabilityCounter;
    uint256 public taskCounter;
    uint256 public subtaskCounter;
    uint256 public constant MAX_CAPABILITIES_PER_AGENT = 50;
    uint256 public constant PROFICIENCY_THRESHOLD = 70;
    uint256 public constant MAX_WORKLOAD_PER_AGENT = 10;
    
    event CapabilityRegistered(uint256 indexed capabilityId, string name, uint256 version);
    event AgentCapabilityAdded(address indexed agent, uint256 indexed capabilityId);
    event ProficiencyUpdated(address indexed agent, uint256 indexed capabilityId, uint256 score);
    event CapabilityMatched(bytes32 indexed taskHash, address[] agents);
    event CapabilityDeprecated(uint256 indexed capabilityId, uint256 deprecationDate, uint256 replacementId);
    event TaskDecomposed(uint256 indexed taskId, uint256[] subtaskIds);
    event SubtaskAssigned(uint256 indexed subtaskId, address indexed agent, uint256 price);
    event WorkloadUpdated(address indexed agent, uint256 currentWorkload, uint256 maxWorkload);
    event CapabilityPriceSet(address indexed agent, uint256 indexed capabilityId, uint256 price);
    
    function initialize(address _agentRegistry) public initializer {
        __AccessControl_init();
        __UUPSUpgradeable_init();
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(CAPABILITY_ADMIN_ROLE, msg.sender);
        
        agentRegistry = IAgentRegistry(_agentRegistry);
        
        // Register core capabilities
        _registerCoreCapabilities();
    }
    
    /**
     * @notice Register a new capability type with versioning support
     */
    function registerCapability(
        string memory name,
        string memory description,
        string[] memory tags,
        string[] memory inputTypes,
        string[] memory outputTypes,
        CapabilityCategory category,
        uint256 version,
        uint256[] memory dependencies,
        uint256[] memory conflicts
    ) external returns (uint256) {
        // Check if this is a new capability or new version
        uint256[] storage versions = capabilityVersions[name];
        for (uint256 i = 0; i < versions.length; i++) {
            require(capabilities[versions[i]].version != version, "Version exists");
        }
        
        uint256 capabilityId = ++capabilityCounter;
        
        capabilities[capabilityId] = Capability({
            name: name,
            description: description,
            tags: tags,
            inputTypes: inputTypes,
            outputTypes: outputTypes,
            category: uint256(category),
            verified: hasRole(CAPABILITY_ADMIN_ROLE, msg.sender),
            version: version,
            status: CapabilityStatus.Active,
            deprecationDate: 0,
            sunsetDate: 0,
            replacedBy: 0,
            dependencies: dependencies,
            conflicts: conflicts
        });
        
        // Store version mapping
        capabilityVersions[name].push(capabilityId);
        
        // Set as current version if first or explicitly requested
        if (versions.length == 0) {
            capabilityNameToId[name] = capabilityId;
        }
        
        // Validate dependencies
        _validateDependencies(dependencies, conflicts);
        
        // Map common intents to this capability
        _mapIntentsToCapability(capabilityId, tags);
        
        emit CapabilityRegistered(capabilityId, name, version);
        return capabilityId;
    }
    
    /**
     * @notice Deprecate a capability version
     */
    function deprecateCapability(
        uint256 capabilityId,
        uint256 deprecationDate,
        uint256 sunsetDate,
        uint256 replacementId
    ) external onlyRole(CAPABILITY_ADMIN_ROLE) {
        require(capabilityId <= capabilityCounter, "Invalid capability");
        require(deprecationDate >= block.timestamp, "Invalid deprecation date");
        require(sunsetDate > deprecationDate, "Invalid sunset date");
        
        Capability storage capability = capabilities[capabilityId];
        require(capability.status == CapabilityStatus.Active, "Not active");
        
        capability.status = CapabilityStatus.Deprecated;
        capability.deprecationDate = deprecationDate;
        capability.sunsetDate = sunsetDate;
        capability.replacedBy = replacementId;
        
        emit CapabilityDeprecated(capabilityId, deprecationDate, replacementId);
    }
    
    /**
     * @notice Create a composite task that requires decomposition
     */
    function createCompositeTask(
        string memory description,
        uint256[] memory requiredCapabilities,
        TaskComplexity complexity,
        uint256 estimatedDuration
    ) external returns (uint256) {
        uint256 taskId = ++taskCounter;
        
        compositeTasks[taskId] = CompositeTask({
            taskId: taskId,
            description: description,
            requiredCapabilities: requiredCapabilities,
            subtaskIds: new uint256[](0),
            complexity: complexity,
            estimatedDuration: estimatedDuration,
            requester: msg.sender,
            status: TaskStatus.Created
        });
        
        return taskId;
    }
    
    /**
     * @notice Decompose a composite task into subtasks
     */
    function decomposeTask(uint256 taskId) external returns (uint256[] memory) {
        CompositeTask storage task = compositeTasks[taskId];
        require(task.requester == msg.sender, "Not task owner");
        require(task.status == TaskStatus.Created, "Task already processed");
        
        uint256[] memory subtaskIds = new uint256[](task.requiredCapabilities.length);
        
        for (uint256 i = 0; i < task.requiredCapabilities.length; i++) {
            uint256 subtaskId = ++subtaskCounter;
            uint256 priority = _calculateSubtaskPriority(task.complexity, i);
            uint256 deadline = block.timestamp + (task.estimatedDuration / task.requiredCapabilities.length);
            
            subtasks[subtaskId] = Subtask({
                subtaskId: subtaskId,
                parentTaskId: taskId,
                requiredCapability: task.requiredCapabilities[i],
                assignedAgent: address(0),
                priority: priority,
                deadline: deadline,
                status: TaskStatus.Created,
                price: 0
            });
            
            subtaskIds[i] = subtaskId;
        }
        
        task.subtaskIds = subtaskIds;
        task.status = TaskStatus.Assigned;
        
        emit TaskDecomposed(taskId, subtaskIds);
        return subtaskIds;
    }
    
    /**
     * @notice Assign subtasks to agents with workload balancing
     */
    function assignSubtasksWithBalancing(uint256[] memory subtaskIds) external {
        for (uint256 i = 0; i < subtaskIds.length; i++) {
            Subtask storage subtask = subtasks[subtaskIds[i]];
            require(subtask.status == TaskStatus.Created, "Subtask already assigned");
            
            // Find best agent considering workload
            address bestAgent = _findBestAgentForSubtask(subtask);
            require(bestAgent != address(0), "No suitable agent found");
            
            // Check agent's workload capacity
            AgentCapabilityProfile storage profile = agentProfiles[bestAgent];
            require(profile.currentWorkload < profile.maxWorkload, "Agent at capacity");
            
            // Assign subtask
            subtask.assignedAgent = bestAgent;
            subtask.status = TaskStatus.Assigned;
            subtask.price = profile.capabilityPrices[subtask.requiredCapability];
            
            // Update agent workload
            profile.currentWorkload++;
            profile.activeTaskIds.push(subtaskIds[i]);
            agentActiveTasks[bestAgent].push(subtaskIds[i]);
            
            emit SubtaskAssigned(subtaskIds[i], bestAgent, subtask.price);
            emit WorkloadUpdated(bestAgent, profile.currentWorkload, profile.maxWorkload);
        }
    }
    
    /**
     * @notice Set pricing for agent capabilities
     */
    function setCapabilityPrice(uint256 capabilityId, uint256 price) external {
        require(agentRegistry.isRegistered(msg.sender), "Not registered");
        
        AgentCapabilityProfile storage profile = agentProfiles[msg.sender];
        
        // Verify agent has this capability
        bool hasCapability = false;
        for (uint256 i = 0; i < profile.capabilityIds.length; i++) {
            if (profile.capabilityIds[i] == capabilityId) {
                hasCapability = true;
                break;
            }
        }
        require(hasCapability, "Agent doesn't have capability");
        
        profile.capabilityPrices[capabilityId] = price;
        
        emit CapabilityPriceSet(msg.sender, capabilityId, price);
    }
    
    /**
     * @notice Update agent's maximum workload capacity
     */
    function setMaxWorkload(uint256 maxWorkload) external {
        require(agentRegistry.isRegistered(msg.sender), "Not registered");
        require(maxWorkload <= MAX_WORKLOAD_PER_AGENT, "Exceeds maximum");
        require(maxWorkload > 0, "Invalid workload");
        
        AgentCapabilityProfile storage profile = agentProfiles[msg.sender];
        profile.maxWorkload = maxWorkload;
        
        emit WorkloadUpdated(msg.sender, profile.currentWorkload, maxWorkload);
    }
    
    /**
     * @notice Get capability versions for a capability name
     */
    function getCapabilityVersions(string memory name) external view returns (uint256[] memory) {
        return capabilityVersions[name];
    }
    
    /**
     * @notice Get active version of a capability
     */
    function getActiveCapabilityVersion(string memory name) external view returns (uint256) {
        uint256[] memory versions = capabilityVersions[name];
        
        for (uint256 i = versions.length; i > 0; i--) {
            uint256 capId = versions[i - 1];
            if (capabilities[capId].status == CapabilityStatus.Active) {
                return capId;
            }
        }
        
        return 0; // No active version found
    }
    
    /**
     * @notice Get agent's current workload status
     */
    function getAgentWorkload(address agent) external view returns (
        uint256 currentWorkload,
        uint256 maxWorkload,
        uint256[] memory activeTaskIds,
        bool availableForWork
    ) {
        AgentCapabilityProfile storage profile = agentProfiles[agent];
        return (
            profile.currentWorkload,
            profile.maxWorkload,
            profile.activeTaskIds,
            profile.currentWorkload < profile.maxWorkload
        );
    }
    
    /**
     * @notice Add capability to agent profile
     */
    function addAgentCapability(
        uint256 capabilityId,
        uint256 initialProficiency
    ) external {
        require(agentRegistry.isRegistered(msg.sender), "Not registered");
        require(capabilityId > 0 && capabilityId <= capabilityCounter, "Invalid capability");
        require(initialProficiency <= 100, "Invalid proficiency");
        
        AgentCapabilityProfile storage profile = agentProfiles[msg.sender];
        require(profile.capabilityIds.length < MAX_CAPABILITIES_PER_AGENT, "Too many capabilities");
        
        // Check if already has this capability
        for (uint256 i = 0; i < profile.capabilityIds.length; i++) {
            require(profile.capabilityIds[i] != capabilityId, "Already has capability");
        }
        
        profile.capabilityIds.push(capabilityId);
        profile.proficiencyScores[capabilityId] = initialProficiency;
        profile.lastUpdated = block.timestamp;
        
        // Initialize workload if not set
        if (profile.maxWorkload == 0) {
            profile.maxWorkload = 5; // Default max workload
        }
        
        capabilityProviders[capabilityId].push(msg.sender);
        
        emit AgentCapabilityAdded(msg.sender, capabilityId);
    }
    
    /**
     * @notice Match agents to task requirements
     */
    function matchAgentsToTask(
        TaskRequirement memory requirement
    ) external view returns (address[] memory, uint256[] memory) {
        address[] memory matchedAgents = new address[](100);
        uint256[] memory matchScores = new uint256[](100);
        uint256 count = 0;
        
        // Score each agent based on capability match
        for (uint256 i = 0; i < requirement.requiredCapabilities.length; i++) {
            address[] memory providers = capabilityProviders[requirement.requiredCapabilities[i]];
            
            for (uint256 j = 0; j < providers.length && count < 100; j++) {
                address agent = providers[j];
                uint256 score = _calculateMatchScore(agent, requirement);
                
                if (score > 0) {
                    // Insert sorted by score
                    uint256 insertPos = count;
                    for (uint256 k = 0; k < count; k++) {
                        if (score > matchScores[k]) {
                            insertPos = k;
                            break;
                        }
                    }
                    
                    // Shift elements
                    for (uint256 k = count; k > insertPos; k--) {
                        matchedAgents[k] = matchedAgents[k-1];
                        matchScores[k] = matchScores[k-1];
                    }
                    
                    matchedAgents[insertPos] = agent;
                    matchScores[insertPos] = score;
                    count++;
                }
            }
        }
        
        // Return only matched agents
        address[] memory result = new address[](count);
        uint256[] memory scores = new uint256[](count);
        for (uint256 i = 0; i < count; i++) {
            result[i] = matchedAgents[i];
            scores[i] = matchScores[i];
        }
        
        return (result, scores);
    }
    
    /**
     * @notice Resolve intent to capabilities
     */
    function resolveIntent(string memory intent) external view returns (uint256[] memory) {
        bytes32 intentHash = keccak256(bytes(_toLowerCase(intent)));
        return intentToCapabilities[intentHash];
    }
    
    /**
     * @notice Update agent proficiency based on task completion
     */
    function updateProficiency(
        address agent,
        uint256 capabilityId,
        bool success,
        uint256 performanceScore
    ) external {
        require(msg.sender == address(agentRegistry), "Only registry");
        
        AgentCapabilityProfile storage profile = agentProfiles[agent];
        uint256 currentScore = profile.proficiencyScores[capabilityId];
        
        if (success) {
            // Increase proficiency based on performance
            uint256 increase = (performanceScore * 10) / 100; // Max 10 point increase
            profile.proficiencyScores[capabilityId] = _min(currentScore + increase, 100);
            profile.completedTasks[capabilityId]++;
        } else {
            // Decrease proficiency on failure
            uint256 decrease = 5;
            profile.proficiencyScores[capabilityId] = currentScore > decrease ? 
                currentScore - decrease : 0;
        }
        
        emit ProficiencyUpdated(agent, capabilityId, profile.proficiencyScores[capabilityId]);
    }
    
    /**
     * @notice Get agent's capability profile
     */
    function getAgentCapabilities(address agent) external view returns (
        uint256[] memory capabilityIds,
        uint256[] memory proficiencies,
        uint256[] memory tasksCompleted
    ) {
        AgentCapabilityProfile storage profile = agentProfiles[agent];
        uint256 length = profile.capabilityIds.length;
        
        capabilityIds = new uint256[](length);
        proficiencies = new uint256[](length);
        tasksCompleted = new uint256[](length);
        
        for (uint256 i = 0; i < length; i++) {
            uint256 capId = profile.capabilityIds[i];
            capabilityIds[i] = capId;
            proficiencies[i] = profile.proficiencyScores[capId];
            tasksCompleted[i] = profile.completedTasks[capId];
        }
    }
    
    /**
     * @notice Calculate match score for agent-task pairing
     */
    function _calculateMatchScore(
        address agent,
        TaskRequirement memory requirement
    ) private view returns (uint256) {
        AgentCapabilityProfile storage profile = agentProfiles[agent];
        uint256 reputation = agentRegistry.getReputation(agent);
        
        if (reputation < requirement.minReputation) return 0;
        
        uint256 score = 0;
        uint256 requiredMatches = 0;
        
        // Check required capabilities
        for (uint256 i = 0; i < requirement.requiredCapabilities.length; i++) {
            uint256 capId = requirement.requiredCapabilities[i];
            uint256 proficiency = profile.proficiencyScores[capId];
            
            if (proficiency >= PROFICIENCY_THRESHOLD) {
                requiredMatches++;
                score += proficiency;
            }
        }
        
        // Must have all required capabilities
        if (requiredMatches < requirement.requiredCapabilities.length) return 0;
        
        // Bonus for preferred capabilities
        for (uint256 i = 0; i < requirement.preferredCapabilities.length; i++) {
            uint256 capId = requirement.preferredCapabilities[i];
            uint256 proficiency = profile.proficiencyScores[capId];
            score += proficiency / 2; // Half weight for preferred
        }
        
        // Factor in reputation and urgency
        score = (score * reputation) / 100;
        score = (score * requirement.urgency) / 10;
        
        return score;
    }
    
    /**
     * @notice Register core system capabilities
     */
    function _registerCoreCapabilities() private {
        string[] memory empty = new string[](0);
        
        // Data Processing
        _registerCapability("data_transformation", "Transform data formats", 
            _toArray("transform", "convert", "parse"), 
            _toArray("json", "xml", "csv"), 
            _toArray("json", "xml", "csv"), 
            CapabilityCategory.DataProcessing);
            
        // Communication
        _registerCapability("message_routing", "Route messages between agents",
            _toArray("route", "forward", "relay"),
            _toArray("message", "request", "data"),
            _toArray("response", "confirmation", "ack"),
            CapabilityCategory.Communication);
            
        // Analysis
        _registerCapability("data_analysis", "Analyze and interpret data",
            _toArray("analyze", "interpret", "evaluate"),
            _toArray("dataset", "metrics", "data"),
            _toArray("report", "insights", "summary"),
            CapabilityCategory.Analysis);
    }
    
    /**
     * @notice Helper to register capability internally
     */
    function _registerCapability(
        string memory name,
        string memory description,
        string[] memory tags,
        string[] memory inputTypes,
        string[] memory outputTypes,
        CapabilityCategory category
    ) private {
        uint256 capabilityId = ++capabilityCounter;
        uint256[] memory emptyDeps = new uint256[](0);
        
        capabilities[capabilityId] = Capability({
            name: name,
            description: description,
            tags: tags,
            inputTypes: inputTypes,
            outputTypes: outputTypes,
            category: uint256(category),
            verified: true,
            version: 1,
            status: CapabilityStatus.Active,
            deprecationDate: 0,
            sunsetDate: 0,
            replacedBy: 0,
            dependencies: emptyDeps,
            conflicts: emptyDeps
        });
        
        capabilityNameToId[name] = capabilityId;
        capabilityVersions[name].push(capabilityId);
        _mapIntentsToCapability(capabilityId, tags);
    }
    
    /**
     * @notice Validate capability dependencies and conflicts
     */
    function _validateDependencies(uint256[] memory dependencies, uint256[] memory conflicts) private view {
        // Check dependencies exist and are active
        for (uint256 i = 0; i < dependencies.length; i++) {
            require(dependencies[i] <= capabilityCounter, "Invalid dependency");
            require(capabilities[dependencies[i]].status == CapabilityStatus.Active, "Inactive dependency");
        }
        
        // Check conflicts don't overlap with dependencies
        for (uint256 i = 0; i < conflicts.length; i++) {
            require(conflicts[i] <= capabilityCounter, "Invalid conflict");
            for (uint256 j = 0; j < dependencies.length; j++) {
                require(conflicts[i] != dependencies[j], "Dependency conflict");
            }
        }
    }
    
    /**
     * @notice Calculate subtask priority based on complexity and position
     */
    function _calculateSubtaskPriority(TaskComplexity complexity, uint256 position) private pure returns (uint256) {
        uint256 baseScore = 50; // Medium priority
        
        if (complexity == TaskComplexity.Simple) {
            baseScore = 30;
        } else if (complexity == TaskComplexity.Moderate) {
            baseScore = 50;
        } else if (complexity == TaskComplexity.Complex) {
            baseScore = 70;
        } else if (complexity == TaskComplexity.ExtraComplex) {
            baseScore = 90;
        }
        
        // Earlier subtasks get higher priority
        return baseScore + (10 - (position % 10));
    }
    
    /**
     * @notice Find best agent for subtask considering capability and workload
     */
    function _findBestAgentForSubtask(Subtask memory subtask) private view returns (address) {
        address[] memory providers = capabilityProviders[subtask.requiredCapability];
        address bestAgent = address(0);
        uint256 bestScore = 0;
        
        for (uint256 i = 0; i < providers.length; i++) {
            address agent = providers[i];
            AgentCapabilityProfile storage profile = agentProfiles[agent];
            
            // Skip if agent is at capacity
            if (profile.currentWorkload >= profile.maxWorkload) continue;
            
            // Calculate score based on proficiency and availability
            uint256 proficiency = profile.proficiencyScores[subtask.requiredCapability];
            if (proficiency < PROFICIENCY_THRESHOLD) continue;
            
            uint256 availability = ((profile.maxWorkload - profile.currentWorkload) * 100) / profile.maxWorkload;
            uint256 score = (proficiency * 60 + availability * 40) / 100;
            
            if (score > bestScore) {
                bestScore = score;
                bestAgent = agent;
            }
        }
        
        return bestAgent;
    }
    
    /**
     * @notice Complete a subtask and update agent workload
     */
    function completeSubtask(uint256 subtaskId, bool success) external {
        Subtask storage subtask = subtasks[subtaskId];
        require(subtask.assignedAgent == msg.sender, "Not assigned agent");
        require(subtask.status == TaskStatus.Assigned || subtask.status == TaskStatus.InProgress, "Invalid status");
        
        // Update subtask status
        subtask.status = success ? TaskStatus.Completed : TaskStatus.Failed;
        
        // Update agent workload
        AgentCapabilityProfile storage profile = agentProfiles[msg.sender];
        profile.currentWorkload--;
        
        // Remove from active tasks
        for (uint256 i = 0; i < profile.activeTaskIds.length; i++) {
            if (profile.activeTaskIds[i] == subtaskId) {
                profile.activeTaskIds[i] = profile.activeTaskIds[profile.activeTaskIds.length - 1];
                profile.activeTaskIds.pop();
                break;
            }
        }
        
        // Remove from global active tasks
        uint256[] storage globalTasks = agentActiveTasks[msg.sender];
        for (uint256 i = 0; i < globalTasks.length; i++) {
            if (globalTasks[i] == subtaskId) {
                globalTasks[i] = globalTasks[globalTasks.length - 1];
                globalTasks.pop();
                break;
            }
        }
        
        emit WorkloadUpdated(msg.sender, profile.currentWorkload, profile.maxWorkload);
    }
    
    /**
     * @notice Get marketplace pricing for a capability
     */
    function getCapabilityMarketPricing(uint256 capabilityId) external view returns (
        uint256 minPrice,
        uint256 maxPrice,
        uint256 avgPrice,
        uint256 providerCount
    ) {
        address[] memory providers = capabilityProviders[capabilityId];
        uint256 totalPrice = 0;
        uint256 validProviders = 0;
        minPrice = type(uint256).max;
        maxPrice = 0;
        
        for (uint256 i = 0; i < providers.length; i++) {
            uint256 price = agentProfiles[providers[i]].capabilityPrices[capabilityId];
            if (price > 0) {
                totalPrice += price;
                validProviders++;
                if (price < minPrice) minPrice = price;
                if (price > maxPrice) maxPrice = price;
            }
        }
        
        avgPrice = validProviders > 0 ? totalPrice / validProviders : 0;
        providerCount = validProviders;
        
        if (validProviders == 0) {
            minPrice = 0;
        }
    }
    
    /**
     * @notice Map intents to capability
     */
    function _mapIntentsToCapability(uint256 capabilityId, string[] memory tags) private {
        for (uint256 i = 0; i < tags.length; i++) {
            bytes32 intentHash = keccak256(bytes(_toLowerCase(tags[i])));
            intentToCapabilities[intentHash].push(capabilityId);
        }
    }
    
    function _toLowerCase(string memory str) private pure returns (string memory) {
        bytes memory bStr = bytes(str);
        bytes memory bLower = new bytes(bStr.length);
        for (uint256 i = 0; i < bStr.length; i++) {
            if ((bStr[i] >= 0x41) && (bStr[i] <= 0x5A)) {
                bLower[i] = bytes1(uint8(bStr[i]) + 32);
            } else {
                bLower[i] = bStr[i];
            }
        }
        return string(bLower);
    }
    
    function _toArray(string memory a, string memory b, string memory c) 
        private pure returns (string[] memory) {
        string[] memory arr = new string[](3);
        arr[0] = a;
        arr[1] = b;
        arr[2] = c;
        return arr;
    }
    
    function _min(uint256 a, uint256 b) private pure returns (uint256) {
        return a < b ? a : b;
    }
    
    function _authorizeUpgrade(address newImplementation) internal override onlyRole(DEFAULT_ADMIN_ROLE) {}
}