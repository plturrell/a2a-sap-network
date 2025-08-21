// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../AgentRegistry.sol";
import "../MessageRouter.sol";
import "../MultiSigPausable.sol";
import "./ProtocolBridge.sol";
import "./IdentityBridge.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title UnifiedProtocolRouter
 * @dev Main router contract that orchestrates message routing and discovery
 * across A2A, ANP, and ACP protocols. Provides a single interface for
 * multi-protocol agent interactions with automatic protocol selection.
 */
contract UnifiedProtocolRouter is MultiSigPausable, ReentrancyGuard {
    AgentRegistry public immutable registry;
    MessageRouter public immutable messageRouter;
    ProtocolBridge public immutable protocolBridge;
    IdentityBridge public immutable identityBridge;

    enum RoutingStrategy { AUTO, EXPLICIT, CAPABILITY_BASED, REPUTATION_BASED }

    struct RoutingDecision {
        ProtocolBridge.ProtocolType selectedProtocol;
        address targetAgent;
        string externalAgent;
        uint256 confidence;
        string reasoning;
    }

    struct UnifiedTask {
        bytes32 taskId;
        address requester;
        string capability;
        string content;
        bytes32 messageType;
        RoutingStrategy strategy;
        ProtocolBridge.ProtocolType[] allowedProtocols;
        uint256 maxBudget;
        uint256 deadline;
        bool completed;
    }

    struct CrossProtocolResponse {
        bytes32 taskId;
        bytes32 messageId;
        ProtocolBridge.ProtocolType usedProtocol;
        string response;
        uint256 completionTime;
        bool success;
    }

    // Storage
    mapping(bytes32 => UnifiedTask) public tasks;
    mapping(bytes32 => CrossProtocolResponse) public responses;
    mapping(bytes32 => RoutingDecision) public routingDecisions;
    
    // Protocol performance tracking
    mapping(ProtocolBridge.ProtocolType => uint256) public protocolSuccessCount;
    mapping(ProtocolBridge.ProtocolType => uint256) public protocolTotalCount;
    mapping(ProtocolBridge.ProtocolType => uint256) public protocolAvgResponseTime;

    // Routing configuration
    uint256 public defaultTimeout = 300; // 5 minutes
    uint256 public maxRetries = 3;
    mapping(bytes32 => uint256) public capabilityWeights;

    // Events
    event TaskCreated(bytes32 indexed taskId, address requester, string capability);
    event RoutingDecisionMade(bytes32 indexed taskId, ProtocolBridge.ProtocolType protocol, uint256 confidence);
    event TaskCompleted(bytes32 indexed taskId, bool success, uint256 responseTime);
    event ProtocolPerformanceUpdated(ProtocolBridge.ProtocolType protocol, uint256 successRate);
    event RoutingStrategyUpdated(RoutingStrategy newStrategy);

    constructor(
        address _registry,
        address _messageRouter,
        address _protocolBridge,
        address _identityBridge,
        uint256 _requiredConfirmations
    ) MultiSigPausable(_requiredConfirmations) {
        registry = AgentRegistry(_registry);
        messageRouter = MessageRouter(_messageRouter);
        protocolBridge = ProtocolBridge(_protocolBridge);
        identityBridge = IdentityBridge(_identityBridge);
        
        // Initialize default capability weights
        _initializeCapabilityWeights();
    }

    modifier onlyRegisteredAgent() {
        AgentRegistry.Agent memory agent = registry.getAgent(msg.sender);
        require(agent.active, "Agent not registered or inactive");
        _;
    }

    /**
     * @notice Submit a unified task that can be routed to any protocol
     * @param capability Required capability identifier
     * @param content Task content/description
     * @param messageType Message type identifier
     * @param strategy Routing strategy to use
     * @param allowedProtocols Array of allowed protocols (empty = all allowed)
     * @param maxBudget Maximum budget for the task
     * @param deadline Task deadline timestamp
     * @return taskId Unique task identifier
     */
    function submitUnifiedTask(
        string memory capability,
        string memory content,
        bytes32 messageType,
        RoutingStrategy strategy,
        ProtocolBridge.ProtocolType[] memory allowedProtocols,
        uint256 maxBudget,
        uint256 deadline
    ) external onlyRegisteredAgent whenNotPaused nonReentrant returns (bytes32) {
        require(bytes(capability).length > 0, "Capability required");
        require(bytes(content).length > 0, "Content required");
        require(deadline > block.timestamp, "Deadline must be in future");

        bytes32 taskId = keccak256(
            abi.encodePacked(msg.sender, capability, content, block.timestamp)
        );

        UnifiedTask storage task = tasks[taskId];
        task.taskId = taskId;
        task.requester = msg.sender;
        task.capability = capability;
        task.content = content;
        task.messageType = messageType;
        task.strategy = strategy;
        task.allowedProtocols = allowedProtocols;
        task.maxBudget = maxBudget;
        task.deadline = deadline;
        task.completed = false;

        emit TaskCreated(taskId, msg.sender, capability);

        // Immediately process routing decision
        _processRoutingDecision(taskId);

        return taskId;
    }

    /**
     * @notice Get optimal routing decision for a task
     * @param taskId Task identifier
     * @return RoutingDecision structure with routing details
     */
    function getRoutingDecision(bytes32 taskId) 
        external 
        view 
        returns (RoutingDecision memory) 
    {
        return routingDecisions[taskId];
    }

    /**
     * @notice Execute a task using the determined routing
     * @param taskId Task identifier
     * @return messageId Generated message ID for tracking
     */
    function executeTask(bytes32 taskId) 
        external 
        onlyRegisteredAgent 
        whenNotPaused 
        nonReentrant 
        returns (bytes32) 
    {
        UnifiedTask storage task = tasks[taskId];
        require(task.requester == msg.sender, "Not task owner");
        require(!task.completed, "Task already completed");
        require(block.timestamp <= task.deadline, "Task deadline passed");

        RoutingDecision memory decision = routingDecisions[taskId];
        require(decision.confidence > 0, "No routing decision available");

        bytes32 messageId;

        if (decision.selectedProtocol == ProtocolBridge.ProtocolType.A2A) {
            // Route via A2A MessageRouter
            messageId = messageRouter.sendMessage(
                decision.targetAgent,
                task.content,
                task.messageType
            );
        } else {
            // Route via external protocol bridge
            messageId = protocolBridge.bridgeMessageToExternal(
                bytes32(0), // Will be generated by bridge
                decision.selectedProtocol,
                decision.externalAgent
            );
        }

        // Update protocol statistics
        protocolTotalCount[decision.selectedProtocol]++;

        // Store response tracking
        responses[taskId] = CrossProtocolResponse({
            taskId: taskId,
            messageId: messageId,
            usedProtocol: decision.selectedProtocol,
            response: "",
            completionTime: 0,
            success: false
        });

        return messageId;
    }

    /**
     * @notice Complete a task with response (called by protocol adapters)
     * @param taskId Task identifier
     * @param response Response content
     * @param success Whether the task was successful
     */
    function completeTask(
        bytes32 taskId,
        string memory response,
        bool success
    ) external onlyRole(DEFAULT_ADMIN_ROLE) nonReentrant {
        UnifiedTask storage task = tasks[taskId];
        require(!task.completed, "Task already completed");

        CrossProtocolResponse storage resp = responses[taskId];
        resp.response = response;
        resp.completionTime = block.timestamp;
        resp.success = success;

        task.completed = true;

        // Update protocol performance metrics
        if (success) {
            protocolSuccessCount[resp.usedProtocol]++;
        }

        uint256 responseTime = block.timestamp - task.deadline; // Simplified
        _updateProtocolMetrics(resp.usedProtocol, responseTime);

        emit TaskCompleted(taskId, success, responseTime);
    }

    /**
     * @notice Get unified discovery results across all protocols
     * @param capability Capability to search for
     * @param maxResults Maximum results to return
     * @param protocols Specific protocols to search (empty = all)
     * @return a2aAgents A2A agent addresses
     * @return externalAgents External agent identifiers
     * @return protocolTypes Protocol types for each external agent
     */
    function unifiedDiscovery(
        string memory capability,
        uint256 maxResults,
        ProtocolBridge.ProtocolType[] memory protocols
    ) external view returns (
        address[] memory a2aAgents,
        string[] memory externalAgents,
        ProtocolBridge.ProtocolType[] memory protocolTypes
    ) {
        bytes32 capabilityHash = keccak256(abi.encodePacked(capability));
        
        // Use ProtocolBridge for cross-protocol discovery
        (address[] memory a2aResults, string[] memory externalResults) = 
            protocolBridge.crossProtocolDiscovery(capabilityHash, protocols, maxResults);

        // Limit results
        uint256 totalResults = a2aResults.length + externalResults.length;
        uint256 resultCount = totalResults > maxResults ? maxResults : totalResults;

        if (resultCount == 0) {
            return (new address[](0), new string[](0), new ProtocolBridge.ProtocolType[](0));
        }

        a2aAgents = a2aResults;
        externalAgents = externalResults;
        
        // Create protocol type array (simplified - would need actual protocol detection)
        protocolTypes = new ProtocolBridge.ProtocolType[](externalResults.length);
        for (uint256 i = 0; i < externalResults.length; i++) {
            protocolTypes[i] = ProtocolBridge.ProtocolType.ANP; // Default to ANP
        }

        return (a2aAgents, externalAgents, protocolTypes);
    }

    /**
     * @notice Update routing configuration
     * @param capability Capability identifier
     * @param weight Weight for routing decisions
     */
    function updateCapabilityWeight(
        bytes32 capability,
        uint256 weight
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        capabilityWeights[capability] = weight;
    }

    /**
     * @notice Get protocol performance statistics
     * @param protocol Protocol type
     * @return successRate Success rate (percentage * 100)
     * @return avgResponseTime Average response time
     * @return totalRequests Total number of requests
     */
    function getProtocolStats(ProtocolBridge.ProtocolType protocol) 
        external 
        view 
        returns (uint256 successRate, uint256 avgResponseTime, uint256 totalRequests) 
    {
        totalRequests = protocolTotalCount[protocol];
        if (totalRequests == 0) {
            return (0, 0, 0);
        }

        successRate = (protocolSuccessCount[protocol] * 10000) / totalRequests; // Basis points
        avgResponseTime = protocolAvgResponseTime[protocol];
        
        return (successRate, avgResponseTime, totalRequests);
    }

    /**
     * @dev Process routing decision for a task
     */
    function _processRoutingDecision(bytes32 taskId) internal {
        UnifiedTask storage task = tasks[taskId];
        RoutingDecision memory decision;

        if (task.strategy == RoutingStrategy.AUTO) {
            decision = _autoSelectProtocol(task);
        } else if (task.strategy == RoutingStrategy.CAPABILITY_BASED) {
            decision = _capabilityBasedRouting(task);
        } else if (task.strategy == RoutingStrategy.REPUTATION_BASED) {
            decision = _reputationBasedRouting(task);
        } else {
            // EXPLICIT strategy - use first allowed protocol
            decision.selectedProtocol = task.allowedProtocols.length > 0 ? 
                task.allowedProtocols[0] : ProtocolBridge.ProtocolType.A2A;
            decision.confidence = 100;
            decision.reasoning = "Explicit protocol selection";
        }

        routingDecisions[taskId] = decision;
        emit RoutingDecisionMade(taskId, decision.selectedProtocol, decision.confidence);
    }

    /**
     * @dev Auto-select optimal protocol based on performance and availability
     */
    function _autoSelectProtocol(UnifiedTask memory task) 
        internal 
        view 
        returns (RoutingDecision memory) 
    {
        RoutingDecision memory decision;
        uint256 bestScore = 0;
        
        // Check A2A protocol first
        bytes32 capabilityHash = keccak256(abi.encodePacked(task.capability));
        address[] memory a2aAgents = registry.findAgentsByCapability(capabilityHash);
        
        if (a2aAgents.length > 0) {
            uint256 a2aScore = _calculateProtocolScore(ProtocolBridge.ProtocolType.A2A);
            if (a2aScore > bestScore) {
                bestScore = a2aScore;
                decision.selectedProtocol = ProtocolBridge.ProtocolType.A2A;
                decision.targetAgent = a2aAgents[0]; // Select first available
                decision.confidence = a2aScore;
                decision.reasoning = "A2A protocol selected based on performance";
            }
        }

        // Check external protocols if no A2A agent or if external protocols score higher
        for (uint256 i = 0; i < task.allowedProtocols.length; i++) {
            if (task.allowedProtocols[i] != ProtocolBridge.ProtocolType.A2A) {
                uint256 protocolScore = _calculateProtocolScore(task.allowedProtocols[i]);
                if (protocolScore > bestScore) {
                    bestScore = protocolScore;
                    decision.selectedProtocol = task.allowedProtocols[i];
                    decision.externalAgent = "external-agent-placeholder";
                    decision.confidence = protocolScore;
                    decision.reasoning = "External protocol selected based on performance";
                }
            }
        }

        if (bestScore == 0) {
            // Fallback to A2A
            decision.selectedProtocol = ProtocolBridge.ProtocolType.A2A;
            decision.confidence = 50;
            decision.reasoning = "Fallback to A2A protocol";
        }

        return decision;
    }

    /**
     * @dev Capability-based routing selection
     */
    function _capabilityBasedRouting(UnifiedTask memory task) 
        internal 
        view 
        returns (RoutingDecision memory) 
    {
        RoutingDecision memory decision;
        bytes32 capabilityHash = keccak256(abi.encodePacked(task.capability));
        
        // Check if we have specific weights for this capability
        uint256 weight = capabilityWeights[capabilityHash];
        
        // Find agents with the specific capability
        address[] memory a2aAgents = registry.findAgentsByCapability(capabilityHash);
        
        if (a2aAgents.length > 0 && weight >= 75) {
            // High weight capabilities prefer A2A
            decision.selectedProtocol = ProtocolBridge.ProtocolType.A2A;
            decision.targetAgent = _selectBestA2AAgent(a2aAgents);
            decision.confidence = weight;
            decision.reasoning = "High-weight capability routed to A2A";
        } else {
            // Lower weight or no A2A agents - consider external protocols
            decision = _autoSelectProtocol(task);
        }

        return decision;
    }

    /**
     * @dev Reputation-based routing selection
     */
    function _reputationBasedRouting(UnifiedTask memory task) 
        internal 
        view 
        returns (RoutingDecision memory) 
    {
        RoutingDecision memory decision;
        bytes32 capabilityHash = keccak256(abi.encodePacked(task.capability));
        address[] memory a2aAgents = registry.findAgentsByCapability(capabilityHash);
        
        if (a2aAgents.length > 0) {
            address bestAgent = _selectBestA2AAgent(a2aAgents);
            AgentRegistry.Agent memory agent = registry.getAgent(bestAgent);
            
            if (agent.reputation >= 150) {
                // High reputation agents get priority in A2A
                decision.selectedProtocol = ProtocolBridge.ProtocolType.A2A;
                decision.targetAgent = bestAgent;
                decision.confidence = agent.reputation / 2; // Convert to percentage-like score
                decision.reasoning = "High reputation A2A agent selected";
            } else {
                // Lower reputation - consider external protocols
                decision = _autoSelectProtocol(task);
            }
        } else {
            // No A2A agents available
            decision = _autoSelectProtocol(task);
        }

        return decision;
    }

    /**
     * @dev Calculate protocol performance score
     */
    function _calculateProtocolScore(ProtocolBridge.ProtocolType protocol) 
        internal 
        view 
        returns (uint256) 
    {
        uint256 totalCount = protocolTotalCount[protocol];
        if (totalCount == 0) {
            return 75; // Default score for untested protocols
        }

        uint256 successCount = protocolSuccessCount[protocol];
        uint256 successRate = (successCount * 100) / totalCount;
        uint256 responseTime = protocolAvgResponseTime[protocol];
        
        // Score based on success rate (60%) and response time (40%)
        uint256 timeScore = responseTime > 0 ? (300 / responseTime) * 40 : 40; // Inverse relationship
        uint256 score = (successRate * 60 / 100) + (timeScore > 40 ? 40 : timeScore);
        
        return score > 100 ? 100 : score;
    }

    /**
     * @dev Select best A2A agent based on reputation
     */
    function _selectBestA2AAgent(address[] memory agents) 
        internal 
        view 
        returns (address) 
    {
        address bestAgent = agents[0];
        uint256 bestReputation = registry.getAgent(agents[0]).reputation;
        
        for (uint256 i = 1; i < agents.length; i++) {
            uint256 reputation = registry.getAgent(agents[i]).reputation;
            if (reputation > bestReputation) {
                bestReputation = reputation;
                bestAgent = agents[i];
            }
        }
        
        return bestAgent;
    }

    /**
     * @dev Update protocol performance metrics
     */
    function _updateProtocolMetrics(
        ProtocolBridge.ProtocolType protocol,
        uint256 responseTime
    ) internal {
        uint256 currentAvg = protocolAvgResponseTime[protocol];
        uint256 totalCount = protocolTotalCount[protocol];
        
        if (currentAvg == 0) {
            protocolAvgResponseTime[protocol] = responseTime;
        } else {
            protocolAvgResponseTime[protocol] = 
                ((currentAvg * (totalCount - 1)) + responseTime) / totalCount;
        }

        emit ProtocolPerformanceUpdated(protocol, _calculateProtocolScore(protocol));
    }

    /**
     * @dev Initialize default capability weights
     */
    function _initializeCapabilityWeights() internal {
        // Set default weights for common capabilities
        capabilityWeights[keccak256("data_analysis")] = 85;
        capabilityWeights[keccak256("text_processing")] = 80;
        capabilityWeights[keccak256("image_processing")] = 70;
        capabilityWeights[keccak256("blockchain_query")] = 90;
        capabilityWeights[keccak256("web_search")] = 60;
    }
}