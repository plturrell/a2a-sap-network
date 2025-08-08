// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./AgentRegistry.sol";
import "./MessageRouter.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";

/**
 * @title BusinessDataCloudA2A
 * @dev Comprehensive smart contract for Business Data Cloud A2A integration
 * Manages all agents, workflows, trust relationships, and cross-agent communication
 */
contract BusinessDataCloudA2A is AccessControl, ReentrancyGuard {
    using EnumerableSet for EnumerableSet.AddressSet;
    using EnumerableSet for EnumerableSet.Bytes32Set;

    // A2A Protocol Version
    string public constant PROTOCOL_VERSION = "0.2.9";
    string public constant CONTRACT_VERSION = "1.0.0";
    
    // Role definitions
    bytes32 public constant AGENT_MANAGER_ROLE = keccak256("AGENT_MANAGER_ROLE");
    bytes32 public constant WORKFLOW_EXECUTOR_ROLE = keccak256("WORKFLOW_EXECUTOR_ROLE");
    bytes32 public constant TRUST_MANAGER_ROLE = keccak256("TRUST_MANAGER_ROLE");

    // Core A2A Agent Types
    enum AgentType {
        DATA_PRODUCT_REGISTRATION,    // Agent 0
        DATA_STANDARDIZATION,         // Agent 1  
        AI_PREPARATION,               // Agent 2
        VECTOR_PROCESSING,            // Agent 3
        CALC_VALIDATION,              // Agent 4
        QA_VALIDATION,                // Agent 5
        DATA_MANAGER,                 // Supporting
        CATALOG_MANAGER,              // Supporting
        AGENT_MANAGER                 // Supporting
    }

    // Agent Information Structure
    struct A2AAgent {
        address agentAddress;
        string agentId;
        string name;
        AgentType agentType;
        string endpoint;
        bytes32[] capabilities;
        bytes32[] skills;
        uint256 trustScore;
        bool active;
        uint256 registeredAt;
        uint256 lastHeartbeat;
        mapping(string => string) metadata;
    }

    // Workflow Structure
    struct A2AWorkflow {
        string workflowId;
        string name;
        string description;
        bytes32[] requiredAgents;
        mapping(uint256 => string) steps;
        uint256 totalSteps;
        bool active;
        uint256 createdAt;
        address createdBy;
    }

    // Trust Relationship Structure
    struct TrustRelationship {
        address agent1;
        address agent2;
        uint256 trustLevel; // 0-100
        bytes32[] sharedCapabilities;
        uint256 establishedAt;
        uint256 lastInteraction;
        bool active;
    }

    // Task Execution Structure
    struct TaskExecution {
        string taskId;
        string workflowId;
        address[] participatingAgents;
        mapping(address => string) agentResults;
        mapping(uint256 => bool) stepCompleted;
        uint256 currentStep;
        uint256 totalSteps;
        uint256 startedAt;
        uint256 completedAt;
        bool completed;
        bool successful;
    }

    // Storage
    mapping(address => A2AAgent) public agents;
    mapping(string => A2AWorkflow) public workflows;
    mapping(bytes32 => TrustRelationship) public trustRelationships;
    mapping(string => TaskExecution) public taskExecutions;
    
    // Agent tracking
    mapping(AgentType => EnumerableSet.AddressSet) private agentsByType;
    mapping(bytes32 => EnumerableSet.AddressSet) private agentsByCapability;
    EnumerableSet.AddressSet private allAgents;
    EnumerableSet.Bytes32Set private allWorkflows;

    // Trust network
    mapping(address => EnumerableSet.AddressSet) private trustedAgents;
    mapping(address => uint256) public agentReputations;

    // Events
    event A2AAgentRegistered(
        address indexed agentAddress,
        string agentId,
        AgentType agentType,
        string endpoint
    );
    
    event WorkflowCreated(
        string indexed workflowId,
        string name,
        address indexed creator
    );
    
    event TaskExecutionStarted(
        string indexed taskId,
        string indexed workflowId,
        address[] participatingAgents
    );
    
    event TaskExecutionCompleted(
        string indexed taskId,
        bool successful,
        uint256 executionTime
    );
    
    event TrustRelationshipEstablished(
        address indexed agent1,
        address indexed agent2,
        uint256 trustLevel
    );
    
    event AgentHeartbeat(
        address indexed agentAddress,
        uint256 timestamp,
        string status
    );

    event CrossAgentCommunication(
        address indexed fromAgent,
        address indexed toAgent,
        string messageType,
        bytes32 messageHash
    );

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(AGENT_MANAGER_ROLE, msg.sender);
        _grantRole(WORKFLOW_EXECUTOR_ROLE, msg.sender);
        _grantRole(TRUST_MANAGER_ROLE, msg.sender);
    }

    /**
     * @dev Register A2A agent with the Business Data Cloud
     */
    function registerA2AAgent(
        string memory agentId,
        string memory name,
        AgentType agentType,
        string memory endpoint,
        bytes32[] memory capabilities,
        bytes32[] memory skills
    ) external nonReentrant {
        require(bytes(agentId).length > 0, "Agent ID required");
        require(bytes(name).length > 0, "Agent name required");
        require(bytes(endpoint).length > 0, "Agent endpoint required");
        require(!allAgents.contains(msg.sender), "Agent already registered");

        // Initialize agent
        A2AAgent storage agent = agents[msg.sender];
        agent.agentAddress = msg.sender;
        agent.agentId = agentId;
        agent.name = name;
        agent.agentType = agentType;
        agent.endpoint = endpoint;
        agent.capabilities = capabilities;
        agent.skills = skills;
        agent.trustScore = 100; // Default trust score
        agent.active = true;
        agent.registeredAt = block.timestamp;
        agent.lastHeartbeat = block.timestamp;

        // Add to tracking sets
        allAgents.add(msg.sender);
        agentsByType[agentType].add(msg.sender);
        
        // Add to capability mapping
        for (uint256 i = 0; i < capabilities.length; i++) {
            agentsByCapability[capabilities[i]].add(msg.sender);
        }

        // Initialize reputation
        agentReputations[msg.sender] = 100;

        emit A2AAgentRegistered(msg.sender, agentId, agentType, endpoint);
    }

    /**
     * @dev Create A2A workflow for Business Data Cloud processing
     */
    function createA2AWorkflow(
        string memory workflowId,
        string memory name,
        string memory description,
        bytes32[] memory requiredAgents,
        string[] memory steps
    ) external onlyRole(WORKFLOW_EXECUTOR_ROLE) {
        require(bytes(workflowId).length > 0, "Workflow ID required");
        require(!allWorkflows.contains(bytes32(bytes(workflowId))), "Workflow already exists");
        require(steps.length > 0, "At least one step required");

        A2AWorkflow storage workflow = workflows[workflowId];
        workflow.workflowId = workflowId;
        workflow.name = name;
        workflow.description = description;
        workflow.requiredAgents = requiredAgents;
        workflow.totalSteps = steps.length;
        workflow.active = true;
        workflow.createdAt = block.timestamp;
        workflow.createdBy = msg.sender;

        // Store steps
        for (uint256 i = 0; i < steps.length; i++) {
            workflow.steps[i] = steps[i];
        }

        allWorkflows.add(bytes32(bytes(workflowId)));
        emit WorkflowCreated(workflowId, name, msg.sender);
    }

    /**
     * @dev Establish trust relationship between agents
     */
    function establishTrustRelationship(
        address agent1,
        address agent2,
        uint256 trustLevel,
        bytes32[] memory sharedCapabilities
    ) external onlyRole(TRUST_MANAGER_ROLE) {
        require(allAgents.contains(agent1), "Agent1 not registered");
        require(allAgents.contains(agent2), "Agent2 not registered");
        require(agent1 != agent2, "Cannot establish trust with self");
        require(trustLevel <= 100, "Trust level must be 0-100");

        bytes32 relationshipId = keccak256(abi.encodePacked(agent1, agent2));
        
        TrustRelationship storage relationship = trustRelationships[relationshipId];
        relationship.agent1 = agent1;
        relationship.agent2 = agent2;
        relationship.trustLevel = trustLevel;
        relationship.sharedCapabilities = sharedCapabilities;
        relationship.establishedAt = block.timestamp;
        relationship.lastInteraction = block.timestamp;
        relationship.active = true;

        // Add to trusted agents mapping (bidirectional)
        trustedAgents[agent1].add(agent2);
        trustedAgents[agent2].add(agent1);

        emit TrustRelationshipEstablished(agent1, agent2, trustLevel);
    }

    /**
     * @dev Execute A2A task across multiple agents
     */
    function executeA2ATask(
        string memory taskId,
        string memory workflowId,
        address[] memory participatingAgents
    ) external onlyRole(WORKFLOW_EXECUTOR_ROLE) nonReentrant {
        require(bytes(taskId).length > 0, "Task ID required");
        require(allWorkflows.contains(bytes32(bytes(workflowId))), "Workflow not found");
        require(participatingAgents.length > 0, "No participating agents");

        // Verify all agents are registered and trusted
        for (uint256 i = 0; i < participatingAgents.length; i++) {
            require(allAgents.contains(participatingAgents[i]), "Agent not registered");
            require(agents[participatingAgents[i]].active, "Agent not active");
        }

        A2AWorkflow storage workflow = workflows[workflowId];
        
        TaskExecution storage execution = taskExecutions[taskId];
        execution.taskId = taskId;
        execution.workflowId = workflowId;
        execution.participatingAgents = participatingAgents;
        execution.currentStep = 0;
        execution.totalSteps = workflow.totalSteps;
        execution.startedAt = block.timestamp;
        execution.completed = false;
        execution.successful = false;

        emit TaskExecutionStarted(taskId, workflowId, participatingAgents);
    }

    /**
     * @dev Report agent result for task step
     */
    function reportAgentResult(
        string memory taskId,
        uint256 stepNumber,
        string memory result
    ) external {
        require(allAgents.contains(msg.sender), "Agent not registered");
        require(bytes(taskExecutions[taskId].taskId).length > 0, "Task not found");
        require(!taskExecutions[taskId].completed, "Task already completed");

        TaskExecution storage execution = taskExecutions[taskId];
        execution.agentResults[msg.sender] = result;
        execution.stepCompleted[stepNumber] = true;

        // Check if we can advance to next step
        _checkStepCompletion(taskId);
    }

    /**
     * @dev Send heartbeat to maintain agent status
     */
    function sendHeartbeat(string memory status) external {
        require(allAgents.contains(msg.sender), "Agent not registered");
        
        agents[msg.sender].lastHeartbeat = block.timestamp;
        emit AgentHeartbeat(msg.sender, block.timestamp, status);
    }

    /**
     * @dev Cross-agent communication
     */
    function sendCrossAgentMessage(
        address toAgent,
        string memory messageType,
        bytes memory messageData
    ) external {
        require(allAgents.contains(msg.sender), "Sender not registered");
        require(allAgents.contains(toAgent), "Recipient not registered");
        require(_canCommunicate(msg.sender, toAgent), "Communication not allowed");

        bytes32 messageHash = keccak256(messageData);
        emit CrossAgentCommunication(msg.sender, toAgent, messageType, messageHash);
    }

    /**
     * @dev Get agents by type
     */
    function getAgentsByType(AgentType agentType) external view returns (address[] memory) {
        uint256 length = agentsByType[agentType].length();
        address[] memory agentList = new address[](length);
        
        for (uint256 i = 0; i < length; i++) {
            agentList[i] = agentsByType[agentType].at(i);
        }
        
        return agentList;
    }

    /**
     * @dev Get agents by capability
     */
    function getAgentsByCapability(bytes32 capability) external view returns (address[] memory) {
        uint256 length = agentsByCapability[capability].length();
        address[] memory agentList = new address[](length);
        
        for (uint256 i = 0; i < length; i++) {
            agentList[i] = agentsByCapability[capability].at(i);
        }
        
        return agentList;
    }

    /**
     * @dev Get trusted agents for a specific agent
     */
    function getTrustedAgents(address agent) external view returns (address[] memory) {
        uint256 length = trustedAgents[agent].length();
        address[] memory trustedList = new address[](length);
        
        for (uint256 i = 0; i < length; i++) {
            trustedList[i] = trustedAgents[agent].at(i);
        }
        
        return trustedList;
    }

    /**
     * @dev Get task execution status
     */
    function getTaskExecutionStatus(string memory taskId) external view returns (
        string memory workflowId,
        uint256 currentStep,
        uint256 totalSteps,
        bool completed,
        bool successful,
        uint256 startedAt,
        uint256 completedAt
    ) {
        TaskExecution storage execution = taskExecutions[taskId];
        return (
            execution.workflowId,
            execution.currentStep,
            execution.totalSteps,
            execution.completed,
            execution.successful,
            execution.startedAt,
            execution.completedAt
        );
    }

    /**
     * @dev Get Business Data Cloud statistics
     */
    function getBDCStatistics() external view returns (
        uint256 totalAgents,
        uint256 activeAgents,
        uint256 totalWorkflows,
        uint256 totalTrustRelationships,
        uint256 avgTrustScore
    ) {
        totalAgents = allAgents.length();
        totalWorkflows = allWorkflows.length();
        
        // Count active agents and calculate average trust
        uint256 activeCounts = 0;
        uint256 totalTrust = 0;
        
        for (uint256 i = 0; i < totalAgents; i++) {
            address agentAddr = allAgents.at(i);
            if (agents[agentAddr].active) {
                activeCounts++;
            }
            totalTrust += agentReputations[agentAddr];
        }
        
        activeAgents = activeCounts;
        avgTrustScore = totalAgents > 0 ? totalTrust / totalAgents : 0;
        
        // Count trust relationships (simplified)
        totalTrustRelationships = 0;
        for (uint256 i = 0; i < totalAgents; i++) {
            address agentAddr = allAgents.at(i);
            totalTrustRelationships += trustedAgents[agentAddr].length();
        }
    }

    // Internal functions

    function _checkStepCompletion(string memory taskId) internal {
        TaskExecution storage execution = taskExecutions[taskId];
        
        // Simple completion check - advance if current step is completed
        if (execution.stepCompleted[execution.currentStep]) {
            execution.currentStep++;
            
            // Check if task is fully completed
            if (execution.currentStep >= execution.totalSteps) {
                execution.completed = true;
                execution.successful = true;
                execution.completedAt = block.timestamp;
                
                uint256 executionTime = execution.completedAt - execution.startedAt;
                emit TaskExecutionCompleted(taskId, true, executionTime);
            }
        }
    }

    function _canCommunicate(address fromAgent, address toAgent) internal view returns (bool) {
        // Check if agents have trust relationship
        bytes32 relationshipId = keccak256(abi.encodePacked(fromAgent, toAgent));
        return trustRelationships[relationshipId].active;
    }

    /**
     * @dev Emergency pause for admin
     */
    function emergencyPause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        // Implement emergency pause functionality
        // This would disable critical functions in case of emergency
    }

    /**
     * @dev Upgrade agent trust score based on performance
     */
    function updateAgentTrustScore(address agent, uint256 newScore) 
        external 
        onlyRole(TRUST_MANAGER_ROLE) 
    {
        require(allAgents.contains(agent), "Agent not registered");
        require(newScore <= 100, "Score must be 0-100");
        
        agents[agent].trustScore = newScore;
        agentReputations[agent] = newScore;
    }

    /**
     * @dev Get complete agent information
     */
    function getAgentInfo(address agent) external view returns (
        string memory agentId,
        string memory name,
        AgentType agentType,
        string memory endpoint,
        uint256 trustScore,
        bool active,
        uint256 registeredAt,
        uint256 lastHeartbeat
    ) {
        require(allAgents.contains(agent), "Agent not registered");
        A2AAgent storage a = agents[agent];
        
        return (
            a.agentId,
            a.name,
            a.agentType,
            a.endpoint,
            a.trustScore,
            a.active,
            a.registeredAt,
            a.lastHeartbeat
        );
    }
}