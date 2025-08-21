// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "./interfaces/IAgentRegistry.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/PausableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/ReentrancyGuardUpgradeable.sol";

/**
 * @title A2ATaskQueue
 * @dev Decentralized task queue for A2A agent coordination
 * Manages task assignment, execution tracking, and result storage on-chain
 */
contract A2ATaskQueue is 
    Initializable, 
    UUPSUpgradeable, 
    OwnableUpgradeable, 
    PausableUpgradeable,
    ReentrancyGuardUpgradeable 
{
    // Task priority levels
    enum Priority {
        LOW,
        MEDIUM,
        HIGH,
        CRITICAL
    }
    
    // Task status
    enum TaskStatus {
        PENDING,
        ASSIGNED,
        PROCESSING,
        COMPLETED,
        FAILED,
        CANCELLED
    }
    
    // Task structure
    struct Task {
        bytes32 taskId;
        address requester;
        address assignedAgent;
        string skillName;
        bytes parameters;
        Priority priority;
        TaskStatus status;
        uint256 createdAt;
        uint256 deadline;
        uint256 gasLimit;
        uint256 reward;
        bytes32 resultHash;
        string errorMessage;
    }
    
    // Queue statistics
    struct QueueStats {
        uint256 totalTasks;
        uint256 pendingTasks;
        uint256 processingTasks;
        uint256 completedTasks;
        uint256 failedTasks;
    }
    
    // Storage
    IAgentRegistry public agentRegistry;
    mapping(bytes32 => Task) public tasks;
    mapping(address => bytes32[]) public agentTasks;
    mapping(address => uint256) public agentRewards;
    
    bytes32[] public taskQueue;
    QueueStats public stats;
    
    uint256 public minTaskReward;
    uint256 public maxTasksPerAgent;
    
    // Events
    event TaskCreated(
        bytes32 indexed taskId,
        address indexed requester,
        string skillName,
        Priority priority
    );
    
    event TaskAssigned(
        bytes32 indexed taskId,
        address indexed agent
    );
    
    event TaskCompleted(
        bytes32 indexed taskId,
        address indexed agent,
        bytes32 resultHash
    );
    
    event TaskFailed(
        bytes32 indexed taskId,
        address indexed agent,
        string reason
    );
    
    event RewardClaimed(
        address indexed agent,
        uint256 amount
    );
    
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }
    
    function initialize(address _agentRegistry) public initializer {
        __Ownable_init();
        __Pausable_init();
        __UUPSUpgradeable_init();
        __ReentrancyGuard_init();
        
        agentRegistry = IAgentRegistry(_agentRegistry);
        minTaskReward = 0.001 ether;
        maxTasksPerAgent = 10;
    }
    
    /**
     * @dev Create a new task
     */
    function createTask(
        string calldata skillName,
        bytes calldata parameters,
        Priority priority,
        uint256 deadline,
        uint256 gasLimit
    ) external payable whenNotPaused returns (bytes32) {
        require(msg.value >= minTaskReward, "Insufficient reward");
        require(deadline > block.timestamp, "Invalid deadline");
        require(gasLimit > 0, "Invalid gas limit");
        
        bytes32 taskId = keccak256(
            abi.encodePacked(
                msg.sender,
                skillName,
                block.timestamp,
                stats.totalTasks
            )
        );
        
        tasks[taskId] = Task({
            taskId: taskId,
            requester: msg.sender,
            assignedAgent: address(0),
            skillName: skillName,
            parameters: parameters,
            priority: priority,
            status: TaskStatus.PENDING,
            createdAt: block.timestamp,
            deadline: deadline,
            gasLimit: gasLimit,
            reward: msg.value,
            resultHash: bytes32(0),
            errorMessage: ""
        });
        
        taskQueue.push(taskId);
        stats.totalTasks++;
        stats.pendingTasks++;
        
        emit TaskCreated(taskId, msg.sender, skillName, priority);
        
        return taskId;
    }
    
    /**
     * @dev Assign task to an agent
     */
    function assignTask(bytes32 taskId) external whenNotPaused nonReentrant {
        Task storage task = tasks[taskId];
        require(task.status == TaskStatus.PENDING, "Task not available");
        require(task.deadline > block.timestamp, "Task expired");
        
        // Verify agent is registered and active
        IAgentRegistry.Agent memory agent = agentRegistry.getAgent(msg.sender);
        require(agent.active, "Agent not active");
        
        // Check agent workload
        require(
            getActiveTaskCount(msg.sender) < maxTasksPerAgent,
            "Agent has too many active tasks"
        );
        
        task.assignedAgent = msg.sender;
        task.status = TaskStatus.ASSIGNED;
        agentTasks[msg.sender].push(taskId);
        
        stats.pendingTasks--;
        stats.processingTasks++;
        
        emit TaskAssigned(taskId, msg.sender);
    }
    
    /**
     * @dev Start processing a task
     */
    function startTask(bytes32 taskId) external whenNotPaused {
        Task storage task = tasks[taskId];
        require(task.assignedAgent == msg.sender, "Not assigned agent");
        require(task.status == TaskStatus.ASSIGNED, "Invalid task status");
        require(task.deadline > block.timestamp, "Task expired");
        
        task.status = TaskStatus.PROCESSING;
    }
    
    /**
     * @dev Submit task result
     */
    function completeTask(
        bytes32 taskId,
        bytes32 resultHash
    ) external whenNotPaused nonReentrant {
        Task storage task = tasks[taskId];
        require(task.assignedAgent == msg.sender, "Not assigned agent");
        require(task.status == TaskStatus.PROCESSING, "Task not processing");
        
        task.status = TaskStatus.COMPLETED;
        task.resultHash = resultHash;
        
        // Update rewards
        agentRewards[msg.sender] += task.reward;
        
        stats.processingTasks--;
        stats.completedTasks++;
        
        emit TaskCompleted(taskId, msg.sender, resultHash);
    }
    
    /**
     * @dev Report task failure
     */
    function failTask(
        bytes32 taskId,
        string calldata reason
    ) external whenNotPaused {
        Task storage task = tasks[taskId];
        require(task.assignedAgent == msg.sender, "Not assigned agent");
        require(
            task.status == TaskStatus.ASSIGNED || 
            task.status == TaskStatus.PROCESSING,
            "Invalid task status"
        );
        
        task.status = TaskStatus.FAILED;
        task.errorMessage = reason;
        
        // Return reward to requester
        payable(task.requester).transfer(task.reward);
        
        stats.processingTasks--;
        stats.failedTasks++;
        
        emit TaskFailed(taskId, msg.sender, reason);
    }
    
    /**
     * @dev Cancel expired task
     */
    function cancelExpiredTask(bytes32 taskId) external whenNotPaused {
        Task storage task = tasks[taskId];
        require(
            task.status == TaskStatus.PENDING || 
            task.status == TaskStatus.ASSIGNED,
            "Cannot cancel"
        );
        require(task.deadline < block.timestamp, "Task not expired");
        
        task.status = TaskStatus.CANCELLED;
        
        // Return reward to requester
        if (task.reward > 0) {
            payable(task.requester).transfer(task.reward);
        }
        
        if (task.status == TaskStatus.PENDING) {
            stats.pendingTasks--;
        } else {
            stats.processingTasks--;
        }
    }
    
    /**
     * @dev Claim accumulated rewards
     */
    function claimRewards() external whenNotPaused nonReentrant {
        uint256 rewards = agentRewards[msg.sender];
        require(rewards > 0, "No rewards to claim");
        
        agentRewards[msg.sender] = 0;
        payable(msg.sender).transfer(rewards);
        
        emit RewardClaimed(msg.sender, rewards);
    }
    
    /**
     * @dev Get active task count for agent
     */
    function getActiveTaskCount(address agent) public view returns (uint256) {
        uint256 count = 0;
        bytes32[] memory taskIds = agentTasks[agent];
        
        for (uint256 i = 0; i < taskIds.length; i++) {
            TaskStatus status = tasks[taskIds[i]].status;
            if (status == TaskStatus.ASSIGNED || status == TaskStatus.PROCESSING) {
                count++;
            }
        }
        
        return count;
    }
    
    /**
     * @dev Get pending tasks by priority
     */
    function getPendingTasks(
        Priority priority,
        uint256 limit
    ) external view returns (bytes32[] memory) {
        bytes32[] memory pendingTasks = new bytes32[](limit);
        uint256 count = 0;
        
        for (uint256 i = 0; i < taskQueue.length && count < limit; i++) {
            Task memory task = tasks[taskQueue[i]];
            if (
                task.status == TaskStatus.PENDING &&
                task.priority == priority &&
                task.deadline > block.timestamp
            ) {
                pendingTasks[count] = task.taskId;
                count++;
            }
        }
        
        // Resize array to actual count
        assembly {
            mstore(pendingTasks, count)
        }
        
        return pendingTasks;
    }
    
    /**
     * @dev Update configuration
     */
    function setMinTaskReward(uint256 _minReward) external onlyOwner {
        minTaskReward = _minReward;
    }
    
    function setMaxTasksPerAgent(uint256 _maxTasks) external onlyOwner {
        maxTasksPerAgent = _maxTasks;
    }
    
    /**
     * @dev Required by UUPSUpgradeable
     */
    function _authorizeUpgrade(address newImplementation) internal override onlyOwner {}
    
    /**
     * @dev Emergency pause
     */
    function pause() external onlyOwner {
        _pause();
    }
    
    function unpause() external onlyOwner {
        _unpause();
    }
}