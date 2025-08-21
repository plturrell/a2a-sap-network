// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "./CrossChainBridge.sol";

/**
 * @title CrossChainMessageRouter
 * @notice Cross-chain message routing for A2A Network agents
 * @dev Enables agents to send messages across different blockchain networks
 */
contract CrossChainMessageRouter is AccessControl, ReentrancyGuard, Pausable {
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    bytes32 public constant ROUTER_OPERATOR_ROLE = keccak256("ROUTER_OPERATOR_ROLE");

    CrossChainBridge public immutable crossChainBridge;

    // Cross-chain message structure
    struct CrossChainMessage {
        bytes32 messageId;
        uint256 sourceChainId;
        uint256 targetChainId;
        address sourceAgent;
        address targetAgent;
        bytes content;
        string messageType;
        uint256 timestamp;
        uint256 expiresAt;
        MessagePriority priority;
        uint256 gasBudget;      // Gas budget for execution on target chain
        bool delivered;
        bytes32 bridgeMessageHash; // Reference to bridge message
    }

    enum MessagePriority {
        Low,
        Medium,
        High,
        Critical
    }

    // Rate limiting structure
    struct RateLimit {
        uint256 messagesPerHour;
        uint256 hourlyVolumeLimit; // In wei
        uint256 currentHourMessages;
        uint256 currentHourVolume;
        uint256 hourStartTime;
    }

    // State variables
    mapping(bytes32 => CrossChainMessage) public crossChainMessages;
    mapping(address => RateLimit) public agentRateLimits;
    mapping(address => bytes32[]) public agentMessages; // agent -> message IDs
    mapping(uint256 => bytes32[]) public chainMessages; // chainId -> message IDs
    mapping(bytes32 => bool) public processedBridgeMessages;
    
    uint256 public totalCrossChainMessages;
    mapping(MessagePriority => uint256) public priorityFees;
    
    // Default rate limits
    uint256 public constant DEFAULT_MESSAGES_PER_HOUR = 10;
    uint256 public constant DEFAULT_HOURLY_VOLUME_LIMIT = 1 ether;
    uint256 public constant MESSAGE_EXPIRY = 24 hours;

    // Events
    event CrossChainMessageSent(
        bytes32 indexed messageId,
        uint256 indexed sourceChainId,
        uint256 indexed targetChainId,
        address sourceAgent,
        address targetAgent,
        string messageType,
        MessagePriority priority
    );
    event CrossChainMessageReceived(
        bytes32 indexed messageId,
        bytes32 indexed bridgeMessageHash,
        address indexed targetAgent
    );
    event CrossChainMessageDelivered(
        bytes32 indexed messageId,
        address indexed targetAgent,
        bool success
    );
    event RateLimitUpdated(
        address indexed agent,
        uint256 messagesPerHour,
        uint256 volumeLimit
    );
    event PriorityFeeUpdated(MessagePriority priority, uint256 fee);

    // Custom errors
    error MessageNotFound(bytes32 messageId);
    error UnauthorizedSender(address sender, address expected);
    error RateLimitExceeded(address agent, string limitType);
    error InsufficientFee(uint256 required, uint256 provided);
    error InvalidChain(uint256 chainId);
    error MessageExpired(bytes32 messageId);
    error MessageAlreadyDelivered(bytes32 messageId);
    error InsufficientGasBudget(uint256 required, uint256 provided);

    constructor(
        address admin,
        address pauser,
        address payable crossChainBridge_
    ) {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(PAUSER_ROLE, pauser);
        _grantRole(ROUTER_OPERATOR_ROLE, admin);
        crossChainBridge = CrossChainBridge(crossChainBridge_);

        // Set default priority fees
        priorityFees[MessagePriority.Low] = 0.001 ether;
        priorityFees[MessagePriority.Medium] = 0.003 ether;
        priorityFees[MessagePriority.High] = 0.01 ether;
        priorityFees[MessagePriority.Critical] = 0.05 ether;
    }

    /**
     * @notice Send cross-chain message to agent on another blockchain
     * @param targetChainId Target blockchain ID
     * @param targetAgent Target agent address
     * @param content Message content
     * @param messageType Type of message (e.g., "task_request", "data_query")
     * @param priority Message priority level
     * @param gasBudget Gas budget for execution on target chain
     * @param expiryDuration How long the message should remain valid
     * @return messageId Unique identifier for the message
     */
    function sendCrossChainMessage(
        uint256 targetChainId,
        address targetAgent,
        bytes calldata content,
        string calldata messageType,
        MessagePriority priority,
        uint256 gasBudget,
        uint256 expiryDuration
    ) external payable whenNotPaused nonReentrant returns (bytes32 messageId) {
        require(targetAgent != address(0), "Invalid target agent");
        require(content.length > 0, "Empty message content");
        require(gasBudget > 0, "Invalid gas budget");
        
        // Check rate limits
        _checkRateLimit(msg.sender);
        
        // Calculate required fee
        uint256 priorityFee = priorityFees[priority];
        uint256 bridgeFee = _estimateBridgeFee(targetChainId);
        uint256 totalFee = priorityFee + bridgeFee + gasBudget;
        
        if (msg.value < totalFee) {
            revert InsufficientFee(totalFee, msg.value);
        }

        // Generate message ID
        messageId = keccak256(abi.encodePacked(
            block.chainid,
            targetChainId,
            msg.sender,
            targetAgent,
            content,
            block.timestamp,
            totalCrossChainMessages++
        ));

        uint256 expiresAt = block.timestamp + 
            (expiryDuration > MESSAGE_EXPIRY ? MESSAGE_EXPIRY : expiryDuration);

        // Store message
        crossChainMessages[messageId] = CrossChainMessage({
            messageId: messageId,
            sourceChainId: block.chainid,
            targetChainId: targetChainId,
            sourceAgent: msg.sender,
            targetAgent: targetAgent,
            content: content,
            messageType: messageType,
            timestamp: block.timestamp,
            expiresAt: expiresAt,
            priority: priority,
            gasBudget: gasBudget,
            delivered: false,
            bridgeMessageHash: bytes32(0)
        });

        // Update rate limits
        _updateRateLimit(msg.sender, msg.value);

        // Add to indexes
        agentMessages[msg.sender].push(messageId);
        chainMessages[targetChainId].push(messageId);

        // Encode message for cross-chain bridge
        bytes memory encodedMessage = abi.encode(
            messageId,
            msg.sender,
            targetAgent,
            content,
            messageType,
            priority,
            gasBudget
        );

        // Send via cross-chain bridge
        bytes32 bridgeMessageHash = crossChainBridge.sendCrossChainMessage{
            value: bridgeFee + gasBudget
        }(
            targetChainId,
            address(this), // Target contract (this router on target chain)
            encodedMessage
        );

        // Update message with bridge hash
        crossChainMessages[messageId].bridgeMessageHash = bridgeMessageHash;

        emit CrossChainMessageSent(
            messageId,
            block.chainid,
            targetChainId,
            msg.sender,
            targetAgent,
            messageType,
            priority
        );

        // Refund excess payment
        if (msg.value > totalFee) {
            (bool success,) = msg.sender.call{value: msg.value - totalFee}("");
            require(success, "Refund failed");
        }
    }

    /**
     * @notice Receive and process cross-chain message from bridge
     * @param bridgeMessageData Encoded message data from bridge
     */
    function receiveCrossChainMessage(
        bytes calldata bridgeMessageData
    ) external whenNotPaused {
        require(msg.sender == address(crossChainBridge), "Only bridge can call");

        // Decode message data
        (
            bytes32 messageId,
            address sourceAgent,
            address targetAgent,
            bytes memory content,
            string memory messageType,
            MessagePriority priority,
            uint256 gasBudget
        ) = abi.decode(
            bridgeMessageData,
            (bytes32, address, address, bytes, string, MessagePriority, uint256)
        );

        // Prevent replay attacks
        bytes32 bridgeMessageHash = keccak256(bridgeMessageData);
        require(!processedBridgeMessages[bridgeMessageHash], "Message already processed");
        processedBridgeMessages[bridgeMessageHash] = true;

        emit CrossChainMessageReceived(messageId, bridgeMessageHash, targetAgent);

        // Attempt to deliver message to target agent
        bool deliverySuccess = _deliverMessage(
            messageId,
            sourceAgent,
            targetAgent,
            content,
            messageType,
            priority,
            gasBudget
        );

        emit CrossChainMessageDelivered(messageId, targetAgent, deliverySuccess);
    }

    /**
     * @notice Internal function to deliver message to target agent
     */
    function _deliverMessage(
        bytes32 messageId,
        address sourceAgent,
        address targetAgent,
        bytes memory content,
        string memory messageType,
        MessagePriority priority,
        uint256 gasBudget
    ) internal returns (bool success) {
        // Check if target agent can receive messages
        if (targetAgent.code.length == 0) {
            return false; // Target is not a contract
        }

        // Prepare message data for target agent
        bytes memory messageData = abi.encodeWithSignature(
            "receiveMessage(bytes32,address,bytes,string,uint8)",
            messageId,
            sourceAgent,
            content,
            messageType,
            uint8(priority)
        );

        // Use limited gas to prevent DoS
        uint256 gasLimit = gasBudget > 200000 ? gasBudget : 200000;
        
        try this.safeDeliverMessage{gas: gasLimit}(targetAgent, messageData) {
            success = true;
        } catch {
            success = false;
        }
    }

    /**
     * @notice Safe message delivery wrapper
     * @param target Target contract address
     * @param data Message data to deliver
     */
    function safeDeliverMessage(
        address target,
        bytes calldata data
    ) external {
        require(msg.sender == address(this), "Only router can call");
        
        (bool success,) = target.call(data);
        require(success, "Message delivery failed");
    }

    /**
     * @notice Check and update rate limits for an agent
     * @param agent Agent address to check
     */
    function _checkRateLimit(address agent) internal view {
        RateLimit storage limit = agentRateLimits[agent];
        
        // Initialize default limits if not set
        uint256 messagesPerHour = limit.messagesPerHour == 0 ? 
            DEFAULT_MESSAGES_PER_HOUR : limit.messagesPerHour;
        uint256 volumeLimit = limit.hourlyVolumeLimit == 0 ? 
            DEFAULT_HOURLY_VOLUME_LIMIT : limit.hourlyVolumeLimit;

        // Check if we're in a new hour
        if (block.timestamp >= limit.hourStartTime + 1 hours) {
            return; // New hour, limits will be reset in _updateRateLimit
        }

        // Check message frequency limit
        if (limit.currentHourMessages >= messagesPerHour) {
            revert RateLimitExceeded(agent, "message_frequency");
        }

        // Check volume limit
        if (limit.currentHourVolume + msg.value > volumeLimit) {
            revert RateLimitExceeded(agent, "hourly_volume");
        }
    }

    /**
     * @notice Update rate limit counters
     * @param agent Agent address
     * @param value Message value/cost
     */
    function _updateRateLimit(address agent, uint256 value) internal {
        RateLimit storage limit = agentRateLimits[agent];
        
        // Check if we're in a new hour
        if (block.timestamp >= limit.hourStartTime + 1 hours) {
            // Reset counters for new hour
            limit.currentHourMessages = 1;
            limit.currentHourVolume = value;
            limit.hourStartTime = block.timestamp;
        } else {
            // Increment counters
            limit.currentHourMessages++;
            limit.currentHourVolume += value;
        }
    }

    /**
     * @notice Estimate bridge fee for target chain
     * @param targetChainId Target chain ID
     * @return Estimated bridge fee
     */
    function _estimateBridgeFee(uint256 targetChainId) internal view returns (uint256) {
        // In a real implementation, this would query the bridge for current fees
        // For now, return a base fee that varies by chain
        if (targetChainId == 1) return 0.01 ether;  // Ethereum mainnet
        if (targetChainId == 137) return 0.001 ether; // Polygon
        if (targetChainId == 42161) return 0.005 ether; // Arbitrum
        return 0.003 ether; // Default fee
    }

    /**
     * @notice Set custom rate limits for an agent
     * @param agent Agent address
     * @param messagesPerHour Maximum messages per hour
     * @param volumeLimit Maximum volume per hour in wei
     */
    function setAgentRateLimit(
        address agent,
        uint256 messagesPerHour,
        uint256 volumeLimit
    ) external onlyRole(ROUTER_OPERATOR_ROLE) {
        require(agent != address(0), "Invalid agent address");
        require(messagesPerHour > 0, "Invalid message limit");
        require(volumeLimit > 0, "Invalid volume limit");

        RateLimit storage limit = agentRateLimits[agent];
        limit.messagesPerHour = messagesPerHour;
        limit.hourlyVolumeLimit = volumeLimit;

        emit RateLimitUpdated(agent, messagesPerHour, volumeLimit);
    }

    /**
     * @notice Update priority fees
     * @param priority Message priority level
     * @param fee New fee amount
     */
    function setPriorityFee(
        MessagePriority priority,
        uint256 fee
    ) external onlyRole(ROUTER_OPERATOR_ROLE) {
        priorityFees[priority] = fee;
        emit PriorityFeeUpdated(priority, fee);
    }

    /**
     * @notice Get messages sent by an agent
     * @param agent Agent address
     * @param offset Pagination offset
     * @param limit Maximum results
     * @return messageIds Array of message IDs
     * @return total Total messages sent by agent
     */
    function getAgentMessages(
        address agent,
        uint256 offset,
        uint256 limit
    ) external view returns (
        bytes32[] memory messageIds,
        uint256 total
    ) {
        bytes32[] storage agentMessageList = agentMessages[agent];
        total = agentMessageList.length;
        
        if (offset >= total) return (new bytes32[](0), total);
        
        uint256 end = offset + limit;
        if (end > total) end = total;
        uint256 resultCount = end - offset;
        
        messageIds = new bytes32[](resultCount);
        for (uint256 i = 0; i < resultCount; i++) {
            messageIds[i] = agentMessageList[offset + i];
        }
    }

    /**
     * @notice Get message details
     * @param messageId Message ID
     * @return CrossChainMessage data
     */
    function getMessage(
        bytes32 messageId
    ) external view returns (CrossChainMessage memory) {
        CrossChainMessage storage message = crossChainMessages[messageId];
        if (message.timestamp == 0) revert MessageNotFound(messageId);
        return message;
    }

    /**
     * @notice Get agent's current rate limit status
     * @param agent Agent address
     * @return messagesUsed Messages used this hour
     * @return messagesLimit Messages allowed per hour
     * @return volumeUsed Volume used this hour
     * @return volumeLimit Volume allowed per hour
     * @return timeUntilReset Seconds until limits reset
     */
    function getRateLimitStatus(address agent) external view returns (
        uint256 messagesUsed,
        uint256 messagesLimit,
        uint256 volumeUsed,
        uint256 volumeLimit,
        uint256 timeUntilReset
    ) {
        RateLimit storage limit = agentRateLimits[agent];
        
        messagesLimit = limit.messagesPerHour == 0 ? 
            DEFAULT_MESSAGES_PER_HOUR : limit.messagesPerHour;
        volumeLimit = limit.hourlyVolumeLimit == 0 ? 
            DEFAULT_HOURLY_VOLUME_LIMIT : limit.hourlyVolumeLimit;

        if (block.timestamp >= limit.hourStartTime + 1 hours) {
            // Limits would reset
            messagesUsed = 0;
            volumeUsed = 0;
            timeUntilReset = 0;
        } else {
            messagesUsed = limit.currentHourMessages;
            volumeUsed = limit.currentHourVolume;
            timeUntilReset = (limit.hourStartTime + 1 hours) - block.timestamp;
        }
    }

    /**
     * @notice Pause contract
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
     * @notice Withdraw collected fees
     */
    function withdrawFees() external onlyRole(DEFAULT_ADMIN_ROLE) {
        uint256 balance = address(this).balance;
        require(balance > 0, "No fees to withdraw");
        
        (bool success,) = msg.sender.call{value: balance}("");
        require(success, "Fee withdrawal failed");
    }

    // Receive function for fee payments
    receive() external payable {}
}