// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "./AgentRegistry.sol";
import "./MultiSigPausable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title MessageRouter
 * @dev Handles secure message routing between registered agents in the A2A Network.
 * Includes rate limiting to prevent spam and message delivery confirmation.
 */
contract MessageRouter is MultiSigPausable, ReentrancyGuard {
    AgentRegistry public immutable registry;

    struct Message {
        address from;
        address to;
        bytes32 messageId;
        string content;
        uint256 timestamp;
        bool delivered;
        bytes32 messageType;
        uint256 expiresAt;
    }

    mapping(bytes32 => Message) public messages;
    mapping(address => bytes32[]) public agentMessages;
    mapping(address => uint256) public messageCounts;

    // Rate limiting
    mapping(address => uint256) public lastMessageTime;
    mapping(address => uint256) public messagesSentInWindow;
    uint256 public constant RATE_LIMIT_WINDOW = 1 hours;
    uint256 public constant MAX_MESSAGES_PER_WINDOW = 100;
    uint256 public messageDelay = 5 seconds; // Minimum delay between messages
    uint256 public messageExpiry = 7 days; // Default message expiry time

    event MessageSent(bytes32 indexed messageId, address indexed from, address indexed to, bytes32 messageType);

    event MessageDelivered(bytes32 indexed messageId);
    event RateLimitUpdated(uint256 newDelay);

    /**
     * @notice Initialize the MessageRouter with a registry address
     * @param _registry Address of the AgentRegistry contract
     * @param _requiredConfirmations Number of confirmations for multi-sig
     */
    constructor(address _registry, uint256 _requiredConfirmations) MultiSigPausable(_requiredConfirmations) {
        registry = AgentRegistry(_registry);
    }

    modifier onlyRegisteredAgent() {
        AgentRegistry.Agent memory agent = registry.getAgent(msg.sender);
        require(agent.active, "Agent not registered or inactive");
        _;
    }

    /**
     * @notice Send a message to another agent
     * @param to The recipient agent's address
     * @param content The message content
     * @param messageType Type identifier for the message
     * @return messageId Unique identifier for the sent message
     */
    function sendMessage(address to, string memory content, bytes32 messageType)
        external
        onlyRegisteredAgent
        whenNotPaused
        nonReentrant
        returns (bytes32)
    {
        // Input validation first
        require(bytes(content).length > 0, "Content required");
        AgentRegistry.Agent memory recipient = registry.getAgent(to);
        require(recipient.active, "Recipient not active");

        // Rate limiting checks after validation
        _checkRateLimit(msg.sender);

        bytes32 messageId =
            keccak256(abi.encodePacked(msg.sender, to, content, block.timestamp, messageCounts[msg.sender]));

        messages[messageId] = Message({
            from: msg.sender,
            to: to,
            messageId: messageId,
            content: content,
            timestamp: block.timestamp,
            delivered: false,
            messageType: messageType,
            expiresAt: block.timestamp + messageExpiry
        });

        agentMessages[to].push(messageId);
        messageCounts[msg.sender]++;

        // Update rate limiting
        lastMessageTime[msg.sender] = block.timestamp;
        messagesSentInWindow[msg.sender]++;

        emit MessageSent(messageId, msg.sender, to, messageType);
        return messageId;
    }

    /**
     * @notice Mark a message as delivered (only callable by recipient)
     * @param messageId The ID of the message to mark as delivered
     */
    function markAsDelivered(bytes32 messageId) external whenNotPaused nonReentrant {
        Message storage message = messages[messageId];
        require(message.to == msg.sender, "Not message recipient");
        require(!message.delivered, "Already delivered");

        message.delivered = true;
        emit MessageDelivered(messageId);
    }

    /**
     * @notice Get all message IDs for a specific agent
     * @param agent The agent's address
     * @return Array of message IDs
     */
    function getMessages(address agent) external view returns (bytes32[] memory) {
        return agentMessages[agent];
    }

    /**
     * @notice Get detailed information about a specific message
     * @param messageId The message ID
     * @return The message data structure
     */
    function getMessage(bytes32 messageId) external view returns (Message memory) {
        Message memory message = messages[messageId];
        require(message.from != address(0), "Message does not exist");
        require(block.timestamp <= message.expiresAt, "Message expired");
        return message;
    }

    /**
     * @notice Get all undelivered messages for an agent
     * @param agent The agent's address
     * @return Array of undelivered message IDs
     */
    function getUndeliveredMessages(address agent) external view returns (bytes32[] memory) {
        bytes32[] memory allMessages = agentMessages[agent];
        uint256 undeliveredCount = 0;

        for (uint256 i = 0; i < allMessages.length; i++) {
            if (!messages[allMessages[i]].delivered) {
                undeliveredCount++;
            }
        }

        bytes32[] memory undelivered = new bytes32[](undeliveredCount);
        uint256 index = 0;

        for (uint256 i = 0; i < allMessages.length; i++) {
            if (!messages[allMessages[i]].delivered) {
                undelivered[index] = allMessages[i];
                index++;
            }
        }

        return undelivered;
    }

    /**
     * @dev Check rate limiting for message sending
     * @param sender The address sending the message
     */
    function _checkRateLimit(address sender) private {
        // Skip delay check for first message
        if (lastMessageTime[sender] > 0) {
            // Check minimum delay between messages
            require(
                block.timestamp >= lastMessageTime[sender] + messageDelay, "MessageRouter: rate limit - too frequent"
            );
        }

        // Reset window if needed
        if (lastMessageTime[sender] == 0 || block.timestamp >= lastMessageTime[sender] + RATE_LIMIT_WINDOW) {
            messagesSentInWindow[sender] = 0;
        }

        // Check messages per window
        require(messagesSentInWindow[sender] < MAX_MESSAGES_PER_WINDOW, "MessageRouter: rate limit - too many messages");
    }

    /**
     * @notice Update the minimum delay between messages (only admin)
     * @param newDelay New delay in seconds
     */
    function updateMessageDelay(uint256 newDelay) external onlyRole(DEFAULT_ADMIN_ROLE) nonReentrant {
        require(newDelay >= 1 seconds && newDelay <= 1 hours, "Invalid delay");
        messageDelay = newDelay;
        emit RateLimitUpdated(newDelay);
    }

    // Note: This is a non-upgradeable contract, no storage gap needed

    // --- Pagination Functions ---

    /**
     * @notice Get paginated messages for an agent
     * @param agent The agent's address
     * @param offset Starting index
     * @param limit Maximum number of messages to return
     * @return messageIds Array of message IDs
     * @return total Total number of messages for the agent
     */
    function getMessagesPaginated(address agent, uint256 offset, uint256 limit)
        external
        view
        returns (bytes32[] memory messageIds, uint256 total)
    {
        bytes32[] memory allMessages = agentMessages[agent];
        total = allMessages.length;
        
        if (offset >= total) {
            return (new bytes32[](0), total);
        }
        
        uint256 end = offset + limit;
        if (end > total) {
            end = total;
        }
        
        uint256 length = end - offset;
        messageIds = new bytes32[](length);
        
        for (uint256 i = 0; i < length; i++) {
            messageIds[i] = allMessages[offset + i];
        }
    }

    /**
     * @notice Update message expiry time (only admin)
     * @param newExpiry New expiry duration in seconds
     */
    function updateMessageExpiry(uint256 newExpiry) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(newExpiry >= 1 days && newExpiry <= 30 days, "Invalid expiry duration");
        messageExpiry = newExpiry;
    }

    /**
     * @notice Check if a message is expired
     * @param messageId The message ID to check
     * @return bool True if the message is expired
     */
    function isMessageExpired(bytes32 messageId) external view returns (bool) {
        Message memory message = messages[messageId];
        return message.from != address(0) && block.timestamp > message.expiresAt;
    }

    /**
     * @notice Get only active (non-expired, non-delivered) messages for an agent
     * @param agent The agent's address
     * @param limit Maximum number of messages to return
     * @return activeMessageIds Array of active message IDs
     */
    function getActiveMessages(address agent, uint256 limit) 
        external 
        view 
        returns (bytes32[] memory activeMessageIds) 
    {
        bytes32[] memory allMessages = agentMessages[agent];
        uint256 activeCount = 0;
        
        // First count active messages
        for (uint256 i = 0; i < allMessages.length && activeCount < limit; i++) {
            Message memory message = messages[allMessages[i]];
            if (!message.delivered && block.timestamp <= message.expiresAt) {
                activeCount++;
            }
        }
        
        // Then collect them
        activeMessageIds = new bytes32[](activeCount);
        uint256 index = 0;
        
        for (uint256 i = 0; i < allMessages.length && index < activeCount; i++) {
            Message memory message = messages[allMessages[i]];
            if (!message.delivered && block.timestamp <= message.expiresAt) {
                activeMessageIds[index] = allMessages[i];
                index++;
            }
        }
    }
}
