// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";
import "./ZKMessageProof.sol";

/**
 * @title PrivateMessageRouter
 * @notice Privacy-preserving message routing using zero-knowledge proofs
 * @dev Extends MessageRouter with ZK proof verification for private communication
 */
contract PrivateMessageRouter is AccessControl, ReentrancyGuard, Pausable {
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    bytes32 public constant AGENT_ROLE = keccak256("AGENT_ROLE");

    ZKMessageProof public immutable zkProofContract;

    // Private message structure
    struct PrivateMessage {
        bytes32 commitmentId;      // Reference to ZK commitment
        bytes encryptedContent;    // Encrypted message content
        bytes32 senderCommitment;  // Hidden sender identity
        bytes32 recipientCommitment; // Hidden recipient identity
        uint256 timestamp;
        uint256 expiresAt;
        bool delivered;
        bytes32 proofHash;         // Hash of the ZK proof
    }

    // Message metadata for discovery without revealing content
    struct MessageMetadata {
        bytes32 messageId;
        bytes32 topicHash;         // Category/topic for filtering
        uint256 timestamp;
        uint256 bounty;           // Optional reward for delivery
        bool requiresProof;       // Whether ZK proof is required
    }

    // State variables
    mapping(bytes32 => PrivateMessage) public privateMessages;
    mapping(bytes32 => MessageMetadata) public messageMetadata;
    mapping(bytes32 => bytes32[]) public topicMessages; // topic -> message IDs
    mapping(address => bytes32[]) public agentInbox;    // For message discovery
    mapping(bytes32 => mapping(address => bool)) public messageAccess; // Who can decrypt
    
    uint256 public totalMessages;
    uint256 public constant MAX_MESSAGE_LIFETIME = 30 days;
    uint256 public constant MESSAGE_FEE = 0.001 ether;

    // Events
    event PrivateMessageSent(
        bytes32 indexed messageId,
        bytes32 indexed commitmentId,
        bytes32 indexed topicHash,
        uint256 bounty
    );
    event MessageDelivered(bytes32 indexed messageId, bytes32 proofHash);
    event MessageExpired(bytes32 indexed messageId);
    event AccessGranted(bytes32 indexed messageId, address indexed recipient);

    // Custom errors
    error InvalidCommitment();
    error MessageNotFound();
    error UnauthorizedAccess();
    error InvalidProof();
    error InsufficientPayment();
    error MessageAlreadyDelivered();
    error MessageExpiredError();

    constructor(address admin, address pauser, address zkProofContract_) {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(PAUSER_ROLE, pauser);
        zkProofContract = ZKMessageProof(zkProofContract_);
    }

    /**
     * @notice Send a private message using zero-knowledge commitment
     * @param encryptedContent The encrypted message content
     * @param commitmentId Reference to ZK commitment in ZKMessageProof contract
     * @param topicHash Hash of message topic for categorization
     * @param recipientHint Encrypted hint for recipient discovery
     * @param expiryDuration How long the message should remain valid
     * @return messageId Unique identifier for the message
     */
    function sendPrivateMessage(
        bytes calldata encryptedContent,
        bytes32 commitmentId,
        bytes32 topicHash,
        bytes calldata recipientHint,
        uint256 expiryDuration
    ) external payable whenNotPaused nonReentrant returns (bytes32 messageId) {
        if (msg.value < MESSAGE_FEE) revert InsufficientPayment();
        if (encryptedContent.length == 0) revert InvalidCommitment();
        if (expiryDuration > MAX_MESSAGE_LIFETIME) expiryDuration = MAX_MESSAGE_LIFETIME;

        // Verify commitment exists in ZK contract
        ZKMessageProof.MessageCommitment memory commitment = zkProofContract.getMessageCommitment(commitmentId);
        if (commitment.timestamp == 0) revert InvalidCommitment();

        // Generate unique message ID
        messageId = keccak256(abi.encodePacked(
            commitmentId,
            encryptedContent,
            block.timestamp,
            totalMessages++
        ));

        uint256 expiresAt = block.timestamp + expiryDuration;

        // Store private message
        privateMessages[messageId] = PrivateMessage({
            commitmentId: commitmentId,
            encryptedContent: encryptedContent,
            senderCommitment: commitment.senderCommitment,
            recipientCommitment: commitment.recipientCommitment,
            timestamp: block.timestamp,
            expiresAt: expiresAt,
            delivered: false,
            proofHash: bytes32(0)
        });

        // Store metadata for discovery
        messageMetadata[messageId] = MessageMetadata({
            messageId: messageId,
            topicHash: topicHash,
            timestamp: block.timestamp,
            bounty: msg.value - MESSAGE_FEE,
            requiresProof: true
        });

        // Add to topic index
        topicMessages[topicHash].push(messageId);

        emit PrivateMessageSent(messageId, commitmentId, topicHash, msg.value - MESSAGE_FEE);
    }

    /**
     * @notice Deliver a message with zero-knowledge proof
     * @param messageId The message to deliver
     * @param proof ZK proof that recipient can decrypt the message
     * @param publicInputs Public inputs for proof verification
     * @param nullifier Unique nullifier to prevent double-claiming
     */
    function deliverWithProof(
        bytes32 messageId,
        ZKMessageProof.Proof calldata proof,
        uint256[] calldata publicInputs,
        bytes32 nullifier
    ) external whenNotPaused nonReentrant {
        PrivateMessage storage message = privateMessages[messageId];
        if (message.timestamp == 0) revert MessageNotFound();
        if (block.timestamp >= message.expiresAt) revert MessageExpiredError();
        if (message.delivered) revert MessageAlreadyDelivered();

        // Verify ZK proof through the proof contract
        try zkProofContract.verifyMessageProof(
            message.commitmentId,
            proof,
            publicInputs,
            nullifier
        ) {
            // Mark as delivered
            message.delivered = true;
            message.proofHash = keccak256(abi.encode(proof));
            
            // Grant access to the prover
            messageAccess[messageId][msg.sender] = true;
            agentInbox[msg.sender].push(messageId);

            // Pay bounty if any
            MessageMetadata storage metadata = messageMetadata[messageId];
            if (metadata.bounty > 0) {
                uint256 bounty = metadata.bounty;
                metadata.bounty = 0;
                (bool success, ) = msg.sender.call{value: bounty}("");
                require(success, "Bounty payment failed");
            }

            emit MessageDelivered(messageId, message.proofHash);
            emit AccessGranted(messageId, msg.sender);
        } catch {
            revert InvalidProof();
        }
    }

    /**
     * @notice Get messages by topic (public metadata only)
     * @param topicHash The topic to search
     * @param offset Starting index for pagination
     * @param limit Maximum number of results
     * @return messageIds Array of message IDs
     * @return total Total number of messages in topic
     */
    function getMessagesByTopic(
        bytes32 topicHash,
        uint256 offset,
        uint256 limit
    ) external view returns (bytes32[] memory messageIds, uint256 total) {
        bytes32[] storage topicMessageIds = topicMessages[topicHash];
        total = topicMessageIds.length;
        
        if (offset >= total) return (new bytes32[](0), total);
        
        uint256 end = offset + limit;
        if (end > total) end = total;
        
        messageIds = new bytes32[](end - offset);
        for (uint256 i = offset; i < end; i++) {
            messageIds[i - offset] = topicMessageIds[i];
        }
    }

    /**
     * @notice Get message metadata (public information only)
     * @param messageId The message ID
     * @return metadata Public metadata
     */
    function getMessageMetadata(
        bytes32 messageId
    ) external view returns (MessageMetadata memory metadata) {
        metadata = messageMetadata[messageId];
        if (metadata.timestamp == 0) revert MessageNotFound();
    }

    /**
     * @notice Get encrypted message content (requires access)
     * @param messageId The message ID
     * @return encryptedContent The encrypted content
     */
    function getEncryptedContent(
        bytes32 messageId
    ) external view returns (bytes memory encryptedContent) {
        if (!messageAccess[messageId][msg.sender]) revert UnauthorizedAccess();
        
        PrivateMessage storage message = privateMessages[messageId];
        if (message.timestamp == 0) revert MessageNotFound();
        
        encryptedContent = message.encryptedContent;
    }

    /**
     * @notice Get agent's inbox (messages they can access)
     * @param agent The agent address
     * @param offset Starting index for pagination
     * @param limit Maximum number of results
     * @return messageIds Array of accessible message IDs
     * @return total Total number of messages in inbox
     */
    function getAgentInbox(
        address agent,
        uint256 offset,
        uint256 limit
    ) external view returns (bytes32[] memory messageIds, uint256 total) {
        bytes32[] storage inbox = agentInbox[agent];
        total = inbox.length;
        
        if (offset >= total) return (new bytes32[](0), total);
        
        uint256 end = offset + limit;
        if (end > total) end = total;
        
        messageIds = new bytes32[](end - offset);
        for (uint256 i = offset; i < end; i++) {
            messageIds[i - offset] = inbox[i];
        }
    }

    /**
     * @notice Clean up expired messages
     * @param messageIds Array of message IDs to check and clean
     */
    function cleanupExpiredMessages(
        bytes32[] calldata messageIds
    ) external {
        for (uint256 i = 0; i < messageIds.length; i++) {
            bytes32 messageId = messageIds[i];
            PrivateMessage storage message = privateMessages[messageId];
            
            if (message.timestamp > 0 && block.timestamp >= message.expiresAt && !message.delivered) {
                // Refund any unclaimed bounty
                MessageMetadata storage metadata = messageMetadata[messageId];
                if (metadata.bounty > 0) {
                    // In production, implement proper refund mechanism
                    metadata.bounty = 0;
                }
                
                // Mark as expired
                delete privateMessages[messageId];
                delete messageMetadata[messageId];
                
                emit MessageExpired(messageId);
            }
        }
    }

    /**
     * @notice Check if user has access to a message
     * @param messageId The message ID
     * @param user The user address
     * @return True if user has access
     */
    function hasMessageAccess(bytes32 messageId, address user) external view returns (bool) {
        return messageAccess[messageId][user];
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
     * @notice Withdraw contract balance (admin only)
     */
    function withdraw() external onlyRole(DEFAULT_ADMIN_ROLE) {
        uint256 balance = address(this).balance;
        require(balance > 0, "No funds to withdraw");
        
        (bool success, ) = msg.sender.call{value: balance}("");
        require(success, "Withdrawal failed");
    }

    // Receive function to accept bounty payments
    receive() external payable {}
}