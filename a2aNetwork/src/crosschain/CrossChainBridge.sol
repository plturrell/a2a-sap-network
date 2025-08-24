// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

/**
 * @title CrossChainBridge
 * @notice Cross-chain communication bridge for A2A Network
 * @dev Enables agents to communicate across different blockchain networks
 */
contract CrossChainBridge is AccessControl, ReentrancyGuard, Pausable {
    bytes32 public constant BRIDGE_OPERATOR_ROLE = keccak256("BRIDGE_OPERATOR_ROLE");
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    bytes32 public constant VALIDATOR_ROLE = keccak256("VALIDATOR_ROLE");

    // Supported chain information
    struct ChainInfo {
        uint256 chainId;
        string name;
        address bridgeContract;
        bool isActive;
        uint256 blockConfirmations; // Required confirmations for finality
        uint256 messageFee;         // Base fee for cross-chain messages
    }

    // Cross-chain message structure
    struct CrossChainMessage {
        uint256 sourceChainId;
        uint256 targetChainId;
        address sourceAgent;
        address targetAgent;
        bytes messageData;
        bytes32 messageHash;
        uint256 timestamp;
        uint256 nonce;
        MessageStatus status;
        bytes32[] validatorSignatures; // Validator consensus signatures
    }

    enum MessageStatus {
        Pending,
        Validated,
        Executed,
        Failed,
        Expired
    }

    // State variables
    mapping(uint256 => ChainInfo) internal _supportedChains;
    mapping(bytes32 => CrossChainMessage) public crossChainMessages;
    mapping(address => uint256) public agentNonces;
    mapping(bytes32 => bool) public processedMessages;
    mapping(address => mapping(uint256 => bool)) public validatorVotes;
    
    uint256[] public activeChains;
    uint256 public currentChainId;
    uint256 public constant MESSAGE_EXPIRY = 7 days;
    uint256 public constant MIN_VALIDATOR_CONSENSUS = 3; // Minimum validators required
    uint256 public validatorCount;

    // Events
    event ChainAdded(uint256 indexed chainId, string name, address bridgeContract);
    event ChainUpdated(uint256 indexed chainId, bool isActive);
    event CrossChainMessageSent(
        bytes32 indexed messageHash,
        uint256 indexed sourceChainId,
        uint256 indexed targetChainId,
        address sourceAgent,
        address targetAgent
    );
    event CrossChainMessageReceived(
        bytes32 indexed messageHash,
        uint256 indexed sourceChainId,
        address sourceAgent,
        address targetAgent
    );
    event MessageValidated(bytes32 indexed messageHash, address indexed validator);
    event MessageExecuted(bytes32 indexed messageHash, bool success);
    event ValidatorAdded(address indexed validator);
    event ValidatorRemoved(address indexed validator);

    // Custom errors
    error UnsupportedChain(uint256 chainId);
    error MessageNotFound(bytes32 messageHash);
    error MessageExpired(bytes32 messageHash);
    error InsufficientFee(uint256 required, uint256 provided);
    error UnauthorizedValidator(address validator);
    error MessageAlreadyProcessed(bytes32 messageHash);
    error InsufficientValidatorConsensus(uint256 required, uint256 provided);

    constructor(uint256 _chainId, address admin, address pauser) {
        currentChainId = _chainId;
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(PAUSER_ROLE, pauser);
        _grantRole(BRIDGE_OPERATOR_ROLE, admin);
    }

    /**
     * @notice Add support for a new blockchain
     * @param chainId The chain ID of the blockchain
     * @param name Human readable name of the chain
     * @param bridgeContract Address of the bridge contract on that chain
     * @param blockConfirmations Required block confirmations
     * @param messageFee Base message fee for this chain
     */
    function addSupportedChain(
        uint256 chainId,
        string calldata name,
        address bridgeContract,
        uint256 blockConfirmations,
        uint256 messageFee
    ) external onlyRole(BRIDGE_OPERATOR_ROLE) {
        require(chainId != 0 && chainId != currentChainId, "Invalid chain ID");
        require(bridgeContract != address(0), "Invalid bridge contract");
        require(bytes(name).length > 0, "Invalid chain name");

        _supportedChains[chainId] = ChainInfo({
            chainId: chainId,
            name: name,
            bridgeContract: bridgeContract,
            isActive: true,
            blockConfirmations: blockConfirmations,
            messageFee: messageFee
        });

        activeChains.push(chainId);
        emit ChainAdded(chainId, name, bridgeContract);
    }

    /**
     * @notice Get supported chain information
     * @param chainId The chain ID to query
     * @return ChainInfo for the requested chain
     */
    function getSupportedChain(uint256 chainId) external view returns (ChainInfo memory) {
        return _supportedChains[chainId];
    }

    /**
     * @notice Send a cross-chain message to an agent on another blockchain
     * @param targetChainId Target blockchain chain ID
     * @param targetAgent Address of the target agent
     * @param messageData Encoded message data
     * @return messageHash Unique identifier for the message
     */
    function sendCrossChainMessage(
        uint256 targetChainId,
        address targetAgent,
        bytes calldata messageData
    ) external payable whenNotPaused nonReentrant returns (bytes32 messageHash) {
        ChainInfo storage targetChain = _supportedChains[targetChainId];
        if (!targetChain.isActive) revert UnsupportedChain(targetChainId);
        if (msg.value < targetChain.messageFee) {
            revert InsufficientFee(targetChain.messageFee, msg.value);
        }

        uint256 nonce = agentNonces[msg.sender]++;
        messageHash = keccak256(abi.encodePacked(
            currentChainId,
            targetChainId,
            msg.sender,
            targetAgent,
            messageData,
            nonce,
            block.timestamp
        ));

        CrossChainMessage storage message = crossChainMessages[messageHash];
        message.sourceChainId = currentChainId;
        message.targetChainId = targetChainId;
        message.sourceAgent = msg.sender;
        message.targetAgent = targetAgent;
        message.messageData = messageData;
        message.messageHash = messageHash;
        message.timestamp = block.timestamp;
        message.nonce = nonce;
        message.status = MessageStatus.Pending;

        emit CrossChainMessageSent(
            messageHash,
            currentChainId,
            targetChainId,
            msg.sender,
            targetAgent
        );
    }

    /**
     * @notice Validate a cross-chain message (validators only)
     * @param messageHash Hash of the message to validate
     * @param signature Validator's signature on the message
     */
    function validateMessage(
        bytes32 messageHash,
        bytes32 signature
    ) external onlyRole(VALIDATOR_ROLE) whenNotPaused {
        CrossChainMessage storage message = crossChainMessages[messageHash];
        if (message.timestamp == 0) revert MessageNotFound(messageHash);
        if (block.timestamp > message.timestamp + MESSAGE_EXPIRY) {
            revert MessageExpired(messageHash);
        }
        if (validatorVotes[msg.sender][uint256(messageHash)]) {
            return; // Already voted
        }

        validatorVotes[msg.sender][uint256(messageHash)] = true;
        message.validatorSignatures.push(signature);

        // Check if we have enough consensus
        // DIVISION BY ZERO PROTECTION: Ensure validator count exists for consensus calculation
        if (validatorCount > 0 && message.validatorSignatures.length >= MIN_VALIDATOR_CONSENSUS) {
            message.status = MessageStatus.Validated;
        }

        emit MessageValidated(messageHash, msg.sender);
    }

    /**
     * @notice Execute a validated cross-chain message
     * @param messageHash Hash of the message to execute
     * @param proof Merkle proof or other cryptographic proof of message validity
     */
    function executeMessage(
        bytes32 messageHash,
        bytes calldata proof
    ) external whenNotPaused nonReentrant {
        CrossChainMessage storage message = crossChainMessages[messageHash];
        if (message.timestamp == 0) revert MessageNotFound(messageHash);
        if (processedMessages[messageHash]) revert MessageAlreadyProcessed(messageHash);
        if (block.timestamp > message.timestamp + MESSAGE_EXPIRY) {
            message.status = MessageStatus.Expired;
            revert MessageExpired(messageHash);
        }

        // Verify sufficient validator consensus
        if (message.validatorSignatures.length < MIN_VALIDATOR_CONSENSUS) {
            revert InsufficientValidatorConsensus(
                MIN_VALIDATOR_CONSENSUS,
                message.validatorSignatures.length
            );
        }

        // Mark as processed to prevent replay
        processedMessages[messageHash] = true;

        // Execute the message by calling the target agent
        bool success = _executeMessageCall(message, proof);
        
        message.status = success ? MessageStatus.Executed : MessageStatus.Failed;
        emit MessageExecuted(messageHash, success);

        // If successful, emit received event
        if (success) {
            emit CrossChainMessageReceived(
                messageHash,
                message.sourceChainId,
                message.sourceAgent,
                message.targetAgent
            );
        }
    }

    /**
     * @notice Internal function to execute the cross-chain message
     * @param message The message to execute
     * @param proof Cryptographic proof of message validity
     * @return success Whether the execution succeeded
     */
    function _executeMessageCall(
        CrossChainMessage storage message,
        bytes calldata proof
    ) internal returns (bool success) {
        // Verify proof (simplified - in production would verify Merkle proof)
        require(proof.length > 0, "Invalid proof");

        // Attempt to call the target agent
        try this.safeExecuteMessage(
            message.targetAgent,
            message.messageData
        ) {
            success = true;
        } catch {
            success = false;
        }
    }

    /**
     * @notice Safe execution wrapper for cross-chain messages
     * @param target Target contract address
     * @param data Message data to execute
     */
    function safeExecuteMessage(
        address target,
        bytes calldata data
    ) external {
        require(msg.sender == address(this), "Only bridge can call");
        require(target != address(0), "Invalid target");
        
        (bool success,) = target.call(data);
        require(success, "Message execution failed");
    }

    /**
     * @notice Get supported chains
     * @param offset Pagination offset
     * @param limit Maximum results
     * @return chainIds Array of chain IDs
     * @return chainInfos Array of chain information
     * @return total Total number of supported chains
     */
    function getSupportedChains(
        uint256 offset,
        uint256 limit
    ) external view returns (
        uint256[] memory chainIds,
        ChainInfo[] memory chainInfos,
        uint256 total
    ) {
        total = activeChains.length;
        if (offset >= total) return (new uint256[](0), new ChainInfo[](0), total);
        
        uint256 end = offset + limit;
        if (end > total) end = total;
        uint256 resultCount = end - offset;
        
        chainIds = new uint256[](resultCount);
        chainInfos = new ChainInfo[](resultCount);
        
        for (uint256 i = 0; i < resultCount; i++) {
            uint256 chainId = activeChains[offset + i];
            chainIds[i] = chainId;
            chainInfos[i] = _supportedChains[chainId];
        }
    }

    /**
     * @notice Get cross-chain message details
     * @param messageHash Hash of the message
     * @return message The cross-chain message
     */
    function getMessage(
        bytes32 messageHash
    ) external view returns (CrossChainMessage memory message) {
        message = crossChainMessages[messageHash];
        if (message.timestamp == 0) revert MessageNotFound(messageHash);
    }

    /**
     * @notice Add a new validator
     * @param validator Address of the new validator
     */
    function addValidator(address validator) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(validator != address(0), "Invalid validator address");
        _grantRole(VALIDATOR_ROLE, validator);
        validatorCount++;
        emit ValidatorAdded(validator);
    }

    /**
     * @notice Remove a validator
     * @param validator Address of the validator to remove
     */
    function removeValidator(address validator) external onlyRole(DEFAULT_ADMIN_ROLE) {
        _revokeRole(VALIDATOR_ROLE, validator);
        if (validatorCount > 0) validatorCount--;
        emit ValidatorRemoved(validator);
    }

    /**
     * @notice Update chain status
     * @param chainId Chain to update
     * @param isActive New status
     */
    function updateChainStatus(
        uint256 chainId,
        bool isActive
    ) external onlyRole(BRIDGE_OPERATOR_ROLE) {
        ChainInfo storage chain = _supportedChains[chainId];
        if (chain.chainId == 0) revert UnsupportedChain(chainId);
        
        chain.isActive = isActive;
        emit ChainUpdated(chainId, isActive);
    }

    /**
     * @notice Emergency pause
     */
    function pause() external onlyRole(PAUSER_ROLE) {
        _pause();
    }

    /**
     * @notice Unpause
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

    /**
     * @notice Get current nonce for an agent
     * @param agent Agent address
     * @return Current nonce
     */
    function getAgentNonce(address agent) external view returns (uint256) {
        return agentNonces[agent];
    }

    /**
     * @notice Check if message has been processed
     * @param messageHash Message hash to check
     * @return True if processed
     */
    function isMessageProcessed(bytes32 messageHash) external view returns (bool) {
        return processedMessages[messageHash];
    }

    // Receive function for fee payments
    receive() external payable {}
}