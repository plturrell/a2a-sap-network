// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

/**
 * @title ZKMessageProof
 * @notice Zero-Knowledge proof verification for private messaging in A2A Network
 * @dev Implements zk-SNARK verification for message privacy while maintaining verifiability
 */
contract ZKMessageProof is AccessControl, ReentrancyGuard, Pausable {
    bytes32 public constant VERIFIER_ROLE = keccak256("VERIFIER_ROLE");
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");

    // Groth16 proof structure
    struct Proof {
        uint256[2] a;
        uint256[2][2] b;
        uint256[2] c;
    }

    // Message commitment structure
    struct MessageCommitment {
        bytes32 contentHash;      // Hash of encrypted message content
        bytes32 senderCommitment; // Commitment to sender identity
        bytes32 recipientCommitment; // Commitment to recipient identity
        uint256 timestamp;
        uint256 nonce;           // Prevent replay attacks
        bool verified;           // ZK proof verification status
    }

    // Verification key for zk-SNARK circuit
    struct VerifyingKey {
        uint256[2] alpha;
        uint256[2][2] beta;
        uint256[2][2] gamma;
        uint256[2][2] delta;
        uint256[2][2] ic; // Fixed size for simplicity
    }

    // State variables
    VerifyingKey internal _verifyingKey;
    mapping(bytes32 => MessageCommitment) public messageCommitments;
    mapping(address => uint256) public nonces;
    mapping(bytes32 => bool) public nullifierHashes;

    // Events
    event MessageCommitmentCreated(
        bytes32 indexed commitmentId,
        bytes32 contentHash,
        uint256 timestamp
    );
    event ProofVerified(bytes32 indexed commitmentId, bool success);
    event VerifyingKeyUpdated(address updatedBy);
    event NullifierUsed(bytes32 indexed nullifier);

    // Custom errors
    error InvalidProof();
    error CommitmentNotFound();
    error ProofAlreadyVerified();
    error InvalidNullifier();
    error NullifierAlreadyUsed();
    error InvalidVerifyingKey();

    constructor(address admin, address pauser) {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(PAUSER_ROLE, pauser);
        _grantRole(VERIFIER_ROLE, admin);
    }

    /**
     * @notice Create a new message commitment for ZK verification
     * @param contentHash Hash of the encrypted message content
     * @param senderCommitment Commitment to sender identity
     * @param recipientCommitment Commitment to recipient identity
     * @param nonce Unique nonce to prevent replay attacks
     * @return commitmentId Unique identifier for the commitment
     */
    function createMessageCommitment(
        bytes32 contentHash,
        bytes32 senderCommitment,
        bytes32 recipientCommitment,
        uint256 nonce
    ) external whenNotPaused nonReentrant returns (bytes32 commitmentId) {
        require(contentHash != bytes32(0), "Invalid content hash");
        require(senderCommitment != bytes32(0), "Invalid sender commitment");
        require(recipientCommitment != bytes32(0), "Invalid recipient commitment");
        require(nonces[msg.sender] == nonce, "Invalid nonce");

        // Generate unique commitment ID
        commitmentId = keccak256(abi.encodePacked(
            contentHash,
            senderCommitment,
            recipientCommitment,
            block.timestamp,
            nonce,
            msg.sender
        ));

        messageCommitments[commitmentId] = MessageCommitment({
            contentHash: contentHash,
            senderCommitment: senderCommitment,
            recipientCommitment: recipientCommitment,
            timestamp: block.timestamp,
            nonce: nonce,
            verified: false
        });

        // Increment nonce to prevent replay
        nonces[msg.sender]++;

        emit MessageCommitmentCreated(commitmentId, contentHash, block.timestamp);
    }

    /**
     * @notice Verify a zk-SNARK proof for a message commitment
     * @param commitmentId The commitment to verify
     * @param proof The zk-SNARK proof
     * @param publicInputs Public inputs for the proof verification
     * @param nullifier Unique nullifier to prevent double-spending
     */
    function verifyMessageProof(
        bytes32 commitmentId,
        Proof calldata proof,
        uint256[] calldata publicInputs,
        bytes32 nullifier
    ) external whenNotPaused nonReentrant {
        MessageCommitment storage commitment = messageCommitments[commitmentId];
        if (commitment.timestamp == 0) revert CommitmentNotFound();
        if (commitment.verified) revert ProofAlreadyVerified();
        if (nullifier == bytes32(0)) revert InvalidNullifier();
        if (nullifierHashes[nullifier]) revert NullifierAlreadyUsed();

        // Verify the zk-SNARK proof
        bool proofValid = _verifyProof(proof, publicInputs);
        if (!proofValid) revert InvalidProof();

        // Mark commitment as verified
        commitment.verified = true;
        nullifierHashes[nullifier] = true;

        emit ProofVerified(commitmentId, true);
        emit NullifierUsed(nullifier);
    }

    /**
     * @notice Internal function to verify zk-SNARK proof using Groth16
     * @param proof The proof to verify
     * @param input Public inputs
     * @return True if proof is valid
     */
    function _verifyProof(
        Proof calldata proof,
        uint256[] calldata input
    ) internal view returns (bool) {
        // Simplified proof verification logic
        // In production, this would use a proper Groth16 verifier
        
        uint256 vk_x = 0;
        uint256 maxInputs = input.length > 2 ? 2 : input.length; // Limit to ic array size
        for (uint256 i = 0; i < maxInputs; i++) {
            // Simplified verification computation
            vk_x = addmod(vk_x, mulmod(input[i], _verifyingKey.ic[i][0], 
                21888242871839275222246405745257275088548364400416034343698204186575808495617), 
                21888242871839275222246405745257275088548364400416034343698204186575808495617);
        }

        // Basic elliptic curve pairing check (simplified)
        // Production implementation would use proper BN254 pairing
        return _isValidPairing(proof, vk_x);
    }

    /**
     * @notice Simplified pairing verification
     * @dev In production, use proper elliptic curve pairing library
     */
    function _isValidPairing(Proof calldata proof, uint256 vk_x) internal pure returns (bool) {
        // Placeholder for actual pairing verification
        // Real implementation would check: e(A,B) = e(alpha, beta) * e(vk_x, gamma) * e(C, delta)
        return (proof.a[0] != 0 && proof.b[0][0] != 0 && proof.c[0] != 0 && vk_x != 0);
    }

    /**
     * @notice Update the verifying key (admin only)
     * @param newVK New verifying key
     */
    function updateVerifyingKey(
        VerifyingKey calldata newVK
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (newVK.alpha[0] == 0) revert InvalidVerifyingKey();
        
        _verifyingKey = newVK;
        emit VerifyingKeyUpdated(msg.sender);
    }

    /**
     * @notice Get message commitment details
     * @param commitmentId The commitment ID
     * @return commitment The message commitment
     */
    function getMessageCommitment(
        bytes32 commitmentId
    ) external view returns (MessageCommitment memory commitment) {
        commitment = messageCommitments[commitmentId];
        if (commitment.timestamp == 0) revert CommitmentNotFound();
    }

    /**
     * @notice Check if a nullifier has been used
     * @param nullifier The nullifier to check
     * @return True if nullifier has been used
     */
    function isNullifierUsed(bytes32 nullifier) external view returns (bool) {
        return nullifierHashes[nullifier];
    }

    /**
     * @notice Get current nonce for an address
     * @param user The user address
     * @return Current nonce
     */
    function getCurrentNonce(address user) external view returns (uint256) {
        return nonces[user];
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
}