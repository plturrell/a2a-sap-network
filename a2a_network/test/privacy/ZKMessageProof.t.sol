// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "forge-std/Test.sol";
import "../../src/privacy/ZKMessageProof.sol";

contract ZKMessageProofTest is Test {
    ZKMessageProof public zkProof;
    
    address public admin = makeAddr("admin");
    address public pauser = makeAddr("pauser");
    address public user1 = makeAddr("user1");
    address public user2 = makeAddr("user2");
    
    // Sample proof data
    ZKMessageProof.Proof internal sampleProof;
    ZKMessageProof.VerifyingKey internal sampleVK;
    
    event MessageCommitmentCreated(
        bytes32 indexed commitmentId,
        bytes32 contentHash,
        uint256 timestamp
    );
    event ProofVerified(bytes32 indexed commitmentId, bool success);
    event NullifierUsed(bytes32 indexed nullifier);
    event VerifyingKeyUpdated(address updatedBy);

    function setUp() public {
        vm.startPrank(admin);
        zkProof = new ZKMessageProof(admin, pauser);
        
        // Setup sample proof data
        sampleProof = ZKMessageProof.Proof({
            a: [uint256(1), uint256(2)],
            b: [[uint256(3), uint256(4)], [uint256(5), uint256(6)]],
            c: [uint256(7), uint256(8)]
        });
        
        // Setup sample verifying key with fixed size array
        sampleVK = ZKMessageProof.VerifyingKey({
            alpha: [uint256(1), uint256(2)],
            beta: [[uint256(3), uint256(4)], [uint256(5), uint256(6)]],
            gamma: [[uint256(7), uint256(8)], [uint256(9), uint256(10)]],
            delta: [[uint256(11), uint256(12)], [uint256(13), uint256(14)]],
            ic: [[uint256(9), uint256(10)], [uint256(11), uint256(12)]]
        });
        
        zkProof.updateVerifyingKey(sampleVK);
        vm.stopPrank();
    }

    function testCreateMessageCommitment() public {
        vm.startPrank(user1);
        
        bytes32 contentHash = keccak256("encrypted message");
        bytes32 senderCommitment = keccak256("sender commitment");
        bytes32 recipientCommitment = keccak256("recipient commitment");
        uint256 nonce = zkProof.getCurrentNonce(user1);
        
        vm.expectEmit(false, true, false, true);
        emit MessageCommitmentCreated(bytes32(0), contentHash, block.timestamp);
        
        bytes32 commitmentId = zkProof.createMessageCommitment(
            contentHash,
            senderCommitment,
            recipientCommitment,
            nonce
        );
        
        // Verify commitment was created
        ZKMessageProof.MessageCommitment memory commitment = zkProof.getMessageCommitment(commitmentId);
        assertEq(commitment.contentHash, contentHash);
        assertEq(commitment.senderCommitment, senderCommitment);
        assertEq(commitment.recipientCommitment, recipientCommitment);
        assertEq(commitment.nonce, nonce);
        assertFalse(commitment.verified);
        
        // Verify nonce was incremented
        assertEq(zkProof.getCurrentNonce(user1), nonce + 1);
        
        vm.stopPrank();
    }

    function testCreateCommitmentRejectsInvalidInputs() public {
        vm.startPrank(user1);
        uint256 nonce = zkProof.getCurrentNonce(user1);
        
        // Test invalid content hash
        vm.expectRevert("Invalid content hash");
        zkProof.createMessageCommitment(
            bytes32(0),
            keccak256("sender"),
            keccak256("recipient"),
            nonce
        );
        
        // Test invalid sender commitment
        vm.expectRevert("Invalid sender commitment");
        zkProof.createMessageCommitment(
            keccak256("content"),
            bytes32(0),
            keccak256("recipient"),
            nonce
        );
        
        // Test invalid recipient commitment
        vm.expectRevert("Invalid recipient commitment");
        zkProof.createMessageCommitment(
            keccak256("content"),
            keccak256("sender"),
            bytes32(0),
            nonce
        );
        
        // Test invalid nonce
        vm.expectRevert("Invalid nonce");
        zkProof.createMessageCommitment(
            keccak256("content"),
            keccak256("sender"),
            keccak256("recipient"),
            nonce + 1
        );
        
        vm.stopPrank();
    }

    function testVerifyMessageProof() public {
        vm.startPrank(user1);
        
        // Create commitment
        bytes32 commitmentId = zkProof.createMessageCommitment(
            keccak256("content"),
            keccak256("sender"),
            keccak256("recipient"),
            zkProof.getCurrentNonce(user1)
        );
        
        // Prepare proof verification data
        uint256[] memory publicInputs = new uint256[](2);
        publicInputs[0] = 100;
        publicInputs[1] = 200;
        bytes32 nullifier = keccak256("unique nullifier");
        
        vm.expectEmit(true, false, false, true);
        emit ProofVerified(commitmentId, true);
        
        vm.expectEmit(true, false, false, false);
        emit NullifierUsed(nullifier);
        
        zkProof.verifyMessageProof(
            commitmentId,
            sampleProof,
            publicInputs,
            nullifier
        );
        
        // Verify commitment is now verified
        ZKMessageProof.MessageCommitment memory commitment = zkProof.getMessageCommitment(commitmentId);
        assertTrue(commitment.verified);
        
        // Verify nullifier is marked as used
        assertTrue(zkProof.isNullifierUsed(nullifier));
        
        vm.stopPrank();
    }

    function testVerifyProofRejectsInvalidInputs() public {
        vm.startPrank(user1);
        
        bytes32 commitmentId = zkProof.createMessageCommitment(
            keccak256("content"),
            keccak256("sender"),
            keccak256("recipient"),
            zkProof.getCurrentNonce(user1)
        );
        
        uint256[] memory publicInputs = new uint256[](2);
        publicInputs[0] = 100;
        publicInputs[1] = 200;
        
        // Test non-existent commitment
        vm.expectRevert(ZKMessageProof.CommitmentNotFound.selector);
        zkProof.verifyMessageProof(
            bytes32("nonexistent"),
            sampleProof,
            publicInputs,
            keccak256("nullifier")
        );
        
        // Test invalid nullifier
        vm.expectRevert(ZKMessageProof.InvalidNullifier.selector);
        zkProof.verifyMessageProof(
            commitmentId,
            sampleProof,
            publicInputs,
            bytes32(0)
        );
        
        // Verify with valid nullifier first
        bytes32 nullifier = keccak256("nullifier");
        zkProof.verifyMessageProof(
            commitmentId,
            sampleProof,
            publicInputs,
            nullifier
        );
        
        // Test already verified commitment
        vm.expectRevert(ZKMessageProof.ProofAlreadyVerified.selector);
        zkProof.verifyMessageProof(
            commitmentId,
            sampleProof,
            publicInputs,
            keccak256("different nullifier")
        );
        
        vm.stopPrank();
    }

    function testVerifyProofRejectsUsedNullifier() public {
        vm.startPrank(user1);
        
        // Create two commitments
        bytes32 commitmentId1 = zkProof.createMessageCommitment(
            keccak256("content1"),
            keccak256("sender"),
            keccak256("recipient"),
            zkProof.getCurrentNonce(user1)
        );
        
        bytes32 commitmentId2 = zkProof.createMessageCommitment(
            keccak256("content2"),
            keccak256("sender"),
            keccak256("recipient"),
            zkProof.getCurrentNonce(user1)
        );
        
        uint256[] memory publicInputs = new uint256[](2);
        publicInputs[0] = 100;
        publicInputs[1] = 200;
        bytes32 nullifier = keccak256("shared nullifier");
        
        // Use nullifier on first commitment
        zkProof.verifyMessageProof(
            commitmentId1,
            sampleProof,
            publicInputs,
            nullifier
        );
        
        // Try to reuse same nullifier on second commitment
        vm.expectRevert(ZKMessageProof.NullifierAlreadyUsed.selector);
        zkProof.verifyMessageProof(
            commitmentId2,
            sampleProof,
            publicInputs,
            nullifier
        );
        
        vm.stopPrank();
    }

    function testUpdateVerifyingKey() public {
        vm.startPrank(admin);
        
        // Create new verifying key
        ZKMessageProof.VerifyingKey memory newVK = ZKMessageProof.VerifyingKey({
            alpha: [uint256(999), uint256(888)],
            beta: [[uint256(777), uint256(666)], [uint256(555), uint256(444)]],
            gamma: [[uint256(333), uint256(222)], [uint256(111), uint256(999)]],
            delta: [[uint256(888), uint256(777)], [uint256(666), uint256(555)]],
            ic: [[uint256(999), uint256(888)], [uint256(777), uint256(666)]]
        });
        
        vm.expectEmit(false, false, false, true);
        emit VerifyingKeyUpdated(admin);
        
        zkProof.updateVerifyingKey(newVK);
        
        // Key update verified through event emission
        // Direct verification would require getter function
        
        vm.stopPrank();
    }

    function testUpdateVerifyingKeyRejectsInvalidKey() public {
        vm.startPrank(admin);
        
        ZKMessageProof.VerifyingKey memory invalidVK = ZKMessageProof.VerifyingKey({
            alpha: [uint256(0), uint256(0)], // Invalid - zero alpha
            beta: [[uint256(1), uint256(2)], [uint256(3), uint256(4)]],
            gamma: [[uint256(5), uint256(6)], [uint256(7), uint256(8)]],
            delta: [[uint256(9), uint256(10)], [uint256(11), uint256(12)]],
            ic: [[uint256(0), uint256(0)], [uint256(0), uint256(0)]]
        });
        
        vm.expectRevert(ZKMessageProof.InvalidVerifyingKey.selector);
        zkProof.updateVerifyingKey(invalidVK);
        
        vm.stopPrank();
    }

    function testUpdateVerifyingKeyRequiresAdmin() public {
        vm.startPrank(user1);
        
        vm.expectRevert();
        zkProof.updateVerifyingKey(sampleVK);
        
        vm.stopPrank();
    }

    function testPauseAndUnpause() public {
        vm.startPrank(pauser);
        
        // Pause contract
        zkProof.pause();
        assertTrue(zkProof.paused());
        
        // Test operations fail when paused
        vm.stopPrank();
        vm.startPrank(user1);
        
        vm.expectRevert();
        zkProof.createMessageCommitment(
            keccak256("content"),
            keccak256("sender"),
            keccak256("recipient"),
            0
        );
        
        vm.stopPrank();
        vm.startPrank(pauser);
        
        // Unpause contract
        zkProof.unpause();
        assertFalse(zkProof.paused());
        
        vm.stopPrank();
    }

    function testPauseRequiresPauserRole() public {
        vm.startPrank(user1);
        
        vm.expectRevert();
        zkProof.pause();
        
        vm.expectRevert();
        zkProof.unpause();
        
        vm.stopPrank();
    }

    function testGetCommitmentNotFound() public {
        vm.expectRevert(ZKMessageProof.CommitmentNotFound.selector);
        zkProof.getMessageCommitment(bytes32("nonexistent"));
    }

    function testGetCurrentNonce() public {
        assertEq(zkProof.getCurrentNonce(user1), 0);
        assertEq(zkProof.getCurrentNonce(user2), 0);
        
        vm.prank(user1);
        zkProof.createMessageCommitment(
            keccak256("content"),
            keccak256("sender"),
            keccak256("recipient"),
            0
        );
        
        assertEq(zkProof.getCurrentNonce(user1), 1);
        assertEq(zkProof.getCurrentNonce(user2), 0);
    }

    function testIsNullifierUsed() public {
        bytes32 nullifier = keccak256("test nullifier");
        assertFalse(zkProof.isNullifierUsed(nullifier));
        
        vm.startPrank(user1);
        bytes32 commitmentId = zkProof.createMessageCommitment(
            keccak256("content"),
            keccak256("sender"),
            keccak256("recipient"),
            zkProof.getCurrentNonce(user1)
        );
        
        uint256[] memory publicInputs = new uint256[](2);
        publicInputs[0] = 100;
        publicInputs[1] = 200;
        
        zkProof.verifyMessageProof(
            commitmentId,
            sampleProof,
            publicInputs,
            nullifier
        );
        
        assertTrue(zkProof.isNullifierUsed(nullifier));
        vm.stopPrank();
    }

    function testFuzzCreateCommitment(
        bytes32 contentHash,
        bytes32 senderCommitment,
        bytes32 recipientCommitment
    ) public {
        vm.assume(contentHash != bytes32(0));
        vm.assume(senderCommitment != bytes32(0));
        vm.assume(recipientCommitment != bytes32(0));
        
        vm.startPrank(user1);
        uint256 nonce = zkProof.getCurrentNonce(user1);
        
        bytes32 commitmentId = zkProof.createMessageCommitment(
            contentHash,
            senderCommitment,
            recipientCommitment,
            nonce
        );
        
        ZKMessageProof.MessageCommitment memory commitment = zkProof.getMessageCommitment(commitmentId);
        assertEq(commitment.contentHash, contentHash);
        assertEq(commitment.senderCommitment, senderCommitment);
        assertEq(commitment.recipientCommitment, recipientCommitment);
        assertEq(commitment.nonce, nonce);
        
        vm.stopPrank();
    }
}