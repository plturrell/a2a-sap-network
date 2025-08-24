// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "forge-std/Test.sol";
import "../../../a2aNetwork/src/MessageRouter.sol";
import "../../../a2aNetwork/src/AgentRegistry.sol";

contract MessageRouterComprehensiveTest is Test {
    MessageRouter public router;
    AgentRegistry public registry;
    
    address public admin = address(0x1);
    address public agent1 = address(0x2);
    address public agent2 = address(0x3);
    address public spammer = address(0x4);
    
    bytes32[] public capabilities;
    
    event MessageSent(bytes32 indexed messageId, address indexed from, address indexed to, bytes32 messageType);
    event MessageDelivered(bytes32 indexed messageId);
    event RateLimitUpdated(uint256 newDelay);
    
    function setUp() public {
        vm.startPrank(admin);
        
        registry = new AgentRegistry(1);
        router = new MessageRouter(address(registry), 1);
        
        capabilities.push(keccak256("MESSAGING"));
        
        vm.stopPrank();
        
        // Register test agents
        vm.startPrank(agent1);
        registry.registerAgent("Agent1", "https://agent1.com", capabilities);
        vm.stopPrank();
        
        vm.startPrank(agent2);
        registry.registerAgent("Agent2", "https://agent2.com", capabilities);
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Message sending edge cases
    function testMessageSendingSuccess() public {
        vm.startPrank(agent1);
        
        bytes32 messageType = keccak256("TEST_MESSAGE");
        string memory content = "Hello Agent2";
        
        vm.expectEmit(true, true, true, true);
        emit MessageSent(bytes32(0), agent1, agent2, messageType);
        
        bytes32 messageId = router.sendMessage(agent2, content, messageType);
        
        MessageRouter.Message memory message = router.getMessage(messageId);
        assertEq(message.from, agent1);
        assertEq(message.to, agent2);
        assertEq(message.content, content);
        assertEq(message.messageType, messageType);
        assertFalse(message.delivered);
        
        vm.stopPrank();
    }
    
    function testMessageSendingFailures() public {
        vm.startPrank(agent1);
        
        // Test empty content
        vm.expectRevert("Content required");
        router.sendMessage(agent2, "", keccak256("TEST"));
        
        // Test inactive recipient
        vm.startPrank(agent2);
        registry.deactivateAgent();
        vm.stopPrank();
        
        vm.startPrank(agent1);
        vm.expectRevert("Recipient not active");
        router.sendMessage(agent2, "test", keccak256("TEST"));
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Rate limiting edge cases
    function testRateLimitingBoundaries() public {
        vm.startPrank(agent1);
        
        // First message should succeed
        bytes32 messageId1 = router.sendMessage(agent2, "Message 1", keccak256("TEST"));
        assertTrue(messageId1 != bytes32(0));
        
        // Second message within delay should fail
        vm.expectRevert("MessageRouter: rate limit - too frequent");
        router.sendMessage(agent2, "Message 2", keccak256("TEST"));
        
        // Advance time past delay
        vm.warp(block.timestamp + 6 seconds);
        
        // Should succeed now
        bytes32 messageId2 = router.sendMessage(agent2, "Message 2", keccak256("TEST"));
        assertTrue(messageId2 != bytes32(0));
        
        vm.stopPrank();
    }
    
    function testRateLimitingWindow() public {
        vm.startPrank(agent1);
        
        // Send messages up to the limit
        for (uint256 i = 0; i < 100; i++) {
            if (i > 0) {
                vm.warp(block.timestamp + 6 seconds);
            }
            router.sendMessage(agent2, string(abi.encodePacked("Message ", vm.toString(i))), keccak256("TEST"));
        }
        
        // Next message should fail due to window limit
        vm.warp(block.timestamp + 6 seconds);
        vm.expectRevert("MessageRouter: rate limit - too many messages");
        router.sendMessage(agent2, "Overflow message", keccak256("TEST"));
        
        // Advance time past window
        vm.warp(block.timestamp + 1 hours + 1 seconds);
        
        // Should succeed now
        bytes32 messageId = router.sendMessage(agent2, "After window", keccak256("TEST"));
        assertTrue(messageId != bytes32(0));
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Message delivery and expiry
    function testMessageDeliveryAndExpiry() public {
        vm.startPrank(agent1);
        bytes32 messageId = router.sendMessage(agent2, "Test message", keccak256("TEST"));
        vm.stopPrank();
        
        // Agent2 marks as delivered
        vm.startPrank(agent2);
        
        vm.expectEmit(true, false, false, false);
        emit MessageDelivered(messageId);
        router.markAsDelivered(messageId);
        
        MessageRouter.Message memory message = router.getMessage(messageId);
        assertTrue(message.delivered);
        
        // Cannot mark as delivered again
        vm.expectRevert("Already delivered");
        router.markAsDelivered(messageId);
        
        vm.stopPrank();
        
        // Test message expiry
        vm.startPrank(agent1);
        bytes32 expiredMessageId = router.sendMessage(agent2, "Expired message", keccak256("TEST"));
        vm.stopPrank();
        
        // Advance time past expiry
        vm.warp(block.timestamp + 8 days);
        
        vm.expectRevert("Message expired");
        router.getMessage(expiredMessageId);
    }
    
    // CRITICAL TEST: Unauthorized access
    function testUnauthorizedAccess() public {
        vm.startPrank(agent1);
        bytes32 messageId = router.sendMessage(agent2, "Test message", keccak256("TEST"));
        vm.stopPrank();
        
        // Unauthorized agent tries to mark as delivered
        vm.startPrank(spammer);
        vm.expectRevert("Not message recipient");
        router.markAsDelivered(messageId);
        vm.stopPrank();
        
        // Unregistered agent tries to send message
        vm.startPrank(spammer);
        vm.expectRevert("Agent not registered or inactive");
        router.sendMessage(agent1, "Spam message", keccak256("SPAM"));
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Message retrieval functions
    function testMessageRetrievalFunctions() public {
        vm.startPrank(agent1);
        
        // Send multiple messages
        bytes32[] memory messageIds = new bytes32[](3);
        for (uint256 i = 0; i < 3; i++) {
            vm.warp(block.timestamp + 6 seconds);
            messageIds[i] = router.sendMessage(agent2, string(abi.encodePacked("Message ", vm.toString(i))), keccak256("TEST"));
        }
        
        vm.stopPrank();
        
        // Test getMessages
        bytes32[] memory allMessages = router.getMessages(agent2);
        assertEq(allMessages.length, 3);
        
        // Test getUndeliveredMessages
        bytes32[] memory undelivered = router.getUndeliveredMessages(agent2);
        assertEq(undelivered.length, 3);
        
        // Mark one as delivered
        vm.startPrank(agent2);
        router.markAsDelivered(messageIds[1]);
        vm.stopPrank();
        
        // Check undelivered count
        undelivered = router.getUndeliveredMessages(agent2);
        assertEq(undelivered.length, 2);
        
        // Test pagination
        (bytes32[] memory paginatedMessages, uint256 total) = router.getMessagesPaginated(agent2, 0, 2);
        assertEq(paginatedMessages.length, 2);
        assertEq(total, 3);
        
        // Test getActiveMessages
        bytes32[] memory activeMessages = router.getActiveMessages(agent2, 10);
        assertEq(activeMessages.length, 2); // 2 undelivered, non-expired
    }
    
    // CRITICAL TEST: Admin functions
    function testAdminFunctions() public {
        vm.startPrank(admin);
        
        // Test updating message delay
        vm.expectEmit(false, false, false, true);
        emit RateLimitUpdated(10 seconds);
        router.updateMessageDelay(10 seconds);
        
        // Test invalid delay bounds
        vm.expectRevert("Invalid delay");
        router.updateMessageDelay(0);
        
        vm.expectRevert("Invalid delay");
        router.updateMessageDelay(2 hours);
        
        // Test updating message expiry
        router.updateMessageExpiry(14 days);
        
        // Test invalid expiry bounds
        vm.expectRevert("Invalid expiry duration");
        router.updateMessageExpiry(12 hours);
        
        vm.expectRevert("Invalid expiry duration");
        router.updateMessageExpiry(31 days);
        
        vm.stopPrank();
        
        // Test unauthorized admin access
        vm.startPrank(agent1);
        vm.expectRevert();
        router.updateMessageDelay(5 seconds);
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Message expiry edge cases
    function testMessageExpiryEdgeCases() public {
        vm.startPrank(agent1);
        bytes32 messageId = router.sendMessage(agent2, "Test message", keccak256("TEST"));
        vm.stopPrank();
        
        // Message should not be expired initially
        assertFalse(router.isMessageExpired(messageId));
        
        // Advance to just before expiry
        vm.warp(block.timestamp + 7 days - 1 seconds);
        assertFalse(router.isMessageExpired(messageId));
        
        // Advance past expiry
        vm.warp(block.timestamp + 2 seconds);
        assertTrue(router.isMessageExpired(messageId));
        
        // Test non-existent message
        bytes32 fakeMessageId = keccak256("fake");
        assertFalse(router.isMessageExpired(fakeMessageId));
    }
    
    // CRITICAL TEST: Gas optimization verification
    function testGasOptimization() public {
        vm.startPrank(agent1);
        
        // Test that message sending gas usage is reasonable
        uint256 gasBefore = gasleft();
        router.sendMessage(agent2, "Test message for gas", keccak256("GAS_TEST"));
        uint256 gasUsed = gasBefore - gasleft();
        
        // Should use less than 200k gas for a simple message
        assertTrue(gasUsed < 200000);
        
        vm.stopPrank();
    }
}
