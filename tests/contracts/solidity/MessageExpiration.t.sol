// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "forge-std/Test.sol";
import "../src/AgentRegistry.sol";
import "../src/MessageRouter.sol";

contract MessageExpirationTest is Test {
    MessageRouter public router;
    AgentRegistry public registry;
    
    address public agent1 = address(0x1);
    address public agent2 = address(0x2);
    bytes32 public messageId;
    
    function setUp() public {
        registry = new AgentRegistry(1);
        router = new MessageRouter(address(registry), 1);
        
        // Register agents
        vm.prank(agent1);
        bytes32[] memory caps = new bytes32[](1);
        caps[0] = keccak256("test");
        registry.registerAgent("Agent1", "http://agent1.com", caps);
        
        vm.prank(agent2);
        registry.registerAgent("Agent2", "http://agent2.com", caps);
        
        // Send a message
        vm.prank(agent1);
        messageId = router.sendMessage(agent2, "Test message", keccak256("TEXT"));
    }
    
    function testMessageNotExpiredInitially() public {
        assertFalse(router.isMessageExpired(messageId));
        
        // Can still get the message
        MessageRouter.Message memory message = router.getMessage(messageId);
        assertEq(message.content, "Test message");
    }
    
    function testMessageExpiresAfterTime() public {
        // Fast forward past expiry time (7 days default)
        vm.warp(block.timestamp + 8 days);
        
        assertTrue(router.isMessageExpired(messageId));
        
        // Cannot get expired message
        vm.expectRevert("Message expired");
        router.getMessage(messageId);
    }
    
    function testUpdateMessageExpiry() public {
        // Update expiry to 2 days
        vm.prank(address(this)); // deployer is admin
        router.updateMessageExpiry(2 days);
        
        // Send new message
        vm.prank(agent1);
        vm.warp(block.timestamp + 10); // Advance to avoid rate limit
        bytes32 newMessageId = router.sendMessage(agent2, "New message", keccak256("TEXT"));
        
        // Should expire after 2 days
        vm.warp(block.timestamp + 3 days);
        assertTrue(router.isMessageExpired(newMessageId));
    }
    
    function testGetActiveMessages() public {
        // Send multiple messages
        vm.startPrank(agent1);
        
        bytes32[] memory sentMessages = new bytes32[](5);
        for (uint256 i = 0; i < 5; i++) {
            vm.warp(block.timestamp + 10); // Advance time
            sentMessages[i] = router.sendMessage(
                agent2, 
                string(abi.encodePacked("Message ", i)), 
                keccak256("TEXT")
            );
        }
        
        vm.stopPrank();
        
        // Mark some as delivered
        vm.startPrank(agent2);
        router.markAsDelivered(sentMessages[0]);
        router.markAsDelivered(sentMessages[2]);
        vm.stopPrank();
        
        // Get active messages
        bytes32[] memory activeMessages = router.getActiveMessages(agent2, 10);
        assertEq(activeMessages.length, 4); // 1 from setUp + 5 sent - 2 delivered = 4 active
        
        // Verify the active messages are the undelivered ones (including setUp message)
        bool foundSetup = false;
        bool found1 = false;
        bool found3 = false; 
        bool found4 = false;
        for (uint256 i = 0; i < activeMessages.length; i++) {
            if (activeMessages[i] == messageId) foundSetup = true; // setUp message
            if (activeMessages[i] == sentMessages[1]) found1 = true;
            if (activeMessages[i] == sentMessages[3]) found3 = true;
            if (activeMessages[i] == sentMessages[4]) found4 = true;
        }
        assertTrue(foundSetup && found1 && found3 && found4);
        
        // Expire some messages
        vm.warp(block.timestamp + 8 days);
        
        // Should return 0 active messages (all expired)
        bytes32[] memory activeAfterExpiry = router.getActiveMessages(agent2, 10);
        assertEq(activeAfterExpiry.length, 0);
    }
    
    function testActiveMessagesWithLimit() public {
        // Send many messages
        vm.startPrank(agent1);
        for (uint256 i = 0; i < 20; i++) {
            vm.warp(block.timestamp + 10);
            router.sendMessage(agent2, "Message", keccak256("TEXT"));
        }
        vm.stopPrank();
        
        // Get only 5 active messages
        bytes32[] memory limitedActive = router.getActiveMessages(agent2, 5);
        assertEq(limitedActive.length, 5);
    }
    
    function testInvalidExpiryUpdate() public {
        // Too short
        vm.expectRevert("Invalid expiry duration");
        router.updateMessageExpiry(12 hours);
        
        // Too long
        vm.expectRevert("Invalid expiry duration");
        router.updateMessageExpiry(31 days);
        
        // Only admin can update
        uint256 initialExpiry = router.messageExpiry();
        
        vm.prank(agent1);
        vm.expectRevert(
            abi.encodeWithSelector(
                bytes4(keccak256("AccessControlUnauthorizedAccount(address,bytes32)")),
                agent1,
                bytes32(0) // DEFAULT_ADMIN_ROLE
            )
        );
        router.updateMessageExpiry(5 days);
        
        // Verify expiry unchanged
        assertEq(router.messageExpiry(), initialExpiry);
    }
}