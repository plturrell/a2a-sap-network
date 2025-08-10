// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "forge-std/Test.sol";
import "../src/AgentRegistry.sol";
import "../src/MessageRouter.sol";

contract EdgeCasesTest is Test {
    AgentRegistry public registry;
    MessageRouter public router;
    
    address public agent1 = address(0x1);
    address public agent2 = address(0x2);
    
    function setUp() public {
        registry = new AgentRegistry(1);
        router = new MessageRouter(address(registry), 1);
        
        // Register test agents
        vm.prank(agent1);
        bytes32[] memory caps = new bytes32[](1);
        caps[0] = keccak256("test");
        registry.registerAgent("Agent1", "http://agent1.com", caps);
        
        vm.prank(agent2);
        registry.registerAgent("Agent2", "http://agent2.com", caps);
    }
    
    function testReputationEdgeCases() public {
        // Test maximum reputation increase
        uint256 maxRep = 200;
        registry.increaseReputation(agent1, maxRep); // Should cap at 200
        assertEq(registry.getAgent(agent1).reputation, maxRep);
        
        // Try to increase beyond maximum
        registry.increaseReputation(agent1, 50);
        assertEq(registry.getAgent(agent1).reputation, maxRep); // Should still be capped
        
        // Test minimum reputation decrease
        registry.decreaseReputation(agent1, 300); // Should floor at 0
        assertEq(registry.getAgent(agent1).reputation, 0);
        
        // Test decrease on zero reputation
        registry.decreaseReputation(agent1, 10); // Should remain 0
        assertEq(registry.getAgent(agent1).reputation, 0);
    }
    
    function testPaginationBoundaryConditions() public {
        // Register exactly 10 agents
        for (uint256 i = 3; i <= 12; i++) {
            address agent = address(uint160(i));
            vm.prank(agent);
            bytes32[] memory caps = new bytes32[](1);
            caps[0] = keccak256("test");
            registry.registerAgent(
                string(abi.encodePacked("Agent", i)), 
                "http://test.com", 
                caps
            );
        }
        
        // Test offset at exact boundary
        (address[] memory page1, uint256 total) = registry.getAgentsPaginated(12, 5);
        assertEq(total, 12);
        assertEq(page1.length, 0); // Should return empty array
        
        // Test offset one before boundary
        (address[] memory page2,) = registry.getAgentsPaginated(11, 5);
        assertEq(page2.length, 1); // Should return 1 item
        
        // Test limit larger than remaining items
        (address[] memory page3,) = registry.getAgentsPaginated(10, 10);
        assertEq(page3.length, 2); // Should return only remaining 2 items
        
        // Test zero limit
        (address[] memory page4,) = registry.getAgentsPaginated(0, 0);
        assertEq(page4.length, 0); // Should return empty array
    }
    
    function testMessageExpiryBoundaryConditions() public {
        // Send message and check exactly at expiry time
        vm.prank(agent1);
        bytes32 messageId = router.sendMessage(agent2, "Test message", keccak256("TEXT"));
        
        // Get the message expiry time
        MessageRouter.Message memory message = router.getMessage(messageId);
        uint256 expiryTime = message.expiresAt;
        
        // At one second before expiry, message should still be valid
        vm.warp(expiryTime - 1);
        assertFalse(router.isMessageExpired(messageId));
        message = router.getMessage(messageId);
        assertEq(message.content, "Test message");
        
        // At exact expiry time + 1 second, message should be expired
        vm.warp(expiryTime + 1);
        assertTrue(router.isMessageExpired(messageId));
        
        // Trying to get expired message should revert
        vm.expectRevert("Message expired");
        router.getMessage(messageId);
        
        // One second after expiry, message should still be expired
        vm.warp(expiryTime + 1);
        assertTrue(router.isMessageExpired(messageId));
    }
    
    function testRateLimitingEdgeCases() public {
        // Test exactly at rate limit boundary
        uint256 maxMessages = 100; // MAX_MESSAGES_PER_WINDOW
        uint256 minDelay = router.messageDelay();
        
        vm.startPrank(agent1);
        
        // Send exactly the maximum number of messages
        for (uint256 i = 0; i < maxMessages; i++) {
            if (i > 0) {
                vm.warp(block.timestamp + minDelay);
            }
            router.sendMessage(agent2, "Message", keccak256("TEXT"));
        }
        
        // The 101st message should fail
        vm.warp(block.timestamp + minDelay);
        vm.expectRevert("MessageRouter: rate limit - too many messages");
        router.sendMessage(agent2, "Overflow message", keccak256("TEXT"));
        
        vm.stopPrank();
    }
    
    function testActiveMessagesWithComplexScenario() public {
        vm.startPrank(agent1);
        
        bytes32[] memory messageIds = new bytes32[](5);
        
        // Send 5 messages with time gaps
        for (uint256 i = 0; i < 5; i++) {
            vm.warp(block.timestamp + 10);
            messageIds[i] = router.sendMessage(agent2, "Message", keccak256("TEXT"));
        }
        
        vm.stopPrank();
        
        // Mark some as delivered
        vm.startPrank(agent2);
        router.markAsDelivered(messageIds[1]);
        router.markAsDelivered(messageIds[3]);
        vm.stopPrank();
        
        // Should have 3 active messages
        bytes32[] memory activeMessages = router.getActiveMessages(agent2, 10);
        assertEq(activeMessages.length, 3);
        
        // Expire all messages
        vm.warp(block.timestamp + 8 days);
        
        // Should have 0 active messages
        bytes32[] memory expiredActive = router.getActiveMessages(agent2, 10);
        assertEq(expiredActive.length, 0);
        
        // But total messages should still be 5
        (,uint256 total) = router.getMessagesPaginated(agent2, 0, 10);
        assertEq(total, 5);
    }
    
    function testCapabilityEdgeCases() public {
        // Test with maximum length capability
        bytes32 longCapability = keccak256("very_long_capability_name_that_tests_boundary_conditions");
        
        vm.prank(agent1);
        bytes32[] memory caps = new bytes32[](1);
        caps[0] = longCapability;
        
        // This should work fine since capabilities are bytes32
        (address[] memory agents,) = registry.getAgentsByCapabilityPaginated(longCapability, 0, 10);
        assertEq(agents.length, 0); // No agents have this capability yet
        
        // Test with empty capability search
        bytes32 emptyCapability = bytes32(0);
        (address[] memory emptyAgents,) = registry.getAgentsByCapabilityPaginated(emptyCapability, 0, 10);
        assertEq(emptyAgents.length, 0);
    }
    
    function testEventEmissionVerification() public {
        // Test AgentRegistered event
        vm.expectEmit(true, false, false, true);
        emit AgentRegistry.AgentRegistered(address(0x123), "TestAgent", "http://test.com");
        
        vm.prank(address(0x123));
        bytes32[] memory caps = new bytes32[](1);
        caps[0] = keccak256("test");
        registry.registerAgent("TestAgent", "http://test.com", caps);
        
        // Test MessageSent event (we can only check the pattern, not exact messageId)
        vm.prank(agent1);
        bytes32 messageId = router.sendMessage(agent2, "Test", keccak256("TEST"));
        
        // Verify the message was actually created
        MessageRouter.Message memory message = router.getMessage(messageId);
        assertEq(message.from, agent1);
        assertEq(message.to, agent2);
        assertEq(message.content, "Test");
    }
    
    function testStateVerificationAfterFailure() public {
        uint256 initialAgentCount = registry.getActiveAgentsCount();
        
        // Try to register with invalid data
        vm.prank(agent1); // agent1 is already registered
        vm.expectRevert("Agent already registered");
        bytes32[] memory caps = new bytes32[](1);
        caps[0] = keccak256("test");
        registry.registerAgent("DuplicateAgent", "http://duplicate.com", caps);
        
        // Verify state unchanged
        assertEq(registry.getActiveAgentsCount(), initialAgentCount);
        
        // Verify original agent data unchanged
        AgentRegistry.Agent memory agent = registry.getAgent(agent1);
        assertEq(agent.name, "Agent1");
        assertEq(agent.endpoint, "http://agent1.com");
    }
}