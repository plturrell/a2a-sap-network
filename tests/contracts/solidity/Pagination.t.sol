// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "forge-std/Test.sol";
import "../src/AgentRegistry.sol";
import "../src/MessageRouter.sol";

contract PaginationTest is Test {
    AgentRegistry public registry;
    MessageRouter public router;
    
    function setUp() public {
        registry = new AgentRegistry(1);
        router = new MessageRouter(address(registry), 1);
    }
    
    function testAgentPagination() public {
        // Register 10 agents
        for (uint256 i = 0; i < 10; i++) {
            address agent = address(uint160(i + 1));
            vm.prank(agent);
            bytes32[] memory caps = new bytes32[](1);
            caps[0] = keccak256("test");
            registry.registerAgent(
                string(abi.encodePacked("Agent", i)), 
                string(abi.encodePacked("http://agent", i, ".com")),
                caps
            );
        }
        
        // Test getting first page
        (address[] memory page1, uint256 total) = registry.getAgentsPaginated(0, 5);
        assertEq(total, 10);
        assertEq(page1.length, 5);
        assertEq(page1[0], address(1));
        assertEq(page1[4], address(5));
        
        // Test getting second page
        (address[] memory page2, ) = registry.getAgentsPaginated(5, 5);
        assertEq(page2.length, 5);
        assertEq(page2[0], address(6));
        assertEq(page2[4], address(10));
        
        // Test partial page
        (address[] memory page3, ) = registry.getAgentsPaginated(8, 5);
        assertEq(page3.length, 2);
        assertEq(page3[0], address(9));
        assertEq(page3[1], address(10));
        
        // Test out of bounds
        (address[] memory page4, ) = registry.getAgentsPaginated(10, 5);
        assertEq(page4.length, 0);
    }
    
    function testCapabilityPagination() public {
        bytes32 capability = keccak256("messaging");
        
        // Register agents with same capability
        for (uint256 i = 0; i < 7; i++) {
            address agent = address(uint160(i + 1));
            vm.prank(agent);
            bytes32[] memory caps = new bytes32[](1);
            caps[0] = capability;
            registry.registerAgent(
                string(abi.encodePacked("Agent", i)), 
                "http://test.com",
                caps
            );
        }
        
        // Test pagination
        (address[] memory page1, uint256 total) = registry.getAgentsByCapabilityPaginated(capability, 0, 3);
        assertEq(total, 7);
        assertEq(page1.length, 3);
        
        (address[] memory page2, ) = registry.getAgentsByCapabilityPaginated(capability, 3, 3);
        assertEq(page2.length, 3);
        
        (address[] memory page3, ) = registry.getAgentsByCapabilityPaginated(capability, 6, 3);
        assertEq(page3.length, 1);
    }
    
    function testMessagePagination() public {
        // Register two agents
        address sender = address(0x1);
        address receiver = address(0x2);
        
        vm.prank(sender);
        bytes32[] memory caps = new bytes32[](1);
        caps[0] = keccak256("test");
        registry.registerAgent("Sender", "http://sender.com", caps);
        
        vm.prank(receiver);
        registry.registerAgent("Receiver", "http://receiver.com", caps);
        
        // Send multiple messages
        vm.startPrank(sender);
        for (uint256 i = 0; i < 15; i++) {
            router.sendMessage(
                receiver, 
                string(abi.encodePacked("Message ", i)), 
                keccak256("TEXT")
            );
            vm.warp(block.timestamp + 6); // Advance time to avoid rate limit
        }
        vm.stopPrank();
        
        // Test message pagination
        (bytes32[] memory page1, uint256 total) = router.getMessagesPaginated(receiver, 0, 10);
        assertEq(total, 15);
        assertEq(page1.length, 10);
        
        (bytes32[] memory page2, ) = router.getMessagesPaginated(receiver, 10, 10);
        assertEq(page2.length, 5);
    }
}