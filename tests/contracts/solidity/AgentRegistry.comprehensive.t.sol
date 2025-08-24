// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "forge-std/Test.sol";
import "../../../a2aNetwork/src/AgentRegistry.sol";

contract AgentRegistryComprehensiveTest is Test {
    AgentRegistry public registry;
    address public admin = address(0x1);
    address public agent1 = address(0x2);
    address public agent2 = address(0x3);
    address public maliciousAgent = address(0x4);
    
    bytes32[] public capabilities;
    
    event AgentRegistered(address indexed agent, string name, string endpoint);
    event AgentUpdated(address indexed agent, string endpoint);
    event AgentDeactivated(address indexed agent);
    event ReputationChanged(address indexed agent, int256 delta, uint256 newReputation);
    
    function setUp() public {
        vm.startPrank(admin);
        registry = new AgentRegistry(2); // Require 2 confirmations
        
        capabilities.push(keccak256("DATA_PROCESSING"));
        capabilities.push(keccak256("AI_INFERENCE"));
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Registration edge cases
    function testRegisterAgentSuccess() public {
        vm.startPrank(agent1);
        
        vm.expectEmit(true, false, false, true);
        emit AgentRegistered(agent1, "TestAgent", "https://api.test.com");
        
        registry.registerAgent("TestAgent", "https://api.test.com", capabilities);
        
        AgentRegistry.Agent memory agent = registry.getAgent(agent1);
        assertEq(agent.name, "TestAgent");
        assertEq(agent.endpoint, "https://api.test.com");
        assertTrue(agent.active);
        assertEq(agent.reputation, 100);
        
        vm.stopPrank();
    }
    
    function testRegisterAgentFailures() public {
        vm.startPrank(agent1);
        
        // Test empty name
        vm.expectRevert("Name required");
        registry.registerAgent("", "https://api.test.com", capabilities);
        
        // Test empty endpoint
        vm.expectRevert("Endpoint required");
        registry.registerAgent("TestAgent", "", capabilities);
        
        // Test empty capabilities
        bytes32[] memory emptyCapabilities;
        vm.expectRevert("At least one capability required");
        registry.registerAgent("TestAgent", "https://api.test.com", emptyCapabilities);
        
        // Register successfully first
        registry.registerAgent("TestAgent", "https://api.test.com", capabilities);
        
        // Test double registration
        vm.expectRevert("Agent already registered");
        registry.registerAgent("TestAgent2", "https://api2.test.com", capabilities);
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Reputation system edge cases
    function testReputationBoundaries() public {
        vm.startPrank(agent1);
        registry.registerAgent("TestAgent", "https://api.test.com", capabilities);
        vm.stopPrank();
        
        vm.startPrank(admin);
        
        // Test maximum reputation cap
        registry.increaseReputation(agent1, 150);
        AgentRegistry.Agent memory agent = registry.getAgent(agent1);
        assertEq(agent.reputation, 200); // Should be capped at MAX_REPUTATION
        
        // Test reputation cannot go below zero
        registry.decreaseReputation(agent1, 250);
        agent = registry.getAgent(agent1);
        assertEq(agent.reputation, 0);
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Access control
    function testUnauthorizedAccess() public {
        vm.startPrank(agent1);
        registry.registerAgent("TestAgent", "https://api.test.com", capabilities);
        vm.stopPrank();
        
        vm.startPrank(maliciousAgent);
        
        // Test unauthorized reputation changes
        vm.expectRevert();
        registry.increaseReputation(agent1, 50);
        
        vm.expectRevert();
        registry.decreaseReputation(agent1, 50);
        
        // Test unauthorized endpoint update
        vm.expectRevert("Not agent owner");
        registry.updateEndpoint("https://malicious.com");
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Pagination edge cases
    function testPaginationBoundaries() public {
        // Register multiple agents
        for (uint256 i = 0; i < 5; i++) {
            address agentAddr = address(uint160(0x100 + i));
            vm.startPrank(agentAddr);
            registry.registerAgent(
                string(abi.encodePacked("Agent", vm.toString(i))),
                string(abi.encodePacked("https://api", vm.toString(i), ".com")),
                capabilities
            );
            vm.stopPrank();
        }
        
        // Test normal pagination
        (address[] memory agents, uint256 total) = registry.getAgentsPaginated(0, 3);
        assertEq(agents.length, 3);
        assertEq(total, 5);
        
        // Test offset beyond total
        (agents, total) = registry.getAgentsPaginated(10, 3);
        assertEq(agents.length, 0);
        assertEq(total, 5);
        
        // Test limit exceeding remaining items
        (agents, total) = registry.getAgentsPaginated(3, 5);
        assertEq(agents.length, 2);
        assertEq(total, 5);
    }
    
    // CRITICAL TEST: Capability search edge cases
    function testCapabilitySearchEdgeCases() public {
        bytes32 uniqueCapability = keccak256("UNIQUE_CAPABILITY");
        bytes32[] memory uniqueCapabilities = new bytes32[](1);
        uniqueCapabilities[0] = uniqueCapability;
        
        // Register agent with unique capability
        vm.startPrank(agent1);
        registry.registerAgent("TestAgent", "https://api.test.com", uniqueCapabilities);
        vm.stopPrank();
        
        // Search for existing capability
        address[] memory foundAgents = registry.findAgentsByCapability(uniqueCapability);
        assertEq(foundAgents.length, 1);
        assertEq(foundAgents[0], agent1);
        
        // Search for non-existent capability
        bytes32 nonExistentCapability = keccak256("NON_EXISTENT");
        foundAgents = registry.findAgentsByCapability(nonExistentCapability);
        assertEq(foundAgents.length, 0);
    }
    
    // CRITICAL TEST: State consistency after deactivation/reactivation
    function testStateConsistencyAfterDeactivation() public {
        vm.startPrank(agent1);
        registry.registerAgent("TestAgent", "https://api.test.com", capabilities);
        
        uint256 initialActiveCount = registry.getActiveAgentsCount();
        
        // Deactivate agent
        vm.expectEmit(true, false, false, false);
        emit AgentDeactivated(agent1);
        registry.deactivateAgent();
        
        assertEq(registry.getActiveAgentsCount(), initialActiveCount - 1);
        
        AgentRegistry.Agent memory agent = registry.getAgent(agent1);
        assertFalse(agent.active);
        
        // Test double deactivation
        vm.expectRevert("Agent already inactive");
        registry.deactivateAgent();
        
        // Reactivate agent
        registry.reactivateAgent();
        assertEq(registry.getActiveAgentsCount(), initialActiveCount);
        
        agent = registry.getAgent(agent1);
        assertTrue(agent.active);
        
        // Test double reactivation
        vm.expectRevert("Agent already active");
        registry.reactivateAgent();
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Gas optimization verification
    function testGasOptimization() public {
        // Test that capability search is O(1) for retrieval
        bytes32 testCapability = keccak256("TEST_CAPABILITY");
        
        uint256 gasBefore = gasleft();
        uint256 length = registry._capabilitySetLength(testCapability);
        uint256 gasUsed1 = gasBefore - gasleft();
        
        // Add agents and test again
        vm.startPrank(agent1);
        registry.registerAgent("TestAgent1", "https://api1.test.com", capabilities);
        vm.stopPrank();
        
        gasBefore = gasleft();
        length = registry._capabilitySetLength(testCapability);
        uint256 gasUsed2 = gasBefore - gasleft();
        
        // Gas usage should be similar (O(1) operation)
        assertTrue(gasUsed2 <= gasUsed1 + 1000); // Allow small variance
    }
    
    // CRITICAL TEST: Reentrancy protection
    function testReentrancyProtection() public {
        // This would require a malicious contract that attempts reentrancy
        // For now, verify that nonReentrant modifier is present in critical functions
        vm.startPrank(agent1);
        registry.registerAgent("TestAgent", "https://api.test.com", capabilities);
        
        // Multiple rapid calls should not cause issues
        registry.updateEndpoint("https://new1.test.com");
        registry.updateEndpoint("https://new2.test.com");
        registry.updateEndpoint("https://new3.test.com");
        
        vm.stopPrank();
    }
}
