// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "forge-std/Test.sol";
import "../src/AgentRegistry.sol";
import "../src/MessageRouter.sol";

contract MultiSigPausableTest is Test {
    AgentRegistry public registry;
    MessageRouter public router;
    
    address public admin = address(0x1);
    address public pauser1 = address(0x2);
    address public pauser2 = address(0x3);
    address public pauser3 = address(0x4);
    address public nonPauser = address(0x5);
    
    function setUp() public {
        vm.startPrank(admin);
        
        // Deploy with 2 required confirmations
        registry = new AgentRegistry(2);
        router = new MessageRouter(address(registry), 2);
        
        // Add additional pausers
        registry.addPauser(pauser1);
        registry.addPauser(pauser2);
        registry.addPauser(pauser3);
        
        router.addPauser(pauser1);
        router.addPauser(pauser2);
        router.addPauser(pauser3);
        
        vm.stopPrank();
    }
    
    function testMultiSigPause() public {
        // First pauser proposes pause
        vm.prank(pauser1);
        uint256 proposalId = registry.proposePause();
        
        // Contract should not be paused yet
        assertFalse(registry.paused());
        
        // Second pauser confirms - should execute
        vm.prank(pauser2);
        registry.confirmPauseProposal(proposalId);
        
        // Contract should now be paused
        assertTrue(registry.paused());
    }
    
    function testMultiSigUnpause() public {
        // First pause the contract
        vm.prank(pauser1);
        uint256 pauseProposalId = registry.proposePause();
        vm.prank(pauser2);
        registry.confirmPauseProposal(pauseProposalId);
        assertTrue(registry.paused());
        
        // Now propose unpause
        vm.prank(pauser1);
        uint256 unpauseProposalId = registry.proposeUnpause();
        
        // Still paused
        assertTrue(registry.paused());
        
        // Second confirmation should unpause
        vm.prank(pauser3);
        registry.confirmPauseProposal(unpauseProposalId);
        
        assertFalse(registry.paused());
    }
    
    function testCannotDoubleConfirm() public {
        vm.prank(pauser1);
        uint256 proposalId = registry.proposePause();
        
        // Try to confirm again as same pauser
        vm.prank(pauser1);
        vm.expectRevert("MultiSigPausable: already confirmed");
        registry.confirmPauseProposal(proposalId);
    }
    
    function testNonPauserCannotPropose() public {
        vm.prank(nonPauser);
        vm.expectRevert("MultiSigPausable: caller is not a pauser");
        registry.proposePause();
    }
    
    function testUpdateRequiredConfirmations() public {
        uint256 initialRequiredConfirmations = registry.requiredConfirmations();
        
        // Only admin can update
        vm.prank(pauser1);
        vm.expectRevert(
            abi.encodeWithSelector(
                bytes4(keccak256("AccessControlUnauthorizedAccount(address,bytes32)")),
                pauser1,
                bytes32(0) // DEFAULT_ADMIN_ROLE
            )
        );
        registry.updateRequiredConfirmations(3);
        
        // Verify unchanged
        assertEq(registry.requiredConfirmations(), initialRequiredConfirmations);
        
        // Admin updates to 3
        vm.prank(admin);
        registry.updateRequiredConfirmations(3);
        
        // Now need 3 confirmations
        vm.prank(pauser1);
        uint256 proposalId = registry.proposePause();
        assertFalse(registry.paused());
        
        vm.prank(pauser2);
        registry.confirmPauseProposal(proposalId);
        assertFalse(registry.paused());
        
        // Third confirmation executes
        vm.prank(pauser3);
        registry.confirmPauseProposal(proposalId);
        assertTrue(registry.paused());
    }
    
    function testSingleConfirmationMode() public {
        // Deploy with single confirmation
        AgentRegistry singleSigRegistry = new AgentRegistry(1);
        
        // Deployer (this test contract) can pause immediately
        singleSigRegistry.proposePause();
        assertTrue(singleSigRegistry.paused());
    }
}