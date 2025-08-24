// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "../../../a2aNetwork/contracts/A2ATimelock.sol";

contract A2ATimelockComprehensiveTest is Test {
    A2ATimelock public timelock;
    
    address public admin = address(0x1);
    address public proposer = address(0x2);
    address public executor = address(0x3);
    address public guardian = address(0x4);
    address public reviewer1 = address(0x5);
    address public reviewer2 = address(0x6);
    address public reviewer3 = address(0x7);
    address public malicious = address(0x8);
    
    address[] public proposers;
    address[] public executors;
    address[] public reviewers;
    
    event OperationScheduledWithMetadata(
        bytes32 indexed id,
        uint256 category,
        uint256 riskLevel,
        string description
    );
    event OperationReviewed(bytes32 indexed id, address reviewer);
    event OperationVetoed(bytes32 indexed id, address guardian, string reason);
    event EmergencyBypassActivated(bytes32 indexed id, address activator);
    
    function setUp() public {
        proposers.push(proposer);
        executors.push(executor);
        reviewers.push(reviewer1);
        reviewers.push(reviewer2);
        reviewers.push(reviewer3);
        
        vm.startPrank(admin);
        
        timelock = new A2ATimelock();
        timelock.initialize(1 days, proposers, executors, admin);
        
        // Grant additional roles
        timelock.grantRole(timelock.GUARDIAN_ROLE(), guardian);
        timelock.grantRole(timelock.EMERGENCY_ROLE(), admin);
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Operation scheduling with metadata
    function testOperationSchedulingWithMetadata() public {
        vm.startPrank(proposer);
        
        address target = address(0x123);
        uint256 value = 0;
        bytes memory data = abi.encodeWithSignature("updateParameter(uint256)", 100);
        bytes32 predecessor = bytes32(0);
        bytes32 salt = keccak256("test");
        uint256 delay = 2 days;
        uint256 category = 1; // Parameter changes
        uint256 riskLevel = uint256(A2ATimelock.RiskLevel.MEDIUM);
        string memory description = "Update system parameter";
        
        vm.expectEmit(true, false, false, true);
        emit OperationScheduledWithMetadata(bytes32(0), category, riskLevel, description);
        
        bytes32 id = timelock.scheduleWithMetadata(
            target,
            value,
            data,
            predecessor,
            salt,
            delay,
            category,
            riskLevel,
            description,
            reviewers
        );
        
        (
            uint256 metaCategory,
            uint256 metaRiskLevel,
            string memory metaDescription,
            address[] memory metaReviewers,
            uint256 reviewCount,
            bool emergencyBypass,
            uint256 scheduledAt
        ) = timelock.getOperationMetadata(id);
        
        assertEq(metaCategory, category);
        assertEq(metaRiskLevel, riskLevel);
        assertEq(metaDescription, description);
        assertEq(metaReviewers.length, 3);
        assertEq(reviewCount, 0);
        assertFalse(emergencyBypass);
        assertTrue(scheduledAt > 0);
        
        vm.stopPrank();
    }
    
    function testSchedulingFailures() public {
        vm.startPrank(proposer);
        
        // Test invalid risk level
        vm.expectRevert("Invalid risk level");
        timelock.scheduleWithMetadata(
            address(0x123),
            0,
            "data",
            bytes32(0),
            keccak256("test"),
            2 days,
            1,
            5, // Invalid risk level
            "test",
            reviewers
        );
        
        // Test delay too short for risk level
        vm.expectRevert("Delay too short for category/risk");
        timelock.scheduleWithMetadata(
            address(0x123),
            0,
            "data",
            bytes32(0),
            keccak256("test"),
            1 hours, // Too short for CRITICAL risk
            0,
            uint256(A2ATimelock.RiskLevel.CRITICAL),
            "test",
            reviewers
        );
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Batch operation scheduling
    function testBatchOperationScheduling() public {
        vm.startPrank(proposer);
        
        address[] memory targets = new address[](2);
        targets[0] = address(0x123);
        targets[1] = address(0x456);
        
        uint256[] memory values = new uint256[](2);
        values[0] = 0;
        values[1] = 1 ether;
        
        bytes[] memory payloads = new bytes[](2);
        payloads[0] = abi.encodeWithSignature("function1()");
        payloads[1] = abi.encodeWithSignature("function2()");
        
        bytes32 id = timelock.batchScheduleWithMetadata(
            targets,
            values,
            payloads,
            bytes32(0),
            keccak256("batch"),
            3 days,
            2, // Treasury operations
            uint256(A2ATimelock.RiskLevel.HIGH),
            "Batch treasury operation"
        );
        
        assertTrue(timelock.isOperationPending(id));
        
        vm.stopPrank();
    }
    
    function testBatchSchedulingFailures() public {
        vm.startPrank(proposer);
        
        address[] memory targets = new address[](2);
        uint256[] memory values = new uint256[](1); // Mismatched length
        bytes[] memory payloads = new bytes[](2);
        
        vm.expectRevert("Length mismatch");
        timelock.batchScheduleWithMetadata(
            targets,
            values,
            payloads,
            bytes32(0),
            keccak256("batch"),
            3 days,
            2,
            uint256(A2ATimelock.RiskLevel.HIGH),
            "test"
        );
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Review process
    function testReviewProcess() public {
        // Schedule a medium risk operation
        vm.startPrank(proposer);
        bytes32 id = timelock.scheduleWithMetadata(
            address(0x123),
            0,
            "data",
            bytes32(0),
            keccak256("review_test"),
            3 days,
            1,
            uint256(A2ATimelock.RiskLevel.MEDIUM),
            "Medium risk operation",
            reviewers
        );
        vm.stopPrank();
        
        // Reviewer1 approves
        vm.startPrank(reviewer1);
        vm.expectEmit(true, true, false, false);
        emit OperationReviewed(id, reviewer1);
        timelock.reviewOperation(id, true, "Approved");
        vm.stopPrank();
        
        // Check review count
        (, , , , uint256 reviewCount, , ) = timelock.getOperationMetadata(id);
        assertEq(reviewCount, 1);
        
        // Reviewer2 approves (should meet majority requirement)
        vm.startPrank(reviewer2);
        timelock.reviewOperation(id, true, "Approved");
        vm.stopPrank();
        
        (, , , , reviewCount, , ) = timelock.getOperationMetadata(id);
        assertEq(reviewCount, 2);
    }
    
    function testReviewFailures() public {
        vm.startPrank(proposer);
        bytes32 id = timelock.scheduleWithMetadata(
            address(0x123),
            0,
            "data",
            bytes32(0),
            keccak256("review_fail_test"),
            3 days,
            1,
            uint256(A2ATimelock.RiskLevel.MEDIUM),
            "Test operation",
            reviewers
        );
        vm.stopPrank();
        
        // Unauthorized reviewer
        vm.startPrank(malicious);
        vm.expectRevert("Not a required reviewer");
        timelock.reviewOperation(id, true, "Unauthorized");
        vm.stopPrank();
        
        // Reviewer disapproval should veto
        vm.startPrank(reviewer1);
        timelock.reviewOperation(id, false, "Rejected");
        vm.stopPrank();
        
        assertTrue(timelock.vetoedOperations(id));
    }
    
    // CRITICAL TEST: Guardian veto functionality
    function testGuardianVeto() public {
        vm.startPrank(proposer);
        bytes32 id = timelock.scheduleWithMetadata(
            address(0x123),
            0,
            "data",
            bytes32(0),
            keccak256("veto_test"),
            2 days,
            1,
            uint256(A2ATimelock.RiskLevel.LOW),
            "Test operation",
            new address[](0)
        );
        vm.stopPrank();
        
        vm.startPrank(guardian);
        vm.expectEmit(true, true, false, true);
        emit OperationVetoed(id, guardian, "Security concern");
        
        timelock.vetoOperation(id, "Security concern");
        vm.stopPrank();
        
        assertTrue(timelock.vetoedOperations(id));
    }
    
    function testVetoFailures() public {
        vm.startPrank(proposer);
        bytes32 id = timelock.scheduleWithMetadata(
            address(0x123),
            0,
            "data",
            bytes32(0),
            keccak256("veto_fail_test"),
            2 days,
            1,
            uint256(A2ATimelock.RiskLevel.LOW),
            "Test operation",
            new address[](0)
        );
        vm.stopPrank();
        
        // Advance time past veto window
        vm.warp(block.timestamp + 25 hours);
        
        vm.startPrank(guardian);
        vm.expectRevert("Veto window expired");
        timelock.vetoOperation(id, "Too late");
        vm.stopPrank();
        
        // Unauthorized veto
        vm.warp(block.timestamp - 25 hours); // Reset time
        vm.startPrank(malicious);
        vm.expectRevert("Not a guardian");
        timelock.vetoOperation(id, "Unauthorized");
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Emergency bypass
    function testEmergencyBypass() public {
        // Activate emergency mode
        vm.startPrank(admin);
        timelock.activateEmergencyMode();
        vm.stopPrank();
        
        vm.startPrank(proposer);
        bytes32 id = timelock.scheduleWithMetadata(
            address(0x123),
            0,
            "data",
            bytes32(0),
            keccak256("emergency_test"),
            7 days,
            3, // Emergency category
            uint256(A2ATimelock.RiskLevel.HIGH),
            "Emergency operation",
            reviewers
        );
        vm.stopPrank();
        
        vm.startPrank(admin);
        vm.expectEmit(true, true, false, false);
        emit EmergencyBypassActivated(id, admin);
        
        timelock.emergencyBypass(id, "Critical security fix");
        vm.stopPrank();
        
        (, , , , , bool emergencyBypass, ) = timelock.getOperationMetadata(id);
        assertTrue(emergencyBypass);
    }
    
    function testEmergencyBypassFailures() public {
        vm.startPrank(proposer);
        bytes32 id = timelock.scheduleWithMetadata(
            address(0x123),
            0,
            "data",
            bytes32(0),
            keccak256("emergency_fail_test"),
            14 days,
            0,
            uint256(A2ATimelock.RiskLevel.CRITICAL),
            "Critical operation",
            reviewers
        );
        vm.stopPrank();
        
        vm.startPrank(admin);
        
        // Test without emergency mode
        vm.expectRevert("Emergency mode not active");
        timelock.emergencyBypass(id, "Test");
        
        // Activate emergency mode
        timelock.activateEmergencyMode();
        
        // Test critical operation bypass (should fail)
        vm.expectRevert("Cannot bypass critical operations");
        timelock.emergencyBypass(id, "Cannot bypass critical");
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Execution with review requirements
    function testExecutionWithReviews() public {
        // Schedule high-risk operation requiring all reviews
        vm.startPrank(proposer);
        address target = address(this);
        bytes memory data = abi.encodeWithSignature("mockFunction()");
        bytes32 id = timelock.scheduleWithMetadata(
            target,
            0,
            data,
            bytes32(0),
            keccak256("execution_test"),
            7 days,
            0,
            uint256(A2ATimelock.RiskLevel.HIGH),
            "High risk operation",
            reviewers
        );
        vm.stopPrank();
        
        // All reviewers approve
        vm.startPrank(reviewer1);
        timelock.reviewOperation(id, true, "Approved");
        vm.stopPrank();
        
        vm.startPrank(reviewer2);
        timelock.reviewOperation(id, true, "Approved");
        vm.stopPrank();
        
        vm.startPrank(reviewer3);
        timelock.reviewOperation(id, true, "Approved");
        vm.stopPrank();
        
        // Advance time past delay
        vm.warp(block.timestamp + 8 days);
        
        // Should be able to execute now
        vm.startPrank(executor);
        timelock.execute(target, 0, data, bytes32(0), keccak256("execution_test"));
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Parameter updates
    function testParameterUpdates() public {
        vm.startPrank(admin);
        
        // Update category delay
        timelock.updateCategoryDelay(1, 5 days);
        assertEq(timelock.categoryDelays(1), 5 days);
        
        // Update risk delay
        timelock.updateRiskDelay(A2ATimelock.RiskLevel.MEDIUM, 4 days);
        assertEq(timelock.riskDelays(A2ATimelock.RiskLevel.MEDIUM), 4 days);
        
        // Update guardian veto window
        timelock.updateGuardianVetoWindow(48 hours);
        assertEq(timelock.guardianVetoWindow(), 48 hours);
        
        vm.stopPrank();
    }
    
    function testParameterUpdateFailures() public {
        vm.startPrank(admin);
        
        // Invalid delay bounds
        vm.expectRevert("Invalid delay");
        timelock.updateCategoryDelay(1, 12 hours); // Too short
        
        vm.expectRevert("Invalid delay");
        timelock.updateRiskDelay(A2ATimelock.RiskLevel.LOW, 31 days); // Too long
        
        // Invalid veto window
        vm.expectRevert("Invalid window");
        timelock.updateGuardianVetoWindow(30 minutes); // Too short
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Guardian management
    function testGuardianManagement() public {
        address newGuardian = address(0x9);
        
        vm.startPrank(admin);
        
        timelock.addGuardian(newGuardian);
        assertTrue(timelock.isGuardian(newGuardian));
        
        timelock.removeGuardian(newGuardian);
        assertFalse(timelock.isGuardian(newGuardian));
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Minimum delay calculations
    function testMinDelayCalculations() public {
        // Test category vs risk delay precedence
        uint256 minDelay = timelock.getMinDelayFor(2, uint256(A2ATimelock.RiskLevel.LOW)); // Treasury + Low
        assertEq(minDelay, 5 days); // Treasury delay is higher
        
        minDelay = timelock.getMinDelayFor(1, uint256(A2ATimelock.RiskLevel.CRITICAL)); // Parameter + Critical
        assertEq(minDelay, 14 days); // Critical delay is higher
    }
    
    // CRITICAL TEST: Division by zero protection
    function testDivisionByZeroProtection() public {
        // Test medium risk operation with no reviewers
        vm.startPrank(proposer);
        bytes32 id = timelock.scheduleWithMetadata(
            address(0x123),
            0,
            "data",
            bytes32(0),
            keccak256("zero_reviewers_test"),
            3 days,
            1,
            uint256(A2ATimelock.RiskLevel.MEDIUM),
            "Medium risk with no reviewers",
            new address[](0) // Empty reviewers array
        );
        vm.stopPrank();
        
        // Advance time past delay
        vm.warp(block.timestamp + 4 days);
        
        // Should be able to execute without division by zero error
        vm.startPrank(executor);
        timelock.execute(address(0x123), 0, "data", bytes32(0), keccak256("zero_reviewers_test"));
        vm.stopPrank();
    }
    
    // Mock function for testing execution
    function mockFunction() external pure returns (bool) {
        return true;
    }
    
    // CRITICAL TEST: Gas optimization verification
    function testGasOptimization() public {
        vm.startPrank(proposer);
        
        uint256 gasBefore = gasleft();
        timelock.scheduleWithMetadata(
            address(0x123),
            0,
            "test data",
            bytes32(0),
            keccak256("gas_test"),
            2 days,
            1,
            uint256(A2ATimelock.RiskLevel.LOW),
            "Gas test operation",
            reviewers
        );
        uint256 gasUsed = gasBefore - gasleft();
        
        // Should use reasonable amount of gas
        assertTrue(gasUsed < 400000);
        
        vm.stopPrank();
    }
}
