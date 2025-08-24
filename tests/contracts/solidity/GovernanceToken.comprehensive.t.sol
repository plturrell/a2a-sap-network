// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "../../../a2aNetwork/contracts/GovernanceToken.sol";

contract GovernanceTokenComprehensiveTest is Test {
    GovernanceToken public token;
    
    address public admin = address(0x1);
    address public user1 = address(0x2);
    address public user2 = address(0x3);
    address public malicious = address(0x4);
    
    event TokensStaked(address indexed user, uint256 amount);
    event TokensUnstaked(address indexed user, uint256 amount);
    event RewardsClaimed(address indexed user, uint256 amount);
    event VestingScheduleCreated(address indexed beneficiary, uint256 amount);
    event TokensVested(address indexed beneficiary, uint256 amount);
    
    function setUp() public {
        vm.startPrank(admin);
        
        token = new GovernanceToken();
        token.initialize("A2A Governance Token", "A2A", admin);
        
        // Transfer some tokens to test users
        token.transfer(user1, 1000000 * 10**18);
        token.transfer(user2, 500000 * 10**18);
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Token initialization and supply limits
    function testTokenInitialization() public {
        assertEq(token.name(), "A2A Governance Token");
        assertEq(token.symbol(), "A2A");
        assertEq(token.totalSupply(), token.INITIAL_SUPPLY());
        assertEq(token.balanceOf(admin), token.INITIAL_SUPPLY() - 1500000 * 10**18);
    }
    
    function testSupplyLimits() public {
        vm.startPrank(admin);
        
        // Test minting up to max supply
        uint256 remainingSupply = token.MAX_SUPPLY() - token.totalSupply();
        token.mint(admin, remainingSupply);
        assertEq(token.totalSupply(), token.MAX_SUPPLY());
        
        // Test exceeding max supply
        vm.expectRevert("Exceeds maximum supply");
        token.mint(admin, 1);
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Staking mechanism edge cases
    function testStakingSuccess() public {
        vm.startPrank(user1);
        
        uint256 stakeAmount = 100000 * 10**18;
        uint256 initialBalance = token.balanceOf(user1);
        
        vm.expectEmit(true, false, false, true);
        emit TokensStaked(user1, stakeAmount);
        
        token.stake(stakeAmount);
        
        assertEq(token.stakingBalances(user1), stakeAmount);
        assertEq(token.balanceOf(user1), initialBalance - stakeAmount);
        assertEq(token.balanceOf(address(token)), stakeAmount);
        
        vm.stopPrank();
    }
    
    function testStakingFailures() public {
        vm.startPrank(user1);
        
        // Test staking more than balance
        uint256 balance = token.balanceOf(user1);
        vm.expectRevert("Insufficient balance");
        token.stake(balance + 1);
        
        // Test staking zero amount
        vm.expectRevert("Amount must be greater than 0");
        token.stake(0);
        
        vm.stopPrank();
    }
    
    function testUnstakingWithMinimumPeriod() public {
        vm.startPrank(user1);
        
        uint256 stakeAmount = 100000 * 10**18;
        token.stake(stakeAmount);
        
        // Try to unstake immediately
        vm.expectRevert("Minimum staking period not met");
        token.unstake(stakeAmount);
        
        // Advance time past minimum period
        vm.warp(block.timestamp + 7 days + 1 seconds);
        
        vm.expectEmit(true, false, false, true);
        emit TokensUnstaked(user1, stakeAmount);
        
        token.unstake(stakeAmount);
        
        assertEq(token.stakingBalances(user1), 0);
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Reward calculation and claiming
    function testRewardCalculation() public {
        vm.startPrank(user1);
        
        uint256 stakeAmount = 100000 * 10**18;
        token.stake(stakeAmount);
        
        // Advance time by 1 year
        vm.warp(block.timestamp + 365 days);
        
        uint256 balanceBefore = token.balanceOf(user1);
        
        vm.expectEmit(true, false, false, true);
        emit RewardsClaimed(user1, 0); // Amount will be calculated
        
        token.claimRewards();
        
        uint256 balanceAfter = token.balanceOf(user1);
        uint256 expectedReward = (stakeAmount * 5) / 100; // 5% annual rate
        
        // Allow for small rounding differences
        assertTrue(balanceAfter >= balanceBefore + expectedReward - 1000);
        assertTrue(balanceAfter <= balanceBefore + expectedReward + 1000);
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Vesting schedule edge cases
    function testVestingScheduleCreation() public {
        vm.startPrank(admin);
        
        uint256 vestAmount = 1000000 * 10**18;
        uint256 cliffDuration = 30 days;
        uint256 vestingDuration = 365 days;
        
        vm.expectEmit(true, false, false, true);
        emit VestingScheduleCreated(user1, vestAmount);
        
        token.createVestingSchedule(user1, vestAmount, cliffDuration, vestingDuration, true);
        
        (uint256 totalAmount, uint256 releasedAmount, uint256 startTime, uint256 cliff, uint256 duration, bool revocable, bool revoked) = token.vestingSchedules(user1);
        
        assertEq(totalAmount, vestAmount);
        assertEq(releasedAmount, 0);
        assertEq(cliff, cliffDuration);
        assertEq(duration, vestingDuration);
        assertTrue(revocable);
        assertFalse(revoked);
        
        vm.stopPrank();
    }
    
    function testVestingScheduleFailures() public {
        vm.startPrank(admin);
        
        // Test invalid beneficiary
        vm.expectRevert("Invalid beneficiary");
        token.createVestingSchedule(address(0), 1000, 30 days, 365 days, true);
        
        // Test insufficient balance
        uint256 adminBalance = token.balanceOf(admin);
        vm.expectRevert("Insufficient balance");
        token.createVestingSchedule(user1, adminBalance + 1, 30 days, 365 days, true);
        
        // Create valid schedule
        token.createVestingSchedule(user1, 1000000 * 10**18, 30 days, 365 days, true);
        
        // Test duplicate schedule
        vm.expectRevert("Vesting schedule already exists");
        token.createVestingSchedule(user1, 500000 * 10**18, 30 days, 365 days, true);
        
        vm.stopPrank();
    }
    
    function testVestingRelease() public {
        vm.startPrank(admin);
        uint256 vestAmount = 1000000 * 10**18;
        token.createVestingSchedule(user1, vestAmount, 30 days, 365 days, true);
        vm.stopPrank();
        
        vm.startPrank(user1);
        
        // Try to release before cliff
        vm.expectRevert("No tokens to release");
        token.releaseVestedTokens();
        
        // Advance past cliff but not full vesting
        vm.warp(block.timestamp + 180 days); // ~50% vested
        
        uint256 balanceBefore = token.balanceOf(user1);
        uint256 releasableBefore = token.getReleasableAmount(user1);
        
        assertTrue(releasableBefore > 0);
        
        vm.expectEmit(true, false, false, true);
        emit TokensVested(user1, releasableBefore);
        
        token.releaseVestedTokens();
        
        uint256 balanceAfter = token.balanceOf(user1);
        assertEq(balanceAfter, balanceBefore + releasableBefore);
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Access control and authorization
    function testUnauthorizedAccess() public {
        vm.startPrank(malicious);
        
        // Test unauthorized minting
        vm.expectRevert();
        token.mint(malicious, 1000);
        
        // Test unauthorized burning
        vm.expectRevert();
        token.burn(1000);
        
        // Test unauthorized vesting schedule creation
        vm.expectRevert();
        token.createVestingSchedule(malicious, 1000, 30 days, 365 days, true);
        
        // Test unauthorized pause
        vm.expectRevert();
        token.pause();
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Transfer restrictions with staked tokens
    function testTransferRestrictions() public {
        vm.startPrank(user1);
        
        uint256 balance = token.balanceOf(user1);
        uint256 stakeAmount = balance / 2;
        
        // Stake half the tokens
        token.stake(stakeAmount);
        
        // Should be able to transfer non-staked tokens
        uint256 transferAmount = balance / 4;
        token.transfer(user2, transferAmount);
        
        // Should not be able to transfer more than non-staked balance
        vm.expectRevert("Insufficient non-staked balance");
        token.transfer(user2, balance - transferAmount);
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Pause functionality
    function testPauseFunctionality() public {
        vm.startPrank(admin);
        
        token.pause();
        
        vm.stopPrank();
        
        vm.startPrank(user1);
        
        // Test that operations are paused
        vm.expectRevert("Contract is paused");
        token.stake(1000);
        
        vm.expectRevert("Contract is paused");
        token.transfer(user2, 1000);
        
        vm.stopPrank();
        
        vm.startPrank(admin);
        
        // Test that minting is also paused
        vm.expectRevert("Contract is paused");
        token.mint(admin, 1000);
        
        // Unpause
        token.unpause();
        
        vm.stopPrank();
        
        vm.startPrank(user1);
        
        // Operations should work again
        token.stake(1000);
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Voting power calculation
    function testVotingPowerCalculation() public {
        vm.startPrank(user1);
        
        uint256 initialBalance = token.balanceOf(user1);
        uint256 stakeAmount = 100000 * 10**18;
        
        // Delegate to self to get voting power
        token.delegate(user1);
        
        uint256 votingPowerBefore = token.getVotingPower(user1);
        assertEq(votingPowerBefore, initialBalance);
        
        // Stake tokens
        token.stake(stakeAmount);
        
        uint256 votingPowerAfter = token.getVotingPower(user1);
        // Voting power should include both held and staked tokens
        assertEq(votingPowerAfter, initialBalance);
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Vesting schedule revocation
    function testVestingRevocation() public {
        vm.startPrank(admin);
        
        uint256 vestAmount = 1000000 * 10**18;
        token.createVestingSchedule(user1, vestAmount, 30 days, 365 days, true);
        
        // Advance time partially through vesting
        vm.warp(block.timestamp + 180 days);
        
        uint256 adminBalanceBefore = token.balanceOf(admin);
        
        // Revoke the schedule
        token.revokeVestingSchedule(user1);
        
        (,,,,,, bool revoked) = token.vestingSchedules(user1);
        assertTrue(revoked);
        
        // Admin should receive unvested tokens back
        uint256 adminBalanceAfter = token.balanceOf(admin);
        assertTrue(adminBalanceAfter > adminBalanceBefore);
        
        vm.stopPrank();
    }
    
    // CRITICAL TEST: Gas optimization verification
    function testGasOptimization() public {
        vm.startPrank(user1);
        
        // Test staking gas usage
        uint256 gasBefore = gasleft();
        token.stake(1000 * 10**18);
        uint256 gasUsed = gasBefore - gasleft();
        
        // Should use reasonable amount of gas
        assertTrue(gasUsed < 150000);
        
        vm.stopPrank();
    }
}
