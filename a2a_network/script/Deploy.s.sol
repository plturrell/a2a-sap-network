// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "forge-std/Script.sol";
import "../src/AgentRegistry.sol";
import "../src/MessageRouter.sol";
import "../contracts/AgentServiceMarketplace.sol";
import "../contracts/CapabilityMatcher.sol";
import "../contracts/PerformanceReputationSystem.sol";

contract DeployScript is Script {
    function run() external {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");

        vm.startBroadcast(deployerPrivateKey);

        // Deploy with 2 required confirmations for multi-sig pause functionality
        uint256 requiredConfirmations = 2;
        
        AgentRegistry registry = new AgentRegistry(requiredConfirmations);
        console.log("AgentRegistry deployed to:", address(registry));
        console.log("Required confirmations:", requiredConfirmations);

        MessageRouter router = new MessageRouter(address(registry), requiredConfirmations);
        console.log("MessageRouter deployed to:", address(router));
        
        // Deploy new contracts with UUPS proxy pattern
        // Deploy implementations
        AgentServiceMarketplace marketplaceImpl = new AgentServiceMarketplace();
        CapabilityMatcher matcherImpl = new CapabilityMatcher();
        PerformanceReputationSystem reputationImpl = new PerformanceReputationSystem();
        
        // Deploy proxies and initialize
        bytes memory marketplaceInit = abi.encodeWithSelector(
            AgentServiceMarketplace.initialize.selector,
            address(registry)
        );
        bytes memory matcherInit = abi.encodeWithSelector(
            CapabilityMatcher.initialize.selector,
            address(registry)
        );
        bytes memory reputationInit = abi.encodeWithSelector(
            PerformanceReputationSystem.initialize.selector,
            address(registry)
        );
        
        // Note: In production, use proper UUPS proxy deployment
        // For now, deploying implementations directly
        AgentServiceMarketplace marketplace = marketplaceImpl;
        marketplace.initialize(address(registry));
        console.log("AgentServiceMarketplace deployed to:", address(marketplace));
        
        CapabilityMatcher matcher = matcherImpl;
        matcher.initialize(address(registry));
        console.log("CapabilityMatcher deployed to:", address(matcher));
        
        PerformanceReputationSystem reputation = reputationImpl;
        reputation.initialize(address(registry));
        console.log("PerformanceReputationSystem deployed to:", address(reputation));
        
        // Grant necessary roles
        registry.grantRole(registry.DEFAULT_ADMIN_ROLE(), address(reputation));
        
        // Note: After deployment, additional pausers should be added using addPauser()

        vm.stopBroadcast();
    }
}
