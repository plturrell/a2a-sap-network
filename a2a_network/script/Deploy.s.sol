// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "forge-std/Script.sol";
import "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
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
        
        // Deploy UUPS proxy for AgentServiceMarketplace using ERC1967Proxy
        ERC1967Proxy marketplaceProxy = new ERC1967Proxy(
            address(marketplaceImpl),
            marketplaceInit
        );
        AgentServiceMarketplace marketplace = AgentServiceMarketplace(address(marketplaceProxy));
        console.log("AgentServiceMarketplace proxy deployed to:", address(marketplace));
        console.log("AgentServiceMarketplace implementation at:", address(marketplaceImpl));
        
        ERC1967Proxy matcherProxy = new ERC1967Proxy(
            address(matcherImpl),
            matcherInit
        );
        CapabilityMatcher matcher = CapabilityMatcher(address(matcherProxy));
        console.log("CapabilityMatcher proxy deployed to:", address(matcher));
        console.log("CapabilityMatcher implementation at:", address(matcherImpl));
        
        ERC1967Proxy reputationProxy = new ERC1967Proxy(
            address(reputationImpl),
            reputationInit
        );
        PerformanceReputationSystem reputation = PerformanceReputationSystem(address(reputationProxy));
        console.log("PerformanceReputationSystem proxy deployed to:", address(reputation));
        console.log("PerformanceReputationSystem implementation at:", address(reputationImpl));
        
        // Grant necessary roles
        registry.grantRole(registry.DEFAULT_ADMIN_ROLE(), address(reputation));
        
        // Note: After deployment, additional pausers should be added using addPauser()

        vm.stopBroadcast();
    }
}
