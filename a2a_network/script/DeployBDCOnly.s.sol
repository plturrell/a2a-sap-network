// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Script} from "forge-std/Script.sol";
import {console} from "forge-std/console.sol";
import "../src/BusinessDataCloudA2A.sol";

/**
 * @title DeployBDCOnly
 * @dev Simple deployment script for BusinessDataCloudA2A contract only
 */
contract DeployBDCOnly is Script {
    
    function run() external {
        uint256 deployerPrivateKey = 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80;
        
        console.log("Deploying BusinessDataCloudA2A contract...");
        
        vm.startBroadcast(deployerPrivateKey);
        
        // Deploy BusinessDataCloudA2A contract
        BusinessDataCloudA2A bdcContract = new BusinessDataCloudA2A();
        console.log("BusinessDataCloudA2A deployed at:", address(bdcContract));
        console.log("Protocol Version:", bdcContract.PROTOCOL_VERSION());
        console.log("Contract Version:", bdcContract.CONTRACT_VERSION());
        
        vm.stopBroadcast();
        
        console.log("Deployment complete!");
    }
}