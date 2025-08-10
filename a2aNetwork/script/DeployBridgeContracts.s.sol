// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "forge-std/Script.sol";
import "../src/AgentRegistry.sol";
import "../src/MessageRouter.sol";
import "../src/crosschain/ProtocolBridge.sol";
import "../src/crosschain/IdentityBridge.sol";
import "../src/crosschain/UnifiedProtocolRouter.sol";
import "../src/crosschain/ExternalProtocolOracle.sol";

/**
 * @title DeployBridgeContracts
 * @dev Deployment script for A2A protocol bridge contracts
 * Deploys the complete bridge infrastructure for ANP/ACP integration
 */
contract DeployBridgeContracts is Script {
    // Configuration
    uint256 public constant REQUIRED_CONFIRMATIONS = 2;
    
    // Contract addresses (will be set during deployment)
    address public agentRegistry;
    address public messageRouter;
    address public externalProtocolOracle;
    address public protocolBridge;
    address public identityBridge;
    address public unifiedProtocolRouter;
    
    function setUp() public {}
    
    function run() public {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        address deployer = vm.addr(deployerPrivateKey);
        
        console.log("Deploying A2A Bridge Contracts...");
        console.log("Deployer:", deployer);
        console.log("Chain ID:", block.chainid);
        
        vm.startBroadcast(deployerPrivateKey);
        
        // Step 1: Deploy core contracts if they don't exist
        agentRegistry = _deployAgentRegistry();
        messageRouter = _deployMessageRouter();
        
        // Step 2: Deploy bridge infrastructure
        externalProtocolOracle = _deployExternalProtocolOracle();
        protocolBridge = _deployProtocolBridge();
        identityBridge = _deployIdentityBridge();
        unifiedProtocolRouter = _deployUnifiedProtocolRouter();
        
        // Step 3: Configure protocols
        _configureProtocols();
        
        // Step 4: Set up initial capabilities and weights
        _setupInitialConfiguration();
        
        vm.stopBroadcast();
        
        // Log deployment results
        _logDeploymentResults();
        
        // Verify deployment
        _verifyDeployment();
    }
    
    function _deployAgentRegistry() internal returns (address) {
        // Check if AgentRegistry already exists from previous deployment
        try vm.envAddress("AGENT_REGISTRY_ADDRESS") returns (address existingRegistry) {
            console.log("Using existing AgentRegistry at:", existingRegistry);
            return existingRegistry;
        } catch {
            console.log("Deploying new AgentRegistry...");
            AgentRegistry registry = new AgentRegistry(REQUIRED_CONFIRMATIONS);
            console.log("AgentRegistry deployed at:", address(registry));
            return address(registry);
        }
    }
    
    function _deployMessageRouter() internal returns (address) {
        // Check if MessageRouter already exists
        try vm.envAddress("MESSAGE_ROUTER_ADDRESS") returns (address existingRouter) {
            console.log("Using existing MessageRouter at:", existingRouter);
            return existingRouter;
        } catch {
            console.log("Deploying new MessageRouter...");
            MessageRouter router = new MessageRouter(agentRegistry, REQUIRED_CONFIRMATIONS);
            console.log("MessageRouter deployed at:", address(router));
            return address(router);
        }
    }
    
    function _deployExternalProtocolOracle() internal returns (address) {
        console.log("Deploying ExternalProtocolOracle...");
        ExternalProtocolOracle oracle = new ExternalProtocolOracle(REQUIRED_CONFIRMATIONS);
        console.log("ExternalProtocolOracle deployed at:", address(oracle));
        return address(oracle);
    }
    
    function _deployProtocolBridge() internal returns (address) {
        console.log("Deploying ProtocolBridge...");
        ProtocolBridge bridge = new ProtocolBridge(
            agentRegistry,
            messageRouter,
            externalProtocolOracle,
            REQUIRED_CONFIRMATIONS
        );
        console.log("ProtocolBridge deployed at:", address(bridge));
        return address(bridge);
    }
    
    function _deployIdentityBridge() internal returns (address) {
        console.log("Deploying IdentityBridge...");
        IdentityBridge identity = new IdentityBridge(
            agentRegistry,
            REQUIRED_CONFIRMATIONS
        );
        console.log("IdentityBridge deployed at:", address(identity));
        return address(identity);
    }
    
    function _deployUnifiedProtocolRouter() internal returns (address) {
        console.log("Deploying UnifiedProtocolRouter...");
        UnifiedProtocolRouter router = new UnifiedProtocolRouter(
            agentRegistry,
            messageRouter,
            protocolBridge,
            identityBridge,
            REQUIRED_CONFIRMATIONS
        );
        console.log("UnifiedProtocolRouter deployed at:", address(router));
        return address(router);
    }
    
    function _configureProtocols() internal {
        console.log("Configuring protocols...");
        
        // Configure ANP protocol
        string memory anpEndpoint = vm.envOr("ANP_ENDPOINT", string("https://anp-gateway.example.com"));
        bool anpEnabled = vm.envOr("ANP_ENABLED", false);
        
        ProtocolBridge(protocolBridge).configureProtocol(
            ProtocolBridge.ProtocolType.ANP,
            anpEndpoint,
            anpEnabled
        );
        console.log("ANP configured - Endpoint:", anpEndpoint, "Enabled:", anpEnabled);
        
        // Configure ACP protocol
        string memory acpEndpoint = vm.envOr("ACP_ENDPOINT", string("https://acp-gateway.example.com"));
        bool acpEnabled = vm.envOr("ACP_ENABLED", false);
        
        ProtocolBridge(protocolBridge).configureProtocol(
            ProtocolBridge.ProtocolType.ACP,
            acpEndpoint,
            acpEnabled
        );
        console.log("ACP configured - Endpoint:", acpEndpoint, "Enabled:", acpEnabled);
    }
    
    function _setupInitialConfiguration() internal {
        console.log("Setting up initial configuration...");
        
        UnifiedProtocolRouter router = UnifiedProtocolRouter(unifiedProtocolRouter);
        
        // Set capability weights for optimal routing
        bytes32[] memory capabilities = new bytes32[](8);
        uint256[] memory weights = new uint256[](8);
        
        capabilities[0] = keccak256("data_analysis");
        weights[0] = 85;
        
        capabilities[1] = keccak256("text_processing");
        weights[1] = 80;
        
        capabilities[2] = keccak256("image_processing");
        weights[2] = 70;
        
        capabilities[3] = keccak256("blockchain_query");
        weights[3] = 90;
        
        capabilities[4] = keccak256("web_search");
        weights[4] = 60;
        
        capabilities[5] = keccak256("document_processing");
        weights[5] = 75;
        
        capabilities[6] = keccak256("semantic_search");
        weights[6] = 65;
        
        capabilities[7] = keccak256("identity_verification");
        weights[7] = 95;
        
        for (uint256 i = 0; i < capabilities.length; i++) {
            router.updateCapabilityWeight(capabilities[i], weights[i]);
            console.log("Set capability weight:", string(abi.encodePacked(capabilities[i])), "=", weights[i]);
        }
    }
    
    function _logDeploymentResults() internal view {
        console.log("\n=== DEPLOYMENT RESULTS ===");
        console.log("AgentRegistry:", agentRegistry);
        console.log("MessageRouter:", messageRouter);
        console.log("ProtocolBridge:", protocolBridge);
        console.log("IdentityBridge:", identityBridge);
        console.log("UnifiedProtocolRouter:", unifiedProtocolRouter);
        console.log("========================\n");
        
        // Save addresses to file for easy access
        string memory addresses = string(abi.encodePacked(
            "AGENT_REGISTRY_ADDRESS=", vm.toString(agentRegistry), "\n",
            "MESSAGE_ROUTER_ADDRESS=", vm.toString(messageRouter), "\n",
            "PROTOCOL_BRIDGE_ADDRESS=", vm.toString(protocolBridge), "\n",
            "IDENTITY_BRIDGE_ADDRESS=", vm.toString(identityBridge), "\n",
            "UNIFIED_PROTOCOL_ROUTER_ADDRESS=", vm.toString(unifiedProtocolRouter), "\n"
        ));
        
        // vm.writeFile("./bridge-addresses.env", addresses);
        console.log("Contract addresses would be saved to bridge-addresses.env");
        console.log("Addresses:\n", addresses);
    }
    
    function _verifyDeployment() internal view {
        console.log("Verifying deployment...");
        
        // Verify AgentRegistry
        require(agentRegistry.code.length > 0, "AgentRegistry deployment failed");
        AgentRegistry registry = AgentRegistry(agentRegistry);
        require(registry.getActiveAgentsCount() >= 0, "AgentRegistry not functional");
        console.log("AgentRegistry verified");
        
        // Verify MessageRouter
        require(messageRouter.code.length > 0, "MessageRouter deployment failed");
        MessageRouter router = MessageRouter(messageRouter);
        require(address(router.registry()) == agentRegistry, "MessageRouter registry mismatch");
        console.log("MessageRouter verified");
        
        // Verify ProtocolBridge
        require(protocolBridge.code.length > 0, "ProtocolBridge deployment failed");
        ProtocolBridge bridge = ProtocolBridge(protocolBridge);
        require(address(bridge.registry()) == agentRegistry, "ProtocolBridge registry mismatch");
        require(address(bridge.router()) == messageRouter, "ProtocolBridge router mismatch");
        console.log("ProtocolBridge verified");
        
        // Verify IdentityBridge
        require(identityBridge.code.length > 0, "IdentityBridge deployment failed");
        IdentityBridge identity = IdentityBridge(identityBridge);
        require(address(identity.registry()) == agentRegistry, "IdentityBridge registry mismatch");
        console.log("IdentityBridge verified");
        
        // Verify UnifiedProtocolRouter
        require(unifiedProtocolRouter.code.length > 0, "UnifiedProtocolRouter deployment failed");
        UnifiedProtocolRouter unifiedRouter = UnifiedProtocolRouter(unifiedProtocolRouter);
        require(address(unifiedRouter.registry()) == agentRegistry, "UnifiedProtocolRouter registry mismatch");
        require(address(unifiedRouter.messageRouter()) == messageRouter, "UnifiedProtocolRouter messageRouter mismatch");
        require(address(unifiedRouter.protocolBridge()) == protocolBridge, "UnifiedProtocolRouter protocolBridge mismatch");
        require(address(unifiedRouter.identityBridge()) == identityBridge, "UnifiedProtocolRouter identityBridge mismatch");
        console.log("UnifiedProtocolRouter verified");
        
        console.log("\nAll contracts deployed and verified successfully!");
    }
    
    // Utility functions for testing deployment
    function deployTestAgents() external {
        require(agentRegistry != address(0), "Deploy contracts first");
        
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        vm.startBroadcast(deployerPrivateKey);
        
        // Deploy test agents
        _deployTestAgent("TestAgent1", "https://test-agent1.com", _getTestCapabilities1());
        _deployTestAgent("TestAgent2", "https://test-agent2.com", _getTestCapabilities2());
        _deployTestAgent("TestAgent3", "https://test-agent3.com", _getTestCapabilities3());
        
        vm.stopBroadcast();
        
        console.log("Test agents deployed successfully");
    }
    
    function _deployTestAgent(
        string memory name,
        string memory endpoint,
        bytes32[] memory capabilities
    ) internal {
        AgentRegistry registry = AgentRegistry(agentRegistry);
        registry.registerAgent(name, endpoint, capabilities);
        console.log("Registered test agent:", name);
    }
    
    function _getTestCapabilities1() internal pure returns (bytes32[] memory) {
        bytes32[] memory capabilities = new bytes32[](3);
        capabilities[0] = keccak256("data_analysis");
        capabilities[1] = keccak256("blockchain_query");
        capabilities[2] = keccak256("document_processing");
        return capabilities;
    }
    
    function _getTestCapabilities2() internal pure returns (bytes32[] memory) {
        bytes32[] memory capabilities = new bytes32[](2);
        capabilities[0] = keccak256("text_processing");
        capabilities[1] = keccak256("web_search");
        return capabilities;
    }
    
    function _getTestCapabilities3() internal pure returns (bytes32[] memory) {
        bytes32[] memory capabilities = new bytes32[](3);
        capabilities[0] = keccak256("image_processing");
        capabilities[1] = keccak256("semantic_search");
        capabilities[2] = keccak256("identity_verification");
        return capabilities;
    }
}

/**
 * Usage:
 * 1. Set environment variables in .env:
 *    - PRIVATE_KEY=your_private_key
 *    - ANP_ENDPOINT=https://your-anp-gateway.com (optional)
 *    - ANP_ENABLED=true (optional)
 *    - ACP_ENDPOINT=https://your-acp-gateway.com (optional)
 *    - ACP_ENABLED=true (optional)
 * 
 * 2. Deploy to local network:
 *    forge script script/DeployBridgeContracts.s.sol --fork-url http://localhost:8545 --broadcast
 * 
 * 3. Deploy to testnet:
 *    forge script script/DeployBridgeContracts.s.sol --rpc-url $SEPOLIA_RPC_URL --broadcast --verify
 * 
 * 4. Deploy test agents (optional):
 *    forge script script/DeployBridgeContracts.s.sol --fork-url http://localhost:8545 --broadcast --sig "deployTestAgents()"
 */