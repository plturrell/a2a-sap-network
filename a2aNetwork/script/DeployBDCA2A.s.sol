// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Script} from "forge-std/Script.sol";
import {console} from "forge-std/console.sol";
import "../src/BusinessDataCloudA2A.sol";
import "../src/AgentRegistry.sol";
import "../src/MessageRouter.sol";

/**
 * @title DeployBDCA2A
 * @dev Deployment script for Business Data Cloud A2A smart contract system
 */
contract DeployBDCA2A is Script {
    
    // Contract addresses (will be populated during deployment)
    BusinessDataCloudA2A public bdcContract;
    AgentRegistry public agentRegistry;
    MessageRouter public messageRouter;
    
    function run() external {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        address deployer = vm.addr(deployerPrivateKey);
        
        console.log("Deploying Business Data Cloud A2A contracts...");
        console.log("Deployer address:", deployer);
        console.log("Deployer balance:", deployer.balance);
        
        vm.startBroadcast(deployerPrivateKey);
        
        // Deploy Agent Registry first (3 required confirmations for multi-sig)
        agentRegistry = new AgentRegistry(3);
        console.log("AgentRegistry deployed at:", address(agentRegistry));
        
        // Deploy Message Router 
        messageRouter = new MessageRouter(address(agentRegistry), 3);
        console.log("MessageRouter deployed at:", address(messageRouter));
        
        // Deploy main Business Data Cloud contract
        bdcContract = new BusinessDataCloudA2A();
        console.log("BusinessDataCloudA2A deployed at:", address(bdcContract));
        
        // Configure initial setup
        _setupInitialConfiguration();
        
        // Register the A2A agents
        _registerA2AAgents();
        
        // Create default workflows
        _createDefaultWorkflows();
        
        // Establish trust relationships
        _establishTrustRelationships();
        
        vm.stopBroadcast();
        
        console.log("\n=== Business Data Cloud A2A Deployment Complete ===");
        console.log("BusinessDataCloudA2A:", address(bdcContract));
        console.log("AgentRegistry:", address(agentRegistry));
        console.log("MessageRouter:", address(messageRouter));
        console.log("Protocol Version:", bdcContract.PROTOCOL_VERSION());
        console.log("Contract Version:", bdcContract.CONTRACT_VERSION());
        
        // Output deployment information for integration
        _outputDeploymentInfo();
    }
    
    function _setupInitialConfiguration() internal {
        console.log("\nSetting up initial configuration...");
        
        // Grant roles to deployer for initial setup
        bdcContract.grantRole(bdcContract.AGENT_MANAGER_ROLE(), msg.sender);
        bdcContract.grantRole(bdcContract.WORKFLOW_EXECUTOR_ROLE(), msg.sender);
        bdcContract.grantRole(bdcContract.TRUST_MANAGER_ROLE(), msg.sender);
        
        console.log("Initial roles granted to deployer");
    }
    
    function _registerA2AAgents() internal {
        console.log("\nRegistering A2A agents...");
        
        // Register Agent 0 - Data Product Registration
        bdcContract.registerA2AAgent(
            "agent0_data_product",
            "Data Product Registration Agent",
            BusinessDataCloudA2A.AgentType.DATA_PRODUCT_REGISTRATION,
            "http://localhost:8003",
            _getDataProductCapabilities(),
            _getDataProductSkills()
        );
        console.log("Registered Agent 0: Data Product Registration");
        
        // Register Agent 1 - Data Standardization
        bdcContract.registerA2AAgent(
            "agent1_standardization",
            "Data Standardization Agent", 
            BusinessDataCloudA2A.AgentType.DATA_STANDARDIZATION,
            "http://localhost:8004",
            _getStandardizationCapabilities(),
            _getStandardizationSkills()
        );
        console.log("Registered Agent 1: Data Standardization");
        
        // Register Agent 2 - AI Preparation
        bdcContract.registerA2AAgent(
            "agent2_ai_preparation",
            "AI Preparation Agent",
            BusinessDataCloudA2A.AgentType.AI_PREPARATION, 
            "http://localhost:8005",
            _getAIPreparationCapabilities(),
            _getAIPreparationSkills()
        );
        console.log("Registered Agent 2: AI Preparation");
        
        // Register Agent 3 - Vector Processing
        bdcContract.registerA2AAgent(
            "agent3_vector_processing",
            "Vector Processing Agent",
            BusinessDataCloudA2A.AgentType.VECTOR_PROCESSING,
            "http://localhost:8008", 
            _getVectorProcessingCapabilities(),
            _getVectorProcessingSkills()
        );
        console.log("Registered Agent 3: Vector Processing");
        
        // Register Agent 4 - Calculation Validation
        bdcContract.registerA2AAgent(
            "agent4_calc_validation", 
            "Calculation Validation Agent",
            BusinessDataCloudA2A.AgentType.CALC_VALIDATION,
            "http://localhost:8006",
            _getCalcValidationCapabilities(),
            _getCalcValidationSkills()
        );
        console.log("Registered Agent 4: Calculation Validation");
        
        // Register Agent 5 - QA Validation
        bdcContract.registerA2AAgent(
            "agent5_qa_validation",
            "QA Validation Agent", 
            BusinessDataCloudA2A.AgentType.QA_VALIDATION,
            "http://localhost:8007",
            _getQAValidationCapabilities(),
            _getQAValidationSkills()
        );
        console.log("Registered Agent 5: QA Validation");
        
        // Register supporting agents
        _registerSupportingAgents();
    }
    
    function _registerSupportingAgents() internal {
        console.log("Registering supporting agents...");
        
        // Data Manager
        bdcContract.registerA2AAgent(
            "data_manager",
            "Data Manager Agent",
            BusinessDataCloudA2A.AgentType.DATA_MANAGER,
            "http://localhost:8001",
            _getDataManagerCapabilities(),
            _getDataManagerSkills()
        );
        console.log("Registered Data Manager");
        
        // Catalog Manager
        bdcContract.registerA2AAgent(
            "catalog_manager", 
            "Catalog Manager Agent",
            BusinessDataCloudA2A.AgentType.CATALOG_MANAGER,
            "http://localhost:8002",
            _getCatalogManagerCapabilities(),
            _getCatalogManagerSkills()
        );
        console.log("Registered Catalog Manager");
        
        // Agent Manager
        bdcContract.registerA2AAgent(
            "agent_manager",
            "Agent Manager",
            BusinessDataCloudA2A.AgentType.AGENT_MANAGER, 
            "http://localhost:8000",
            _getAgentManagerCapabilities(),
            _getAgentManagerSkills()
        );
        console.log("Registered Agent Manager");
    }
    
    function _createDefaultWorkflows() internal {
        console.log("\nCreating default workflows...");
        
        // Complete A2A Processing Workflow
        string[] memory completeWorkflowSteps = new string[](6);
        completeWorkflowSteps[0] = "Data Product Registration (Agent 0)";
        completeWorkflowSteps[1] = "Data Standardization (Agent 1)";
        completeWorkflowSteps[2] = "AI Preparation (Agent 2)";
        completeWorkflowSteps[3] = "Vector Processing (Agent 3)";
        completeWorkflowSteps[4] = "Calculation Validation (Agent 4)";
        completeWorkflowSteps[5] = "QA Validation (Agent 5)";
        
        bytes32[] memory completeWorkflowAgents = new bytes32[](9);
        completeWorkflowAgents[0] = keccak256("agent0_data_product");
        completeWorkflowAgents[1] = keccak256("agent1_standardization");
        completeWorkflowAgents[2] = keccak256("agent2_ai_preparation");
        completeWorkflowAgents[3] = keccak256("agent3_vector_processing");
        completeWorkflowAgents[4] = keccak256("agent4_calc_validation");
        completeWorkflowAgents[5] = keccak256("agent5_qa_validation");
        completeWorkflowAgents[6] = keccak256("data_manager");
        completeWorkflowAgents[7] = keccak256("catalog_manager");
        completeWorkflowAgents[8] = keccak256("agent_manager");
        
        bdcContract.createA2AWorkflow(
            "complete_a2a_processing",
            "Complete A2A Data Processing",
            "End-to-end data processing workflow through all A2A agents",
            completeWorkflowAgents,
            completeWorkflowSteps
        );
        console.log("Created Complete A2A Processing Workflow");
        
        // Validation Only Workflow
        string[] memory validationSteps = new string[](2);
        validationSteps[0] = "Calculation Validation (Agent 4)";
        validationSteps[1] = "QA Validation (Agent 5)";
        
        bytes32[] memory validationAgents = new bytes32[](2);
        validationAgents[0] = keccak256("agent4_calc_validation");
        validationAgents[1] = keccak256("agent5_qa_validation");
        
        bdcContract.createA2AWorkflow(
            "validation_only",
            "Validation Only Workflow",
            "Run only validation agents for testing data",
            validationAgents,
            validationSteps
        );
        console.log("Created Validation Only Workflow");
    }
    
    function _establishTrustRelationships() internal {
        console.log("\nEstablishing trust relationships...");
        
        // Get agent addresses (simplified - in real deployment these would be actual addresses)
        address agent0 = address(uint160(uint256(keccak256("agent0"))));
        address agent1 = address(uint160(uint256(keccak256("agent1"))));
        address agent2 = address(uint160(uint256(keccak256("agent2"))));
        address agent3 = address(uint160(uint256(keccak256("agent3"))));
        address agent4 = address(uint160(uint256(keccak256("agent4"))));
        address agent5 = address(uint160(uint256(keccak256("agent5"))));
        address dataManager = address(uint160(uint256(keccak256("data_manager"))));
        address catalogManager = address(uint160(uint256(keccak256("catalog_manager"))));
        
        // Establish high trust between sequential agents
        bytes32[] memory sharedCaps = new bytes32[](1);
        sharedCaps[0] = keccak256("data_processing");
        
        bdcContract.establishTrustRelationship(agent0, agent1, 95, sharedCaps);
        bdcContract.establishTrustRelationship(agent1, agent2, 95, sharedCaps);
        bdcContract.establishTrustRelationship(agent2, agent3, 95, sharedCaps);
        
        // Validation agents trust each other
        bdcContract.establishTrustRelationship(agent4, agent5, 90, sharedCaps);
        
        // All agents trust data and catalog managers
        bdcContract.establishTrustRelationship(agent0, dataManager, 100, sharedCaps);
        bdcContract.establishTrustRelationship(agent1, dataManager, 100, sharedCaps);
        bdcContract.establishTrustRelationship(agent2, dataManager, 100, sharedCaps);
        bdcContract.establishTrustRelationship(agent3, dataManager, 100, sharedCaps);
        bdcContract.establishTrustRelationship(agent4, dataManager, 100, sharedCaps);
        bdcContract.establishTrustRelationship(agent5, dataManager, 100, sharedCaps);
        
        bdcContract.establishTrustRelationship(agent0, catalogManager, 100, sharedCaps);
        bdcContract.establishTrustRelationship(agent1, catalogManager, 100, sharedCaps);
        bdcContract.establishTrustRelationship(agent2, catalogManager, 100, sharedCaps);
        bdcContract.establishTrustRelationship(agent3, catalogManager, 100, sharedCaps);
        bdcContract.establishTrustRelationship(agent4, catalogManager, 100, sharedCaps);
        bdcContract.establishTrustRelationship(agent5, catalogManager, 100, sharedCaps);
        
        console.log("Trust relationships established");
    }
    
    function _outputDeploymentInfo() internal view {
        console.log("\n=== Deployment Information for Integration ===");
        console.log("Copy this information to your agent configurations:");
        console.log("");
        console.log("Business Data Cloud Contract:");
        console.log("  Address:", address(bdcContract));
        console.log("  ABI: BusinessDataCloudA2A.json");
        console.log("");
        console.log("Agent Registry Contract:");
        console.log("  Address:", address(agentRegistry));
        console.log("  ABI: AgentRegistry.json");
        console.log("");
        console.log("Message Router Contract:");
        console.log("  Address:", address(messageRouter));
        console.log("  ABI: MessageRouter.json");
        console.log("");
        console.log("Environment Variables:");
        console.log("  BDC_CONTRACT_ADDRESS=", address(bdcContract));
        console.log("  AGENT_REGISTRY_ADDRESS=", address(agentRegistry));
        console.log("  MESSAGE_ROUTER_ADDRESS=", address(messageRouter));
        console.log("  A2A_PROTOCOL_VERSION=0.2.9");
        console.log("  CONTRACT_VERSION=1.0.0");
    }
    
    // Capability and skill definitions
    function _getDataProductCapabilities() internal pure returns (bytes32[] memory) {
        bytes32[] memory caps = new bytes32[](4);
        caps[0] = keccak256("data_product_registration");
        caps[1] = keccak256("dublin_core_metadata");
        caps[2] = keccak256("ord_integration");
        caps[3] = keccak256("trust_system");
        return caps;
    }
    
    function _getDataProductSkills() internal pure returns (bytes32[] memory) {
        bytes32[] memory skills = new bytes32[](4);
        skills[0] = keccak256("register_data_product");
        skills[1] = keccak256("validate_metadata");
        skills[2] = keccak256("generate_dublin_core");
        skills[3] = keccak256("ord_registration");
        return skills;
    }
    
    function _getStandardizationCapabilities() internal pure returns (bytes32[] memory) {
        bytes32[] memory caps = new bytes32[](4);
        caps[0] = keccak256("data_standardization");
        caps[1] = keccak256("schema_validation");
        caps[2] = keccak256("multi_format_support");
        caps[3] = keccak256("trust_system");
        return caps;
    }
    
    function _getStandardizationSkills() internal pure returns (bytes32[] memory) {
        bytes32[] memory skills = new bytes32[](5);
        skills[0] = keccak256("standardize_account");
        skills[1] = keccak256("standardize_book");
        skills[2] = keccak256("standardize_location");
        skills[3] = keccak256("standardize_measure");
        skills[4] = keccak256("standardize_product");
        return skills;
    }
    
    function _getAIPreparationCapabilities() internal pure returns (bytes32[] memory) {
        bytes32[] memory caps = new bytes32[](4);
        caps[0] = keccak256("semantic_enrichment");
        caps[1] = keccak256("grok_api_integration");
        caps[2] = keccak256("data_preparation");
        caps[3] = keccak256("trust_system");
        return caps;
    }
    
    function _getAIPreparationSkills() internal pure returns (bytes32[] memory) {
        bytes32[] memory skills = new bytes32[](4);
        skills[0] = keccak256("semantic_analysis");
        skills[1] = keccak256("grok_enhancement");
        skills[2] = keccak256("data_enrichment");
        skills[3] = keccak256("ai_optimization");
        return skills;
    }
    
    function _getVectorProcessingCapabilities() internal pure returns (bytes32[] memory) {
        bytes32[] memory caps = new bytes32[](4);
        caps[0] = keccak256("vector_embeddings");
        caps[1] = keccak256("knowledge_graph");
        caps[2] = keccak256("similarity_search");
        caps[3] = keccak256("trust_system");
        return caps;
    }
    
    function _getVectorProcessingSkills() internal pure returns (bytes32[] memory) {
        bytes32[] memory skills = new bytes32[](4);
        skills[0] = keccak256("generate_embeddings");
        skills[1] = keccak256("similarity_search");
        skills[2] = keccak256("knowledge_graph_ops");
        skills[3] = keccak256("vector_clustering");
        return skills;
    }
    
    function _getCalcValidationCapabilities() internal pure returns (bytes32[] memory) {
        bytes32[] memory caps = new bytes32[](4);
        caps[0] = keccak256("template_based_testing");
        caps[1] = keccak256("computation_validation");
        caps[2] = keccak256("quality_metrics");
        caps[3] = keccak256("trust_system");
        return caps;
    }
    
    function _getCalcValidationSkills() internal pure returns (bytes32[] memory) {
        bytes32[] memory skills = new bytes32[](4);
        skills[0] = keccak256("generate_test_cases");
        skills[1] = keccak256("validate_calculations");
        skills[2] = keccak256("quality_assessment");
        skills[3] = keccak256("performance_testing");
        return skills;
    }
    
    function _getQAValidationCapabilities() internal pure returns (bytes32[] memory) {
        bytes32[] memory caps = new bytes32[](4);
        caps[0] = keccak256("simpleqa_testing");
        caps[1] = keccak256("ord_discovery");
        caps[2] = keccak256("factuality_validation");
        caps[3] = keccak256("trust_system");
        return caps;
    }
    
    function _getQAValidationSkills() internal pure returns (bytes32[] memory) {
        bytes32[] memory skills = new bytes32[](4);
        skills[0] = keccak256("generate_qa_tests");
        skills[1] = keccak256("ord_product_discovery");
        skills[2] = keccak256("factuality_check");
        skills[3] = keccak256("metadata_validation");
        return skills;
    }
    
    function _getDataManagerCapabilities() internal pure returns (bytes32[] memory) {
        bytes32[] memory caps = new bytes32[](4);
        caps[0] = keccak256("data_storage");
        caps[1] = keccak256("hana_integration");
        caps[2] = keccak256("sqlite_fallback");
        caps[3] = keccak256("trust_system");
        return caps;
    }
    
    function _getDataManagerSkills() internal pure returns (bytes32[] memory) {
        bytes32[] memory skills = new bytes32[](4);
        skills[0] = keccak256("store_data");
        skills[1] = keccak256("retrieve_data");
        skills[2] = keccak256("data_backup");
        skills[3] = keccak256("data_sync");
        return skills;
    }
    
    function _getCatalogManagerCapabilities() internal pure returns (bytes32[] memory) {
        bytes32[] memory caps = new bytes32[](4);
        caps[0] = keccak256("service_discovery");
        caps[1] = keccak256("catalog_management");
        caps[2] = keccak256("semantic_search");
        caps[3] = keccak256("trust_system");
        return caps;
    }
    
    function _getCatalogManagerSkills() internal pure returns (bytes32[] memory) {
        bytes32[] memory skills = new bytes32[](4);
        skills[0] = keccak256("discover_services");
        skills[1] = keccak256("manage_catalog");
        skills[2] = keccak256("search_services");
        skills[3] = keccak256("service_health_check");
        return skills;
    }
    
    function _getAgentManagerCapabilities() internal pure returns (bytes32[] memory) {
        bytes32[] memory caps = new bytes32[](4);
        caps[0] = keccak256("workflow_orchestration");
        caps[1] = keccak256("agent_coordination");
        caps[2] = keccak256("trust_verification");
        caps[3] = keccak256("message_routing");
        return caps;
    }
    
    function _getAgentManagerSkills() internal pure returns (bytes32[] memory) {
        bytes32[] memory skills = new bytes32[](4);
        skills[0] = keccak256("orchestrate_workflow");
        skills[1] = keccak256("coordinate_agents");
        skills[2] = keccak256("verify_trust");
        skills[3] = keccak256("route_messages");
        return skills;
    }
}