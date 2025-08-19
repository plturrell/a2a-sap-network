#!/usr/bin/env node
/**
 * Register A2A Agents on Blockchain Registry
 * This script registers the Quality Control Manager and Data Manager agents on the blockchain
 */

require('dotenv').config();
const { ethers } = require('ethers');
const path = require('path');
const fs = require('fs');

// Load actual ABI from compiled artifacts
async function loadAgentRegistryABI() {
    try {
        const artifactPath = path.join(__dirname, '../out/AgentRegistry.sol/AgentRegistry.json');
        const artifact = JSON.parse(await fs.readFile(artifactPath, 'utf8'));
        return artifact.abi;
    } catch (error) {
        console.error('Failed to load AgentRegistry ABI:', error);
        // Fallback to basic ABI
        return [
            "function registerAgent(string memory name, string memory endpoint, bytes32[] memory capabilities) external",
            "function getAgent(address agentAddress) external view returns (tuple(address owner, string name, string endpoint, bytes32[] capabilities, uint256 reputation, bool active, uint256 registeredAt))",
            "function getActiveAgentsCount() external view returns (uint256)",
            "function findAgentsByCapability(bytes32 capability) external view returns (address[] memory)",
            "event AgentRegistered(address indexed agent, string name, string endpoint)"
        ];
    }
}

const MESSAGE_ROUTER_ABI = [
    "function sendMessage(address to, string memory content, bytes32 messageType) external returns (bytes32)",
    "function getMessages(address recipient) external view returns (bytes32[] memory)",
    "function getMessage(bytes32 messageId) external view returns (tuple(address from, address to, string content, bytes32 messageType, uint256 timestamp, bool delivered))"
];

// Agent configurations - all 16 agents with unique addresses
const AGENTS = {
    // Core Processing Agents (0-5)
    dataProductAgent: {
        name: "Data Product Agent",
        endpoint: "http://localhost:8000",
        capabilities: [
            "data_product_creation",
            "data_ingestion",
            "data_transformation",
            "quality_control",
            "metadata_management"
        ],
        privateKey: process.env.AGENT0_PRIVATE_KEY || "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a", // Anvil account #4
        description: "Agent 0 - Creates and manages data products from raw data sources"
    },
    
    dataStandardizationAgent: {
        name: "Data Standardization Agent",
        endpoint: "http://localhost:8001",
        capabilities: [
            "data_standardization",
            "format_conversion",
            "schema_validation",
            "data_cleansing",
            "normalization"
        ],
        privateKey: process.env.AGENT1_PRIVATE_KEY || "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba", // Anvil account #5
        description: "Agent 1 - Standardizes data formats and ensures consistency"
    },
    
    aiPreparationAgent: {
        name: "AI Preparation Agent",
        endpoint: "http://localhost:8002",
        capabilities: [
            "ai_data_preparation",
            "feature_engineering",
            "data_preprocessing",
            "model_input_formatting",
            "training_data_generation"
        ],
        privateKey: process.env.AGENT2_PRIVATE_KEY || "0x92db14e403b83dfe3df233f83dfa3a0d7096f21ca9b0d6d6b8d88b2b4ec1564e", // Anvil account #6
        description: "Agent 2 - Prepares data for AI/ML model consumption"
    },
    
    vectorProcessingAgent: {
        name: "Vector Processing Agent",
        endpoint: "http://localhost:8003",
        capabilities: [
            "vector_generation",
            "embedding_creation",
            "similarity_search",
            "vector_storage",
            "dimension_reduction"
        ],
        privateKey: process.env.AGENT3_PRIVATE_KEY || "0x4bbbf85ce3377467afe5d46f804f221813b2bb87f24d81f60f1fcdbf7cbf4356", // Anvil account #7
        description: "Agent 3 - Processes and manages vector embeddings"
    },
    
    calculationValidationAgent: {
        name: "Calculation Validation Agent",
        endpoint: "http://localhost:8004",
        capabilities: [
            "calculation_validation",
            "numerical_verification",
            "formula_checking",
            "result_validation",
            "mathematical_accuracy"
        ],
        privateKey: process.env.AGENT4_PRIVATE_KEY || "0xdbda1821b80551c9d65939329250298aa3472ba22feea921c0cf5d620ea67b97", // Anvil account #8
        description: "Agent 4 - Validates calculations and mathematical operations"
    },
    
    qaValidationAgent: {
        name: "QA Validation Agent",
        endpoint: "http://localhost:8005",
        capabilities: [
            "qa_validation",
            "quality_assurance",
            "test_execution",
            "validation_reporting",
            "compliance_checking"
        ],
        privateKey: process.env.AGENT5_PRIVATE_KEY || "0x2a871d0798f97d79848a013d4936a73bf4cc922c825d33c1cf7073dff6d409c6", // Anvil account #9
        description: "Agent 5 - Performs quality assurance and validation checks"
    },
    
    // Management & Control Agents
    qualityControlManager: {
        name: "Quality Control Manager Agent",
        endpoint: "http://localhost:8009",
        capabilities: [
            "quality_assessment",
            "routing_decision", 
            "improvement_recommendations",
            "agent_registry",
            "message_routing",
            "trust_verification",
            "reputation_tracking"
        ],
        privateKey: process.env.QC_AGENT_PRIVATE_KEY || "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d", // Anvil account #1
        description: "Agent 6 - Manages quality control and routing decisions"
    },
    
    agentManager: {
        name: "Agent Manager Agent",
        endpoint: "http://localhost:8010",
        capabilities: [
            "agent_lifecycle_management",
            "agent_registration",
            "health_monitoring",
            "performance_tracking",
            "agent_coordination"
        ],
        privateKey: process.env.AGENT_MANAGER_PRIVATE_KEY || "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6", // Anvil account #3
        description: "Central agent that manages other agents in the network"
    },
    
    // Specialized Agents
    reasoningAgent: {
        name: "Reasoning Agent",
        endpoint: "http://localhost:8011",
        capabilities: [
            "logical_reasoning",
            "inference_generation",
            "decision_making",
            "knowledge_synthesis",
            "problem_solving"
        ],
        privateKey: process.env.REASONING_PRIVATE_KEY || "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80", // Anvil account #0 (admin)
        description: "Advanced reasoning and decision-making agent"
    },
    
    calculationAgent: {
        name: "Calculation Agent",
        endpoint: "http://localhost:8012",
        capabilities: [
            "mathematical_calculations",
            "statistical_analysis",
            "formula_execution",
            "numerical_processing",
            "computation_services"
        ],
        privateKey: process.env.CALC_PRIVATE_KEY || "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d", // Share with QC for now
        description: "Performs complex calculations and mathematical operations"
    },
    
    sqlAgent: {
        name: "SQL Agent",
        endpoint: "http://localhost:8013",
        capabilities: [
            "sql_query_execution",
            "database_operations",
            "query_optimization",
            "data_extraction",
            "schema_management"
        ],
        privateKey: process.env.SQL_PRIVATE_KEY || "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a", // Share with DM for now
        description: "Handles SQL operations and database interactions"
    },
    
    dataManager: {
        name: "Data Manager Agent",
        endpoint: "http://localhost:8001",
        capabilities: [
            "data_storage",
            "caching",
            "persistence",
            "versioning",
            "bulk_operations",
            "reputation_tracking"
        ],
        privateKey: process.env.DM_AGENT_PRIVATE_KEY || "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a", // Anvil account #2
        description: "Centralized data storage and retrieval agent"
    },
    
    catalogManager: {
        name: "Catalog Manager Agent",
        endpoint: "http://localhost:8014",
        capabilities: [
            "catalog_management",
            "metadata_indexing",
            "service_discovery",
            "catalog_search",
            "resource_registration"
        ],
        privateKey: process.env.CATALOG_PRIVATE_KEY || "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a", // Share with agent0
        description: "Manages service catalogs and resource discovery"
    },
    
    agentBuilder: {
        name: "Agent Builder Agent",
        endpoint: "http://localhost:8015",
        capabilities: [
            "agent_creation",
            "code_generation",
            "template_management",
            "deployment_automation",
            "agent_configuration"
        ],
        privateKey: process.env.BUILDER_PRIVATE_KEY || "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba", // Share with agent1
        description: "Creates and deploys new agents dynamically"
    },
    
    embeddingFineTuner: {
        name: "Embedding Fine-Tuner Agent",
        endpoint: "http://localhost:8016",
        capabilities: [
            "embedding_optimization",
            "model_fine_tuning",
            "vector_improvement",
            "performance_tuning",
            "embedding_evaluation"
        ],
        privateKey: process.env.EMBEDDING_PRIVATE_KEY || "0x92db14e403b83dfe3df233f83dfa3a0d7096f21ca9b0d6d6b8d88b2b4ec1564e", // Share with agent2
        description: "Fine-tunes and optimizes embedding models"
    },
    
    orchestratorAgent: {
        name: "Orchestrator Agent",
        endpoint: "http://localhost:8017",
        capabilities: [
            "workflow_orchestration",
            "task_scheduling",
            "pipeline_management",
            "coordination_services",
            "execution_monitoring"
        ],
        privateKey: process.env.ORCHESTRATOR_PRIVATE_KEY || "0x4bbbf85ce3377467afe5d46f804f221813b2bb87f24d81f60f1fcdbf7cbf4356", // Share with agent3
        description: "Orchestrates complex workflows across multiple agents"
    }
};

async async function main() {
(async () => {
    console.log("🚀 Starting A2A Agent Blockchain Registration");
    
    try {
        // Initialize provider and contracts
        const provider = new ethers.JsonRpcProvider(process.env.A2A_RPC_URL || "http://localhost:8545");
        
        // Get contract addresses
        const agentRegistryAddress = process.env.A2A_AGENT_REGISTRY_ADDRESS;
        const messageRouterAddress = process.env.A2A_MESSAGE_ROUTER_ADDRESS;
        
        if (!agentRegistryAddress || !messageRouterAddress) {
            throw new Error("Contract addresses not configured. Please set A2A_AGENT_REGISTRY_ADDRESS and A2A_MESSAGE_ROUTER_ADDRESS");
        }
        
        console.log(`📄 Agent Registry: ${agentRegistryAddress}`);
        console.log(`📄 Message Router: ${messageRouterAddress}`);
        
        // Register each agent
        for (const [agentType, config] of Object.entries(AGENTS)) {
            console.log(`\n🔧 Registering ${config.name}...`);
            
            // Create wallet for agent
            const wallet = new ethers.Wallet(config.privateKey, provider);
            console.log(`🔑 Agent address: ${wallet.address}`);
            
            // Create contract instances
            const registryABI = loadAgentRegistryABI();
            const agentRegistry = new ethers.Contract(agentRegistryAddress, registryABI, wallet);
            const messageRouter = new ethers.Contract(messageRouterAddress, MESSAGE_ROUTER_ABI, wallet);
            
            // Check if already registered
            try {
                const existingAgent = await agentRegistry.getAgent(wallet.address);
                const isRegistered = existingAgent.owner !== '0x0000000000000000000000000000000000000000';
                
                if (isRegistered) {
                    console.log(`✅ ${config.name} is already registered`);
                    console.log(`   Name: ${existingAgent.name}`);
                    console.log(`   Endpoint: ${existingAgent.endpoint}`);
                    console.log(`   Reputation: ${existingAgent.reputation}`);
                    console.log(`   Active: ${existingAgent.active}`);
                    continue;
                }
            } catch (error) {
                // Agent not found, proceed with registration
                console.log(`📝 ${config.name} not yet registered, proceeding...`);
            }
            
            // Convert capabilities to bytes32
            const capabilityHashes = config.capabilities.map(cap => ethers.id(cap));
            
            // Estimate gas for registration
            const gasEstimate = await agentRegistry.registerAgent.estimateGas(
                config.name,
                config.endpoint,
                capabilityHashes
            );
            
            console.log(`⛽ Estimated gas: ${gasEstimate.toString()}`);
            
            // Register agent
            console.log(`📝 Registering agent on blockchain...`);
            const tx = await agentRegistry.registerAgent(
                config.name,
                config.endpoint,
                capabilityHashes,
                {
                    gasLimit: gasEstimate * BigInt(120) / BigInt(100) // Add 20% buffer
                }
            );
            
            console.log(`⏳ Transaction hash: ${tx.hash}`);
            
            // Wait for confirmation
            const receipt = await tx.wait();
            console.log(`✅ Registration confirmed in block ${receipt.blockNumber}`);
            
            // Parse registration event
            const registrationEvent = receipt.logs.find(log => {
                try {
                    const decoded = agentRegistry.interface.parseLog(log);
                    return decoded.name === 'AgentRegistered';
                } catch {
                    return false;
                }
            });
            
            if (registrationEvent) {
                const decoded = agentRegistry.interface.parseLog(registrationEvent);
                console.log(`🎉 Agent registered successfully!`);
                console.log(`   Agent Address: ${decoded.args.agent || decoded.args.agentAddress}`);
                console.log(`   Name: ${decoded.args.name}`);
                console.log(`   Endpoint: ${decoded.args.endpoint}`);
                if (decoded.args.capabilities) {
                    try {
                        console.log(`   Capabilities: ${decoded.args.capabilities.map(cap => ethers.toUtf8String(cap.slice(0, 32)))}`);
                    } catch (e) {
                        console.log(`   Capabilities: [${decoded.args.capabilities.length} items]`);
                    }
                }
            }
            
            // Verify registration
            const agentInfo = await agentRegistry.getAgent(wallet.address);
            console.log(`\n📊 Agent Info:`);
            console.log(`   Name: ${agentInfo.name}`);
            console.log(`   Endpoint: ${agentInfo.endpoint}`);
            console.log(`   Reputation: ${agentInfo.reputation}`);
            console.log(`   Active: ${agentInfo.active}`);
            
            // Save agent configuration for future use
            const agentConfig = {
                agentType,
                name: config.name,
                address: wallet.address,
                endpoint: config.endpoint,
                capabilities: config.capabilities,
                description: config.description,
                registrationTx: tx.hash,
                registrationBlock: receipt.blockNumber,
                registeredAt: new Date().toISOString()
            };
            
            const configPath = path.join(__dirname, `../data/agents/${agentType}.json`);
            fs.mkdirSync(path.dirname(configPath), { recursive: true });
            await fs.writeFile(configPath, JSON.stringify(agentConfig));
            console.log(`💾 Agent config saved to ${configPath}`);
        }
        
        // Display summary
        console.log(`\n📈 Registration Summary:`);
        const registryABI = loadAgentRegistryABI();
        const agentRegistry = new ethers.Contract(
            agentRegistryAddress, 
            registryABI, 
            new ethers.Wallet(AGENTS.qualityControlManager.privateKey, provider)
        );
        
        const totalAgents = await agentRegistry.getActiveAgentsCount();
        console.log(`   Total active agents: ${totalAgents}`);
        
        console.log(`\n🎯 Next Steps:`);
        console.log(`   1. Start Quality Control Manager: cd a2aAgents/backend/app/a2a/agents/agent6QualityControl/active && python qualityControlManagerAgent.py`);
        console.log(`   2. Start Data Manager: cd a2aAgents/backend/services/dataManager && python src/server.py`);
        console.log(`   3. Test blockchain communication between agents`);
        console.log(`   4. Verify reputation tracking and trust verification`);
        
    } catch (error) {
        console.error(`❌ Registration failed:`, error);
        
        if (error.code === 'CALL_EXCEPTION') {
            console.error(`Contract call failed. Check:`)
            console.error(`- Contract addresses are correct`);
            console.error(`- Network is running and accessible`);
            console.error(`- Agent has sufficient ETH for gas`);
        }
        
        process.exit(1);
    }
}

// Helper function to convert string to bytes32
async function stringToBytes32(str) {
    const bytes = ethers.toUtf8Bytes(str);
    if (bytes.length > 32) {
        throw new Error(`String too long: ${str}`);
    }
    const padded = ethers.zeroPadRight(bytes, 32);
    return padded;
}

// Test blockchain connectivity
async async function testConnectivity() {
    console.log("🔍 Testing blockchain connectivity...");
    
    try {
        const provider = new ethers.JsonRpcProvider(process.env.A2A_RPC_URL || "http://localhost:8545");
        
        const network = await provider.getNetwork();
        console.log(`🌐 Connected to network: ${network.name} (chainId: ${network.chainId})`);
        
        const blockNumber = await provider.getBlockNumber();
        console.log(`📦 Latest block: ${blockNumber}`);
        
        return true;
    } catch (error) {
        console.error(`❌ Connectivity test failed:`, error.message);
        return false;
    }
}

// Enhanced error handling
process.on('uncaughtException', (error) => {
    console.error('❌ Uncaught Exception:', error);
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('❌ Unhandled Rejection at:', promise, 'reason:', reason);
    process.exit(1);
});

// Run the registration
if (require.main === module) {
    testConnectivity().then(connected => {
        if (connected) {
            main().catch(error => {
                console.error('❌ Main execution failed:', error);
                process.exit(1);
            });
        } else {
            console.error('❌ Cannot proceed without blockchain connectivity');
            process.exit(1);
        }
    });
}

module.exports = { main, AGENTS, loadAgentRegistryABI, MESSAGE_ROUTER_ABI };
})().catch(console.error);