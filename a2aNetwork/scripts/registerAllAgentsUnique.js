#!/usr/bin/env node
/**
 * Register ALL A2A Agents with Unique Addresses
 * Each agent gets its own Anvil account for proper blockchain identity
 */

require('dotenv').config();
const { ethers } = require('ethers');
const path = require('path');
const fs = require('fs').promises;

// Load AgentRegistry ABI
async function loadAgentRegistryABI() {
    try {
        const artifactPath = path.join(__dirname, '../out/AgentRegistry.sol/AgentRegistry.json');
        const artifact = JSON.parse(await fs.readFile(artifactPath, 'utf8'));
        return artifact.abi;
    } catch (error) {
        console.error('Failed to load AgentRegistry ABI:', error);
        return [
            "function registerAgent(string memory name, string memory endpoint, bytes32[] memory capabilities) external",
            "function getAgent(address agentAddress) external view returns (tuple(address owner, string name, string endpoint, bytes32[] capabilities, uint256 reputation, bool active, uint256 registeredAt))",
            "function getActiveAgentsCount() external view returns (uint256)",
            "event AgentRegistered(address indexed agent, string name, string endpoint)"
        ];
    }
}

// All 16 agents with unique Anvil accounts (0-15)
const ALL_AGENTS = {
    // Core Processing Agents (0-5)
    dataProductAgent: {
        name: "Data Product Agent",
        endpoint: "http://localhost:8000",
        capabilities: ["data_product_creation", "data_ingestion", "data_transformation", "quality_control", "metadata_management"],
        privateKey: "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80", // Account 0
        description: "Agent 0 - Creates and manages data products with Dublin Core metadata"
    },

    dataStandardizationAgent: {
        name: "Data Standardization Agent",
        endpoint: "http://localhost:8001",
        capabilities: ["data_standardization", "schema_validation", "format_conversion", "data_normalization", "quality_improvement"],
        privateKey: "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d", // Account 1
        description: "Agent 1 - Standardizes data formats and validates schemas"
    },

    aiPreparationAgent: {
        name: "AI Preparation Agent",
        endpoint: "http://localhost:8002",
        capabilities: ["ai_data_preparation", "feature_engineering", "data_preprocessing", "ml_optimization", "embedding_preparation"],
        privateKey: "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a", // Account 2
        description: "Agent 2 - Prepares data for AI/ML processing with advanced preprocessing"
    },

    vectorProcessingAgent: {
        name: "Vector Processing Agent",
        endpoint: "http://localhost:8003",
        capabilities: ["vector_generation", "embedding_creation", "similarity_search", "vector_optimization", "semantic_analysis"],
        privateKey: "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6", // Account 3
        description: "Agent 3 - Generates and processes vector embeddings for semantic analysis"
    },

    calculationValidationAgent: {
        name: "Calculation Validation Agent",
        endpoint: "http://localhost:8004",
        capabilities: ["calculation_validation", "numerical_verification", "statistical_analysis", "accuracy_checking", "error_detection"],
        privateKey: "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a", // Account 4
        description: "Agent 4 - Validates calculations and numerical computations"
    },

    qaValidationAgent: {
        name: "QA Validation Agent",
        endpoint: "http://localhost:8005",
        capabilities: ["qa_validation", "quality_assurance", "test_execution", "validation_reporting", "compliance_checking"],
        privateKey: "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba", // Account 5
        description: "Agent 5 - Performs quality assurance and validation checks"
    },

    // Management & Control Agents (6-9)
    qualityControlManager: {
        name: "Quality Control Manager Agent",
        endpoint: "http://localhost:8006",
        capabilities: ["quality_assessment", "routing_decision", "improvement_recommendations", "workflow_control", "trust_verification"],
        privateKey: "0x92db14e403b83dfe3df233f83dfa3a0d7096f21ca9b0d6d6b8d88b2b4ec1564e", // Account 6
        description: "Agent 6 - Manages quality control and routing decisions"
    },

    agentManager: {
        name: "Agent Manager Agent",
        endpoint: "http://localhost:8007",
        capabilities: ["agent_lifecycle_management", "agent_registration", "health_monitoring", "performance_tracking", "agent_coordination"],
        privateKey: "0x4bbbf85ce3377467afe5d46f804f221813b2bb87f24d81f60f1fcdbf7cbf4356", // Account 7
        description: "Agent 7 - Central agent that manages other agents in the network"
    },

    dataManager: {
        name: "Data Manager Agent",
        endpoint: "http://localhost:8008",
        capabilities: ["data_storage", "caching", "persistence", "versioning", "bulk_operations"],
        privateKey: "0xdbda1821b80551c9d65939329250298aa3472ba22feea921c0cf5d620ea67b97", // Account 8
        description: "Agent 8 - Centralized data storage and retrieval agent"
    },

    reasoningAgent: {
        name: "Reasoning Agent",
        endpoint: "http://localhost:8009",
        capabilities: ["logical_reasoning", "inference_generation", "decision_making", "knowledge_synthesis", "problem_solving"],
        privateKey: "0x2a871d0798f97d79848a013d4936a73bf4cc922c825d33c1cf7073dff6d409c6", // Account 9
        description: "Agent 9 - Advanced reasoning and decision-making agent"
    },

    // Specialized Agents (10-15)
    calculationAgent: {
        name: "Calculation Agent",
        endpoint: "http://localhost:8010",
        capabilities: ["mathematical_calculations", "statistical_analysis", "formula_execution", "numerical_processing", "computation_services"],
        privateKey: "0xf214f2b2cd398c806f84e317254e0f0b801d0643303237d97a22a48e01628897", // Account 10
        description: "Agent 10 - Performs complex calculations and mathematical operations"
    },

    sqlAgent: {
        name: "SQL Agent",
        endpoint: "http://localhost:8011",
        capabilities: ["sql_query_execution", "database_operations", "query_optimization", "data_extraction", "schema_management"],
        privateKey: "0x701b615bbdfb9de65240bc28bd21bbc0d996645a3dd57e7b12bc2bdf6f192c82", // Account 11
        description: "Agent 11 - Handles SQL operations and database interactions"
    },

    catalogManager: {
        name: "Catalog Manager Agent",
        endpoint: "http://localhost:8012",
        capabilities: ["catalog_management", "metadata_indexing", "service_discovery", "catalog_search", "resource_registration"],
        privateKey: "0xa267530f49f8280200edf313ee7af6b827f2a8bce2897751d06a843f644967b1", // Account 12
        description: "Agent 12 - Manages service catalogs and resource discovery"
    },

    agentBuilder: {
        name: "Agent Builder Agent",
        endpoint: "http://localhost:8013",
        capabilities: ["agent_creation", "code_generation", "template_management", "deployment_automation", "agent_configuration"],
        privateKey: "0x47c99abed3324a2707c28affff1267e45918ec8c3f20b8aa892e8b065d2942dd", // Account 13
        description: "Agent 13 - Creates and deploys new agents dynamically"
    },

    embeddingFineTuner: {
        name: "Embedding Fine-Tuner Agent",
        endpoint: "http://localhost:8014",
        capabilities: ["embedding_optimization", "model_fine_tuning", "vector_improvement", "performance_tuning", "embedding_evaluation"],
        privateKey: "0xc526ee95bf44d8fc405a158bb884d9d1238d99f0612e9f33d006bb0789009aaa", // Account 14
        description: "Agent 14 - Fine-tunes and optimizes embedding models"
    },

    orchestratorAgent: {
        name: "Orchestrator Agent",
        endpoint: "http://localhost:8015",
        capabilities: ["workflow_orchestration", "task_scheduling", "pipeline_management", "coordination_services", "execution_monitoring"],
        privateKey: "0x8166f546bab6da521a8369cab06c5d2b9e46670292d85c875ee9ec20e84ffb61", // Account 15
        description: "Agent 15 - Orchestrates complex workflows across multiple agents"
    }
};

async function registerAllAgents() {
    console.log('🚀 Starting Complete A2A Agent Registration with Unique Addresses');
    console.log(`📊 Total agents to register: ${Object.keys(ALL_AGENTS).length}`);

    const provider = new ethers.JsonRpcProvider(process.env.A2A_RPC_URL);
    const agentRegistryAddress = process.env.A2A_AGENT_REGISTRY_ADDRESS;
    const abi = loadAgentRegistryABI();

    console.log(`📄 Agent Registry: ${agentRegistryAddress}`);

    // Clear existing agent data
    const dataDir = path.join(__dirname, '../data/agents');
    if (fs.existsSync(dataDir)) {
        fs.rmSync(dataDir, { recursive: true });
    }
    fs.mkdirSync(dataDir, { recursive: true });

    let successCount = 0;
    let failCount = 0;

    for (const [agentType, config] of Object.entries(ALL_AGENTS)) {
        try {
            console.log(`\n🔧 Registering ${config.name}...`);

            const wallet = new ethers.Wallet(config.privateKey, provider);
            const registry = new ethers.Contract(agentRegistryAddress, abi, wallet);

            console.log(`🔑 Agent address: ${wallet.address}`);

            // Convert capabilities to bytes32
            const capabilityHashes = config.capabilities.map(cap => ethers.id(cap));

            // Register agent
            const tx = await registry.registerAgent(
                config.name,
                config.endpoint,
                capabilityHashes
            );

            console.log(`⏳ Transaction submitted: ${tx.hash}`);
            const receipt = await tx.wait();
            console.log(`✅ ${config.name} registered successfully!`);
            console.log(`   Block: ${receipt.blockNumber}, Gas used: ${receipt.gasUsed.toString()}`);

            // Save agent data
            const agentData = {
                agentType: agentType,
                name: config.name,
                address: wallet.address,
                endpoint: config.endpoint,
                capabilities: config.capabilities,
                description: config.description,
                registrationTx: tx.hash,
                registrationBlock: receipt.blockNumber,
                registeredAt: new Date().toISOString()
            };

            const dataPath = path.join(dataDir, `${agentType}.json`);
            await fs.writeFile(dataPath, JSON.stringify(agentData));
            console.log(`💾 Data saved: ${dataPath}`);

            successCount++;

        } catch (error) {
            console.error(`❌ Failed to register ${config.name}:`, error.message);
            failCount++;
        }
    }

    // Get final count from blockchain
    const registry = new ethers.Contract(agentRegistryAddress, abi, provider);
    const activeCount = await registry.getActiveAgentsCount();

    console.log('\n🎯 Registration Summary:');
    console.log(`   ✅ Successfully registered: ${successCount} agents`);
    console.log(`   ❌ Failed registrations: ${failCount} agents`);
    console.log(`   📊 Total active on blockchain: ${activeCount}`);
    console.log(`   🎯 TARGET ACHIEVED: ${successCount === 16 ? '100/100' : `${Math.round(successCount/16*100)}/100`}`);

    return successCount === 16;
}

// Run registration
if (require.main === module) {
    registerAllAgents().then(success => {
        if (success) {
            console.log('\n🏆 ALL 16 AGENTS SUCCESSFULLY REGISTERED WITH UNIQUE ADDRESSES!');
            process.exit(0);
        } else {
            console.log('\n❌ Registration incomplete. Check errors above.');
            process.exit(1);
        }
    }).catch(error => {
        console.error('Registration failed:', error);
        process.exit(1);
    });
}

module.exports = { registerAllAgents };