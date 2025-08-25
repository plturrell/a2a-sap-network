#!/usr/bin/env node
/**
 * Register Missing A2A Agents on Blockchain
 * Registers the 6 missing agents with unique addresses
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
            "function isRegistered(address agentAddress) external view returns (bool)",
            "event AgentRegistered(address indexed agent, string name, string endpoint)"
        ];
    }
}

// Missing agents with unique private keys from unused Anvil accounts
const MISSING_AGENTS = {
    calculationAgent: {
        name: "Calculation Agent",
        endpoint: "http://localhost:8020",
        capabilities: [
            "mathematical_calculations",
            "statistical_analysis", 
            "formula_execution",
            "numerical_processing",
            "computation_services"
        ],
        privateKey: "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6", // Anvil account #3
        description: "Performs complex calculations and mathematical operations"
    },
    
    catalogManager: {
        name: "Catalog Manager Agent", 
        endpoint: "http://localhost:8021",
        capabilities: [
            "catalog_management",
            "metadata_indexing",
            "service_discovery",
            "catalog_search",
            "resource_registration"
        ],
        privateKey: "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a", // Anvil account #4
        description: "Manages service catalogs and resource discovery"
    },
    
    agentBuilder: {
        name: "Agent Builder Agent",
        endpoint: "http://localhost:8022", 
        capabilities: [
            "agent_creation",
            "code_generation",
            "template_management",
            "deployment_automation",
            "agent_configuration"
        ],
        privateKey: "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba", // Anvil account #5
        description: "Creates and deploys new agents dynamically"
    },
    
    embeddingFineTuner: {
        name: "Embedding Fine-Tuner Agent",
        endpoint: "http://localhost:8023",
        capabilities: [
            "embedding_optimization",
            "model_fine_tuning", 
            "vector_improvement",
            "performance_tuning",
            "embedding_evaluation"
        ],
        privateKey: "0x92db14e403b83dfe3df233f83dfa3a0d7096f21ca9b0d6d6b8d88b2b4ec1564e", // Anvil account #6
        description: "Fine-tunes and optimizes embedding models"
    },
    
    reasoningAgent: {
        name: "Reasoning Agent",
        endpoint: "http://localhost:8024",
        capabilities: [
            "logical_reasoning",
            "inference_generation",
            "decision_making", 
            "knowledge_synthesis",
            "problem_solving"
        ],
        privateKey: "0x4bbbf85ce3377467afe5d46f804f221813b2bb87f24d81f60f1fcdbf7cbf4356", // Anvil account #7
        description: "Advanced reasoning and decision-making agent"
    },
    
    sqlAgent: {
        name: "SQL Agent",
        endpoint: "http://localhost:8025",
        capabilities: [
            "sql_query_execution",
            "database_operations",
            "query_optimization",
            "data_extraction", 
            "schema_management"
        ],
        privateKey: "0xdbda1821b80551c9d65939329250298aa3472ba22feea921c0cf5d620ea67b97", // Anvil account #8
        description: "Handles SQL operations and database interactions"
    }
};

async function registerMissingAgents() {
    console.log('🚀 Starting Missing A2A Agent Blockchain Registration');
    
    const provider = new ethers.JsonRpcProvider(process.env.A2A_RPC_URL);
    const agentRegistryAddress = process.env.A2A_AGENT_REGISTRY_ADDRESS;
    const abi = loadAgentRegistryABI();
    
    console.log(`📄 Agent Registry: ${agentRegistryAddress}`);
    
    for (const [agentType, config] of Object.entries(MISSING_AGENTS)) {
        try {
            console.log(`\n🔧 Registering ${config.name}...`);
            
            const wallet = new ethers.Wallet(config.privateKey, provider);
            const registry = new ethers.Contract(agentRegistryAddress, abi, wallet);
            
            console.log(`🔑 Agent address: ${wallet.address}`);
            
            // Check if already registered by trying to get agent info
            try {
                const existingAgent = await registry.getAgent(wallet.address);
                if (existingAgent.active) {
                    console.log(`⚠️  ${config.name} is already registered at ${wallet.address}`);
                    console.log(`   Name: ${existingAgent.name}, Active: ${existingAgent.active}`);
                    continue;
                }
            } catch (error) {
                // Agent not registered, continue with registration
                console.log(`📝 ${config.name} not registered yet, proceeding...`);
            }
            
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
            console.log(`   Transaction confirmed in block: ${receipt.blockNumber}`);
            
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
            
            const dataPath = path.join(__dirname, '../data/agents', `${agentType}.json`);
            await fs.writeFile(dataPath, JSON.stringify(agentData));
            console.log(`💾 Agent data saved to ${dataPath}`);
            
        } catch (error) {
            console.error(`❌ Failed to register ${config.name}:`, error.message);
        }
    }
    
    console.log('\n🎯 Missing agent registration complete!');
}

// Run registration
if (require.main === module) {
    registerMissingAgents().catch(error => {
        console.error('Registration failed:', error);
        process.exit(1);
    });
}

module.exports = { registerMissingAgents };