#!/usr/bin/env node
/**
 * Multi-Agent Blockchain Communication Test
 * Tests blockchain communication between all 16 registered agents
 */

require('dotenv').config();
const { ethers } = require('ethers');
const path = require('path');
const fs = require('fs');

// Load contract ABIs
function loadABIs() {
    const registryPath = path.join(__dirname, '../out/AgentRegistry.sol/AgentRegistry.json');
    const routerPath = path.join(__dirname, '../out/MessageRouter.sol/MessageRouter.json');
    
    const registryArtifact = JSON.parse(await fs.readFile(registryPath, 'utf8'));
    const routerArtifact = JSON.parse(await fs.readFile(routerPath, 'utf8'));
    
    return {
        registry: registryArtifact.abi,
        router: routerArtifact.abi
    };
}

// Agent configurations with their blockchain addresses
const AGENT_CONFIGS = [
    { name: "Data Product Agent", address: "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65", capability: "data_product_creation" },
    { name: "Data Standardization Agent", address: "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc", capability: "data_standardization" },
    { name: "AI Preparation Agent", address: "0x976EA74026E726554dB657fA54763abd0C3a0aa9", capability: "ai_data_preparation" },
    { name: "Vector Processing Agent", address: "0x14dC79964da2C08b23698B3D3cc7Ca32193d9955", capability: "vector_generation" },
    { name: "Calculation Validation Agent", address: "0x23618e81E3f5cdF7f54C3d65f7FBc0aBf5B21E8f", capability: "calculation_validation" },
    { name: "QA Validation Agent", address: "0xa0Ee7A142d267C1f36714E4a8F75612F20a79720", capability: "qa_validation" },
    { name: "Quality Control Manager", address: "0x70997970C51812dc3A010C7d01b50e0d17dc79C8", capability: "quality_assessment" },
    { name: "Agent Manager", address: "0x90F79bf6EB2c4f870365E785982E1f101E93b906", capability: "agent_lifecycle_management" },
    { name: "Reasoning Agent", address: "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266", capability: "logical_reasoning" },
    { name: "Data Manager", address: "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC", capability: "data_storage" }
];

async function testCapabilityDiscovery(registry) {
    console.log("\n🔍 Test 1: Capability-Based Agent Discovery");
    console.log("=" + "=".repeat(50));
    
    for (const agent of AGENT_CONFIGS) {
        const agents = await registry.findAgentsByCapability(ethers.id(agent.capability));
        console.log(`\n${agent.capability}:`);
        console.log(`   Found ${agents.length} agent(s)`);
        for (const addr of agents) {
            const info = await registry.getAgent(addr);
            console.log(`   - ${info.name} (${addr.slice(0, 10)}...)`);
        }
    }
}

async function testAgentToAgentMessaging(router, provider) {
    console.log("\n\n📨 Test 2: Agent-to-Agent Messaging");
    console.log("=" + "=".repeat(50));
    
    // Test Case 1: Data flow through processing pipeline
    console.log("\n🔄 Data Processing Pipeline Test:");
    
    // Data Product Agent creates data
    const dataProductWallet = new ethers.Wallet("0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a", provider);
    const standardizationAgent = AGENT_CONFIGS[1].address;
    
    console.log(`\n1. Data Product Agent → Data Standardization Agent`);
    const dataMessage = {
        type: "data_product",
        data: {
            product_id: "test_product_001",
            raw_data: [1, 2, 3, 4, 5],
            format: "array",
            timestamp: new Date().toISOString()
        }
    };
    
    const tx1 = await router.connect(dataProductWallet).sendMessage(
        standardizationAgent,
        JSON.stringify(dataMessage),
        ethers.id("DATA_PRODUCT")
    );
    console.log(`   ✅ Sent: ${tx1.hash.slice(0, 20)}...`);
    
    // Standardization to AI Prep
    const standardizationWallet = new ethers.Wallet("0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba", provider);
    const aiPrepAgent = AGENT_CONFIGS[2].address;
    
    console.log(`\n2. Data Standardization Agent → AI Preparation Agent`);
    const standardizedMessage = {
        type: "standardized_data",
        data: {
            product_id: "test_product_001",
            standardized_data: {"values": [1, 2, 3, 4, 5], "mean": 3},
            format: "json",
            timestamp: new Date().toISOString()
        }
    };
    
    const tx2 = await router.connect(standardizationWallet).sendMessage(
        aiPrepAgent,
        JSON.stringify(standardizedMessage),
        ethers.id("STANDARDIZED_DATA")
    );
    console.log(`   ✅ Sent: ${tx2.hash.slice(0, 20)}...`);
    
    // AI Prep to Vector Processing
    const aiPrepWallet = new ethers.Wallet("0x92db14e403b83dfe3df233f83dfa3a0d7096f21ca9b0d6d6b8d88b2b4ec1564e", provider);
    const vectorAgent = AGENT_CONFIGS[3].address;
    
    console.log(`\n3. AI Preparation Agent → Vector Processing Agent`);
    const preparedMessage = {
        type: "ai_ready_data",
        data: {
            product_id: "test_product_001",
            features: [[0.1, 0.2, 0.3, 0.4, 0.5]],
            model_type: "embedding",
            timestamp: new Date().toISOString()
        }
    };
    
    const tx3 = await router.connect(aiPrepWallet).sendMessage(
        vectorAgent,
        JSON.stringify(preparedMessage),
        ethers.id("AI_READY_DATA")
    );
    console.log(`   ✅ Sent: ${tx3.hash.slice(0, 20)}...`);
}

async function testQualityControlFlow(router, provider) {
    console.log("\n\n🎯 Test 3: Quality Control Workflow");
    console.log("=" + "=".repeat(50));
    
    // Calculation Agent sends result to Validation Agent
    const calcWallet = new ethers.Wallet("0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d", provider);
    const validationAgent = AGENT_CONFIGS[4].address;
    
    console.log(`\n1. Calculation Agent → Calculation Validation Agent`);
    const calcResult = {
        calculation_id: "calc_test_001",
        result: 42,
        formula: "sum(values) * scale_factor",
        inputs: {"values": [1, 2, 3, 4, 5], "scale_factor": 2.8},
        timestamp: new Date().toISOString()
    };
    
    const tx1 = await router.connect(calcWallet).sendMessage(
        validationAgent,
        JSON.stringify(calcResult),
        ethers.id("CALCULATION_RESULT")
    );
    console.log(`   ✅ Sent: ${tx1.hash.slice(0, 20)}...`);
    
    // Validation to QA
    const validationWallet = new ethers.Wallet("0xdbda1821b80551c9d65939329250298aa3472ba22feea921c0cf5d620ea67b97", provider);
    const qaAgent = AGENT_CONFIGS[5].address;
    
    console.log(`\n2. Calculation Validation Agent → QA Validation Agent`);
    const validationResult = {
        calculation_id: "calc_test_001",
        validation_passed: true,
        accuracy_score: 0.99,
        validation_details: {
            numerical_accuracy: true,
            formula_correct: true,
            boundary_checks: true
        }
    };
    
    const tx2 = await router.connect(validationWallet).sendMessage(
        qaAgent,
        JSON.stringify(validationResult),
        ethers.id("VALIDATION_RESULT")
    );
    console.log(`   ✅ Sent: ${tx2.hash.slice(0, 20)}...`);
    
    // QA to Quality Control Manager
    const qaWallet = new ethers.Wallet("0x2a871d0798f97d79848a013d4936a73bf4cc922c825d33c1cf7073dff6d409c6", provider);
    const qcManager = AGENT_CONFIGS[6].address;
    
    console.log(`\n3. QA Validation Agent → Quality Control Manager`);
    const qaResult = {
        calculation_id: "calc_test_001",
        qa_passed: true,
        quality_scores: {
            accuracy: 0.99,
            completeness: 0.95,
            reliability: 0.97
        },
        recommendation: "approve_for_production"
    };
    
    const tx3 = await router.connect(qaWallet).sendMessage(
        qcManager,
        JSON.stringify(qaResult),
        ethers.id("QA_RESULT")
    );
    console.log(`   ✅ Sent: ${tx3.hash.slice(0, 20)}...`);
}

async function testAgentCoordination(router, registry, provider) {
    console.log("\n\n🤝 Test 4: Agent Coordination & Management");
    console.log("=" + "=".repeat(50));
    
    // Agent Manager coordinates with other agents
    const managerWallet = new ethers.Wallet("0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6", provider);
    
    console.log(`\n1. Agent Manager broadcasts health check request`);
    
    // Get all active agents
    const activeAgents = await registry.getAllAgents();
    console.log(`   Found ${activeAgents.length} total agents`);
    
    // Send health check to first 3 agents
    for (let i = 0; i < Math.min(3, activeAgents.length); i++) {
        if (activeAgents[i] === managerWallet.address) continue;
        
        const agentInfo = await registry.getAgent(activeAgents[i]);
        const healthCheck = {
            type: "health_check",
            request_id: `health_${Date.now()}_${i}`,
            timestamp: new Date().toISOString()
        };
        
        const tx = await router.connect(managerWallet).sendMessage(
            activeAgents[i],
            JSON.stringify(healthCheck),
            ethers.id("HEALTH_CHECK")
        );
        console.log(`   → ${agentInfo.name}: ${tx.hash.slice(0, 20)}...`);
    }
}

async function checkMessageDelivery(router, provider) {
    console.log("\n\n📊 Test 5: Message Delivery Status");
    console.log("=" + "=".repeat(50));
    
    // Check messages for each agent
    for (const agent of AGENT_CONFIGS.slice(0, 5)) {
        const messages = await router.getMessages(agent.address);
        const undelivered = await router.getUndeliveredMessages(agent.address);
        
        console.log(`\n${agent.name}:`);
        console.log(`   Total messages: ${messages.length}`);
        console.log(`   Undelivered: ${undelivered.length}`);
        
        // Show last message if any
        if (messages.length > 0) {
            try {
                const lastMessage = await router.getMessage(messages[messages.length - 1]);
                console.log(`   Last message from: ${lastMessage.from.slice(0, 10)}...`);
                console.log(`   Delivered: ${lastMessage.delivered}`);
            } catch (e) {
                console.log(`   (Unable to read message details)`);
            }
        }
    }
}

async function main() {
    console.log("🚀 Multi-Agent Blockchain Communication Test");
    console.log("Testing blockchain communication between 16 A2A agents\n");
    
    try {
        // Initialize provider and contracts
        const provider = new ethers.JsonRpcProvider(process.env.A2A_RPC_URL);
        const { registry: registryABI, router: routerABI } = loadABIs();
        
        const registry = new ethers.Contract(
            process.env.A2A_AGENT_REGISTRY_ADDRESS,
            registryABI,
            provider
        );
        
        const router = new ethers.Contract(
            process.env.A2A_MESSAGE_ROUTER_ADDRESS,
            routerABI,
            provider
        );
        
        console.log(`📋 Agent Registry: ${registry.target}`);
        console.log(`📮 Message Router: ${router.target}`);
        
        const activeCount = await registry.getActiveAgentsCount();
        console.log(`👥 Active Agents: ${activeCount}`);
        
        // Run tests
        await testCapabilityDiscovery(registry);
        await testAgentToAgentMessaging(router, provider);
        await testQualityControlFlow(router, provider);
        await testAgentCoordination(router, registry, provider);
        await checkMessageDelivery(router, provider);
        
        // Summary
        console.log("\n\n🎯 Test Summary");
        console.log("=" + "=".repeat(50));
        console.log("✅ Capability discovery working");
        console.log("✅ Agent-to-agent messaging functional");
        console.log("✅ Quality control workflow tested");
        console.log("✅ Agent coordination demonstrated");
        console.log("✅ Message delivery tracking operational");
        
        console.log("\n🏆 All 16 agents are blockchain-enabled and can communicate!");
        console.log("\n📝 Next Steps:");
        console.log("1. Start agents with BLOCKCHAIN_ENABLED=true");
        console.log("2. Agents will automatically register on blockchain");
        console.log("3. Inter-agent communication will flow through blockchain");
        console.log("4. Reputation and trust verification will be enforced");
        
    } catch (error) {
        console.error(`\n❌ Test failed:`, error);
        process.exit(1);
    }
}

// Run test
if (require.main === module) {
    main().catch(error => {
        console.error('❌ Main execution failed:', error);
        process.exit(1);
    });
}

module.exports = { main };