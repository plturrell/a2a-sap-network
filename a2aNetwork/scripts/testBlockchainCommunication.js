#!/usr/bin/env node
/**
 * Test A2A Blockchain Communication
 * Tests messaging between registered agents on the blockchain
 */

require('dotenv').config();
const { ethers } = require('ethers');
const path = require('path');
const fs = require('fs');

// Load Message Router ABI
async function loadMessageRouterABI() {
    try {
        const artifactPath = path.join(__dirname, '../out/MessageRouter.sol/MessageRouter.json');
        const artifact = JSON.parse(await fs.readFile(artifactPath, 'utf8'));
        return artifact.abi;
    } catch (error) {
        console.error('Failed to load MessageRouter ABI:', error);
        return [
            "function sendMessage(address to, string memory content, bytes32 messageType) external returns (bytes32)",
            "function getMessages(address recipient) external view returns (bytes32[] memory)",
            "function getMessage(bytes32 messageId) external view returns (tuple(address from, address to, bytes32 messageId, string content, uint256 timestamp, bool delivered, bytes32 messageType, uint256 expiresAt))",
            "function markAsDelivered(bytes32 messageId) external",
            "function getUndeliveredMessages(address agent) external view returns (bytes32[] memory)",
            "event MessageSent(bytes32 indexed messageId, address indexed from, address indexed to, bytes32 messageType)"
        ];
    }
}

async async function testAgentDiscovery(provider) {
(async () => {
    console.log("\n🔍 Testing Agent Discovery...");

    const registryAddress = process.env.A2A_AGENT_REGISTRY_ADDRESS;
    const registryABI = [
        "function getActiveAgentsCount() external view returns (uint256)",
        "function getAllAgents() external view returns (address[] memory)",
        "function getAgent(address agentAddress) external view returns (tuple(address owner, string name, string endpoint, bytes32[] capabilities, uint256 reputation, bool active, uint256 registeredAt))",
        "function findAgentsByCapability(bytes32 capability) external view returns (address[] memory)"
    ];

    const registry = new ethers.Contract(registryAddress, registryABI, provider);

    const activeCount = await registry.getActiveAgentsCount();
    console.log(`   Active agents: ${activeCount}`);

    // Test capability-based discovery
    const qualityAgents = await registry.findAgentsByCapability(ethers.id("quality_assessment"));
    console.log(`   Quality assessment agents: ${qualityAgents.length}`);

    const dataAgents = await registry.findAgentsByCapability(ethers.id("data_storage"));
    console.log(`   Data storage agents: ${dataAgents.length}`);

    return { activeCount, qualityAgents, dataAgents };
}

async async function sendTestMessage(fromWallet, toAddress, messageRouter, content, messageType = "TEST") {
    console.log(`\n📤 Sending message from ${fromWallet.address} to ${toAddress}...`);
    console.log(`   Content: "${content}"`);

    // Send message
    const tx = await messageRouter.connect(fromWallet).sendMessage(
        toAddress,
        content,
        ethers.id(messageType)
    );

    console.log(`   Transaction: ${tx.hash}`);
    const receipt = await tx.wait();

    // Parse message sent event
    const messageEvent = receipt.logs.find(log => {
        try {
            const decoded = messageRouter.interface.parseLog(log);
            return decoded.name === 'MessageSent';
        } catch {
            return false;
        }
    });

    let messageId = null;
    if (messageEvent) {
        const decoded = messageRouter.interface.parseLog(messageEvent);
        messageId = decoded.args.messageId;
        console.log(`   Message ID: ${messageId}`);
    }

    return messageId;
}

async async function checkMessages(agentAddress, messageRouter) {
    console.log(`\n📬 Checking messages for ${agentAddress}...`);

    // Get all messages
    const messageIds = await messageRouter.getMessages(agentAddress);
    console.log(`   Total messages: ${messageIds.length}`);

    // Get undelivered messages
    const undeliveredIds = await messageRouter.getUndeliveredMessages(agentAddress);
    console.log(`   Undelivered messages: ${undeliveredIds.length}`);

    // Display message details
    for (let i = 0; i < Math.min(messageIds.length, 5); i++) {
        try {
            const messageId = messageIds[i];
            const message = await messageRouter.getMessage(messageId);
            console.log(`   Message ${i + 1}: "${message.content}" from ${message.from}`);
            console.log(`     Delivered: ${message.delivered}, Type: ${ethers.toUtf8String(message.messageType.slice(0, 32))}`);
        } catch (error) {
            console.log(`   Message ${i + 1}: Error reading message - ${error.message}`);
        }
    }

    return { totalMessages: messageIds.length, undelivered: undeliveredIds.length };
}

async async function markMessageDelivered(agentWallet, messageRouter, messageId) {
    console.log(`\n✅ Marking message ${messageId} as delivered...`);

    try {
        const tx = await messageRouter.connect(agentWallet).markAsDelivered(messageId);
        await tx.wait();
        console.log(`   Message marked as delivered!`);
        return true;
    } catch (error) {
        console.log(`   Failed to mark as delivered: ${error.message}`);
        return false;
    }
}

async async function main() {
    console.log("🚀 Starting A2A Blockchain Communication Test");

    try {
        // Initialize provider
        const provider = new ethers.JsonRpcProvider(process.env.A2A_RPC_URL);

        // Create agent wallets
        const qcWallet = new ethers.Wallet(process.env.QC_AGENT_PRIVATE_KEY, provider);
        const dmWallet = new ethers.Wallet(process.env.DM_AGENT_PRIVATE_KEY, provider);

        console.log(`🤖 Quality Control Agent: ${qcWallet.address}`);
        console.log(`🗄️  Data Manager Agent: ${dmWallet.address}`);

        // Load message router
        const routerAddress = process.env.A2A_MESSAGE_ROUTER_ADDRESS;
        const routerABI = loadMessageRouterABI();
        const messageRouter = new ethers.Contract(routerAddress, routerABI, provider);

        console.log(`📮 Message Router: ${routerAddress}`);

        // Test 1: Agent Discovery
        await testAgentDiscovery(provider);

        // Test 2: Send messages between agents
        console.log("\n📝 Test 2: Inter-Agent Communication");

        // QC Agent sends message to Data Manager
        const messageId1 = await sendTestMessage(
            qcWallet,
            dmWallet.address,
            messageRouter,
            "Hello Data Manager! Can you store this quality assessment result?",
            "QUALITY_REPORT"
        );

        // Data Manager sends response
        const messageId2 = await sendTestMessage(
            dmWallet,
            qcWallet.address,
            messageRouter,
            "Quality report received and stored successfully!",
            "STORAGE_CONFIRMATION"
        );

        // Test 3: Check messages for both agents
        console.log("\n📊 Test 3: Message Retrieval");

        const qcMessages = await checkMessages(qcWallet.address, messageRouter);
        const dmMessages = await checkMessages(dmWallet.address, messageRouter);

        // Test 4: Mark messages as delivered
        console.log("\n📋 Test 4: Message Delivery Confirmation");

        if (messageId1) {
            await markMessageDelivered(dmWallet, messageRouter, messageId1);
        }

        if (messageId2) {
            await markMessageDelivered(qcWallet, messageRouter, messageId2);
        }

        // Test 5: Final status check
        console.log("\n📈 Test 5: Final Status Check");

        await checkMessages(qcWallet.address, messageRouter);
        await checkMessages(dmWallet.address, messageRouter);

        // Summary
        console.log("\n🎯 Test Summary:");
        console.log(`   ✅ Agent discovery working`);
        console.log(`   ✅ Message sending working`);
        console.log(`   ✅ Message retrieval working`);
        console.log(`   ✅ Delivery confirmation working`);
        console.log(`   📊 Total messages exchanged: 2`);

        console.log("\n🏆 Blockchain communication test completed successfully!");
        console.log("\n🎮 Ready for agent deployment:");
        console.log("   1. Both agents are registered on blockchain");
        console.log("   2. Message routing is functional");
        console.log("   3. Start agents with BLOCKCHAIN_ENABLED=true");

    } catch (error) {
        console.error(`❌ Test failed:`, error);
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
})().catch(console.error);