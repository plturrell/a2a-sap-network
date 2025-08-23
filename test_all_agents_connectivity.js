#!/usr/bin/env node
/**
 * Test connectivity for all 16 A2A agents
 * Verifies that adapters can reach their configured backend ports
 */

const axios = require('axios');

// Agent configuration with correct ports
const agents = [
    { id: 0, name: 'Data Product Agent', port: 8000 },
    { id: 1, name: 'Data Standardization Agent', port: 8001 },
    { id: 2, name: 'AI Preparation Agent', port: 8002 }, // Fixed from 8001
    { id: 3, name: 'Vector Processing Agent', port: 8003 },
    { id: 4, name: 'Calculation Validation Agent', port: 8004 },
    { id: 5, name: 'QA Validation Agent', port: 8005 },
    { id: 6, name: 'Quality Control Agent', port: 8006 },
    { id: 7, name: 'Agent Manager', port: 8007 },
    { id: 8, name: 'Data Manager', port: 8008 },
    { id: 9, name: 'Reasoning Agent', port: 8086 }, // Non-sequential port
    { id: 10, name: 'Calculator Agent', port: 8010 },
    { id: 11, name: 'SQL Query Engine', port: 8011 },
    { id: 12, name: 'Registry Agent', port: 8012 },
    { id: 13, name: 'Agent Builder', port: 8013 },
    { id: 14, name: 'Embedding Fine-Tuner', port: 8014 },
    { id: 15, name: 'Orchestrator Agent', port: 8015 }
];

async function checkAgent(agent) {
    const url = `http://localhost:${agent.port}/health`;
    
    try {
        const response = await axios.get(url, { timeout: 2000 });
        return {
            ...agent,
            status: 'online',
            response: response.data
        };
    } catch (error) {
        return {
            ...agent,
            status: 'offline',
            error: error.code || error.message
        };
    }
}

async function testAllAgents() {
    console.log('🔍 Testing connectivity for all 16 A2A agents...\n');
    
    const results = await Promise.all(agents.map(checkAgent));
    
    // Display results
    console.log('📊 Connectivity Test Results:\n');
    console.log('| Agent | Name | Port | Status |');
    console.log('|-------|------|------|--------|');
    
    let onlineCount = 0;
    results.forEach(result => {
        const statusEmoji = result.status === 'online' ? '✅' : '❌';
        console.log(`| ${result.id.toString().padEnd(5)} | ${result.name.padEnd(28)} | ${result.port} | ${statusEmoji} ${result.status} |`);
        if (result.status === 'online') onlineCount++;
    });
    
    console.log(`\n📈 Summary: ${onlineCount}/${agents.length} agents online`);
    
    // Check for port conflicts
    const portMap = new Map();
    let conflicts = false;
    agents.forEach(agent => {
        if (portMap.has(agent.port)) {
            console.log(`\n⚠️  Port conflict detected: Agent ${agent.id} and Agent ${portMap.get(agent.port)} both use port ${agent.port}`);
            conflicts = true;
        } else {
            portMap.set(agent.port, agent.id);
        }
    });
    
    if (!conflicts) {
        console.log('\n✅ No port conflicts detected');
    }
    
    // Show offline agents details
    const offlineAgents = results.filter(r => r.status === 'offline');
    if (offlineAgents.length > 0) {
        console.log('\n❌ Offline agents:');
        offlineAgents.forEach(agent => {
            console.log(`   - Agent ${agent.id} (${agent.name}): ${agent.error}`);
        });
        
        console.log('\n💡 To start offline agents:');
        if (offlineAgents.some(a => a.id === 9)) {
            console.log('   cd a2aAgents/backend/app/a2a/agents/reasoningAgent/active && ./start_agent9.sh');
        }
        if (offlineAgents.some(a => a.id === 14)) {
            console.log('   cd a2aAgents/backend/app/a2a/agents/embeddingFineTuner/active && ./start_agent14.sh');
        }
        if (offlineAgents.some(a => a.id === 15)) {
            console.log('   cd a2aAgents/backend/app/a2a/agents/orchestratorAgent/active && ./start_agent15.sh');
        }
    }
}

// Run the test
testAllAgents().catch(console.error);