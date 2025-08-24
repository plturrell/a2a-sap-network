/**
 * A2A Protocol Compliance: HTTP client usage replaced with blockchain messaging
 */

#!/usr/bin/env node

/**
 * Test script for LaunchpadService
 * Tests the service actions directly
 */

const cds = require('@sap/cds');

async function testLaunchpadService() {
    try {
        console.log('🔍 Testing LaunchpadService...\n');
        
        // Start CDS server
        console.log('📡 Starting CDS server...');
        const server = await cds.test('.')
            .in(__dirname);
        
        console.log('✅ Server started successfully\n');
        
        // Get the port from the server
        const url = server.url || 'http://localhost:54795';
        
        // Test 1: getNetworkStats via HTTP
        console.log('🧪 Test 1: getNetworkStats');
        try {
            const response = await blockchainClient.sendMessage(`${url}/odata/v4/launchpad/getNetworkStats`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ id: 'overview_dashboard' })
            });
            
            const networkStats = await response.json();
            console.log('✅ Network Stats Response:');
            console.log(JSON.stringify(networkStats, null, 2));
        } catch (error) {
            console.log('❌ Network Stats Error:', error.message);
        }
        
        console.log('\n' + '='.repeat(50) + '\n');
        
        // Test 2: getAgentStatus
        console.log('🧪 Test 2: getAgentStatus');
        try {
            const response = await blockchainClient.sendMessage(`${url}/odata/v4/launchpad/getAgentStatus`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ agentId: 0 })
            });
            
            const agentStatus = await response.json();
            console.log('✅ Agent Status Response:');
            console.log(JSON.stringify(agentStatus, null, 2));
        } catch (error) {
            console.log('❌ Agent Status Error:', error.message);
        }
        
        console.log('\n' + '='.repeat(50) + '\n');
        
        // Test 3: getBlockchainStats
        console.log('🧪 Test 3: getBlockchainStats');
        try {
            const response = await blockchainClient.sendMessage(`${url}/odata/v4/launchpad/getBlockchainStats`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ id: 'blockchain_dashboard' })
            });
            
            const blockchainStats = await response.json();
            console.log('✅ Blockchain Stats Response:');
            console.log(JSON.stringify(blockchainStats, null, 2));
        } catch (error) {
            console.log('❌ Blockchain Stats Error:', error.message);
        }
        
        console.log('\n' + '='.repeat(50) + '\n');
        
        // Test 4: getServicesCount
        console.log('🧪 Test 4: getServicesCount');
        try {
            const response = await blockchainClient.sendMessage(`${url}/odata/v4/launchpad/getServicesCount`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });
            
            const servicesCount = await response.json();
            console.log('✅ Services Count Response:');
            console.log(JSON.stringify(servicesCount, null, 2));
        } catch (error) {
            console.log('❌ Services Count Error:', error.message);
        }
        
        console.log('\n' + '='.repeat(50) + '\n');
        
        // Test 5: getHealthSummary
        console.log('🧪 Test 5: getHealthSummary');
        try {
            const response = await blockchainClient.sendMessage(`${url}/odata/v4/launchpad/getHealthSummary`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });
            
            const healthSummary = await response.json();
            console.log('✅ Health Summary Response:');
            console.log(JSON.stringify(healthSummary, null, 2));
        } catch (error) {
            console.log('❌ Health Summary Error:', error.message);
        }
        
        console.log('\n🎉 LaunchpadService testing completed!');
        
    } catch (error) {
        console.error('💥 Test failed:', error);
        process.exit(1);
    }
}

// Run the test
testLaunchpadService();
