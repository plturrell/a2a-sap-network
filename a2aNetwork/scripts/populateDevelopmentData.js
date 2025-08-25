/**
 * Script to populate development data for all Fiori launchpad tiles
 * Uses actual database schema
 */

const cds = require('@sap/cds');

async function populateDevelopmentData() {
    try {
        log.debug('🚀 Populating development data for tiles...\n');

        // Load model and connect to database
        await cds.load('*');
        const db = await cds.connect.to('db');

        // 1. Populate Agent Performance data
        log.debug('📊 Creating agent performance records...');
        const agents = await db.run('SELECT ID FROM a2a_network_Agents');

        // Clear existing performance data
        await db.run('DELETE FROM a2a_network_AgentPerformance');

        for (const agent of agents) {
            const totalTasks = Math.floor(Math.random() * 1000) + 100;
            const successRate = Math.random() * 0.3 + 0.7; // 70-100%
            const successfulTasks = Math.floor(totalTasks * successRate);

            const performanceData = {
                ID: cds.utils.uuid(),
                agent_ID: agent.ID,
                totalTasks: totalTasks,
                successfulTasks: successfulTasks,
                failedTasks: totalTasks - successfulTasks,
                averageResponseTime: Math.floor(Math.random() * 500) + 100, // 100-600ms
                averageGasUsage: Math.floor(Math.random() * 50000) + 20000,
                reputationScore: Math.floor(Math.random() * 400) + 600, // 600-1000
                trustScore: Math.round((0.5 + Math.random() * 0.5) * 100) / 100, // 0.50-1.00
                lastUpdated: new Date().toISOString()
            };

            await db.run(`INSERT INTO a2a_network_AgentPerformance
                (ID, agent_ID, totalTasks, successfulTasks, failedTasks, averageResponseTime,
                 averageGasUsage, reputationScore, trustScore, lastUpdated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                [performanceData.ID, performanceData.agent_ID, performanceData.totalTasks,
                 performanceData.successfulTasks, performanceData.failedTasks,
                 performanceData.averageResponseTime, performanceData.averageGasUsage,
                 performanceData.reputationScore, performanceData.trustScore,
                 performanceData.lastUpdated]
            );
        }
        log.debug(`✅ Created performance data for ${agents.length} agents`);

        // 2. Populate Blockchain Stats
        log.debug('\n⛓️  Creating blockchain statistics...');
        await db.run('DELETE FROM BlockchainService_BlockchainStats');

        const blockchainStats = {
            ID: cds.utils.uuid(),
            blockHeight: 42150,
            gasPrice: 25.5,
            networkStatus: 'connected',
            totalTransactions: 128450,
            averageBlockTime: 13.2,
            timestamp: new Date().toISOString()
        };

        await db.run(`INSERT INTO BlockchainService_BlockchainStats
            (ID, blockHeight, gasPrice, networkStatus, totalTransactions, averageBlockTime, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)`,
            [blockchainStats.ID, blockchainStats.blockHeight, blockchainStats.gasPrice,
             blockchainStats.networkStatus, blockchainStats.totalTransactions,
             blockchainStats.averageBlockTime, blockchainStats.timestamp]
        );
        log.debug('✅ Blockchain statistics created');

        // 3. Populate Capabilities (without foreign key constraints)
        log.debug('\n🔧 Creating capabilities...');
        await db.run('DELETE FROM a2a_network_Capabilities');

        const capabilities = [
            { name: 'Data Processing', description: 'Advanced data processing and transformation' },
            { name: 'Machine Learning', description: 'ML model training and inference' },
            { name: 'Natural Language Processing', description: 'Text analysis and language understanding' },
            { name: 'Image Recognition', description: 'Computer vision and image analysis' },
            { name: 'Blockchain Integration', description: 'Smart contract interaction' },
            { name: 'API Gateway', description: 'API management and routing' },
            { name: 'Data Analytics', description: 'Business intelligence and reporting' },
            { name: 'Security Services', description: 'Authentication and encryption' }
        ];

        for (const cap of capabilities) {
            await db.run(`INSERT INTO a2a_network_Capabilities
                (ID, name, description, createdAt, modifiedAt)
                VALUES (?, ?, ?, ?, ?)`,
                [cds.utils.uuid(), cap.name, cap.description,
                 new Date().toISOString(), new Date().toISOString()]
            );
        }
        log.debug(`✅ Created ${capabilities.length} capabilities`);

        // 4. Link capabilities to agents
        log.debug('\n🔗 Linking capabilities to agents...');
        await db.run('DELETE FROM a2a_network_AgentCapabilities');

        const capabilityIds = await db.run('SELECT ID FROM a2a_network_Capabilities');

        // Give each agent 1-3 random capabilities
        for (const agent of agents) {
            const numCaps = Math.floor(Math.random() * 3) + 1;
            const assignedCaps = new Set();

            for (let i = 0; i < numCaps; i++) {
                const capIndex = Math.floor(Math.random() * capabilityIds.length);
                const capId = capabilityIds[capIndex].ID;

                if (!assignedCaps.has(capId)) {
                    assignedCaps.add(capId);
                    await db.run(`INSERT INTO a2a_network_AgentCapabilities
                        (ID, agent_ID, capability_ID, proficiencyLevel, createdAt, modifiedAt)
                        VALUES (?, ?, ?, ?, ?, ?)`,
                        [cds.utils.uuid(), agent.ID, capId,
                         Math.floor(Math.random() * 5) + 1, // Proficiency 1-5
                         new Date().toISOString(), new Date().toISOString()]
                    );
                }
            }
        }
        log.debug('✅ Agent capabilities linked');

        // 5. Create messages between agents
        log.debug('\n💬 Creating messages...');
        await db.run('DELETE FROM a2a_network_Messages');

        const messageStatuses = ['sent', 'delivered', 'read', 'failed'];
        const protocols = ['REST', 'GRPC', 'WEBSOCKET', 'MQTT'];

        for (let i = 0; i < 50; i++) {
            const fromIndex = Math.floor(Math.random() * agents.length);
            const toIndex = Math.floor(Math.random() * agents.length);

            if (fromIndex !== toIndex) {
                await db.run(`INSERT INTO a2a_network_Messages
                    (ID, sender_ID, recipient_ID, messageHash, protocol, status,
                     timestamp, createdAt, modifiedAt)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                    [cds.utils.uuid(), agents[fromIndex].ID, agents[toIndex].ID,
                     `0x${Math.random().toString(16).substr(2, 64)}`,
                     protocols[Math.floor(Math.random() * protocols.length)],
                     messageStatuses[Math.floor(Math.random() * messageStatuses.length)],
                     new Date(Date.now() - Math.random() * 86400000).toISOString(), // Last 24h
                     new Date().toISOString(), new Date().toISOString()]
                );
            }
        }
        log.debug('✅ Created 50 messages');

        // 6. Create workflows
        log.debug('\n🔄 Creating workflows...');
        await db.run('DELETE FROM a2a_network_Workflows');

        const workflows = [
            { name: 'Data Processing Pipeline', description: 'ETL workflow for data transformation', status: 'active' },
            { name: 'ML Training Workflow', description: 'Automated machine learning pipeline', status: 'active' },
            { name: 'Multi-Agent Coordination', description: 'Complex task orchestration', status: 'active' },
            { name: 'Security Audit Workflow', description: 'Automated security scanning', status: 'draft' },
            { name: 'Backup and Recovery', description: 'System backup automation', status: 'inactive' }
        ];

        for (const wf of workflows) {
            const definition = {
                steps: [
                    { id: 1, name: 'Initialize', type: 'start' },
                    { id: 2, name: 'Process', type: 'action' },
                    { id: 3, name: 'Complete', type: 'end' }
                ]
            };

            await db.run(`INSERT INTO a2a_network_Workflows
                (ID, name, description, definition, status, gasEstimate,
                 createdBy, createdAt, modifiedAt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                [cds.utils.uuid(), wf.name, wf.description,
                 JSON.stringify(definition), wf.status,
                 Math.floor(Math.random() * 50000) + 20000,
                 agents[0].ID, new Date().toISOString(), new Date().toISOString()]
            );
        }
        log.debug('✅ Created 5 workflows');

        // 7. Create performance snapshots
        log.debug('\n📸 Creating performance snapshots...');
        await db.run('DELETE FROM a2a_network_PerformanceSnapshots');

        // Create hourly snapshots for the last 24 hours
        const now = Date.now();
        for (let h = 0; h < 24; h++) {
            for (const agent of agents.slice(0, 5)) { // Top 5 agents only
                const timestamp = new Date(now - h * 3600000);
                await db.run(`INSERT INTO a2a_network_PerformanceSnapshots
                    (ID, agent_ID, timestamp, cpuUsage, memoryUsage, networkLatency,
                     requestsPerMinute, errorRate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
                    [cds.utils.uuid(), agent.ID, timestamp.toISOString(),
                     Math.random() * 80 + 10, // CPU 10-90%
                     Math.random() * 70 + 20, // Memory 20-90%
                     Math.random() * 100 + 10, // Latency 10-110ms
                     Math.floor(Math.random() * 100) + 10, // RPM 10-110
                     Math.random() * 5 // Error rate 0-5%
                    ]
                );
            }
        }
        log.debug('✅ Created performance snapshots');

        // 8. Update Network Stats
        log.debug('\n📈 Updating network statistics...');
        await db.run('DELETE FROM a2a_network_NetworkStats');

        const agentCount = await db.run('SELECT COUNT(*) as count FROM a2a_network_Agents');
        const capCount = await db.run('SELECT COUNT(*) as count FROM a2a_network_Capabilities');
        const msgCount = await db.run('SELECT COUNT(*) as count FROM a2a_network_Messages');
        const avgRep = await db.run('SELECT AVG(reputation) as avg FROM a2a_network_Agents');

        await db.run(`INSERT INTO a2a_network_NetworkStats
            (ID, totalAgents, activeAgents, totalServices, totalCapabilities,
             totalMessages, totalTransactions, averageReputation, networkLoad,
             gasPrice, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
            [cds.utils.uuid(),
             agentCount[0].count, agentCount[0].count, // All agents active
             0, // Services will be added separately
             capCount[0].count,
             msgCount[0].count,
             blockchainStats.totalTransactions,
             Math.floor(avgRep[0].avg || 850),
             Math.floor(Math.random() * 50) + 30, // Load 30-80%
             blockchainStats.gasPrice,
             new Date().toISOString()]
        );
        log.debug('✅ Network statistics updated');

        // Summary
        log.info('\n🎉 Development data population complete!\n');
        log.debug('📊 Summary:');
        log.debug(`   - Agents: ${agents.length}`);
        log.debug(`   - Performance records: ${agents.length}`);
        log.debug(`   - Capabilities: ${capabilities.length}`);
        log.debug('   - Messages: 50');
        log.debug(`   - Workflows: ${workflows.length}`);
        log.debug(`   - Performance snapshots: ${5 * 24}`);
        log.debug('\nAll tiles should now display properly! 🚀');

        process.exit(0);

    } catch (error) {
        console.error('❌ Error populating development data:', error);
        process.exit(1);
    }
}

// Run the population
populateDevelopmentData();