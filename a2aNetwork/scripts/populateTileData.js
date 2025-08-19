/**
 * Script to populate all necessary data for Fiori launchpad tiles
 */

const cds = require('@sap/cds');

async function populateTileData() {
    try {
        log.debug('🚀 Populating tile data for development...');
        
        // Load model and connect to database
        await cds.load('*');
        const db = await cds.connect.to('db');
        
        // 1. Populate Agent Performance data
        log.debug('\n📊 Creating agent performance records...');
        const agents = await db.run(`SELECT ID FROM a2a_network_Agents`);
        
        // Batch insert agent performance data instead of individual inserts
        const performanceDataBatch = agents.map(agent => ({
            ID: cds.utils.uuid(),
            agent_ID: agent.ID,
            successRate: Math.floor(Math.random() * 30) + 70, // 70-100%
            averageResponseTime: Math.floor(Math.random() * 500) + 100, // 100-600ms
            totalRequests: Math.floor(Math.random() * 10000) + 1000,
            failedRequests: Math.floor(Math.random() * 100),
            lastActive: new Date().toISOString(),
            uptime: Math.floor(Math.random() * 95) + 5, // 5-100%
            createdAt: new Date().toISOString(),
            modifiedAt: new Date().toISOString()
        }));
        
        // Batch insert all performance data at once
        if (performanceDataBatch.length > 0) {
            const values = performanceDataBatch.map(data => 
                `('${data.ID}', '${data.agent_ID}', ${data.successRate}, ${data.averageResponseTime}, ${data.totalRequests}, ${data.failedRequests}, '${data.lastActive}', ${data.uptime}, '${data.createdAt}', '${data.modifiedAt}')`
            ).join(', ');
            
            await db.run(`INSERT OR IGNORE INTO a2a_network_AgentPerformance 
                (ID, agent_ID, successRate, averageResponseTime, totalRequests, failedRequests, lastActive, uptime, createdAt, modifiedAt) 
                VALUES ${values}`);
        }
        log.debug('✅ Agent performance data created');
        
        // 2. Populate Blockchain Stats
        log.debug('\n⛓️  Creating blockchain statistics...');
        const blockchainStats = {
            ID: cds.utils.uuid(),
            blockHeight: 42150,
            gasPrice: 25.5,
            networkStatus: 'connected',
            totalTransactions: 128450,
            averageBlockTime: 13.2,
            timestamp: new Date().toISOString()
        };
        
        await db.run(`DELETE FROM BlockchainService_BlockchainStats`);
        await db.run(`INSERT INTO BlockchainService_BlockchainStats 
            (ID, blockHeight, gasPrice, networkStatus, totalTransactions, averageBlockTime, timestamp) 
            VALUES (?, ?, ?, ?, ?, ?, ?)`,
            [blockchainStats.ID, blockchainStats.blockHeight, blockchainStats.gasPrice,
             blockchainStats.networkStatus, blockchainStats.totalTransactions,
             blockchainStats.averageBlockTime, blockchainStats.timestamp]
        );
        log.debug('✅ Blockchain statistics created');
        
        // 3. Populate Capabilities
        log.debug('\n🔧 Creating capabilities...');
        const capabilities = [
            { name: 'Data Processing', category: 'COMPUTE', description: 'Advanced data processing and transformation' },
            { name: 'Machine Learning', category: 'AI', description: 'ML model training and inference' },
            { name: 'Natural Language Processing', category: 'AI', description: 'Text analysis and language understanding' },
            { name: 'Image Recognition', category: 'AI', description: 'Computer vision and image analysis' },
            { name: 'Blockchain Integration', category: 'BLOCKCHAIN', description: 'Smart contract interaction' },
            { name: 'API Gateway', category: 'NETWORK', description: 'API management and routing' }
        ];
        
        for (const cap of capabilities) {
            await db.run(`INSERT OR IGNORE INTO a2a_network_Capabilities 
                (ID, name, category, description, isActive, createdAt, modifiedAt) 
                VALUES (?, ?, ?, ?, ?, ?, ?)`,
                [cds.utils.uuid(), cap.name, cap.category, cap.description, 1,
                 new Date().toISOString(), new Date().toISOString()]
            );
        }
        log.debug('✅ Capabilities created');
        
        // 4. Link capabilities to agents
        log.debug('\n🔗 Linking capabilities to agents...');
        const capabilityIds = await db.run(`SELECT ID FROM a2a_network_Capabilities`);
        let capIndex = 0;
        
        for (const agent of agents.slice(0, 10)) { // First 10 agents get capabilities
            const numCaps = Math.floor(Math.random() * 3) + 1; // 1-3 capabilities per agent
            for (let i = 0; i < numCaps && capIndex < capabilityIds.length; i++) {
                await db.run(`INSERT OR IGNORE INTO a2a_network_AgentCapabilities 
                    (ID, agent_ID, capability_ID, proficiencyLevel, createdAt, modifiedAt) 
                    VALUES (?, ?, ?, ?, ?, ?)`,
                    [cds.utils.uuid(), agent.ID, capabilityIds[capIndex % capabilityIds.length].ID,
                     Math.floor(Math.random() * 5) + 1, new Date().toISOString(), new Date().toISOString()]
                );
                capIndex++;
            }
        }
        log.debug('✅ Agent capabilities linked');
        
        // 5. Create some messages
        log.debug('\n💬 Creating messages...');
        for (let i = 0; i < 25; i++) {
            const fromAgent = agents[Math.floor(Math.random() * agents.length)];
            const toAgent = agents[Math.floor(Math.random() * agents.length)];
            
            if (fromAgent.ID !== toAgent.ID) {
                await db.run(`INSERT OR IGNORE INTO a2a_network_Messages 
                    (ID, sender_ID, recipient_ID, messageHash, protocol, status, timestamp, createdAt, modifiedAt) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                    [cds.utils.uuid(), fromAgent.ID, toAgent.ID, 
                     `0x${Math.random().toString(16).substr(2, 64)}`, 'REST',
                     ['delivered', 'pending', 'failed'][Math.floor(Math.random() * 3)],
                     new Date().toISOString(), new Date().toISOString(), new Date().toISOString()]
                );
            }
        }
        log.debug('✅ Messages created');
        
        // 6. Create network configuration
        log.debug('\n⚙️  Creating network configuration...');
        await db.run(`INSERT OR IGNORE INTO a2a_network_NetworkConfig 
            (ID, configKey, configValue, category, isEditable, lastModified, modifiedBy) 
            VALUES (?, ?, ?, ?, ?, ?, ?)`,
            [cds.utils.uuid(), 'network.consensus.algorithm', 'PoS', 'NETWORK', 1,
             new Date().toISOString(), 'system']
        );
        log.debug('✅ Network configuration created');
        
        // 7. Create some workflows
        log.debug('\n🔄 Creating workflows...');
        const workflows = [
            { name: 'Data Pipeline', description: 'ETL data processing workflow' },
            { name: 'ML Training Pipeline', description: 'Automated ML model training' },
            { name: 'Agent Coordination', description: 'Multi-agent task coordination' }
        ];
        
        for (const wf of workflows) {
            await db.run(`INSERT OR IGNORE INTO a2a_network_Workflows 
                (ID, name, description, definition, isActive, createdBy, createdAt, modifiedAt) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
                [cds.utils.uuid(), wf.name, wf.description, '{"steps": []}', 1,
                 agents[0].ID, new Date().toISOString(), new Date().toISOString()]
            );
        }
        log.debug('✅ Workflows created');
        
        // 8. Update network stats
        log.debug('\n📈 Updating network statistics...');
        const stats = await db.run(`SELECT COUNT(*) as count FROM a2a_network_NetworkStats`);
        if (stats[0].count === 0) {
            await db.run(`INSERT INTO a2a_network_NetworkStats 
                (ID, totalAgents, activeAgents, totalServices, totalCapabilities, totalMessages, 
                 totalTransactions, averageReputation, networkLoad, gasPrice, timestamp) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                [cds.utils.uuid(), 15, 15, 0, 6, 25, 128450, 850, 45, 25.5, new Date().toISOString()]
            );
        }
        log.debug('✅ Network statistics updated');
        
        log.info('\n🎉 All tile data populated successfully!');
        
        // Display summary
        const summary = {
            agents: await db.run(`SELECT COUNT(*) as count FROM a2a_network_Agents`),
            performance: await db.run(`SELECT COUNT(*) as count FROM a2a_network_AgentPerformance`),
            capabilities: await db.run(`SELECT COUNT(*) as count FROM a2a_network_Capabilities`),
            messages: await db.run(`SELECT COUNT(*) as count FROM a2a_network_Messages`),
            workflows: await db.run(`SELECT COUNT(*) as count FROM a2a_network_Workflows`),
            blockchainStats: await db.run(`SELECT COUNT(*) as count FROM BlockchainService_BlockchainStats`)
        };
        
        log.debug('\n📊 Database Summary:');
        log.debug(`   Agents: ${summary.agents[0].count}`);
        log.debug(`   Performance Records: ${summary.performance[0].count}`);
        log.debug(`   Capabilities: ${summary.capabilities[0].count}`);
        log.debug(`   Messages: ${summary.messages[0].count}`);
        log.debug(`   Workflows: ${summary.workflows[0].count}`);
        log.debug(`   Blockchain Stats: ${summary.blockchainStats[0].count}`);
        
        process.exit(0);
        
    } catch (error) {
        console.error('❌ Error populating tile data:', error);
        process.exit(1);
    }
}

// Run the population
populateTileData();