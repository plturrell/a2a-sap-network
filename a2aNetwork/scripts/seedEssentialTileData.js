/**
 * Script to seed only essential data needed for tiles to work
 * Focuses on fixing tile endpoint errors
 */

const cds = require('@sap/cds');

async function seedEssentialData() {
    try {
        log.debug('🚀 Seeding essential tile data...\n');
        
        // Load model and connect to database
        await cds.load('*');
        const db = await cds.connect.to('db');
        
        log.debug('📊 Checking what we have...');
        const agentCount = await db.run('SELECT COUNT(*) as count FROM a2a_network_Agents');
        log.debug(`Current agents: ${agentCount[0].count}`);
        
        // 1. Add essential agent performance data if missing
        const perfCount = await db.run('SELECT COUNT(*) as count FROM a2a_network_AgentPerformance');
        log.debug(`Current performance records: ${perfCount[0].count}`);
        
        if (perfCount[0].count === 0) {
            log.debug('\n🎯 Adding agent performance data...');
            const agents = await db.run('SELECT ID FROM a2a_network_Agents LIMIT 15');
            
            // Batch insert performance data instead of individual inserts
            const performanceDataBatch = agents.map(agent => {
                const totalTasks = Math.floor(Math.random() * 1000) + 100;
                const successRate = Math.random() * 0.3 + 0.7; // 70-100%
                
                return {
                    ID: cds.utils.uuid(),
                    agent_ID: agent.ID,
                    totalTasks: totalTasks,
                    successfulTasks: Math.floor(totalTasks * successRate),
                    failedTasks: Math.floor(totalTasks * (1 - successRate)),
                    averageResponseTime: Math.floor(Math.random() * 300) + 100, // 100-400ms
                    reputationScore: Math.floor(Math.random() * 400) + 600, // 600-1000
                    trustScore: Math.round((0.7 + Math.random() * 0.3) * 100) / 100, // 0.70-1.00
                    lastUpdated: new Date().toISOString()
                };
            });
            
            // Batch insert all performance data
            if (performanceDataBatch.length > 0) {
                const values = performanceDataBatch.map(data =>
                    `('${data.ID}', '${data.agent_ID}', ${data.totalTasks}, ${data.successfulTasks}, ${data.failedTasks}, ${data.averageResponseTime}, ${data.reputationScore}, ${data.trustScore}, '${data.lastUpdated}')`
                ).join(', ');
                
                await db.run(`INSERT INTO a2a_network_AgentPerformance 
                    (ID, agent_ID, totalTasks, successfulTasks, failedTasks, 
                     averageResponseTime, reputationScore, trustScore, lastUpdated) 
                    VALUES ${values}`);
            }
            log.debug(`✅ Added performance data for ${agents.length} agents`);
        }
        
        // 2. Add essential blockchain stats if missing
        const blockchainCount = await db.run('SELECT COUNT(*) as count FROM BlockchainService_BlockchainStats');
        log.debug(`Current blockchain stats: ${blockchainCount[0].count}`);
        
        if (blockchainCount[0].count === 0) {
            log.debug('\n⛓️  Adding blockchain statistics...');
            await db.run(`INSERT INTO BlockchainService_BlockchainStats 
                (ID, blockHeight, gasPrice, networkStatus, totalTransactions, averageBlockTime, timestamp) 
                VALUES (?, ?, ?, ?, ?, ?, ?)`,
                [cds.utils.uuid(), 42150, 25.5, 'connected', 128450, 13.2, new Date().toISOString()]
            );
            log.debug('✅ Blockchain statistics added');
        }
        
        // 3. Test the tile endpoints to see what's missing
        log.debug('\n🔍 Testing tile endpoints...');
        
        const { SELECT } = cds.ql;
        
        try {
            // Test NetworkStats endpoint data
            const totalAgents = await db.run('SELECT COUNT(*) as count FROM a2a_network_Agents');
            const activeAgents = await db.run('SELECT COUNT(*) as count FROM a2a_network_Agents WHERE isActive = 1');
            log.debug(`✅ NetworkStats data: ${activeAgents[0].count} active agents of ${totalAgents[0].count} total`);
            
            // Test agent performance data for agent visualization
            const topPerformers = await db.run(`
                SELECT a.name, p.reputationScore, p.totalTasks, p.successfulTasks 
                FROM a2a_network_Agents a 
                LEFT JOIN a2a_network_AgentPerformance p ON a.ID = p.agent_ID 
                ORDER BY p.reputationScore DESC 
                LIMIT 3
            `);
            log.debug(`✅ Top performers data available: ${topPerformers.length} records`);
            
            // Check messages count
            const messageCount = await db.run('SELECT COUNT(*) as count FROM a2a_network_Messages');
            log.debug(`✅ Messages: ${messageCount[0].count}`);
            
        } catch (error) {
            log.error('❌ Error testing endpoints:', error.message);
        }
        
        // 4. Add some messages if none exist
        const msgCount = await db.run('SELECT COUNT(*) as count FROM a2a_network_Messages');
        if (msgCount[0].count < 10) {
            log.debug('\n💬 Adding sample messages...');
            const agents = await db.run('SELECT ID FROM a2a_network_Agents LIMIT 10');
            
            for (let i = 0; i < 20; i++) {
                const from = agents[Math.floor(Math.random() * agents.length)];
                const to = agents[Math.floor(Math.random() * agents.length)];
                
                if (from.ID !== to.ID) {
                    await db.run(`INSERT INTO a2a_network_Messages 
                        (ID, sender_ID, recipient_ID, messageHash, protocol, status, timestamp, createdAt, modifiedAt) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                        [cds.utils.uuid(), from.ID, to.ID,
                         `0x${Math.random().toString(16).substr(2, 64)}`,
                         'REST', 'delivered',
                         new Date(Date.now() - Math.random() * 86400000).toISOString(),
                         new Date().toISOString(), new Date().toISOString()]
                    );
                }
            }
            log.debug('✅ Added sample messages');
        }
        
        // 5. Update network stats
        const statsCount = await db.run('SELECT COUNT(*) as count FROM a2a_network_NetworkStats');
        if (statsCount[0].count === 0) {
            log.debug('\n📈 Adding network statistics...');
            const finalAgentCount = await db.run('SELECT COUNT(*) as count FROM a2a_network_Agents');
            const finalMsgCount = await db.run('SELECT COUNT(*) as count FROM a2a_network_Messages');
            const avgRep = await db.run('SELECT AVG(reputation) as avg FROM a2a_network_Agents');
            
            await db.run(`INSERT INTO a2a_network_NetworkStats 
                (ID, totalAgents, activeAgents, totalServices, totalCapabilities, totalMessages, 
                 totalTransactions, averageReputation, networkLoad, gasPrice, timestamp) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                [cds.utils.uuid(),
                 finalAgentCount[0].count,
                 finalAgentCount[0].count, // All agents active
                 0, // No services yet
                 0, // No capabilities due to FK constraints
                 finalMsgCount[0].count,
                 128450, // Blockchain transactions
                 Math.floor(avgRep[0].avg || 850),
                 Math.floor(Math.random() * 50) + 30, // Network load 30-80%
                 25.5, // Gas price
                 new Date().toISOString()]
            );
            log.debug('✅ Network statistics added');
        }
        
        log.info('\n🎉 Essential tile data seeding complete!');
        log.debug('\n📊 Final Summary:');
        
        const finalStats = {
            agents: await db.run('SELECT COUNT(*) as count FROM a2a_network_Agents'),
            performance: await db.run('SELECT COUNT(*) as count FROM a2a_network_AgentPerformance'),
            messages: await db.run('SELECT COUNT(*) as count FROM a2a_network_Messages'),
            blockchain: await db.run('SELECT COUNT(*) as count FROM BlockchainService_BlockchainStats'),
            networkStats: await db.run('SELECT COUNT(*) as count FROM a2a_network_NetworkStats')
        };
        
        log.debug(`   - Agents: ${finalStats.agents[0].count}`);
        log.debug(`   - Performance Records: ${finalStats.performance[0].count}`);
        log.debug(`   - Messages: ${finalStats.messages[0].count}`);
        log.debug(`   - Blockchain Stats: ${finalStats.blockchain[0].count}`);
        log.debug(`   - Network Stats: ${finalStats.networkStats[0].count}`);
        
        log.debug('\n🚀 Tiles should now display properly!');
        
        process.exit(0);
        
    } catch (error) {
        console.error('❌ Error seeding essential data:', error);
        process.exit(1);
    }
}

seedEssentialData();