/**
 * Safely populate services data handling foreign key constraints
 */

const cds = require('@sap/cds');
const { v4: uuidv4 } = require('uuid');

async function populateServices() {
    log.info('🔧 Starting safe services population...\n');

    try {
        // Connect to database
        const db = await cds.connect.to('db');

        // First, check if we have agents to reference
        log.debug('1. Checking existing agents...');
        const agentResult = await db.run('SELECT ID, name FROM a2a_network_Agents LIMIT 5');

        if (agentResult.length === 0) {
            log.debug('❌ No agents found. Running agent seeding first...');
            const { execSync } = require('child_process');
            execSync('node scripts/seedTestAgents.js', { stdio: 'inherit' });

            // Re-check agents
            const newAgentResult = await db.run('SELECT ID, name FROM a2a_network_Agents LIMIT 5');
            if (newAgentResult.length === 0) {
                throw new Error('Failed to seed agents');
            }
            log.debug(`✅ Found ${newAgentResult.length} agents after seeding`);
        } else {
            log.debug(`✅ Found ${agentResult.length} existing agents`);
        }

        // Get available agent IDs
        const availableAgents = await db.run('SELECT ID FROM a2a_network_Agents LIMIT 10');
        const agentIds = availableAgents.map(a => a.ID);

        log.debug('2. Setting up currencies...');
        // First ensure EUR currency exists
        try {
            await db.run('INSERT OR IGNORE INTO sap_common_Currencies (code, name) VALUES (\'EUR\', \'Euro\')');
            await db.run('INSERT OR IGNORE INTO sap_common_Currencies (code, name) VALUES (\'USD\', \'US Dollar\')');
        } catch (error) {
            console.warn('Currency setup warning:', error.message);
        }

        log.debug('3. Clearing existing services...');
        await db.run('DELETE FROM a2a_network_Services');

        log.debug('4. Creating service data...');
        const services = [
            {
                ID: uuidv4(),
                name: 'Data Analytics Service',
                description: 'Advanced data processing and analytics capabilities',
                category: 'Analytics',
                isActive: true,
                averageRating: 4.8,
                pricePerCall: 0.001,
                maxCallsPerDay: 1000,
                minReputation: 500,
                totalCalls: 250,
                escrowAmount: 0.1,
                provider_ID: agentIds[0] || null,
                createdAt: new Date().toISOString(),
                modifiedAt: new Date().toISOString()
            },
            {
                ID: uuidv4(),
                name: 'ML Training Service',
                description: 'Machine learning model training and optimization',
                category: 'AI/ML',
                isActive: true,
                averageRating: 4.6,
                pricePerCall: 0.005,
                maxCallsPerDay: 500,
                minReputation: 700,
                totalCalls: 180,
                escrowAmount: 0.25,
                provider_ID: agentIds[1] || null,
                createdAt: new Date().toISOString(),
                modifiedAt: new Date().toISOString()
            },
            {
                ID: uuidv4(),
                name: 'Blockchain Integration Service',
                description: 'Smart contract interaction and blockchain operations',
                category: 'Blockchain',
                isActive: true,
                averageRating: 4.5,
                pricePerCall: 0.002,
                maxCallsPerDay: 750,
                minReputation: 600,
                totalCalls: 320,
                escrowAmount: 0.15,
                provider_ID: agentIds[2] || null,
                createdAt: new Date().toISOString(),
                modifiedAt: new Date().toISOString()
            },
            {
                ID: uuidv4(),
                name: 'Document Processing Service',
                description: 'OCR, text extraction, and document analysis',
                category: 'Document Processing',
                isActive: true,
                averageRating: 4.3,
                pricePerCall: 0.003,
                maxCallsPerDay: 300,
                minReputation: 400,
                totalCalls: 140,
                escrowAmount: 0.2,
                provider_ID: agentIds[3] || null,
                createdAt: new Date().toISOString(),
                modifiedAt: new Date().toISOString()
            },
            {
                ID: uuidv4(),
                name: 'Real-time Messaging Service',
                description: 'WebSocket-based real-time communication',
                category: 'Communication',
                isActive: true,
                averageRating: 4.7,
                pricePerCall: 0.0005,
                maxCallsPerDay: 2000,
                minReputation: 300,
                totalCalls: 410,
                escrowAmount: 0.05,
                provider_ID: agentIds[4] || null,
                createdAt: new Date().toISOString(),
                modifiedAt: new Date().toISOString()
            },
            {
                ID: uuidv4(),
                name: 'Image Recognition Service',
                description: 'Computer vision and image classification',
                category: 'AI/ML',
                isActive: false,
                averageRating: 4.1,
                pricePerCall: 0.004,
                maxCallsPerDay: 250,
                minReputation: 800,
                totalCalls: 90,
                escrowAmount: 0.3,
                provider_ID: agentIds[0] || null,
                createdAt: new Date().toISOString(),
                modifiedAt: new Date().toISOString()
            },
            {
                ID: uuidv4(),
                name: 'API Gateway Service',
                description: 'Request routing and API management',
                category: 'Integration',
                isActive: true,
                averageRating: 4.4,
                pricePerCall: 0.0002,
                maxCallsPerDay: 5000,
                minReputation: 200,
                totalCalls: 2200,
                escrowAmount: 0.02,
                provider_ID: agentIds[1] || null,
                createdAt: new Date().toISOString(),
                modifiedAt: new Date().toISOString()
            },
            {
                ID: uuidv4(),
                name: 'Data Backup Service',
                description: 'Automated data backup and recovery',
                category: 'Storage',
                isActive: false,
                averageRating: 3.9,
                pricePerCall: 0.001,
                maxCallsPerDay: 100,
                minReputation: 500,
                totalCalls: 70,
                escrowAmount: 0.1,
                provider_ID: agentIds[2] || null,
                createdAt: new Date().toISOString(),
                modifiedAt: new Date().toISOString()
            }
        ];

        log.debug('5. Inserting services...');
        for (const service of services) {
            try {
                await db.run(`
                    INSERT INTO a2a_network_Services (
                        ID, name, description, category, isActive, averageRating,
                        pricePerCall, maxCallsPerDay, minReputation, totalCalls,
                        escrowAmount, provider_ID, createdAt, modifiedAt
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                `, [
                    service.ID, service.name, service.description, service.category,
                    service.isActive ? 1 : 0, service.averageRating, service.pricePerCall,
                    service.maxCallsPerDay, service.minReputation, service.totalCalls,
                    service.escrowAmount, service.provider_ID, service.createdAt, service.modifiedAt
                ]);
                log.debug(`✅ Inserted: ${service.name}`);
            } catch (error) {
                console.error(`❌ Failed to insert ${service.name}:`, error.message);
                // Try without provider_ID if foreign key constraint fails
                if (error.message.includes('FOREIGN KEY constraint')) {
                    try {
                        await db.run(`
                            INSERT INTO a2a_network_Services (
                                ID, name, description, category, isActive, averageRating,
                                pricePerCall, maxCallsPerDay, minReputation, totalCalls,
                                escrowAmount, createdAt, modifiedAt
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        `, [
                            service.ID, service.name, service.description, service.category,
                            service.isActive ? 1 : 0, service.averageRating, service.pricePerCall,
                            service.maxCallsPerDay, service.minReputation, service.totalCalls,
                            service.escrowAmount, service.createdAt, service.modifiedAt
                        ]);
                        log.debug(`✅ Inserted ${service.name} (without provider reference)`);
                    } catch (retryError) {
                        console.error(`❌ Retry failed for ${service.name}:`, retryError.message);
                    }
                }
            }
        }

        log.debug('6. Verifying services...');
        const serviceCount = await db.run('SELECT COUNT(*) as count FROM a2a_network_Services');
        const activeServiceCount = await db.run('SELECT COUNT(*) as count FROM a2a_network_Services WHERE isActive = 1');

        log.debug(`✅ Total services: ${serviceCount[0]?.count || 0}`);
        log.debug(`✅ Active services: ${activeServiceCount[0]?.count || 0}`);

        log.debug('7. Creating sample service usage records...');
        const usageRecords = [];
        const serviceIds = services.map(s => s.ID);

        for (let i = 0; i < 15; i++) {
            usageRecords.push({
                ID: uuidv4(),
                service_ID: serviceIds[i % serviceIds.length],
                agent_ID: agentIds[i % agentIds.length],
                callCount: Math.floor(Math.random() * 100) + 1,
                totalCost: Math.random() * 0.5,
                avgResponseTime: Math.floor(Math.random() * 500) + 50,
                successRate: Math.random() * 0.3 + 0.7, // 70-100%
                lastUsed: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString()
            });
        }

        for (const usage of usageRecords) {
            try {
                await db.run(`
                    INSERT INTO a2a_network_ServiceUsage (
                        ID, service_ID, agent_ID, callCount, totalCost, avgResponseTime, successRate, lastUsed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                `, [
                    usage.ID, usage.service_ID, usage.agent_ID, usage.callCount,
                    usage.totalCost, usage.avgResponseTime, usage.successRate, usage.lastUsed
                ]);
            } catch (error) {
                // Skip if table doesn't exist or constraint fails
                if (!error.message.includes('no such table') && !error.message.includes('FOREIGN KEY')) {
                    console.warn(`Warning creating usage record: ${error.message}`);
                }
            }
        }

        log.debug('\n🎉 Services population completed successfully!');
        return true;

    } catch (error) {
        console.error('❌ Error populating services:', error);
        throw error;
    }
}

// Run if called directly
if (require.main === module) {
    populateServices()
        .then(() => {
            log.debug('\n✅ Services population completed');
            process.exit(0);
        })
        .catch(error => {
            console.error('❌ Services population failed:', error);
            process.exit(1);
        });
}

module.exports = { populateServices };