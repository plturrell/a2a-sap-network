const cds = require('@sap/cds');
const { SELECT, INSERT, UPDATE, DELETE } = cds.ql;
/**
 * @fileoverview SAP Data Initialization Service
 * @since 1.0.0
 * @module sapDataInit
 *
 * Data initialization service for A2A Network using CAP framework
 * to properly initialize master data, reference data, and sample records
 * for development and testing environments
 */

module.exports = async (srv) => {
    const { Agents, Services, Capabilities, Workflows } = srv.entities;

    // Only initialize in development
    if (process.env.NODE_ENV !== 'production') {
        srv.on('listening', async () => {
            const db = await cds.connect.to('db');

            try {
                // Check if data already exists
                const agentCount = await SELECT.one`count(*) as count`.from(Agents);
                if (agentCount?.count > 0) {
                    cds.log('service').info('Sample data already exists, skipping initialization');
                    return;
                }

                // Only initialize sample data in development environment
                if (process.env.NODE_ENV === 'production') {
                    cds.log('service').info('Skipping sample data initialization in production');
                    return;
                }

                cds.log('service').info('Initializing sample data for development...');

                // Create sample agents
                const agents = await INSERT.into(Agents).entries([
                    {
                        name: 'DataProcessor Alpha',
                        endpoint: 'https://agent1.a2a.network',
                        reputation: 150,
                        isActive: true,
                        country_code: 'US',
                        address: '0x1111111111111111111111111111111111111111'
                    },
                    {
                        name: 'Analytics Bot Beta',
                        endpoint: 'https://agent2.a2a.network',
                        reputation: 180,
                        isActive: true,
                        country_code: 'DE',
                        address: '0x2222222222222222222222222222222222222222'
                    },
                    {
                        name: 'Security Guardian',
                        endpoint: 'https://agent3.a2a.network',
                        reputation: 195,
                        isActive: true,
                        country_code: 'JP',
                        address: '0x3333333333333333333333333333333333333333'
                    }
                ]);

                cds.log('service').info(`Created ${agents.length} sample agents`);

                // Create sample capabilities
                await INSERT.into(Capabilities).entries([
                    {
                        name: 'data-processing',
                        description: 'General data processing and transformation',
                        category: 'COMPUTE'
                    },
                    {
                        name: 'ml-inference',
                        description: 'Machine learning model inference',
                        category: 'AI_ML'
                    },
                    {
                        name: 'security-audit',
                        description: 'Security vulnerability scanning',
                        category: 'SECURITY'
                    }
                ]);

                cds.log('service').info('Created sample capabilities');

                // Create sample workflows
                await INSERT.into(Workflows).entries([
                    {
                        name: 'ETL Pipeline Workflow',
                        description: 'Extract, transform, and load data workflow',
                        definition: JSON.stringify({
                            steps: [
                                { name: 'extract', action: 'extract-data' },
                                { name: 'transform', action: 'transform-data' },
                                { name: 'load', action: 'load-data' }
                            ]
                        }),
                        isActive: true
                    }
                ]);

                cds.log('service').info('Created sample workflows');
                cds.log('service').info('✅ Sample data initialization completed!');

            } catch (error) {
                cds.log('service').error('Error initializing data:', error);
            }
        });
    }
};
