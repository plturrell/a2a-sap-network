#!/usr/bin/env node

/**
 * Database Seeder for A2A Network
 * Populates database with initial development/test data
 */

const cds = require('@sap/cds');
const { v4: uuidv4 } = require('uuid');

class DatabaseSeeder {
    constructor() {
        this.db = null;
    }

    async initialize() {
        try {
            const env = process.env.NODE_ENV || 'development';
            
            if (env === 'production') {
                this.db = await cds.connect.to('db');
                log.debug('✅ Connected to SAP HANA Cloud');
            } else {
                this.db = await cds.connect.to('db', {
                    kind: 'sqlite',
                    credentials: { url: './data/a2a-network.db' }
                });
                log.debug('✅ Connected to SQLite database');
            }
        } catch (error) {
            console.error('❌ Database connection failed:', error);
            throw error;
        }
    }

    async seedAgents() {
        log.debug('👥 Seeding agents...');

        const agents = [
            {
                ID: uuidv4(),
                name: 'DataStandardizationAgent',
                description: 'Handles data standardization and cleansing operations',
                version: '1.2.0',
                status: 'active',
                endpoint: 'http://localhost:8001/api',
                blockchain_address: '0x742d35Cc6634C0532925a3b8D2aD4E0e1d36b8b3',
                public_key: 'MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA2K...',
                trust_level: 95,
                reputation_score: 92.5
            },
            {
                ID: uuidv4(),
                name: 'AIPreparationAgent',
                description: 'Prepares data for AI/ML processing and analysis',
                version: '1.1.0',
                status: 'active',
                endpoint: 'http://localhost:8002/api',
                blockchain_address: '0x8f2a55949038a9610f5d4c65d5b7b37d2a59e8c4',
                public_key: 'MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA3L...',
                trust_level: 88,
                reputation_score: 89.2
            },
            {
                ID: uuidv4(),
                name: 'VectorProcessingAgent',
                description: 'Handles vector operations and embeddings processing',
                version: '1.0.5',
                status: 'active',
                endpoint: 'http://localhost:8003/api',
                blockchain_address: '0x91b9f3ef4c7a3b9d8e2f1c5a6b8d9e3f4a5b6c7d',
                public_key: 'MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA4M...',
                trust_level: 91,
                reputation_score: 87.8
            },
            {
                ID: uuidv4(),
                name: 'CalculationValidationAgent',
                description: 'Validates calculations and mathematical operations',
                version: '1.0.0',
                status: 'active',
                endpoint: 'http://localhost:8004/api',
                blockchain_address: '0xa2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1',
                public_key: 'MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA5N...',
                trust_level: 93,
                reputation_score: 91.0
            },
            {
                ID: uuidv4(),
                name: 'QAValidationAgent',
                description: 'Performs quality assurance and validation checks',
                version: '1.0.0',
                status: 'active',
                endpoint: 'http://localhost:8005/api',
                blockchain_address: '0xb3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2',
                public_key: 'MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA6O...',
                trust_level: 87,
                reputation_score: 88.5
            }
        ];

        for (const agent of agents) {
            await this.db.run(`
                INSERT OR REPLACE INTO Agents 
                (ID, name, description, version, status, endpoint, blockchain_address, public_key, trust_level, reputation_score, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            `, [
                agent.ID, agent.name, agent.description, agent.version, agent.status,
                agent.endpoint, agent.blockchain_address, agent.public_key, agent.trust_level, agent.reputation_score
            ]);
            
            // Seed reputation scores for each agent
            await this.db.run(`
                INSERT OR REPLACE INTO ReputationScores
                (ID, agent_ID, overall_score, reliability_score, performance_score, security_score, total_interactions, successful_interactions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            `, [
                uuidv4(), agent.ID, agent.reputation_score, agent.reputation_score * 0.95, 
                agent.reputation_score * 1.02, agent.reputation_score * 0.98, 
                Math.floor(Math.random() * 500) + 100, Math.floor(Math.random() * 450) + 90
            ]);
        }

        log.debug(`✅ Seeded ${agents.length} agents`);
        return agents;
    }

    async seedServices(agents) {
        log.debug('🔧 Seeding services...');

        const serviceTemplates = [
            { type: 'standardization', description: 'Data standardization and cleansing' },
            { type: 'preparation', description: 'AI/ML data preparation' },
            { type: 'vectorization', description: 'Vector processing and embeddings' },
            { type: 'calculation', description: 'Mathematical calculations' },
            { type: 'validation', description: 'Quality assurance validation' }
        ];

        let servicesCount = 0;
        
        for (let i = 0; i < agents.length; i++) {
            const agent = agents[i];
            const serviceTemplate = serviceTemplates[i % serviceTemplates.length];
            
            const service = {
                ID: uuidv4(),
                agent_ID: agent.ID,
                name: `${serviceTemplate.type}_service`,
                description: serviceTemplate.description,
                service_type: serviceTemplate.type,
                endpoint: `${agent.endpoint}/${serviceTemplate.type}`,
                price: (Math.random() * 10 + 1).toFixed(2),
                currency: 'USD',
                is_active: true
            };

            await this.db.run(`
                INSERT OR REPLACE INTO Services
                (ID, agent_ID, name, description, service_type, endpoint, price, currency, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            `, [
                service.ID, service.agent_ID, service.name, service.description,
                service.service_type, service.endpoint, service.price, service.currency, service.is_active
            ]);

            servicesCount++;
        }

        log.debug(`✅ Seeded ${servicesCount} services`);
    }

    async seedCapabilities(agents) {
        log.debug('⚡ Seeding capabilities...');

        const capabilityTemplates = [
            { name: 'data_cleaning', description: 'Clean and standardize data formats' },
            { name: 'data_validation', description: 'Validate data integrity and quality' },
            { name: 'format_conversion', description: 'Convert between different data formats' },
            { name: 'vector_embedding', description: 'Generate vector embeddings' },
            { name: 'similarity_search', description: 'Perform similarity searches' },
            { name: 'mathematical_operations', description: 'Execute complex calculations' },
            { name: 'quality_scoring', description: 'Calculate quality metrics' }
        ];

        let capabilitiesCount = 0;

        for (const agent of agents) {
            // Each agent gets 2-4 random capabilities
            const numCapabilities = Math.floor(Math.random() * 3) + 2;
            const selectedCapabilities = capabilityTemplates
                .sort(() => Math.random() - 0.5)
                .slice(0, numCapabilities);

            for (const capability of selectedCapabilities) {
                await this.db.run(`
                    INSERT OR REPLACE INTO Capabilities
                    (ID, agent_ID, name, description, capability_type, is_enabled, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                `, [
                    uuidv4(), agent.ID, capability.name, capability.description, 'skill', true
                ]);

                capabilitiesCount++;
            }
        }

        log.debug(`✅ Seeded ${capabilitiesCount} capabilities`);
    }

    async seedNetworkStatistics() {
        log.debug('📊 Seeding network statistics...');

        const stats = [
            { metric_name: 'total_agents', metric_value: '5', data_type: 'integer', category: 'agents' },
            { metric_name: 'active_agents', metric_value: '5', data_type: 'integer', category: 'agents' },
            { metric_name: 'total_services', metric_value: '5', data_type: 'integer', category: 'services' },
            { metric_name: 'total_capabilities', metric_value: '15', data_type: 'integer', category: 'capabilities' },
            { metric_name: 'network_uptime', metric_value: '99.97', data_type: 'decimal', category: 'health' },
            { metric_name: 'average_response_time', metric_value: '125.3', data_type: 'decimal', category: 'performance' }
        ];

        for (const stat of stats) {
            await this.db.run(`
                INSERT OR REPLACE INTO NetworkStatistics
                (ID, metric_name, metric_value, data_type, category, timestamp)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            `, [uuidv4(), stat.metric_name, stat.metric_value, stat.data_type, stat.category]);
        }

        log.debug(`✅ Seeded ${stats.length} network statistics`);
    }

    async seedWorkflows() {
        log.debug('🔄 Seeding workflows...');

        const workflows = [
            {
                ID: uuidv4(),
                name: 'Data Processing Pipeline',
                description: 'Standard data processing workflow with validation',
                definition: JSON.stringify({
                    steps: [
                        { step: 1, agent: 'DataStandardizationAgent', action: 'standardize' },
                        { step: 2, agent: 'AIPreparationAgent', action: 'prepare' },
                        { step: 3, agent: 'QAValidationAgent', action: 'validate' }
                    ]
                }),
                status: 'active'
            },
            {
                ID: uuidv4(),
                name: 'Vector Analysis Workflow',
                description: 'Workflow for vector processing and analysis',
                definition: JSON.stringify({
                    steps: [
                        { step: 1, agent: 'VectorProcessingAgent', action: 'vectorize' },
                        { step: 2, agent: 'CalculationValidationAgent', action: 'validate_vectors' },
                        { step: 3, agent: 'QAValidationAgent', action: 'quality_check' }
                    ]
                }),
                status: 'active'
            }
        ];

        for (const workflow of workflows) {
            await this.db.run(`
                INSERT OR REPLACE INTO Workflows
                (ID, name, description, definition, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            `, [workflow.ID, workflow.name, workflow.description, workflow.definition, workflow.status]);
        }

        log.debug(`✅ Seeded ${workflows.length} workflows`);
    }

    async seed() {
        log.info('🌱 Starting database seeding...');

        try {
            const agents = await this.seedAgents();
            await this.seedServices(agents);
            await this.seedCapabilities(agents);
            await this.seedNetworkStatistics();
            await this.seedWorkflows();

            log.debug('🎉 Database seeding completed successfully');
        } catch (error) {
            console.error('❌ Seeding failed:', error);
            throw error;
        }
    }

    async disconnect() {
        if (this.db) {
            await this.db.disconnect();
            log.debug('🔌 Database disconnected');
        }
    }
}

// CLI Interface
async function main() {
    const seeder = new DatabaseSeeder();
    
    try {
        await seeder.initialize();
        await seeder.seed();
    } catch (error) {
        console.error('💥 Seeding failed:', error);
        process.exit(1);
    } finally {
        await seeder.disconnect();
    }
}

if (require.main === module) {
    main();
}

module.exports = DatabaseSeeder;