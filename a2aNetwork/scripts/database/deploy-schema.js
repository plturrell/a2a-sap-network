#!/usr/bin/env node

/**
 * Direct SQL Database Schema Deployment for A2A Network
 * Bypasses CDS compilation issues and deploys schema directly
 */

const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const fs = require('fs');

class DirectDatabaseDeployer {
    constructor() {
        this.db = null;
        this.dbPath = './data/a2a-network.db';
    }

    async initialize() {
        console.log('🔌 Initializing SQLite database...');
        
        // Ensure data directory exists
        const dataDir = path.dirname(this.dbPath);
        if (!fs.existsSync(dataDir)) {
            fs.mkdirSync(dataDir, { recursive: true });
            console.log(`📁 Created data directory: ${dataDir}`);
        }

        return new Promise((resolve, reject) => {
            this.db = new sqlite3.Database(this.dbPath, (err) => {
                if (err) {
                    console.error('❌ Database connection failed:', err);
                    reject(err);
                } else {
                    console.log('✅ Connected to SQLite database');
                    resolve();
                }
            });
        });
    }

    async executeSQL(sql, description) {
        return new Promise((resolve, reject) => {
            this.db.run(sql, (err) => {
                if (err) {
                    console.error(`❌ ${description} failed:`, err.message);
                    reject(err);
                } else {
                    console.log(`✅ ${description} completed`);
                    resolve();
                }
            });
        });
    }

    async createCoreSchema() {
        console.log('🏗️  Creating core A2A Network schema...');

        // Core Agents table
        await this.executeSQL(`
            CREATE TABLE IF NOT EXISTS Agents (
                ID TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                version TEXT DEFAULT '1.0.0',
                status TEXT DEFAULT 'active',
                endpoint TEXT,
                blockchain_address TEXT,
                public_key TEXT,
                trust_level INTEGER DEFAULT 100,
                reputation_score DECIMAL(5,2) DEFAULT 100.00,
                last_heartbeat DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                updated_by TEXT
            )
        `, 'Create Agents table');

        // Services table
        await this.executeSQL(`
            CREATE TABLE IF NOT EXISTS Services (
                ID TEXT PRIMARY KEY,
                agent_ID TEXT,
                name TEXT NOT NULL,
                description TEXT,
                service_type TEXT,
                endpoint TEXT,
                price DECIMAL(15,2) DEFAULT 0.00,
                currency TEXT DEFAULT 'USD',
                is_active BOOLEAN DEFAULT TRUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_ID) REFERENCES Agents(ID)
            )
        `, 'Create Services table');

        // Capabilities table
        await this.executeSQL(`
            CREATE TABLE IF NOT EXISTS Capabilities (
                ID TEXT PRIMARY KEY,
                agent_ID TEXT,
                name TEXT NOT NULL,
                description TEXT,
                capability_type TEXT,
                input_schema TEXT,
                output_schema TEXT,
                is_enabled BOOLEAN DEFAULT TRUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_ID) REFERENCES Agents(ID)
            )
        `, 'Create Capabilities table');

        // Messages table
        await this.executeSQL(`
            CREATE TABLE IF NOT EXISTS Messages (
                ID TEXT PRIMARY KEY,
                from_agent_ID TEXT,
                to_agent_ID TEXT,
                message_type TEXT,
                content TEXT,
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 5,
                expires_at DATETIME,
                processed_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (from_agent_ID) REFERENCES Agents(ID),
                FOREIGN KEY (to_agent_ID) REFERENCES Agents(ID)
            )
        `, 'Create Messages table');

        // Workflows table
        await this.executeSQL(`
            CREATE TABLE IF NOT EXISTS Workflows (
                ID TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                definition TEXT,
                status TEXT DEFAULT 'draft',
                created_by TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        `, 'Create Workflows table');
    }

    async createReputationSchema() {
        console.log('🏆 Creating reputation system schema...');

        await this.executeSQL(`
            CREATE TABLE IF NOT EXISTS ReputationScores (
                ID TEXT PRIMARY KEY,
                agent_ID TEXT NOT NULL,
                overall_score DECIMAL(5,2) DEFAULT 100.00,
                reliability_score DECIMAL(5,2) DEFAULT 100.00,
                performance_score DECIMAL(5,2) DEFAULT 100.00,
                security_score DECIMAL(5,2) DEFAULT 100.00,
                total_interactions INTEGER DEFAULT 0,
                successful_interactions INTEGER DEFAULT 0,
                failed_interactions INTEGER DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_ID) REFERENCES Agents(ID)
            )
        `, 'Create ReputationScores table');

        await this.executeSQL(`
            CREATE TABLE IF NOT EXISTS ReputationEvents (
                ID TEXT PRIMARY KEY,
                agent_ID TEXT NOT NULL,
                event_type TEXT NOT NULL,
                impact_score DECIMAL(5,2),
                description TEXT,
                metadata TEXT,
                reported_by TEXT,
                verified BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_ID) REFERENCES Agents(ID),
                FOREIGN KEY (reported_by) REFERENCES Agents(ID)
            )
        `, 'Create ReputationEvents table');

        await this.executeSQL(`
            CREATE TABLE IF NOT EXISTS TrustRelationships (
                ID TEXT PRIMARY KEY,
                truster_agent_ID TEXT NOT NULL,
                trustee_agent_ID TEXT NOT NULL,
                trust_level DECIMAL(5,2) DEFAULT 50.00,
                relationship_type TEXT DEFAULT 'peer',
                established_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_interaction DATETIME,
                interaction_count INTEGER DEFAULT 0,
                FOREIGN KEY (truster_agent_ID) REFERENCES Agents(ID),
                FOREIGN KEY (trustee_agent_ID) REFERENCES Agents(ID),
                UNIQUE(truster_agent_ID, trustee_agent_ID)
            )
        `, 'Create TrustRelationships table');
    }

    async createBlockchainSchema() {
        console.log('⛓️  Creating blockchain integration schema...');

        await this.executeSQL(`
            CREATE TABLE IF NOT EXISTS BlockchainTransactions (
                ID TEXT PRIMARY KEY,
                transaction_hash TEXT UNIQUE,
                block_number INTEGER,
                from_address TEXT,
                to_address TEXT,
                contract_address TEXT,
                transaction_type TEXT,
                function_name TEXT,
                function_args TEXT,
                gas_used INTEGER,
                gas_price INTEGER,
                value_wei TEXT,
                status TEXT DEFAULT 'pending',
                confirmation_count INTEGER DEFAULT 0,
                agent_ID TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                confirmed_at DATETIME,
                FOREIGN KEY (agent_ID) REFERENCES Agents(ID)
            )
        `, 'Create BlockchainTransactions table');

        await this.executeSQL(`
            CREATE TABLE IF NOT EXISTS NetworkStatistics (
                ID TEXT PRIMARY KEY,
                metric_name TEXT NOT NULL,
                metric_value TEXT,
                data_type TEXT DEFAULT 'string',
                category TEXT,
                chain_id INTEGER,
                block_number INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        `, 'Create NetworkStatistics table');
    }

    async createIndexes() {
        console.log('📊 Creating database indexes...');

        const indexes = [
            'CREATE INDEX IF NOT EXISTS idx_agents_status ON Agents(status)',
            'CREATE INDEX IF NOT EXISTS idx_agents_blockchain ON Agents(blockchain_address)',
            'CREATE INDEX IF NOT EXISTS idx_services_agent ON Services(agent_ID)',
            'CREATE INDEX IF NOT EXISTS idx_services_active ON Services(is_active)',
            'CREATE INDEX IF NOT EXISTS idx_capabilities_agent ON Capabilities(agent_ID)',
            'CREATE INDEX IF NOT EXISTS idx_messages_status ON Messages(status)',
            'CREATE INDEX IF NOT EXISTS idx_messages_created ON Messages(created_at)',
            'CREATE INDEX IF NOT EXISTS idx_reputation_agent ON ReputationScores(agent_ID)',
            'CREATE INDEX IF NOT EXISTS idx_reputation_score ON ReputationScores(overall_score)',
            'CREATE INDEX IF NOT EXISTS idx_blockchain_tx_hash ON BlockchainTransactions(transaction_hash)',
            'CREATE INDEX IF NOT EXISTS idx_blockchain_tx_status ON BlockchainTransactions(status)',
            'CREATE INDEX IF NOT EXISTS idx_network_stats_metric ON NetworkStatistics(metric_name)'
        ];

        for (const indexSQL of indexes) {
            await this.executeSQL(indexSQL, `Create index`);
        }
    }

    async createMigrationTable() {
        console.log('📋 Creating migration tracking table...');

        await this.executeSQL(`
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                checksum TEXT,
                execution_time INTEGER
            )
        `, 'Create schema_migrations table');

        // Record this initial schema creation
        const version = '001_initial_direct_schema';
        await this.executeSQL(`
            INSERT OR REPLACE INTO schema_migrations (version, execution_time) 
            VALUES ('${version}', 0)
        `, 'Record migration');
    }

    async seedInitialData() {
        console.log('🌱 Seeding initial data...');

        // Generate UUID function for SQLite
        const generateId = () => {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                const r = Math.random() * 16 | 0;
                const v = c == 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        };

        // Initial network statistics
        const stats = [
            { metric_name: 'total_agents', metric_value: '0', data_type: 'integer', category: 'agents' },
            { metric_name: 'active_agents', metric_value: '0', data_type: 'integer', category: 'agents' },
            { metric_name: 'total_services', metric_value: '0', data_type: 'integer', category: 'services' },
            { metric_name: 'network_health', metric_value: '100.0', data_type: 'decimal', category: 'health' },
            { metric_name: 'last_block_processed', metric_value: '0', data_type: 'bigint', category: 'blockchain' }
        ];

        for (const stat of stats) {
            const id = generateId();
            await this.executeSQL(`
                INSERT OR REPLACE INTO NetworkStatistics 
                (ID, metric_name, metric_value, data_type, category) 
                VALUES ('${id}', '${stat.metric_name}', '${stat.metric_value}', '${stat.data_type}', '${stat.category}')
            `, `Seed ${stat.metric_name} statistic`);
        }
    }

    async deploySchema() {
        console.log('🚀 Starting direct schema deployment...');

        try {
            await this.createCoreSchema();
            await this.createReputationSchema();
            await this.createBlockchainSchema();
            await this.createIndexes();
            await this.createMigrationTable();
            await this.seedInitialData();

            console.log('🎉 Database schema deployed successfully!');
            return true;
        } catch (error) {
            console.error('💥 Schema deployment failed:', error);
            throw error;
        }
    }

    async getStatus() {
        console.log('📊 Checking database status...');

        return new Promise((resolve, reject) => {
            const tables = [];
            
            this.db.all(`
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            `, (err, rows) => {
                if (err) {
                    reject(err);
                } else {
                    console.log('\n📋 Database Tables:');
                    rows.forEach(row => {
                        console.log(`  ✅ ${row.name}`);
                        tables.push(row.name);
                    });
                    
                    console.log(`\n📈 Total tables: ${tables.length}`);
                    resolve(tables);
                }
            });
        });
    }

    async disconnect() {
        if (this.db) {
            return new Promise((resolve) => {
                this.db.close((err) => {
                    if (err) {
                        console.error('⚠️  Database disconnect error:', err);
                    } else {
                        console.log('🔌 Database disconnected');
                    }
                    resolve();
                });
            });
        }
    }
}

// CLI Interface
async function main() {
    const deployer = new DirectDatabaseDeployer();
    
    try {
        await deployer.initialize();
        
        const command = process.argv[2] || 'deploy';
        
        switch (command) {
            case 'deploy':
                await deployer.deploySchema();
                await deployer.getStatus();
                break;
                
            case 'status':
                await deployer.getStatus();
                break;
                
            default:
                console.log(`
A2A Network Direct Database Deployer

Usage:
  node deploy-schema.js deploy   - Deploy complete schema
  node deploy-schema.js status   - Show database status

Examples:
  node deploy-schema.js deploy
  node deploy-schema.js status
                `);
                break;
        }
        
    } catch (error) {
        console.error('💥 Deployment failed:', error);
        process.exit(1);
    } finally {
        await deployer.disconnect();
    }
}

if (require.main === module) {
    main();
}

module.exports = DirectDatabaseDeployer;