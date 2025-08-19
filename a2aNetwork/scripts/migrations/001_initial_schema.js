/**
 * Migration 001: Initial A2A Network Schema
 * Creates the core tables for agent management, services, and messaging
 */

async function up(db) {
    console.log('🏗️  Creating initial A2A Network schema...');

    // Core Agents table
    await db.run(`
        CREATE TABLE IF NOT EXISTS Agents (
            ID NVARCHAR(36) PRIMARY KEY,
            name NVARCHAR(255) NOT NULL,
            description NVARCHAR(1000),
            version NVARCHAR(50) DEFAULT '1.0.0',
            status NVARCHAR(20) DEFAULT 'active',
            endpoint NVARCHAR(500),
            blockchain_address NVARCHAR(42),
            public_key NVARCHAR(1000),
            trust_level INTEGER DEFAULT 100,
            last_heartbeat TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by NVARCHAR(255),
            updated_by NVARCHAR(255)
        )
    `);

    // Services table
    await db.run(`
        CREATE TABLE IF NOT EXISTS Services (
            ID NVARCHAR(36) PRIMARY KEY,
            agent_ID NVARCHAR(36),
            name NVARCHAR(255) NOT NULL,
            description NVARCHAR(1000),
            service_type NVARCHAR(50),
            endpoint NVARCHAR(500),
            price DECIMAL(15,2) DEFAULT 0.00,
            currency NVARCHAR(10) DEFAULT 'USD',
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (agent_ID) REFERENCES Agents(ID)
        )
    `);

    // Capabilities table
    await db.run(`
        CREATE TABLE IF NOT EXISTS Capabilities (
            ID NVARCHAR(36) PRIMARY KEY,
            agent_ID NVARCHAR(36),
            name NVARCHAR(255) NOT NULL,
            description NVARCHAR(1000),
            capability_type NVARCHAR(50),
            input_schema NCLOB,
            output_schema NCLOB,
            is_enabled BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (agent_ID) REFERENCES Agents(ID)
        )
    `);

    // Messages table for inter-agent communication
    await db.run(`
        CREATE TABLE IF NOT EXISTS Messages (
            ID NVARCHAR(36) PRIMARY KEY,
            from_agent_ID NVARCHAR(36),
            to_agent_ID NVARCHAR(36),
            message_type NVARCHAR(50),
            content NCLOB,
            status NVARCHAR(20) DEFAULT 'pending',
            priority INTEGER DEFAULT 5,
            expires_at TIMESTAMP,
            processed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (from_agent_ID) REFERENCES Agents(ID),
            FOREIGN KEY (to_agent_ID) REFERENCES Agents(ID)
        )
    `);

    // Workflows table
    await db.run(`
        CREATE TABLE IF NOT EXISTS Workflows (
            ID NVARCHAR(36) PRIMARY KEY,
            name NVARCHAR(255) NOT NULL,
            description NVARCHAR(1000),
            definition NCLOB,
            status NVARCHAR(20) DEFAULT 'draft',
            created_by NVARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    `);

    // Performance Metrics table
    await db.run(`
        CREATE TABLE IF NOT EXISTS PerformanceMetrics (
            ID NVARCHAR(36) PRIMARY KEY,
            agent_ID NVARCHAR(36),
            metric_name NVARCHAR(100),
            metric_value DECIMAL(15,6),
            unit NVARCHAR(20),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (agent_ID) REFERENCES Agents(ID)
        )
    `);

    // Create essential indexes
    await db.run('CREATE INDEX IF NOT EXISTS idx_agents_status ON Agents(status)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_agents_blockchain_address ON Agents(blockchain_address)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_services_agent_id ON Services(agent_ID)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_services_active ON Services(is_active)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_capabilities_agent_id ON Capabilities(agent_ID)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_messages_status ON Messages(status)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_messages_created_at ON Messages(created_at)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON PerformanceMetrics(timestamp)');

    console.log('✅ Initial schema created successfully');
}

async function down(db) {
    console.log('🔄 Rolling back initial schema...');

    const tables = [
        'PerformanceMetrics',
        'Workflows', 
        'Messages',
        'Capabilities',
        'Services',
        'Agents'
    ];

    for (const table of tables) {
        await db.run(`DROP TABLE IF EXISTS ${table}`);
    }

    console.log('✅ Initial schema rollback completed');
}

module.exports = { up, down };