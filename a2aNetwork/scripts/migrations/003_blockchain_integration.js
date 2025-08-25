/**
 * Migration 003: Blockchain Integration
 * Adds blockchain-related tables for smart contract integration
 */

async function up(db) {
    console.log('⛓️  Adding blockchain integration...');

    // Blockchain Transactions table
    await db.run(`
        CREATE TABLE IF NOT EXISTS BlockchainTransactions (
            ID NVARCHAR(36) PRIMARY KEY,
            transaction_hash NVARCHAR(66) UNIQUE,
            block_number BIGINT,
            from_address NVARCHAR(42),
            to_address NVARCHAR(42),
            contract_address NVARCHAR(42),
            transaction_type NVARCHAR(50),
            function_name NVARCHAR(100),
            function_args NCLOB,
            gas_used BIGINT,
            gas_price BIGINT,
            value_wei NVARCHAR(100),
            status NVARCHAR(20) DEFAULT 'pending',
            confirmation_count INTEGER DEFAULT 0,
            agent_ID NVARCHAR(36),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confirmed_at TIMESTAMP,
            FOREIGN KEY (agent_ID) REFERENCES Agents(ID)
        )
    `);

    // Smart Contract Events table
    await db.run(`
        CREATE TABLE IF NOT EXISTS SmartContractEvents (
            ID NVARCHAR(36) PRIMARY KEY,
            transaction_hash NVARCHAR(66),
            block_number BIGINT,
            log_index INTEGER,
            contract_address NVARCHAR(42),
            event_name NVARCHAR(100),
            event_args NCLOB,
            topics NCLOB,
            raw_log NCLOB,
            processed BOOLEAN DEFAULT FALSE,
            agent_ID NVARCHAR(36),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP,
            FOREIGN KEY (agent_ID) REFERENCES Agents(ID)
        )
    `);

    // Network Statistics table
    await db.run(`
        CREATE TABLE IF NOT EXISTS NetworkStatistics (
            ID NVARCHAR(36) PRIMARY KEY,
            metric_name NVARCHAR(100) NOT NULL,
            metric_value NVARCHAR(500),
            data_type NVARCHAR(20) DEFAULT 'string',
            category NVARCHAR(50),
            chain_id INTEGER,
            block_number BIGINT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata NCLOB
        )
    `);

    // Agent Registry Contract State table
    await db.run(`
        CREATE TABLE IF NOT EXISTS AgentRegistryState (
            ID NVARCHAR(36) PRIMARY KEY,
            agent_ID NVARCHAR(36) NOT NULL,
            contract_address NVARCHAR(42),
            blockchain_agent_id NVARCHAR(100),
            registered_block BIGINT,
            last_updated_block BIGINT,
            stake_amount NVARCHAR(100) DEFAULT '0',
            is_active_onchain BOOLEAN DEFAULT FALSE,
            metadata_hash NVARCHAR(66),
            last_sync TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (agent_ID) REFERENCES Agents(ID)
        )
    `);

    // Cross-Chain Operations table
    await db.run(`
        CREATE TABLE IF NOT EXISTS CrossChainOperations (
            ID NVARCHAR(36) PRIMARY KEY,
            operation_type NVARCHAR(50) NOT NULL,
            source_chain_id INTEGER,
            destination_chain_id INTEGER,
            source_tx_hash NVARCHAR(66),
            destination_tx_hash NVARCHAR(66),
            bridge_contract NVARCHAR(42),
            amount NVARCHAR(100),
            token_address NVARCHAR(42),
            status NVARCHAR(20) DEFAULT 'initiated',
            agent_ID NVARCHAR(36),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (agent_ID) REFERENCES Agents(ID)
        )
    `);

    // Create blockchain-related indexes
    await db.run('CREATE INDEX IF NOT EXISTS idx_blockchain_tx_hash ON BlockchainTransactions(transaction_hash)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_blockchain_tx_status ON BlockchainTransactions(status)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_blockchain_tx_agent ON BlockchainTransactions(agent_ID)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_blockchain_tx_block ON BlockchainTransactions(block_number)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_smart_contract_events_tx ON SmartContractEvents(transaction_hash)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_smart_contract_events_contract ON SmartContractEvents(contract_address)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_smart_contract_events_processed ON SmartContractEvents(processed)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_network_stats_metric ON NetworkStatistics(metric_name)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_network_stats_timestamp ON NetworkStatistics(timestamp)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_agent_registry_agent ON AgentRegistryState(agent_ID)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_agent_registry_active ON AgentRegistryState(is_active_onchain)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_cross_chain_status ON CrossChainOperations(status)');

    // Insert initial network statistics
    const initialStats = [
        { metric_name: 'total_agents', metric_value: '0', data_type: 'integer', category: 'agents' },
        { metric_name: 'active_agents', metric_value: '0', data_type: 'integer', category: 'agents' },
        { metric_name: 'total_transactions', metric_value: '0', data_type: 'integer', category: 'blockchain' },
        { metric_name: 'total_gas_used', metric_value: '0', data_type: 'bigint', category: 'blockchain' },
        { metric_name: 'network_health', metric_value: '100', data_type: 'decimal', category: 'health' },
        { metric_name: 'last_block_processed', metric_value: '0', data_type: 'bigint', category: 'blockchain' }
    ];

    for (const stat of initialStats) {
        await db.run(`
            INSERT OR IGNORE INTO NetworkStatistics
            (ID, metric_name, metric_value, data_type, category)
            VALUES (?, ?, ?, ?, ?)
        `, [
            require('crypto').randomUUID(),
            stat.metric_name,
            stat.metric_value,
            stat.data_type,
            stat.category
        ]);
    }

    console.log('✅ Blockchain integration added successfully');
}

async function down(db) {
    console.log('🔄 Rolling back blockchain integration...');

    const tables = [
        'CrossChainOperations',
        'AgentRegistryState',
        'NetworkStatistics',
        'SmartContractEvents',
        'BlockchainTransactions'
    ];

    for (const table of tables) {
        await db.run(`DROP TABLE IF EXISTS ${table}`);
    }

    console.log('✅ Blockchain integration rollback completed');
}

module.exports = { up, down };