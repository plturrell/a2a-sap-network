#!/usr/bin/env node

/**
 * Initialize Blockchain Statistics with correct table structure
 */

const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const { v4: uuidv4 } = require('uuid');

const dbPath = path.join(__dirname, '../data/a2a_development.db');

console.log('Initializing blockchain statistics...');

const db = new sqlite3.Database(dbPath, (err) => {
    if (err) {
        console.error('Failed to connect to database:', err.message);
        process.exit(1);
    }
    console.log('Connected to database');
});

// Initialize blockchain stats data
const blockchainStats = {
    ID: uuidv4(),
    blockHeight: 18500000,
    gasPrice: 25.5,
    networkStatus: 'operational',
    totalTransactions: 2145678932,
    averageBlockTime: 12.1,
    timestamp: new Date().toISOString()
};

// Insert blockchain stats
db.run(`
    INSERT OR REPLACE INTO BlockchainService_BlockchainStats 
    (ID, blockHeight, gasPrice, networkStatus, totalTransactions, averageBlockTime, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?)
`, [
    blockchainStats.ID,
    blockchainStats.blockHeight,
    blockchainStats.gasPrice,
    blockchainStats.networkStatus,
    blockchainStats.totalTransactions,
    blockchainStats.averageBlockTime,
    blockchainStats.timestamp
], function(err) {
    if (err) {
        console.error('Error inserting blockchain stats:', err);
    } else {
        console.log('✅ Blockchain stats initialized successfully');
        console.log(`   - Block Height: ${blockchainStats.blockHeight}`);
        console.log(`   - Gas Price: ${blockchainStats.gasPrice} Gwei`);
        console.log(`   - Network Status: ${blockchainStats.networkStatus}`);
        console.log(`   - Total Transactions: ${blockchainStats.totalTransactions}`);
        console.log(`   - Average Block Time: ${blockchainStats.averageBlockTime}s`);
        
        // Verify the data was inserted
        db.get('SELECT * FROM BlockchainService_BlockchainStats ORDER BY timestamp DESC LIMIT 1', (err, row) => {
            if (err) {
                console.error('Error verifying data:', err);
            } else {
                console.log('\n✅ Verification: Data successfully stored in database');
                console.log('   Current blockchain stats:', row);
            }
            db.close();
        });
    }
});