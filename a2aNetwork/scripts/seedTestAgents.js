/**
 * Simple script to seed test agents into the database
 */

const cds = require('@sap/cds');
const { SELECT, INSERT, DELETE } = cds.ql;

async function seedTestAgents() {
    try {
        log.debug('🚀 Seeding test agents...');

        // Load model and connect to database
        await cds.load('*');
        const db = await cds.connect.to('db');

        // Use table name directly for SQLite
        const agentTableName = 'a2a_network_Agents';

        // Create 15 test agents
        const agents = [];
        for (let i = 1; i <= 15; i++) {
            agents.push({
                ID: cds.utils.uuid(),
                address: `0x${i.toString().padStart(40, '0')}`,
                name: `Agent ${i}`,
                endpoint: `http://localhost:800${i}`,
                reputation: Math.floor(Math.random() * 1000) + 500, // 500-1500
                isActive: 1, // SQLite boolean as integer
                createdAt: new Date().toISOString(),
                modifiedAt: new Date().toISOString()
            });
        }

        // Clear existing agents first - use raw SQL for SQLite
        await db.run(`DELETE FROM ${agentTableName}`);
        log.debug('🧹 Cleared existing agents');

        // Insert new agents using raw SQL
        const insertSQL = `INSERT INTO ${agentTableName} (ID, address, name, endpoint, reputation, isActive, createdAt, modifiedAt) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`;

        for (const agent of agents) {
            await db.run(insertSQL, [
                agent.ID,
                agent.address,
                agent.name,
                agent.endpoint,
                agent.reputation,
                agent.isActive,
                agent.createdAt,
                agent.modifiedAt
            ]);
        }

        log.info(`✅ Successfully seeded ${agents.length} test agents`);

        // Verify the count
        const countResult = await db.run(`SELECT COUNT(*) as total FROM ${agentTableName}`);
        log.debug(`📊 Total agents in database: ${countResult[0]?.total || 0}`);

        process.exit(0);

    } catch (error) {
        console.error('❌ Error seeding test agents:', error);
        process.exit(1);
    }
}

// Run the seeding
seedTestAgents();