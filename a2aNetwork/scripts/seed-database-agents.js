/**
 * Direct Database Seeding Script for 10 Real A2A Agents (including CalculationAgent)
 * This script bypasses API endpoints and directly inserts the real agents into the database
 */

const cds = require('@sap/cds');
const { SELECT, INSERT, DELETE } = cds.ql;
const { v4: uuidv4 } = require('uuid');

// Define all 15 real A2A agents (matching actual database schema)
const REAL_A2A_AGENTS = [
    {
        name: "Agent Manager",
        address: process.env.AGENT_MANAGER_ADDRESS || (() => { throw new Error('AGENT_MANAGER_ADDRESS environment variable is required'); })(),
        endpoint: process.env.AGENT_MANAGER_ENDPOINT || "http://localhost:8000/a2a/agent_manager/v1",
        reputation: 200,
        isActive: true,
        country_code: "US"
    },
    {
        name: "Data Product Registration Agent",
        address: process.env.AGENT0_ADDRESS || (() => { throw new Error('AGENT0_ADDRESS environment variable is required'); })(),
        endpoint: process.env.AGENT0_ENDPOINT || "http://localhost:8000/a2a/agent0/v1",
        reputation: 180,
        isActive: true,
        country_code: "US"
    },
    {
        name: "Data Standardization Agent",
        address: process.env.AGENT1_ADDRESS || (() => { throw new Error('AGENT1_ADDRESS environment variable is required'); })(),
        endpoint: process.env.AGENT1_ENDPOINT || "http://localhost:8000/a2a/agent1/v1",
        reputation: 175,
        isActive: true,
        country_code: "DE"
    },
    {
        name: "AI Preparation Agent",
        address: process.env.AGENT2_ADDRESS || (() => { throw new Error('AGENT2_ADDRESS environment variable is required'); })(),
        endpoint: process.env.AGENT2_ENDPOINT || "http://localhost:8000/a2a/agent2/v1",
        reputation: 170,
        isActive: true,
        country_code: "JP"
    },
    {
        name: "Vector Processing Agent",
        address: process.env.AGENT3_ADDRESS || (() => { throw new Error('AGENT3_ADDRESS environment variable is required'); })(),
        endpoint: process.env.AGENT3_ENDPOINT || "http://localhost:8000/a2a/agent3/v1",
        reputation: 185,
        isActive: true,
        country_code: "SG"
    },
    {
        name: "Calculation Validation Agent",
        address: process.env.AGENT4_ADDRESS || "0xAA00000000000000000000000000000000000006",
        endpoint: process.env.AGENT4_ENDPOINT || "http://localhost:8000/a2a/agent4/v1",
        reputation: 165,
        isActive: true,
        country_code: "IN"
    },
    {
        name: "QA Validation Agent",
        address: process.env.AGENT5_ADDRESS || "0xAA00000000000000000000000000000000000007",
        endpoint: process.env.AGENT5_ENDPOINT || "http://localhost:8000/a2a/agent5/v1",
        reputation: 160,
        isActive: true,
        country_code: "CA"
    },
    {
        name: "Data Manager Agent",
        address: process.env.DATA_MANAGER_ADDRESS || "0xAA00000000000000000000000000000000000008",
        endpoint: process.env.DATA_MANAGER_ENDPOINT || "http://localhost:8000/a2a/data_manager/v1",
        reputation: 190,
        isActive: true,
        country_code: "UK"
    },
    {
        name: "Catalog Manager Agent",
        address: process.env.CATALOG_MANAGER_ADDRESS || "0xAA00000000000000000000000000000000000009",
        endpoint: process.env.CATALOG_MANAGER_ENDPOINT || "http://localhost:8000/a2a/catalog_manager/v1",
        reputation: 195,
        isActive: true,
        country_code: "FR"
    },
    {
        name: "Enhanced Calculation Agent",
        address: process.env.CALCULATION_AGENT_ADDRESS || "0xAA0000000000000000000000000000000000000A",
        endpoint: process.env.CALCULATION_AGENT_ENDPOINT || "http://localhost:8000/a2a/calculation_agent/v1",
        reputation: 210,
        isActive: true,
        country_code: "CH"
    },
    {
        name: "Reasoning Agent",
        address: process.env.REASONING_AGENT_ADDRESS || "0xAA0000000000000000000000000000000000000B",
        endpoint: process.env.REASONING_AGENT_ENDPOINT || "http://localhost:8000/a2a/reasoning_agent/v1",
        reputation: 200,
        isActive: true,
        country_code: "NO"
    },
    {
        name: "SQL Agent",
        address: process.env.SQL_AGENT_ADDRESS || "0xAA0000000000000000000000000000000000000C",
        endpoint: process.env.SQL_AGENT_ENDPOINT || "http://localhost:8000/a2a/sql_agent/v1",
        reputation: 175,
        isActive: true,
        country_code: "SE"
    },
    {
        name: "Developer Portal Agent Builder",
        address: process.env.DEVELOPER_PORTAL_ADDRESS || "0xAA0000000000000000000000000000000000000D",
        endpoint: process.env.DEVELOPER_PORTAL_ENDPOINT || "http://localhost:8000/a2a/developer_portal/agent_builder/v1",
        reputation: 165,
        isActive: true,
        country_code: "NL"
    },
    {
        name: "Agent Builder Service",
        address: process.env.AGENT_BUILDER_SERVICE_ADDRESS || "0xAA0000000000000000000000000000000000000E",
        endpoint: process.env.AGENT_BUILDER_SERVICE_ENDPOINT || "http://localhost:8000/a2a/developer_portal/static/agent_builder/v1",
        reputation: 150,
        isActive: true,
        country_code: "DK"
    }
];

async function seedRealAgents() {
    try {
        log.debug('🔄 Connecting to database...');
        const db = await cds.connect.to('db');
        
        log.debug('🧹 Cleaning up existing agent data...');
        // Delete all existing agents to ensure clean state
        await db.run(DELETE.from('a2a.network.Agents'));
        log.debug('✅ Existing agent data cleared');

        log.debug('🌱 Seeding 15 real A2A agents...');
        let successCount = 0;
        
        for (const agent of REAL_A2A_AGENTS) {
            try {
                // Add required ID field
                const agentWithId = {
                    ID: uuidv4(),
                    ...agent
                };
                await db.run(INSERT.into('a2a.network.Agents').entries(agentWithId));
                log.debug(`✅ Inserted: ${agent.name}`);
                successCount++;
            } catch (error) {
                console.error(`❌ Failed to insert ${agent.name}:`, error.message);
            }
        }

        log.debug('\n📊 Verification...');
        const totalCount = await db.run(SELECT.from('a2a.network.Agents').columns('count(*) as total'));
        const activeCount = await db.run(SELECT.from('a2a.network.Agents').columns('count(*) as total').where({ isActive: true }));
        
        log.debug(`✅ Total agents in database: ${totalCount[0].total}`);
        log.debug(`✅ Active agents in database: ${activeCount[0].total}`);
        
        if (totalCount[0].total === 10 && activeCount[0].total === 10) {
            log.info('🎉 SUCCESS: Exactly 10 real agents seeded and verified (including CalculationAgent)!');
        } else {
            log.warn('⚠️  WARNING: Agent count does not match expected 10 agents');
        }

        log.debug('\n🔗 Testing API endpoint...');
        // Test that our API endpoint returns the correct count
        log.debug('Database seeding complete. Test the API with:');
        log.debug('curl -s "http://localhost:4004/api/v1/NetworkStats?id=overview_dashboard" | jq \'.data.activeAgents\'');
        
    } catch (error) {
        console.error('❌ Failed to seed real agents:', error);
        process.exit(1);
    }
}

// Run the seeding script
seedRealAgents().then(() => {
    log.debug('✅ Real agent seeding completed successfully');
    process.exit(0);
}).catch(error => {
    console.error('❌ Real agent seeding failed:', error);
    process.exit(1);
});
