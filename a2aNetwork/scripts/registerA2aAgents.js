/**
 * Register A2A Agents from a2a_agents project into A2A Network
 * This script creates agent records in the SAP CAP A2A Network system
 */

const axios = require('axios');

const A2A_NETWORK_API = process.env.A2A_NETWORK_API_URL || 'http://localhost:4004/api/v1';

// Define all 15 real agents from the a2a_agents project
const A2A_AGENTS = [
    {
        name: "Agent Manager",
        endpoint: process.env.AGENT_MANAGER_ENDPOINT || "http://localhost:8000/a2a/agent_manager/v1",
        reputation: 200,
        isActive: true,
        country_code: "US",
        address: process.env.AGENT_MANAGER_ADDRESS || "0xAA00000000000000000000000000000000000001",
        description: "Manages A2A ecosystem registration, trust contracts, and workflow orchestration",
        capabilities: ["agent-registration", "trust-management", "workflow-orchestration", "ecosystem-management"]
    },
    {
        name: "Data Product Registration Agent",
        endpoint: process.env.AGENT0_ENDPOINT || "http://localhost:8000/a2a/agent0/v1",
        reputation: 180,
        isActive: true,
        country_code: "US",
        address: process.env.AGENT0_ADDRESS || "0xAA00000000000000000000000000000000000002",
        description: "A2A v0.2.9 compliant agent for data product registration with Dublin Core metadata",
        capabilities: ["cds-csn-generation", "ord-descriptor-creation", "dublin-core-metadata", "crd-integration"]
    },
    {
        name: "Data Standardization Agent",
        endpoint: process.env.AGENT1_ENDPOINT || "http://localhost:8000/a2a/agent1/v1",
        reputation: 175,
        isActive: true,
        country_code: "DE",
        address: process.env.AGENT1_ADDRESS || "0xAA00000000000000000000000000000000000003",
        description: "A2A v0.2.9 compliant agent for standardizing financial data to L4 hierarchical structure",
        capabilities: ["location-standardization", "account-standardization", "financial-data-processing", "hierarchy-generation"]
    },
    {
        name: "AI Preparation Agent",
        endpoint: process.env.AGENT2_ENDPOINT || "http://localhost:8000/a2a/agent2/v1",
        reputation: 170,
        isActive: true,
        country_code: "JP",
        address: process.env.AGENT2_ADDRESS || "0xAA00000000000000000000000000000000000004",
        description: "A2A v0.2.9 compliant agent for AI data preparation and vectorization",
        capabilities: ["data-vectorization", "ai-preparation", "embedding-generation", "feature-extraction"]
    },
    {
        name: "Vector Processing Agent",
        endpoint: process.env.AGENT3_ENDPOINT || "http://localhost:8000/a2a/agent3/v1",
        reputation: 185,
        isActive: true,
        country_code: "SG",
        address: process.env.AGENT3_ADDRESS || "0xAA00000000000000000000000000000000000005",
        description: "A2A v0.2.9 compliant agent for vector processing and knowledge graph management",
        capabilities: ["vector-processing", "hana-vector-engine", "knowledge-graph", "semantic-search"]
    },
    {
        name: "Calculation Validation Agent",
        endpoint: process.env.AGENT4_ENDPOINT || "http://localhost:8000/a2a/agent4/v1",
        reputation: 165,
        isActive: true,
        country_code: "IN",
        address: process.env.AGENT4_ADDRESS || "0xAA00000000000000000000000000000000000006",
        description: "A2A v0.2.9 compliant agent for dynamic computation quality testing",
        capabilities: ["calculation-validation", "quality-testing", "computation-verification", "formula-checking"]
    },
    {
        name: "QA Validation Agent",
        endpoint: process.env.AGENT5_ENDPOINT || "http://localhost:8000/a2a/agent5/v1",
        reputation: 160,
        isActive: true,
        country_code: "CA",
        address: process.env.AGENT5_ADDRESS || "0xAA00000000000000000000000000000000000007",
        description: "A2A compliant agent for dynamic factuality testing using ORD registry data",
        capabilities: ["qa-validation", "factuality-testing", "ord-validation", "data-quality-assurance"]
    },
    {
        name: "Data Manager Agent",
        endpoint: process.env.DATA_MANAGER_ENDPOINT || "http://localhost:8000/a2a/data_manager/v1",
        reputation: 190,
        isActive: true,
        country_code: "UK",
        address: process.env.DATA_MANAGER_ADDRESS || "0xAA00000000000000000000000000000000000008",
        description: "A2A v0.2.9 compliant agent for data management and storage operations",
        capabilities: ["data-storage", "data-retrieval", "data-lifecycle", "storage-optimization"]
    },
    {
        name: "Catalog Manager Agent",
        endpoint: process.env.CATALOG_MANAGER_ENDPOINT || "http://localhost:8000/a2a/catalog_manager/v1",
        reputation: 195,
        isActive: true,
        country_code: "FR",
        address: process.env.CATALOG_MANAGER_ADDRESS || "0xAA00000000000000000000000000000000000009",
        description: "A2A v0.2.9 compliant agent for ORD repository management",
        capabilities: ["ord-management", "catalog-operations", "metadata-management", "repository-sync"]
    },
    {
        name: "Agent Builder Agent",
        endpoint: process.env.AGENT_BUILDER_ENDPOINT || "http://localhost:8000/a2a/agent_builder/v1",
        reputation: 155,
        isActive: true,
        country_code: "AU",
        address: process.env.AGENT_BUILDER_ADDRESS || "0xAA00000000000000000000000000000000000010",
        description: "A2A v0.2.9 compliant agent for generating and managing other A2A agents",
        capabilities: ["agent-generation", "agent-deployment", "sdk-management", "agent-templates"]
    },
    {
        name: "Enhanced Calculation Agent",
        endpoint: process.env.CALCULATION_AGENT_ENDPOINT || "http://localhost:8000/a2a/calculation_agent/v1",
        reputation: 210,
        isActive: true,
        country_code: "CH",
        address: process.env.CALCULATION_AGENT_ADDRESS || "0xAA00000000000000000000000000000000000011",
        description: "Advanced calculation engine with self-healing capabilities and dynamic computation",
        capabilities: ["dynamic-calculation", "self-healing", "formula-optimization", "computation-validation"]
    },
    {
        name: "Reasoning Agent",
        endpoint: process.env.REASONING_AGENT_ENDPOINT || "http://localhost:8000/a2a/reasoning_agent/v1",
        reputation: 200,
        isActive: true,
        country_code: "NO",
        address: process.env.REASONING_AGENT_ADDRESS || "0xAA00000000000000000000000000000000000012",
        description: "Advanced reasoning and inference engine for complex decision making",
        capabilities: ["logical-reasoning", "inference-engine", "decision-making", "knowledge-reasoning"]
    },
    {
        name: "SQL Agent",
        endpoint: process.env.SQL_AGENT_ENDPOINT || "http://localhost:8000/a2a/sql_agent/v1",
        reputation: 175,
        isActive: true,
        country_code: "SE",
        address: process.env.SQL_AGENT_ADDRESS || "0xAA00000000000000000000000000000000000013",
        description: "Specialized agent for SQL query generation, optimization, and database operations",
        capabilities: ["sql-generation", "query-optimization", "database-operations", "schema-analysis"]
    },
    {
        name: "Developer Portal Agent Builder",
        endpoint: process.env.DEVELOPER_PORTAL_ENDPOINT || "http://localhost:8000/a2a/developer_portal/agent_builder/v1",
        reputation: 165,
        isActive: true,
        country_code: "NL",
        address: process.env.DEVELOPER_PORTAL_ADDRESS || "0xAA00000000000000000000000000000000000014",
        description: "Developer portal integration for agent building and deployment workflows",
        capabilities: ["portal-integration", "developer-tools", "agent-deployment", "workflow-management"]
    },
    {
        name: "Agent Builder Service",
        endpoint: process.env.AGENT_BUILDER_SERVICE_ENDPOINT || "http://localhost:8000/a2a/developer_portal/static/agent_builder/v1",
        reputation: 150,
        isActive: true,
        country_code: "DK",
        address: process.env.AGENT_BUILDER_SERVICE_ADDRESS || "0xAA00000000000000000000000000000000000015",
        description: "Static agent builder service for template-based agent generation",
        capabilities: ["template-generation", "static-building", "agent-scaffolding", "code-generation"]
    }
];

async function registerAgents() {
    console.log('Starting A2A Agents registration into A2A Network...\n');

    const successfulAgents = [];
    const failedAgents = [];

    for (const agent of A2A_AGENTS) {
        try {
            console.log(`Registering ${agent.name}...`);

            // First, create the agent
            const agentData = {
                name: agent.name,
                endpoint: agent.endpoint,
                reputation: agent.reputation,
                isActive: agent.isActive,
                country_code: agent.country_code,
                address: agent.address
            };

            const response = await axios.post(`${A2A_NETWORK_API}/Agents`, agentData, {
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            });
            const createdAgent = response.data;

            console.log(`✓ Created agent: ${agent.name} (ID: ${createdAgent.ID})`);

            // Store agent ID with capabilities for later use
            successfulAgents.push({
                id: createdAgent.ID,
                name: agent.name,
                capabilities: agent.capabilities
            });

            // TODO: Create capabilities and link them to the agent
            // This requires the Capabilities entity to be working properly

        } catch (error) {
            console.error(`✗ Failed to register ${agent.name}:`, error.response?.data || error.message);
            failedAgents.push(agent.name);
        }
    }

    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('Registration Summary:');
    console.log('='.repeat(60));
    console.log(`✓ Successfully registered: ${successfulAgents.length} agents`);
    successfulAgents.forEach(agent => {
        console.log(`  - ${agent.name} (ID: ${agent.id})`);
    });

    if (failedAgents.length > 0) {
        console.log(`\n✗ Failed to register: ${failedAgents.length} agents`);
        failedAgents.forEach(name => {
            console.log(`  - ${name}`);
        });
    }

    console.log('\n' + '='.repeat(60));
    console.log('Next Steps:');
    console.log('='.repeat(60));
    console.log('1. Create capabilities for each agent');
    console.log('2. Link capabilities to agents via AgentCapabilities');
    console.log('3. Create sample workflows using these agents');
    console.log('4. Register services provided by each agent');
    console.log('\nAccess the A2A Network UI at: ' + (process.env.A2A_NETWORK_UI_URL || 'http://localhost:4004/fiori-launchpad.html'));
}

// Execute the registration
registerAgents().catch(error => {
    console.error('Registration script failed:', error);
    process.exit(1);
});