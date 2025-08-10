/**
 * Register A2A Agents from a2a_agents project into A2A Network
 * This script creates agent records in the SAP CAP A2A Network system
 */

const axios = require('axios');

const A2A_NETWORK_API = 'http://localhost:4004/api/v1';

// Define all agents from the a2a_agents project
const A2A_AGENTS = [
    {
        name: "Agent Manager",
        endpoint: "http://localhost:8000/a2a/agent_manager/v1",
        reputation: 200, // Highest reputation for the orchestrator
        isActive: true,
        country_code: "US",
        address: "0xAA00000000000000000000000000000000000001",
        description: "Manages A2A ecosystem registration, trust contracts, and workflow orchestration",
        capabilities: ["agent-registration", "trust-management", "workflow-orchestration", "ecosystem-management"]
    },
    {
        name: "Data Product Registration Agent",
        endpoint: "http://localhost:8000/a2a/agent0/v1",
        reputation: 180,
        isActive: true,
        country_code: "US",
        address: "0xAA00000000000000000000000000000000000002",
        description: "A2A v0.2.9 compliant agent for data product registration with Dublin Core metadata",
        capabilities: ["cds-csn-generation", "ord-descriptor-creation", "dublin-core-metadata", "crd-integration"]
    },
    {
        name: "Data Standardization Agent",
        endpoint: "http://localhost:8000/a2a/agent1/v1",
        reputation: 175,
        isActive: true,
        country_code: "DE",
        address: "0xAA00000000000000000000000000000000000003",
        description: "A2A v0.2.9 compliant agent for standardizing financial data to L4 hierarchical structure",
        capabilities: ["location-standardization", "account-standardization", "financial-data-processing", "hierarchy-generation"]
    },
    {
        name: "AI Preparation Agent",
        endpoint: "http://localhost:8000/a2a/agent2/v1",
        reputation: 170,
        isActive: true,
        country_code: "JP",
        address: "0xAA00000000000000000000000000000000000004",
        description: "A2A v0.2.9 compliant agent for AI data preparation and vectorization",
        capabilities: ["data-vectorization", "ai-preparation", "embedding-generation", "feature-extraction"]
    },
    {
        name: "Vector Processing Agent",
        endpoint: "http://localhost:8000/a2a/agent3/v1",
        reputation: 185,
        isActive: true,
        country_code: "SG",
        address: "0xAA00000000000000000000000000000000000005",
        description: "A2A v0.2.9 compliant agent for vector processing and knowledge graph management",
        capabilities: ["vector-processing", "hana-vector-engine", "knowledge-graph", "semantic-search"]
    },
    {
        name: "Calculation Validation Agent",
        endpoint: "http://localhost:8000/a2a/agent4/v1",
        reputation: 165,
        isActive: true,
        country_code: "IN",
        address: "0xAA00000000000000000000000000000000000006",
        description: "A2A v0.2.9 compliant agent for dynamic computation quality testing",
        capabilities: ["calculation-validation", "quality-testing", "computation-verification", "formula-checking"]
    },
    {
        name: "QA Validation Agent",
        endpoint: "http://localhost:8000/a2a/agent5/v1",
        reputation: 160,
        isActive: true,
        country_code: "CA",
        address: "0xAA00000000000000000000000000000000000007",
        description: "A2A compliant agent for dynamic factuality testing using ORD registry data",
        capabilities: ["qa-validation", "factuality-testing", "ord-validation", "data-quality-assurance"]
    },
    {
        name: "Data Manager Agent",
        endpoint: "http://localhost:8000/a2a/data_manager/v1",
        reputation: 190,
        isActive: true,
        country_code: "UK",
        address: "0xAA00000000000000000000000000000000000008",
        description: "A2A v0.2.9 compliant agent for data management and storage operations",
        capabilities: ["data-storage", "data-retrieval", "data-lifecycle", "storage-optimization"]
    },
    {
        name: "Catalog Manager Agent",
        endpoint: "http://localhost:8000/a2a/catalog_manager/v1",
        reputation: 195,
        isActive: true,
        country_code: "FR",
        address: "0xAA00000000000000000000000000000000000009",
        description: "A2A v0.2.9 compliant agent for ORD repository management",
        capabilities: ["ord-management", "catalog-operations", "metadata-management", "repository-sync"]
    },
    {
        name: "Agent Builder Agent",
        endpoint: "http://localhost:8000/a2a/agent_builder/v1",
        reputation: 155,
        isActive: true,
        country_code: "AU",
        address: "0xAA00000000000000000000000000000000000010",
        description: "A2A v0.2.9 compliant agent for generating and managing other A2A agents",
        capabilities: ["agent-generation", "agent-deployment", "sdk-management", "agent-templates"]
    }
];

async function registerAgents() {
    console.log('Starting A2A Agents registration into A2A Network...\n');
    
    const headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    };
    
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
            
            const response = await axios.post(`${A2A_NETWORK_API}/Agents`, agentData, { headers });
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
    console.log('\nAccess the A2A Network UI at: http://localhost:4004/fiori-launchpad.html');
}

// Execute the registration
registerAgents().catch(error => {
    console.error('Registration script failed:', error);
    process.exit(1);
});