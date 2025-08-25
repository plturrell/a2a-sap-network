#!/usr/bin/env node
/**
 * Test script to verify Agent 14 and 15 integration
 * Tests that the adapters can communicate with the real Python backends
 */

const Agent14Adapter = require('./srv/adapters/agent14-adapter');
const Agent15Adapter = require('./srv/adapters/agent15-adapter');

async function testAgent14() {
    console.log('\n=== Testing Agent 14 (Embedding Fine-Tuner) ===');
    const adapter = new Agent14Adapter();

    try {
        // Test creating an embedding model
        console.log('\n1. Creating embedding model...');
        const model = await adapter.createEmbeddingModel({
            name: 'test-embedding-model',
            description: 'Test model for integration',
            baseModel: 'sentence-transformers/all-MiniLM-L6-v2',
            modelType: 'sentence_transformer',
            optimizationStrategy: 'contrastive_learning'
        });
        console.log('✓ Model created:', model);

        // Test listing models
        console.log('\n2. Listing embedding models...');
        const models = await adapter.getEmbeddingModels({});
        console.log('✓ Models found:', models.length);

        console.log('\n✅ Agent 14 integration test passed!');
        return true;
    } catch (error) {
        console.error('\n❌ Agent 14 test failed:', error.message);
        return false;
    }
}

async function testAgent15() {
    console.log('\n=== Testing Agent 15 (Orchestrator) ===');
    const adapter = new Agent15Adapter();

    try {
        // Test creating a workflow
        console.log('\n1. Creating workflow...');
        const workflow = await adapter.createWorkflow({
            name: 'test-workflow',
            description: 'Test workflow for integration',
            tasks: [
                { agent_id: 'agent1', task_type: 'standardize', parameters: {} },
                { agent_id: 'agent2', task_type: 'prepare', parameters: {} }
            ],
            strategy: 'sequential'
        });
        console.log('✓ Workflow created:', workflow);

        // Test listing workflows
        console.log('\n2. Listing workflows...');
        const workflows = await adapter.getWorkflows({});
        console.log('✓ Workflows found:', workflows.length);

        console.log('\n✅ Agent 15 integration test passed!');
        return true;
    } catch (error) {
        console.error('\n❌ Agent 15 test failed:', error.message);
        return false;
    }
}

async function runTests() {
    console.log('Starting integration tests for Agents 14 and 15...');
    console.log('Make sure the Python backend servers are running:');
    console.log('- Agent 14: http://localhost:8014');
    console.log('- Agent 15: http://localhost:8015');

    const results = await Promise.all([
        testAgent14(),
        testAgent15()
    ]);

    const allPassed = results.every(r => r === true);

    if (allPassed) {
        console.log('\n🎉 All integration tests passed!');
        console.log('\nThe mock implementations have been successfully replaced with real backends.');
    } else {
        console.log('\n⚠️  Some tests failed. Please check:');
        console.log('1. Python backend servers are running');
        console.log('2. Ports 8014 and 8015 are accessible');
        console.log('3. No firewall blocking connections');
    }
}

// Run the tests
runTests().catch(console.error);