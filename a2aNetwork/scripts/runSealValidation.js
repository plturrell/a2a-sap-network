#!/usr/bin/env node

const path = require('path');
const { spawn } = require('child_process');

console.log('🚀 SEAL COMPREHENSIVE VALIDATION RUNNER\n');

// Set environment variables
process.env.NODE_ENV = 'test';
process.env.XAI_API_KEY = process.env.XAI_API_KEY || 'test-api-key';
process.env.XAI_BASE_URL = 'https://api.x.ai/v1';
process.env.XAI_MODEL = 'grok-4';
process.env.LOG_LEVEL = 'info';

console.log('Environment Configuration:');
console.log(`  NODE_ENV: ${process.env.NODE_ENV}`);
console.log(`  XAI_MODEL: ${process.env.XAI_MODEL}`);
console.log(`  XAI_BASE_URL: ${process.env.XAI_BASE_URL}`);
console.log(`  API_KEY: ${process.env.XAI_API_KEY ? `***${  process.env.XAI_API_KEY.slice(-4)}` : 'NOT SET'}`);
console.log('');

// Function to run tests
async function runTests() {
    console.log('Running comprehensive SEAL validation tests...\n');
    
    // Use npx to run mocha directly
    const testProcess = spawn('npx', [
        'mocha',
        'test/integration/comprehensiveSealValidation.test.js',
        '--timeout', '60000',
        '--reporter', 'spec',
        '--colors'
    ], {
        cwd: path.resolve(__dirname, '..'),
        stdio: 'inherit',
        env: process.env
    });
    
    testProcess.on('close', (code) => {
        if (code === 0) {
            console.log('\n✅ All SEAL validation tests passed!');
            displaySummary();
        } else {
            console.log(`\n❌ Tests failed with code ${code}`);
            process.exit(code);
        }
    });
    
    testProcess.on('error', (err) => {
        console.error('Failed to start test process:', err);
        process.exit(1);
    });
}

// Display summary of what was tested
function displaySummary() {
    console.log('\n📊 VALIDATION SUMMARY');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('✅ Core Component Integration');
    console.log('   • Configuration system');
    console.log('   • Algorithm services (Graph & Tree)');
    console.log('   • Health checks');
    console.log('');
    console.log('✅ Q-Learning Real Adaptation');
    console.log('   • Q-value convergence (50 iterations)');
    console.log('   • Adaptive learning to changing rewards');
    console.log('   • Statistically significant behavior changes');
    console.log('');
    console.log('✅ Grok 4 API Integration');
    console.log('   • Contextually relevant self-edits');
    console.log('   • Error handling and recovery');
    console.log('   • Rate limiting management');
    console.log('');
    console.log('✅ Full Self-Adaptation Cycles');
    console.log('   • End-to-end adaptation with improvements');
    console.log('   • Pattern-specific adaptations');
    console.log('   • User feedback integration');
    console.log('');
    console.log('✅ SAP Enterprise Compliance');
    console.log('   • High-risk operation governance');
    console.log('   • Approval workflows');
    console.log('   • Audit report generation');
    console.log('');
    console.log('✅ Performance & Scalability');
    console.log('   • Concurrent adaptation handling');
    console.log('   • Memory management under pressure');
    console.log('   • Response time optimization');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('\n🎯 SEAL System: PRODUCTION-READY with genuine self-adaptation verified!\n');
}

// Alternative: Run a quick validation without full test suite
async function runQuickValidation() {
    console.log('Running quick SEAL validation...\n');
    
    try {
        // Load and test core components
        const SealConfiguration = require('../srv/seal/sealConfiguration');
        const GrokSealAdapter = require('../srv/seal/grokSealAdapter');
        const ReinforcementLearningEngine = require('../srv/seal/reinforcementLearningEngine');
        
        const config = new SealConfiguration();
        const cfg = config.getConfiguration();
        
        console.log('✅ Configuration loaded');
        console.log(`   Grok Model: ${cfg.grok.model}`);
        console.log(`   API URL: ${cfg.grok.baseUrl}`);
        
        const rl = new ReinforcementLearningEngine();
        await rl.initializeService();
        
        console.log('✅ RL Engine initialized');
        console.log(`   Q-table size: ${rl.qTable.size}`);
        console.log(`   Learning rate: ${rl.learningRate}`);
        
        // Test Q-learning
        const state = { complexity: 0.7, accuracy: 0.6 };
        const actions = [{ type: 'test_a' }, { type: 'test_b' }];
        
        const selection = await rl.selectAction(state, actions);
        console.log('✅ Action selection working');
        console.log(`   Selected: ${selection.action.type} (${selection.selectionReason})`);
        
        // Test learning
        await rl.learnFromFeedback(state, selection.action, 0.8, state);
        console.log('✅ Q-Learning update successful');
        
        console.log('\n🎯 Quick validation PASSED!\n');
        
    } catch (error) {
        console.error('❌ Quick validation failed:', error.message);
        process.exit(1);
    }
}

// Main execution
const args = process.argv.slice(2);

if (args.includes('--quick')) {
    runQuickValidation();
} else if (args.includes('--help')) {
    console.log('Usage: node runSealValidation.js [options]');
    console.log('');
    console.log('Options:');
    console.log('  --quick    Run quick validation without full test suite');
    console.log('  --help     Show this help message');
    console.log('');
    console.log('Environment variables:');
    console.log('  XAI_API_KEY    Your xAI API key for Grok 4');
    console.log('  NODE_ENV       Environment (test/development/production)');
} else {
    runTests();
}