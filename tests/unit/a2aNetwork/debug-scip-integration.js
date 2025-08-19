#!/usr/bin/env node

const SCIPIndexer = require('../srv/glean/scipIndexer');
const path = require('path');

async function debugSCIPIntegration() {
    console.log('🔍 Debugging SCIP Integration');
    console.log('============================\n');
    
    try {
        const indexer = new SCIPIndexer(process.cwd());
        
        console.log('📋 Initializing SCIP indexer...');
        await indexer.initialize();
        
        console.log('📋 Checking language servers configuration:');
        console.log('Available language servers:', Array.from(indexer.languageServers.keys()));
        
        indexer.languageServers.forEach((server, language) => {
            console.log(`  ${language}:`, {
                command: typeof server.command === 'function' ? 'function' : server.command,
                extensions: server.extensions
            });
        });
        
        console.log('\n📋 Testing language lookup:');
        const testLanguages = ['typescript', 'javascript', 'python', 'solidity'];
        testLanguages.forEach(lang => {
            const server = indexer.languageServers.get(lang);
            console.log(`  ${lang}: ${server ? '✓ Found' : '✗ Not found'}`);
        });
        
        console.log('\n📋 Testing file finding for typescript:');
        const files = await indexer.findFilesForLanguage('typescript');
        console.log(`Found ${files.length} TypeScript/JavaScript files:`, files.slice(0, 5));
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        console.error('📚 Stack:', error.stack);
    }
}

debugSCIPIntegration();