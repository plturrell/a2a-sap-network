#!/usr/bin/env node

const SCIPIndexer = require('../srv/glean/scipIndexer');
const GleanFactTransformer = require('../srv/glean/gleanFactTransformer');
const path = require('path');
const fs = require('fs').promises;

async function debugSecurityDetection() {
    console.log('🔍 Debugging Security Detection');
    console.log('==============================\n');
    
    try {
        const indexer = new SCIPIndexer(process.cwd());
        await indexer.initialize();
        
        const transformer = new GleanFactTransformer();
        
        // Create test file with security issues
        const testContent = `
// Test JavaScript file with security issues
function calculateSum(a, b) {
    if (a > 10) {
        return a + b;
    }
    return 0;
}

function dangerousFunction(userInput) {
    // Security issue: eval usage
    return eval(userInput);
}

class TestClass {
    constructor(name) {
        this.name = name;
    }
    
    getName() {
        return this.name;
    }
}

export { calculateSum, TestClass, dangerousFunction };
        `.trim();
        
        const testFilePath = path.join(process.cwd(), 'debug_security_test.js');
        await fs.writeFile(testFilePath, testContent);
        
        try {
            console.log('📋 Test file content:');
            console.log(testContent);
            console.log(`\n📋 File size: ${testContent.length} characters`);
            
            // Parse with SCIP indexer
            console.log('\n📋 Parsing with SCIP indexer...');
            const scipDoc = await indexer.runSCIPIndexer(
                { command: 'scip-typescript', extensions: ['.js'] },
                testFilePath
            );
            
            console.log(`✅ SCIP parsing completed`);
            console.log(`   Symbols: ${scipDoc.scip.symbols.length}`);
            console.log(`   Direct facts: ${scipDoc.glean.length}`);
            
            // Transform through fact transformer
            console.log('\n📋 Transforming through fact transformer...');
            const scipIndex = {
                documents: [scipDoc.scip],
                external_symbols: [],
                symbol_roles: []
            };
            
            const gleanFacts = transformer.transformSCIPToGlean(scipIndex);
            
            console.log(`✅ Fact transformation completed`);
            console.log(`   Total fact types: ${Object.keys(gleanFacts).length}`);
            Object.entries(gleanFacts).forEach(([type, facts]) => {
                console.log(`   ${type}: ${facts.length} facts`);
            });
            
            // Check security facts specifically
            const securityFacts = gleanFacts['src.SecurityIssue'] || [];
            console.log(`\n🔍 Security analysis results:`);
            console.log(`   Security issues found: ${securityFacts.length}`);
            
            if (securityFacts.length > 0) {
                console.log(`   Security issues:`);
                securityFacts.forEach((fact, index) => {
                    console.log(`      ${index + 1}. ${fact.value.issue_type} at line ${fact.value.line} (${fact.value.severity})`);
                    console.log(`         Description: ${fact.value.description}`);
                });
            } else {
                console.log(`   ❌ No security issues detected - checking why...`);
                
                // Check if file content is being analyzed
                if (scipDoc.scip.relative_path) {
                    console.log(`   📄 File path: ${scipDoc.scip.relative_path}`);
                }
                
                // Check language detection
                const fileFacts = gleanFacts['src.File'] || [];
                if (fileFacts.length > 0) {
                    console.log(`   🔍 File analysis:`);
                    fileFacts.forEach(fact => {
                        console.log(`      Language: ${fact.value.language}`);
                        console.log(`      File: ${fact.value.file}`);
                    });
                }
            }
            
        } finally {
            // Clean up
            try {
                await fs.unlink(testFilePath);
            } catch (e) {
                // Ignore cleanup errors
            }
        }
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        console.error('📚 Stack:', error.stack);
    }
}

debugSecurityDetection();