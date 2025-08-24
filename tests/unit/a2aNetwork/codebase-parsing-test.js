#!/usr/bin/env node

/**
 * Real Enhanced AST Parsing Test against A2A Codebase
 * Tests the new enhanced parsing against actual production files
 */

const SCIPIndexer = require('../srv/glean/scipIndexer');
const path = require('path');
const fs = require('fs').promises;

async function testRealEnhancedParsing() {
    console.log('🚀 Real Enhanced AST Parsing Test');
    console.log('================================\n');
    
    const indexer = new SCIPIndexer(path.join(__dirname, '..'));
    await indexer.initialize();
    
    // Test files from the real A2A codebase
    const testFiles = [
        'srv/server.js',
        'srv/sapA2aService.js', 
        'app/a2aFiori/webapp/Component.js',
        'srv/middleware/security.js'
    ];
    
    const results = {};
    
    for (const filePath of testFiles) {
        console.log(`📋 Testing Enhanced Parsing: ${filePath}`);
        
        try {
            const fullPath = path.join(__dirname, '..', filePath);
            const content = await fs.readFile(fullPath, 'utf8');
            
            const document = {
                relative_path: filePath,
                occurrences: [],
                symbols: []
            };
            
            // Parse with enhanced AST parsing
            const enhancedResult = await indexer.parseTypeScript(fullPath, content, document);
            
            // Parse with fallback for comparison
            const fallbackDocument = {
                relative_path: filePath,
                occurrences: [],
                symbols: []
            };
            const fallbackResult = await indexer.parseTypeScriptFallback(fullPath, content, fallbackDocument);
            
            // Generate facts from enhanced parsing
            const enhancedFacts = indexer.scipToGleanFacts(enhancedResult, filePath);
            const fallbackFacts = indexer.scipToGleanFacts(fallbackResult, filePath);
            
            const analysis = {
                file: filePath,
                fileSize: content.length,
                enhanced: {
                    symbols: enhancedResult.symbols.length,
                    facts: enhancedFacts.length,
                    imports: enhancedResult.symbols.filter(s => s.definition?.syntax_kind === 'ImportDeclaration').length,
                    functions: enhancedResult.symbols.filter(s => s.definition?.syntax_kind === 'FunctionDeclaration').length,
                    classes: enhancedResult.symbols.filter(s => s.definition?.syntax_kind === 'ClassDeclaration').length,
                    variables: enhancedResult.symbols.filter(s => s.definition?.syntax_kind === 'VariableDeclaration').length,
                    methods: enhancedResult.symbols.filter(s => s.definition?.syntax_kind === 'MethodDefinition').length,
                    hasEnhancedMetadata: enhancedFacts.some(f => f.value.parameterCount !== undefined || f.value.methodCount !== undefined)
                },
                fallback: {
                    symbols: fallbackResult.symbols.length,
                    facts: fallbackFacts.length
                }
            };
            
            analysis.improvement = {
                symbolRatio: analysis.enhanced.symbols / Math.max(analysis.fallback.symbols, 1),
                factRatio: analysis.enhanced.facts / Math.max(analysis.fallback.facts, 1)
            };
            
            results[filePath] = analysis;
            
            console.log(`✅ ${filePath}`);
            console.log(`   Enhanced: ${analysis.enhanced.symbols} symbols, ${analysis.enhanced.facts} facts`);
            console.log(`   Fallback: ${analysis.fallback.symbols} symbols, ${analysis.fallback.facts} facts`);
            console.log(`   Improvement: ${analysis.improvement.symbolRatio.toFixed(1)}x symbols, ${analysis.improvement.factRatio.toFixed(1)}x facts`);
            console.log(`   Types: ${analysis.enhanced.imports}i ${analysis.enhanced.functions}f ${analysis.enhanced.classes}c ${analysis.enhanced.variables}v ${analysis.enhanced.methods}m`);
            console.log(`   Enhanced metadata: ${analysis.enhanced.hasEnhancedMetadata ? '✓' : '✗'}`);
            console.log('');
            
        } catch (error) {
            console.error(`❌ Error processing ${filePath}: ${error.message}`);
            results[filePath] = { error: error.message };
        }
    }
    
    // Summary analysis
    console.log('📊 Enhanced Parsing Performance Summary');
    console.log('======================================');
    
    const successfulTests = Object.values(results).filter(r => !r.error);
    const totalFiles = testFiles.length;
    const successfulFiles = successfulTests.length;
    
    if (successfulFiles > 0) {
        const avgSymbolImprovement = successfulTests.reduce((sum, r) => sum + r.improvement.symbolRatio, 0) / successfulFiles;
        const avgFactImprovement = successfulTests.reduce((sum, r) => sum + r.improvement.factRatio, 0) / successfulFiles;
        const totalEnhancedSymbols = successfulTests.reduce((sum, r) => sum + r.enhanced.symbols, 0);
        const totalFallbackSymbols = successfulTests.reduce((sum, r) => sum + r.fallback.symbols, 0);
        const filesWithEnhancedMetadata = successfulTests.filter(r => r.enhanced.hasEnhancedMetadata).length;
        
        console.log(`✅ Success Rate: ${successfulFiles}/${totalFiles} files (${((successfulFiles/totalFiles)*100).toFixed(1)}%)`);
        console.log(`📈 Average Symbol Improvement: ${avgSymbolImprovement.toFixed(1)}x`);
        console.log(`📈 Average Fact Improvement: ${avgFactImprovement.toFixed(1)}x`);
        console.log(`🔢 Total Enhanced Symbols: ${totalEnhancedSymbols} vs ${totalFallbackSymbols} fallback`);
        console.log(`🔍 Files with Enhanced Metadata: ${filesWithEnhancedMetadata}/${successfulFiles}`);
        
        // Type distribution
        const totalImports = successfulTests.reduce((sum, r) => sum + r.enhanced.imports, 0);
        const totalFunctions = successfulTests.reduce((sum, r) => sum + r.enhanced.functions, 0);
        const totalClasses = successfulTests.reduce((sum, r) => sum + r.enhanced.classes, 0);
        const totalVariables = successfulTests.reduce((sum, r) => sum + r.enhanced.variables, 0);
        const totalMethods = successfulTests.reduce((sum, r) => sum + r.enhanced.methods, 0);
        
        console.log('📋 Symbol Distribution:');
        console.log(`   Imports: ${totalImports}`);
        console.log(`   Functions: ${totalFunctions}`);
        console.log(`   Classes: ${totalClasses}`);
        console.log(`   Variables: ${totalVariables}`);
        console.log(`   Methods: ${totalMethods}`);
        
        const testsPassed = 
            successfulFiles === totalFiles &&
            avgSymbolImprovement >= 1.0 &&
            filesWithEnhancedMetadata > 0;
        
        console.log('\n============================================================');
        if (testsPassed) {
            console.log('🎉 SUCCESS: Enhanced AST parsing works with real A2A code!');
            console.log('\n✅ Real Code Validation:');
            console.log('   • All test files processed successfully');
            console.log('   • Enhanced parsing provides equal or better symbol extraction');
            console.log('   • Rich metadata available for enhanced analysis');
            console.log('   • Production-ready for A2A codebase');
        } else {
            console.log('⚠️  PARTIAL SUCCESS: Enhanced parsing has limitations with real code');
        }
        
        return testsPassed;
        
    } else {
        console.log('❌ FAILED: No files processed successfully');
        return false;
    }
}

if (require.main === module) {
    testRealEnhancedParsing().then(success => {
        process.exit(success ? 0 : 1);
    });
}

module.exports = { testRealEnhancedParsing };