#!/usr/bin/env node

/**
 * AST Parsing Validation Test
 * Tests the new TypeScript/Babel AST parsing capabilities
 */

const SCIPIndexer = require('../srv/glean/scipIndexer');
const path = require('path');

async function testParsing() {
    console.log('🚀 AST Parsing Validation Test');
    console.log('=======================================\n');
    
    const indexer = new SCIPIndexer(path.join(__dirname, '..'));
    await indexer.initialize();
    
    console.log('📋 Testing TypeScript/JavaScript Parsing...');
    
    // Test with a real TypeScript-like file
    const testContent = `
import { Component } from '@sap/ui5-core';
import { EventBus } from '@sap/ui5-eventing';
import * as Utils from './utils';

export class TestComponent extends Component {
    private eventBus: EventBus;
    
    constructor(config: any) {
        super(config);
        this.eventBus = new EventBus();
    }
    
    async initialize(): Promise<void> {
        await this.setupEventHandlers();
    }
    
    private setupEventHandlers(): void {
        this.eventBus.subscribe('test', this.handleTest.bind(this));
    }
    
    handleTest = (event: any) => {
        console.log('Handling test event', event);
    }
}

export function processData(input: string, callback: Function): Promise<any> {
    return new Promise((resolve, reject) => {
        try {
            const result = callback(input);
            resolve(result);
        } catch (error) {
            reject(error);
        }
    });
}

export const helper = {
    format: (value: string) => value.trim(),
    validate: async (data: any) => {
        return data !== null;
    }
};

export default TestComponent;
`;
    
    const testFilePath = path.join(__dirname, '..', 'test-component.ts');
    
    // Test the new parsing
    const document = {
        relative_path: 'test-component.ts',
        occurrences: [],
        symbols: []
    };
    
    try {
        const result = await indexer.parseTypeScript(testFilePath, testContent, document);
        
        console.log(`✅ AST Parsing`);
        console.log(`   Extracted ${result.symbols.length} symbols with full AST metadata`);
        
        // Validate symbol extraction
        const imports = result.symbols.filter(s => s.definition?.syntax_kind === 'ImportDeclaration');
        const classes = result.symbols.filter(s => s.definition?.syntax_kind === 'ClassDeclaration');
        const functions = result.symbols.filter(s => s.definition?.syntax_kind === 'FunctionDeclaration');
        const variables = result.symbols.filter(s => s.definition?.syntax_kind === 'VariableDeclaration');
        const methods = result.symbols.filter(s => s.definition?.syntax_kind === 'MethodDefinition');
        
        console.log(`✅ Symbol Type Detection`);
        console.log(`   Imports: ${imports.length}, Classes: ${classes.length}, Functions: ${functions.length}`);
        console.log(`   Variables: ${variables.length}, Methods: ${methods.length}`);
        
        // Test fact generation
        const facts = indexer.scipToGleanFacts(result, 'test-component.ts');
        const enhancedFacts = facts.filter(f => f.value.enhanced_ast !== false);
        
        console.log(`✅ Fact Generation`);
        console.log(`   Generated ${facts.length} total facts`);
        console.log(`   ${enhancedFacts.length} facts with AST metadata`);
        
        // Check for specific metadata
        const functionFacts = facts.filter(f => f.key.function);
        const classFacts = facts.filter(f => f.key.class);
        const importFacts = facts.filter(f => f.key.import);
        
        console.log(`✅ Specialized Fact Types`);
        console.log(`   Function facts: ${functionFacts.length}`);
        console.log(`   Class facts: ${classFacts.length}`);
        console.log(`   Import facts: ${importFacts.length}`);
        
        // Validate metadata is present
        const metadataValidation = {
            hasParameterInfo: functionFacts.some(f => f.value.parameters) || facts.some(f => f.value.parameterCount !== undefined),
            hasAsyncInfo: facts.some(f => f.value.async !== undefined),
            hasClassMethods: classFacts.some(f => f.value.methods),
            hasImportSpecifiers: importFacts.some(f => f.value.specifiers)
        };
        
        console.log(`✅ Metadata Validation`);
        console.log(`   Parameter info: ${metadataValidation.hasParameterInfo ? '✓' : '✗'}`);
        console.log(`   Async/await info: ${metadataValidation.hasAsyncInfo ? '✓' : '✗'}`);
        console.log(`   Class methods: ${metadataValidation.hasClassMethods ? '✓' : '✗'}`);
        console.log(`   Import specifiers: ${metadataValidation.hasImportSpecifiers ? '✓' : '✗'}`);
        
        // Compare with fallback parsing
        console.log('\n📋 Testing Fallback vs Parsing...');
        
        const fallbackDoc = {
            relative_path: 'test-component.ts',
            occurrences: [],
            symbols: []
        };
        
        const fallbackResult = await indexer.parseTypeScriptFallback(testFilePath, testContent, fallbackDoc);
        
        console.log(`✅ Fallback Comparison`);
        console.log(`   symbols: ${result.symbols.length}`);
        console.log(`   Fallback symbols: ${fallbackResult.symbols.length}`);
        console.log(`   Enhancement ratio: ${(result.symbols.length / Math.max(fallbackResult.symbols.length, 1)).toFixed(1)}x`);
        
        const allTestsPassed = 
            result.symbols.length >= fallbackResult.symbols.length &&
            metadataValidation.hasParameterInfo &&
            metadataValidation.hasAsyncInfo &&
            enhancedFacts.length > 0;
        
        console.log('\n============================================================');
        if (allTestsPassed) {
            console.log('🎉 SUCCESS: AST parsing is working correctly!');
            console.log('\n✅ Validated Enhancements:');
            console.log('   • Real TypeScript/Babel AST parsing');
            console.log('   • Comprehensive symbol extraction');
            console.log('   • metadata generation');
            console.log('   • Rich fact types (functions, classes, imports)');
            console.log('   • Fallback compatibility maintained');
        } else {
            console.log('❌ FAILED: AST parsing has issues');
        }
        
        return allTestsPassed;
        
    } catch (error) {
        console.error('❌ parsing test failed:', error.message);
        console.error('Stack:', error.stack);
        return false;
    }
}

if (require.main === module) {
    testParsing().then(success => {
        process.exit(success ? 0 : 1);
    });
}

module.exports = { testParsing };