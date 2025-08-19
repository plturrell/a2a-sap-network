#!/usr/bin/env node

const { AngleParser, AngleQueryExecutor } = require('../srv/glean/angleParser');

async function debugQueryExecution() {
    console.log('🔍 Debugging Query Execution');
    console.log('============================\n');
    
    try {
        const angleParser = new AngleParser();
        
        // Set up test fact database
        const factDb = {
            'src.File': [
                {
                    id: 'file1',
                    key: { file: 'test.js' },
                    value: {
                        file: 'test.js',
                        language: 'javascript',
                        size: 1000,
                        symbols: 5,
                        lines: 50
                    }
                }
            ]
        };
        
        console.log('📄 Test fact database:');
        console.log(JSON.stringify(factDb, null, 2));
        
        // Test query (note: using value.language instead of language)
        const queryText = 'query FindJSFiles() : [File] = src.File { file, value.language = "javascript" }';
        console.log(`\n🔧 Parsing query: ${queryText}`);
        
        const parsedQuery = angleParser.parseQuery(queryText);
        console.log('✅ Parsed query:');
        console.log(JSON.stringify(parsedQuery, null, 2));
        
        // Execute query
        console.log('\n⚡ Executing query...');
        const executor = new AngleQueryExecutor(factDb, angleParser.predicateDefinitions);
        
        // Debug: Check if facts exist for the predicate
        console.log('\n🔍 Debugging fact lookup:');
        console.log(`Looking for predicate: "src.File"`);
        console.log(`Available predicates in factDb:`, Object.keys(factDb));
        console.log(`Facts for src.File:`, factDb['src.File']);
        
        // Debug: Manual fact value checking
        const testFact = factDb['src.File'][0];
        console.log('\n🔍 Testing fact value extraction:');
        console.log(`Test fact:`, testFact);
        console.log(`language value (from fact.value.language):`, testFact.value.language);
        console.log(`language value (from fact.language):`, testFact.language);
        
        const results = executor.execute(parsedQuery);
        
        console.log('\n📊 Query results:');
        console.log(JSON.stringify(results, null, 2));
        console.log(`\nResult type: ${typeof results}`);
        console.log(`Is array: ${Array.isArray(results)}`);
        console.log(`Length: ${results.length || 'N/A'}`);
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        console.error('📚 Stack:', error.stack);
    }
}

debugQueryExecution();