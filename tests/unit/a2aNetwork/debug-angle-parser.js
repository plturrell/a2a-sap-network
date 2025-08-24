#!/usr/bin/env node

const { AngleParser } = require('../srv/glean/angleParser');
const fs = require('fs').promises;
const path = require('path');

async function debugAngleParser() {
    console.log('🔍 Debugging Angle Parser');
    console.log('=========================\n');
    
    try {
        const angleParser = new AngleParser();
        
        // Load the schema file
        const schemaPath = path.join(__dirname, 'integration/simple_test_schema.angle');
        console.log(`📋 Loading schema from: ${schemaPath}`);
        
        const schemaContent = await fs.readFile(schemaPath, 'utf8');
        console.log(`📄 Schema content:\n${schemaContent}\n`);
        
        // Parse the schema
        console.log('🔧 Parsing schema...');
        const parsedSchema = angleParser.parseSchema(schemaContent);
        
        console.log('✅ Parsed schema:', JSON.stringify(parsedSchema, null, 2));
        
        // Check predicates
        console.log('\n📊 Schema analysis:');
        console.log(`   Name: ${parsedSchema.name}`);
        console.log(`   Version: ${parsedSchema.version}`);
        console.log(`   Predicates count: ${Object.keys(parsedSchema.predicates).length}`);
        console.log('   Predicates:', Object.keys(parsedSchema.predicates));
        
        // Validate each predicate
        Object.entries(parsedSchema.predicates).forEach(([name, predicate]) => {
            console.log(`\n🔍 Predicate "${name}":`, predicate);
        });
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        console.error('📚 Stack:', error.stack);
    }
}

debugAngleParser();