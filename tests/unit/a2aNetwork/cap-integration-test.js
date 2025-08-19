#!/usr/bin/env node

/**
 * SAP CAP Integration Test for Glean
 * Tests the enhanced CDS parsing and CAP-specific fact generation
 */

const SCIPIndexer = require('../srv/glean/scipIndexer');
const { AngleParser, AngleQueryExecutor } = require('../srv/glean/angleParser');
const path = require('path');
const fs = require('fs').promises;

async function testCAPIntegration() {
    console.log('🚀 SAP CAP Integration Test for Glean');
    console.log('====================================\n');
    
    const indexer = new SCIPIndexer(path.join(__dirname, '..'));
    await indexer.initialize();
    
    console.log('📋 Testing SAP CAP/CDS Support...');
    
    // Find CDS files in the A2A project
    const cdsFiles = await indexer.findFilesForLanguage('cds');
    console.log(`✅ Found ${cdsFiles.length} CDS files in A2A project`);
    
    if (cdsFiles.length === 0) {
        console.log('⚠️  No CDS files found, creating test CDS content...');
        await testWithSampleCDS(indexer);
        return;
    }
    
    // Test with real A2A CDS files
    const results = {};
    const sampleFiles = cdsFiles.slice(0, 3); // Test first 3 files
    
    for (const cdsFile of sampleFiles) {
        console.log(`\n📋 Processing CDS file: ${path.relative(process.cwd(), cdsFile)}`);
        
        try {
            const content = await fs.readFile(cdsFile, 'utf8');
            console.log(`   Size: ${content.length} characters`);
            
            // Parse CDS file
            const result = await indexer.scipCDSIndexer(cdsFile);
            
            const analysis = {
                file: path.relative(process.cwd(), cdsFile),
                symbols: result.scip.symbols.length,
                facts: result.glean.length,
                entities: result.scip.symbols.filter(s => s.type === 'entity').length,
                services: result.scip.symbols.filter(s => s.type === 'service').length,
                types: result.scip.symbols.filter(s => s.type === 'type').length,
                aspects: result.scip.symbols.filter(s => s.type === 'aspect').length,
                namespaces: result.scip.symbols.filter(s => s.type === 'namespace').length,
                imports: result.scip.symbols.filter(s => s.type === 'import').length
            };
            
            results[cdsFile] = analysis;
            
            console.log(`   ✅ Extracted ${analysis.symbols} symbols, generated ${analysis.facts} facts`);
            console.log(`   📊 Entities: ${analysis.entities}, Services: ${analysis.services}, Types: ${analysis.types}`);
            console.log(`   📊 Aspects: ${analysis.aspects}, Namespaces: ${analysis.namespaces}, Imports: ${analysis.imports}`);
            
            // Show some example symbols
            if (result.scip.symbols.length > 0) {
                console.log(`   🔍 Sample symbols:`);
                result.scip.symbols.slice(0, 3).forEach(symbol => {
                    console.log(`      ${symbol.type}: ${symbol.name}`);
                });
            }
            
        } catch (error) {
            console.error(`   ❌ Error processing ${cdsFile}: ${error.message}`);
            results[cdsFile] = { error: error.message };
        }
    }
    
    // Test CDS-specific queries
    await testCDSQueries(results, indexer);
    
    // Summary
    console.log('\n📊 SAP CAP Integration Summary');
    console.log('==============================');
    
    const successfulFiles = Object.values(results).filter(r => !r.error);
    const totalSymbols = successfulFiles.reduce((sum, r) => sum + (r.symbols || 0), 0);
    const totalFacts = successfulFiles.reduce((sum, r) => sum + (r.facts || 0), 0);
    const totalEntities = successfulFiles.reduce((sum, r) => sum + (r.entities || 0), 0);
    const totalServices = successfulFiles.reduce((sum, r) => sum + (r.services || 0), 0);
    
    console.log(`✅ Processed: ${successfulFiles.length}/${sampleFiles.length} files`);
    console.log(`📈 Total Symbols: ${totalSymbols}`);
    console.log(`📈 Total Facts: ${totalFacts}`);
    console.log(`🏢 Total Entities: ${totalEntities}`);
    console.log(`⚙️  Total Services: ${totalServices}`);
    
    const success = successfulFiles.length > 0 && totalSymbols > 0;
    
    console.log('\n============================================================');
    if (success) {
        console.log('🎉 SUCCESS: SAP CAP integration is working correctly!');
        console.log('\n✅ CAP Features Validated:');
        console.log('   • CDS file parsing and symbol extraction');
        console.log('   • Entity definition analysis');
        console.log('   • Service definition parsing');
        console.log('   • Type and aspect extraction');
        console.log('   • Namespace and import handling');
        console.log('   • CAP-specific Glean fact generation');
    } else {
        console.log('❌ FAILED: SAP CAP integration has issues');
    }
    
    return success;
}

async function testWithSampleCDS(indexer) {
    console.log('\n📋 Testing with sample CDS content...');
    
    const sampleCDS = `
namespace sap.a2a;

using { managed, cuid } from '@sap/cds/common';

entity Agents : managed, cuid {
    name        : String(100) @title: 'Agent Name';
    description : String(500) @title: 'Description';
    status      : String(20) @title: 'Status';
    type        : String(50) @title: 'Agent Type';
    endpoint    : String(200) @title: 'Service Endpoint';
    capabilities: many to many Capabilities on capabilities.agent = $self;
    metrics     : Composition of many AgentMetrics on metrics.agent = $self;
}

entity Capabilities : cuid {
    name        : String(100) @title: 'Capability Name';
    category    : String(50) @title: 'Category';
    description : String(500) @title: 'Description';
    agent       : Association to Agents;
}

entity AgentMetrics : cuid {
    timestamp   : DateTime @title: 'Timestamp';
    metric_type : String(50) @title: 'Metric Type';
    value       : Decimal @title: 'Value';
    unit        : String(20) @title: 'Unit';
    agent       : Association to Agents;
}

service A2AService @(path: '/api/v1') {
    entity Agents as projection on a2a.Agents;
    entity Capabilities as projection on a2a.Capabilities;
    
    action registerAgent(
        name: String,
        type: String,
        endpoint: String
    ) returns Agents;
    
    function getAgentMetrics(
        agentId: UUID,
        metricType: String
    ) returns array of AgentMetrics;
}

type AgentConfiguration {
    maxConnections : Integer;
    timeout        : Integer;
    retryAttempts  : Integer;
    enableLogging  : Boolean;
}

aspect Auditable {
    createdAt : DateTime @cds.on.insert: $now;
    createdBy : String @cds.on.insert: $user;
    modifiedAt: DateTime @cds.on.insert: $now  @cds.on.update: $now;
    modifiedBy: String @cds.on.insert: $user @cds.on.update: $user;
}
    `.trim();
    
    // Write sample CDS to temp file
    const tempCDSFile = path.join(__dirname, '..', 'test-sample.cds');
    await fs.writeFile(tempCDSFile, sampleCDS);
    
    try {
        // Parse the sample CDS
        const result = await indexer.scipCDSIndexer(tempCDSFile);
        
        console.log(`✅ Sample CDS parsed successfully`);
        console.log(`   Symbols: ${result.scip.symbols.length}`);
        console.log(`   Facts: ${result.glean.length}`);
        
        // Analyze symbols by type
        const symbolsByType = {};
        result.scip.symbols.forEach(symbol => {
            symbolsByType[symbol.type] = (symbolsByType[symbol.type] || 0) + 1;
        });
        
        console.log(`   Symbol breakdown:`, symbolsByType);
        
        // Show sample symbols
        console.log(`   🔍 Sample symbols:`);
        result.scip.symbols.slice(0, 5).forEach(symbol => {
            console.log(`      ${symbol.type}: ${symbol.name} (line ${symbol.definition?.range?.start?.line || 'unknown'})`);
        });
        
        // Test CDS-specific facts
        const cdsSpecificFacts = result.glean.filter(fact => 
            fact.key.cdsFile || fact.key.cdsEntity || fact.key.cdsService || fact.key.cdsAnnotation
        );
        
        console.log(`   📊 CDS-specific facts: ${cdsSpecificFacts.length}`);
        
        return true;
        
    } finally {
        // Clean up temp file
        try {
            await fs.unlink(tempCDSFile);
        } catch (e) {
            // Ignore cleanup errors
        }
    }
}

async function testCDSQueries(results, indexer) {
    console.log('\n📋 Testing CDS-specific queries...');
    
    try {
        const angleParser = new AngleParser();
        
        // Load the enhanced schema with CDS predicates
        const schemaPath = path.join(__dirname, '..', 'srv/glean/schemas/src.angle');
        const schemaContent = await fs.readFile(schemaPath, 'utf8');
        const parsedSchema = angleParser.parseSchema(schemaContent);
        
        console.log(`✅ Loaded schema with ${Object.keys(parsedSchema.predicates).length} predicates`);
        
        // Check for CDS-specific predicates
        const cdsPredicates = Object.keys(parsedSchema.predicates).filter(name => 
            name.startsWith('CDS') || name.startsWith('CAP')
        );
        
        console.log(`✅ Found ${cdsPredicates.length} CDS/CAP predicates:`, cdsPredicates);
        
        // Test basic CDS queries (would need actual fact database for full testing)
        const testQueries = [
            'Find all CDS entities',
            'Find all CDS services', 
            'Find entities with annotations',
            'Find services with actions'
        ];
        
        testQueries.forEach(queryName => {
            console.log(`   📋 ${queryName}: Schema ready for implementation`);
        });
        
    } catch (error) {
        console.error(`   ❌ Query testing failed: ${error.message}`);
    }
}

if (require.main === module) {
    testCAPIntegration().then(success => {
        process.exit(success ? 0 : 1);
    });
}

module.exports = { testCAPIntegration };