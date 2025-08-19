#!/usr/bin/env node

/**
 * A2A Network - Glean Integration Test
 * Validates that SCIP indexing, fact transformation, and Angle queries work correctly
 */

const { AngleParser, AngleQueryExecutor } = require('../../srv/glean/angleParser');
const SCIPIndexer = require('../../srv/glean/scipIndexer');
const GleanFactTransformer = require('../../srv/glean/gleanFactTransformer');
const fs = require('fs').promises;
const path = require('path');

class GleanIntegrationTest {
    constructor() {
        this.angleParser = new AngleParser();
        this.scipIndexer = new SCIPIndexer(process.cwd());
        this.factTransformer = new GleanFactTransformer();
        this.testResults = [];
        this.verbose = process.argv.includes('--verbose') || process.argv.includes('-v');
    }

    log(message, level = 'info') {
        const timestamp = new Date().toISOString();
        const prefix = {
            'info': '📋',
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'test': '🧪'
        }[level] || '📋';
        
        console.log(`${prefix} [${timestamp}] ${message}`);
    }

    async runTests() {
        this.log('🚀 Starting Glean Integration Test - Phase 1 Validation', 'info');
        
        try {
            // Test 1: Parse Angle Schema
            await this.testAngleSchemaParser();
            
            // Test 2: Generate Test Facts
            await this.testFactGeneration();
            
            // Test 3: Execute Angle Queries
            await this.testAngleQueries();
            
            // Test 4: Real SCIP Integration
            await this.testSCIPIntegration();
            
            // Test 5: End-to-End Workflow
            await this.testEndToEndWorkflow();

            this.printSummary();
            
            const passed = this.testResults.filter(r => r.status === 'PASSED').length;
            const total = this.testResults.length;
            
            if (passed === total) {
                this.log(`🎉 All ${total} tests passed! Phase 1 integration is working correctly.`, 'success');
                process.exit(0);
            } else {
                this.log(`❌ ${total - passed} out of ${total} tests failed.`, 'error');
                process.exit(1);
            }
            
        } catch (error) {
            this.log(`Integration test failed: ${error.message}`, 'error');
            if (this.verbose) {
                console.error(error.stack);
            }
            process.exit(1);
        }
    }

    async testAngleSchemaParser() {
        this.log('Testing Angle Schema Parser...', 'test');
        
        try {
            // Load a simple test schema first
            const schemaPath = path.join(process.cwd(), 'test/integration/simple_test_schema.angle');
            const schemaContent = await fs.readFile(schemaPath, 'utf8');
            
            // Parse the schema
            const parsedSchema = this.angleParser.parseSchema(schemaContent);
            
            if (this.verbose) {
                this.log(`Parsed schema: ${JSON.stringify(parsedSchema, null, 2)}`, 'info');
            }
            
            // Validate schema structure
            this.assert(parsedSchema.name === 'test', 'Schema name should be "test"');
            this.assert(parsedSchema.version === 1, 'Schema version should be 1');
            this.assert(Object.keys(parsedSchema.predicates).length > 0, 'Schema should have predicates');
            
            // Check for key predicates
            const expectedPredicates = ['File', 'Function'];
            expectedPredicates.forEach(pred => {
                this.assert(parsedSchema.predicates[pred], `Schema should define ${pred} predicate`);
            });
            
            this.addTestResult('Angle Schema Parser', 'PASSED', 'Schema parsed successfully');
            
        } catch (error) {
            this.addTestResult('Angle Schema Parser', 'FAILED', error.message);
            throw error;
        }
    }

    async testFactGeneration() {
        this.log('Testing Fact Generation...', 'test');
        
        try {
            // Create mock SCIP data
            const mockSCIPIndex = {
                metadata: {
                    version: '0.3.0',
                    project_root: 'file:///test'
                },
                documents: [
                    {
                        relative_path: 'test/sample.js',
                        symbols: [
                            {
                                symbol: 'local 0',
                                definition: {
                                    range: {
                                        start: { line: 0, character: 0 },
                                        end: { line: 0, character: 20 }
                                    },
                                    syntax_kind: 'Function'
                                }
                            }
                        ],
                        occurrences: [
                            {
                                range: {
                                    start: { line: 0, character: 0 },
                                    end: { line: 0, character: 20 }
                                },
                                symbol: 'local 0',
                                symbol_roles: ['Definition']
                            }
                        ]
                    }
                ],
                external_symbols: []
            };
            
            // Transform to Glean facts
            const gleanFacts = this.factTransformer.transformSCIPToGlean(mockSCIPIndex);
            
            // Validate fact structure
            this.assert(gleanFacts['src.File'], 'Should generate File facts');
            this.assert(gleanFacts['src.Symbol'], 'Should generate Symbol facts');
            this.assert(gleanFacts['src.File'].length > 0, 'Should have at least one File fact');
            
            // Validate fact format
            const fileFact = gleanFacts['src.File'][0];
            this.assert(fileFact.id, 'File fact should have ID');
            this.assert(fileFact.key, 'File fact should have key');
            this.assert(fileFact.value, 'File fact should have value');
            this.assert(fileFact.value.file, 'File fact should have file path');
            this.assert(fileFact.value.language, 'File fact should have language');
            
            this.addTestResult('Fact Generation', 'PASSED', `Generated ${Object.values(gleanFacts).reduce((sum, facts) => sum + facts.length, 0)} facts`);
            
        } catch (error) {
            this.addTestResult('Fact Generation', 'FAILED', error.message);
            throw error;
        }
    }

    async testAngleQueries() {
        this.log('Testing Angle Query Execution...', 'test');
        
        try {
            // Create test fact database
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
                ],
                'src.Function': [
                    {
                        id: 'func1',
                        key: { file: 'test.js', name: 'testFunction' },
                        value: {
                            file: 'test.js',
                            name: 'testFunction',
                            complexity: 15,
                            async: false,
                            exported: true
                        }
                    }
                ],
                'src.SecurityIssue': [
                    {
                        id: 'sec1',
                        key: { file: 'test.js', line: 10 },
                        value: {
                            file: 'test.js',
                            line: 10,
                            issue_type: 'unsafe_eval',
                            severity: 'high',
                            description: 'Use of eval() detected'
                        }
                    }
                ]
            };
            
            // Test queries
            await this.runTestQuery(
                'Find all JavaScript files',
                'query FindJSFiles() : [File] = src.File { file, value.language = "javascript" }',
                factDb,
                (results) => {
                    this.assert(results.length === 1, 'Should find 1 JavaScript file');
                    this.assert(results[0].value.file === 'test.js', 'Should find test.js');
                }
            );
            
            await this.runTestQuery(
                'Find complex functions',
                'query FindComplexFunctions() : [Function] = src.Function { file, name, value.complexity } where value.complexity > 10',
                factDb,
                (results) => {
                    this.assert(results.length === 1, 'Should find 1 complex function');
                    this.assert(results[0].value.name === 'testFunction', 'Should find testFunction');
                }
            );
            
            await this.runTestQuery(
                'Find high severity security issues',
                'query FindHighSecurityIssues() : [SecurityIssue] = src.SecurityIssue { file, line, value.issue_type, value.severity = "high" }',
                factDb,
                (results) => {
                    this.assert(results.length === 1, 'Should find 1 high severity issue');
                    this.assert(results[0].value.issue_type === 'unsafe_eval', 'Should find unsafe_eval issue');
                }
            );
            
            this.addTestResult('Angle Queries', 'PASSED', 'All test queries executed successfully');
            
        } catch (error) {
            this.addTestResult('Angle Queries', 'FAILED', error.message);
            throw error;
        }
    }

    async runTestQuery(testName, queryText, factDb, validator) {
        this.log(`  Testing query: ${testName}`, 'test');
        
        try {
            const parsedQuery = this.angleParser.parseQuery(queryText);
            const executor = new AngleQueryExecutor(factDb, this.angleParser.predicateDefinitions);
            const results = executor.execute(parsedQuery);
            
            if (this.verbose) {
                this.log(`    Query results: ${JSON.stringify(results, null, 2)}`, 'info');
            }
            
            validator(results);
            this.log(`  ✅ ${testName} passed`, 'success');
            
        } catch (error) {
            this.log(`  ❌ ${testName} failed: ${error.message}`, 'error');
            throw error;
        }
    }

    async testSCIPIntegration() {
        this.log('Testing SCIP Integration...', 'test');
        
        try {
            // Initialize SCIP indexer
            await this.scipIndexer.initialize();
            // Create a test file with security issues for detection
            const testFileContent = `
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
            
            const testFilePath = path.join(process.cwd(), 'test_temp_file.js');
            await fs.writeFile(testFilePath, testFileContent);
            
            try {
                // Index just the test file by creating a temporary indexer
                const scipDoc = await this.scipIndexer.runSCIPIndexer(
                    { command: 'scip-typescript', extensions: ['.js'] },
                    testFilePath
                );
                
                // Validate SCIP results
                this.assert(scipDoc.scip, 'Should generate SCIP document');
                this.assert(scipDoc.scip.symbols && scipDoc.scip.symbols.length > 0, 'Should extract symbols');
                this.assert(scipDoc.glean && scipDoc.glean.length > 0, 'Should generate Glean facts');
                
                // Transform SCIP results through the fact transformer to add security analysis
                const scipIndex = {
                    documents: [scipDoc.scip],
                    external_symbols: [],
                    symbol_roles: []
                };
                
                const gleanFacts = this.factTransformer.transformSCIPToGlean(scipIndex);
                const totalFacts = Object.values(gleanFacts).reduce((sum, facts) => sum + facts.length, 0);
                this.assert(totalFacts > 0, 'Should generate facts from SCIP data');
                
                // Verify security analysis was performed
                const securityFacts = gleanFacts['src.SecurityIssue'] || [];
                this.assert(securityFacts.length > 0, 'Should detect security issues');
                
                this.addTestResult('SCIP Integration', 'PASSED', `Generated ${totalFacts} facts including ${securityFacts.length} security issues`);
                
            } finally {
                // Clean up test file
                try {
                    await fs.unlink(testFilePath);
                } catch (e) {
                    // Ignore cleanup errors
                }
            }
            
        } catch (error) {
            this.addTestResult('SCIP Integration', 'FAILED', error.message);
            throw error;
        }
    }

    async testEndToEndWorkflow() {
        this.log('Testing End-to-End Workflow...', 'test');
        
        try {
            // 1. Create test code
            const testCode = `
function vulnerableFunction() {
    eval("console.log('This is dangerous')");
    return true;
}

class SecureClass {
    validate(input) {
        return input.length > 0;
    }
}
            `.trim();
            
            const testFile = path.join(process.cwd(), 'e2e_test.js');
            await fs.writeFile(testFile, testCode);
            
            try {
                // 2. Index with SCIP
                const scipResults = await this.scipIndexer.indexProject(['javascript']);
                
                // 3. Transform to Glean facts
                const gleanFacts = this.factTransformer.transformSCIPToGlean(scipResults.scipIndex);
                
                // 4. Execute security analysis query
                const securityQuery = `
                    query FindSecurityIssues() : [SecurityIssue] = 
                        src.SecurityIssue { file, line, issue_type, severity }
                `;
                
                const parsedQuery = this.angleParser.parseQuery(securityQuery);
                const executor = new AngleQueryExecutor(gleanFacts, this.angleParser.predicateDefinitions);
                const securityIssues = executor.execute(parsedQuery);
                
                // 5. Validate end-to-end results
                this.assert(securityIssues.length > 0, 'Should detect security issues');
                
                const evalIssue = securityIssues.find(issue => 
                    issue.value && issue.value.issue_type === 'unsafe_eval'
                );
                this.assert(evalIssue, 'Should detect eval() usage as security issue');
                
                this.addTestResult('End-to-End Workflow', 'PASSED', 'Complete workflow from code to security analysis working');
                
            } finally {
                // Clean up
                try {
                    await fs.unlink(testFile);
                } catch (e) {
                    // Ignore cleanup errors
                }
            }
            
        } catch (error) {
            this.addTestResult('End-to-End Workflow', 'FAILED', error.message);
            throw error;
        }
    }

    assert(condition, message) {
        if (!condition) {
            throw new Error(`Assertion failed: ${message}`);
        }
    }

    addTestResult(testName, status, details) {
        this.testResults.push({
            name: testName,
            status,
            details,
            timestamp: new Date().toISOString()
        });
    }

    printSummary() {
        this.log('\n📊 Test Summary:', 'info');
        this.log('='.repeat(50), 'info');
        
        this.testResults.forEach(result => {
            const icon = result.status === 'PASSED' ? '✅' : '❌';
            this.log(`${icon} ${result.name}: ${result.status}`, result.status === 'PASSED' ? 'success' : 'error');
            if (this.verbose && result.details) {
                this.log(`   ${result.details}`, 'info');
            }
        });
        
        const passed = this.testResults.filter(r => r.status === 'PASSED').length;
        const total = this.testResults.length;
        
        this.log('='.repeat(50), 'info');
        this.log(`Total: ${passed}/${total} tests passed`, passed === total ? 'success' : 'error');
    }
}

// Run tests if called directly
if (require.main === module) {
    const test = new GleanIntegrationTest();
    test.runTests().catch(error => {
        console.error('❌ Integration test failed:', error.message);
        process.exit(1);
    });
}

module.exports = GleanIntegrationTest;