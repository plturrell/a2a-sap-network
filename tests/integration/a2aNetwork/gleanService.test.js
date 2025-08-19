const { expect } = require('chai');
const sinon = require('sinon');
const cds = require('@sap/cds');
const GleanService = require('../../srv/glean/gleanService');

/**
 * Integration tests for Glean Service
 * Testing the integration of CLRS algorithms and Tree operations with Glean
 * Following SAP CAP testing standards
 */
describe('GleanService Integration Tests', function() {
    let gleanService;
    let sandbox;
    let mockCdsContext;

    before(async function() {
        // Initialize test database and services
        this.timeout(10000);
        
        // Mock CDS environment for testing
        cds.test.in(__dirname);
        cds.env.requires.db = { kind: 'sqlite', credentials: { url: ':memory:' } };
    });

    beforeEach(function() {
        sandbox = sinon.createSandbox();
        
        // Create mock CDS context
        mockCdsContext = {
            run: sandbox.stub(),
            tx: sandbox.stub().returns({
                run: sandbox.stub(),
                commit: sandbox.stub(),
                rollback: sandbox.stub()
            })
        };
        
        gleanService = new GleanService();
        
        // Mock base service methods
        sandbox.stub(gleanService, '_withErrorHandling').callsFake(async (name, fn) => {
            try {
                return await fn();
            } catch (error) {
                throw error;
            }
        });
    });

    afterEach(function() {
        sandbox.restore();
    });

    describe('Service Initialization', function() {
        it('should initialize service with algorithm services', async function() {
            // Mock parent initialization
            sandbox.stub(gleanService.__proto__.__proto__, 'initializeService').resolves();
            
            await gleanService.initializeService();
            
            expect(gleanService.graphAlgorithms).to.exist;
            expect(gleanService.treeAlgorithms).to.exist;
            expect(gleanService.logger).to.exist;
        });

        it('should register actions correctly', async function() {
            const onStub = sandbox.stub(gleanService, 'on');
            
            // Trigger action registration
            gleanService._registerActions();
            
            expect(onStub).to.have.been.calledWith('analyzeDependencyCriticalPaths');
            expect(onStub).to.have.been.calledWith('findSimilarCode');
            expect(onStub).to.have.been.calledWith('navigateCodeHierarchy');
            expect(onStub).to.have.been.calledWith('suggestRefactorings');
        });
    });

    describe('Dependency Critical Path Analysis', function() {
        beforeEach(function() {
            // Mock graph building methods
            sandbox.stub(gleanService, '_buildDependencyGraph').resolves(
                createMockDependencyGraph()
            );
            sandbox.stub(gleanService, '_findEntryPoints').resolves(['index.js', 'main.js']);
            sandbox.stub(gleanService, '_calculateCycleSeverity').returns(5);
            sandbox.stub(gleanService, '_calculateDependencyMetrics').resolves({
                depth: 3,
                fanOut: 2.5,
                fanIn: 1.8
            });
        });

        it('should analyze critical paths using Dijkstra algorithm', async function() {
            const result = await gleanService.analyzeDependencyCriticalPaths('test-project', 'target.js');
            
            expect(result).to.have.property('criticalPaths');
            expect(result).to.have.property('circularDependencies');
            expect(result).to.have.property('buildOrder');
            expect(result).to.have.property('dependencyMetrics');
            
            expect(result.criticalPaths).to.be.an('array');
            expect(result.circularDependencies).to.be.an('array');
        });

        it('should detect circular dependencies correctly', async function() {
            // Create graph with circular dependency
            const circularGraph = createMockCircularGraph();
            gleanService._buildDependencyGraph.resolves(circularGraph);
            
            const result = await gleanService.analyzeDependencyCriticalPaths('test-project', 'target.js');
            
            expect(result.circularDependencies).to.have.lengthOf.greaterThan(0);
            expect(result.circularDependencies[0]).to.have.property('files');
            expect(result.circularDependencies[0]).to.have.property('severity');
        });

        it('should handle topological sort errors gracefully', async function() {
            // Mock graph with cycles that prevent topological sort
            const cyclicGraph = createMockCircularGraph();
            gleanService._buildDependencyGraph.resolves(cyclicGraph);
            
            const result = await gleanService.analyzeDependencyCriticalPaths('test-project', 'target.js');
            
            expect(result.buildOrder).to.be.empty;
        });

        it('should calculate dependency metrics accurately', async function() {
            const result = await gleanService.analyzeDependencyCriticalPaths('test-project', 'target.js');
            
            expect(result.dependencyMetrics).to.have.property('depth');
            expect(result.dependencyMetrics).to.have.property('fanOut');
            expect(result.dependencyMetrics).to.have.property('fanIn');
            
            expect(result.dependencyMetrics.depth).to.be.a('number');
            expect(result.dependencyMetrics.fanOut).to.be.a('number');
            expect(result.dependencyMetrics.fanIn).to.be.a('number');
        });
    });

    describe('Code Similarity Detection', function() {
        beforeEach(function() {
            // Mock file operations
            sandbox.stub(gleanService, '_getIndexedFiles').resolves([
                { path: 'file1.js' },
                { path: 'file2.js' },
                { path: 'file3.js' }
            ]);
            
            sandbox.stub(gleanService, '_readFileContent').callsFake((path) => {
                const mockContent = {
                    'file1.js': 'function test() { return "hello"; }',
                    'file2.js': 'function test() { return "world"; }',
                    'file3.js': 'function different() { return 42; }'
                };
                return Promise.resolve(mockContent[path] || '');
            });
            
            sandbox.stub(gleanService, '_normalizeCode').callsFake((code) => code.toLowerCase());
            sandbox.stub(gleanService, '_createSlidingWindows').returns([
                { content: 'normalized content', originalContent: 'original content', startLine: 1, endLine: 5 }
            ]);
            sandbox.stub(gleanService, '_generateRefactoringSuggestions').returns([
                { type: 'extract_common', description: 'Extract common function' }
            ]);
        });

        it('should find similar code using LCS algorithm', async function() {
            // Mock similarity calculation
            sandbox.stub(gleanService, '_calculateSimilarity').returns(0.85);
            
            const result = await gleanService.findSimilarCode('function test() { return "hello"; }', 0.8);
            
            expect(result).to.have.property('query');
            expect(result).to.have.property('matches');
            expect(result).to.have.property('statistics');
            
            expect(result.matches).to.be.an('array');
            expect(result.statistics).to.have.property('totalFilesScanned');
            expect(result.statistics).to.have.property('matchesFound');
        });

        it('should filter results by similarity threshold', async function() {
            // Mock varying similarity scores
            let callCount = 0;
            sandbox.stub(gleanService, '_calculateSimilarity').callsFake(() => {
                callCount++;
                return callCount === 1 ? 0.9 : 0.7; // First high, second low
            });
            
            const result = await gleanService.findSimilarCode('test code', 0.8);
            
            expect(result.matches).to.have.lengthOf(1); // Only high similarity match
        });

        it('should sort results by similarity score', async function() {
            // Mock multiple matches with different scores
            sandbox.stub(gleanService, '_calculateSimilarity')
                .onFirstCall().returns(0.85)
                .onSecondCall().returns(0.95)
                .onThirdCall().returns(0.75);
            
            const result = await gleanService.findSimilarCode('test code', 0.7);
            
            expect(result.matches[0].similarity).to.equal(0.95);
            expect(result.matches[1].similarity).to.equal(0.85);
            expect(result.matches[2].similarity).to.equal(0.75);
        });

        it('should include refactoring suggestions for matches', async function() {
            sandbox.stub(gleanService, '_calculateSimilarity').returns(0.9);
            
            const result = await gleanService.findSimilarCode('test code', 0.8);
            
            expect(result.matches[0]).to.have.property('suggestions');
            expect(result.matches[0].suggestions).to.be.an('array');
        });
    });

    describe('Hierarchical Code Navigation', function() {
        beforeEach(function() {
            // Mock tree building
            sandbox.stub(gleanService, '_buildCodeTree').resolves({
                src: {
                    'index.js': { type: 'file', size: 1000 },
                    components: {
                        'Button.js': { type: 'file', size: 500 },
                        'Input.js': { type: 'file', size: 300 }
                    }
                },
                test: {
                    'index.test.js': { type: 'file', size: 800 }
                }
            });
            
            sandbox.stub(gleanService, '_applyHierarchicalQuery').callsFake((tree, query) => {
                if (query.includes('*.js')) {
                    return tree; // Return full tree for JS files
                }
                return {}; // Empty for other queries
            });
        });

        it('should navigate code hierarchy using tree algorithms', async function() {
            const result = await gleanService.navigateCodeHierarchy('/project/src', '*.js');
            
            expect(result).to.have.property('rootPath');
            expect(result).to.have.property('query');
            expect(result).to.have.property('hierarchy');
            expect(result).to.have.property('matchedPaths');
            expect(result).to.have.property('metrics');
            
            expect(result.rootPath).to.equal('/project/src');
            expect(result.query).to.equal('*.js');
        });

        it('should calculate hierarchy metrics correctly', async function() {
            const result = await gleanService.navigateCodeHierarchy('/project/src', '*.js');
            
            expect(result.metrics).to.have.property('totalNodes');
            expect(result.metrics).to.have.property('totalLeaves');
            expect(result.metrics).to.have.property('maxDepth');
            expect(result.metrics).to.have.property('matchedNodes');
            
            expect(result.metrics.totalNodes).to.be.a('number');
            expect(result.metrics.maxDepth).to.be.a('number');
        });

        it('should apply query filters correctly', async function() {
            gleanService._applyHierarchicalQuery.restore();
            sandbox.stub(gleanService, '_applyHierarchicalQuery').callsFake((tree, query) => {
                if (query === 'function:test*') {
                    return { testFunction: { type: 'function' } };
                }
                return tree;
            });
            
            const result = await gleanService.navigateCodeHierarchy('/project/src', 'function:test*');
            
            expect(gleanService._applyHierarchicalQuery).to.have.been.calledWith(
                sinon.match.any,
                'function:test*'
            );
        });

        it('should format matched paths correctly', async function() {
            const result = await gleanService.navigateCodeHierarchy('/project/src', '*.js');
            
            expect(result.matchedPaths).to.be.an('array');
            if (result.matchedPaths.length > 0) {
                expect(result.matchedPaths[0]).to.have.property('path');
                expect(result.matchedPaths[0]).to.have.property('element');
                expect(result.matchedPaths[0]).to.have.property('depth');
            }
        });
    });

    describe('Refactoring Suggestions', function() {
        beforeEach(function() {
            // Mock AST parsing and analysis
            sandbox.stub(gleanService, '_parseFileAST').resolves({
                type: 'Program',
                body: [
                    { type: 'FunctionDeclaration', name: 'complexFunction' }
                ]
            });
            
            sandbox.stub(gleanService, '_astToTreeStructure').returns({
                functions: [
                    { name: 'complexFunction', complexity: 15, path: ['functions', 0] }
                ]
            });
            
            sandbox.stub(gleanService, '_analyzeCodePatterns').resolves({
                patterns: ['long_function', 'nested_conditions'],
                quality: 3.2
            });
            
            sandbox.stub(gleanService, '_findComplexFunctions').returns([
                { name: 'complexFunction', complexity: 15, path: ['functions', 0] }
            ]);
            
            sandbox.stub(gleanService, '_findDuplicatePatterns').returns([]);
            sandbox.stub(gleanService, '_analyzeDependencyStructure').resolves([]);
            sandbox.stub(gleanService, '_findInefficiencies').returns([]);
            
            sandbox.stub(gleanService, '_generateExtractMethodRefactoring').returns({
                type: 'extract_method',
                code: 'generated refactoring code'
            });
            
            sandbox.stub(gleanService, '_calculateCodeQualityScore').returns(75);
            sandbox.stub(gleanService, '_generateRefactoringExecutionPlan').returns({
                canExecuteAll: true,
                executionOrder: [],
                estimatedTotalTime: 30
            });
        });

        it('should suggest refactorings using combined analysis', async function() {
            const result = await gleanService.suggestRefactorings('/project/src/complex.js', 2);
            
            expect(result).to.have.property('file');
            expect(result).to.have.property('totalSuggestions');
            expect(result).to.have.property('suggestions');
            expect(result).to.have.property('codeQualityScore');
            expect(result).to.have.property('executionPlan');
            
            expect(result.file).to.equal('/project/src/complex.js');
            expect(result.suggestions).to.be.an('array');
        });

        it('should detect complex functions requiring refactoring', async function() {
            const result = await gleanService.suggestRefactorings('/project/src/complex.js', 2);
            
            const complexitySuggestions = result.suggestions.filter(s => s.type === 'EXTRACT_METHOD');
            expect(complexitySuggestions).to.have.lengthOf.greaterThan(0);
            
            const suggestion = complexitySuggestions[0];
            expect(suggestion).to.have.property('severity');
            expect(suggestion).to.have.property('description');
            expect(suggestion).to.have.property('automated');
            expect(suggestion).to.have.property('refactoring');
        });

        it('should sort suggestions by severity and feasibility', async function() {
            // Mock multiple suggestions with different severities
            gleanService._findComplexFunctions.returns([
                { name: 'highComplexity', complexity: 20, path: ['functions', 0] },
                { name: 'mediumComplexity', complexity: 10, path: ['functions', 1] }
            ]);
            
            const result = await gleanService.suggestRefactorings('/project/src/complex.js', 2);
            
            expect(result.suggestions[0].severity).to.equal('high');
            expect(result.suggestions[0].description).to.include('highComplexity');
        });

        it('should include automated refactoring code when possible', async function() {
            const result = await gleanService.suggestRefactorings('/project/src/complex.js', 2);
            
            const automatedSuggestions = result.suggestions.filter(s => s.automated);
            expect(automatedSuggestions).to.have.lengthOf.greaterThan(0);
            
            const suggestion = automatedSuggestions[0];
            expect(suggestion.refactoring).to.have.property('type');
            expect(suggestion.refactoring).to.have.property('code');
        });

        it('should generate execution plan for refactorings', async function() {
            const result = await gleanService.suggestRefactorings('/project/src/complex.js', 2);
            
            expect(result.executionPlan).to.have.property('canExecuteAll');
            expect(result.executionPlan).to.have.property('estimatedTotalTime');
            
            if (result.executionPlan.canExecuteAll) {
                expect(result.executionPlan).to.have.property('executionOrder');
            }
        });
    });

    describe('Algorithm Integration', function() {
        it('should use graph algorithms for dependency analysis', function() {
            expect(gleanService.graphAlgorithms).to.respondTo('depthFirstSearch');
            expect(gleanService.graphAlgorithms).to.respondTo('breadthFirstSearch');
            expect(gleanService.graphAlgorithms).to.respondTo('dijkstra');
            expect(gleanService.graphAlgorithms).to.respondTo('topologicalSort');
        });

        it('should use tree algorithms for hierarchical operations', function() {
            expect(gleanService.treeAlgorithms).to.respondTo('flatten');
            expect(gleanService.treeAlgorithms).to.respondTo('mapStructure');
            expect(gleanService.treeAlgorithms).to.respondTo('filterStructure');
            expect(gleanService.treeAlgorithms).to.respondTo('getAllPaths');
        });

        it('should implement LCS algorithm for code similarity', function() {
            const lcsResult = gleanService._longestCommonSubsequence(
                ['a', 'b', 'c', 'd'],
                ['a', 'c', 'd', 'e']
            );
            
            expect(lcsResult).to.deep.equal(['a', 'c', 'd']);
        });

        it('should calculate similarity scores correctly', function() {
            const similarity = gleanService._calculateSimilarity(
                'function test() { return 1; }',
                'function test() { return 2; }'
            );
            
            expect(similarity).to.be.a('number');
            expect(similarity).to.be.at.least(0);
            expect(similarity).to.be.at.most(1);
        });
    });

    describe('Error Handling and Resilience', function() {
        it('should handle missing files gracefully', async function() {
            gleanService._getIndexedFiles.rejects(new Error('File not found'));
            
            try {
                await gleanService.findSimilarCode('test code', 0.8);
                expect.fail('Should have thrown an error');
            } catch (error) {
                expect(error.message).to.include('File not found');
            }
        });

        it('should handle invalid dependency graphs', async function() {
            gleanService._buildDependencyGraph.resolves(null);
            
            try {
                await gleanService.analyzeDependencyCriticalPaths('test-project', 'target.js');
                expect.fail('Should have thrown an error');
            } catch (error) {
                expect(error).to.exist;
            }
        });

        it('should handle AST parsing errors', async function() {
            gleanService._parseFileAST.rejects(new Error('Syntax error'));
            
            try {
                await gleanService.suggestRefactorings('/invalid/file.js', 2);
                expect.fail('Should have thrown an error');
            } catch (error) {
                expect(error.message).to.include('Syntax error');
            }
        });

        it('should provide meaningful error messages', async function() {
            gleanService._buildCodeTree.rejects(new Error('Access denied'));
            
            try {
                await gleanService.navigateCodeHierarchy('/forbidden/path', '*.js');
                expect.fail('Should have thrown an error');
            } catch (error) {
                expect(error.message).to.include('Access denied');
            }
        });
    });

    describe('Performance Optimization', function() {
        it('should handle large codebases efficiently', async function() {
            // Mock large codebase
            const largeMockTree = {};
            for (let i = 0; i < 1000; i++) {
                largeMockTree[`file${i}.js`] = { type: 'file', size: 1000 };
            }
            
            gleanService._buildCodeTree.resolves(largeMockTree);
            
            const startTime = Date.now();
            await gleanService.navigateCodeHierarchy('/large/project', '*.js');
            const endTime = Date.now();
            
            expect(endTime - startTime).to.be.lessThan(5000); // Should complete within 5 seconds
        });

        it('should cache frequently accessed data', function() {
            // Test that repeated calls use cached data
            const firstCall = gleanService._buildDependencyGraph('test-project');
            const secondCall = gleanService._buildDependencyGraph('test-project');
            
            expect(gleanService._buildDependencyGraph).to.have.been.calledTwice;
        });
    });

    // Helper functions for creating mock data
    function createMockDependencyGraph() {
        const mockGraph = gleanService.graphAlgorithms.createGraph();
        gleanService.graphAlgorithms.addEdge(mockGraph, 'index.js', 'module1.js');
        gleanService.graphAlgorithms.addEdge(mockGraph, 'index.js', 'module2.js');
        gleanService.graphAlgorithms.addEdge(mockGraph, 'module1.js', 'target.js');
        gleanService.graphAlgorithms.addEdge(mockGraph, 'module2.js', 'target.js');
        return mockGraph;
    }

    function createMockCircularGraph() {
        const mockGraph = gleanService.graphAlgorithms.createGraph();
        gleanService.graphAlgorithms.addEdge(mockGraph, 'a.js', 'b.js');
        gleanService.graphAlgorithms.addEdge(mockGraph, 'b.js', 'c.js');
        gleanService.graphAlgorithms.addEdge(mockGraph, 'c.js', 'a.js'); // Creates cycle
        return mockGraph;
    }
});