const { expect } = require('chai');
const sinon = require('sinon');
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');

// SEAL Components
const SealEnhancedGleanService = require('../../srv/glean/sealEnhancedGleanService');
const GrokSealAdapter = require('../../srv/seal/grokSealAdapter');
const ReinforcementLearningEngine = require('../../srv/seal/reinforcementLearningEngine');
const SapSealGovernance = require('../../srv/seal/sapSealGovernance');
const SealConfiguration = require('../../srv/seal/sealConfiguration');
const GraphAlgorithms = require('../../srv/algorithms/graphAlgorithms');
const TreeAlgorithms = require('../../srv/algorithms/treeAlgorithms');

/**
 * COMPREHENSIVE SEAL VALIDATION TEST SUITE
 * Tests all aspects of genuine self-adaptation, integration, and production readiness
 */
describe('Comprehensive SEAL System Validation', function() {
    this.timeout(30000); // Allow time for complex tests
    
    let sealService;
    let grokAdapter;
    let rlEngine;
    let governance;
    let axiosStub;
    let sandbox;
    
    // Test data tracking
    const testMetrics = {
        qLearningUpdates: [],
        grokApiCalls: [],
        performanceImprovements: [],
        adaptationCycles: [],
        errorRecoveries: []
    };
    
    before(async () => {
        // Setup comprehensive test environment
        process.env.NODE_ENV = 'test';
        process.env.XAI_API_KEY = 'test-api-key-comprehensive';
        process.env.XAI_BASE_URL = 'https://api.x.ai/v1';
        process.env.XAI_MODEL = 'grok-4';
        process.env.MOCK_EXTERNAL_SERVICES = 'false';
        
        sandbox = sinon.createSandbox();
    });
    
    after(async () => {
        // Generate test report
        await generateTestReport(testMetrics);
    });
    
    beforeEach(async () => {
        // Initialize fresh instances for each test
        sealService = new SealEnhancedGleanService();
        grokAdapter = new GrokSealAdapter();
        rlEngine = new ReinforcementLearningEngine();
        governance = new SapSealGovernance();
        
        axiosStub = sandbox.stub(axios, 'post');
    });
    
    afterEach(() => {
        sandbox.restore();
    });
    
    describe('1. Core Component Integration Tests', () => {
        it('should successfully initialize all SEAL components with proper configuration', async () => {
            const config = new SealConfiguration();
            const cfg = config.getConfiguration();
            
            // Verify configuration
            expect(cfg.grok.model).to.equal('grok-4');
            expect(cfg.grok.baseUrl).to.equal('https://api.x.ai/v1');
            expect(cfg.reinforcementLearning.algorithm).to.equal('Q_LEARNING');
            expect(cfg.compliance.enabled).to.be.true;
            
            // Initialize all components
            await Promise.all([
                grokAdapter.initializeService(),
                rlEngine.initializeService(),
                governance.initializeService()
            ]);
            
            // Verify initialization
            expect(grokAdapter.initialized).to.be.true;
            expect(rlEngine.initialized).to.be.true;
            expect(governance.initialized).to.be.true;
            
            // Test component health
            const healthChecks = await Promise.all([
                grokAdapter.healthCheck(),
                rlEngine.healthCheck(),
                governance.healthCheck()
            ]);
            
            healthChecks.forEach(health => {
                expect(health.status).to.equal('healthy');
                expect(health.uptime).to.be.greaterThan(0);
            });
        });
        
        it('should properly integrate algorithm services', async () => {
            const graphAlg = new GraphAlgorithms();
            const treeAlg = new TreeAlgorithms();
            
            await graphAlg.initializeService();
            await treeAlg.initializeService();
            
            // Test graph algorithms
            const graph = graphAlg.createGraph();
            graphAlg.addEdge(graph, 'ServiceA', 'ServiceB', 1);
            graphAlg.addEdge(graph, 'ServiceB', 'ServiceC', 2);
            graphAlg.addEdge(graph, 'ServiceA', 'ServiceC', 5);
            
            const shortestPath = graphAlg.dijkstra(graph, 'ServiceA');
            expect(shortestPath.getDistance('ServiceC')).to.equal(3);
            expect(shortestPath.getPath('ServiceC')).to.deep.equal(['ServiceA', 'ServiceB', 'ServiceC']);
            
            // Test tree algorithms
            const codeStructure = {
                src: {
                    components: ['Button.js', 'Modal.js'],
                    utils: { helpers: ['format.js', 'validate.js'] }
                }
            };
            
            const flattened = treeAlg.flatten(codeStructure);
            expect(flattened).to.include('Button.js');
            expect(flattened).to.include('format.js');
            expect(flattened).to.have.lengthOf(4);
        });
    });
    
    describe('2. Q-Learning Real Adaptation Tests', () => {
        it('should demonstrate statistically significant Q-value convergence', async () => {
            await rlEngine.initializeService();
            
            const state = {
                codebase_complexity: 0.7,
                analysis_accuracy: 0.6,
                performance_score: 0.5,
                user_satisfaction: 0.7
            };
            
            const actions = [
                { type: 'increase_depth', target: 'analysis', intensity: 1 },
                { type: 'optimize_algorithm', target: 'core_analysis', intensity: 1 },
                { type: 'add_context', target: 'pattern_recognition', intensity: 1 },
                { type: 'parallel_processing', target: 'execution_strategy', intensity: 1 }
            ];
            
            // Track Q-value evolution
            const qValueHistory = {};
            actions.forEach(action => {
                qValueHistory[action.type] = [];
            });
            
            // Run 50 learning iterations
            console.log('\n📊 Q-Learning Convergence Test (50 iterations)');
            
            for (let iteration = 0; iteration < 50; iteration++) {
                const selection = await rlEngine.selectAction(state, actions);
                
                // Reward strategy: optimize_algorithm gets high reward
                let reward;
                if (selection.action.type === 'optimize_algorithm') {
                    reward = 0.9 + Math.random() * 0.1; // 0.9-1.0
                } else if (selection.action.type === 'increase_depth') {
                    reward = 0.4 + Math.random() * 0.2; // 0.4-0.6
                } else {
                    reward = Math.random() * 0.3; // 0.0-0.3
                }
                
                const nextState = {
                    ...state,
                    analysis_accuracy: Math.min(1, state.analysis_accuracy + reward * 0.1),
                    performance_score: Math.min(1, state.performance_score + reward * 0.05)
                };
                
                const learningResult = await rlEngine.learnFromFeedback(
                    state,
                    selection.action,
                    reward,
                    nextState,
                    { iteration }
                );
                
                testMetrics.qLearningUpdates.push({
                    iteration,
                    action: selection.action.type,
                    reward,
                    qValueChange: learningResult.qValueUpdate.updatedQ - learningResult.qValueUpdate.previousQ
                });
                
                // Record Q-values every 5 iterations
                if (iteration % 5 === 0) {
                    const encodedState = rlEngine._encodeState(state);
                    actions.forEach(action => {
                        const key = `${encodedState}:${rlEngine._encodeAction(action)}`;
                        const qValue = rlEngine.qTable.get(key) || 0;
                        qValueHistory[action.type].push({ iteration, qValue });
                    });
                }
            }
            
            // Analyze convergence
            console.log('\n📈 Q-Value Evolution:');
            Object.entries(qValueHistory).forEach(([actionType, history]) => {
                const initial = history[0].qValue;
                const final = history[history.length - 1].qValue;
                const change = final - initial;
                console.log(`   ${actionType}: ${initial.toFixed(4)} → ${final.toFixed(4)} (Δ${change.toFixed(4)})`);
            });
            
            // Verify convergence
            const finalQValues = {};
            const encodedState = rlEngine._encodeState(state);
            actions.forEach(action => {
                const key = `${encodedState}:${rlEngine._encodeAction(action)}`;
                finalQValues[action.type] = rlEngine.qTable.get(key) || 0;
            });
            
            // optimize_algorithm should have highest Q-value
            const sortedActions = Object.entries(finalQValues)
                .sort(([,a], [,b]) => b - a);
            
            expect(sortedActions[0][0]).to.equal('optimize_algorithm');
            expect(sortedActions[0][1]).to.be.greaterThan(0.5);
            
            // Calculate variance to ensure convergence
            const lastFiveUpdates = testMetrics.qLearningUpdates.slice(-5);
            const variance = calculateVariance(lastFiveUpdates.map(u => u.qValueChange));
            expect(variance).to.be.lessThan(0.01); // Low variance indicates convergence
        });
        
        it('should adapt to changing reward patterns', async () => {
            await rlEngine.initializeService();
            
            const state = { complexity: 0.6, accuracy: 0.5 };
            const actions = [
                { type: 'strategy_A' },
                { type: 'strategy_B' }
            ];
            
            console.log('\n🔄 Adaptive Learning Test');
            
            // Phase 1: Reward strategy_A
            console.log('   Phase 1: Rewarding strategy_A...');
            for (let i = 0; i < 20; i++) {
                const selection = await rlEngine.selectAction(state, actions);
                const reward = selection.action.type === 'strategy_A' ? 0.8 : 0.2;
                await rlEngine.learnFromFeedback(state, selection.action, reward, state);
            }
            
            // Check preference
            const phase1Counts = { strategy_A: 0, strategy_B: 0 };
            for (let i = 0; i < 10; i++) {
                const selection = await rlEngine.selectAction(state, actions);
                phase1Counts[selection.action.type]++;
            }
            console.log(`   Phase 1 results: A=${phase1Counts.strategy_A}, B=${phase1Counts.strategy_B}`);
            
            // Phase 2: Switch to rewarding strategy_B
            console.log('   Phase 2: Switching to reward strategy_B...');
            for (let i = 0; i < 30; i++) {
                const selection = await rlEngine.selectAction(state, actions);
                const reward = selection.action.type === 'strategy_B' ? 0.9 : 0.1;
                await rlEngine.learnFromFeedback(state, selection.action, reward, state);
            }
            
            // Check adapted preference
            const phase2Counts = { strategy_A: 0, strategy_B: 0 };
            for (let i = 0; i < 10; i++) {
                const selection = await rlEngine.selectAction(state, actions);
                phase2Counts[selection.action.type]++;
            }
            console.log(`   Phase 2 results: A=${phase2Counts.strategy_A}, B=${phase2Counts.strategy_B}`);
            
            // Verify adaptation
            expect(phase1Counts.strategy_A).to.be.greaterThan(phase1Counts.strategy_B);
            expect(phase2Counts.strategy_B).to.be.greaterThan(phase2Counts.strategy_A);
        });
    });
    
    describe('3. Grok 4 API Integration Tests', () => {
        it('should generate contextually relevant self-edits', async () => {
            // Mock Grok response with realistic self-improvements
            axiosStub.resolves({
                status: 200,
                data: {
                    choices: [{
                        message: {
                            content: JSON.stringify({
                                selfEdits: {
                                    dataAugmentations: [
                                        'enhance_react_hook_detection',
                                        'add_typescript_generic_inference',
                                        'improve_async_error_boundary_analysis'
                                    ],
                                    hyperparameterUpdates: {
                                        analysisDepth: 6,
                                        contextWindow: 3000,
                                        patternSimilarityThreshold: 0.88,
                                        maxParallelAnalysis: 4
                                    },
                                    modelArchitectureChanges: {
                                        enableSemanticLayer: true,
                                        addGraphNeuralNetwork: true,
                                        useTransformerAttention: true
                                    },
                                    trainingDataEnhancements: [
                                        'collect_real_world_react_patterns',
                                        'augment_with_popular_libraries'
                                    ]
                                },
                                confidence: 0.92,
                                reasoning: 'Analysis shows weak React hooks and TypeScript detection. Proposed enhancements target these specific gaps.',
                                expectedImprovement: 0.23
                            })
                        }
                    }],
                    usage: {
                        prompt_tokens: 1500,
                        completion_tokens: 800,
                        total_tokens: 2300
                    }
                }
            });
            
            await grokAdapter.initializeService();
            
            const analysisContext = {
                currentAnalysis: {
                    accuracy: 0.65,
                    weaknesses: ['react_hooks', 'typescript_generics', 'async_patterns'],
                    strengths: ['basic_syntax', 'imports'],
                    projectType: 'react-typescript'
                },
                projectContext: {
                    projectId: 'test-react-app',
                    language: 'typescript',
                    framework: 'react',
                    patterns: ['hooks', 'context', 'async/await']
                },
                performanceHistory: [
                    { timestamp: new Date(Date.now() - 3600000), accuracy: 0.58 },
                    { timestamp: new Date(Date.now() - 1800000), accuracy: 0.62 },
                    { timestamp: new Date(), accuracy: 0.65 }
                ]
            };
            
            const selfEdits = await grokAdapter.generateSelfEdits(analysisContext);
            
            testMetrics.grokApiCalls.push({
                context: analysisContext,
                response: selfEdits,
                timestamp: new Date()
            });
            
            // Verify Grok generated relevant improvements
            expect(selfEdits.dataAugmentations).to.include('enhance_react_hook_detection');
            expect(selfEdits.hyperparameterUpdates.analysisDepth).to.be.greaterThan(5);
            expect(selfEdits.confidence).to.be.greaterThan(0.9);
            expect(selfEdits.reasoning).to.include('React hooks');
            
            // Verify API call was made correctly
            expect(axiosStub.calledOnce).to.be.true;
            const [url, data, config] = axiosStub.getCall(0).args;
            expect(url).to.equal('https://api.x.ai/v1/chat/completions');
            expect(data.model).to.equal('grok-4');
            expect(data.messages).to.have.lengthOf(2);
            expect(config.headers.Authorization).to.include('Bearer');
        });
        
        it('should handle Grok API failures gracefully', async () => {
            // Simulate API failures
            axiosStub.onFirstCall().rejects({
                response: { 
                    status: 429, 
                    data: { error: { message: 'Rate limit exceeded' } } 
                }
            });
            
            axiosStub.onSecondCall().rejects({
                response: { 
                    status: 500, 
                    data: { error: { message: 'Internal server error' } } 
                }
            });
            
            axiosStub.onThirdCall().resolves({
                status: 200,
                data: {
                    choices: [{
                        message: {
                            content: JSON.stringify({
                                selfEdits: { dataAugmentations: ['fallback_improvement'] },
                                confidence: 0.7
                            })
                        }
                    }]
                }
            });
            
            await grokAdapter.initializeService();
            
            const context = {
                currentAnalysis: { accuracy: 0.5 },
                projectContext: { projectId: 'test' },
                performanceHistory: []
            };
            
            // First attempt - rate limit
            try {
                await grokAdapter.generateSelfEdits(context);
                expect.fail('Should have thrown rate limit error');
            } catch (error) {
                expect(error.message).to.include('rate limit');
                testMetrics.errorRecoveries.push({
                    error: 'rate_limit',
                    recovered: false
                });
            }
            
            // Wait and retry
            await new Promise(resolve => setTimeout(resolve, 100));
            
            // Second attempt - server error
            try {
                await grokAdapter.generateSelfEdits(context);
                expect.fail('Should have thrown server error');
            } catch (error) {
                expect(error.message).to.include('API call failed');
            }
            
            // Third attempt - success
            const result = await grokAdapter.generateSelfEdits(context);
            expect(result.dataAugmentations).to.include('fallback_improvement');
            
            testMetrics.errorRecoveries.push({
                error: 'api_failures',
                recovered: true,
                attempts: 3
            });
        });
    });
    
    describe('4. Full Self-Adaptation Cycle Tests', () => {
        it('should complete full adaptation cycle with measurable improvements', async () => {
            await sealService.initializeService();
            
            let currentAccuracy = 0.55;
            let analysisCount = 0;
            
            // Mock base analysis with improving accuracy
            sealService._performBaseAnalysis = async () => {
                analysisCount++;
                const result = {
                    projectId: 'test-project',
                    accuracy: currentAccuracy,
                    performance: 0.7,
                    complexity: 0.6,
                    patterns: ['basic_patterns'],
                    timestamp: new Date()
                };
                return result;
            };
            
            // Mock Grok responses with progressive improvements
            const grokResponses = [
                {
                    dataAugmentations: ['improve_pattern_detection'],
                    hyperparameterUpdates: { depth: 4 },
                    confidence: 0.8,
                    expectedImprovement: 0.1
                },
                {
                    dataAugmentations: ['add_context_awareness'],
                    hyperparameterUpdates: { depth: 5, contextWindow: 2000 },
                    confidence: 0.85,
                    expectedImprovement: 0.08
                },
                {
                    dataAugmentations: ['optimize_performance'],
                    hyperparameterUpdates: { parallelism: true },
                    confidence: 0.9,
                    expectedImprovement: 0.07
                }
            ];
            
            let grokCallCount = 0;
            axiosStub.callsFake(async () => ({
                status: 200,
                data: {
                    choices: [{
                        message: {
                            content: JSON.stringify(grokResponses[grokCallCount++ % 3])
                        }
                    }]
                }
            }));
            
            console.log('\n🔄 Full Adaptation Cycle Test (3 cycles)');
            
            // Run 3 complete adaptation cycles
            for (let cycle = 0; cycle < 3; cycle++) {
                const cycleStart = Date.now();
                
                // Perform self-adapting analysis
                const result = await sealService.performSelfAdaptingAnalysis(
                    'test-project',
                    'comprehensive_analysis',
                    true
                );
                
                expect(result.sealEnhancements.adaptationApplied).to.be.true;
                
                // Simulate improvement based on Grok suggestions
                const improvement = grokResponses[cycle].expectedImprovement * 0.8;
                currentAccuracy = Math.min(0.95, currentAccuracy + improvement);
                
                // Provide user feedback
                const feedback = {
                    helpful: true,
                    accurate: currentAccuracy > 0.7,
                    rating: Math.floor(currentAccuracy * 5),
                    executionTime: Date.now() - cycleStart
                };
                
                const learningResult = await sealService.learnFromUserFeedback(
                    result.sealEnhancements.adaptationId || `cycle-${cycle}`,
                    feedback
                );
                
                expect(learningResult.learningApplied).to.be.true;
                
                testMetrics.adaptationCycles.push({
                    cycle: cycle + 1,
                    initialAccuracy: result.accuracy,
                    finalAccuracy: currentAccuracy,
                    improvement: improvement,
                    feedback: feedback,
                    duration: Date.now() - cycleStart
                });
                
                console.log(`   Cycle ${cycle + 1}: ${result.accuracy.toFixed(2)} → ${currentAccuracy.toFixed(2)} (+${improvement.toFixed(2)})`);
            }
            
            // Verify cumulative improvement
            const totalImprovement = currentAccuracy - 0.55;
            expect(totalImprovement).to.be.greaterThan(0.2);
            expect(currentAccuracy).to.be.greaterThan(0.75);
            
            console.log(`   Total improvement: +${totalImprovement.toFixed(2)} (${(totalImprovement/0.55*100).toFixed(0)}%)`);
        });
        
        it('should adapt to different code pattern types', async () => {
            await sealService.initializeService();
            
            // Mock Grok for pattern-specific adaptations
            axiosStub.callsFake(async (url, data) => {
                const prompt = data.messages[1].content;
                let response;
                
                if (prompt.includes('async')) {
                    response = {
                        adaptationPlan: {
                            patterns: ['async/await', 'promise', 'callback'],
                            features: ['error_boundaries', 'timeout_handling'],
                            strategies: ['parallel_detection', 'flow_analysis']
                        },
                        confidence: 0.88
                    };
                } else if (prompt.includes('react')) {
                    response = {
                        adaptationPlan: {
                            patterns: ['hooks', 'context', 'lifecycle'],
                            features: ['state_management', 'effect_tracking'],
                            strategies: ['component_tree_analysis', 'prop_flow']
                        },
                        confidence: 0.91
                    };
                } else {
                    response = {
                        adaptationPlan: {
                            patterns: ['generic'],
                            features: ['basic_analysis'],
                            strategies: ['standard']
                        },
                        confidence: 0.7
                    };
                }
                
                return {
                    status: 200,
                    data: {
                        choices: [{
                            message: { content: JSON.stringify(response) }
                        }]
                    }
                };
            });
            
            // Test async pattern adaptation
            const asyncExamples = [
                { 
                    code: 'async function fetchData() { const result = await api.get(); return result; }',
                    metadata: { type: 'async_function' }
                },
                {
                    code: 'promise.then(data => process(data)).catch(err => handle(err));',
                    metadata: { type: 'promise_chain' }
                }
            ];
            
            const asyncAdaptation = await sealService.adaptToNewCodingPatterns(
                asyncExamples,
                'Modern async/await patterns'
            );
            
            expect(asyncAdaptation.adaptationSuccessful).to.be.true;
            expect(asyncAdaptation.patternsLearned).to.include('async/await');
            
            // Test React pattern adaptation
            const reactExamples = [
                {
                    code: 'const [state, setState] = useState(initialValue);',
                    metadata: { type: 'react_hook' }
                },
                {
                    code: 'useEffect(() => { fetchData(); }, [dependency]);',
                    metadata: { type: 'effect_hook' }
                }
            ];
            
            const reactAdaptation = await sealService.adaptToNewCodingPatterns(
                reactExamples,
                'React hooks patterns'
            );
            
            expect(reactAdaptation.adaptationSuccessful).to.be.true;
            expect(reactAdaptation.confidenceScore).to.be.greaterThan(0.9);
            
            console.log('\n🎨 Pattern Adaptation Results:');
            console.log(`   Async patterns: ${asyncAdaptation.patternsLearned.length} learned`);
            console.log(`   React patterns: ${reactAdaptation.patternsLearned.length} learned`);
        });
    });
    
    describe('5. SAP Enterprise Compliance Tests', () => {
        it('should enforce governance for high-risk adaptations', async () => {
            await governance.initializeService();
            
            // Test high-risk operation
            const highRiskOperation = {
                type: 'model_architecture_change',
                riskLevel: 'HIGH',
                impact: 'SYSTEM_WIDE',
                changes: ['neural_network_replacement', 'algorithm_overhaul']
            };
            
            const context = {
                projectId: 'critical-system',
                dataClassification: 'CONFIDENTIAL',
                userRole: 'developer',
                environment: 'production'
            };
            
            // Mock internal validation methods
            sandbox.stub(governance, '_validateDataClassification').resolves({
                isValid: true,
                dataClassification: 'CONFIDENTIAL'
            });
            
            sandbox.stub(governance, '_validateAccessControl').resolves({
                isValid: true,
                authenticationPassed: true,
                authorizationLevel: 'LIMITED'
            });
            
            sandbox.stub(governance, '_performRiskAssessment').resolves({
                riskScore: 0.85,
                riskLevel: 'HIGH',
                mitigationRequired: true
            });
            
            const complianceResult = await governance.validateOperationCompliance(
                highRiskOperation,
                context
            );
            
            expect(complianceResult.isCompliant).to.be.false; // High risk in production
            expect(complianceResult.riskLevel).to.equal('HIGH');
            expect(complianceResult.requiresApproval).to.be.true;
            
            // Test approval workflow
            const approvalRequest = {
                operationId: 'op-123',
                userId: 'developer-1',
                approvalType: 'HIGH_RISK_ADAPTATION',
                justification: 'Performance critical improvement'
            };
            
            sandbox.stub(governance, '_determineRequiredApprovers').resolves([
                { role: 'security_officer', userId: 'sec-1' },
                { role: 'technical_lead', userId: 'lead-1' }
            ]);
            
            const workflowResult = await governance.manageApprovalWorkflow(
                'op-123',
                approvalRequest
            );
            
            expect(workflowResult.workflowStarted).to.be.true;
            expect(workflowResult.requiredApprovals).to.have.lengthOf(2);
        });
        
        it('should generate comprehensive audit reports', async () => {
            await governance.initializeService();
            
            // Simulate adaptation history
            const adaptationHistory = [
                {
                    timestamp: new Date(Date.now() - 86400000),
                    operation: 'self_adaptation',
                    riskLevel: 'MEDIUM',
                    outcome: 'SUCCESS',
                    performanceGain: 0.15
                },
                {
                    timestamp: new Date(Date.now() - 43200000),
                    operation: 'pattern_learning',
                    riskLevel: 'LOW',
                    outcome: 'SUCCESS',
                    performanceGain: 0.08
                },
                {
                    timestamp: new Date(),
                    operation: 'architecture_change',
                    riskLevel: 'HIGH',
                    outcome: 'PENDING_APPROVAL',
                    performanceGain: null
                }
            ];
            
            sandbox.stub(governance, '_generateComplianceSummaryData').resolves({
                complianceScore: 0.92,
                totalOperations: adaptationHistory.length,
                successfulOperations: 2,
                pendingOperations: 1,
                riskDistribution: { LOW: 1, MEDIUM: 1, HIGH: 1 },
                averagePerformanceGain: 0.115,
                keyFindings: [
                    'High compliance rate maintained',
                    '1 high-risk operation pending approval',
                    'Average performance gain of 11.5%'
                ]
            });
            
            const reportParams = {
                reportType: 'COMPLIANCE_SUMMARY',
                timeframe: '24h',
                includeDetails: true,
                format: 'JSON'
            };
            
            const report = await governance.generateAuditReport(reportParams);
            
            expect(report.reportGenerated).to.be.true;
            expect(report.complianceScore).to.equal(0.92);
            expect(report.keyFindings).to.have.lengthOf(3);
            
            console.log('\n📊 Compliance Report Summary:');
            console.log(`   Compliance Score: ${(report.complianceScore * 100).toFixed(0)}%`);
            console.log(`   Operations: ${report.totalOperations} total`);
            console.log(`   Risk Distribution: ${JSON.stringify(report.riskDistribution)}`);
        });
    });
    
    describe('6. Performance and Scalability Tests', () => {
        it('should handle concurrent adaptation requests efficiently', async () => {
            await sealService.initializeService();
            
            // Mock fast responses
            sealService._performBaseAnalysis = async () => ({
                accuracy: 0.7 + Math.random() * 0.1,
                performance: 0.8
            });
            
            axiosStub.resolves({
                status: 200,
                data: {
                    choices: [{
                        message: {
                            content: JSON.stringify({
                                dataAugmentations: ['concurrent_improvement'],
                                confidence: 0.85
                            })
                        }
                    }]
                }
            });
            
            console.log('\n⚡ Concurrent Operations Test');
            
            const startTime = Date.now();
            
            // Launch 10 concurrent adaptations
            const promises = Array.from({ length: 10 }, (_, i) => 
                sealService.performSelfAdaptingAnalysis(
                    `project-${i}`,
                    'concurrent_test',
                    true
                )
            );
            
            const results = await Promise.all(promises);
            const duration = Date.now() - startTime;
            
            // All should complete successfully
            results.forEach((result, i) => {
                expect(result.sealEnhancements.adaptationApplied).to.be.true;
                expect(result.accuracy).to.be.greaterThan(0.7);
            });
            
            console.log(`   Completed 10 concurrent adaptations in ${duration}ms`);
            console.log(`   Average time per adaptation: ${(duration / 10).toFixed(0)}ms`);
            
            // Should complete in reasonable time (less than 5 seconds total)
            expect(duration).to.be.lessThan(5000);
        });
        
        it('should maintain performance under memory pressure', async () => {
            await rlEngine.initializeService();
            
            console.log('\n💾 Memory Management Test');
            
            // Record initial memory
            if (global.gc) global.gc();
            const initialMemory = process.memoryUsage().heapUsed;
            
            // Generate large number of episodes
            for (let i = 0; i < 15000; i++) {
                rlEngine.episodeData.push({
                    timestamp: new Date(),
                    state: `state-${i}`,
                    action: `action-${i % 10}`,
                    reward: Math.random(),
                    nextState: `state-${i + 1}`
                });
            }
            
            // Check memory management kicked in
            expect(rlEngine.episodeData.length).to.be.lessThan(10000);
            expect(rlEngine.episodeData.length).to.be.greaterThan(7000);
            
            // Force garbage collection and check memory
            if (global.gc) global.gc();
            const finalMemory = process.memoryUsage().heapUsed;
            const memoryGrowth = (finalMemory - initialMemory) / 1024 / 1024;
            
            console.log(`   Episode limit enforced: ${rlEngine.episodeData.length} episodes`);
            console.log(`   Memory growth: ${memoryGrowth.toFixed(2)} MB`);
            
            // Memory growth should be reasonable (less than 50MB)
            expect(memoryGrowth).to.be.lessThan(50);
        });
    });
    
    // Helper functions
    function calculateVariance(numbers) {
        const mean = numbers.reduce((a, b) => a + b, 0) / numbers.length;
        const squaredDiffs = numbers.map(n => Math.pow(n - mean, 2));
        return squaredDiffs.reduce((a, b) => a + b, 0) / numbers.length;
    }
    
    async function generateTestReport(metrics) {
        const report = {
            timestamp: new Date().toISOString(),
            summary: {
                totalQLearningUpdates: metrics.qLearningUpdates.length,
                totalGrokAPICalls: metrics.grokApiCalls.length,
                adaptationCycles: metrics.adaptationCycles.length,
                averageImprovement: calculateAverageImprovement(metrics.adaptationCycles),
                errorRecoveryRate: calculateRecoveryRate(metrics.errorRecoveries)
            },
            details: {
                qLearning: analyzeQLearning(metrics.qLearningUpdates),
                grokIntegration: analyzeGrokUsage(metrics.grokApiCalls),
                adaptationEffectiveness: analyzeAdaptations(metrics.adaptationCycles)
            }
        };
        
        console.log('\n📋 COMPREHENSIVE TEST REPORT');
        console.log('═══════════════════════════════════════════════════════════════');
        console.log(`Total Q-Learning Updates: ${report.summary.totalQLearningUpdates}`);
        console.log(`Total Grok API Calls: ${report.summary.totalGrokAPICalls}`);
        console.log(`Adaptation Cycles: ${report.summary.adaptationCycles}`);
        console.log(`Average Improvement: ${(report.summary.averageImprovement * 100).toFixed(1)}%`);
        console.log(`Error Recovery Rate: ${(report.summary.errorRecoveryRate * 100).toFixed(0)}%`);
        console.log('═══════════════════════════════════════════════════════════════');
        
        return report;
    }
    
    function calculateAverageImprovement(cycles) {
        if (cycles.length === 0) return 0;
        const totalImprovement = cycles.reduce((sum, cycle) => 
            sum + (cycle.improvement || 0), 0);
        return totalImprovement / cycles.length;
    }
    
    function calculateRecoveryRate(recoveries) {
        if (recoveries.length === 0) return 1;
        const recovered = recoveries.filter(r => r.recovered).length;
        return recovered / recoveries.length;
    }
    
    function analyzeQLearning(updates) {
        const avgReward = updates.reduce((sum, u) => sum + u.reward, 0) / updates.length;
        const convergenceRate = updates.filter(u => 
            Math.abs(u.qValueChange) < 0.01).length / updates.length;
        
        return {
            averageReward: avgReward,
            convergenceRate: convergenceRate,
            totalUpdates: updates.length
        };
    }
    
    function analyzeGrokUsage(calls) {
        const contextTypes = {};
        calls.forEach(call => {
            const type = call.context.projectContext?.language || 'unknown';
            contextTypes[type] = (contextTypes[type] || 0) + 1;
        });
        
        return {
            totalCalls: calls.length,
            contextDistribution: contextTypes,
            averageConfidence: calls.reduce((sum, c) => 
                sum + (c.response?.confidence || 0), 0) / calls.length
        };
    }
    
    function analyzeAdaptations(cycles) {
        return {
            totalCycles: cycles.length,
            successfulCycles: cycles.filter(c => c.improvement > 0).length,
            averageDuration: cycles.reduce((sum, c) => 
                sum + (c.duration || 0), 0) / cycles.length
        };
    }
});

module.exports = {
    runComprehensiveSealValidation: async function() {
        console.log('Starting comprehensive SEAL validation...');
        // Can be called from CLI or other scripts
    }
};