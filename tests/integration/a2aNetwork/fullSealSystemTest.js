const { expect } = require('chai');
const sinon = require('sinon');
const axios = require('axios');

// SEAL Components
const SealConfiguration = require('../../srv/seal/sealConfiguration');
const GrokSealAdapter = require('../../srv/seal/grokSealAdapter');
const ReinforcementLearningEngine = require('../../srv/seal/reinforcementLearningEngine');
const SapSealGovernance = require('../../srv/seal/sapSealGovernance');
const SealEnhancedGleanService = require('../../srv/glean/sealEnhancedGleanService');

// Base Components
const GraphAlgorithms = require('../../srv/algorithms/graphAlgorithms');
const TreeAlgorithms = require('../../srv/algorithms/treeAlgorithms');

/**
 * Comprehensive Full System Test for SEAL Implementation
 * Tests entire codebase integration and CLI functionality
 */
describe('Full SEAL System Integration Test', () => {
    let sandbox;
    let mockAxios;
    
    // Test environment setup
    const testConfig = {
        XAI_API_KEY: 'test-xai-key-12345',
        XAI_BASE_URL: 'https://api.x.ai/v1',
        XAI_MODEL: 'grok-4',
        NODE_ENV: 'test',
        MOCK_EXTERNAL_SERVICES: 'true'
    };

    before(async function() {
        this.timeout(15000);
        
        // Set test environment variables
        Object.keys(testConfig).forEach(key => {
            process.env[key] = testConfig[key];
        });
    });

    beforeEach(() => {
        sandbox = sinon.createSandbox();
        
        // Mock axios for xAI API calls
        mockAxios = sandbox.stub(axios, 'post');
        mockAxios.resolves({
            status: 200,
            data: {
                choices: [{
                    message: {
                        content: JSON.stringify({
                            improvements: ['test_improvement'],
                            confidence: 0.85,
                            reasoning: 'Test reasoning from Grok 4'
                        })
                    }
                }],
                usage: {
                    prompt_tokens: 100,
                    completion_tokens: 50,
                    total_tokens: 150
                }
            }
        });
    });

    afterEach(() => {
        sandbox.restore();
    });

    describe('1. Configuration System Test', () => {
        let configuration;

        beforeEach(() => {
            configuration = new SealConfiguration();
        });

        it('should load configuration correctly across all environments', () => {
            const config = configuration.getConfiguration();
            
            expect(config).to.be.an('object');
            expect(config.grok).to.have.property('apiKey');
            expect(config.grok).to.have.property('baseUrl');
            expect(config.grok).to.have.property('model');
            expect(config.reinforcementLearning).to.have.property('learningRate');
            expect(config.compliance).to.have.property('enabled');
            expect(config.monitoring).to.have.property('enabled');
        });

        it('should validate xAI Grok 4 configuration', () => {
            const validation = configuration.validateConfiguration();
            
            expect(validation.isValid).to.be.true;
            expect(validation.errors).to.be.empty;
        });

        it('should apply correct environment overrides', () => {
            const config = configuration.getConfiguration();
            
            // Test environment should have specific settings
            expect(config.grok.model).to.equal('grok-4');
            expect(config.grok.baseUrl).to.equal('https://api.x.ai/v1');
            expect(config.development.mockExternalServices).to.be.true;
        });

        it('should handle Grok 4 specific features', () => {
            const config = configuration.getConfiguration();
            
            expect(config.grok.features).to.have.property('reasoning');
            expect(config.grok.features).to.have.property('vision');
            expect(config.grok.features).to.have.property('structuredOutputs');
            expect(config.grok.features.reasoning).to.be.true;
        });
    });

    describe('2. Grok 4 API Integration Test', () => {
        let grokAdapter;

        beforeEach(async () => {
            grokAdapter = new GrokSealAdapter();
            await grokAdapter.initializeService();
        });

        it('should initialize with correct xAI credentials', () => {
            expect(grokAdapter.grokApiKey).to.equal(testConfig.XAI_API_KEY);
            expect(grokAdapter.grokBaseUrl).to.equal(testConfig.XAI_BASE_URL);
        });

        it('should make correct API calls to xAI Grok 4', async () => {
            const analysisContext = {
                currentAnalysis: { accuracy: 0.7 },
                projectContext: { projectId: 'test-proj' },
                performanceHistory: []
            };

            await grokAdapter.generateSelfEdits(analysisContext);

            // Verify API call was made correctly
            expect(mockAxios.calledOnce).to.be.true;
            
            const callArgs = mockAxios.getCall(0).args;
            expect(callArgs[0]).to.equal('https://api.x.ai/v1/chat/completions');
            
            const requestData = callArgs[1];
            expect(requestData.model).to.equal('grok-4');
            expect(requestData.stream).to.equal(false);
            expect(requestData).to.not.have.property('presence_penalty');
            expect(requestData).to.not.have.property('frequency_penalty');
            expect(requestData).to.not.have.property('stop');
            
            const headers = callArgs[2].headers;
            expect(headers.Authorization).to.equal(`Bearer ${testConfig.XAI_API_KEY}`);
            expect(headers['Content-Type']).to.equal('application/json');
        });

        it('should handle xAI rate limiting correctly', async () => {
            // Mock rate limit response
            mockAxios.onFirstCall().rejects({
                response: { status: 429, data: { error: { message: 'Rate limit exceeded' } } }
            });
            mockAxios.onSecondCall().resolves({
                status: 200,
                data: { choices: [{ message: { content: '{"test": "success"}' } }] }
            });

            try {
                const result = await grokAdapter.generateSelfEdits({
                    currentAnalysis: {},
                    projectContext: {},
                    performanceHistory: []
                });
                expect(result).to.exist;
            } catch (error) {
                expect(error.message).to.include('xAI API rate limit exceeded');
            }
        });

        it('should perform few-shot learning with Grok 4', async () => {
            const codeExamples = [
                { code: 'async function test() {}', metadata: { type: 'async' } }
            ];
            const patternContext = { type: 'async_patterns', domain: 'javascript' };

            const result = await grokAdapter.adaptToNewCodePatterns(codeExamples, patternContext);

            expect(result).to.have.property('adaptationPlan');
            expect(result).to.have.property('confidenceScore');
            expect(mockAxios.calledOnce).to.be.true;
        });
    });

    describe('3. Reinforcement Learning Engine Test', () => {
        let rlEngine;

        beforeEach(async () => {
            rlEngine = new ReinforcementLearningEngine();
            await rlEngine.initializeService();
        });

        it('should initialize Q-learning properly', () => {
            expect(rlEngine.stateSpace).to.be.instanceOf(Map);
            expect(rlEngine.actionSpace).to.be.instanceOf(Map);
            expect(rlEngine.qTable).to.be.instanceOf(Map);
            expect(rlEngine.qTable.size).to.be.greaterThan(0);
        });

        it('should perform Q-learning updates correctly', async () => {
            const state = { codebase_complexity: 0.6, analysis_accuracy: 0.7 };
            const action = { type: 'increase_depth', intensity: 1 };
            const reward = 0.3;
            const nextState = { codebase_complexity: 0.6, analysis_accuracy: 0.8 };

            const result = await rlEngine.learnFromFeedback(
                state, action, reward, nextState, { projectId: 'test' }
            );

            expect(result).to.have.property('episodeId');
            expect(result).to.have.property('qValueUpdate');
            expect(result.complianceStatus).to.equal('COMPLIANT');
        });

        it('should select actions using epsilon-greedy policy', async () => {
            const state = { codebase_complexity: 0.5 };
            const actions = [
                { type: 'action_a', intensity: 1 },
                { type: 'action_b', intensity: 2 }
            ];

            const result = await rlEngine.selectAction(state, actions);

            expect(result).to.have.property('action');
            expect(result).to.have.property('selectionReason');
            expect(result.selectionReason).to.be.oneOf(['EXPLORATION', 'EXPLOITATION']);
        });

        it('should perform multi-armed bandit optimization', async () => {
            const actions = [
                { type: 'action_1' }, { type: 'action_2' }, { type: 'action_3' }
            ];

            const result = await rlEngine.performBanditOptimization(actions, {});

            expect(result).to.have.property('selectedAction');
            expect(result).to.have.property('allCandidates');
            expect(result.selectionReason).to.equal('MULTI_ARMED_BANDIT');
        });
    });

    describe('4. SAP Compliance and Governance Test', () => {
        let governance;

        beforeEach(async () => {
            governance = new SapSealGovernance();
            await governance.initializeService();
        });

        it('should validate operation compliance', async () => {
            const operation = { type: 'self_adaptation', riskLevel: 'MEDIUM' };
            const context = { projectId: 'test', dataClassification: 'INTERNAL' };

            // Mock compliance validation methods
            sandbox.stub(governance, '_validateDataClassification').resolves({
                isValid: true, dataClassification: 'INTERNAL'
            });
            sandbox.stub(governance, '_validateAccessControl').resolves({
                isValid: true, authenticationPassed: true
            });
            sandbox.stub(governance, '_performRiskAssessment').resolves({
                riskScore: 0.4, riskLevel: 'MEDIUM'
            });
            sandbox.stub(governance, '_checkRegulatoryCompliance').resolves({
                isCompliant: true, frameworks: ['GDPR', 'SOX']
            });
            sandbox.stub(governance, '_validateBusinessPolicies').resolves({
                isValid: true, policiesChecked: 5
            });
            sandbox.stub(governance, '_validateTechnicalStandards').resolves({
                isValid: true, standardsCompliant: true
            });

            const result = await governance.validateOperationCompliance(operation, context);

            expect(result).to.have.property('isCompliant');
            expect(result).to.have.property('complianceScore');
            expect(result).to.have.property('validationResults');
            expect(result).to.have.property('riskLevel');
        });

        it('should manage approval workflows', async () => {
            const operationId = 'test-op-123';
            const approvalRequest = {
                userId: 'test-user',
                approvalType: 'HIGH_RISK_ADAPTATION',
                riskLevel: 'HIGH'
            };

            sandbox.stub(governance, '_determineRequiredApprovers').resolves([
                { role: 'security_officer', userId: 'sec-123' }
            ]);
            sandbox.stub(governance, '_sendApprovalNotifications').resolves();

            const result = await governance.manageApprovalWorkflow(operationId, approvalRequest);

            expect(result).to.have.property('workflowStarted');
            expect(result).to.have.property('workflowId');
            expect(result.workflowStarted).to.be.true;
        });

        it('should generate audit reports', async () => {
            const reportParams = {
                reportType: 'COMPLIANCE_SUMMARY',
                timeframe: '24h',
                userId: 'test-user'
            };

            sandbox.stub(governance, '_generateComplianceSummaryData').resolves({
                complianceScore: 0.95,
                totalOperations: 10,
                keyFindings: ['High compliance maintained']
            });
            sandbox.stub(governance, '_generateExecutiveSummary').resolves(
                'Compliance maintained at 95% with no critical issues'
            );
            sandbox.stub(governance, '_generateReportRecommendations').resolves([
                'Continue current compliance practices'
            ]);
            sandbox.stub(governance, '_formatAuditReport').resolves({
                location: '/reports/compliance-123.pdf'
            });
            sandbox.stub(governance, '_storeAuditReport').resolves();

            const result = await governance.generateAuditReport(reportParams);

            expect(result).to.have.property('reportGenerated');
            expect(result).to.have.property('reportId');
            expect(result.reportGenerated).to.be.true;
        });
    });

    describe('5. Algorithm Integration Test', () => {
        let graphAlgorithms, treeAlgorithms;

        beforeEach(async () => {
            graphAlgorithms = new GraphAlgorithms();
            treeAlgorithms = new TreeAlgorithms();
            await graphAlgorithms.initializeService();
            await treeAlgorithms.initializeService();
        });

        it('should integrate graph algorithms with SEAL', () => {
            const graph = graphAlgorithms.createGraph();
            graphAlgorithms.addEdge(graph, 'A', 'B', 1);
            graphAlgorithms.addEdge(graph, 'B', 'C', 2);

            const result = graphAlgorithms.dijkstra(graph, 'A');
            expect(result.getDistance('C')).to.equal(3);
            expect(result.hasPath('C')).to.be.true;
        });

        it('should integrate tree algorithms with SEAL', () => {
            const structure = { a: [1, 2], b: { c: [3, 4] } };
            const flattened = treeAlgorithms.flatten(structure);
            expect(flattened).to.deep.equal([1, 2, 3, 4]);

            const mapped = treeAlgorithms.mapStructure(x => x * 2, structure);
            expect(mapped.a).to.deep.equal([2, 4]);
            expect(mapped.b.c).to.deep.equal([6, 8]);
        });
    });

    describe('6. Full SEAL Enhanced Glean Service Test', () => {
        let sealService;

        beforeEach(async () => {
            sealService = new SealEnhancedGleanService();
            
            // Mock parent class methods
            sandbox.stub(sealService, '_performBaseAnalysis').resolves({
                projectId: 'test-project',
                accuracy: 0.7,
                performance: 0.6,
                confidenceScore: 0.75
            });

            // Mock SEAL adapter methods
            sandbox.stub(sealService.sealAdapter, 'generateSelfEdits').resolves({
                dataAugmentations: ['test_aug'],
                hyperparameterUpdates: { lr: 0.01 },
                confidence: 0.8
            });

            // Mock RL engine methods
            sandbox.stub(sealService.rlEngine, 'selectAction').resolves({
                action: { type: 'test_action', intensity: 1 },
                selectionReason: 'EXPLOITATION'
            });

            sandbox.stub(sealService.rlEngine, 'learnFromFeedback').resolves({
                episodeId: 'test-episode',
                qValueUpdate: { updatedQ: 0.6 }
            });

            await sealService.initializeService();
        });

        it('should perform complete self-adapting analysis', async () => {
            const result = await sealService.performSelfAdaptingAnalysis(
                'test-project',
                'dependency_analysis',
                true
            );

            expect(result).to.have.property('sealEnhancements');
            expect(result.sealEnhancements).to.have.property('adaptationApplied');
            expect(result.sealEnhancements).to.have.property('performanceImprovement');
            expect(result.sealEnhancements.adaptationApplied).to.be.true;
        });

        it('should learn from user feedback', async () => {
            // Mock finding analysis record
            sandbox.stub(sealService, '_findAnalysisRecord').returns({
                projectId: 'test-project',
                sealEnhancements: { rlEpisodeId: 'test-episode' }
            });
            sandbox.stub(sealService, '_updateRLWithUserFeedback').resolves();
            sandbox.stub(sealService, '_recordLearningEvent').resolves({
                learningId: 'learning-123'
            });

            const result = await sealService.learnFromUserFeedback('analysis-123', {
                helpful: true,
                accurate: true,
                rating: 4
            });

            expect(result).to.have.property('learningApplied');
            expect(result).to.have.property('rewardCalculated');
            expect(result.learningApplied).to.be.true;
        });

        it('should adapt to new coding patterns', async () => {
            sandbox.stub(sealService, '_analyzePatternContext').returns({
                type: 'test_pattern'
            });
            sandbox.stub(sealService, '_validatePatternAdaptation').resolves({
                validationPassed: true,
                score: 0.9
            });
            sandbox.stub(sealService, '_applyPatternAdaptation').resolves();
            sandbox.stub(sealService, '_recordPatternAdaptation').resolves({
                adaptationId: 'adapt-123'
            });

            const result = await sealService.adaptToNewCodingPatterns(
                [{ code: 'test code', metadata: {} }],
                'Test pattern'
            );

            expect(result).to.have.property('adaptationSuccessful');
            expect(result.adaptationSuccessful).to.be.true;
        });

        it('should get comprehensive performance metrics', async () => {
            sandbox.stub(sealService, '_calculateAdaptationMetrics').returns({
                successRate: 0.85
            });
            sandbox.stub(sealService, '_calculateLearningProgress').returns({
                convergenceScore: 0.9
            });
            sandbox.stub(sealService, '_calculateSystemImpact').returns({
                performanceOverhead: 0.05
            });
            sandbox.stub(sealService, '_calculateUserSatisfactionTrends').returns({
                averageRating: 4.2
            });

            const result = await sealService.getSealPerformanceMetrics('24h');

            expect(result).to.have.property('overallSealScore');
            expect(result).to.have.property('adaptationMetrics');
            expect(result).to.have.property('learningProgress');
            expect(result.overallSealScore).to.be.a('number');
        });

        it('should handle failures gracefully', async () => {
            // Force SEAL adapter to fail
            sealService.sealAdapter.generateSelfEdits.restore();
            sandbox.stub(sealService.sealAdapter, 'generateSelfEdits').rejects(
                new Error('Test failure')
            );

            const result = await sealService.performSelfAdaptingAnalysis(
                'test-project',
                'dependency_analysis',
                true
            );

            expect(result.sealEnhancements).to.have.property('adaptationApplied');
            expect(result.sealEnhancements).to.have.property('errorOccurred');
            expect(result.sealEnhancements.adaptationApplied).to.be.false;
            expect(result.sealEnhancements.errorOccurred).to.be.true;
        });
    });

    describe('7. CLI Integration Test', () => {
        it('should provide CLI-compatible configuration', () => {
            const config = new SealConfiguration();
            const grokConfig = config.getGrokConfig();
            const rlConfig = config.getRLConfig();
            const complianceConfig = config.getComplianceConfig();

            expect(grokConfig).to.have.property('apiKey');
            expect(grokConfig).to.have.property('model');
            expect(rlConfig).to.have.property('learningRate');
            expect(complianceConfig).to.have.property('enabled');
        });

        it('should export all necessary components for CLI usage', () => {
            // Test that all components can be imported
            expect(SealConfiguration).to.be.a('function');
            expect(GrokSealAdapter).to.be.a('function');
            expect(ReinforcementLearningEngine).to.be.a('function');
            expect(SapSealGovernance).to.be.a('function');
            expect(SealEnhancedGleanService).to.be.a('function');
        });
    });

    describe('8. Environment and Deployment Test', () => {
        it('should work across different environments', () => {
            const environments = ['development', 'staging', 'production', 'test'];
            
            environments.forEach(env => {
                process.env.NODE_ENV = env;
                const config = new SealConfiguration();
                const envConfig = config.getConfiguration();
                
                expect(envConfig).to.be.an('object');
                expect(envConfig.grok.model).to.equal('grok-4');
                expect(envConfig.grok.baseUrl).to.equal('https://api.x.ai/v1');
            });
            
            // Reset to test environment
            process.env.NODE_ENV = 'test';
        });

        it('should validate production readiness', () => {
            process.env.NODE_ENV = 'production';
            const config = new SealConfiguration();
            const prodConfig = config.getConfiguration();
            
            // Production should have stricter settings
            expect(prodConfig.grok.timeout).to.equal(60000);
            expect(prodConfig.monitoring.alerting.thresholds.responseTime).to.equal(3000);
            expect(prodConfig.development.debugMode).to.be.false;
            
            process.env.NODE_ENV = 'test';
        });
    });

    describe('9. Performance and Scalability Test', () => {
        it('should handle concurrent SEAL operations', async () => {
            const sealService = new SealEnhancedGleanService();
            
            // Mock all dependencies
            sandbox.stub(sealService, '_performBaseAnalysis').resolves({ accuracy: 0.8 });
            sandbox.stub(sealService.sealAdapter, 'generateSelfEdits').resolves({
                confidence: 0.8, dataAugmentations: []
            });
            sandbox.stub(sealService.rlEngine, 'selectAction').resolves({
                action: { type: 'test' }, selectionReason: 'EXPLOITATION'
            });
            sandbox.stub(sealService.rlEngine, 'learnFromFeedback').resolves({
                episodeId: 'test'
            });
            
            await sealService.initializeService();

            // Run multiple concurrent operations
            const promises = Array.from({ length: 5 }, (_, i) =>
                sealService.performSelfAdaptingAnalysis(`project-${i}`, 'dependency_analysis', true)
            );

            const results = await Promise.all(promises);
            
            expect(results).to.have.lengthOf(5);
            results.forEach(result => {
                expect(result).to.have.property('sealEnhancements');
            });
        });

        it('should handle memory management correctly', () => {
            const rlEngine = new ReinforcementLearningEngine();
            
            // Simulate many episodes
            for (let i = 0; i < 15000; i++) {
                rlEngine.episodeData.push({
                    timestamp: new Date(),
                    reward: Math.random(),
                    action: `action-${i}`
                });
            }
            
            // Episode data should be managed within limits
            expect(rlEngine.episodeData.length).to.be.lessThan(10000);
        });
    });

    describe('10. Error Handling and Recovery Test', () => {
        it('should handle xAI API errors gracefully', async () => {
            const grokAdapter = new GrokSealAdapter();
            await grokAdapter.initializeService();

            // Mock API failure
            mockAxios.rejects(new Error('Network error'));

            try {
                await grokAdapter.generateSelfEdits({
                    currentAnalysis: {},
                    projectContext: {},
                    performanceHistory: []
                });
                expect.fail('Should have thrown an error');
            } catch (error) {
                expect(error.message).to.include('xAI Grok API call failed');
            }
        });

        it('should provide fallback mechanisms', async () => {
            const sealService = new SealEnhancedGleanService();
            
            // Mock base analysis but fail SEAL
            sandbox.stub(sealService, '_performBaseAnalysis').resolves({
                accuracy: 0.7,
                fallbackResult: true
            });
            sandbox.stub(sealService.sealAdapter, 'generateSelfEdits').rejects(
                new Error('SEAL failure')
            );
            
            await sealService.initializeService();

            const result = await sealService.performSelfAdaptingAnalysis(
                'test-project',
                'dependency_analysis',
                true
            );

            expect(result.sealEnhancements.fallbackUsed).to.be.true;
            expect(result.accuracy).to.equal(0.7);
        });
    });
});