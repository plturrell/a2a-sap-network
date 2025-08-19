const { expect } = require('chai');
const sinon = require('sinon');
const cds = require('@sap/cds');
const SealEnhancedGleanService = require('../../srv/glean/sealEnhancedGleanService');
const GrokSealAdapter = require('../../srv/seal/grokSealAdapter');
const ReinforcementLearningEngine = require('../../srv/seal/reinforcementLearningEngine');
const SapSealGovernance = require('../../srv/seal/sapSealGovernance');
const SealConfiguration = require('../../srv/seal/sealConfiguration');

/**
 * Integration tests for real SEAL implementation
 * Tests Grok API integration, RL learning, and SAP compliance
 */
describe('SEAL Integration Tests', function() {
    let sealService;
    let grokAdapter;
    let rlEngine;
    let governance;
    let configuration;
    let sandbox;

    before(async function() {
        this.timeout(10000);
        
        // Initialize test database
        cds.test.in(__dirname);
        cds.env.requires.db = { kind: 'sqlite', credentials: { url: ':memory:' } };
        
        // Load test configuration
        process.env.NODE_ENV = 'test';
        process.env.GROK_API_KEY = 'test-api-key';
        process.env.MOCK_EXTERNAL_SERVICES = 'true';
    });

    beforeEach(function() {
        sandbox = sinon.createSandbox();
        
        // Initialize components
        configuration = new SealConfiguration();
        sealService = new SealEnhancedGleanService();
        grokAdapter = new GrokSealAdapter();
        rlEngine = new ReinforcementLearningEngine();
        governance = new SapSealGovernance();
        
        // Mock external dependencies
        this._mockExternalServices();
    });

    afterEach(function() {
        sandbox.restore();
    });

    describe('SEAL Configuration', function() {
        it('should load valid configuration for test environment', function() {
            const config = configuration.getConfiguration();
            
            expect(config).to.be.an('object');
            expect(config.grok).to.have.property('apiKey');
            expect(config.reinforcementLearning).to.have.property('learningRate');
            expect(config.compliance).to.have.property('enabled');
            expect(config.monitoring).to.have.property('enabled');
        });

        it('should validate configuration successfully', function() {
            const validation = configuration.validateConfiguration();
            
            expect(validation.isValid).to.be.true;
            expect(validation.errors).to.be.empty;
        });

        it('should apply test environment overrides', function() {
            const config = configuration.getConfiguration();
            
            expect(config.grok.timeout).to.equal(5000); // Test override
            expect(config.grok.retryAttempts).to.equal(1); // Test override
            expect(config.monitoring.enabled).to.be.false; // Test override
            expect(config.development.mockExternalServices).to.be.true; // Test override
        });
    });

    describe('Grok SEAL Adapter', function() {
        beforeEach(async function() {
            // Mock Grok API calls
            sandbox.stub(grokAdapter, '_callGrokApi').resolves({
                choices: [{
                    message: {
                        content: JSON.stringify({
                            dataAugmentations: ['increase_training_diversity'],
                            hyperparameterUpdates: { learning_rate: 0.001 },
                            modelModifications: ['add_attention_layer'],
                            confidence: 0.8,
                            reasoning: 'Test improvement strategy'
                        })
                    }
                }]
            });
            
            await grokAdapter.initializeService();
        });

        it('should initialize successfully', async function() {
            expect(grokAdapter.sealAdapter).to.exist;
            expect(grokAdapter.adaptationHistory).to.be.instanceOf(Map);
            expect(grokAdapter.performanceMetrics).to.be.instanceOf(Map);
        });

        it('should generate self-edits using Grok API', async function() {
            const analysisContext = {
                currentAnalysis: { accuracy: 0.7, performance: 0.6 },
                projectContext: { projectId: 'test-project' },
                performanceHistory: []
            };
            
            const selfEdits = await grokAdapter.generateSelfEdits(analysisContext);
            
            expect(selfEdits).to.have.property('dataAugmentations');
            expect(selfEdits).to.have.property('hyperparameterUpdates');
            expect(selfEdits).to.have.property('modelModifications');
            expect(selfEdits).to.have.property('confidence');
            expect(selfEdits.confidence).to.be.a('number');
            expect(selfEdits.confidence).to.be.at.least(0).and.at.most(1);
        });

        it('should perform self-directed learning', async function() {
            const analysisResults = {
                accuracy: 0.85,
                completeness: 0.9,
                executionTime: 3000
            };
            
            const userFeedback = {
                helpful: true,
                accurate: true,
                rating: 4
            };
            
            const learningResult = await grokAdapter.performSelfDirectedLearning(
                analysisResults,
                userFeedback
            );
            
            expect(learningResult).to.have.property('performanceScore');
            expect(learningResult).to.have.property('learningStrategy');
            expect(learningResult).to.have.property('improvements');
            expect(learningResult).to.have.property('adaptationId');
            
            expect(learningResult.performanceScore).to.be.a('number');
            expect(learningResult.adaptationId).to.be.a('string');
        });

        it('should adapt to new code patterns using few-shot learning', async function() {
            const codeExamples = [
                {
                    code: 'function testFunction() { return "hello"; }',
                    metadata: { type: 'function', language: 'javascript' }
                },
                {
                    code: 'const testArrow = () => "world";',
                    metadata: { type: 'arrow_function', language: 'javascript' }
                }
            ];
            
            const patternContext = {
                type: 'javascript_functions',
                domain: 'frontend',
                description: 'JavaScript function patterns'
            };
            
            const adaptationResult = await grokAdapter.adaptToNewCodePatterns(
                codeExamples,
                patternContext
            );
            
            expect(adaptationResult).to.have.property('adaptationPlan');
            expect(adaptationResult).to.have.property('syntheticDataGenerated');
            expect(adaptationResult).to.have.property('newCapabilities');
            expect(adaptationResult).to.have.property('confidenceScore');
            
            expect(adaptationResult.syntheticDataGenerated).to.be.a('number');
            expect(adaptationResult.confidenceScore).to.be.a('number');
        });

        it('should handle Grok API failures gracefully', async function() {
            // Restore stub and create failing stub
            grokAdapter._callGrokApi.restore();
            sandbox.stub(grokAdapter, '_callGrokApi').rejects(new Error('API failure'));
            
            try {
                await grokAdapter.generateSelfEdits({ currentAnalysis: {} });
                expect.fail('Should have thrown an error');
            } catch (error) {
                expect(error.message).to.include('SEAL self-edit generation failed');
            }
        });
    });

    describe('Reinforcement Learning Engine', function() {
        beforeEach(async function() {
            await rlEngine.initializeService();
        });

        it('should initialize with correct state and action spaces', async function() {
            expect(rlEngine.stateSpace).to.be.instanceOf(Map);
            expect(rlEngine.actionSpace).to.be.instanceOf(Map);
            expect(rlEngine.qTable).to.be.instanceOf(Map);
            
            expect(rlEngine.stateSpace.size).to.be.greaterThan(0);
            expect(rlEngine.actionSpace.size).to.be.greaterThan(0);
            expect(rlEngine.qTable.size).to.be.greaterThan(0);
        });

        it('should learn from feedback using Q-Learning', async function() {
            const state = {
                codebase_complexity: 0.6,
                analysis_accuracy: 0.7,
                performance_score: 0.8
            };
            
            const action = {
                type: 'increase_depth',
                target: 'analysis_depth',
                intensity: 1
            };
            
            const reward = 0.3;
            const nextState = {
                codebase_complexity: 0.6,
                analysis_accuracy: 0.8,
                performance_score: 0.75
            };
            
            const learningResult = await rlEngine.learnFromFeedback(
                state,
                action,
                reward,
                nextState,
                { projectId: 'test-project' }
            );
            
            expect(learningResult).to.have.property('episodeId');
            expect(learningResult).to.have.property('qValueUpdate');
            expect(learningResult).to.have.property('explorationRate');
            expect(learningResult).to.have.property('complianceStatus');
            
            expect(learningResult.episodeId).to.be.a('string');
            expect(learningResult.qValueUpdate).to.have.property('updatedQ');
            expect(learningResult.complianceStatus).to.equal('COMPLIANT');
        });

        it('should select actions using epsilon-greedy policy', async function() {
            const currentState = {
                codebase_complexity: 0.5,
                analysis_accuracy: 0.6,
                performance_score: 0.7
            };
            
            const availableActions = [
                { type: 'increase_depth', target: 'analysis', intensity: 1 },
                { type: 'add_pattern_check', target: 'validation', intensity: 1 },
                { type: 'optimize_algorithm', target: 'performance', intensity: 1 }
            ];
            
            const actionSelection = await rlEngine.selectAction(
                currentState,
                availableActions,
                { projectId: 'test-project' }
            );
            
            expect(actionSelection).to.have.property('action');
            expect(actionSelection).to.have.property('selectionReason');
            expect(actionSelection).to.have.property('explorationRate');
            expect(actionSelection).to.have.property('qValue');
            expect(actionSelection).to.have.property('complianceStatus');
            
            expect(actionSelection.selectionReason).to.be.oneOf(['EXPLORATION', 'EXPLOITATION']);
            expect(actionSelection.complianceStatus).to.be.oneOf(['COMPLIANT', 'FALLBACK_APPLIED']);
        });

        it('should evaluate policy performance', async function() {
            // Add some episode data for evaluation
            rlEngine.episodeData.push({
                timestamp: new Date(),
                reward: 0.5,
                action: 'test_action'
            });
            
            rlEngine.episodeData.push({
                timestamp: new Date(),
                reward: 0.7,
                action: 'test_action_2'
            });
            
            const evaluation = await rlEngine.evaluatePolicyPerformance(1); // 1 hour
            
            expect(evaluation).to.have.property('averageReward');
            expect(evaluation).to.have.property('rewardTrend');
            expect(evaluation).to.have.property('actionDistribution');
            expect(evaluation).to.have.property('overallPerformance');
            
            expect(evaluation.averageReward).to.be.a('number');
            expect(evaluation.episodeCount).to.equal(2);
        });

        it('should perform multi-armed bandit optimization', async function() {
            const actionCandidates = [
                { type: 'action_a', intensity: 1 },
                { type: 'action_b', intensity: 2 },
                { type: 'action_c', intensity: 1 }
            ];
            
            const contextualFeatures = {
                projectType: 'javascript',
                complexity: 'medium'
            };
            
            const banditResult = await rlEngine.performBanditOptimization(
                actionCandidates,
                contextualFeatures
            );
            
            expect(banditResult).to.have.property('selectedAction');
            expect(banditResult).to.have.property('allCandidates');
            expect(banditResult).to.have.property('selectionReason');
            expect(banditResult).to.have.property('confidence');
            
            expect(banditResult.selectionReason).to.equal('MULTI_ARMED_BANDIT');
            expect(banditResult.allCandidates).to.be.an('array');
            expect(banditResult.allCandidates).to.have.lengthOf(3);
        });
    });

    describe('SAP SEAL Governance', function() {
        beforeEach(async function() {
            await governance.initializeService();
        });

        it('should validate operation compliance', async function() {
            const operation = {
                type: 'self_adaptation',
                target: 'code_analysis',
                riskLevel: 'MEDIUM'
            };
            
            const context = {
                projectId: 'test-project',
                dataClassification: 'INTERNAL',
                userId: 'test-user'
            };
            
            const complianceResult = await governance.validateOperationCompliance(
                operation,
                context
            );
            
            expect(complianceResult).to.have.property('operationId');
            expect(complianceResult).to.have.property('isCompliant');
            expect(complianceResult).to.have.property('complianceScore');
            expect(complianceResult).to.have.property('validationResults');
            expect(complianceResult).to.have.property('riskLevel');
            
            expect(complianceResult.operationId).to.be.a('string');
            expect(complianceResult.isCompliant).to.be.a('boolean');
            expect(complianceResult.complianceScore).to.be.a('number');
        });

        it('should monitor SEAL operations', async function() {
            const operationId = 'test-operation-123';
            const mockSealService = {
                getCurrentMetrics: sandbox.stub().resolves({
                    accuracy: 0.8,
                    performance: 0.7,
                    errorRate: 0.05
                })
            };
            
            const monitoringResult = await governance.monitorSealOperation(
                operationId,
                mockSealService
            );
            
            expect(monitoringResult).to.have.property('monitoringStarted');
            expect(monitoringResult).to.have.property('operationId');
            expect(monitoringResult).to.have.property('monitoringFrequency');
            
            expect(monitoringResult.monitoringStarted).to.be.true;
            expect(monitoringResult.operationId).to.equal(operationId);
            
            // Clean up monitoring
            await governance.completeGovernanceAssessment(operationId, {});
        });

        it('should generate audit reports', async function() {
            const reportParams = {
                reportType: 'COMPLIANCE_SUMMARY',
                timeframe: '24h',
                scope: 'all_operations',
                userId: 'test-user'
            };
            
            // Mock report data generation
            sandbox.stub(governance, '_generateComplianceSummaryData').resolves({
                keyFindings: ['High compliance score', 'No critical violations'],
                complianceScore: 0.95,
                totalOperations: 10
            });
            
            const reportResult = await governance.generateAuditReport(reportParams);
            
            expect(reportResult).to.have.property('reportGenerated');
            expect(reportResult).to.have.property('reportId');
            expect(reportResult).to.have.property('executiveSummary');
            expect(reportResult).to.have.property('keyFindings');
            
            expect(reportResult.reportGenerated).to.be.true;
            expect(reportResult.reportId).to.be.a('string');
        });

        it('should manage approval workflows for high-risk operations', async function() {
            const operationId = 'high-risk-operation-123';
            const approvalRequest = {
                userId: 'test-user',
                approvalType: 'HIGH_RISK_ADAPTATION',
                riskLevel: 'HIGH'
            };
            
            // Mock required approvers
            sandbox.stub(governance, '_determineRequiredApprovers').resolves([
                { role: 'security_officer', userId: 'security-123' },
                { role: 'data_protection_officer', userId: 'dpo-123' }
            ]);
            
            sandbox.stub(governance, '_sendApprovalNotifications').resolves();
            
            const workflowResult = await governance.manageApprovalWorkflow(
                operationId,
                approvalRequest
            );
            
            expect(workflowResult).to.have.property('workflowStarted');
            expect(workflowResult).to.have.property('workflowId');
            expect(workflowResult).to.have.property('requiredApprovers');
            expect(workflowResult).to.have.property('status');
            
            expect(workflowResult.workflowStarted).to.be.true;
            expect(workflowResult.requiredApprovers).to.include('security_officer');
            expect(workflowResult.status).to.equal('PENDING_APPROVAL');
        });
    });

    describe('SEAL Enhanced Glean Service Integration', function() {
        beforeEach(async function() {
            // Mock all external dependencies
            sandbox.stub(sealService, '_performBaseAnalysis').resolves({
                projectId: 'test-project',
                accuracy: 0.7,
                performance: 0.6,
                confidenceScore: 0.75
            });
            
            sandbox.stub(sealService.sealAdapter, 'generateSelfEdits').resolves({
                dataAugmentations: ['test_augmentation'],
                hyperparameterUpdates: { learning_rate: 0.001 },
                confidence: 0.8,
                reasoning: 'Test improvement'
            });
            
            sandbox.stub(sealService.rlEngine, 'selectAction').resolves({
                action: { type: 'test_action', intensity: 1 },
                selectionReason: 'EXPLOITATION',
                qValue: 0.5,
                complianceStatus: 'COMPLIANT'
            });
            
            sandbox.stub(sealService.rlEngine, 'learnFromFeedback').resolves({
                episodeId: 'test-episode-123',
                qValueUpdate: { updatedQ: 0.6 }
            });
            
            await sealService.initializeService();
        });

        it('should perform self-adapting analysis successfully', async function() {
            const result = await sealService.performSelfAdaptingAnalysis(
                'test-project',
                'dependency_analysis',
                true
            );
            
            expect(result).to.have.property('sealEnhancements');
            expect(result).to.have.property('executionTime');
            expect(result.sealEnhancements).to.have.property('adaptationApplied');
            expect(result.sealEnhancements).to.have.property('performanceImprovement');
            expect(result.sealEnhancements).to.have.property('actionSelected');
            
            expect(result.sealEnhancements.adaptationApplied).to.be.true;
            expect(result.sealEnhancements.performanceImprovement).to.be.a('number');
        });

        it('should learn from user feedback', async function() {
            // First create an analysis to provide feedback on
            const analysisResult = await sealService.performSelfAdaptingAnalysis(
                'test-project',
                'dependency_analysis',
                true
            );
            
            // Mock finding analysis record
            sandbox.stub(sealService, '_findAnalysisRecord').returns(analysisResult);
            sandbox.stub(sealService, '_updateRLWithUserFeedback').resolves();
            sandbox.stub(sealService, '_recordLearningEvent').resolves({
                learningId: 'learning-123'
            });
            
            const userFeedback = {
                helpful: true,
                accurate: true,
                rating: 4,
                executionTime: 3000
            };
            
            const learningResult = await sealService.learnFromUserFeedback(
                'analysis-123',
                userFeedback
            );
            
            expect(learningResult).to.have.property('learningApplied');
            expect(learningResult).to.have.property('learningId');
            expect(learningResult).to.have.property('rewardCalculated');
            
            expect(learningResult.learningApplied).to.be.true;
            expect(learningResult.rewardCalculated).to.be.a('number');
        });

        it('should adapt to new coding patterns', async function() {
            const codeExamples = [
                {
                    code: 'async function fetchData() { return await api.get("/data"); }',
                    metadata: { type: 'async_function', framework: 'express' }
                }
            ];
            
            const patternDescription = 'Async/await API patterns';
            
            // Mock pattern adaptation methods
            sandbox.stub(sealService, '_analyzePatternContext').returns({
                type: 'async_patterns',
                complexity: 'medium'
            });
            
            sandbox.stub(sealService, '_validatePatternAdaptation').resolves({
                validationPassed: true,
                score: 0.85
            });
            
            sandbox.stub(sealService, '_applyPatternAdaptation').resolves();
            sandbox.stub(sealService, '_recordPatternAdaptation').resolves({
                adaptationId: 'adaptation-123'
            });
            
            const adaptationResult = await sealService.adaptToNewCodingPatterns(
                codeExamples,
                patternDescription
            );
            
            expect(adaptationResult).to.have.property('adaptationSuccessful');
            expect(adaptationResult).to.have.property('adaptationId');
            expect(adaptationResult).to.have.property('newCapabilities');
            
            expect(adaptationResult.adaptationSuccessful).to.be.true;
        });

        it('should get SEAL performance metrics', async function() {
            // Mock metrics calculation methods
            sandbox.stub(sealService, '_calculateAdaptationMetrics').returns({
                successRate: 0.85,
                averageImprovement: 0.15
            });
            
            sandbox.stub(sealService, '_calculateLearningProgress').returns({
                episodeCount: 100,
                convergenceScore: 0.9
            });
            
            sandbox.stub(sealService, '_calculateSystemImpact').returns({
                performanceOverhead: 0.05,
                resourceUtilization: 0.7
            });
            
            sandbox.stub(sealService, '_calculateUserSatisfactionTrends').returns({
                averageRating: 4.2,
                trendDirection: 'improving'
            });
            
            const metricsResult = await sealService.getSealPerformanceMetrics('24h');
            
            expect(metricsResult).to.have.property('timeRange');
            expect(metricsResult).to.have.property('reinforcementLearning');
            expect(metricsResult).to.have.property('adaptationMetrics');
            expect(metricsResult).to.have.property('overallSealScore');
            
            expect(metricsResult.timeRange).to.equal('24h');
            expect(metricsResult.overallSealScore).to.be.a('number');
        });

        it('should handle failures gracefully with fallback', async function() {
            // Make SEAL components fail
            sealService.sealAdapter.generateSelfEdits.restore();
            sandbox.stub(sealService.sealAdapter, 'generateSelfEdits').rejects(
                new Error('SEAL generation failed')
            );
            
            const result = await sealService.performSelfAdaptingAnalysis(
                'test-project',
                'dependency_analysis',
                true
            );
            
            expect(result.sealEnhancements).to.have.property('adaptationApplied');
            expect(result.sealEnhancements).to.have.property('errorOccurred');
            expect(result.sealEnhancements).to.have.property('fallbackUsed');
            
            expect(result.sealEnhancements.adaptationApplied).to.be.false;
            expect(result.sealEnhancements.errorOccurred).to.be.true;
            expect(result.sealEnhancements.fallbackUsed).to.be.true;
        });
    });

    /**
     * Mock external services for testing
     * @private
     */
    _mockExternalServices() {
        // Mock Grok API calls
        sandbox.stub(GrokSealAdapter.prototype, '_callGrokApi').resolves({
            choices: [{
                message: {
                    content: '{"improvements": "test", "confidence": 0.8}'
                }
            }]
        });
        
        // Mock file system operations
        sandbox.stub(SealEnhancedGleanService.prototype, '_performBaseAnalysis').resolves({
            accuracy: 0.7,
            performance: 0.6
        });
        
        // Mock external monitoring services
        sandbox.stub(SapSealGovernance.prototype, '_sendApprovalNotifications').resolves();
    }
});