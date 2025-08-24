/**
 * Agent 9 Adapter - Advanced Logical Reasoning and Decision-Making Agent
 * Converts between REST API and OData formats for reasoning tasks, knowledge base management, 
 * inference generation, decision making, problem solving, and logical analysis operations
 */

const { BlockchainClient } = require('../core/blockchain-client') = const { BlockchainClient } = require('../core/blockchain-client');
const { v4: uuidv4 } = require('uuid');

class Agent9Adapter {
    constructor() {
        this.baseUrl = process.env.AGENT9_BASE_URL || 'http://localhost:8008';
        this.apiVersion = 'v1';
        this.timeout = 60000; // Longer timeout for complex reasoning operations
    }

    // ===== REASONING TASKS =====
    async getReasoningTasks(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/reasoning-tasks`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'ReasoningTask');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createReasoningTask(data) {
        try {
            const restData = this._convertODataReasoningTaskToREST(data);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/reasoning-tasks`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTReasoningTaskToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateReasoningTask(id, data) {
        try {
            const restData = this._convertODataReasoningTaskToREST(data);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/reasoning-tasks/${id}`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTReasoningTaskToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async deleteReasoningTask(id) {
        try {
            await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/reasoning-tasks/${id}`, {
                timeout: this.timeout
            });
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== REASONING OPERATIONS =====
    async startReasoning(taskId, configuration) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/reasoning-tasks/${taskId}/start`, {
                configuration
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                taskName: response.data.task_name,
                reasoningType: response.data.reasoning_type?.toUpperCase(),
                engineType: response.data.engine_type?.toUpperCase(),
                problemDomain: response.data.problem_domain?.toUpperCase(),
                estimatedDuration: response.data.estimated_duration,
                sessionId: response.data.session_id
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async pauseReasoning(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/reasoning-tasks/${taskId}/pause`, {}, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                pausedAt: response.data.paused_at
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async resumeReasoning(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/reasoning-tasks/${taskId}/resume`, {}, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                resumedAt: response.data.resumed_at
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async cancelReasoning(taskId, reason) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/reasoning-tasks/${taskId}/cancel`, {
                reason
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                cancelledAt: response.data.cancelled_at
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async validateConclusion(taskId, validationMethod) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/reasoning-tasks/${taskId}/validate`, {
                validation_method: validationMethod?.toLowerCase()
            }, {
                timeout: this.timeout
            });
            
            return {
                isValid: response.data.is_valid,
                confidence: response.data.confidence,
                validationResults: response.data.validation_results,
                contradictions: response.data.contradictions || [],
                supportingEvidence: response.data.supporting_evidence || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async explainReasoning(taskId, detailLevel) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/reasoning-tasks/${taskId}/explain`, {
                detail_level: parseInt(detailLevel) || 3
            }, {
                timeout: this.timeout
            });
            
            return {
                explanation: response.data.explanation,
                reasoningChain: response.data.reasoning_chain || [],
                keyDecisionPoints: response.data.key_decision_points || [],
                alternativePaths: response.data.alternative_paths || [],
                confidence: response.data.confidence
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== KNOWLEDGE BASE MANAGEMENT =====
    async getKnowledgeBaseElements(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/knowledge-base`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'KnowledgeBaseElement');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createKnowledgeBaseElement(data) {
        try {
            const restData = this._convertODataKnowledgeElementToREST(data);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/knowledge-base`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTKnowledgeElementToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async addKnowledge(elementType, content, domain, confidenceLevel) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/knowledge-base/add`, {
                element_type: elementType?.toLowerCase(),
                content,
                domain: domain?.toLowerCase(),
                confidence_level: confidenceLevel
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                elementName: response.data.element_name,
                message: response.data.message,
                elementId: response.data.element_id
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateKnowledge(elementId, content, confidenceLevel) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/knowledge-base/${elementId}`, {
                content,
                confidence_level: confidenceLevel
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                elementType: response.data.element_type?.toUpperCase(),
                domain: response.data.domain?.toUpperCase()
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async validateKnowledgeBase(domain) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/knowledge-base/validate`, {
                domain: domain?.toLowerCase()
            }, {
                timeout: this.timeout
            });
            
            return {
                isConsistent: response.data.is_consistent,
                contradictions: response.data.contradictions?.map(c => ({
                    elements: c.conflicting_elements,
                    severity: c.severity?.toUpperCase(),
                    autoResolvable: c.auto_resolvable,
                    description: c.description
                })) || [],
                warnings: response.data.warnings || [],
                statistics: response.data.statistics
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== INFERENCE GENERATION =====
    async getLogicalInferences(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/inferences`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'LogicalInference');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createLogicalInference(data) {
        try {
            const restData = this._convertODataInferenceToREST(data);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/inferences`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTInferenceToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async generateInferences(taskId, inferenceTypes, maxInferences) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/reasoning-tasks/${taskId}/generate-inferences`, {
                inference_types: inferenceTypes?.map(t => t.toLowerCase()),
                max_inferences: maxInferences || 10
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                inferencesGenerated: response.data.inferences_generated,
                inferences: response.data.inferences?.map(inf => ({
                    statement: inf.statement,
                    type: inf.inference_type,
                    confidence: inf.confidence,
                    premises: inf.premises || [],
                    derivationPath: inf.derivation_path
                })) || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async verifyInference(inferenceId, verificationMethod) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/inferences/${inferenceId}/verify`, {
                verification_method: verificationMethod?.toLowerCase()
            }, {
                timeout: this.timeout
            });
            
            return {
                isValid: response.data.is_valid,
                confidence: response.data.confidence,
                evidence: response.data.evidence || [],
                counterExamples: response.data.counter_examples || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== DECISION MAKING =====
    async getDecisionRecords(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/decisions`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'DecisionRecord');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createDecisionRecord(data) {
        try {
            const restData = this._convertODataDecisionToREST(data);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/decisions`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTDecisionToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async makeDecision(taskId, decisionCriteria, alternatives) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/reasoning-tasks/${taskId}/make-decision`, {
                decision_criteria: decisionCriteria,
                alternatives: alternatives?.split(',').map(a => a.trim())
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                decision: response.data.decision,
                context: response.data.context,
                confidence: response.data.confidence,
                justification: response.data.justification,
                riskAssessment: response.data.risk_assessment,
                expectedOutcome: response.data.expected_outcome,
                riskLevel: response.data.risk_level?.toUpperCase(),
                decisionType: response.data.decision_type?.toUpperCase()
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async evaluateDecision(decisionId, actualOutcome) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/decisions/${decisionId}/evaluate`, {
                actual_outcome: actualOutcome
            }, {
                timeout: this.timeout
            });
            
            return {
                successRate: response.data.success_rate,
                lessonsLearned: response.data.lessons_learned,
                accuracyScore: response.data.accuracy_score,
                improvementSuggestions: response.data.improvement_suggestions || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== PROBLEM SOLVING =====
    async getProblemSolvingRecords(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/problems`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'ProblemSolvingRecord');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createProblemSolvingRecord(data) {
        try {
            const restData = this._convertODataProblemToREST(data);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/problems`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTProblemToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async solveProblem(problemDescription, problemType, solvingStrategy, constraints) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/problems/solve`, {
                problem_description: problemDescription,
                problem_type: problemType?.toLowerCase(),
                solving_strategy: solvingStrategy?.toLowerCase(),
                constraints: constraints
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                solution: response.data.solution,
                solutionSteps: response.data.solution_steps || [],
                qualityScore: response.data.quality_score,
                timeComplexity: response.data.time_complexity,
                spaceComplexity: response.data.space_complexity,
                solvingTime: response.data.solving_time,
                alternativeSolutions: response.data.alternative_solutions || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async optimizeSolution(problemId, optimizationCriteria) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/problems/${problemId}/optimize`, {
                optimization_criteria: optimizationCriteria
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                optimizedSolution: response.data.optimized_solution,
                improvementScore: response.data.improvement_score,
                tradeoffs: response.data.tradeoffs || [],
                performanceGain: response.data.performance_gain
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== REASONING ENGINES =====
    async getReasoningEngines(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/engines`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'ReasoningEngine');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async optimizeEngine(engineId, optimizationType, targetMetrics) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/engines/${engineId}/optimize`, {
                optimization_type: optimizationType?.toLowerCase(),
                target_metrics: targetMetrics
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                newConfiguration: response.data.new_configuration,
                performanceGain: response.data.performance_gain,
                optimizationResults: response.data.optimization_results
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async calibrateEngine(engineId, testDataset) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/engines/${engineId}/calibrate`, {
                test_dataset: testDataset
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                accuracyScore: response.data.accuracy_score,
                calibrationResults: response.data.calibration_results,
                recommendedAdjustments: response.data.recommended_adjustments || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== FUNCTION IMPLEMENTATIONS =====
    async getReasoningOptions() {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/reasoning-options`, {
                timeout: this.timeout
            });
            
            return {
                reasoningTypes: response.data.reasoning_types?.map(t => t.toUpperCase()) || [],
                engineTypes: response.data.engine_types?.map(t => t.toUpperCase()) || [],
                problemDomains: response.data.problem_domains?.map(d => d.toUpperCase()) || [],
                strategies: response.data.strategies?.map(s => s.toUpperCase()) || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getDashboardData(timeRange) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/dashboard`, {
                params: { time_range: timeRange },
                timeout: this.timeout
            });
            
            return {
                activeTasks: response.data.active_tasks || 0,
                completedTasks: response.data.completed_tasks || 0,
                totalInferences: response.data.total_inferences || 0,
                averageConfidence: response.data.average_confidence || 0,
                knowledgeElements: response.data.knowledge_elements || 0,
                activeEngines: response.data.active_engines || 0,
                recentActivity: response.data.recent_activity || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getKnowledgeBaseStats(domain) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/knowledge-base/stats`, {
                params: { domain: domain?.toLowerCase() },
                timeout: this.timeout
            });
            
            return {
                totalElements: response.data.total_elements || 0,
                factCount: response.data.fact_count || 0,
                ruleCount: response.data.rule_count || 0,
                ontologyCount: response.data.ontology_count || 0,
                averageConfidence: response.data.average_confidence || 0,
                lastUpdated: response.data.last_updated,
                domains: response.data.domains?.map(d => d.toUpperCase()) || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getReasoningChain(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/reasoning-tasks/${taskId}/chain`, {
                timeout: this.timeout
            });
            
            return {
                taskId: response.data.task_id,
                steps: response.data.steps?.map(step => ({
                    stepNumber: step.step_number,
                    operation: step.operation?.toUpperCase(),
                    input: step.input,
                    output: step.output,
                    confidence: step.confidence,
                    duration: step.duration
                })) || [],
                totalSteps: response.data.total_steps || 0,
                conclusion: response.data.conclusion
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async analyzeContradictions(domain) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/analyze/contradictions`, {
                params: { domain: domain?.toLowerCase() },
                timeout: this.timeout
            });
            
            return {
                contradictionCount: response.data.contradiction_count || 0,
                severityBreakdown: response.data.severity_breakdown || {},
                topContradictions: response.data.top_contradictions?.map(c => ({
                    id: c.id,
                    description: c.description,
                    severity: c.severity?.toUpperCase(),
                    conflictingElements: c.conflicting_elements || [],
                    suggestedResolution: c.suggested_resolution
                })) || [],
                resolutionSuggestions: response.data.resolution_suggestions || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getEngineComparison(engineTypes, problemDomain) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/engines/compare`, {
                params: { 
                    engine_types: engineTypes?.join(','),
                    problem_domain: problemDomain?.toLowerCase()
                },
                timeout: this.timeout
            });
            
            return {
                comparison: response.data.comparison?.map(comp => ({
                    engineType: comp.engine_type?.toUpperCase(),
                    accuracyScore: comp.accuracy_score,
                    performanceScore: comp.performance_score,
                    reliability: comp.reliability,
                    strengths: comp.strengths || [],
                    weaknesses: comp.weaknesses || []
                })) || [],
                recommendation: response.data.recommendation?.toUpperCase(),
                bestForDomain: response.data.best_for_domain?.toUpperCase()
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getDecisionAnalysis(decisionId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/decisions/${decisionId}/analysis`, {
                timeout: this.timeout
            });
            
            return {
                decisionId: response.data.decision_id,
                decisionQuality: response.data.decision_quality,
                factorsConsidered: response.data.factors_considered || [],
                alternativesEvaluated: response.data.alternatives_evaluated || [],
                riskFactors: response.data.risk_factors || [],
                confidenceFactors: response.data.confidence_factors || [],
                improvementOpportunities: response.data.improvement_opportunities || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getProblemSolvingInsights(problemType, timeRange) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/problems/insights`, {
                params: {
                    problem_type: problemType?.toLowerCase(),
                    time_range: timeRange
                },
                timeout: this.timeout
            });
            
            return {
                totalProblems: response.data.total_problems || 0,
                solvedProblems: response.data.solved_problems || 0,
                averageSolutionTime: response.data.average_solution_time || 0,
                averageQualityScore: response.data.average_quality_score || 0,
                commonStrategies: response.data.common_strategies?.map(s => s.toUpperCase()) || [],
                successRates: response.data.success_rates || {},
                trends: response.data.trends || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getPerformanceMetrics(engineType, timeRange) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/performance/metrics`, {
                params: {
                    engine_type: engineType?.toLowerCase(),
                    time_range: timeRange
                },
                timeout: this.timeout
            });
            
            return {
                engineType: response.data.engine_type?.toUpperCase(),
                averageResponseTime: response.data.average_response_time || 0,
                throughput: response.data.throughput || 0,
                accuracyRate: response.data.accuracy_rate || 0,
                errorRate: response.data.error_rate || 0,
                resourceUtilization: response.data.resource_utilization || {},
                performanceTrends: response.data.performance_trends || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async optimizeKnowledgeBase(domain, optimizationStrategy) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/knowledge-base/optimize`, {
                domain: domain?.toLowerCase(),
                optimization_strategy: optimizationStrategy?.toLowerCase()
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                optimizationResults: response.data.optimization_results,
                performanceImprovement: response.data.performance_improvement,
                elementsOptimized: response.data.elements_optimized || 0
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== PERFORMANCE METRICS =====
    async getReasoningPerformanceMetrics(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/performance/reasoning-metrics`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'ReasoningPerformanceMetric');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== UTILITY METHODS =====
    _convertODataToREST(query) {
        const params = {};
        
        if (query.$top) params.limit = query.$top;
        if (query.$skip) params.offset = query.$skip;
        if (query.$orderby) params.sort = query.$orderby.replace(/ desc/gi, '-').replace(/ asc/gi, '');
        if (query.$filter) params.filter = this._parseODataFilter(query.$filter);
        if (query.$select) params.fields = query.$select;
        
        return params;
    }

    _parseODataFilter(filter) {
        // Convert OData filter to REST query parameters
        return filter
            .replace(/ eq /g, '=')
            .replace(/ ne /g, '!=')
            .replace(/ gt /g, '>')
            .replace(/ ge /g, '>=')
            .replace(/ lt /g, '<')
            .replace(/ le /g, '<=')
            .replace(/ and /g, '&')
            .replace(/ or /g, '|');
    }

    _convertRESTToOData(data, entityType) {
        if (Array.isArray(data)) {
            return data.map(item => this._convertRESTItemToOData(item, entityType));
        }
        return this._convertRESTItemToOData(data, entityType);
    }

    _convertRESTItemToOData(item, entityType) {
        switch (entityType) {
            case 'ReasoningTask':
                return this._convertRESTReasoningTaskToOData(item);
            case 'KnowledgeBaseElement':
                return this._convertRESTKnowledgeElementToOData(item);
            case 'LogicalInference':
                return this._convertRESTInferenceToOData(item);
            case 'DecisionRecord':
                return this._convertRESTDecisionToOData(item);
            case 'ProblemSolvingRecord':
                return this._convertRESTProblemToOData(item);
            case 'ReasoningEngine':
                return this._convertRESTEngineToOData(item);
            case 'ReasoningPerformanceMetric':
                return this._convertRESTMetricToOData(item);
            default:
                return item;
        }
    }

    // ===== ENTITY CONVERSION METHODS =====
    _convertODataReasoningTaskToREST(data) {
        return {
            task_name: data.taskName,
            description: data.description,
            reasoning_type: data.reasoningType?.toLowerCase(),
            problem_domain: data.problemDomain?.toLowerCase(),
            reasoning_engine: data.reasoningEngine?.toLowerCase(),
            priority: data.priority?.toLowerCase(),
            initial_premises: data.initialPremises,
            target_conclusions: data.targetConclusions,
            configuration: data.configuration,
            metadata: data.metadata
        };
    }

    _convertRESTReasoningTaskToOData(item) {
        return {
            ID: item.id,
            taskName: item.task_name,
            description: item.description,
            reasoningType: item.reasoning_type?.toUpperCase(),
            problemDomain: item.problem_domain?.toUpperCase(),
            reasoningEngine: item.reasoning_engine?.toUpperCase(),
            status: item.status?.toUpperCase(),
            priority: item.priority?.toUpperCase(),
            progress: item.progress || 0,
            initialPremises: item.initial_premises,
            targetConclusions: item.target_conclusions,
            currentConclusions: item.current_conclusions,
            confidenceLevel: item.confidence_level || 0,
            startTime: item.start_time,
            endTime: item.end_time,
            estimatedDuration: item.estimated_duration,
            actualDuration: item.actual_duration,
            inferencesGenerated: item.inferences_generated || 0,
            validationStatus: item.validation_status?.toUpperCase(),
            validationConfidence: item.validation_confidence || 0,
            validationResults: item.validation_results,
            configuration: item.configuration,
            metadata: item.metadata,
            createdAt: item.created_at,
            createdBy: item.created_by,
            modifiedAt: item.modified_at,
            modifiedBy: item.modified_by
        };
    }

    _convertODataKnowledgeElementToREST(data) {
        return {
            element_name: data.elementName,
            element_type: data.elementType?.toLowerCase(),
            content: data.content,
            domain: data.domain?.toLowerCase(),
            confidence_level: data.confidenceLevel,
            priority_weight: data.priorityWeight,
            source: data.source,
            metadata: data.metadata
        };
    }

    _convertRESTKnowledgeElementToOData(item) {
        return {
            ID: item.id,
            elementName: item.element_name,
            elementType: item.element_type?.toUpperCase(),
            content: item.content,
            domain: item.domain?.toUpperCase(),
            confidenceLevel: item.confidence_level || 0,
            priorityWeight: item.priority_weight || 0.5,
            source: item.source,
            isActive: item.is_active !== false,
            usageCount: item.usage_count || 0,
            lastUsed: item.last_used,
            createdAt: item.created_at,
            createdBy: item.created_by,
            modifiedAt: item.modified_at,
            modifiedBy: item.modified_by,
            metadata: item.metadata
        };
    }

    _convertODataInferenceToREST(data) {
        return {
            statement: data.statement,
            inference_type: data.inferenceType?.toLowerCase(),
            confidence: data.confidence,
            premises: data.premises,
            derivation_path: data.derivationPath,
            metadata: data.metadata
        };
    }

    _convertRESTInferenceToOData(item) {
        return {
            ID: item.id,
            task_ID: item.task_id,
            statement: item.statement,
            inferenceType: item.inference_type?.toUpperCase(),
            confidence: item.confidence || 0,
            premises: item.premises,
            derivationPath: item.derivation_path,
            validationStatus: item.validation_status?.toUpperCase(),
            validationConfidence: item.validation_confidence || 0,
            validationEvidence: item.validation_evidence,
            isActive: item.is_active !== false,
            createdAt: item.created_at,
            modifiedAt: item.modified_at,
            metadata: item.metadata
        };
    }

    _convertODataDecisionToREST(data) {
        return {
            decision_context: data.decisionContext,
            decision_criteria: data.decisionCriteria,
            recommended_action: data.recommendedAction,
            confidence: data.confidence,
            risk_assessment: data.riskAssessment,
            justification: data.justification,
            metadata: data.metadata
        };
    }

    _convertRESTDecisionToOData(item) {
        return {
            ID: item.id,
            task_ID: item.task_id,
            decisionContext: item.decision_context,
            decisionCriteria: item.decision_criteria,
            alternativesEvaluated: item.alternatives_evaluated || 0,
            recommendedAction: item.recommended_action,
            confidence: item.confidence || 0,
            riskAssessment: item.risk_assessment,
            impactAnalysis: item.impact_analysis,
            justification: item.justification,
            actualOutcome: item.actual_outcome,
            successRate: item.success_rate || 0,
            lessonsLearned: item.lessons_learned,
            status: item.status?.toUpperCase(),
            createdAt: item.created_at,
            evaluationDate: item.evaluation_date,
            metadata: item.metadata
        };
    }

    _convertODataProblemToREST(data) {
        return {
            problem_description: data.problemDescription,
            problem_type: data.problemType?.toLowerCase(),
            solving_strategy: data.solvingStrategy?.toLowerCase(),
            constraints: data.constraints,
            solution: data.solution,
            solution_steps: data.solutionSteps,
            metadata: data.metadata
        };
    }

    _convertRESTProblemToOData(item) {
        return {
            ID: item.id,
            problemDescription: item.problem_description,
            problemType: item.problem_type?.toUpperCase(),
            solvingStrategy: item.solving_strategy?.toUpperCase(),
            constraints: item.constraints,
            solution: item.solution,
            optimizedSolution: item.optimized_solution,
            solutionSteps: item.solution_steps,
            qualityScore: item.quality_score || 0,
            solutionOptimality: item.solution_optimality || 0,
            timeComplexity: item.time_complexity,
            spaceComplexity: item.space_complexity,
            improvementScore: item.improvement_score || 0,
            optimizationTradeoffs: item.optimization_tradeoffs,
            status: item.status?.toUpperCase(),
            createdAt: item.created_at,
            solvedAt: item.solved_at,
            modifiedAt: item.modified_at,
            metadata: item.metadata
        };
    }

    _convertRESTEngineToOData(item) {
        return {
            ID: item.id,
            engineName: item.engine_name,
            engineType: item.engine_type?.toUpperCase(),
            version: item.version,
            status: item.status?.toUpperCase(),
            configuration: item.configuration,
            capabilities: item.capabilities,
            supportedDomains: item.supported_domains,
            accuracyScore: item.accuracy_score || 0,
            performanceScore: item.performance_score || 0,
            reliabilityScore: item.reliability_score || 0,
            lastCalibration: item.last_calibration,
            lastOptimization: item.last_optimization,
            performanceGain: item.performance_gain || 0,
            calibrationResults: item.calibration_results,
            isActive: item.is_active !== false,
            createdAt: item.created_at,
            modifiedAt: item.modified_at,
            metadata: item.metadata
        };
    }

    _convertRESTMetricToOData(item) {
        return {
            ID: item.id,
            engine_ID: item.engine_id,
            metricName: item.metric_name,
            metricValue: item.metric_value || 0,
            metricUnit: item.metric_unit,
            timestamp: item.timestamp,
            category: item.category?.toUpperCase(),
            context: item.context,
            metadata: item.metadata
        };
    }

    _handleError(error) {
        const errorMessage = error.response?.data?.message || error.message || 'Unknown error occurred';
        const errorCode = error.response?.status || 500;
        
        const customError = new Error(`Agent 9 Reasoning service error: ${errorMessage}`);
        customError.statusCode = errorCode;
        customError.originalError = error;
        
        return customError;
    }
}

module.exports = Agent9Adapter;