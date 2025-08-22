/**
 * Agent 6 Adapter - Quality Control & Workflow Routing
 * Converts between REST API and OData formats for quality assessment and routing operations
 */

const axios = require('axios');
const { v4: uuidv4 } = require('uuid');

class Agent6Adapter {
    constructor() {
        this.baseUrl = process.env.AGENT6_BASE_URL || 'http://localhost:8005';
        this.apiVersion = 'v1';
        this.timeout = 30000;
    }

    // Quality Control Tasks
    async getQualityControlTasks(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/quality-control/tasks`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'QualityControlTask');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createQualityControlTask(data) {
        try {
            const restData = this._convertODataTaskToREST(data);
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/quality-control/tasks`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTTaskToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateQualityControlTask(id, data) {
        try {
            const restData = this._convertODataTaskToREST(data);
            const response = await axios.put(`${this.baseUrl}/api/${this.apiVersion}/quality-control/tasks/${id}`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTTaskToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async deleteQualityControlTask(id) {
        try {
            await axios.delete(`${this.baseUrl}/api/${this.apiVersion}/quality-control/tasks/${id}`, {
                timeout: this.timeout
            });
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Quality Assessment
    async startQualityAssessment(taskId) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/quality-control/tasks/${taskId}/assess`, {}, {
                timeout: this.timeout
            });
            
            return {
                assessmentId: response.data.assessment_id,
                status: response.data.status,
                startTime: response.data.start_time,
                estimatedDuration: response.data.estimated_duration
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async performQualityAssessment(taskId, criteria) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/quality-control/assess`, {
                task_id: taskId,
                criteria: this._convertCriteriaToREST(criteria)
            }, {
                timeout: this.timeout
            });
            
            return {
                overallScore: response.data.overall_score,
                details: {
                    accuracy: response.data.details.accuracy,
                    completeness: response.data.details.completeness,
                    consistency: response.data.details.consistency,
                    reliability: response.data.details.reliability,
                    performance: response.data.details.performance
                },
                issues: response.data.issues?.map(issue => ({
                    type: issue.type,
                    severity: issue.severity,
                    description: issue.description,
                    recommendation: issue.recommendation
                })) || [],
                recommendation: response.data.recommendation
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Routing Operations
    async makeRoutingDecision(taskId, decision) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/quality-control/tasks/${taskId}/route`, {
                decision: decision.decision,
                target_agent: decision.targetAgent,
                confidence: decision.confidence,
                reason: decision.reason
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                routingId: response.data.routing_id,
                estimatedProcessingTime: response.data.estimated_processing_time
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getRoutingRecommendations(taskId) {
        try {
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/quality-control/tasks/${taskId}/routing-recommendations`, {
                timeout: this.timeout
            });
            
            return response.data.recommendations.map(rec => ({
                agent: rec.agent,
                confidence: rec.confidence,
                score: rec.score,
                reasoning: rec.reasoning,
                estimatedProcessingTime: rec.estimated_processing_time,
                historicalSuccessRate: rec.historical_success_rate
            }));
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Trust Verification
    async verifyTrust(taskId) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/quality-control/tasks/${taskId}/verify-trust`, {}, {
                timeout: this.timeout
            });
            
            return {
                overallScore: response.data.overall_score,
                trustLevel: response.data.trust_level,
                factors: response.data.factors.map(factor => ({
                    name: factor.name,
                    score: factor.score,
                    weight: factor.weight,
                    status: factor.status,
                    details: factor.details
                })),
                blockchainHash: response.data.blockchain_hash,
                consensusResult: response.data.consensus_result,
                anomaliesDetected: response.data.anomalies_detected,
                method: response.data.verification_method
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Workflow Optimization
    async optimizeWorkflow(taskId, optimization) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/quality-control/workflow/optimize`, {
                task_id: taskId,
                optimization_type: optimization.optimizationType,
                parameters: optimization.parameters
            }, {
                timeout: this.timeout
            });
            
            return {
                bottlenecks: response.data.bottlenecks.map(b => ({
                    stage: b.stage,
                    type: b.type,
                    duration: b.duration,
                    impact: b.impact,
                    rootCause: b.root_cause,
                    suggestedFix: b.suggested_fix
                })),
                recommendations: response.data.recommendations.map(r => ({
                    name: r.name,
                    description: r.description,
                    expectedImprovement: r.expected_improvement,
                    risk: r.risk,
                    effort: r.effort,
                    priority: r.priority
                })),
                expectedImprovement: response.data.expected_improvement,
                implementationPlan: response.data.implementation_plan
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async analyzeWorkflowBottlenecks(workflowId) {
        try {
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/quality-control/workflow/${workflowId}/bottlenecks`, {
                timeout: this.timeout
            });
            
            return {
                workflowId: response.data.workflow_id,
                analysisTime: response.data.analysis_time,
                bottlenecks: response.data.bottlenecks.map(b => ({
                    id: b.id,
                    location: b.location,
                    severity: b.severity,
                    waitTime: b.wait_time,
                    processingTime: b.processing_time,
                    queueLength: b.queue_length,
                    impact: b.impact
                })),
                overallEfficiency: response.data.overall_efficiency,
                recommendations: response.data.recommendations
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Task Escalation
    async escalateTask(taskId, escalation) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/quality-control/tasks/${taskId}/escalate`, {
                escalation_level: escalation.escalationLevel,
                reason: escalation.reason
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                escalationId: response.data.escalation_id,
                assignedTo: response.data.assigned_to
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Quality Metrics
    async getQualityMetrics(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/quality-control/metrics`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'QualityMetric');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Routing Rules
    async getRoutingRules(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/quality-control/routing-rules`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'RoutingRule');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createRoutingRule(data) {
        try {
            const restData = {
                rule_name: data.ruleName,
                condition_type: data.conditionType,
                condition_value: data.conditionValue,
                target_agent: data.targetAgent,
                priority: data.priority,
                enabled: data.enabled,
                metadata: data.metadata
            };
            
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/quality-control/routing-rules`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTRuleToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Trust Verifications
    async getTrustVerifications(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/quality-control/trust-verifications`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'TrustVerification');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Quality Gates
    async getQualityGates(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/quality-control/quality-gates`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'QualityGate');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Workflow Optimizations
    async getWorkflowOptimizations(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/quality-control/workflow-optimizations`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'WorkflowOptimization');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Streaming support for real-time assessment
    async* streamQualityAssessment(taskId) {
        try {
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/quality-control/tasks/${taskId}/assessment-stream`, {
                responseType: 'stream',
                timeout: 0
            });
            
            for await (const chunk of response.data) {
                const lines = chunk.toString().split('\n').filter(line => line.trim());
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.substring(6));
                        yield {
                            phase: data.phase,
                            progress: data.progress,
                            currentScore: data.current_score,
                            details: data.details,
                            issues: data.issues
                        };
                    }
                }
            }
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Utility methods
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
            case 'QualityControlTask':
                return this._convertRESTTaskToOData(item);
            case 'QualityMetric':
                return this._convertRESTMetricToOData(item);
            case 'RoutingRule':
                return this._convertRESTRuleToOData(item);
            case 'TrustVerification':
                return this._convertRESTVerificationToOData(item);
            case 'QualityGate':
                return this._convertRESTGateToOData(item);
            case 'WorkflowOptimization':
                return this._convertRESTOptimizationToOData(item);
            default:
                return item;
        }
    }

    _convertRESTTaskToOData(task) {
        return {
            ID: task.id || uuidv4(),
            taskName: task.task_name,
            qualityGate: task.quality_gate,
            status: task.status?.toUpperCase() || 'DRAFT',
            overallQuality: task.overall_quality,
            trustScore: task.trust_score,
            routingDecision: task.routing_decision?.toUpperCase(),
            currentAgent: task.current_agent,
            targetAgent: task.target_agent,
            processingTime: task.processing_time,
            issuesFound: task.issues_found,
            issuesSeverity: task.issues_severity?.toUpperCase(),
            completenessScore: task.completeness_score,
            accuracyScore: task.accuracy_score,
            consistencyScore: task.consistency_score,
            reliabilityScore: task.reliability_score,
            performanceScore: task.performance_score,
            metadata: JSON.stringify(task.metadata || {}),
            assessmentStartTime: task.assessment_start_time,
            assessmentEndTime: task.assessment_end_time,
            routingTimestamp: task.routing_timestamp,
            escalationLevel: task.escalation_level,
            escalationReason: task.escalation_reason,
            recommendations: JSON.stringify(task.recommendations || []),
            createdAt: task.created_at,
            createdBy: task.created_by,
            modifiedAt: task.modified_at,
            modifiedBy: task.modified_by
        };
    }

    _convertODataTaskToREST(task) {
        const restTask = {
            task_name: task.taskName,
            quality_gate: task.qualityGate,
            status: task.status?.toLowerCase()
        };
        
        if (task.overallQuality !== undefined) restTask.overall_quality = task.overallQuality;
        if (task.trustScore !== undefined) restTask.trust_score = task.trustScore;
        if (task.routingDecision) restTask.routing_decision = task.routingDecision.toLowerCase();
        if (task.currentAgent) restTask.current_agent = task.currentAgent;
        if (task.targetAgent) restTask.target_agent = task.targetAgent;
        if (task.metadata) restTask.metadata = JSON.parse(task.metadata);
        
        return restTask;
    }

    _convertRESTMetricToOData(metric) {
        return {
            ID: metric.id || uuidv4(),
            taskId: metric.task_id,
            metricType: metric.metric_type?.toUpperCase(),
            value: metric.value,
            unit: metric.unit,
            threshold: metric.threshold,
            status: metric.status?.toUpperCase(),
            metadata: JSON.stringify(metric.metadata || {}),
            measuredAt: metric.measured_at
        };
    }

    _convertRESTRuleToOData(rule) {
        return {
            ID: rule.id || uuidv4(),
            ruleName: rule.rule_name,
            description: rule.description,
            conditionType: rule.condition_type?.toUpperCase(),
            conditionValue: rule.condition_value,
            targetAgent: rule.target_agent,
            priority: rule.priority,
            enabled: rule.enabled,
            successRate: rule.success_rate,
            lastTriggered: rule.last_triggered,
            metadata: JSON.stringify(rule.metadata || {}),
            createdAt: rule.created_at,
            createdBy: rule.created_by,
            modifiedAt: rule.modified_at,
            modifiedBy: rule.modified_by
        };
    }

    _convertRESTVerificationToOData(verification) {
        return {
            ID: verification.id || uuidv4(),
            taskId: verification.task_id,
            verificationMethod: verification.verification_method?.toUpperCase(),
            overallScore: verification.overall_score,
            trustLevel: verification.trust_level?.toUpperCase(),
            factors: JSON.stringify(verification.factors || []),
            blockchainHash: verification.blockchain_hash,
            blockchainBlockNumber: verification.blockchain_block_number,
            consensusResult: verification.consensus_result,
            consensusParticipants: verification.consensus_participants,
            anomaliesDetected: verification.anomalies_detected,
            anomalies: JSON.stringify(verification.anomalies || []),
            verificationTime: verification.verification_time,
            expiresAt: verification.expires_at
        };
    }

    _convertRESTGateToOData(gate) {
        return {
            ID: gate.id || uuidv4(),
            gateName: gate.gate_name,
            description: gate.description,
            gateType: gate.gate_type?.toUpperCase(),
            thresholdValue: gate.threshold_value,
            comparisonOperator: gate.comparison_operator,
            enabled: gate.enabled,
            criticalGate: gate.critical_gate,
            metadata: JSON.stringify(gate.metadata || {}),
            createdAt: gate.created_at,
            createdBy: gate.created_by,
            modifiedAt: gate.modified_at,
            modifiedBy: gate.modified_by
        };
    }

    _convertRESTOptimizationToOData(optimization) {
        return {
            ID: optimization.id || uuidv4(),
            taskId: optimization.task_id,
            workflowId: optimization.workflow_id,
            optimizationType: optimization.optimization_type?.toUpperCase(),
            bottlenecks: JSON.stringify(optimization.bottlenecks || []),
            recommendations: JSON.stringify(optimization.recommendations || []),
            expectedImprovement: optimization.expected_improvement,
            actualImprovement: optimization.actual_improvement,
            implementationStatus: optimization.implementation_status?.toUpperCase(),
            implementationDate: optimization.implementation_date,
            parameters: JSON.stringify(optimization.parameters || {}),
            results: JSON.stringify(optimization.results || {}),
            createdAt: optimization.created_at,
            createdBy: optimization.created_by,
            modifiedAt: optimization.modified_at,
            modifiedBy: optimization.modified_by
        };
    }

    _convertCriteriaToREST(criteria) {
        if (!criteria) return {};
        
        return {
            accuracy_weight: criteria.accuracyWeight || 20,
            completeness_weight: criteria.completenessWeight || 20,
            consistency_weight: criteria.consistencyWeight || 20,
            reliability_weight: criteria.reliabilityWeight || 20,
            performance_weight: criteria.performanceWeight || 20,
            custom_criteria: criteria.customCriteria
        };
    }

    _handleError(error) {
        if (error.response) {
            const status = error.response.status;
            const message = error.response.data?.message || error.message;
            
            switch (status) {
                case 400:
                    return new Error(`Bad Request: ${message}`);
                case 401:
                    return new Error(`Unauthorized: ${message}`);
                case 403:
                    return new Error(`Forbidden: ${message}`);
                case 404:
                    return new Error(`Not Found: ${message}`);
                case 500:
                    return new Error(`Internal Server Error: ${message}`);
                default:
                    return new Error(`HTTP ${status}: ${message}`);
            }
        } else if (error.request) {
            return new Error(`No response from Agent 6 service: ${error.message}`);
        } else {
            return new Error(`Agent 6 adapter error: ${error.message}`);
        }
    }
}

module.exports = Agent6Adapter;