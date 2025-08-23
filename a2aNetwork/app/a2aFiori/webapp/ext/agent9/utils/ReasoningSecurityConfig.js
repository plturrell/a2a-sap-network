sap.ui.define([], function () {
    "use strict";

    /**
     * Security Configuration for Agent 9 Reasoning Agent
     * Specific security policies for reasoning operations and knowledge management
     */
    return {
        
        // Reasoning-specific security settings
        reasoning: {
            maxInferenceDepth: 50, // Maximum depth to prevent logic bombs
            maxReasoningTime: 300000, // 5 minutes maximum reasoning time
            maxConcurrentReasoningTasks: 10,
            maxFactsPerTask: 10000,
            maxRulesPerEngine: 5000,
            maxContradictionsAllowed: 100,
            
            // Confidence thresholds
            confidenceThresholds: {
                minimum: 0.1,
                warning: 0.5,
                acceptable: 0.7,
                high: 0.85,
                certain: 0.95
            },
            
            // Rate limiting for reasoning operations
            rateLimiting: {
                maxReasoningRequestsPerMinute: 20,
                maxInferenceRequestsPerMinute: 50,
                maxKnowledgeUpdatesPerHour: 100,
                maxContradictionAnalysesPerHour: 30
            },
            
            // Allowed reasoning types
            allowedReasoningTypes: [
                'DEDUCTIVE',
                'INDUCTIVE',
                'ABDUCTIVE',
                'ANALOGICAL',
                'PROBABILISTIC',
                'CAUSAL',
                'TEMPORAL',
                'MODAL'
            ],
            
            // Blocked patterns in reasoning rules
            blockedRulePatterns: [
                /eval\(/gi,
                /Function\(/gi,
                /require\(/gi,
                /import\s+/gi,
                /__proto__/gi,
                /constructor\[/gi,
                /process\./gi,
                /child_process/gi,
                /fs\./gi,
                /exec\(/gi
            ]
        },

        // Knowledge base security
        knowledgeBase: {
            maxFactSize: 10000, // Maximum characters per fact
            maxRuleComplexity: 1000, // Maximum rule length
            maxOntologyDepth: 20,
            requireFactValidation: true,
            requireRuleValidation: true,
            auditAllModifications: true,
            
            // Access control for knowledge operations
            accessControl: {
                read: ['user', 'analyst', 'manager', 'admin'],
                create: ['analyst', 'manager', 'admin'],
                update: ['manager', 'admin'],
                delete: ['admin'],
                bulkOperations: ['admin']
            },
            
            // Data integrity checks
            integrityChecks: {
                validateConsistency: true,
                preventCircularReferences: true,
                checkContradictions: true,
                validateDependencies: true
            }
        },

        // Inference engine security
        inferenceEngine: {
            maxChainingDepth: 30,
            maxParallelInferences: 100,
            timeoutPerInference: 60000, // 1 minute per inference
            requireExplainability: true,
            
            // Chain validation
            chainValidation: {
                validatePremises: true,
                checkLogicalConsistency: true,
                preventInfiniteLoops: true,
                maxChainLength: 100
            },
            
            // Resource limits
            resourceLimits: {
                maxMemoryPerTask: 536870912, // 512MB
                maxCPUTimePerTask: 300000, // 5 minutes
                maxQueueSize: 1000
            }
        },

        // Decision making security
        decisionMaking: {
            requireMultipleCriteria: true,
            minAlternativesRequired: 2,
            maxDecisionComplexity: 1000,
            requireJustification: true,
            auditAllDecisions: true,
            
            // Risk assessment
            riskAssessment: {
                requireRiskAnalysis: true,
                maxAcceptableRisk: 0.3,
                requireFallbackOptions: true,
                escalateHighRiskDecisions: true
            }
        },

        // Contradiction handling security
        contradictionHandling: {
            maxContradictionsPerAnalysis: 500,
            requireResolutionJustification: true,
            preventMaliciousContradictions: true,
            
            // Resolution strategies
            allowedResolutionStrategies: [
                'CONFIDENCE_BASED',
                'TEMPORAL_PRECEDENCE',
                'SOURCE_AUTHORITY',
                'MANUAL_REVIEW'
            ],
            
            // Exploitation prevention
            exploitationPrevention: {
                detectPatternedContradictions: true,
                blockRepeatedContradictions: true,
                limitContradictionCreation: true
            }
        },

        // Validation rules
        validation: {
            // Input validation patterns
            patterns: {
                taskName: /^[a-zA-Z0-9\s\-_]{3,100}$/,
                factPattern: /^[a-zA-Z0-9\s\-_\.\,\(\)\[\]]{1,1000}$/,
                rulePattern: /^[a-zA-Z0-9\s\-_\.\,\(\)\[\]\&\|\!\=\>\<]{1,1000}$/,
                conclusionPattern: /^[a-zA-Z0-9\s\-_\.\,]{1,500}$/
            },
            
            // Semantic validation
            semantic: {
                checkLogicalValidity: true,
                validatePredicates: true,
                checkArgumentStructure: true,
                validateQuantifiers: true
            }
        },

        // Audit and monitoring
        monitoring: {
            // Events to monitor
            monitoredEvents: [
                'reasoning_started',
                'inference_generated',
                'contradiction_detected',
                'decision_made',
                'knowledge_updated',
                'validation_failed',
                'security_violation',
                'performance_anomaly'
            ],
            
            // Anomaly detection
            anomalyDetection: {
                unusualInferencePatterns: true,
                suspiciousContradictions: true,
                abnormalConfidenceScores: true,
                excessiveResourceUsage: true
            },
            
            // Alert thresholds
            alertThresholds: {
                failedReasoningAttempts: 5,
                contradictionSpikes: 10,
                lowConfidenceResults: 0.3,
                resourceExhaustion: 0.9
            }
        },

        // Security headers for reasoning operations
        securityHeaders: {
            'X-Reasoning-Depth-Limit': '50',
            'X-Inference-Timeout': '60000',
            'X-Knowledge-Integrity': 'required',
            'X-Contradiction-Protection': 'enabled',
            'X-Logic-Bomb-Protection': 'active'
        },

        /**
         * Validates reasoning task parameters
         * @param {Object} params - Task parameters
         * @returns {Object} - Validation result
         */
        validateReasoningTask: function(params) {
            const errors = [];
            
            // Check task name
            if (!this.validation.patterns.taskName.test(params.taskName)) {
                errors.push('Invalid task name format');
            }
            
            // Check reasoning type
            if (!this.reasoning.allowedReasoningTypes.includes(params.reasoningType)) {
                errors.push('Invalid reasoning type');
            }
            
            // Check confidence threshold
            if (params.confidenceThreshold < this.reasoning.confidenceThresholds.minimum) {
                errors.push('Confidence threshold too low');
            }
            
            // Check inference depth
            if (params.maxInferenceDepth > this.reasoning.maxInferenceDepth) {
                errors.push('Inference depth exceeds maximum allowed');
            }
            
            return {
                isValid: errors.length === 0,
                errors: errors
            };
        },

        /**
         * Checks if a reasoning rule is safe
         * @param {string} rule - Rule to check
         * @returns {boolean} - True if safe
         */
        isRuleSafe: function(rule) {
            for (let pattern of this.reasoning.blockedRulePatterns) {
                if (pattern.test(rule)) {
                    return false;
                }
            }
            
            return rule.length <= this.knowledgeBase.maxRuleComplexity;
        },

        /**
         * Checks if user has permission for knowledge operation
         * @param {string} operation - Operation type
         * @param {string} userRole - User's role
         * @returns {boolean} - True if permitted
         */
        hasKnowledgePermission: function(operation, userRole) {
            const allowedRoles = this.knowledgeBase.accessControl[operation];
            return allowedRoles && allowedRoles.includes(userRole);
        },

        /**
         * Calculates risk score for a decision
         * @param {Object} decision - Decision parameters
         * @returns {number} - Risk score between 0 and 1
         */
        calculateDecisionRisk: function(decision) {
            let riskScore = 0;
            
            // Factor in confidence
            if (decision.confidence < this.reasoning.confidenceThresholds.acceptable) {
                riskScore += 0.3;
            }
            
            // Factor in alternatives
            if (!decision.alternatives || decision.alternatives.length < 2) {
                riskScore += 0.2;
            }
            
            // Factor in impact
            if (decision.impactLevel === 'HIGH') {
                riskScore += 0.3;
            }
            
            // Factor in reversibility
            if (!decision.reversible) {
                riskScore += 0.2;
            }
            
            return Math.min(1, riskScore);
        }
    };
});