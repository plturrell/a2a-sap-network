/**
 * Agent 5 QA Validation Adapter
 * Converts between REST API format (Python backend) and OData format (SAP CAP)
 */

const cds = require('@sap/cds');
const axios = require('axios');

const log = cds.log('agent5-adapter');

class Agent5Adapter {
    constructor() {
        this.baseUrl = process.env.AGENT5_BASE_URL || 'http://localhost:8004';
        log.info(`Agent 5 Adapter initialized with base URL: ${this.baseUrl}`);
    }

    // =================================
    // QA Validation Tasks Conversion
    // =================================

    /**
     * Convert REST QA validation task to OData format
     */
    convertTaskToOData(restTask) {
        return {
            ID: restTask.id || cds.utils.uuid(),
            taskName: restTask.task_name,
            description: restTask.description,
            dataProductId: restTask.data_product_id,
            ordRegistryUrl: restTask.ord_registry_url,
            validationType: this.mapValidationType(restTask.validation_type),
            qaScope: this.mapQAScope(restTask.qa_scope),
            testGenerationMethod: this.mapTestGenerationMethod(restTask.test_generation_method),
            simpleQaTestCount: restTask.simple_qa_test_count || 10,
            qualityThreshold: restTask.quality_threshold || 0.8,
            factualityThreshold: restTask.factuality_threshold || 0.85,
            complianceThreshold: restTask.compliance_threshold || 0.95,
            vectorSimilarityThreshold: restTask.vector_similarity_threshold || 0.7,
            enableFactualityTesting: restTask.enable_factuality_testing !== false,
            enableComplianceCheck: restTask.enable_compliance_check !== false,
            enableVectorSimilarity: restTask.enable_vector_similarity !== false,
            enableRegressionTesting: restTask.enable_regression_testing || false,
            requireApproval: restTask.require_approval !== false,
            status: this.mapStatus(restTask.status),
            priority: this.mapPriority(restTask.priority),
            progressPercent: restTask.progress || 0,
            currentStage: restTask.current_stage,
            overallScore: restTask.overall_score,
            qualityScore: restTask.quality_score,
            factualityScore: restTask.factuality_score,
            complianceScore: restTask.compliance_score,
            testsGenerated: restTask.tests_generated || 0,
            testsPassed: restTask.tests_passed || 0,
            testsFailed: restTask.tests_failed || 0,
            validationTime: restTask.validation_time,
            approvalStatus: this.mapApprovalStatus(restTask.approval_status),
            approvedBy: restTask.approved_by,
            approvedAt: restTask.approved_at,
            rejectionReason: restTask.rejection_reason,
            errorDetails: restTask.error_details,
            startedAt: restTask.started_at,
            completedAt: restTask.completed_at,
            createdBy: restTask.created_by,
            createdAt: restTask.created_at || new Date().toISOString(),
            modifiedAt: restTask.modified_at || new Date().toISOString(),
            createdBy_ID: restTask.created_by_id,
            modifiedBy_ID: restTask.modified_by_id
        };
    }

    /**
     * Convert OData QA validation task to REST format
     */
    convertODataToTask(odataTask) {
        return {
            id: odataTask.ID,
            task_name: odataTask.taskName,
            description: odataTask.description,
            data_product_id: odataTask.dataProductId,
            ord_registry_url: odataTask.ordRegistryUrl,
            validation_type: odataTask.validationType?.toLowerCase(),
            qa_scope: odataTask.qaScope?.toLowerCase(),
            test_generation_method: odataTask.testGenerationMethod?.toLowerCase(),
            simple_qa_test_count: odataTask.simpleQaTestCount,
            quality_threshold: odataTask.qualityThreshold,
            factuality_threshold: odataTask.factualityThreshold,
            compliance_threshold: odataTask.complianceThreshold,
            vector_similarity_threshold: odataTask.vectorSimilarityThreshold,
            enable_factuality_testing: odataTask.enableFactualityTesting,
            enable_compliance_check: odataTask.enableComplianceCheck,
            enable_vector_similarity: odataTask.enableVectorSimilarity,
            enable_regression_testing: odataTask.enableRegressionTesting,
            require_approval: odataTask.requireApproval,
            status: odataTask.status?.toLowerCase(),
            priority: odataTask.priority?.toLowerCase(),
            progress: odataTask.progressPercent,
            current_stage: odataTask.currentStage,
            overall_score: odataTask.overallScore,
            quality_score: odataTask.qualityScore,
            factuality_score: odataTask.factualityScore,
            compliance_score: odataTask.complianceScore,
            tests_generated: odataTask.testsGenerated,
            tests_passed: odataTask.testsPassed,
            tests_failed: odataTask.testsFailed,
            validation_time: odataTask.validationTime,
            approval_status: odataTask.approvalStatus?.toLowerCase(),
            approved_by: odataTask.approvedBy,
            approved_at: odataTask.approvedAt,
            rejection_reason: odataTask.rejectionReason,
            error_details: odataTask.errorDetails,
            started_at: odataTask.startedAt,
            completed_at: odataTask.completedAt,
            created_by: odataTask.createdBy,
            created_at: odataTask.createdAt,
            modified_at: odataTask.modifiedAt,
            created_by_id: odataTask.createdBy_ID,
            modified_by_id: odataTask.modifiedBy_ID
        };
    }

    // =================================
    // QA Validation Rules Conversion
    // =================================

    /**
     * Convert REST validation rule to OData format
     */
    convertRuleToOData(restRule) {
        return {
            ID: restRule.id || cds.utils.uuid(),
            ruleName: restRule.rule_name,
            description: restRule.description,
            ruleCategory: this.mapRuleCategory(restRule.rule_category),
            ruleType: this.mapRuleType(restRule.rule_type),
            ruleExpression: restRule.rule_expression,
            expectedResult: restRule.expected_result,
            severityLevel: this.mapSeverityLevel(restRule.severity_level),
            isActive: restRule.is_active !== false,
            isBlocking: restRule.is_blocking || false,
            executionOrder: restRule.execution_order || 100,
            timeoutSeconds: restRule.timeout_seconds || 30,
            successRate: restRule.success_rate || 0,
            usageCount: restRule.usage_count || 0,
            lastRunDate: restRule.last_run_date,
            tags: restRule.tags,
            createdAt: restRule.created_at || new Date().toISOString(),
            modifiedAt: restRule.modified_at || new Date().toISOString(),
            createdBy_ID: restRule.created_by_id,
            modifiedBy_ID: restRule.modified_by_id
        };
    }

    /**
     * Convert OData validation rule to REST format
     */
    convertODataToRule(odataRule) {
        return {
            id: odataRule.ID,
            rule_name: odataRule.ruleName,
            description: odataRule.description,
            rule_category: odataRule.ruleCategory?.toLowerCase(),
            rule_type: odataRule.ruleType?.toLowerCase(),
            rule_expression: odataRule.ruleExpression,
            expected_result: odataRule.expectedResult,
            severity_level: odataRule.severityLevel?.toLowerCase(),
            is_active: odataRule.isActive,
            is_blocking: odataRule.isBlocking,
            execution_order: odataRule.executionOrder,
            timeout_seconds: odataRule.timeoutSeconds,
            success_rate: odataRule.successRate,
            usage_count: odataRule.usageCount,
            last_run_date: odataRule.lastRunDate,
            tags: odataRule.tags,
            created_at: odataRule.createdAt,
            modified_at: odataRule.modifiedAt,
            created_by_id: odataRule.createdBy_ID,
            modified_by_id: odataRule.modifiedBy_ID
        };
    }

    // =================================
    // QA Test Results Conversion
    // =================================

    /**
     * Convert REST test result to OData format
     */
    convertTestResultToOData(restResult) {
        return {
            ID: restResult.id || cds.utils.uuid(),
            testId: restResult.test_id,
            testName: restResult.test_name,
            testType: this.mapTestType(restResult.test_type),
            testQuestion: restResult.test_question,
            expectedAnswer: restResult.expected_answer,
            actualAnswer: restResult.actual_answer,
            isPassed: restResult.is_passed || false,
            confidenceScore: restResult.confidence_score || 0,
            processingTime: restResult.processing_time,
            errorMessage: restResult.error_message,
            testData: JSON.stringify(restResult.test_data || {}),
            validationDetails: JSON.stringify(restResult.validation_details || {}),
            createdAt: restResult.created_at || new Date().toISOString(),
            task_ID: restResult.task_id
        };
    }

    // =================================
    // QA Approval Workflows Conversion
    // =================================

    /**
     * Convert REST approval workflow to OData format
     */
    convertApprovalToOData(restApproval) {
        return {
            ID: restApproval.id || cds.utils.uuid(),
            workflowName: restApproval.workflow_name,
            description: restApproval.description,
            status: this.mapWorkflowStatus(restApproval.status),
            currentStep: restApproval.current_step,
            currentStepIndex: restApproval.current_step_index || 1,
            totalSteps: restApproval.total_steps || 1,
            progress: restApproval.progress || 0,
            decision: this.mapDecision(restApproval.decision),
            approvalConditions: restApproval.approval_conditions,
            requiredChanges: restApproval.required_changes,
            rejectionReason: restApproval.rejection_reason,
            escalationReason: restApproval.escalation_reason,
            escalateTo: this.mapEscalateTo(restApproval.escalate_to),
            comments: restApproval.comments,
            priority: this.mapPriority(restApproval.priority),
            notifyStakeholders: restApproval.notify_stakeholders,
            canApprove: restApproval.can_approve !== false,
            canReject: restApproval.can_reject !== false,
            canRequestInfo: restApproval.can_request_info !== false,
            canEscalate: restApproval.can_escalate !== false,
            criteriaScore: restApproval.criteria_score || 0,
            autoApprovalMessage: restApproval.auto_approval_message,
            autoApprovalType: this.mapMessageType(restApproval.auto_approval_type),
            createdAt: restApproval.created_at || new Date().toISOString(),
            modifiedAt: restApproval.modified_at || new Date().toISOString(),
            task_ID: restApproval.task_id,
            createdBy_ID: restApproval.created_by_id,
            modifiedBy_ID: restApproval.modified_by_id
        };
    }

    /**
     * Convert OData approval workflow to REST format
     */
    convertODataToApproval(odataApproval) {
        return {
            id: odataApproval.ID,
            workflow_name: odataApproval.workflowName,
            description: odataApproval.description,
            status: odataApproval.status?.toLowerCase(),
            current_step: odataApproval.currentStep,
            current_step_index: odataApproval.currentStepIndex,
            total_steps: odataApproval.totalSteps,
            progress: odataApproval.progress,
            decision: odataApproval.decision,
            approval_conditions: odataApproval.approvalConditions,
            required_changes: odataApproval.requiredChanges,
            rejection_reason: odataApproval.rejectionReason,
            escalation_reason: odataApproval.escalationReason,
            escalate_to: odataApproval.escalateTo?.toLowerCase(),
            comments: odataApproval.comments,
            priority: odataApproval.priority?.toLowerCase(),
            notify_stakeholders: odataApproval.notifyStakeholders,
            can_approve: odataApproval.canApprove,
            can_reject: odataApproval.canReject,
            can_request_info: odataApproval.canRequestInfo,
            can_escalate: odataApproval.canEscalate,
            criteria_score: odataApproval.criteriaScore,
            auto_approval_message: odataApproval.autoApprovalMessage,
            auto_approval_type: odataApproval.autoApprovalType?.toLowerCase(),
            created_at: odataApproval.createdAt,
            modified_at: odataApproval.modifiedAt,
            task_id: odataApproval.task_ID,
            created_by_id: odataApproval.createdBy_ID,
            modified_by_id: odataApproval.modifiedBy_ID
        };
    }

    // =================================
    // Utility Result Conversions
    // =================================

    convertValidationResultToOData(restResult) {
        return {
            ID: cds.utils.uuid(),
            status: restResult.status?.toUpperCase() || 'COMPLETED',
            overallScore: restResult.overall_score,
            qualityScore: restResult.quality_score,
            factualityScore: restResult.factuality_score,
            complianceScore: restResult.compliance_score,
            testsGenerated: restResult.tests_generated || 0,
            testsPassed: restResult.tests_passed || 0,
            testsFailed: restResult.tests_failed || 0,
            validationTime: restResult.validation_time,
            results: JSON.stringify(restResult.results || {}),
            timestamp: new Date().toISOString()
        };
    }

    convertRuleTestResultToOData(restResult) {
        return {
            ID: cds.utils.uuid(),
            status: restResult.status?.toUpperCase() || 'PASSED',
            executionTime: restResult.execution_time,
            result: restResult.result,
            errorMessage: restResult.error_message,
            output: restResult.output,
            timestamp: new Date().toISOString()
        };
    }

    convertTestGenerationResultToOData(restResult) {
        return {
            ID: cds.utils.uuid(),
            testsGenerated: restResult.tests_generated || 0,
            testTypes: restResult.test_types,
            generationTime: restResult.generation_time,
            qualityMetrics: JSON.stringify(restResult.quality_metrics || {}),
            tests: JSON.stringify(restResult.tests || []),
            timestamp: new Date().toISOString()
        };
    }

    convertORDDiscoveryResultToOData(restResult) {
        return {
            ID: cds.utils.uuid(),
            registryUrl: restResult.registry_url,
            resourcesDiscovered: restResult.resources_discovered || 0,
            dataProducts: restResult.data_products || 0,
            apis: restResult.apis || 0,
            events: restResult.events || 0,
            discoveryTime: restResult.discovery_time,
            resources: JSON.stringify(restResult.resources || []),
            timestamp: new Date().toISOString()
        };
    }

    convertMetricsToOData(restMetrics) {
        return {
            ID: cds.utils.uuid(),
            totalTasks: restMetrics.total_tasks || 0,
            activeTasks: restMetrics.active_tasks || 0,
            completedTasks: restMetrics.completed_tasks || 0,
            averageQualityScore: restMetrics.average_quality_score || 0,
            averageFactualityScore: restMetrics.average_factuality_score || 0,
            averageComplianceScore: restMetrics.average_compliance_score || 0,
            totalTestsGenerated: restMetrics.total_tests_generated || 0,
            totalTestsPassed: restMetrics.total_tests_passed || 0,
            totalTestsFailed: restMetrics.total_tests_failed || 0,
            averageValidationTime: restMetrics.average_validation_time || 0,
            timestamp: new Date().toISOString()
        };
    }

    convertTrendsToOData(restTrends) {
        return {
            ID: cds.utils.uuid(),
            period: restTrends.period,
            qualityTrend: restTrends.quality_trend,
            factualityTrend: restTrends.factuality_trend,
            complianceTrend: restTrends.compliance_trend,
            volumeTrend: restTrends.volume_trend,
            performanceTrend: restTrends.performance_trend,
            data: JSON.stringify(restTrends.data || []),
            timestamp: new Date().toISOString()
        };
    }

    // =================================
    // Mapping Functions
    // =================================

    mapValidationType(type) {
        const mappings = {
            'factuality': 'FACTUALITY',
            'quality_assurance': 'QUALITY_ASSURANCE',
            'compliance': 'COMPLIANCE',
            'end_to_end': 'END_TO_END',
            'regression': 'REGRESSION',
            'integration': 'INTEGRATION'
        };
        return mappings[type?.toLowerCase()] || 'QUALITY_ASSURANCE';
    }

    mapQAScope(scope) {
        const mappings = {
            'data_integrity': 'DATA_INTEGRITY',
            'business_rules': 'BUSINESS_RULES',
            'regulatory_compliance': 'REGULATORY_COMPLIANCE',
            'performance': 'PERFORMANCE',
            'security': 'SECURITY',
            'completeness': 'COMPLETENESS'
        };
        return mappings[scope?.toLowerCase()] || 'DATA_INTEGRITY';
    }

    mapTestGenerationMethod(method) {
        const mappings = {
            'dynamic_simpleqa': 'DYNAMIC_SIMPLEQA',
            'static_rules': 'STATIC_RULES',
            'hybrid': 'HYBRID',
            'custom_template': 'CUSTOM_TEMPLATE'
        };
        return mappings[method?.toLowerCase()] || 'DYNAMIC_SIMPLEQA';
    }

    mapStatus(status) {
        const mappings = {
            'draft': 'DRAFT',
            'pending': 'PENDING',
            'in_progress': 'IN_PROGRESS',
            'running': 'RUNNING',
            'paused': 'PAUSED',
            'completed': 'COMPLETED',
            'failed': 'FAILED',
            'cancelled': 'CANCELLED'
        };
        return mappings[status?.toLowerCase()] || 'DRAFT';
    }

    mapPriority(priority) {
        const mappings = {
            'low': 'LOW',
            'medium': 'MEDIUM',
            'high': 'HIGH',
            'urgent': 'URGENT'
        };
        return mappings[priority?.toLowerCase()] || 'MEDIUM';
    }

    mapApprovalStatus(status) {
        const mappings = {
            'pending': 'PENDING',
            'approved': 'APPROVED',
            'conditional_approval': 'CONDITIONAL_APPROVAL',
            'rejected': 'REJECTED',
            'escalated': 'ESCALATED'
        };
        return mappings[status?.toLowerCase()] || 'PENDING';
    }

    mapRuleCategory(category) {
        const mappings = {
            'data_quality': 'DATA_QUALITY',
            'business_logic': 'BUSINESS_LOGIC',
            'regulatory': 'REGULATORY',
            'security': 'SECURITY',
            'performance': 'PERFORMANCE',
            'completeness': 'COMPLETENESS'
        };
        return mappings[category?.toLowerCase()] || 'DATA_QUALITY';
    }

    mapRuleType(type) {
        const mappings = {
            'simple_qa': 'SIMPLE_QA',
            'sql_query': 'SQL_QUERY',
            'python_script': 'PYTHON_SCRIPT',
            'rest_api_check': 'REST_API_CHECK',
            'regex_pattern': 'REGEX_PATTERN',
            'threshold_check': 'THRESHOLD_CHECK'
        };
        return mappings[type?.toLowerCase()] || 'SIMPLE_QA';
    }

    mapSeverityLevel(level) {
        const mappings = {
            'low': 'LOW',
            'medium': 'MEDIUM',
            'high': 'HIGH',
            'critical': 'CRITICAL'
        };
        return mappings[level?.toLowerCase()] || 'MEDIUM';
    }

    mapTestType(type) {
        const mappings = {
            'simple_qa': 'SIMPLE_QA',
            'factual': 'FACTUAL',
            'computational': 'COMPUTATIONAL',
            'relational': 'RELATIONAL',
            'compliance': 'COMPLIANCE'
        };
        return mappings[type?.toLowerCase()] || 'SIMPLE_QA';
    }

    mapWorkflowStatus(status) {
        const mappings = {
            'active': 'ACTIVE',
            'pending': 'PENDING',
            'in_review': 'IN_REVIEW',
            'approved': 'APPROVED',
            'rejected': 'REJECTED',
            'escalated': 'ESCALATED',
            'completed': 'COMPLETED'
        };
        return mappings[status?.toLowerCase()] || 'PENDING';
    }

    mapDecision(decision) {
        const mappings = {
            'approve': 0,
            'conditional_approval': 1,
            'request_changes': 2,
            'reject': 3,
            'escalate': 4
        };
        return mappings[decision?.toLowerCase()] || null;
    }

    mapEscalateTo(escalateTo) {
        const mappings = {
            'senior_qa': 'SENIOR_QA',
            'compliance_officer': 'COMPLIANCE_OFFICER',
            'data_governance': 'DATA_GOVERNANCE',
            'executive': 'EXECUTIVE'
        };
        return mappings[escalateTo?.toLowerCase()] || 'SENIOR_QA';
    }

    mapMessageType(type) {
        const mappings = {
            'success': 'Success',
            'information': 'Information',
            'warning': 'Warning',
            'error': 'Error'
        };
        return mappings[type?.toLowerCase()] || 'Information';
    }
}

module.exports = Agent5Adapter;