using a2a.network as db from '../db/schema';

@requires: ['authenticated-user', {kind: 'any', grant: 'READ'}]
service A2AService @(path: '/api/v1') {
    
    // SAP Fiori Launchpad Tile Data Endpoints - 100% Real Data
    // These replace the Express.js endpoints with proper CAP actions
    
    // Individual Agent Status (16 endpoints: /agents/0/status through /agents/15/status)
    @Common.Label: 'Agent Status'
    action getAgentStatus(agentId: Integer) returns {
        d: {
            title: String;
            number: String;
            numberUnit: String;
            numberState: String;
            subtitle: String;
            stateArrow: String;
            info: String;
            status: String;
            agent_id: String;
            version: String;
            port: Integer;
            capabilities: {
                skills: Integer;
                handlers: Integer;
                mcp_tools: Integer;
                mcp_resources: Integer;
            };
            performance: {
                cpu_usage: Decimal;
                memory_usage: Decimal;
                uptime_seconds: Decimal;
                success_rate: Decimal;
                avg_response_time_ms: Decimal;
                processed_today: Integer;
                error_rate: Decimal;
                queue_depth: Integer;
            };
            timestamp: String;
        };
    };
    
    // Network Overview Dashboard
    @Common.Label: 'Network Statistics'
    action getNetworkStats(id: String) returns {
        d: {
            title: String;
            number: String;
            numberUnit: String;
            numberState: String;
            subtitle: String;
            stateArrow: String;
            info: String;
            real_metrics: {
                healthy_agents: Integer;
                total_agents: Integer;
                agent_health_score: Integer;
                total_active_tasks: Integer;
                total_skills: Integer;
                total_mcp_tools: Integer;
                blockchain_status: String;
                blockchain_score: Integer;
                mcp_status: String;
                mcp_score: Integer;
                overall_system_health: Integer;
            };
            timestamp: String;
        };
    };
    
    // Blockchain Monitor
    @Common.Label: 'Blockchain Statistics'
    action getBlockchainStats(id: String) returns {
        d: {
            title: String;
            number: String;
            numberUnit: String;
            numberState: String;
            subtitle: String;
            stateArrow: String;
            info: String;
            blockchain_metrics: {
                network: String;
                contracts: {
                    registry: String;
                    message_router: String;
                };
                registered_agents_count: Integer;
                contract_count: Integer;
                trust_integration: Boolean;
                avg_trust_score: Decimal;
            };
            timestamp: String;
        };
    };
    
    // Service Marketplace
    @Common.Label: 'Service Count'
    action getServicesCount() returns {
        d: {
            title: String;
            number: String;
            numberUnit: String;
            numberState: String;
            subtitle: String;
            stateArrow: String;
            info: String;
            service_breakdown: {
                agent_skills: Integer;
                agent_handlers: Integer;
                mcp_tools: Integer;
                database_services: Integer;
                total_services: Integer;
            };
            provider_health: {
                active_providers: Integer;
                total_providers: Integer;
                provider_health_percentage: Integer;
            };
            timestamp: String;
        };
    };
    
    // System Health Summary
    @Common.Label: 'Health Summary'
    action getHealthSummary() returns {
        d: {
            title: String;
            number: String;
            numberUnit: String;
            numberState: String;
            subtitle: String;
            stateArrow: String;
            info: String;
            component_health: {
                agents_health: Integer;
                blockchain_health: Integer;
                mcp_health: Integer;
                api_health: Integer;
            };
            system_performance: {
                avg_cpu_usage: Decimal;
                avg_memory_usage: Decimal;
                network_latency: Integer;
            };
            error_tracking: {
                agent_error_rate: Decimal;
                blockchain_tx_failure_rate: Decimal;
                api_error_rate: Decimal;
            };
            timestamp: String;
        };
    };
    
    // Network Statistics for UI tiles - using projection from database
    
    // Agent Management
    // @odata.draft.enabled - Temporarily disabled for HANA Cloud compatibility
    @cds.redirection.target
    @requires: ['authenticated-user', {kind: 'any', grant: 'READ'}]
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE', 'UPDATE'], to: 'AgentManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity Agents as projection on db.Agents {
        ID,
        createdAt,
        createdBy,
        modifiedAt,
        modifiedBy,
        address,
        name,
        endpoint,
        reputation,
        isActive,
        country,
        capabilities : redirected to AgentCapabilities,
        services : redirected to Services,
        performance : redirected to AgentPerformance,
        reputationTransactions : redirected to ReputationTransactions,
        endorsementsReceived : redirected to PeerEndorsements,
        endorsementsGiven : redirected to PeerEndorsements,
        milestones : redirected to ReputationMilestones
    } actions {
        @requires: ['AgentManager', 'Admin']
        action registerOnBlockchain() returns String;
        @requires: ['AgentManager', 'Admin']
        action updateReputation(score: Integer) returns Boolean;
        @requires: ['Admin']
        action deactivate() returns Boolean;
        @requires: ['authenticated-user']
        action endorsePeer(
            toAgentId: String,
            amount: Integer,
            reason: String,
            description: String
        ) returns String;
    };
    
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE', 'UPDATE'], to: 'AgentManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity AgentCapabilities as projection on db.AgentCapabilities {
        *,
        agent : redirected to Agents
    };
    
    @readonly
    @requires: ['authenticated-user']
    entity AgentPerformance as projection on db.AgentPerformance;
    
    @readonly
    @requires: ['authenticated-user']
    entity PerformanceSnapshots as projection on db.PerformanceSnapshots;
    
    // Capability Registry
    // @odata.draft.enabled - Temporarily disabled for HANA Cloud compatibility
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE', 'UPDATE'], to: 'ServiceManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity Capabilities as projection on db.Capabilities actions {
        @requires: ['ServiceManager', 'Admin']
        action registerOnBlockchain() returns String;
        @requires: ['ServiceManager', 'Admin']
        action deprecate(replacementId: String) returns Boolean;
        @requires: ['Admin']
        action sunset() returns Boolean;
    };
    
    // Service Marketplace
    @cds.redirection.target
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE', 'UPDATE'], to: 'ServiceManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity Services as projection on db.Services {
        *,
        provider : redirected to Agents
    } actions {
        @requires: ['ServiceManager', 'Admin']
        action listOnMarketplace() returns String;
        @requires: ['ServiceManager', 'Admin']
        action updatePricing(newPrice: Decimal(10,4)) returns Boolean;
        @requires: ['Admin']
        action deactivate() returns Boolean;
    };
    
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE', 'UPDATE'], to: 'ServiceManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity ServiceOrders as projection on db.ServiceOrders {
        *,
        service : redirected to Services,
        consumer : redirected to Agents
    } actions {
        @requires: ['ServiceManager', 'Admin']
        action complete(rating: Integer, feedback: String) returns Boolean;
        @requires: ['ServiceManager', 'Admin']
        action dispute(reason: String) returns Boolean;
        @requires: ['Admin']
        action releaseEscrow() returns Boolean;
    };
    
    // Message Routing
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE'], to: 'MessageSender' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity Messages as projection on db.Messages {
        *,
        sender : redirected to Agents,
        recipient : redirected to Agents
    } actions {
        @requires: ['MessageSender', 'Admin']
        action retry() returns Boolean;
        @requires: ['MessageSender', 'Admin']
        action markDelivered() returns Boolean;
    };
    
    // Workflows
    // @odata.draft.enabled - Temporarily disabled for HANA Cloud compatibility
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE', 'UPDATE'], to: 'WorkflowManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity Workflows as projection on db.Workflows {
        *,
        steps : redirected to WorkflowSteps,
        executions: redirected to WorkflowExecutions
    } actions {
        @requires: ['WorkflowManager', 'Admin']
        action execute(parameters: String) returns String; // returns executionId
        @requires: ['WorkflowManager', 'Admin']
        action validate() returns Boolean;
        @requires: ['WorkflowManager', 'Admin']
        action publish() returns Boolean;
    };

    @cds.redirection.target
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE', 'UPDATE'], to: 'WorkflowManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity WorkflowExecutions as projection on db.WorkflowExecutions {
        *,
        workflowId : redirected to Workflows
    } actions {
        @requires: ['WorkflowManager', 'Admin']
        action cancel() returns Boolean;
        @requires: ['WorkflowManager', 'Admin']
        action retry() returns String;
    };

    @readonly
    @requires: ['authenticated-user']
    entity WorkflowSteps as projection on db.WorkflowSteps {
        *,
        workflowId: redirected to Workflows
    };
    
    // Cross-chain
    @requires: ['Admin']
    entity ChainBridges as projection on db.ChainBridges;
    
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity CrossChainTransfers as projection on db.CrossChainTransfers;
    
    // Privacy
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE', 'UPDATE'], to: 'PrivacyManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity PrivateChannels as projection on db.PrivateChannels actions {
        @requires: ['PrivacyManager', 'Admin']
        action addParticipant(agentAddress: String) returns Boolean;
        @requires: ['PrivacyManager', 'Admin']
        action removeParticipant(agentAddress: String) returns Boolean;
    };
    
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE'], to: 'PrivacyManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity PrivateMessages as projection on db.PrivateMessages;
    
    // Analytics Views
    @readonly
    @requires: ['authenticated-user']
    entity TopAgents as projection on db.TopAgents;
    
    @readonly
    @requires: ['authenticated-user']
    entity ActiveServices as projection on db.ActiveServices;
    
    @readonly
    @requires: ['authenticated-user']
    entity RecentWorkflows as projection on db.RecentWorkflows;
    
    @readonly
    @requires: ['authenticated-user']
    entity NetworkStats as projection on db.NetworkStats;
    
    // ================================
    // AGENT 2 - AI PREPARATION ENTITIES
    // ================================
    
    @cds.redirection.target
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE', 'UPDATE'], to: 'AIManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity AIPreparationTasks as projection on db.AIPreparationTasks {
        *,
        agent : redirected to Agents,
        features : redirected to AIPreparationFeatures
    } actions {
        @requires: ['AIManager', 'Admin']
        action startPreparation() returns String;
        @requires: ['AIManager', 'Admin']
        action pausePreparation() returns Boolean;
        @requires: ['AIManager', 'Admin']
        action resumePreparation() returns Boolean;
        @requires: ['AIManager', 'Admin']
        action cancelPreparation() returns Boolean;
        @requires: ['AIManager', 'Admin']
        action analyzeFeatures() returns String;
        @requires: ['AIManager', 'Admin']
        action generateEmbeddings(
            model: String,
            dimensions: Integer,
            normalization: Boolean,
            batchSize: Integer,
            useGPU: Boolean
        ) returns String;
        @requires: ['AIManager', 'Admin']
        action exportPreparedData(
            format: String,
            includeMetadata: Boolean,
            splitData: Boolean,
            compression: String
        ) returns String;
        @requires: ['AIManager', 'Admin']
        action optimizeHyperparameters(
            method: String,
            trials: Integer,
            timeout: Integer,
            earlyStop: Boolean
        ) returns String;
    };
    
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE', 'UPDATE'], to: 'AIManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity AIPreparationFeatures as projection on db.AIPreparationFeatures;
    
    // Agent 2 specific actions for data profiling and AutoML
    @requires: ['AIManager', 'Admin']
    action getDataProfile() returns String;
    
    @requires: ['AIManager', 'Admin'] 
    action batchPrepare(
        taskIds: array of String,
        parallel: Boolean,
        gpuAcceleration: Boolean
    ) returns String;
    
    @requires: ['AIManager', 'Admin']
    action startAutoML(
        dataset: String,
        problemType: String,
        targetColumn: String,
        evaluationMetric: String,
        timeLimit: Integer,
        maxModels: Integer,
        includeEnsemble: Boolean,
        crossValidation: Integer
    ) returns String;
    
    // ================================
    // REPUTATION SYSTEM ENTITIES
    // ================================
    
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['READ'], to: 'authenticated-user' },
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' }
    ]
    entity ReputationTransactions as projection on db.ReputationTransactions {
            *,
            agent : redirected to Agents,
            createdByAgent : redirected to Agents,
            serviceOrder : redirected to ServiceOrders,
            workflow : redirected to Workflows
        } actions {
            @requires: ['authenticated-user']
            action verify() returns Boolean;
        };
        
        @requires: ['authenticated-user']
        @restrict: [
            { grant: ['CREATE'], to: 'authenticated-user' },
            { grant: ['READ'], to: 'authenticated-user' },
            { grant: ['UPDATE', 'DELETE'], to: 'Admin' }
        ]
        entity PeerEndorsements as projection on db.PeerEndorsements actions {
            @requires: ['authenticated-user']
            action verify() returns Boolean;
        };
        
        @readonly
        @requires: ['authenticated-user']
        entity ReputationMilestones as projection on db.ReputationMilestones;
        
        @requires: ['authenticated-user']
        @restrict: [
            { grant: ['CREATE'], to: 'authenticated-user' },
            { grant: ['READ'], to: 'authenticated-user' },
            { grant: ['UPDATE', 'DELETE'], to: 'Admin' }
        ]
        entity ReputationRecovery as projection on db.ReputationRecovery actions {
            @requires: ['authenticated-user']
            action startProgram() returns Boolean;
            @requires: ['authenticated-user']
            action updateProgress(progress: String) returns Boolean;
        };
        
        event ReputationMilestoneReached : { 
            target: db.ReputationMilestones 
        };

    entity DailyReputationLimits as projection on db.DailyReputationLimits;
    
    event ServiceOrderRated : {
        serviceOrderId: String;
        providerId: String;
        clientId: String;
        rating: Integer;
        feedback: String;
        reputationChange: Integer;
    };
    
    // ============================================
    // AGENT 6 - QUALITY CONTROL & WORKFLOW ROUTING
    // ============================================
    
    @requires: ['authenticated-user']
    entity QualityControlTasks as projection on db.QualityControlTasks {
        *,
        agent : redirected to Agents
    } actions {
        @requires: ['QualityManager', 'Admin']
        action startAssessment() returns String;
        @requires: ['QualityManager', 'Admin']
        action pauseAssessment() returns String;
        @requires: ['QualityManager', 'Admin']
        action resumeAssessment() returns String;
        @requires: ['QualityManager', 'Admin']
        action cancelAssessment() returns String;
        @requires: ['QualityManager', 'Admin']
        action makeRoutingDecision(
            decision: String,
            targetAgent: String,
            confidence: Decimal,
            reason: String,
            priority: String
        ) returns String;
        @requires: ['QualityManager', 'Admin']
        action verifyTrust() returns String;
        @requires: ['QualityManager', 'Admin']
        action optimizeWorkflow(
            analysisDepth: String,
            includeResourceOptimization: Boolean,
            applyOptimizations: Boolean
        ) returns String;
        @requires: ['QualityManager', 'Admin']
        action generateReport(
            reportType: String,
            includeCharts: Boolean,
            includeRecommendations: Boolean,
            format: String
        ) returns String;
        @requires: ['QualityManager', 'Admin']
        action escalateIssues(
            issues: array of String,
            priority: String,
            notifyStakeholders: Boolean
        ) returns String;
    };
    
    @requires: ['authenticated-user']
    entity QualityMetrics as projection on db.QualityMetrics;
    
    @requires: ['authenticated-user']
    entity RoutingRules as projection on db.RoutingRules actions {
        @requires: ['QualityManager', 'Admin']
        action testRule(
            testData: String
        ) returns String;
        @requires: ['QualityManager', 'Admin']
        action activateRule() returns String;
        @requires: ['QualityManager', 'Admin']
        action deactivateRule() returns String;
    };
    
    @requires: ['authenticated-user']
    entity TrustVerifications as projection on db.TrustVerifications;
    
    // Agent 6 Utility Functions
    @requires: ['authenticated-user']
    function getQualityDashboard() returns String;
    
    @requires: ['authenticated-user']
    function getRoutingOptions(taskId: String) returns String;
    
    @requires: ['authenticated-user']
    function getTrustMetrics() returns String;
    
    @requires: ['authenticated-user']
    function getWorkflowAnalysis(taskId: String) returns String;
    
    @requires: ['QualityManager', 'Admin']
    function runBatchAssessment(
        taskIds: array of String,
        assessmentType: String,
        parallel: Boolean
    ) returns String;
    
    @requires: ['authenticated-user']
    function getQualityMetrics(taskId: String) returns String;
    
    @requires: ['authenticated-user']
    function getAvailableAgents() returns String;
    
    @requires: ['authenticated-user']
    function getRoutingRecommendations(taskId: String) returns String;
    
    // Agent 6 Events
    event QualityAssessmentStarted : {
        taskId: String;
        assessmentType: String;
        timestamp: DateTime;
    };
    
    event QualityAssessmentCompleted : {
        taskId: String;
        overallScore: Decimal;
        issuesFound: Integer;
        timestamp: DateTime;
    };
    
    event RoutingDecisionMade : {
        taskId: String;
        decision: String;
        targetAgent: String;
        confidence: Decimal;
        timestamp: DateTime;
    };
    
    event TrustVerificationCompleted : {
        taskId: String;
        trustScore: Decimal;
        anomaliesDetected: Boolean;
        timestamp: DateTime;
    };
    
    event WorkflowOptimized : {
        taskId: String;
        optimizationType: String;
        expectedImprovement: Decimal;
        timestamp: DateTime;
    };
    
    event IssuesEscalated : {
        taskId: String;
        issueCount: Integer;
        priority: String;
        escalationId: String;
        timestamp: DateTime;
    };
}

// Agent 7 Service - Agent Management & Orchestration - TEMPORARILY DISABLED
/*
service Agent7Service {
    // Main entity for registered agents with full management capabilities
    entity RegisteredAgents as projection on db.RegisteredAgents;
    
    action registerAgent(
        @title: 'Agent Data'
        agentData: {
            agentName: String;
            agentType: String;
            endpointUrl: String;
            capabilities: String;
            configuration: String;
        }
    ) returns String;
    
    action updateStatus(
        @title: 'New Status'
        status: String,
        @title: 'Reason'
        reason: String
    ) returns String;
        
    action performHealthCheck() returns String;
    
    action updateConfiguration(
        @title: 'Configuration'
        configuration: String,
        @title: 'Restart Required'
        restartRequired: Boolean
    ) returns String;
    
    action deactivateAgent(
        @title: 'Reason'
        reason: String,
        @title: 'Graceful Shutdown'
        gracefulShutdown: Boolean
    ) returns String;
    
    action scheduleTask(
        @title: 'Task Data'
        taskData: {
            taskType: String;
            parameters: String;
            scheduledTime: DateTime;
            priority: String;
        }
    ) returns String;
    
    action assignWorkload(
        @title: 'Workload Data'
        workloadData: {
            workloadType: String;
            parameters: String;
            priority: String;
            expectedDuration: Integer;
        }
    ) returns String;
}
    
    // Management tasks for coordinating agent activities
    entity ManagementTasks as projection on db.ManagementTasks;
    
    action executeTask() returns String;
    
    action pauseTask() returns String;
    
    action resumeTask() returns String;
    
    action cancelTask(
        @title: 'Reason'
        reason: String
    ) returns String;
    
    action retryTask(
        @title: 'Force Retry'
        forceRetry: Boolean
    ) returns String;
    
    action rollbackTask() returns String;
    
    // Health monitoring for all agents
    entity AgentHealthChecks as projection on db.AgentHealthChecks;
    
    // Performance metrics tracking
    entity AgentPerformanceMetrics as projection on db.AgentPerformanceMetrics;
    
    // Agent coordination and orchestration
    entity AgentCoordination as projection on db.AgentCoordination;
    
    action activateCoordination() returns String;
    
    action pauseCoordination() returns String;
    
    action updateRules(
        @title: 'New Rules'
        rules: String
    ) returns String;
    
    action addAgent(
        @title: 'Agent ID'
        agentId: String,
        @title: 'Role'
        role: String
    ) returns String;
    
    action removeAgent(
        @title: 'Agent ID'
        agentId: String,
        @title: 'Graceful Removal'
        graceful: Boolean
    ) returns String;
    
    // Bulk operations for managing multiple agents
    entity BulkOperations as projection on db.BulkOperations;
    
    action executeBulkOperation() returns String;
    
    action rollbackOperation() returns String;
    
    action pauseOperation() returns String;
    
    action resumeOperation() returns String;
    
    // Agent management functions
    function getAgentTypes() returns array of {
        type: String;
        description: String;
        capabilities: String;
        requirements: String;
    };
    
    function getDashboardData() returns {
        totalAgents: Integer;
        activeAgents: Integer;
        healthyAgents: Integer;
        tasksInProgress: Integer;
        averageResponseTime: Decimal;
        systemLoad: Decimal;
        alerts: array of String;
        trends: String;
    };
    
    function getHealthStatus(
        @title: 'Agent ID'
        agentId: String
    ) returns {
        agentId: String;
        status: String;
        lastCheck: DateTime;
        responseTime: Integer;
        errorRate: Decimal;
        alerts: array of String;
        recommendations: array of String;
    };
    
    function getPerformanceAnalysis(
        @title: 'Agent ID'
        agentId: String,
        @title: 'Time Range'
        timeRange: String
    ) returns {
        agentId: String;
        metrics: array of {
            metricType: String;
            value: Decimal;
            trend: String;
            benchmark: Decimal;
        };
        bottlenecks: array of String;
        recommendations: array of String;
    };
    
    function getCoordinationStatus() returns {
        activeCoordinations: Integer;
        totalWorkflows: Integer;
        averageSuccess: Decimal;
        currentLoad: Decimal;
        connections: array of {
            source: String;
            target: String;
            status: String;
            latency: Integer;
        };
    };
    
    function getAgentCapabilities(
        @title: 'Agent Type'
        agentType: String
    ) returns {
        type: String;
        capabilities: array of String;
        supportedProtocols: array of String;
        requirements: String;
        limitations: String;
    };
    
    function validateConfiguration(
        @title: 'Configuration'
        configuration: String,
        @title: 'Agent Type'
        agentType: String
    ) returns {
        valid: Boolean;
        errors: array of String;
        warnings: array of String;
        suggestions: array of String;
    };
    
    function getLoadBalancingRecommendations() returns {
        strategy: String;
        distribution: array of {
            agentId: String;
            recommendedWeight: Integer;
            currentLoad: Decimal;
            capacity: Decimal;
        };
        expectedImprovement: Decimal;
    };
    
    // Event definitions for agent management
    event AgentRegistered : {
        agentId: String;
        agentName: String;
        agentType: String;
        timestamp: DateTime;
    };
    
    event AgentStatusChanged : {
        agentId: String;
        oldStatus: String;
        newStatus: String;
        reason: String;
        timestamp: DateTime;
    };
    
    event HealthCheckFailed : {
        agentId: String;
        checkType: String;
        errorDetails: String;
        alertLevel: String;
        timestamp: DateTime;
    };
    
    event PerformanceAnomaly : {
        agentId: String;
        metricType: String;
        currentValue: Decimal;
        threshold: Decimal;
        severity: String;
        timestamp: DateTime;
    };
    
    event TaskCompleted : {
        taskId: String;
        agentId: String;
        taskType: String;
        status: String;
        duration: Integer;
        timestamp: DateTime;
    };
    
    event CoordinationUpdated : {
        coordinationId: String;
        coordinationType: String;
        action: String;
        affectedAgents: Integer;
        timestamp: DateTime;
    };
    
    event BulkOperationCompleted : {
        operationId: String;
        operationType: String;
        targetCount: Integer;
        successfulCount: Integer;
        failedCount: Integer;
        duration: Integer;
        timestamp: DateTime;
    };
}
*/

// Agent 8 Service - Data Management Agent - TEMPORARILY DISABLED
/*
service Agent8Service {
    // Main entity for data management tasks with comprehensive operations
    entity DataTasks as projection on db.DataTasks actions {
        action executeTask() returns String;
        
        action pauseTask() returns String;
        
        action resumeTask() returns String;
        
        action cancelTask(
            @title: 'Reason'
            reason: String;
        ) returns String;
        
        action retryTask() returns String;
        
        action optimizeTask(
            @title: 'Optimization Type'
            optimizationType: String;
        ) returns String;
        
        action createCheckpoint() returns String;
        
        action restoreFromCheckpoint(
            @title: 'Checkpoint ID'
            checkpointId: String;
        ) returns String;
        
        action validateData() returns String;
        
        action compressData(
            @title: 'Compression Type'
            compressionType: String;
        ) returns String;
        
        action encryptData(
            @title: 'Encryption Algorithm'
            encryptionAlgorithm: String;
        ) returns String;
        
        action migrateData(
            @title: 'Target Backend'
            targetBackend: String;
            @title: 'Migration Strategy'
            migrationStrategy: String;
        ) returns String;
    };
    
    // Storage backend management
    entity StorageBackends as projection on db.StorageBackends actions {
        action testConnection() returns String;
        
        action performHealthCheck() returns String;
        
        action optimizeStorage() returns String;
        
        action performMaintenance() returns String;
        
        action createBackup() returns String;
        
        action restoreFromBackup(
            @title: 'Backup ID'
            backupId: String;
        ) returns String;
        
        action updateConfiguration(
            @title: 'Configuration'
            configuration: String;
        ) returns String;
        
        action scaleCapacity(
            @title: 'Target Capacity (GB)'
            targetCapacity: Decimal;
        ) returns String;
    };
    
    // Cache configuration and management
    entity CacheConfigurations as projection on db.CacheConfigurations actions {
        action warmupCache() returns String;
        
        action clearCache() returns String;
        
        action flushCache() returns String;
        
        action invalidateKeys(
            @title: 'Key Pattern'
            keyPattern: String;
        ) returns String;
        
        action preloadData(
            @title: 'Data Source'
            dataSource: String;
        ) returns String;
        
        action optimizeCache() returns String;
        
        action adjustSize(
            @title: 'New Size (MB)'
            newSize: Integer;
        ) returns String;
    };
    
    // Data version management
    entity DataVersions as projection on db.DataVersions actions {
        action createVersion(
            @title: 'Version Data'
            versionData: {
                versionNumber: String;
                versionType: String;
                description: String;
                tags: String;
            }
        ) returns String;
        
        action tagVersion(
            @title: 'Tags'
            tags: String;
        ) returns String;
        
        action promoteVersion() returns String;
        
        action rollbackToVersion() returns String;
        
        action deleteVersion() returns String;
        
        action compareVersions(
            @title: 'Target Version ID'
            targetVersionId: String;
        ) returns String;
    };
    
    // Backup management
    entity DataBackups as projection on db.DataBackups actions {
        action executeBackup() returns String;
        
        action restoreBackup() returns String;
        
        action verifyBackup() returns String;
        
        action scheduleBackup(
            @title: 'Schedule Data'
            scheduleData: {
                scheduleType: String;
                scheduleExpression: String;
                retentionPeriod: Integer;
            }
        ) returns String;
        
        action cancelBackup() returns String;
        
        action cloneBackup(
            @title: 'Target Location'
            targetLocation: String;
        ) returns String;
    };
    
    // Storage utilization and cache operations (read-only)
    entity StorageUtilizations as projection on db.StorageUtilizations;
    entity CacheOperations as projection on db.CacheOperations;
    entity DataPerformanceMetrics as projection on db.DataPerformanceMetrics;
    
    // Data Management Functions
    function getStorageOptions() returns array of {
        backendType: String;
        name: String;
        description: String;
        capabilities: String;
        performanceProfile: String;
        costProfile: String;
    };
    
    function getDashboardData() returns {
        totalTasks: Integer;
        activeTasks: Integer;
        completedTasks: Integer;
        failedTasks: Integer;
        totalStorageUsed: Decimal;
        totalCacheHitRate: Decimal;
        averageProcessingSpeed: Decimal;
        storageBackends: array of {
            name: String;
            type: String;
            status: String;
            usedCapacity: Decimal;
            totalCapacity: Decimal;
            healthScore: Decimal;
        };
        cacheMetrics: {
            memoryHitRate: Decimal;
            redisHitRate: Decimal;
            totalOperations: Integer;
            averageResponseTime: Decimal;
        };
        performanceTrends: array of {
            timestamp: DateTime;
            throughput: Decimal;
            latency: Decimal;
            errorRate: Decimal;
        };
        alerts: array of String;
    };
    
    function getStorageBackendDetails(
        @title: 'Backend ID'
        backendId: String
    ) returns {
        backendId: String;
        name: String;
        type: String;
        status: String;
        healthScore: Decimal;
        capacity: {
            total: Decimal;
            used: Decimal;
            available: Decimal;
            utilizationPercent: Decimal;
        };
        performance: {
            readIOPS: Integer;
            writeIOPS: Integer;
            latency: Decimal;
            throughput: Decimal;
        };
        configuration: String;
        lastHealthCheck: DateTime;
        metrics: array of {
            metricType: String;
            value: Decimal;
            unit: String;
            trend: String;
        };
    };
    
    function getCacheStatus() returns {
        memoryCache: {
            status: String;
            hitRate: Decimal;
            missRate: Decimal;
            currentSize: Decimal;
            maxSize: Decimal;
            operationsPerSecond: Integer;
        };
        redisCache: {
            status: String;
            hitRate: Decimal;
            missRate: Decimal;
            currentSize: Decimal;
            maxSize: Decimal;
            operationsPerSecond: Integer;
        };
        overall: {
            totalHitRate: Decimal;
            totalOperations: Integer;
            averageResponseTime: Decimal;
        };
    };
    
    function getVersionHistory(
        @title: 'Task ID'
        taskId: String
    ) returns array of {
        versionId: String;
        versionNumber: String;
        versionType: String;
        dataSize: Integer64;
        compressedSize: Integer64;
        description: String;
        tags: String;
        isCurrent: Boolean;
        createdAt: DateTime;
        createdBy: String;
    };
    
    function getExportOptions() returns array of {
        format: String;
        description: String;
        supportedBackends: array of String;
        compressionOptions: array of String;
        encryptionOptions: array of String;
    };
    
    function getBackupStatus() returns {
        totalBackups: Integer;
        activeBackups: Integer;
        completedBackups: Integer;
        failedBackups: Integer;
        totalBackupSize: Decimal;
        nextScheduledBackup: DateTime;
        backupRetentionPolicy: String;
        storageUtilization: Decimal;
        recentBackups: array of {
            backupId: String;
            name: String;
            type: String;
            status: String;
            dataSize: Decimal;
            compressionRatio: Decimal;
            startedAt: DateTime;
            duration: Integer;
        };
    };
    
    function analyzePerformance(
        @title: 'Time Range'
        timeRange: String;
        @title: 'Metric Types'
        metricTypes: array of String
    ) returns {
        timeRange: String;
        summary: {
            averageThroughput: Decimal;
            averageLatency: Decimal;
            totalOperations: Integer;
            errorRate: Decimal;
        };
        trends: array of {
            timestamp: DateTime;
            throughput: Decimal;
            latency: Decimal;
            errorRate: Decimal;
            queueDepth: Integer;
        };
        bottlenecks: array of {
            component: String;
            severity: String;
            description: String;
            recommendation: String;
        };
        recommendations: array of {
            category: String;
            recommendation: String;
            expectedImprovement: Decimal;
            priority: String;
        };
    };
    
    function optimizeConfiguration(
        @title: 'Optimization Target'
        optimizationTarget: String
    ) returns {
        currentConfiguration: String;
        optimizedConfiguration: String;
        expectedImprovements: array of {
            metric: String;
            currentValue: Decimal;
            expectedValue: Decimal;
            improvementPercent: Decimal;
        };
        risks: array of String;
        implementationSteps: array of String;
    };
    
    // Event definitions for data management
    event DataTaskCreated : {
        taskId: String;
        taskName: String;
        taskType: String;
        storageBackend: String;
        priority: String;
        timestamp: DateTime;
    };
    
    event DataTaskCompleted : {
        taskId: String;
        taskType: String;
        status: String;
        processingTime: Integer;
        dataSize: Integer64;
        processedSize: Integer64;
        timestamp: DateTime;
    };
    
    event DataTaskFailed : {
        taskId: String;
        taskType: String;
        errorMessage: String;
        retryCount: Integer;
        timestamp: DateTime;
    };
    
    event StorageBackendAlert : {
        backendId: String;
        backendName: String;
        alertType: String;
        severity: String;
        message: String;
        currentValue: Decimal;
        threshold: Decimal;
        timestamp: DateTime;
    };
    
    event CachePerformanceAlert : {
        cacheId: String;
        cacheName: String;
        alertType: String;
        hitRate: Decimal;
        threshold: Decimal;
        timestamp: DateTime;
    };
    
    event BackupCompleted : {
        backupId: String;
        backupName: String;
        backupType: String;
        dataSize: Integer64;
        compressionRatio: Decimal;
        duration: Integer;
        timestamp: DateTime;
    };
    
    event VersionCreated : {
        versionId: String;
        taskId: String;
        versionNumber: String;
        versionType: String;
        dataSize: Integer64;
        timestamp: DateTime;
    };
    
    event PerformanceThresholdExceeded : {
        metricType: String;
        component: String;
        currentValue: Decimal;
        threshold: Decimal;
        severity: String;
        timestamp: DateTime;
    };
}
*/

// Agent 9 Service - Advanced Logical Reasoning and Decision-Making Agent - TEMPORARILY DISABLED
/*
@requires: ['authenticated-user']
service Agent9Service @(path: '/api/agent9/v1') {
    
    // Entity Exposures
    entity ReasoningTasks as projection on db.ReasoningTasks;
    entity KnowledgeBaseElements as projection on db.KnowledgeBaseElements;
    entity LogicalInferences as projection on db.LogicalInferences;
    entity ReasoningEngines as projection on db.ReasoningEngines;
    entity DecisionRecords as projection on db.DecisionRecords;
    entity ProblemSolvingRecords as projection on db.ProblemSolvingRecords;
    entity ReasoningPerformanceMetrics as projection on db.ReasoningPerformanceMetrics;
    
    // Reasoning Task Actions
    action startReasoning(
        @title: 'Task ID'
        taskId: String;
        @title: 'Configuration'
        configuration: String
    ) returns {
        success: Boolean;
        message: String;
        executionId: String;
        estimatedDuration: Integer;
    };
    
    action pauseReasoning(
        @title: 'Task ID'
        taskId: String
    ) returns {
        success: Boolean;
        message: String;
    };
    
    action resumeReasoning(
        @title: 'Task ID'
        taskId: String
    ) returns {
        success: Boolean;
        message: String;
    };
    
    action cancelReasoning(
        @title: 'Task ID'
        taskId: String;
        @title: 'Reason'
        reason: String
    ) returns {
        success: Boolean;
        message: String;
    };
    
    action validateConclusion(
        @title: 'Task ID'
        taskId: String;
        @title: 'Validation Method'
        validationMethod: String
    ) returns {
        isValid: Boolean;
        confidence: Decimal;
        validationResults: String;
        issues: String;
    };
    
    action explainReasoning(
        @title: 'Task ID'
        taskId: String;
        @title: 'Detail Level'
        detailLevel: String
    ) returns {
        explanation: String;
        reasoningChain: String;
        premises: String;
        conclusions: String;
        confidenceFactors: String;
    };
    
    // Knowledge Base Actions
    action addKnowledge(
        @title: 'Element Type'
        elementType: String;
        @title: 'Content'
        content: String;
        @title: 'Domain'
        domain: String;
        @title: 'Confidence Level'
        confidenceLevel: Decimal
    ) returns {
        success: Boolean;
        elementId: String;
        message: String;
    };
    
    action updateKnowledge(
        @title: 'Element ID'
        elementId: String;
        @title: 'Content'
        content: String;
        @title: 'Confidence Level'
        confidenceLevel: Decimal
    ) returns {
        success: Boolean;
        message: String;
    };
    
    action validateKnowledgeBase(
        @title: 'Domain'
        domain: String
    ) returns {
        isConsistent: Boolean;
        contradictions: String;
        recommendations: String;
    };
    
    // Inference Generation Actions
    action generateInferences(
        @title: 'Task ID'
        taskId: String;
        @title: 'Inference Types'
        inferenceTypes: array of String;
        @title: 'Max Inferences'
        maxInferences: Integer
    ) returns {
        success: Boolean;
        inferencesGenerated: Integer;
        newKnowledge: String;
        message: String;
    };
    
    action verifyInference(
        @title: 'Inference ID'
        inferenceId: String;
        @title: 'Verification Method'
        verificationMethod: String
    ) returns {
        isValid: Boolean;
        confidence: Decimal;
        evidence: String;
        counterEvidence: String;
    };
    
    // Decision Making Actions
    action makeDecision(
        @title: 'Task ID'
        taskId: String;
        @title: 'Decision Criteria'
        decisionCriteria: String;
        @title: 'Alternatives'
        alternatives: String
    ) returns {
        decision: String;
        confidence: Decimal;
        justification: String;
        riskAssessment: String;
        expectedOutcome: String;
    };
    
    action evaluateDecision(
        @title: 'Decision ID'
        decisionId: String;
        @title: 'Actual Outcome'
        actualOutcome: String
    ) returns {
        successRate: Decimal;
        lessonsLearned: String;
        recommendations: String;
    };
    
    // Problem Solving Actions
    action solveProblem(
        @title: 'Problem Description'
        problemDescription: String;
        @title: 'Problem Type'
        problemType: String;
        @title: 'Solving Strategy'
        solvingStrategy: String;
        @title: 'Constraints'
        constraints: String
    ) returns {
        solution: String;
        solutionSteps: String;
        qualityScore: Decimal;
        timeComplexity: String;
        spaceComplexity: String;
    };
    
    action optimizeSolution(
        @title: 'Problem ID'
        problemId: String;
        @title: 'Optimization Criteria'
        optimizationCriteria: String
    ) returns {
        optimizedSolution: String;
        improvementScore: Decimal;
        tradeoffs: String;
    };
    
    // Engine Management Actions
    action optimizeEngine(
        @title: 'Engine ID'
        engineId: String;
        @title: 'Optimization Type'
        optimizationType: String;
        @title: 'Target Metrics'
        targetMetrics: String
    ) returns {
        success: Boolean;
        performanceGain: Decimal;
        optimizationResults: String;
        newConfiguration: String;
    };
    
    action calibrateEngine(
        @title: 'Engine ID'
        engineId: String;
        @title: 'Test Dataset'
        testDataset: String
    ) returns {
        accuracyScore: Decimal;
        calibrationResults: String;
        recommendations: String;
    };
    
    // Functions
    function getReasoningOptions() returns {
        reasoningTypes: array of String;
        problemDomains: array of String;
        reasoningEngines: array of String;
        solvingStrategies: array of String;
    };
    
    function getDashboardData(
        @title: 'Time Range'
        timeRange: String
    ) returns {
        summary: {
            totalTasks: Integer;
            activeTasks: Integer;
            completedTasks: Integer;
            averageConfidence: Decimal;
            averageProcessingTime: Integer;
            knowledgeBaseSize: Integer;
            inferencesGenerated: Integer;
            decisionsProcessed: Integer;
        };
        enginePerformance: array of {
            engineType: String;
            tasksProcessed: Integer;
            averageAccuracy: Decimal;
            averageSpeed: Integer;
            successRate: Decimal;
        };
        domainDistribution: array of {
            domain: String;
            taskCount: Integer;
            successRate: Decimal;
            averageConfidence: Decimal;
        };
        trends: array of {
            timestamp: DateTime;
            tasksCompleted: Integer;
            averageConfidence: Decimal;
            processingTime: Integer;
            accuracy: Decimal;
        };
    };
    
    function getKnowledgeBaseStats(
        @title: 'Domain'
        domain: String
    ) returns {
        totalElements: Integer;
        facts: Integer;
        rules: Integer;
        ontologies: Integer;
        axioms: Integer;
        definitions: Integer;
        averageConfidence: Decimal;
        consistency: {
            isConsistent: Boolean;
            contradictions: Integer;
            inconsistencyScore: Decimal;
        };
        coverage: {
            domainCoverage: Decimal;
            knowledgeGaps: array of String;
        };
        usage: {
            mostUsedElements: array of String;
            underutilizedElements: array of String;
        };
    };
    
    function getReasoningChain(
        @title: 'Task ID'
        taskId: String
    ) returns {
        premises: array of {
            statement: String;
            confidence: Decimal;
            source: String;
        };
        inferences: array of {
            step: Integer;
            inferenceType: String;
            statement: String;
            confidence: Decimal;
            rulesApplied: array of String;
            derivationPath: String;
        };
        conclusions: array of {
            statement: String;
            confidence: Decimal;
            supportingEvidence: String;
            certaintyLevel: String;
        };
        metadata: {
            totalSteps: Integer;
            processingTime: Integer;
            engineUsed: String;
            complexityScore: Decimal;
        };
    };
    
    function analyzeContradictions(
        @title: 'Domain'
        domain: String
    ) returns {
        contradictions: array of {
            contradictionId: String;
            conflictingStatements: array of String;
            severity: String;
            resolutionStrategy: String;
            confidence: Decimal;
        };
        summary: {
            totalContradictions: Integer;
            highSeverity: Integer;
            mediumSeverity: Integer;
            lowSeverity: Integer;
            autoResolvable: Integer;
        };
        recommendations: array of {
            action: String;
            priority: String;
            expectedBenefit: String;
        };
    };
    
    function getEngineComparison(
        @title: 'Engine Types'
        engineTypes: array of String;
        @title: 'Problem Domain'
        problemDomain: String
    ) returns {
        comparison: array of {
            engineType: String;
            accuracy: Decimal;
            speed: Integer;
            memoryUsage: Integer;
            complexity: String;
            strengths: array of String;
            weaknesses: array of String;
            bestUseCases: array of String;
        };
        recommendation: {
            bestEngine: String;
            reason: String;
            confidenceLevel: Decimal;
        };
    };
    
    function getDecisionAnalysis(
        @title: 'Decision ID'
        decisionId: String
    ) returns {
        decision: {
            name: String;
            recommendedOption: String;
            confidence: Decimal;
            riskLevel: String;
        };
        criteria: array of {
            criterion: String;
            weight: Decimal;
            scores: array of {
                alternative: String;
                score: Decimal;
            };
        };
        analysis: {
            riskFactors: array of String;
            opportunities: array of String;
            expectedOutcome: String;
            contingencyPlans: array of String;
        };
        simulation: {
            scenarios: array of {
                scenario: String;
                probability: Decimal;
                outcome: String;
                impact: String;
            };
        };
    };
    
    function getProblemSolvingInsights(
        @title: 'Problem Type'
        problemType: String;
        @title: 'Time Range'
        timeRange: String
    ) returns {
        patterns: array of {
            pattern: String;
            frequency: Integer;
            successRate: Decimal;
            averageTime: Integer;
        };
        strategies: array of {
            strategy: String;
            effectiveness: Decimal;
            complexity: String;
            bestFor: array of String;
        };
        trends: array of {
            timestamp: DateTime;
            problemsSolved: Integer;
            averageQuality: Decimal;
            averageTime: Integer;
        };
        recommendations: array of {
            recommendation: String;
            expectedImprovement: Decimal;
            implementationEffort: String;
        };
    };
    
    function getPerformanceMetrics(
        @title: 'Engine Type'
        engineType: String;
        @title: 'Time Range'
        timeRange: String
    ) returns {
        overall: {
            accuracy: Decimal;
            precision: Decimal;
            recall: Decimal;
            f1Score: Decimal;
            processingSpeed: Integer;
            memoryEfficiency: Decimal;
            throughput: Integer;
            errorRate: Decimal;
        };
        trends: array of {
            timestamp: DateTime;
            accuracy: Decimal;
            speed: Integer;
            confidence: Decimal;
        };
        benchmarks: {
            industry: Decimal;
            historical: Decimal;
            target: Decimal;
        };
        breakdown: array of {
            domain: String;
            accuracy: Decimal;
            taskCount: Integer;
            averageTime: Integer;
        };
    };
    
    function optimizeKnowledgeBase(
        @title: 'Domain'
        domain: String;
        @title: 'Optimization Strategy'
        optimizationStrategy: String
    ) returns {
        optimizations: array of {
            type: String;
            description: String;
            expectedBenefit: String;
            effort: String;
        };
        redundancies: array of {
            elements: array of String;
            redundancyType: String;
            consolidationStrategy: String;
        };
        gaps: array of {
            area: String;
            priority: String;
            suggestedContent: String;
        };
        summary: {
            currentSize: Integer;
            projectedSize: Integer;
            expectedPerformanceGain: Decimal;
            confidenceImprovement: Decimal;
        };
    };
    
    // Events
    event ReasoningStarted : {
        taskId: String;
        taskName: String;
        reasoningType: String;
        engineType: String;
        problemDomain: String;
        timestamp: DateTime;
    };
    
    event ReasoningCompleted : {
        taskId: String;
        taskName: String;
        reasoningType: String;
        conclusionsReached: Integer;
        confidenceScore: Decimal;
        processingTime: Integer;
        timestamp: DateTime;
    };
    
    event InferenceGenerated : {
        taskId: String;
        inferenceId: String;
        inferenceType: String;
        statement: String;
        confidence: Decimal;
        validationStatus: String;
        timestamp: DateTime;
    };
    
    event DecisionMade : {
        taskId: String;
        decisionId: String;
        decisionType: String;
        recommendedOption: String;
        confidence: Decimal;
        riskLevel: String;
        timestamp: DateTime;
    };
    
    event ProblemSolved : {
        problemId: String;
        problemType: String;
        solvingStrategy: String;
        qualityScore: Decimal;
        solvingTime: Integer;
        timestamp: DateTime;
    };
    
    event ContradictionDetected : {
        contradictionId: String;
        conflictingElements: array of String;
        severity: String;
        domain: String;
        autoResolvable: Boolean;
        timestamp: DateTime;
    };
    
    event KnowledgeUpdated : {
        elementId: String;
        elementType: String;
        updateType: String;
        domain: String;
        confidenceChange: Decimal;
        timestamp: DateTime;
    };
    
    event PerformanceThresholdReached : {
        engineType: String;
        metricType: String;
        currentValue: Decimal;
        threshold: Decimal;
        trend: String;
        timestamp: DateTime;
    };
}
*/

    // Agent 10 Service - Calculation Engine
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE', 'UPDATE'], to: 'DataManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity CalculationTasks as projection on db.CalculationTasks actions {
        @requires: ['DataManager', 'Admin']
        action startCalculation() returns String;
        @requires: ['DataManager', 'Admin']
        action pauseCalculation() returns String;
        @requires: ['DataManager', 'Admin']
        action resumeCalculation() returns String;
        @requires: ['DataManager', 'Admin']
        action cancelCalculation() returns String;
        @requires: ['DataManager', 'Admin']
        action validateFormula(formula: String) returns String;
        @requires: ['DataManager', 'Admin']
        action previewCalculation(
            formula: String,
            sampleData: String
        ) returns String;
        @requires: ['DataManager', 'Admin']
        action exportResults(
            format: String,
            includeSteps: Boolean,
            includeStatistics: Boolean
        ) returns String;
    };
    
    @requires: ['authenticated-user']
    entity CalculationSteps as projection on db.CalculationSteps;
    
    @requires: ['authenticated-user']
    entity StatisticalAnalysisResults as projection on db.StatisticalAnalysisResults;
    
    @requires: ['authenticated-user']
    entity CalculationErrorCorrections as projection on db.CalculationErrorCorrections;
    
    // Agent 10 specific actions for calculation engine
    @requires: ['DataManager', 'Admin']
    action performCalculation(
        formula: String,
        inputData: String,
        calculationType: String,
        method: String,
        precision: String,
        enableSelfHealing: Boolean
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action performStatisticalAnalysis(
        data: String,
        analysisType: String,
        confidenceLevel: Decimal,
        options: String
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action batchCalculate(
        calculations: String,
        parallel: Boolean,
        priority: String
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action evaluateCustomFormula(
        formula: String,
        variables: String,
        verify: Boolean
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action getCalculationMethods() returns String;
    
    @requires: ['DataManager', 'Admin']
    action getStatisticalMethods() returns String;
    
    @requires: ['DataManager', 'Admin']
    action getSelfHealingStrategies() returns String;
    
    @requires: ['DataManager', 'Admin']
    action configurePrecision(
        type: String,
        accuracy: Decimal
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action configureParallelProcessing(
        maxThreads: Integer,
        chunkSize: Integer
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action getCalculationHistory(
        limit: Integer,
        offset: Integer,
        filter: String
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action getPerformanceMetrics(
        taskId: String
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action clearCalculationCache() returns String;
    
    // Events
    event CalculationStarted : {
        taskId: String;
        taskName: String;
        calculationType: String;
        method: String;
        timestamp: DateTime;
    };
    
    event CalculationCompleted : {
        taskId: String;
        taskName: String;
        executionTime: Integer;
        status: String;
        accuracy: Decimal;
        timestamp: DateTime;
    };
    
    event SelfHealingTriggered : {
        taskId: String;
        errorType: String;
        strategy: String;
        originalValue: String;
        correctedValue: String;
        confidence: Decimal;
        timestamp: DateTime;
    };
    
    event StatisticalAnalysisCompleted : {
        taskId: String;
        analysisType: String;
        sampleSize: Integer;
        significanceLevel: Decimal;
        timestamp: DateTime;
    };
    
    event CalculationError : {
        taskId: String;
        errorType: String;
        errorMessage: String;
        step: Integer;
        timestamp: DateTime;
    };

    // Agent 11 Service - SQL Engine
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE', 'UPDATE'], to: 'DataManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity SQLQueryTasks as projection on db.SQLQueryTasks actions {
        @requires: ['DataManager', 'Admin']
        action executeQuery() returns String;
        @requires: ['DataManager', 'Admin']
        action validateSQL() returns String;
        @requires: ['DataManager', 'Admin']
        action optimizeQuery() returns String;
        @requires: ['DataManager', 'Admin']
        action generateFromNL(naturalLanguage: String) returns String;
        @requires: ['DataManager', 'Admin']
        action explainQuery() returns String;
        @requires: ['DataManager', 'Admin']
        action approveQuery() returns String;
        @requires: ['DataManager', 'Admin']
        action exportResults(
            format: String,
            includeMetadata: Boolean
        ) returns String;
    };
    
    @requires: ['authenticated-user']
    entity QueryOptimizations as projection on db.QueryOptimizations;
    
    @requires: ['authenticated-user']
    entity QueryExecutionHistory as projection on db.QueryExecutionHistory;
    
    @requires: ['authenticated-user']
    entity SchemaReferences as projection on db.SchemaReferences;
    
    @requires: ['authenticated-user']
    entity NLProcessingResults as projection on db.NLProcessingResults;
    
    // Agent 11 specific actions for SQL operations
    @requires: ['DataManager', 'Admin']
    action executeSQL(
        sql: String,
        parameters: String,
        database: String,
        timeout: Integer
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action translateNaturalLanguage(
        naturalLanguage: String,
        context: String,
        database: String
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action optimizeSQL(
        sql: String,
        database: String,
        explain: Boolean
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action validateSQL(
        sql: String,
        dialect: String
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action explainExecutionPlan(
        sql: String,
        database: String
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action getSchemaInfo(
        database: String,
        schema: String
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action getTableInfo(
        database: String,
        table: String
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action suggestIndexes(
        table: String,
        database: String
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action analyzeQueryPerformance(
        queryId: String
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action getQueryHistory(
        database: String,
        limit: Integer,
        offset: Integer
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action createQueryTemplate(
        name: String,
        sql: String,
        parameters: String
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action manageDatabaseConnection(
        operation: String,
        connectionConfig: String
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action backupQuery(
        queryId: String,
        includeResults: Boolean
    ) returns String;
    
    @requires: ['DataManager', 'Admin']
    action restoreQuery(
        backupId: String
    ) returns String;
    
    // Events
    event QueryExecuted : {
        queryId: String;
        queryName: String;
        database: String;
        executionTime: Integer;
        rowsAffected: Integer;
        status: String;
        timestamp: DateTime;
    };
    
    event QueryOptimized : {
        queryId: String;
        originalSQL: String;
        optimizedSQL: String;
        improvementPercent: Decimal;
        optimizationType: String;
        timestamp: DateTime;
    };
    
    event NLQueryTranslated : {
        originalText: String;
        generatedSQL: String;
        confidenceScore: Decimal;
        language: String;
        processingTime: Integer;
        timestamp: DateTime;
    };
    
    event QueryError : {
        queryId: String;
        errorCode: String;
        errorMessage: String;
        database: String;
        timestamp: DateTime;
    };
    
    event SchemaUpdated : {
        database: String;
        schemaName: String;
        operation: String;
        affectedObjects: String;
        timestamp: DateTime;
    };
    
    event PerformanceAlert : {
        queryId: String;
        alertType: String;
        threshold: Decimal;
        currentValue: Decimal;
        recommendation: String;
        timestamp: DateTime;
    };

//=====================================================
// Agent 12 Service: Catalog Manager
//=====================================================

service Agent12Service @(path : '/a2a/agent12') {
    // Catalog Entry Management
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE', 'UPDATE'], to: 'CatalogManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity CatalogEntries as projection on db.CatalogEntries {
        *,
        dependencies : redirected to CatalogDependencies,
        reviews : redirected to CatalogReviews,
        metadata_entries : redirected to CatalogMetadata
    } actions {
        @requires: ['CatalogManager', 'Admin']
        action publish() returns String;
        @requires: ['CatalogManager', 'Admin']
        action deprecate(reason: String) returns Boolean;
        @requires: ['CatalogManager', 'Admin']
        action updateMetadata(metadata: String) returns Boolean;
        @requires: ['Admin']
        action archive() returns Boolean;
        @requires: ['CatalogManager', 'Admin']
        action generateDocumentation() returns String;
        @requires: ['CatalogManager', 'Admin']
        action validateEntry() returns String;
        @requires: ['CatalogManager', 'Admin']
        action duplicateEntry() returns String;
        @requires: ['CatalogManager', 'Admin']
        action exportEntry(format: String) returns String;
    };
    
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE', 'UPDATE'], to: 'CatalogManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity CatalogDependencies as projection on db.CatalogDependencies;
    
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE'], to: 'authenticated-user' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity CatalogReviews as projection on db.CatalogReviews {
        *
    } actions {
        @requires: ['Admin']
        action approveReview() returns Boolean;
        @requires: ['Admin'] 
        action rejectReview(reason: String) returns Boolean;
        @requires: ['Admin']
        action flagReview(reason: String) returns Boolean;
    };
    
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE', 'UPDATE'], to: 'CatalogManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity CatalogMetadata as projection on db.CatalogMetadata;
    
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity CatalogSearches as projection on db.CatalogSearches;
    
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE', 'UPDATE'], to: 'CatalogManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity RegistryManagement as projection on db.RegistryManagement {
        *
    } actions {
        @requires: ['CatalogManager', 'Admin']
        action syncRegistry() returns String;
        @requires: ['CatalogManager', 'Admin']
        action testConnection() returns Boolean;
        @requires: ['Admin']
        action resetRegistry() returns Boolean;
        @requires: ['CatalogManager', 'Admin']
        action exportRegistry(format: String) returns String;
        @requires: ['CatalogManager', 'Admin']
        action importRegistry(data: String) returns String;
    };
    
    // Catalog Management Actions
    @requires: ['CatalogManager', 'Admin']
    action searchCatalog(
        query: String,
        category: String,
        tags: String,
        filters: String,
        sortBy: String,
        limit: Integer
    ) returns String;
    
    @requires: ['CatalogManager', 'Admin']
    action discoverServices(
        registryType: String,
        filters: String,
        autoRegister: Boolean
    ) returns String;
    
    @requires: ['CatalogManager', 'Admin']
    action registerService(
        serviceName: String,
        serviceUrl: String,
        serviceType: String,
        metadata: String,
        healthCheckUrl: String
    ) returns String;
    
    @requires: ['CatalogManager', 'Admin']
    action updateServiceHealth(
        serviceId: String,
        healthStatus: String,
        healthDetails: String
    ) returns Boolean;
    
    @requires: ['CatalogManager', 'Admin']
    action analyzeDependencies(
        entryId: String,
        depth: Integer,
        includeIndirect: Boolean
    ) returns String;
    
    @requires: ['CatalogManager', 'Admin']
    action validateMetadata(
        entryId: String,
        schemaValidation: Boolean,
        qualityChecks: Boolean
    ) returns String;
    
    @requires: ['CatalogManager', 'Admin']
    action generateCatalogReport(
        format: String,
        includeStats: Boolean,
        includeReviews: Boolean,
        dateRange: String
    ) returns String;
    
    @requires: ['CatalogManager', 'Admin']
    action bulkImport(
        importFormat: String,
        data: String,
        validateBeforeImport: Boolean,
        updateExisting: Boolean
    ) returns String;
    
    @requires: ['CatalogManager', 'Admin']
    action bulkExport(
        exportFormat: String,
        categories: String,
        filters: String,
        includeMetadata: Boolean
    ) returns String;
    
    @requires: ['CatalogManager', 'Admin']
    action syncExternalCatalog(
        catalogUrl: String,
        catalogType: String,
        syncMode: String,
        mappingRules: String
    ) returns String;
    
    @requires: ['CatalogManager', 'Admin']
    action optimizeSearchIndex() returns Boolean;
    
    @requires: ['CatalogManager', 'Admin']
    action generateRecommendations(
        userId: String,
        context: String,
        limit: Integer
    ) returns String;
    
    @requires: ['Admin']
    action rebuildCatalog() returns String;
    
    @requires: ['CatalogManager', 'Admin']
    action createCategory(
        categoryName: String,
        description: String,
        parentCategory: String,
        icon: String
    ) returns String;
    
    @requires: ['CatalogManager', 'Admin']
    action manageVersioning(
        entryId: String,
        operation: String,
        version: String,
        changeLog: String
    ) returns String;
    
    // Events for catalog operations
    event CatalogEntryCreated : {
        entryId: String;
        entryName: String;
        category: String;
        provider: String;
        timestamp: DateTime;
    };
    
    event CatalogEntryUpdated : {
        entryId: String;
        entryName: String;
        changes: String;
        updatedBy: String;
        timestamp: DateTime;
    };
    
    event CatalogEntryPublished : {
        entryId: String;
        entryName: String;
        version: String;
        publishedBy: String;
        timestamp: DateTime;
    };
    
    event CatalogEntryDeprecated : {
        entryId: String;
        entryName: String;
        reason: String;
        replacementEntry: String;
        timestamp: DateTime;
    };
    
    event ServiceDiscovered : {
        serviceName: String;
        serviceType: String;
        discoveryMethod: String;
        registrySource: String;
        timestamp: DateTime;
    };
    
    event ServiceHealthUpdated : {
        serviceId: String;
        serviceName: String;
        previousHealth: String;
        currentHealth: String;
        timestamp: DateTime;
    };
    
    event DependencyAdded : {
        entryId: String;
        dependentEntryId: String;
        dependencyType: String;
        timestamp: DateTime;
    };
    
    event ReviewSubmitted : {
        entryId: String;
        reviewer: String;
        rating: Integer;
        reviewId: String;
        timestamp: DateTime;
    };
    
    event SearchPerformed : {
        searchQuery: String;
        searchType: String;
        resultsCount: Integer;
        searchTime: Integer;
        timestamp: DateTime;
    };
    
    event CatalogSyncStarted : {
        registryId: String;
        registryName: String;
        syncType: String;
        timestamp: DateTime;
    };
    
    event CatalogSyncCompleted : {
        registryId: String;
        registryName: String;
        entriesProcessed: Integer;
        entriesAdded: Integer;
        entriesUpdated: Integer;
        errors: Integer;
        duration: Integer;
        timestamp: DateTime;
    };
    
    event MetadataValidationFailed : {
        entryId: String;
        validationType: String;
        errors: String;
        timestamp: DateTime;
    };
    
    event CatalogError : {
        operation: String;
        errorCode: String;
        errorMessage: String;
        entryId: String;
        timestamp: DateTime;
    };
    
    // ============================================
    // AGENT 13 - AGENT BUILDER SERVICE DEFINITIONS
    // ============================================
    
    // Core Entities for Agent 13
    @odata.draft.enabled
    entity AgentTemplates as projection on db.AgentTemplates;
    
    @odata.draft.enabled
    entity AgentBuilds as projection on db.AgentBuilds;
    
    @odata.draft.enabled
    entity TemplateComponents as projection on db.TemplateComponents;
    
    @odata.draft.enabled
    entity AgentDeployments as projection on db.AgentDeployments;
    
    @odata.draft.enabled
    entity BuildPipelines as projection on db.BuildPipelines;
    
    // Actions for Agent 13 (Agent Builder)
    @requires: ['AgentBuilder', 'Admin']
    action createAgentTemplate(
        templateName: String,
        agentType: String,
        baseTemplate: String,
        capabilities: String,
        configuration: String,
        description: String
    ) returns String;
    
    @requires: ['AgentBuilder', 'Developer']
    action generateAgentFromTemplate(
        templateId: String,
        agentName: String,
        customConfiguration: String,
        targetEnvironment: String
    ) returns String;
    
    @requires: ['AgentBuilder', 'Admin']
    action deployAgent(
        buildId: String,
        targetEnvironment: String,
        deploymentConfig: String,
        autoStart: Boolean
    ) returns String;
    
    @requires: ['AgentBuilder', 'Admin']
    action createBuildPipeline(
        pipelineName: String,
        templateIds: String,
        stages: String,
        triggers: String,
        configuration: String
    ) returns String;
    
    @requires: ['AgentBuilder', 'Developer']
    action validateTemplate(
        templateId: String,
        validationType: String
    ) returns String;
    
    @requires: ['AgentBuilder', 'Developer']
    action testAgent(
        buildId: String,
        testSuite: String,
        testConfiguration: String
    ) returns String;
    
    @requires: ['AgentBuilder', 'Admin']
    action manageTemplateComponent(
        templateId: String,
        componentType: String,
        componentName: String,
        sourceCode: String,
        dependencies: String,
        operation: String
    ) returns String;
    
    @requires: ['AgentBuilder', 'Developer']
    action generateCode(
        templateId: String,
        targetLanguage: String,
        framework: String,
        optimizations: String
    ) returns String;
    
    @requires: ['AgentBuilder', 'DevOps']
    action triggerPipeline(
        pipelineId: String,
        parameters: String,
        priority: String
    ) returns String;
    
    @requires: ['AgentBuilder', 'Admin']
    action cloneTemplate(
        sourceTemplateId: String,
        newTemplateName: String,
        modifications: String
    ) returns String;
    
    // Functions for Agent 13 (Agent Builder)
    function GetBuilderStatistics() returns {
        totalTemplates: Integer;
        totalBuilds: Integer;
        successfulBuilds: Integer;
        failedBuilds: Integer;
        activeDeployments: Integer;
        totalPipelines: Integer;
        buildsToday: Integer;
        deploymentsToday: Integer;
        averageBuildTime: Integer;
        templateUsageStats: String;
    };
    
    function GetTemplateDetails(templateId: String) returns {
        templateId: String;
        templateName: String;
        agentType: String;
        version: String;
        baseTemplate: String;
        capabilities: String;
        configuration: String;
        components: String;
        lastModified: DateTime;
        buildCount: Integer;
        deploymentCount: Integer;
        successRate: Decimal;
        documentation: String;
        dependencies: String;
    };
    
    function GetDeploymentTargets() returns {
        environments: String;
        kubernetesTargets: String;
        dockerTargets: String;
        cloudTargets: String;
        onPremiseTargets: String;
        availableResources: String;
    };
    
    function GetBuildPipelines() returns {
        pipelines: String;
        activeBuilds: Integer;
        queuedBuilds: Integer;
        buildHistory: String;
        pipelineMetrics: String;
    };
    
    function GetAgentComponents(templateId: String) returns {
        components: String;
        dependencies: String;
        interfaces: String;
        configurations: String;
        tests: String;
    };
    
    function GetTestConfiguration(templateId: String) returns {
        testSuites: String;
        testCases: String;
        mockData: String;
        testEnvironments: String;
        automationScripts: String;
    };
    
    function ValidateAgentCode(
        sourceCode: String,
        agentType: String,
        validationRules: String
    ) returns {
        isValid: Boolean;
        errors: String;
        warnings: String;
        suggestions: String;
        codeQualityScore: Integer;
    };
    
    function GetBuildArtifacts(buildId: String) returns {
        artifacts: String;
        containerImages: String;
        configurations: String;
        documentation: String;
        testResults: String;
    };
    
    function GetDeploymentStatus(deploymentId: String) returns {
        status: String;
        progress: Integer;
        logs: String;
        healthMetrics: String;
        errors: String;
    };
    
    function StartBatchBuild(templateIds: String) returns {
        batchId: String;
        queuedBuilds: Integer;
        estimatedDuration: Integer;
        status: String;
    };
    
    // Events for Agent 13 (Agent Builder)
    event TemplateCreated : {
        templateId: String;
        templateName: String;
        agentType: String;
        createdBy: String;
        timestamp: DateTime;
    };
    
    event TemplateUpdated : {
        templateId: String;
        templateName: String;
        version: String;
        changes: String;
        updatedBy: String;
        timestamp: DateTime;
    };
    
    event AgentBuildStarted : {
        buildId: String;
        templateId: String;
        templateName: String;
        buildType: String;
        triggeredBy: String;
        timestamp: DateTime;
    };
    
    event AgentBuildCompleted : {
        buildId: String;
        templateId: String;
        buildStatus: String;
        duration: Integer;
        artifacts: String;
        timestamp: DateTime;
    };
    
    event AgentBuildFailed : {
        buildId: String;
        templateId: String;
        errorMessage: String;
        failureStage: String;
        duration: Integer;
        timestamp: DateTime;
    };
    
    event AgentDeploymentStarted : {
        deploymentId: String;
        buildId: String;
        targetEnvironment: String;
        deployedBy: String;
        timestamp: DateTime;
    };
    
    event AgentDeploymentCompleted : {
        deploymentId: String;
        buildId: String;
        targetEnvironment: String;
        status: String;
        endpoint: String;
        timestamp: DateTime;
    };
    
    event AgentDeploymentFailed : {
        deploymentId: String;
        buildId: String;
        targetEnvironment: String;
        errorMessage: String;
        timestamp: DateTime;
    };
    
    event PipelineStarted : {
        pipelineId: String;
        pipelineName: String;
        triggeredBy: String;
        parameters: String;
        timestamp: DateTime;
    };
    
    event PipelineCompleted : {
        pipelineId: String;
        pipelineName: String;
        status: String;
        duration: Integer;
        buildsProcessed: Integer;
        timestamp: DateTime;
    };
    
    event PipelineFailed : {
        pipelineId: String;
        pipelineName: String;
        failedStage: String;
        errorMessage: String;
        timestamp: DateTime;
    };
    
    event ComponentUpdated : {
        templateId: String;
        componentId: String;
        componentType: String;
        componentName: String;
        updatedBy: String;
        timestamp: DateTime;
    };
    
    event CodeGenerated : {
        templateId: String;
        targetLanguage: String;
        framework: String;
        codeSize: Integer;
        generatedBy: String;
        timestamp: DateTime;
    };
    
    event TestExecuted : {
        buildId: String;
        testSuite: String;
        testResults: String;
        passed: Integer;
        failed: Integer;
        duration: Integer;
        timestamp: DateTime;
    };
    
    event TemplateValidated : {
        templateId: String;
        validationType: String;
        validationStatus: String;
        issues: String;
        timestamp: DateTime;
    };
    
    event BatchBuildStarted : {
        batchId: String;
        templateIds: String;
        buildCount: Integer;
        triggeredBy: String;
        timestamp: DateTime;
    };
    
    event BatchBuildCompleted : {
        batchId: String;
        totalBuilds: Integer;
        successfulBuilds: Integer;
        failedBuilds: Integer;
        duration: Integer;
        timestamp: DateTime;
    };
    
    event AgentBuilderError : {
        operation: String;
        errorCode: String;
        errorMessage: String;
        templateId: String;
        buildId: String;
        timestamp: DateTime;
    };
    
    // ============================================
    // AGENT 14 - EMBEDDING FINE-TUNER SERVICE DEFINITIONS
    // ============================================
    
    // Core Entities for Agent 14
    @odata.draft.enabled
    entity EmbeddingModels as projection on db.EmbeddingModels;
    
    @odata.draft.enabled
    entity TrainingRuns as projection on db.TrainingRuns;
    
    @odata.draft.enabled
    entity ModelEvaluations as projection on db.ModelEvaluations;
    
    @odata.draft.enabled
    entity ModelOptimizations as projection on db.ModelOptimizations;
    
    @odata.draft.enabled
    entity FineTuningTasks as projection on db.FineTuningTasks;
    
    // Actions for Agent 14 (Embedding Fine-Tuner)
    @requires: ['EmbeddingTuner', 'Admin']
    action createEmbeddingModel(
        modelName: String,
        baseModel: String,
        modelType: String,
        domain: String,
        language: String,
        embeddingDimensions: Integer,
        configuration: String
    ) returns String;
    
    @requires: ['EmbeddingTuner', 'Developer']
    action startFineTuning(
        modelId: String,
        trainingName: String,
        trainingStrategy: String,
        datasetPath: String,
        trainingConfig: String,
        batchSize: Integer,
        learningRate: Decimal,
        epochs: Integer
    ) returns String;
    
    @requires: ['EmbeddingTuner', 'Developer']
    action evaluateModel(
        modelId: String,
        evaluationName: String,
        evaluationType: String,
        benchmarkDataset: String,
        testDataPath: String
    ) returns String;
    
    @requires: ['EmbeddingTuner', 'Developer']
    action optimizeModel(
        modelId: String,
        optimizationName: String,
        optimizationType: String,
        optimizationConfig: String
    ) returns String;
    
    @requires: ['EmbeddingTuner', 'Admin']
    action deployModelToProduction(
        modelId: String,
        deploymentConfig: String,
        performanceThreshold: Decimal
    ) returns String;
    
    @requires: ['EmbeddingTuner', 'Developer']
    action compareModels(
        baseModelId: String,
        compareModelId: String,
        comparisonMetrics: String,
        testDataset: String
    ) returns String;
    
    @requires: ['EmbeddingTuner', 'Admin']
    action createFineTuningTask(
        taskName: String,
        baseModel: String,
        targetDomain: String,
        taskType: String,
        priority: String,
        taskConfiguration: String
    ) returns String;
    
    @requires: ['EmbeddingTuner', 'Developer']
    action pauseTraining(
        trainingRunId: String,
        reason: String
    ) returns String;
    
    @requires: ['EmbeddingTuner', 'Developer']
    action resumeTraining(
        trainingRunId: String,
        resumeConfig: String
    ) returns String;
    
    @requires: ['EmbeddingTuner', 'Developer']
    action exportModel(
        modelId: String,
        exportFormat: String,
        optimizationLevel: String
    ) returns String;
    
    @requires: ['EmbeddingTuner', 'Developer']
    action validateModelQuality(
        modelId: String,
        qualityThresholds: String,
        validationDataset: String
    ) returns String;
    
    @requires: ['EmbeddingTuner', 'Admin']
    action batchOptimizeModels(
        modelIds: String,
        optimizationType: String,
        optimizationConfig: String
    ) returns String;
    
    // Functions for Agent 14 (Embedding Fine-Tuner)
    function GetFineTunerStatistics() returns {
        totalModels: Integer;
        trainingModels: Integer;
        productionModels: Integer;
        totalTrainingRuns: Integer;
        successfulTrainings: Integer;
        failedTrainings: Integer;
        averageTrainingTime: Integer;
        averageModelAccuracy: Decimal;
        totalEvaluations: Integer;
        totalOptimizations: Integer;
        averageInferenceSpeed: Integer;
        modelsOptimizedToday: Integer;
        trainingHoursToday: Integer;
        performanceImprovementAvg: Decimal;
    };
    
    function GetModelPerformanceMetrics(modelId: String) returns {
        modelId: String;
        modelName: String;
        modelType: String;
        status: String;
        version: String;
        accuracy: Decimal;
        f1Score: Decimal;
        cosineSimilarity: Decimal;
        inferenceSpeedMs: Integer;
        modelSizeMB: Integer;
        memoryFootprintMB: Integer;
        throughput: Integer;
        lastEvaluationDate: DateTime;
        comparisonToBaseline: Decimal;
        qualityScore: Decimal;
        recommendedForProduction: Boolean;
    };
    
    function GetTrainingProgress(trainingRunId: String) returns {
        trainingRunId: String;
        trainingName: String;
        status: String;
        currentEpoch: Integer;
        totalEpochs: Integer;
        progressPercent: Integer;
        trainingLoss: Decimal;
        validationLoss: Decimal;
        trainingAccuracy: Decimal;
        validationAccuracy: Decimal;
        estimatedTimeRemaining: Integer;
        gpuUtilization: Decimal;
        memoryUsage: Integer;
        learningRate: Decimal;
        batchSize: Integer;
    };
    
    function GetBenchmarkResults(evaluationId: String) returns {
        evaluationId: String;
        evaluationName: String;
        modelId: String;
        benchmarkDataset: String;
        evaluationType: String;
        overallScore: Decimal;
        cosineSimilarity: Decimal;
        precisionAtK: Decimal;
        recallAtK: Decimal;
        f1Score: Decimal;
        silhouetteScore: Decimal;
        daviesBouldinIndex: Decimal;
        inferenceSpeedMs: Integer;
        comparisonBaseline: String;
        improvementPercent: Decimal;
        evaluationTime: DateTime;
    };
    
    function GetOptimizationReport(optimizationId: String) returns {
        optimizationId: String;
        optimizationName: String;
        modelId: String;
        optimizationType: String;
        status: String;
        originalSizeMB: Integer;
        optimizedSizeMB: Integer;
        sizeReductionPercent: Decimal;
        originalInferenceMs: Integer;
        optimizedInferenceMs: Integer;
        speedImprovementPercent: Decimal;
        accuracyLossPercent: Decimal;
        qualityScoreBefore: Decimal;
        qualityScoreAfter: Decimal;
        optimizationTime: DateTime;
    };
    
    function GetAvailableBaseModels() returns {
        sentenceTransformers: String;
        openAIModels: String;
        bertModels: String;
        customModels: String;
        supportedLanguages: String;
        recommendedModels: String;
    };
    
    function GetTrainingRecommendations(domain: String, dataSize: Integer) returns {
        recommendedStrategy: String;
        suggestedBatchSize: Integer;
        suggestedLearningRate: Decimal;
        suggestedEpochs: Integer;
        estimatedTrainingTime: Integer;
        recommendedBaseModel: String;
        dataAugmentationTips: String;
        resourceRequirements: String;
    };
    
    function ValidateTrainingData(datasetPath: String, validationType: String) returns {
        isValid: Boolean;
        datasetSize: Integer;
        dataQualityScore: Decimal;
        issues: String;
        recommendations: String;
        formatCorrect: Boolean;
        labelDistribution: String;
        duplicateRate: Decimal;
    };
    
    function GetModelComparison(modelIds: String) returns {
        comparisonMatrix: String;
        performanceMetrics: String;
        strengthsWeaknesses: String;
        recommendations: String;
        bestForUseCase: String;
        costBenefitAnalysis: String;
    };
    
    function GetFineTuningQueue() returns {
        queuedTasks: String;
        runningTasks: String;
        completedToday: Integer;
        estimatedWaitTime: Integer;
        resourceUtilization: String;
        priorityQueue: String;
    };
    
    // Events for Agent 14 (Embedding Fine-Tuner)
    event ModelCreated : {
        modelId: String;
        modelName: String;
        modelType: String;
        baseModel: String;
        domain: String;
        createdBy: String;
        timestamp: DateTime;
    };
    
    event TrainingStarted : {
        trainingRunId: String;
        modelId: String;
        trainingName: String;
        trainingStrategy: String;
        batchSize: Integer;
        learningRate: Decimal;
        epochs: Integer;
        startedBy: String;
        timestamp: DateTime;
    };
    
    event TrainingCompleted : {
        trainingRunId: String;
        modelId: String;
        trainingName: String;
        status: String;
        durationMinutes: Integer;
        finalAccuracy: Decimal;
        trainingLoss: Decimal;
        validationLoss: Decimal;
        timestamp: DateTime;
    };
    
    event TrainingFailed : {
        trainingRunId: String;
        modelId: String;
        trainingName: String;
        errorMessage: String;
        failureStage: String;
        durationMinutes: Integer;
        timestamp: DateTime;
    };
    
    event EvaluationStarted : {
        evaluationId: String;
        modelId: String;
        evaluationName: String;
        evaluationType: String;
        benchmarkDataset: String;
        startedBy: String;
        timestamp: DateTime;
    };
    
    event EvaluationCompleted : {
        evaluationId: String;
        modelId: String;
        evaluationName: String;
        overallScore: Decimal;
        improvementPercent: Decimal;
        durationMinutes: Integer;
        timestamp: DateTime;
    };
    
    event OptimizationStarted : {
        optimizationId: String;
        modelId: String;
        optimizationName: String;
        optimizationType: String;
        startedBy: String;
        timestamp: DateTime;
    };
    
    event OptimizationCompleted : {
        optimizationId: String;
        modelId: String;
        optimizationType: String;
        sizeReductionPercent: Decimal;
        speedImprovementPercent: Decimal;
        accuracyLossPercent: Decimal;
        timestamp: DateTime;
    };
    
    event ModelDeployedToProduction : {
        modelId: String;
        modelName: String;
        deploymentEndpoint: String;
        performanceMetrics: String;
        deployedBy: String;
        timestamp: DateTime;
    };
    
    event QualityThresholdBreached : {
        modelId: String;
        modelName: String;
        metricType: String;
        currentValue: Decimal;
        thresholdValue: Decimal;
        severity: String;
        timestamp: DateTime;
    };
    
    event FineTuningTaskCreated : {
        taskId: String;
        taskName: String;
        baseModel: String;
        targetDomain: String;
        taskType: String;
        priority: String;
        createdBy: String;
        timestamp: DateTime;
    };
    
    event FineTuningTaskCompleted : {
        taskId: String;
        taskName: String;
        status: String;
        generatedModelId: String;
        actualDurationHours: Integer;
        qualityAchieved: Decimal;
        performanceAchievedMs: Integer;
        timestamp: DateTime;
    };
    
    event ModelComparisonCompleted : {
        comparisonId: String;
        baseModelId: String;
        compareModelId: String;
        winnerModelId: String;
        improvementMetrics: String;
        comparisonResults: String;
        timestamp: DateTime;
    };
    
    event ModelExported : {
        modelId: String;
        exportFormat: String;
        exportPath: String;
        optimizationLevel: String;
        exportedBy: String;
        timestamp: DateTime;
    };
    
    event BatchOptimizationStarted : {
        batchId: String;
        modelIds: String;
        optimizationType: String;
        modelCount: Integer;
        startedBy: String;
        timestamp: DateTime;
    };
    
    event BatchOptimizationCompleted : {
        batchId: String;
        totalModels: Integer;
        successfulOptimizations: Integer;
        failedOptimizations: Integer;
        averageSpeedImprovement: Decimal;
        averageSizeReduction: Decimal;
        durationMinutes: Integer;
        timestamp: DateTime;
    };
    
    event EmbeddingTunerError : {
        operation: String;
        errorCode: String;
        errorMessage: String;
        modelId: String;
        trainingRunId: String;
        timestamp: DateTime;
    };
    
    // ============================================
    // AGENT 15 - ORCHESTRATOR SERVICE DEFINITIONS
    // ============================================
    
    // Core Entities for Agent 15
    @odata.draft.enabled
    entity Workflows as projection on db.Workflows;
    

    
    @odata.draft.enabled
    entity WorkflowSteps as projection on db.WorkflowSteps;
    
    @odata.draft.enabled
    entity StepExecutions as projection on db.StepExecutions;
    
    @odata.draft.enabled
    entity OrchestratorTasks as projection on db.OrchestratorTasks;
    
    @odata.draft.enabled
    entity PipelineConfigurations as projection on db.PipelineConfigurations;
    
    // Actions for Agent 15 (Orchestrator)
    @requires: ['Orchestrator', 'Admin']
    action createWorkflow(
        workflowName: String,
        description: String,
        workflowType: String,
        workflowDefinition: String,
        priority: String,
        timeoutMinutes: Integer,
        scheduleConfig: String
    ) returns String;
    
    @requires: ['Orchestrator', 'Developer']
    action executeWorkflow(
        workflowId: String,
        executionName: String,
        inputData: String,
        priority: String,
        executionContext: String
    ) returns String;
    
    @requires: ['Orchestrator', 'Developer']
    action pauseWorkflowExecution(
        executionId: String,
        reason: String
    ) returns String;
    
    @requires: ['Orchestrator', 'Developer']
    action resumeWorkflowExecution(
        executionId: String,
        resumeFromStep: String
    ) returns String;
    
    @requires: ['Orchestrator', 'Developer']
    action cancelWorkflowExecution(
        executionId: String,
        reason: String
    ) returns String;
    
    @requires: ['Orchestrator', 'Developer']
    action retryWorkflowExecution(
        executionId: String,
        retryFromStep: String,
        retryConfig: String
    ) returns String;
    
    @requires: ['Orchestrator', 'Admin']
    action scheduleWorkflow(
        workflowId: String,
        scheduleExpression: String,
        scheduleConfig: String,
        isActive: Boolean
    ) returns String;
    
    @requires: ['Orchestrator', 'Developer']
    action addWorkflowStep(
        workflowId: String,
        stepName: String,
        stepType: String,
        targetAgent: String,
        actionName: String,
        stepConfiguration: String,
        dependencies: String
    ) returns String;
    
    @requires: ['Orchestrator', 'Developer']
    action updateWorkflowStep(
        stepId: String,
        stepConfiguration: String,
        conditions: String,
        retryPolicy: String
    ) returns String;
    
    @requires: ['Orchestrator', 'Developer']
    action createTask(
        taskName: String,
        taskType: String,
        priority: String,
        targetAgent: String,
        taskData: String,
        scheduledTime: DateTime
    ) returns String;
    
    @requires: ['Orchestrator', 'Admin']
    action createPipeline(
        pipelineName: String,
        pipelineType: String,
        configuration: String,
        sourceSystems: String,
        targetSystems: String,
        processingAgents: String
    ) returns String;
    
    @requires: ['Orchestrator', 'Developer']
    action executePipeline(
        pipelineId: String,
        executionParameters: String,
        priority: String
    ) returns String;
    
    @requires: ['Orchestrator', 'Developer']
    action validateWorkflow(
        workflowDefinition: String,
        validationType: String
    ) returns String;
    
    @requires: ['Orchestrator', 'Admin']
    action deployWorkflow(
        workflowId: String,
        targetEnvironment: String,
        deploymentConfig: String
    ) returns String;
    
    @requires: ['Orchestrator', 'Developer']
    action cloneWorkflow(
        sourceWorkflowId: String,
        newWorkflowName: String,
        modifications: String
    ) returns String;
    
    @requires: ['Orchestrator', 'Admin']
    action optimizeWorkflowPerformance(
        workflowId: String,
        optimizationStrategy: String,
        performanceTargets: String
    ) returns String;
    
    // Functions for Agent 15 (Orchestrator)
    function GetOrchestratorStatistics() returns {
        totalWorkflows: Integer;
        activeWorkflows: Integer;
        totalExecutions: Integer;
        runningExecutions: Integer;
        completedExecutions: Integer;
        failedExecutions: Integer;
        totalTasks: Integer;
        queuedTasks: Integer;
        completedTasks: Integer;
        averageWorkflowDuration: Integer;
        totalPipelines: Integer;
        activePipelines: Integer;
        executionsToday: Integer;
        successRate: Decimal;
    };
    
    function GetWorkflowExecutionStatus(executionId: String) returns {
        executionId: String;
        workflowId: String;
        workflowName: String;
        status: String;
        progressPercent: Integer;
        currentStep: String;
        totalSteps: Integer;
        completedSteps: Integer;
        failedSteps: Integer;
        startTime: DateTime;
        estimatedEndTime: DateTime;
        durationMinutes: Integer;
        executionType: String;
        triggeredBy: String;
        errorMessage: String;
        nextSteps: String;
    };
    
    function GetWorkflowPerformanceMetrics(workflowId: String) returns {
        workflowId: String;
        workflowName: String;
        totalExecutions: Integer;
        successfulExecutions: Integer;
        failedExecutions: Integer;
        successRate: Decimal;
        averageDurationMinutes: Integer;
        minDurationMinutes: Integer;
        maxDurationMinutes: Integer;
        lastExecutionTime: DateTime;
        averageStepsCompleted: Integer;
        mostCommonFailureStep: String;
        performanceTrend: String;
        recommendedOptimizations: String;
    };
    
    function GetActiveExecutions() returns {
        executions: String;
        totalRunning: Integer;
        totalQueued: Integer;
        totalPaused: Integer;
        priorityDistribution: String;
        resourceUtilization: String;
        estimatedCompletionTimes: String;
    };
    
    function GetTaskQueue() returns {
        queuedTasks: String;
        runningTasks: String;
        totalTasks: Integer;
        queuedCount: Integer;
        runningCount: Integer;
        completedToday: Integer;
        failedToday: Integer;
        averageWaitTime: Integer;
        priorityQueues: String;
        workerUtilization: String;
    };
    
    function GetWorkflowDependencies(workflowId: String) returns {
        workflowId: String;
        workflowName: String;
        requiredAgents: String;
        stepDependencies: String;
        externalDependencies: String;
        resourceRequirements: String;
        dataFlowMap: String;
        criticalPath: String;
    };
    
    function GetExecutionLogs(executionId: String, logLevel: String) returns {
        executionId: String;
        logs: String;
        stepLogs: String;
        errorLogs: String;
        warningLogs: String;
        debugLogs: String;
        executionTrace: String;
    };
    
    function ValidateWorkflowDefinition(workflowDefinition: String) returns {
        isValid: Boolean;
        syntaxErrors: String;
        logicErrors: String;
        warnings: String;
        recommendations: String;
        requiredAgents: String;
        estimatedDuration: Integer;
        complexityScore: Integer;
    };
    
    function GetPipelineStatus(pipelineId: String) returns {
        pipelineId: String;
        pipelineName: String;
        status: String;
        lastRunTime: DateTime;
        nextRunTime: DateTime;
        totalRuns: Integer;
        successfulRuns: Integer;
        failedRuns: Integer;
        averageRuntimeMinutes: Integer;
        currentBacklog: Integer;
        throughputRate: String;
        healthScore: Decimal;
    };
    
    function GetWorkflowTemplates() returns {
        dataProcessingTemplates: String;
        mlPipelineTemplates: String;
        etlTemplates: String;
        customTemplates: String;
        popularTemplates: String;
        recentlyUsedTemplates: String;
    };
    
    function GetResourceUtilization() returns {
        agentUtilization: String;
        queueUtilization: String;
        systemResources: String;
        bottlenecks: String;
        recommendations: String;
        scalingAdvice: String;
    };
    
    function GetWorkflowInsights(workflowId: String, timeRange: String) returns {
        executionTrends: String;
        performanceAnalytics: String;
        failurePatterns: String;
        optimizationOpportunities: String;
        resourceConsumption: String;
        costAnalysis: String;
    };
    
    // Events for Agent 15 (Orchestrator)
    event WorkflowCreated : {
        workflowId: String;
        workflowName: String;
        workflowType: String;
        createdBy: String;
        timestamp: DateTime;
    };
    
    event WorkflowExecutionStarted : {
        executionId: String;
        workflowId: String;
        workflowName: String;
        executionType: String;
        triggeredBy: String;
        priority: String;
        timestamp: DateTime;
    };
    
    event WorkflowExecutionCompleted : {
        executionId: String;
        workflowId: String;
        workflowName: String;
        status: String;
        durationMinutes: Integer;
        completedSteps: Integer;
        totalSteps: Integer;
        timestamp: DateTime;
    };
    
    event WorkflowExecutionFailed : {
        executionId: String;
        workflowId: String;
        workflowName: String;
        failedStep: String;
        errorMessage: String;
        durationMinutes: Integer;
        timestamp: DateTime;
    };
    
    event WorkflowStepStarted : {
        executionId: String;
        stepId: String;
        stepName: String;
        targetAgent: String;
        actionName: String;
        timestamp: DateTime;
    };
    
    event WorkflowStepCompleted : {
        executionId: String;
        stepId: String;
        stepName: String;
        status: String;
        durationMinutes: Integer;
        agentResponseTimeMs: Integer;
        timestamp: DateTime;
    };
    
    event WorkflowStepFailed : {
        executionId: String;
        stepId: String;
        stepName: String;
        targetAgent: String;
        errorMessage: String;
        retryAttempt: Integer;
        timestamp: DateTime;
    };
    
    event TaskCreated : {
        taskId: String;
        taskName: String;
        taskType: String;
        priority: String;
        targetAgent: String;
        createdBy: String;
        timestamp: DateTime;
    };
    
    event TaskCompleted : {
        taskId: String;
        taskName: String;
        status: String;
        durationMinutes: Integer;
        assignedWorker: String;
        timestamp: DateTime;
    };
    
    event PipelineExecutionStarted : {
        pipelineId: String;
        pipelineName: String;
        executionId: String;
        triggeredBy: String;
        timestamp: DateTime;
    };
    
    event PipelineExecutionCompleted : {
        pipelineId: String;
        pipelineName: String;
        executionId: String;
        status: String;
        recordsProcessed: Integer;
        durationMinutes: Integer;
        timestamp: DateTime;
    };
    
    event WorkflowScheduled : {
        workflowId: String;
        workflowName: String;
        scheduleExpression: String;
        nextRunTime: DateTime;
        scheduledBy: String;
        timestamp: DateTime;
    };
    
    event ResourceThresholdExceeded : {
        resourceType: String;
        currentUsage: Decimal;
        thresholdValue: Decimal;
        affectedWorkflows: String;
        recommendedAction: String;
        timestamp: DateTime;
    };
    
    event WorkflowOptimized : {
        workflowId: String;
        workflowName: String;
        optimizationType: String;
        performanceImprovementPercent: Decimal;
        optimizedBy: String;
        timestamp: DateTime;
    };
    
    event WorkflowDeployed : {
        workflowId: String;
        workflowName: String;
        targetEnvironment: String;
        deploymentStatus: String;
        deployedBy: String;
        timestamp: DateTime;
    };
    
    event SLAViolationDetected : {
        workflowId: String;
        executionId: String;
        slaType: String;
        expectedTime: Integer;
        actualTime: Integer;
        severity: String;
        timestamp: DateTime;
    };
    
    event WorkflowTemplateUsed : {
        templateId: String;
        templateName: String;
        workflowId: String;
        usedBy: String;
        customizations: String;
        timestamp: DateTime;
    };
    
    event OrchestratorError : {
        operation: String;
        errorCode: String;
        errorMessage: String;
        workflowId: String;
        executionId: String;
        timestamp: DateTime;
    };
    
    // ============================================
    // AGENT 0 - DATA PRODUCT AGENT
    // ============================================
    
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE', 'UPDATE'], to: 'DataProductManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity DataProducts as projection on db.DataProducts {
        *
    } actions {
        @requires: ['DataProductManager', 'Admin']
        action generateDublinCore() returns String;
        @requires: ['DataProductManager', 'Admin']
        action updateDublinCore(
            title: String,
            creator: String,
            subject: String,
            description: String,
            publisher: String,
            contributor: String,
            date: Date,
            type: String,
            format: String,
            identifier: String,
            source: String,
            language: String,
            relation: String,
            coverage: String,
            rights: String
        ) returns String;
        @requires: ['DataProductManager', 'Admin']
        action validateMetadata(
            validationOptions: String
        ) returns String;
        @requires: ['DataProductManager', 'Admin']
        action validateSchema(
            schemaData: String
        ) returns String;
        @requires: ['DataProductManager', 'Admin']
        action assessQuality(
            completenessWeight: Integer,
            accuracyWeight: Integer,
            consistencyWeight: Integer,
            timelinessWeight: Integer,
            validityWeight: Integer,
            uniquenessWeight: Integer
        ) returns String;
        @requires: ['DataProductManager', 'Admin']
        action publish(
            targetCatalog: String,
            visibility: String,
            approvalRequired: Boolean,
            notificationEnabled: Boolean
        ) returns String;
        @requires: ['DataProductManager', 'Admin']
        action archive(
            reason: String
        ) returns String;
        @requires: ['authenticated-user']
        action getLineage() returns String;
        @requires: ['DataProductManager', 'Admin']
        action createVersion(
            versionNumber: String,
            changeDescription: String,
            versionType: String,
            autoIncrement: Boolean
        ) returns String;
        @requires: ['authenticated-user']
        action compareVersions(
            fromVersion: String,
            toVersion: String
        ) returns String;
    };

    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE', 'UPDATE'], to: 'DataProductManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity DublinCoreMetadata as projection on db.DublinCoreMetadata;

    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE'], to: 'DataProductManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity IngestionSessions as projection on db.IngestionSessions;

    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE'], to: 'DataProductManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity QualityAssessments as projection on db.QualityAssessments;

    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' },
        { grant: ['CREATE'], to: 'DataProductManager' },
        { grant: 'READ', to: 'authenticated-user' }
    ]
    entity ProductTransformations as projection on db.ProductTransformations;

    // Agent 0 specific actions for data product management
    @requires: ['DataProductManager', 'Admin']
    action getDashboardMetrics() returns {
        totalProducts: Integer;
        activeProducts: Integer;
        averageQuality: Decimal;
        productsByType: String;
        qualityDistribution: String;
        recentActivity: String;
        topContributors: String;
    };

    @requires: ['DataProductManager', 'Admin']
    action importMetadata(
        format: String,
        data: String,
        overwriteExisting: Boolean,
        validateBeforeImport: Boolean,
        createBackup: Boolean
    ) returns String;

    @requires: ['DataProductManager', 'Admin']
    action exportCatalog(
        format: String,
        includePrivate: Boolean
    ) returns {
        success: Boolean;
        downloadUrl: String;
        exportId: String;
        expiresAt: String;
    };

    @requires: ['DataProductManager', 'Admin']
    action bulkUpdateProducts(
        productIds: array of String,
        updateData: String
    ) returns String;

    @requires: ['DataProductManager', 'Admin']
    action batchValidateProducts(
        productIds: array of String,
        validationType: String
    ) returns String;

    // ================================
    // DATA PRODUCT EVENTS
    // ================================

    event DataProductCreated : {
        productId: String;
        productName: String;
        timestamp: DateTime;
        createdBy: String;
    };

    event DataProductUpdated : {
        productId: String;
        timestamp: DateTime;
        modifiedBy: String;
    };

    event DataProductDeleted : {
        productId: String;
        timestamp: DateTime;
        deletedBy: String;
    };

    event DublinCoreGenerated : {
        productId: String;
        timestamp: DateTime;
        generatedBy: String;
    };

    event DublinCoreUpdated : {
        productId: String;
        timestamp: DateTime;
        updatedBy: String;
    };

    event MetadataValidated : {
        productId: String;
        isValid: Boolean;
        score: Decimal;
        timestamp: DateTime;
    };

    event SchemaValidated : {
        productId: String;
        isValid: Boolean;
        errors: String;
        timestamp: DateTime;
    };

    event QualityAssessed : {
        productId: String;
        overallScore: Decimal;
        qualityDimensions: String;
        timestamp: DateTime;
    };

    event DataProductPublished : {
        productId: String;
        publicationId: String;
        catalogUrl: String;
        timestamp: DateTime;
        publishedBy: String;
    };

    event DataProductArchived : {
        productId: String;
        reason: String;
        timestamp: DateTime;
        archivedBy: String;
    };

    event DataProductVersionCreated : {
        productId: String;
        versionId: String;
        versionNumber: String;
        timestamp: DateTime;
        createdBy: String;
    };

    event MetadataImported : {
        importedCount: Integer;
        skippedCount: Integer;
        errorCount: Integer;
        timestamp: DateTime;
        importedBy: String;
    };

    event CatalogExported : {
        exportId: String;
        format: String;
        timestamp: DateTime;
        exportedBy: String;
    };

    event ProductsBulkUpdated : {
        productIds: array of String;
        updatedCount: Integer;
        failedCount: Integer;
        timestamp: DateTime;
        updatedBy: String;
    };

    event ProductsBatchValidated : {
        validationId: String;
        totalProducts: Integer;
        validProducts: Integer;
        invalidProducts: Integer;
        timestamp: DateTime;
    };
}