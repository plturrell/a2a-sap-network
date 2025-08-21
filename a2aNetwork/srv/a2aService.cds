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
    entity AgentCapabilities as projection on db.AgentCapabilities;
    
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
    entity Services as projection on db.Services actions {
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
    entity ServiceOrders as projection on db.ServiceOrders actions {
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
    entity Workflows as projection on db.Workflows actions {
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
        workflow : redirected to Workflows,
        steps : redirected to WorkflowSteps
    } actions {
        @requires: ['WorkflowManager', 'Admin']
        action cancel() returns Boolean;
        @requires: ['WorkflowManager', 'Admin']
        action retry() returns String;
    };
    
    @readonly
    @requires: ['authenticated-user']
    entity WorkflowSteps as projection on db.WorkflowSteps;
    
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
    // REPUTATION SYSTEM ENTITIES
    // ================================
    
    @requires: ['authenticated-user']
    @restrict: [
        { grant: ['READ'], to: 'authenticated-user' },
        { grant: ['CREATE', 'UPDATE', 'DELETE'], to: 'Admin' }
    ]
    entity ReputationTransactions as projection on db.ReputationTransactions;
    
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
    
    @readonly
    @requires: ['authenticated-user']
    entity DailyReputationLimits as projection on db.DailyReputationLimits;
    
    @readonly
    @requires: ['authenticated-user']
    entity ReputationAnalytics as projection on db.ReputationAnalytics;
    
    // Configuration
    @requires: 'Admin'
    entity NetworkConfig as projection on db.NetworkConfig;
    
    // Functions for complex operations
    @requires: ['authenticated-user']
    function matchCapabilities(requirements: array of String) returns array of String;
    
    @requires: ['authenticated-user']
    function calculateReputation(agentAddress: String) returns String;
    
    @requires: ['authenticated-user']
    function getNetworkHealth() returns String;
    
    @requires: ['authenticated-user']
    function searchAgents(
        capabilities: array of String,
        minReputation: Integer,
        maxPrice: Decimal(10,4)
    ) returns array of String;
    
    // ================================
    // REPUTATION SYSTEM FUNCTIONS
    // ================================
    
    @requires: ['authenticated-user']
    function getReputationHistory(
        agentId: String,
        startDate: Date,
        endDate: Date
    ) returns array of String;
    
    @requires: ['authenticated-user']
    function getReputationAnalytics(
        agentId: String,
        period: String
    ) returns String;
    
    @requires: ['authenticated-user']
    function getEndorsementNetwork(
        agentId: String,
        depth: Integer
    ) returns String;
    
    @requires: ['authenticated-user']
    function calculateReputationScore(
        agentId: String
    ) returns String;
    
    @requires: ['authenticated-user']
    function getReputationBadge(
        reputation: Integer
    ) returns String;
    
    @requires: ['authenticated-user']
    function canEndorsePeer(
        fromAgentId: String,
        toAgentId: String,
        amount: Integer
    ) returns Boolean;
    
    // Actions for blockchain operations
    @requires: ['Admin']
    action deployContract(
        contractType: String enum { Agent; Service; Workflow; },
        parameters: String
    ) returns String;
    
    @requires: ['Admin']
    action syncBlockchain() returns String;
    
    // Events for real-time updates
    event AgentRegistered : {
        agentId: String;
        address: String;
        name: String;
        timestamp: DateTime;
    };
    
    event ServiceCreated : {
        serviceId: String;
        providerId: String;
        name: String;
        price: Decimal(10,4);
    };
    
    event ReputationUpdated : {
        agentId: String;
        oldScore: Integer;
        newScore: Integer;
        reason: String;
    };
    
    event WorkflowCompleted : {
        executionId: String;
        workflowId: String;
        status: String;
        gasUsed: Integer;
    };
    
    // ================================
    // REPUTATION SYSTEM EVENTS
    // ================================
    
    event ReputationChanged : {
        agentId: String;
        oldReputation: Integer;
        newReputation: Integer;
        change: Integer;
        reason: String;
        timestamp: DateTime;
    };
    
    event ReputationEndorsed : {
        endorsementId: String;
        fromAgent: String;
        toAgent: String;
        amount: Integer;
        reason: String;
        timestamp: DateTime;
    };
    
    event ReputationMilestoneReached : {
        agentId: String;
        milestone: Integer;
        badge: String;
        timestamp: DateTime;
    };
    
    event TaskCompleted : {
        agentId: String;
        taskId: String;
        complexity: String;
        performance: String;
        reputationEarned: Integer;
    };
    
    event ServiceOrderRated : {
        serviceOrderId: String;
        providerId: String;
        clientId: String;
        rating: Integer;
        feedback: String;
        reputationChange: Integer;
    };
}