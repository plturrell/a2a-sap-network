using a2a.network as db from '../db/schema';

@requires: ['authenticated-user', {kind: 'any', grant: 'READ'}]
service A2AService @(path: '/api/v1') {
    
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
        *,
        capabilities : redirected to AgentCapabilities,
        services : redirected to Services,
        performance : redirected to AgentPerformance
    } actions {
        @requires: ['AgentManager', 'Admin']
        action registerOnBlockchain() returns String;
        @requires: ['AgentManager', 'Admin']
        action updateReputation(score: Integer) returns Boolean;
        @requires: ['Admin']
        action deactivate() returns Boolean;
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
}