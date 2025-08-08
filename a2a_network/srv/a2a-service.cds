using a2a.network as db from '../db/schema';

@requires: 'authenticated-user'
service A2AService @(path: '/api/v1') {
    
    // Agent Management
    @odata.draft.enabled
    @cds.redirection.target
    entity Agents as projection on db.Agents {
        *,
        capabilities : redirected to AgentCapabilities,
        services : redirected to Services,
        performance : redirected to AgentPerformance
    } actions {
        action registerOnBlockchain() returns String;
        action updateReputation(score: Integer) returns Boolean;
        action deactivate() returns Boolean;
    };
    
    entity AgentCapabilities as projection on db.AgentCapabilities;
    entity AgentPerformance as projection on db.AgentPerformance;
    entity PerformanceSnapshots as projection on db.PerformanceSnapshots;
    
    // Capability Registry
    @odata.draft.enabled
    entity Capabilities as projection on db.Capabilities actions {
        action registerOnBlockchain() returns String;
        action deprecate(replacementId: String) returns Boolean;
        action sunset() returns Boolean;
    };
    
    // Service Marketplace
    @cds.redirection.target
    entity Services as projection on db.Services actions {
        action listOnMarketplace() returns String;
        action updatePricing(newPrice: Decimal(10,4)) returns Boolean;
        action deactivate() returns Boolean;
    };
    
    entity ServiceOrders as projection on db.ServiceOrders actions {
        action complete(rating: Integer, feedback: String) returns Boolean;
        action dispute(reason: String) returns Boolean;
        action releaseEscrow() returns Boolean;
    };
    
    // Message Routing
    entity Messages as projection on db.Messages {
        *,
        sender : redirected to Agents,
        recipient : redirected to Agents
    } actions {
        action retry() returns Boolean;
        action markDelivered() returns Boolean;
    };
    
    // Workflows
    @odata.draft.enabled
    entity Workflows as projection on db.Workflows actions {
        action execute(parameters: String) returns String; // returns executionId
        action validate() returns Boolean;
        action publish() returns Boolean;
    };
    
    @cds.redirection.target
    entity WorkflowExecutions as projection on db.WorkflowExecutions {
        *,
        workflow : redirected to Workflows,
        steps : redirected to WorkflowSteps
    } actions {
        action cancel() returns Boolean;
        action retry() returns String;
    };
    
    entity WorkflowSteps as projection on db.WorkflowSteps;
    
    // Cross-chain
    entity ChainBridges as projection on db.ChainBridges;
    entity CrossChainTransfers as projection on db.CrossChainTransfers;
    
    // Privacy
    entity PrivateChannels as projection on db.PrivateChannels actions {
        action addParticipant(agentAddress: String) returns Boolean;
        action removeParticipant(agentAddress: String) returns Boolean;
    };
    
    entity PrivateMessages as projection on db.PrivateMessages;
    
    // Analytics Views
    @readonly entity TopAgents as projection on db.TopAgents;
    @readonly entity ActiveServices as projection on db.ActiveServices;
    @readonly entity RecentWorkflows as projection on db.RecentWorkflows;
    @readonly entity NetworkStats as projection on db.NetworkStats;
    
    // Configuration
    @requires: 'Admin'
    entity NetworkConfig as projection on db.NetworkConfig;
    
    // Functions for complex operations
    function matchCapabilities(requirements: array of String) returns array of String;
    
    function calculateReputation(agentAddress: String) returns String;
    
    function getNetworkHealth() returns String;
    
    function searchAgents(
        capabilities: array of String,
        minReputation: Integer,
        maxPrice: Decimal(10,4)
    ) returns array of String;
    
    // Actions for blockchain operations
    action deployContract(
        contractType: String enum { Agent; Service; Workflow; },
        parameters: String
    ) returns String;
    
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