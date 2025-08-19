/**
 * @fileoverview SAP Draft-Enabled Service for Fiori Elements
 * @since 1.0.0
 * @module draft-service
 * 
 * Provides draft-enabled entities for UI5 applications with
 * automatic draft handling, validation, and conflict resolution
 */

using a2a.network.draft as draft from '../db/draft_entities';
using a2a.network as db from '../db/schema';

/**
 * Draft-enabled service for Fiori Elements applications
 * @since 1.0.0
 */
@requires: ['authenticated-user']
@Common.Label: 'A2A Draft Service'
service A2ADraftService @(path: '/api/v1/drafts') {
    
    /**
     * Draft-enabled Agent entity with full CRUD + draft operations
     * @since 1.0.0
     */
    @odata.draft.enabled
    @Common.Label: 'Agents'
    @Capabilities: {
        SearchRestrictions: { Searchable: true },
        InsertRestrictions: { Insertable: true },
        UpdateRestrictions: { Updatable: true },
        DeleteRestrictions: { Deletable: true }
    }
    entity Agents as projection on draft.AgentDrafts {
        *,
        @UI.Hidden: false
        capabilities : Composition of many AgentCapabilities on capabilities.agent.ID = $self.ID,
        @UI.Hidden: false
        services : Association to many ServiceDrafts on services.provider = $self,
        @UI.Hidden: false
        performance : Association to one AgentPerformance on performance.agent.ID = $self.ID
    } actions {
        @Common.SideEffects: {
            TargetProperties: ['reputation', 'isActive']
        }
        action registerOnBlockchain() returns String;
        
        @Common.SideEffects: {
            TargetProperties: ['reputation']
        }
        action updateReputation(
            @Common.Label: 'New Score'
            score: Integer @assert.range: [0, 1000]
        ) returns Boolean;
        
        @Common.SideEffects: {
            TargetProperties: ['isActive']
        }
        action deactivate() returns Boolean;
    };
    
    /**
     * Draft-enabled Service entity
     * @since 1.0.0
     */
    @odata.draft.enabled
    @Common.Label: 'Service Drafts'
    @Capabilities: {
        SearchRestrictions: { Searchable: true },
        InsertRestrictions: { Insertable: true },
        UpdateRestrictions: { Updatable: true },
        DeleteRestrictions: { Deletable: true }
    }
    entity ServiceDrafts as projection on draft.ServiceDrafts {
        *,
        @UI.Hidden: false
        provider : Association to one Agents on provider.ID = $self.provider.ID,
        @UI.Hidden: false
        orders : Composition of many ServiceOrders on orders.service.ID = $self.ID
    } actions {
        @Common.SideEffects: {
            TargetProperties: ['isListed', 'lastUpdated']
        }
        action listOnMarketplace() returns String;
        
        @Common.SideEffects: {
            TargetProperties: ['fee', 'lastUpdated']
        }
        action updatePricing(
            @Common.Label: 'New Price'
            newPrice: Decimal(10,4) @assert.range: [0, 999999.9999]
        ) returns Boolean;
        
        @Common.SideEffects: {
            TargetProperties: ['isActive', 'isListed']
        }
        action deactivate() returns Boolean;
    };
    
    /**
     * Draft-enabled Workflow entity with simulation
     * @since 1.0.0
     */
    @odata.draft.enabled
    @Common.Label: 'Workflows'
    @Capabilities: {
        SearchRestrictions: { Searchable: true },
        InsertRestrictions: { Insertable: true },
        UpdateRestrictions: { Updatable: true },
        DeleteRestrictions: { Deletable: true }
    }
    entity Workflows as projection on draft.WorkflowDrafts {
        *,
        @UI.Hidden: false
        creator : Association to one Agents on creator.ID = $self.creator.ID,
        @UI.Hidden: false
        executions : Association to many WorkflowExecutions on executions.workflow.ID = $self.ID
    } actions {
        @Common.IsActionCritical: true
        action execute(
            @Common.Label: 'Parameters (JSON)'
            parameters: String
        ) returns String;
        
        action validate() returns {
            valid : Boolean;
            errors : array of {
                path : String;
                message : String;
            };
        };
        
        @Common.SideEffects: {
            TargetProperties: ['isPublished', 'publishedAt']
        }
        action publish() returns Boolean;
    };
    
    /**
     * Draft-enabled Request entity
     * @since 1.0.0
     */
    @odata.draft.enabled
    @Common.Label: 'Requests'
    @Capabilities: {
        SearchRestrictions: { Searchable: true },
        InsertRestrictions: { Insertable: true },
        UpdateRestrictions: { Updatable: true },
        DeleteRestrictions: { Deletable: false } // Requests cannot be deleted, only cancelled
    }
    entity Requests as projection on draft.RequestDrafts {
        *,
        @UI.Hidden: false
        requester : Association to one Agents on requester.ID = $self.requester.ID,
        @UI.Hidden: false
        responses : Composition of many Responses on responses.request.ID = $self.ID
    } actions {
        @Common.SideEffects: {
            TargetProperties: ['status', 'processedAt']
        }
        action submit() returns String;
        
        @Common.IsActionCritical: true
        @Common.SideEffects: {
            TargetProperties: ['status']
        }
        action cancel(
            @Common.Label: 'Cancellation Reason'
            reason: String @mandatory
        ) returns Boolean;
        
        action assignAgent(
            @Common.Label: 'Agent'
            agentId: String @mandatory
        ) returns Boolean;
    };
    
    // Supporting entities (non-draft)
    entity AgentCapabilities as projection on db.AgentCapabilities;
    entity AgentPerformance as projection on db.AgentPerformance;
    entity ServiceOrders as projection on db.ServiceOrders;
    entity WorkflowExecutions as projection on db.WorkflowExecutions;
    entity Responses as projection on db.Responses;
    
    // Read-only conflict view
    @readonly
    @Common.Label: 'Draft Conflicts'
    entity DraftConflicts as projection on draft.DraftConflicts;
    
    /**
     * Functions for draft management
     * @since 1.0.0
     */
    
    @Common.Label: 'Get My Drafts'
    function getMyDrafts() returns array of {
        draftId : String;
        entityType : String;
        entityName : String;
        lastModified : Timestamp;
        isExpiring : Boolean;
    };
    
    @Common.Label: 'Check Draft Status'
    function checkDraftStatus(
        @Common.Label: 'Draft ID'
        draftId: String
    ) returns {
        entityExists : Boolean;
        isLocked : Boolean;
        lockedBy : String;
        canEdit : Boolean;
        validationStatus : String enum { valid; invalid; unknown };
    };
    
    @Common.Label: 'Resolve Conflict'
    function resolveConflict(
        @Common.Label: 'Conflict ID'
        conflictId: String,
        @Common.Label: 'Resolution'
        resolution: String enum { forceDraft; keepActive; merge }
    ) returns Boolean;
    
    /**
     * Actions for draft operations
     * @since 1.0.0
     */
    
    @Common.Label: 'Clean Up My Drafts'
    action cleanupMyDrafts() returns {
        cleaned : Integer;
        remaining : Integer;
    };
    
    @Common.Label: 'Extend Draft Timeout'
    action extendDraftTimeout(
        @Common.Label: 'Draft ID'
        draftId: String
    ) returns Timestamp;
}