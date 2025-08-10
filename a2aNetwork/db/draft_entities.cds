/**
 * @fileoverview SAP Draft-Enabled Entities for UI5 Applications
 * @since 1.0.0
 * @module draft-entities
 * 
 * Implements SAP Fiori draft handling patterns for improved user experience
 * with automatic save, draft indicators, and conflict resolution
 */

namespace a2a.network.draft;

using { a2a.network } from './schema';
using { 
    cuid, 
    managed,
    sap.common.CodeList
} from '@sap/cds/common';

/**
 * Draft-enabled Agent entity for UI5 editing scenarios
 * Allows users to save work in progress without validation
 * @since 1.0.0
 */
@odata.draft.enabled
@Common.Label: 'Agent Drafts'
@UI.HeaderInfo: {
    TypeName: 'Agent',
    TypeNamePlural: 'Agents',
    Title: { Value: name },
    Description: { Value: address }
}
entity AgentDrafts as projection on network.Agents {
    *,
    @UI.Hidden
    HasActiveEntity : Boolean,
    @UI.Hidden
    HasDraftEntity  : Boolean,
    @UI.Hidden
    DraftAdministrativeData : DraftAdministrativeData
} actions {
    /**
     * Prepare action for draft activation
     * @since 1.0.0
     */
    action draftPrepare() returns AgentDrafts;
    
    /**
     * Edit action to create draft from active instance
     * @since 1.0.0
     */
    action draftEdit() returns AgentDrafts;
    
    /**
     * Activate draft and create active instance
     * @since 1.0.0
     */
    action draftActivate() returns network.Agents;
    
    /**
     * Discard draft changes
     * @since 1.0.0
     */
    action draftDiscard() returns network.Agents;
};

/**
 * Draft-enabled Service entity for UI5 editing
 * @since 1.0.0
 */
@odata.draft.enabled
@Common.Label: 'Service Drafts'
@UI.HeaderInfo: {
    TypeName: 'Service',
    TypeNamePlural: 'Services',
    Title: { Value: name },
    Description: { Value: description }
}
entity ServiceDrafts as projection on network.Services {
    *,
    @UI.Hidden
    HasActiveEntity : Boolean,
    @UI.Hidden
    HasDraftEntity  : Boolean,
    @UI.Hidden
    DraftAdministrativeData : DraftAdministrativeData
} actions {
    action draftPrepare() returns ServiceDrafts;
    action draftEdit() returns ServiceDrafts;
    action draftActivate() returns network.Services;
    action draftDiscard() returns network.Services;
};

/**
 * Draft-enabled Request entity with validation
 * @since 1.0.0
 */
@odata.draft.enabled
@Common.Label: 'Request Drafts'
@UI.HeaderInfo: {
    TypeName: 'Request',
    TypeNamePlural: 'Requests',
    Title: { Value: ID },
    Description: { Value: status }
}
entity RequestDrafts as projection on network.Requests {
    *,
    @UI.Hidden
    HasActiveEntity : Boolean,
    @UI.Hidden
    HasDraftEntity  : Boolean,
    @UI.Hidden
    DraftAdministrativeData : DraftAdministrativeData,
    
    // Additional draft-specific fields
    @Common.Label: 'Validation Messages'
    virtual validationMessages : array of {
        severity : String enum { error; warning; info };
        message  : String;
        target   : String;
    };
    
    @Common.Label: 'Draft Status'
    virtual draftStatus : String enum { 
        new; 
        modified; 
        locked; 
        outdated 
    };
} actions {
    action draftPrepare() returns RequestDrafts;
    action draftEdit() returns RequestDrafts;
    action draftActivate() returns network.Requests;
    action draftDiscard() returns network.Requests;
    
    /**
     * Validate draft before activation
     * @since 1.0.0
     */
    action validateDraft() returns {
        valid : Boolean;
        messages : array of {
            severity : String;
            message  : String;
            target   : String;
        };
    };
};

/**
 * Draft-enabled Workflow configuration
 * @since 1.0.0
 */
@odata.draft.enabled
@Common.Label: 'Workflow Drafts'
@UI.HeaderInfo: {
    TypeName: 'Workflow',
    TypeNamePlural: 'Workflows',
    Title: { Value: name },
    Description: { Value: description }
}
entity WorkflowDrafts as projection on network.Workflows {
    *,
    @UI.Hidden
    HasActiveEntity : Boolean,
    @UI.Hidden
    HasDraftEntity  : Boolean,
    @UI.Hidden
    DraftAdministrativeData : DraftAdministrativeData,
    
    // Workflow-specific draft features
    @Common.Label: 'Preview Available'
    virtual canPreview : Boolean;
    
    @Common.Label: 'Simulation Results'
    virtual simulationResults : {
        estimatedDuration : Integer;
        estimatedCost     : Decimal(10,2);
        warnings          : array of String;
    };
} actions {
    action draftPrepare() returns WorkflowDrafts;
    action draftEdit() returns WorkflowDrafts;
    action draftActivate() returns network.Workflows;
    action draftDiscard() returns network.Workflows;
    
    /**
     * Simulate workflow execution
     * @since 1.0.0
     */
    action simulateWorkflow() returns {
        success : Boolean;
        results : {
            steps : array of {
                name     : String;
                duration : Integer;
                status   : String;
            };
            totalDuration : Integer;
            estimatedCost : Decimal(10,2);
        };
    };
    
    /**
     * Preview workflow visualization
     * @since 1.0.0
     */
    action previewWorkflow() returns {
        graphData : String; // JSON representation
        warnings  : array of String;
    };
};

/**
 * Draft administrative data following SAP standards
 * @since 1.0.0
 */
type DraftAdministrativeData {
    DraftUUID            : UUID;
    CreationDateTime     : Timestamp;
    CreatedByUser        : String;
    LastChangeDateTime   : Timestamp;
    LastChangedByUser    : String;
    DraftIsProcessedByMe : Boolean;
    DraftIsCreatedByMe   : Boolean;
    InProcessByUser      : String;
    DraftIsKeptByUser    : Boolean;
}

/**
 * Conflict resolution view for concurrent editing
 * @since 1.0.0
 */
@readonly
entity DraftConflicts : cuid {
    entityType   : String;
    entityID     : String;
    draftUser    : String;
    activeUser   : String;
    conflictTime : Timestamp;
    resolution   : String enum { 
        pending; 
        forceDraft; 
        keepActive; 
        merged 
    };
}