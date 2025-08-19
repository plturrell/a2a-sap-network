// Draft Entities for A2A Network
// SAP Standard Draft-enabled entities for enterprise workflows

namespace a2a.network.draft;

using { cuid, managed } from '@sap/cds/common';

/**
 * Draft Administrative Data
 * Standard SAP draft administrative information
 */
entity DraftAdministrativeData : cuid, managed {
    DraftUUID : UUID;
    CreationDateTime : DateTime;
    CreatedByUser : String(256);
    DraftIsCreatedByMe : Boolean default false;
    DraftIsProcessedByMe : Boolean default false;
    DraftIsKeptByMe : Boolean default false;
    EnqueueStartDateTime : DateTime;
    DraftEntityCreationDateTime : DateTime;
    DraftEntityLastChangeDateTime : DateTime;
    HasActiveEntity : Boolean default false;
    HasDraftEntity : Boolean default false;
    ProcessingStartedByUser : String(256);
    ProcessingStartedByUserDescription : String(256);
    LastChangedByUser : String(256);
    LastChangedByUserDescription : String(256);
    LastChangeDateTime : DateTime;
    InProcessByUser : String(256);
    InProcessByUserDescription : String(256);
}

/**
 * Draft-enabled Requests entity
 * Supports draft mode for request creation and editing
 */
entity RequestDrafts : cuid, managed {
    // Core request fields
    title : String(200);
    description : String(2000);
    status : String(20) default 'DRAFT';
    priority : String(10) default 'MEDIUM';
}

/**
 * Draft-enabled Responses entity
 * Supports draft mode for response creation and editing
 */
entity ResponseDrafts : cuid, managed {
    // Core response fields
    content : String(5000);
    responseType : String(50);
    status : String(20) default 'DRAFT';
}

/**
 * Draft-enabled Agents entity
 * Supports draft mode for agent configuration
 */
entity AgentDrafts : cuid, managed {
    // Core agent fields
    name : String(200);
    description : String(1000);
    agentType : String(50);
    status : String(20) default 'DRAFT';
}

/**
 * Draft-enabled Services entity
 * Supports draft mode for service creation and editing
 */
entity ServiceDrafts : cuid, managed {
    // Core service fields
    name : String(200);
    description : String(2000);
    serviceType : String(50);
    status : String(20) default 'DRAFT';
}

/**
 * Draft-enabled Workflows entity
 * Supports draft mode for workflow creation and editing
 */
entity WorkflowDrafts : cuid, managed {
    // Core workflow fields
    name : String(200);
    description : String(2000);
    workflowType : String(50);
    status : String(20) default 'DRAFT';
}

/**
 * Draft Conflicts entity
 * Manages draft conflicts and resolution
 */
entity DraftConflicts : cuid, managed {
    // Conflict information
    conflictType : String(50);
    conflictDescription : String(2000);
    conflictStatus : String(20) default 'PENDING';
    
    // Resolution information
    resolutionStrategy : String(50);
    resolutionNotes : String(2000);
    resolvedBy : String(256);
    resolvedAt : DateTime;
}
