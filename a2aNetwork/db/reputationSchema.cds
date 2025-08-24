namespace a2a.reputation;

using { a2a.network as network } from './schema';
using { cuid, managed } from '@sap/cds/common';

// Reputation transaction tracking
entity ReputationTransactions : cuid, managed {
    @Common.Label: 'Agent'
    @assert.integrity
    agent              : Association to network.Agents;
    
    @Common.Label: 'Transaction Type'
    transactionType    : String(50) not null enum {
        TASK_COMPLETION   = 'TASK_COMPLETION';
        SERVICE_RATING    = 'SERVICE_RATING';
        PEER_ENDORSEMENT  = 'PEER_ENDORSEMENT';
        QUALITY_BONUS     = 'QUALITY_BONUS';
        PENALTY           = 'PENALTY';
        MILESTONE_BONUS   = 'MILESTONE_BONUS';
        RECOVERY_REWARD   = 'RECOVERY_REWARD';
    };
    
    @Common.Label: 'Amount'
    @assert.range: [-50, 30]
    amount            : Integer not null;
    
    @Common.Label: 'Reason'
    reason            : String(200);
    
    @Common.Label: 'Context'
    @Core.MediaType: 'application/json'
    context           : LargeString; // JSON context data
    
    @Common.Label: 'Is Automated'
    isAutomated       : Boolean default false;
    
    @Common.Label: 'Created By Agent'
    createdByAgent    : Association to network.Agents;
}

// Peer-to-peer endorsements
entity PeerEndorsements : cuid, managed {
    @Common.Label: 'From Agent'
    @assert.integrity
    fromAgent         : Association to network.Agents not null;
    
    @Common.Label: 'To Agent'
    @assert.integrity
    toAgent           : Association to network.Agents not null;
    
    @Common.Label: 'Endorsement Amount'
    @assert.range: [1, 10]
    amount            : Integer not null;
    
    @Common.Label: 'Endorsement Reason'
    reason            : String(50) not null enum {
        EXCELLENT_COLLABORATION = 'EXCELLENT_COLLABORATION';
        TIMELY_ASSISTANCE = 'TIMELY_ASSISTANCE';
        HIGH_QUALITY_WORK = 'HIGH_QUALITY_WORK';
        KNOWLEDGE_SHARING = 'KNOWLEDGE_SHARING';
        PROBLEM_SOLVING = 'PROBLEM_SOLVING';
        INNOVATION = 'INNOVATION';
        MENTORING = 'MENTORING';
        RELIABILITY = 'RELIABILITY';
    };
    
    @Common.Label: 'Context'
    @Core.MediaType: 'application/json'
    context           : LargeString; // JSON with workflowId, taskId, etc.
    
    @Common.Label: 'Related Workflow'
    workflow          : Association to network.Workflows;
    
    @Common.Label: 'Related Task'
    task              : Association to network.WorkflowSteps;
    
    @Common.Label: 'Expires At'
    expiresAt         : DateTime;
    
    @Common.Label: 'Is Reciprocal'
    @readonly
    isReciprocal      : Boolean default false;
    
    @Common.Label: 'Verification Hash'
    verificationHash  : String(64); // For blockchain verification
}

// Reputation milestones and badges
entity ReputationMilestones : cuid {
    @Common.Label: 'Agent'
    @assert.integrity
    agent             : Association to network.Agents;
    
    @Common.Label: 'Milestone'
    @assert.range: [50, 100, 150, 200]
    milestone         : Integer not null;
    
    @Common.Label: 'Badge Name'
    badgeName         : String(20) not null enum {
        NEWCOMER = 'NEWCOMER';
        ESTABLISHED = 'ESTABLISHED';
        TRUSTED = 'TRUSTED';
        EXPERT = 'EXPERT';
        LEGENDARY = 'LEGENDARY';
    };
    
    @Common.Label: 'Achieved At'
    achievedAt        : DateTime not null;
    
    @Common.Label: 'Badge Metadata'
    @Core.MediaType: 'application/json'
    badgeMetadata     : String(500); // Icon, color, description
}

// Reputation recovery programs
entity ReputationRecovery : cuid, managed {
    @Common.Label: 'Agent'
    @assert.integrity
    agent             : Association to network.Agents;
    
    @Common.Label: 'Recovery Type'
    recoveryType      : String(30) not null enum {
        PROBATION_TASKS = 'PROBATION_TASKS';
        PEER_VOUCHING = 'PEER_VOUCHING';
        TRAINING_COMPLETION = 'TRAINING_COMPLETION';
        COMMUNITY_SERVICE = 'COMMUNITY_SERVICE';
    };
    
    @Common.Label: 'Status'
    status            : String(20) enum {
        PENDING = 'PENDING';
        IN_PROGRESS = 'IN_PROGRESS';
        COMPLETED = 'COMPLETED';
        FAILED = 'FAILED';
    } default 'PENDING';
    
    @Common.Label: 'Requirements'
    @Core.MediaType: 'application/json'
    requirements      : LargeString; // JSON requirements
    
    @Common.Label: 'Progress'
    @Core.MediaType: 'application/json'
    progress          : LargeString; // JSON progress tracking
    
    @Common.Label: 'Reputation Reward'
    reputationReward  : Integer default 20;
    
    @Common.Label: 'Started At'
    startedAt         : DateTime;
    
    @Common.Label: 'Completed At'
    completedAt       : DateTime;
}

// Reputation analytics aggregation
@Analytics: { DCL: 'REPUTATION_ANALYTICS' }
entity ReputationAnalytics : cuid {
    @Common.Label: 'Agent'
    @assert.integrity
    agent                  : Association to network.Agents;
    
    @Common.Label: 'Period Start'
    periodStart           : Date not null;
    
    @Common.Label: 'Period End'
    periodEnd             : Date not null;
    
    @Common.Label: 'Starting Reputation'
    startingReputation    : Integer;
    
    @Common.Label: 'Ending Reputation'
    endingReputation      : Integer;
    
    @Common.Label: 'Total Earned'
    @Analytics.Measure: true
    totalEarned           : Integer default 0;
    
    @Common.Label: 'Total Lost'
    @Analytics.Measure: true
    totalLost             : Integer default 0;
    
    @Common.Label: 'Endorsements Received'
    @Analytics.Measure: true
    endorsementsReceived  : Integer default 0;
    
    @Common.Label: 'Endorsements Given'
    @Analytics.Measure: true
    endorsementsGiven     : Integer default 0;
    
    @Common.Label: 'Unique Endorsers'
    @Analytics.Measure: true
    uniqueEndorsers       : Integer default 0;
    
    @Common.Label: 'Average Transaction'
    @Analytics.Measure: true
    averageTransaction    : Decimal(5,2);
    
    @Common.Label: 'Task Success Rate'
    @Analytics.Measure: true
    taskSuccessRate       : Decimal(5,2);
    
    @Common.Label: 'Service Rating Average'
    @Analytics.Measure: true
    serviceRatingAverage  : Decimal(3,2);
}

// Reputation rules configuration
entity ReputationRules : cuid, managed {
    @Common.Label: 'Rule Name'
    ruleName          : String(100) not null;
    
    @Common.Label: 'Rule Type'
    ruleType          : String(20) not null enum {
        EARNING = 'EARNING';
        PENALTY = 'PENALTY';
        ENDORSEMENT = 'ENDORSEMENT';
        RECOVERY = 'RECOVERY';
        DECAY = 'DECAY';
    };
    
    @Common.Label: 'Condition'
    @Core.MediaType: 'application/json'
    condition         : LargeString; // JSON condition
    
    @Common.Label: 'Action'
    @Core.MediaType: 'application/json'
    action            : LargeString; // JSON action
    
    @Common.Label: 'Is Active'
    isActive          : Boolean default true;
    
    @Common.Label: 'Priority'
    priority          : Integer default 100;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description       : String(500);
}

// Note: Reputation tracking fields are already defined in the main schema.cds file
// The following computed fields would be added to network.Agents if not already present:
// - currentBadge: String(20) @Core.Computed
// - endorsementPower: Integer @Core.Computed  
// - reputationTrend: String(10) @Core.Computed

// Service definition extensions
// Note: The reputation entities are already exposed in the A2AService
// extend service A2AService with {
//     
//     // Note: Reputation actions, functions and events are already defined in the main A2AService
// }