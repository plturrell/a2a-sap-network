using { managed, cuid } from '@sap/cds/common';

namespace a2a.goalmanagement;

/**
 * A2A Goal Management Service
 * CAP service definitions for enterprise-grade goal tracking
 */

entity Agents : managed {
  key agentId     : String(50);
  agentName       : String(100);
  agentType       : String(50);
  status          : String(20) default 'active';
  capabilities    : array of String;
  lastSeen        : Timestamp;
  goals           : Composition of many Goals on goals.agent = $self;
  metrics         : Composition of many AgentMetrics on metrics.agent = $self;
}

entity Goals : managed, cuid {
  agent           : Association to Agents;
  goalType        : String(50);
  priority        : String(20) enum { critical; high; medium; low };
  status          : String(20) enum { draft; active; paused; completed; cancelled } default 'draft';
  
  // SMART Goal Components
  specific        : LargeString;
  measurable      : Composition of many MeasurableTargets on measurable.goal = $self;
  achievable      : Boolean default true;
  relevant        : LargeString;
  timeBound       : Date;
  
  // Progress Tracking
  overallProgress : Decimal(5,2) default 0.00;
  startDate       : Date;
  targetDate      : Date;
  completedDate   : Date;
  
  // A2A Integration
  assignedVia     : String(20) enum { manual; automated; ai_recommended } default 'manual';
  trackingFrequency : String(20) enum { hourly; daily; weekly } default 'daily';
  
  // AI Enhancement
  aiEnabled       : Boolean default false;
  aiPredictions   : Composition of many AIPredictions on aiPredictions.goal = $self;
  
  // Related Data
  progress        : Composition of many GoalProgress on progress.goal = $self;
  milestones      : Composition of many Milestones on milestones.goal = $self;
  notifications   : Composition of many GoalNotifications on notifications.goal = $self;
}

entity MeasurableTargets : cuid {
  goal            : Association to Goals;
  metricName      : String(100);
  targetValue     : Decimal(10,2);
  currentValue    : Decimal(10,2) default 0.00;
  unit            : String(20);
  progressPercent : Decimal(5,2) default 0.00;
  achieved        : Boolean default false;
}

entity GoalProgress : managed, cuid {
  goal            : Association to Goals;
  timestamp       : Timestamp;
  overallProgress : Decimal(5,2);
  metrics         : LargeString; // JSON string of metric values
  milestoneCount  : Integer default 0;
  notes           : LargeString;
  reportedBy      : String(50) enum { agent; system; ai; manual };
}

entity Milestones : managed, cuid {
  goal            : Association to Goals;
  title           : String(200);
  description     : LargeString;
  achievedDate    : Timestamp;
  detectedBy      : String(50) enum { agent; ai; manual };
  significance    : String(20) enum { minor; major; critical } default 'minor';
  metrics         : LargeString; // JSON string of metrics at achievement
}

entity AIPredictions : managed, cuid {
  goal            : Association to Goals;
  predictionType  : String(50) enum { completion_date; risk_assessment; optimization; milestone };
  prediction      : LargeString; // JSON string with prediction details
  confidence      : Decimal(5,2);
  validUntil      : Timestamp;
  accuracy        : Decimal(5,2); // Measured accuracy after fact
}

entity AgentMetrics : managed, cuid {
  agent           : Association to Agents;
  timestamp       : Timestamp;
  metricType      : String(50);
  metricName      : String(100);
  value           : Decimal(10,4);
  unit            : String(20);
  source          : String(50) enum { agent_sdk; monitoring; blockchain; manual };
}

entity GoalNotifications : managed, cuid {
  goal            : Association to Goals;
  notificationType : String(50) enum { assignment; progress; milestone; completion; risk_alert };
  recipient       : String(100);
  title           : String(200);
  message         : LargeString;
  sent            : Boolean default false;
  sentAt          : Timestamp;
  priority        : String(20) enum { low; medium; high; urgent } default 'medium';
}

// Goal Dependencies for dependency management
entity GoalDependencies : cuid {
  goal            : Association to Goals;
  dependsOnGoal   : Association to Goals;
  dependencyType  : String(50) enum { prerequisite; co_requisite; blocking; informational } default 'prerequisite';
  isBlocking      : Boolean default true;
  description     : String(500);
  validatedAt     : Timestamp;
}

// Collaborative Goals for cross-agent collaboration
entity CollaborativeGoals : managed, cuid {
  title           : String(200);
  description     : LargeString;
  status          : String(20) enum { planning; active; completed; cancelled } default 'planning';
  overallProgress : Decimal(5,2) default 0.00;
  coordinator     : Association to Agents;
  participants    : Composition of many CollaborativeParticipants on participants.collaborativeGoal = $self;
  milestones      : Composition of many CollaborativeMilestones on milestones.collaborativeGoal = $self;
  targetDate      : Date;
  completedDate   : Date;
}

entity CollaborativeParticipants : cuid {
  collaborativeGoal : Association to CollaborativeGoals;
  agent           : Association to Agents;
  role            : String(50) enum { leader; contributor; reviewer; observer } default 'contributor';
  contribution    : Decimal(5,2) default 0.00;
  joinedAt        : Timestamp;
  responsibilities : LargeString;
}

entity CollaborativeMilestones : managed, cuid {
  collaborativeGoal : Association to CollaborativeGoals;
  title           : String(200);
  description     : LargeString;
  achievedDate    : Timestamp;
  significance    : String(20) enum { low; medium; high; critical } default 'medium';
  contributingAgents : array of String(50);
}

// Goal Conflicts for tracking and resolving conflicts
entity GoalConflicts : managed, cuid {
  goal1           : Association to Goals;
  goal2           : Association to Goals;
  conflictType    : String(50) enum { resource; timeline; priority; objective } default 'resource';
  severity        : String(20) enum { low; medium; high; critical } default 'medium';
  status          : String(20) enum { identified; analyzing; resolved; accepted } default 'identified';
  description     : LargeString;
  resolution      : LargeString;
  resolvedAt      : Timestamp;
  resolvedBy      : String(100);
}

// Goal Activity for tracking all goal-related activities
entity GoalActivity : managed, cuid {
  goal            : Association to Goals;
  agent           : Association to Agents;
  timestamp       : Timestamp;
  activityType    : String(50) enum { created; updated; progress_tracked; milestone_achieved; status_changed; conflict_detected; dependency_added };
  description     : String(500);
  metadata        : LargeString; // JSON with activity-specific data
}

entity SystemAnalytics : managed, cuid {
  timestamp       : Timestamp;
  totalAgents     : Integer;
  activeGoals     : Integer;
  completedGoals  : Integer;
  averageProgress : Decimal(5,2);
  totalMilestones : Integer;
  agentsAbove50   : Integer;
  systemHealth    : String(20) enum { excellent; good; fair; poor } default 'good';
  analyticsData   : LargeString; // JSON string with detailed analytics
}

// Views for efficient querying
view AgentGoalSummary as select from Agents {
  agentId,
  agentName,
  status,
  goals.goalType,
  goals.priority,
  goals.overallProgress,
  goals.status as goalStatus,
  goals.targetDate,
  count(goals) as totalGoals : Integer,
  avg(goals.overallProgress) as avgProgress : Decimal(5,2)
} group by agentId, agentName, status, goals.goalType, goals.priority, goals.overallProgress, goals.status, goals.targetDate;

view GoalProgressSummary as select from Goals {
  ID,
  agent.agentId,
  agent.agentName,
  goalType,
  priority,
  status,
  overallProgress,
  targetDate,
  count(progress) as progressUpdates : Integer,
  count(milestones) as milestonesAchieved : Integer,
  max(progress.timestamp) as lastUpdate : Timestamp
} group by ID, agent.agentId, agent.agentName, goalType, priority, status, overallProgress, targetDate;

view SystemMetricsSummary as select from SystemAnalytics {
  timestamp,
  totalAgents,
  activeGoals,
  completedGoals,
  averageProgress,
  totalMilestones,
  agentsAbove50,
  systemHealth
} order by timestamp desc;

// Visualization-specific views
view GoalDependencyGraph as select from GoalDependencies {
  goal.ID as sourceGoalId,
  goal.specific as sourceGoalTitle,
  goal.agent.agentId as sourceAgentId,
  goal.status as sourceStatus,
  goal.overallProgress as sourceProgress,
  dependsOnGoal.ID as targetGoalId,
  dependsOnGoal.specific as targetGoalTitle,
  dependsOnGoal.agent.agentId as targetAgentId,
  dependsOnGoal.status as targetStatus,
  dependsOnGoal.overallProgress as targetProgress,
  dependencyType,
  isBlocking
};

view CollaborativeGoalNetwork as select from CollaborativeParticipants {
  collaborativeGoal.ID as goalId,
  collaborativeGoal.title as goalTitle,
  collaborativeGoal.status as goalStatus,
  collaborativeGoal.overallProgress as goalProgress,
  agent.agentId,
  agent.agentName,
  role,
  contribution,
  joinedAt
};

view ConflictAnalysis as select from GoalConflicts {
  ID as conflictId,
  goal1.ID as goal1Id,
  goal1.specific as goal1Title,
  goal1.agent.agentId as goal1AgentId,
  goal2.ID as goal2Id,
  goal2.specific as goal2Title,
  goal2.agent.agentId as goal2AgentId,
  conflictType,
  severity,
  status,
  createdAt,
  resolvedAt
} where status != 'resolved';

// Service definition
service GoalManagementService @(path: '/api/v1/goal-management') {
  // Core entities
  entity Agents as projection on a2a.goalmanagement.Agents;
  entity Goals as projection on a2a.goalmanagement.Goals;
  entity GoalProgress as projection on a2a.goalmanagement.GoalProgress;
  entity Milestones as projection on a2a.goalmanagement.Milestones;
  entity AgentMetrics as projection on a2a.goalmanagement.AgentMetrics;
  
  // Enhanced entities
  entity GoalDependencies as projection on a2a.goalmanagement.GoalDependencies;
  entity CollaborativeGoals as projection on a2a.goalmanagement.CollaborativeGoals;
  entity GoalConflicts as projection on a2a.goalmanagement.GoalConflicts;
  entity GoalActivity as projection on a2a.goalmanagement.GoalActivity;
  
  // Analytics and views
  entity SystemAnalytics as projection on a2a.goalmanagement.SystemAnalytics;
  entity AgentGoalSummary as projection on a2a.goalmanagement.AgentGoalSummary;
  entity GoalProgressSummary as projection on a2a.goalmanagement.GoalProgressSummary;
  
  // Visualization endpoints
  @readonly entity GoalVisualization {
    key type : String enum { overview; progress_timeline; agent_comparison; goal_heatmap; dependency_graph; collaborative_goals };
    agentId : String;
    dateRange : Integer;
    data : LargeString; // JSON response from visualization functions
  }
  
  // Actions for goal management
  action assignGoal(agentId: String, goalData: String) returns String;
  action updateProgress(goalId: String, progressData: String) returns String;
  action detectConflicts(goalId: String) returns array of String;
  action resolveConflict(conflictId: String, resolution: String) returns Boolean;
  action createCollaborativeGoal(goalData: String, participants: array of String) returns String;
  action generateAIPredictions(goalId: String) returns String;
  
  // Goal synchronization actions
  action syncGoals() returns {
    status: String;
    message: String;
    result: {
      status: String;
      timestamp: Timestamp;
      successCount: Integer;
      failureCount: Integer;
      totalAgents: Integer;
    }
  };
  
  action getSyncStatus() returns {
    running: Boolean;
    interval: Integer;
    nextSync: Timestamp;
    serverTime: Timestamp;
  };
}
