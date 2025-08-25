using { managed, cuid } from '@sap/cds/common';
using a2a.goalmanagement.db as db from '../db/goalManagementSchema';

namespace a2a.goalmanagement;

/**
 * A2A Goal Management Service
 * CAP service definitions for enterprise-grade goal tracking
 */

// Service definition
service GoalManagementService @(path: '/api/v1/goal-management') {
  // Core entities - projections on database schema
  @cds.redirection.target entity Agents as projection on db.Agents;
  @cds.redirection.target entity Goals as projection on db.Goals;
  entity GoalProgress as projection on db.GoalProgress;
  entity Milestones as projection on db.Milestones;
  entity AgentMetrics as projection on db.AgentMetrics;
  
  // Enhanced entities
  entity GoalDependencies as projection on db.GoalDependencies;
  entity CollaborativeGoals as projection on db.CollaborativeGoals;
  entity GoalConflicts as projection on db.GoalConflicts;
  entity GoalActivity as projection on db.GoalActivity;
  
  // Analytics and views
  entity SystemAnalytics as projection on db.SystemAnalytics;
  entity AgentGoalSummary as projection on db.AgentGoalSummary;
  entity GoalProgressSummary as projection on db.GoalProgressSummary;
  
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