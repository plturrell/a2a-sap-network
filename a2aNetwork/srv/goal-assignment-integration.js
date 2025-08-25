const cds = require('@sap/cds');
const { SELECT, INSERT, UPDATE, DELETE, UPSERT } = cds.ql;

/**
 * Goal Assignment Integration Service
 * Connects the A2A orchestrator goal assignment system with the CAP UI
 */
module.exports = class GoalAssignmentIntegration {
    
    constructor() {
        this.LOG = cds.log('goal-assignment-integration');
        this.ORCHESTRATOR_URL = process.env.A2A_ORCHESTRATOR_URL || 'http://localhost:8001';
    }
    
    /**
     * Sync all agent goals from orchestrator to CAP database
     */
    async syncAllAgentGoals() {
        try {
            this.LOG.info('Starting full goal synchronization from orchestrator');
            
            const axios = require('axios');
            
            // Get all 16 agents
            const agents = [
                { id: 'agent0_data_product', name: 'Data Product Agent', type: 'data_pipeline' },
                { id: 'agent1_standardization', name: 'Data Standardization Agent', type: 'data_pipeline' },
                { id: 'agent2_ai_preparation', name: 'AI Data Preparation Agent', type: 'data_pipeline' },
                { id: 'agent3_vector_processing', name: 'Vector Processing Agent', type: 'data_pipeline' },
                { id: 'agent4_calc_validation', name: 'Calculation Validation Agent', type: 'data_pipeline' },
                { id: 'agent5_qa_validation', name: 'QA Validation Agent', type: 'data_pipeline' },
                { id: 'agent6_quality_control', name: 'Quality Control Manager', type: 'management' },
                { id: 'agent7_agent_manager', name: 'Agent Manager', type: 'management' },
                { id: 'agent8_data_manager', name: 'Data Manager', type: 'management' },
                { id: 'agent9_reasoning', name: 'Reasoning Agent', type: 'specialized' },
                { id: 'agent10_calculation', name: 'Calculation Agent', type: 'specialized' },
                { id: 'agent11_sql', name: 'SQL Agent', type: 'specialized' },
                { id: 'agent12_catalog_manager', name: 'Catalog Manager', type: 'infrastructure' },
                { id: 'agent13_agent_builder', name: 'Agent Builder', type: 'infrastructure' },
                { id: 'agent14_embedding_finetuner', name: 'Embedding Fine-Tuner', type: 'infrastructure' },
                { id: 'agent15_orchestrator', name: 'Orchestrator Agent', type: 'infrastructure' }
            ];
            
            let successCount = 0;
            let failureCount = 0;
            
            for (const agent of agents) {
                try {
                    // Get goals from orchestrator
                    const response = await axios.get(`${this.ORCHESTRATOR_URL}/api/v1/agents/${agent.id}/goals`, {
                        headers: {
                            'X-A2A-Service': 'goal-assignment-integration',
                            'Authorization': `Bearer system`
                        }
                    });
                    
                    if (response.data && response.data.goals) {
                        await this._syncAgentToCAP(agent, response.data);
                        successCount++;
                        this.LOG.info(`Successfully synced goals for ${agent.id}`);
                    }
                    
                } catch (error) {
                    // If orchestrator is not available, use the last assignment results
                    this.LOG.warn(`Failed to fetch from orchestrator for ${agent.id}, using fallback`);
                    await this._syncFromAssignmentResults(agent);
                    failureCount++;
                }
            }
            
            // Update system analytics
            await this._updateSystemAnalytics();
            
            this.LOG.info(`Goal sync completed: ${successCount} success, ${failureCount} failures`);
            
            return {
                status: 'completed',
                timestamp: new Date(),
                successCount,
                failureCount,
                totalAgents: agents.length
            };
            
        } catch (error) {
            this.LOG.error('Failed to sync agent goals', { error: error.message });
            throw error;
        }
    }
    
    /**
     * Sync individual agent data to CAP database
     */
    async _syncAgentToCAP(agent, goalsData) {
        const srv = await cds.connect.to('GoalManagementService');
        
        // Upsert agent
        await srv.run(UPSERT.into('Agents').entries({
            agentId: agent.id,
            agentName: agent.name,
            agentType: agent.type,
            status: 'active',
            lastSeen: new Date(),
            capabilities: goalsData.capabilities || []
        }));
        
        // Sync goals
        if (goalsData.goals && goalsData.goals.primary_objectives) {
            for (const goal of goalsData.goals.primary_objectives) {
                // Create or update goal
                await srv.run(UPSERT.into('Goals').entries({
                    ID: goal.goal_id,
                    agent_agentId: agent.id,
                    goalType: goal.goal_type,
                    priority: goal.priority || 'medium',
                    status: goal.status || 'active',
                    specific: goal.specific,
                    achievable: goal.achievable !== false,
                    relevant: goal.relevant || goal.specific,
                    timeBound: goal.time_bound || '30 days',
                    overallProgress: goal.progress || 0,
                    targetDate: goal.target_date,
                    startDate: goal.assigned_date || goal.created_at,
                    aiEnabled: goal.ai_enabled || false,
                    assignedVia: 'automated',
                    trackingFrequency: 'daily'
                }));
                
                // Sync measurable targets
                if (goal.measurable) {
                    // Delete existing measurable targets
                    await srv.run(DELETE.from('MeasurableTargets').where({ goal_ID: goal.goal_id }));
                    
                    // Insert new measurable targets
                    const measurableEntries = Object.entries(goal.measurable).map(([metric, target]) => ({
                        goal_ID: goal.goal_id,
                        metricName: metric,
                        targetValue: parseFloat(target) || 0,
                        currentValue: 0,
                        unit: this._getMetricUnit(metric),
                        progressPercent: 0,
                        achieved: false
                    }));
                    
                    await srv.run(INSERT.into('MeasurableTargets').entries(measurableEntries));
                }
                
                // Create initial progress entry
                await srv.run(INSERT.into('GoalProgress').entries({
                    goal_ID: goal.goal_id,
                    timestamp: new Date(),
                    overallProgress: 0,
                    metrics: JSON.stringify({}),
                    reportedBy: 'system',
                    notes: 'Goal synchronized from orchestrator'
                }));
                
                // Create goal activity
                await srv.run(INSERT.into('GoalActivity').entries({
                    goal_ID: goal.goal_id,
                    agent_agentId: agent.id,
                    timestamp: new Date(),
                    activityType: 'created',
                    description: `Goal ${goal.goal_type} assigned via orchestrator`,
                    metadata: JSON.stringify({
                        source: 'orchestrator',
                        syncedAt: new Date().toISOString()
                    })
                }));
            }
        }
    }
    
    /**
     * Fallback sync from assignment results file
     */
    async _syncFromAssignmentResults(agent) {
        try {
            const fs = require('fs').promises;
            const path = require('path');
            
            // Find the latest assignment results file
            const resultsDir = path.join(__dirname, '../../a2aAgents/backend/app/a2a/agents/orchestratorAgent/active');
            const files = await fs.readdir(resultsDir);
            const resultFiles = files.filter(f => f.startsWith('goal_assignment_results_')).sort().reverse();
            
            if (resultFiles.length > 0) {
                const latestFile = path.join(resultsDir, resultFiles[0]);
                const content = await fs.readFile(latestFile, 'utf8');
                const results = JSON.parse(content);
                
                if (results.assignments && results.assignments[agent.id]) {
                    const assignment = results.assignments[agent.id];
                    if (assignment.status === 'success' && assignment.goals) {
                        // Convert to expected format
                        const goalsData = {
                            goals: {
                                primary_objectives: assignment.goals
                            }
                        };
                        
                        await this._syncAgentToCAP(agent, goalsData);
                        this.LOG.info(`Synced ${agent.id} from assignment results file`);
                    }
                }
            }
        } catch (error) {
            this.LOG.error(`Failed to sync from assignment results for ${agent.id}`, { error: error.message });
        }
    }
    
    /**
     * Update system-wide analytics
     */
    async _updateSystemAnalytics() {
        const srv = await cds.connect.to('GoalManagementService');
        
        // Get current statistics
        const agents = await srv.run(SELECT.from('Agents'));
        const goals = await srv.run(SELECT.from('Goals').where({ status: 'active' }));
        const completedGoals = await srv.run(SELECT.from('Goals').where({ status: 'completed' }));
        const milestones = await srv.run(SELECT.from('Milestones'));
        
        // Calculate metrics
        const totalProgress = goals.reduce((sum, g) => sum + (g.overallProgress || 0), 0);
        const averageProgress = goals.length > 0 ? totalProgress / goals.length : 0;
        const agentsAbove50 = new Set(goals.filter(g => g.overallProgress >= 50).map(g => g.agent_agentId)).size;
        
        // Determine system health
        let systemHealth = 'good';
        if (averageProgress < 25) systemHealth = 'poor';
        else if (averageProgress < 50) systemHealth = 'fair';
        else if (averageProgress >= 75) systemHealth = 'excellent';
        
        // Store analytics
        await srv.run(INSERT.into('SystemAnalytics').entries({
            timestamp: new Date(),
            totalAgents: agents.length,
            activeGoals: goals.length,
            completedGoals: completedGoals.length,
            averageProgress: averageProgress,
            totalMilestones: milestones.length,
            agentsAbove50: agentsAbove50,
            systemHealth: systemHealth,
            analyticsData: JSON.stringify({
                timestamp: new Date().toISOString(),
                agentBreakdown: this._getAgentBreakdown(goals),
                goalTypeDistribution: this._getGoalTypeDistribution(goals),
                priorityDistribution: this._getPriorityDistribution(goals),
                progressDistribution: this._getProgressDistribution(goals)
            })
        }));
    }
    
    /**
     * Get metric unit based on metric name
     */
    _getMetricUnit(metricName) {
        const unitMap = {
            'success_rate': '%',
            'registration_success_rate': '%',
            'standardization_success_rate': '%',
            'validation_accuracy': '%',
            'qa_pass_rate': '%',
            'agent_uptime_percentage': '%',
            'error_rate': '%',
            'false_positive_rate': '%',
            'response_time': 'sec',
            'avg_registration_time': 'sec',
            'avg_transformation_time': 'sec',
            'avg_review_time_hours': 'hours',
            'mean_time_to_repair': 'min',
            'avg_retrieval_time_ms': 'ms',
            'avg_query_time_ms': 'ms',
            'avg_query_execution_time': 'ms',
            'health_check_response_time': 'ms',
            'throughput': '/hr',
            'vector_generation_throughput': '/hr',
            'validation_throughput': '/hr',
            'calculation_throughput': '/hr',
            'inference_throughput': '/hr',
            'pipeline_throughput': '/hr',
            'quality_score': 'points',
            'data_quality_score': 'points',
            'feature_quality_score': 'points',
            'embedding_quality_score': 'points',
            'code_quality_score': 'points',
            'deployment_time': 'min',
            'avg_deployment_time': 'min',
            'convergence_speed': 'epochs'
        };
        
        return unitMap[metricName] || '';
    }
    
    /**
     * Analytics helper methods
     */
    _getAgentBreakdown(goals) {
        const breakdown = {};
        goals.forEach(goal => {
            if (!breakdown[goal.agent_agentId]) {
                breakdown[goal.agent_agentId] = {
                    totalGoals: 0,
                    averageProgress: 0,
                    goalTypes: {}
                };
            }
            breakdown[goal.agent_agentId].totalGoals++;
            breakdown[goal.agent_agentId].averageProgress += goal.overallProgress || 0;
            breakdown[goal.agent_agentId].goalTypes[goal.goalType] = 
                (breakdown[goal.agent_agentId].goalTypes[goal.goalType] || 0) + 1;
        });
        
        // Calculate averages
        Object.keys(breakdown).forEach(agentId => {
            if (breakdown[agentId].totalGoals > 0) {
                breakdown[agentId].averageProgress = 
                    breakdown[agentId].averageProgress / breakdown[agentId].totalGoals;
            }
        });
        
        return breakdown;
    }
    
    _getGoalTypeDistribution(goals) {
        const distribution = {};
        goals.forEach(goal => {
            distribution[goal.goalType] = (distribution[goal.goalType] || 0) + 1;
        });
        return distribution;
    }
    
    _getPriorityDistribution(goals) {
        const distribution = {
            critical: 0,
            high: 0,
            medium: 0,
            low: 0
        };
        goals.forEach(goal => {
            if (distribution.hasOwnProperty(goal.priority)) {
                distribution[goal.priority]++;
            }
        });
        return distribution;
    }
    
    _getProgressDistribution(goals) {
        const bins = {
            '0-25': 0,
            '26-50': 0,
            '51-75': 0,
            '76-100': 0
        };
        
        goals.forEach(goal => {
            const progress = goal.overallProgress || 0;
            if (progress <= 25) bins['0-25']++;
            else if (progress <= 50) bins['26-50']++;
            else if (progress <= 75) bins['51-75']++;
            else bins['76-100']++;
        });
        
        return bins;
    }
    
    /**
     * Create collaborative goals based on agent profiles
     */
    async createCollaborativeGoals() {
        const srv = await cds.connect.to('GoalManagementService');
        
        const collaborativeConfigs = [
            {
                title: 'End-to-End Data Pipeline Optimization',
                description: 'Optimize the complete data pipeline from ingestion to AI-ready data',
                participants: [
                    { agentId: 'agent0_data_product', role: 'leader' },
                    { agentId: 'agent1_standardization', role: 'contributor' },
                    { agentId: 'agent2_ai_preparation', role: 'contributor' }
                ],
                targetDate: new Date(Date.now() + 60 * 24 * 60 * 60 * 1000) // 60 days
            },
            {
                title: 'Advanced Embedding Quality Enhancement',
                description: 'Improve embedding quality through collaborative processing and fine-tuning',
                participants: [
                    { agentId: 'agent3_vector_processing', role: 'leader' },
                    { agentId: 'agent14_embedding_finetuner', role: 'contributor' }
                ],
                targetDate: new Date(Date.now() + 45 * 24 * 60 * 60 * 1000) // 45 days
            },
            {
                title: 'Mathematical Computation Excellence',
                description: 'Achieve 99.99% accuracy in mathematical computations with validation',
                participants: [
                    { agentId: 'agent4_calc_validation', role: 'contributor' },
                    { agentId: 'agent10_calculation', role: 'contributor' }
                ],
                targetDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000) // 30 days
            },
            {
                title: 'Comprehensive Quality Assurance Framework',
                description: 'Implement end-to-end quality monitoring and control system',
                participants: [
                    { agentId: 'agent5_qa_validation', role: 'contributor' },
                    { agentId: 'agent6_quality_control', role: 'leader' }
                ],
                targetDate: new Date(Date.now() + 45 * 24 * 60 * 60 * 1000) // 45 days
            },
            {
                title: 'Agent Lifecycle Management Excellence',
                description: 'Streamline agent deployment, management, and orchestration',
                participants: [
                    { agentId: 'agent7_agent_manager', role: 'contributor' },
                    { agentId: 'agent13_agent_builder', role: 'contributor' },
                    { agentId: 'agent15_orchestrator', role: 'leader' }
                ],
                targetDate: new Date(Date.now() + 60 * 24 * 60 * 60 * 1000) // 60 days
            }
        ];
        
        for (const config of collaborativeConfigs) {
            try {
                // Create collaborative goal
                const collaborativeGoal = await srv.run(INSERT.into('CollaborativeGoals').entries({
                    title: config.title,
                    description: config.description,
                    status: 'active',
                    overallProgress: 0,
                    coordinator_agentId: config.participants.find(p => p.role === 'leader').agentId,
                    targetDate: config.targetDate
                }));
                
                // Add participants
                for (const participant of config.participants) {
                    await srv.run(INSERT.into('CollaborativeParticipants').entries({
                        collaborativeGoal_ID: collaborativeGoal.ID,
                        agent_agentId: participant.agentId,
                        role: participant.role,
                        contribution: 0,
                        joinedAt: new Date(),
                        responsibilities: `${participant.role} responsibilities for ${config.title}`
                    }));
                }
                
                this.LOG.info(`Created collaborative goal: ${config.title}`);
                
            } catch (error) {
                this.LOG.error(`Failed to create collaborative goal: ${config.title}`, { error: error.message });
            }
        }
    }
}

// Export singleton instance
module.exports = new GoalAssignmentIntegration();