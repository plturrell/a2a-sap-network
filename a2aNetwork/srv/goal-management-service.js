const cds = require('@sap/cds');
const { SELECT, INSERT, UPDATE, DELETE, UPSERT } = cds.ql;
const goalSyncScheduler = require('./goal-sync-scheduler');

module.exports = (srv) => {
    srv.service.impl(function() {
        
        const { Agents, Goals } = this.entities;
        
        // Constants
        const A2A_BASE_URL = process.env.A2A_SERVICE_URL || 'http://localhost:8000';
        const PROGRESS_HISTORY_LIMIT = 10;
        const ANALYTICS_REFRESH_INTERVAL = 5 * 60 * 1000; // 5 minutes
        const MIN_SPECIFIC_LENGTH = 10;
        const PREDICTION_MIN_CONFIDENCE = 0.3;
        
        const axios = require('axios');
        const LOG = cds.log('goal-management');
        
        // Cleanup tracking
        let analyticsRefreshTimer;
        
        /**
         * Manual goal sync action
         */
        this.on('syncGoals', async (req) => {
            try {
                LOG.info('Manual goal sync requested');
                const result = await goalSyncScheduler.triggerManualSync();
                return {
                    status: 'success',
                    message: 'Goal synchronization completed',
                    result: result
                };
            } catch (error) {
                LOG.error('Manual goal sync failed', { error: error.message });
                throw new Error('Failed to synchronize goals');
            }
        });
        
        /**
         * Get sync status
         */
        this.on('getSyncStatus', async (req) => {
            const status = goalSyncScheduler.getStatus();
            return {
                ...status,
                serverTime: new Date()
            };
        });
        
        /**
         * Get visualization data for goal dashboard
         */
        this.on('READ', 'GoalVisualization', async (req) => {
            try {
                const visualizationType = req.data.type || 'overview';
                const agentFilter = req.data.agentId;
                const dateRange = req.data.dateRange || 30; // days
                
                let visualizationData = {};
                
                switch (visualizationType) {
                    case 'overview':
                        visualizationData = await this._getOverviewVisualization(agentFilter, dateRange);
                        break;
                    case 'progress_timeline':
                        visualizationData = await this._getProgressTimelineVisualization(agentFilter, dateRange);
                        break;
                    case 'agent_comparison':
                        visualizationData = await this._getAgentComparisonVisualization();
                        break;
                    case 'goal_heatmap':
                        visualizationData = await this._getGoalHeatmapVisualization();
                        break;
                    case 'dependency_graph':
                        visualizationData = await this._getDependencyGraphVisualization(agentFilter);
                        break;
                    case 'collaborative_goals':
                        visualizationData = await this._getCollaborativeGoalsVisualization();
                        break;
                    default:
                        throw new Error(`Unknown visualization type: ${visualizationType}`);
                }
                
                return visualizationData;
                
            } catch (error) {
                LOG.error('Failed to generate visualization data', { error: error.message });
                throw error;
            }
        });
        
        /**
         * Get comprehensive goal analytics
         */
        this.on('READ', 'GoalAnalytics', async (req) => {
            try {
                // Get system-wide analytics from A2A backend
                const response = await axios.get(`${A2A_BASE_URL}/api/v1/goals/analytics`, {
                    headers: {
                        'X-A2A-Service': 'goal-management-ui',
                        'Authorization': `Bearer ${req.user.token || 'system'}`
                    }
                });
                
                const analytics = response.data;
                
                // Store in CAP database for caching
                await srv.run(DELETE.from('SystemAnalytics'));
                await srv.run(INSERT.into('SystemAnalytics').entries({
                    timestamp: new Date(),
                    totalAgents: analytics.total_agents_with_goals || 0,
                    activeGoals: analytics.active_goals || 0,
                    completedGoals: analytics.completed_goals || 0,
                    averageProgress: analytics.average_progress || 0,
                    totalMilestones: analytics.total_milestones || 0,
                    agentsAbove50: analytics.agents_above_50_percent || 0,
                    systemHealth: analytics.system_health || 'good',
                    analyticsData: JSON.stringify(analytics)
                }));
                
                return analytics;
                
            } catch (error) {
                LOG.warn('Failed to fetch goal analytics from A2A network', { error: error.message });
                
                // Fallback to cached data
                const cached = await srv.run(SELECT.from('SystemAnalytics').orderBy('timestamp desc').limit(1));
                if (cached.length > 0 && cached[0].analyticsData) {
                    LOG.info('Using cached analytics data as fallback');
                    return JSON.parse(cached[0].analyticsData);
                }
                
                LOG.error('No cached analytics data available');
                throw new Error('Goal analytics service unavailable');
            }
        });
        
        /**
         * Get agent-specific goal data
         */
        this.on('READ', 'AgentGoals', async (req) => {
            const agentId = req.params[0];
            
            try {
                const response = await axios.get(`${A2A_BASE_URL}/api/v1/goals/agent/${agentId}`, {
                    headers: {
                        'X-A2A-Service': 'goal-management-ui',
                        'Authorization': `Bearer ${req.user.token || 'system'}`
                    }
                });
                
                const agentGoals = response.data;
                
                // Sync with CAP database
                await this._syncAgentGoals(agentId, agentGoals);
                
                return agentGoals;
                
            } catch (error) {
                LOG.warn('Failed to fetch goals from A2A network', { agentId, error: error.message });
                
                // Fallback to CAP database
                const agent = await srv.run(SELECT.one.from(Agents, a => {
                    a.agentId, a.agentName, a.status;
                    a.goals(g => {
                        g.ID, g.goalType, g.priority, g.status, g.overallProgress;
                        g.specific, g.targetDate;
                        g.measurable(m => { m.metricName, m.targetValue, m.currentValue, m.unit; });
                        g.progress(p => { p.timestamp, p.overallProgress, p.metrics; });
                        g.milestones(ms => { ms.title, ms.achievedDate, ms.significance; });
                    });
                }).where({ agentId }));
                
                if (agent) {
                    LOG.info('Using cached agent goals as fallback', { agentId });
                    return agent;
                } else {
                    LOG.warn('No cached goals found for agent', { agentId });
                    return { agent_id: agentId, goals: [], status: 'unknown' };
                }
            }
        });
        
        /**
         * Create new SMART goal
         */
        this.on('CREATE', 'Goals', async (req) => {
            const goalData = req.data;
            
            // Input validation
            const validationErrors = this._validateGoalData(goalData);
            if (validationErrors.length > 0) {
                req.error(400, `Goal validation failed: ${validationErrors.join(', ')}`);
                return;
            }
            
            try {
                // Send goal assignment to A2A backend
                const a2aPayload = {
                    operation: 'set_agent_goals',
                    data: {
                        agent_id: goalData.agent_agentId,
                        goals: {
                            primary_objectives: [{
                                goal_id: goalData.ID,
                                goal_type: goalData.goalType,
                                specific: goalData.specific,
                                measurable: goalData.measurable || {},
                                achievable: goalData.achievable,
                                relevant: goalData.relevant,
                                time_bound: goalData.timeBound,
                                priority: goalData.priority,
                                ai_enabled: goalData.aiEnabled
                            }]
                        }
                    }
                };
                
                const response = await axios.post(`${A2A_BASE_URL}/api/v1/goals/assign`, a2aPayload, {
                    headers: {
                        'Content-Type': 'application/json',
                        'X-A2A-Service': 'goal-management-ui',
                        'Authorization': `Bearer ${req.user.token || 'system'}`
                    }
                });
                
                if (response.data.status === 'success') {
                    // Goal successfully assigned via A2A
                    goalData.assignedVia = 'manual';
                    goalData.status = 'active';
                    goalData.startDate = new Date();
                    
                    // Create goal progress entry
                    await srv.run(INSERT.into('GoalProgress').entries({
                        goal_ID: goalData.ID,
                        timestamp: new Date(),
                        overallProgress: 0,
                        metrics: JSON.stringify({}),
                        reportedBy: 'system',
                        notes: 'Goal created and assigned via A2A network'
                    }));
                    
                    return req.reply(goalData);
                } else {
                    throw new Error('Failed to assign goal via A2A network');
                }
                
            } catch (error) {
                const errorInfo = this._handleServiceError(error, 'goal_creation', { goalData });
                req.error(500, errorInfo.message, errorInfo.errorId);
                return;
            }
        });
        
        /**
         * Update goal progress from A2A network
         */
        this.on('UPDATE', 'GoalProgress', async (req) => {
            const progressData = req.data;
            
            // Update goal overall progress
            await UPDATE(Goals, progressData.goal_ID).set({
                overallProgress: progressData.overallProgress,
                modifiedAt: new Date()
            });
            
            // Check for milestone achievements
            await this._detectMilestones(progressData.goal_ID, progressData);
            
            // Trigger AI predictions if enabled
            const goal = await SELECT.one.from(Goals).where({ ID: progressData.goal_ID });
            if (goal && goal.aiEnabled) {
                await this._generateAIPredictions(goal);
            }
            
            return req.reply(progressData);
        });
        
        /**
         * Get real-time agent metrics
         */
        this.on('READ', 'AgentMetrics', async (req) => {
            const agentId = req.params[0];
            
            try {
                const response = await axios.get(`${A2A_BASE_URL}/api/v1/agents/${agentId}/metrics`, {
                    headers: {
                        'X-A2A-Service': 'goal-management-ui',
                        'Authorization': `Bearer ${req.user.token || 'system'}`
                    }
                });
                
                const metrics = response.data;
                
                // Store metrics in CAP database
                const metricsEntries = Object.entries(metrics).map(([key, value]) => ({
                    agent_agentId: agentId,
                    timestamp: new Date(),
                    metricType: 'performance',
                    metricName: key,
                    value: parseFloat(value) || 0,
                    unit: this._getMetricUnit(key),
                    source: 'agent_sdk'
                }));
                
                await srv.run(INSERT.into('AgentMetrics').entries(metricsEntries));
                
                return metrics;
                
            } catch (error) {
                LOG.warn('Failed to fetch metrics from A2A network', { agentId, error: error.message });
                
                // Return cached metrics
                const cached = await srv.run(SELECT.from('AgentMetrics').where({ agent_agentId: agentId }).orderBy('timestamp desc').limit(PROGRESS_HISTORY_LIMIT));
                
                if (cached.length > 0) {
                    LOG.info('Using cached metrics as fallback', { agentId, metricsCount: cached.length });
                    return cached.reduce((acc, metric) => {
                        acc[metric.metricName] = metric.value;
                        return acc;
                    }, {});
                } else {
                    LOG.warn('No cached metrics found for agent', { agentId });
                    return {};
                }
            }
        });
        
        /**
         * Sync agent goals with CAP database
         */
        this._syncAgentGoals = async function(agentId, agentGoals) {
            try {
                // Upsert agent
                await srv.run(UPSERT.into('Agents').entries({
                    agentId: agentId,
                    agentName: agentGoals.agent_name || agentId,
                    status: agentGoals.status || 'active',
                    lastSeen: new Date()
                }));
                
                // Sync goals
                if (agentGoals.goals && agentGoals.goals.length > 0) {
                    for (const goal of agentGoals.goals) {
                        await srv.run(UPSERT.into('Goals').entries({
                            ID: goal.goal_id,
                            agent_agentId: agentId,
                            goalType: goal.goal_type,
                            priority: goal.priority,
                            status: goal.status || 'active',
                            specific: goal.specific,
                            achievable: goal.achievable,
                            relevant: goal.relevant,
                            timeBound: goal.time_bound,
                            overallProgress: agentGoals.overall_progress || 0,
                            targetDate: goal.target_date,
                            aiEnabled: goal.ai_enabled || false
                        }));
                        
                        // Sync measurable targets
                        if (goal.measurable) {
                            const measurableEntries = Object.entries(goal.measurable).map(([key, value]) => ({
                                goal_ID: goal.goal_id,
                                metricName: key,
                                targetValue: parseFloat(value) || 0,
                                currentValue: agentGoals.current_metrics?.[key] || 0,
                                unit: this._getMetricUnit(key),
                                progressPercent: this._calculateMetricProgress(key, value, agentGoals.current_metrics?.[key])
                            }));
                            
                            await srv.run(DELETE.from('a2a.goalmanagement.MeasurableTargets').where({ goal_ID: goal.goal_id }));
                            await srv.run(INSERT.into('a2a.goalmanagement.MeasurableTargets').entries(measurableEntries));
                        }
                    }
                }
                
            } catch (error) {
                LOG.error('Failed to sync agent goals', { agentId, error: error.message });
            }
        };

        /**
         * Helper functions
         */
        this._getMetricUnit = function(metricName) {
            const unitMap = {
                'success_rate': '%',
                'response_time': 'sec',
                'quality_score': '%',
                'throughput': '/hr',
                'uptime': '%',
                'error_rate': '%'
            };
            return unitMap[metricName] || '';
        };

        this._calculateMetricProgress = function(metricName, target, current) {
            if (!target || !current) return 0;

            // For metrics where lower is better (like response_time)
            if (metricName.includes('time') || metricName.includes('error')) {
                return Math.max(0, Math.min(100, ((target - current) / target) * 100));
            }

            // For metrics where higher is better
            return Math.max(0, Math.min(100, (current / target) * 100));
        };

        /**
         * Detect and record milestone achievements based on progress updates
         */
        this._detectMilestones = async function(goalId, progressData) {
            try {
                const goal = await SELECT.one.from(Goals).where({ ID: goalId });
                if (!goal) return;

                const currentProgress = progressData.overallProgress || 0;
                const previousProgress = goal.overallProgress || 0;

                // Define milestone thresholds
                const milestones = [
                    { threshold: 25, title: 'Quarter Progress', significance: 'low' },
                    { threshold: 50, title: 'Halfway Point', significance: 'medium' },
                    { threshold: 75, title: 'Three-Quarter Progress', significance: 'medium' },
                    { threshold: 90, title: 'Near Completion', significance: 'high' },
                    { threshold: 100, title: 'Goal Completed', significance: 'critical' }
                ];

                // Check for crossed milestones
                for (const milestone of milestones) {
                    if (currentProgress >= milestone.threshold && previousProgress < milestone.threshold) {
                        // Milestone achieved
                        await srv.run(INSERT.into('GoalMilestones').entries({
                            goal_ID: goalId,
                            title: milestone.title,
                            description: `Goal reached ${milestone.threshold}% completion`,
                            achievedDate: new Date(),
                            significance: milestone.significance,
                            progressAtAchievement: currentProgress,
                            autoDetected: true,
                            metadata: JSON.stringify({
                                threshold: milestone.threshold,
                                previousProgress: previousProgress,
                                currentProgress: currentProgress,
                                detectedAt: new Date().toISOString()
                            })
                        }));

                        // Send milestone notification to A2A network
                        try {
                            await axios.post(`${A2A_BASE_URL}/api/v1/goals/milestone`, {
                                goal_id: goalId,
                                agent_id: goal.agent_agentId,
                                milestone: {
                                    title: milestone.title,
                                    threshold: milestone.threshold,
                                    significance: milestone.significance,
                                    achieved_at: new Date().toISOString()
                                }
                            }, {
                                headers: {
                                    'Content-Type': 'application/json',
                                    'X-A2A-Service': 'goal-management-ui'
                                }
                            });
                        } catch (notificationError) {
                            // Failed to send milestone notification - continue processing
                            LOG.warn('Failed to send milestone notification', { goalId, error: notificationError.message });
                        }
                    }
                }

                // Check for custom milestones based on specific metrics
                if (progressData.metrics) {
                    const metrics = JSON.parse(progressData.metrics);
                    await this._checkCustomMilestones(goalId, metrics, goal);
                }

            } catch (error) {
                LOG.error('Failed to detect milestones', { goalId, error: error.message });
            }
        };

        /**
         * Check for custom milestones based on specific metrics
         */
        this._checkCustomMilestones = async function(goalId, metrics, goal) {
            try {
                // Define custom milestone patterns for different goal types
                const customMilestones = {
                    'performance_improvement': [
                        { metricName: 'success_rate', threshold: 95, title: 'Excellence Achieved' },
                        { metricName: 'response_time', threshold: 0.5, title: 'Speed Optimized', isLowerBetter: true }
                    ],
                    'quality_enhancement': [
                        { metricName: 'quality_score', threshold: 90, title: 'Quality Standard Met' },
                        { metricName: 'error_rate', threshold: 1, title: 'Error Rate Minimized', isLowerBetter: true }
                    ],
                    'throughput_increase': [
                        { metricName: 'throughput', threshold: 1000, title: 'Throughput Target Reached' }
                    ]
                };

                const goalTypeMilestones = customMilestones[goal.goalType] || [];

                for (const milestone of goalTypeMilestones) {
                    const metricValue = metrics[milestone.metricName];
                    if (metricValue !== undefined) {
                        const thresholdMet = milestone.isLowerBetter 
                            ? metricValue <= milestone.threshold
                            : metricValue >= milestone.threshold;

                        if (thresholdMet) {
                            // Check if this custom milestone was already recorded
                            const existing = await srv.run(SELECT.one.from('GoalMilestones').where({
                                goal_ID: goalId,
                                title: milestone.title
                            }));

                            if (!existing) {
                                await srv.run(INSERT.into('GoalMilestones').entries({
                                    goal_ID: goalId,
                                    title: milestone.title,
                                    description: `Custom milestone: ${milestone.metricName} ${milestone.isLowerBetter ? 'reduced to' : 'reached'} ${milestone.threshold}`,
                                    achievedDate: new Date(),
                                    significance: 'high',
                                    progressAtAchievement: goal.overallProgress || 0,
                                    autoDetected: true,
                                    metadata: JSON.stringify({
                                        metricName: milestone.metricName,
                                        metricValue: metricValue,
                                        threshold: milestone.threshold,
                                        isCustom: true
                                    })
                                }));
                            }
                        }
                    }
                }

            } catch (error) {
                LOG.error('Failed to check custom milestones', { goalId, error: error.message });
            }
        };

        /**
         * Generate AI predictions for goal completion and recommendations
         */
        this._generateAIPredictions = async function(goal) {
            try {
                // Get historical progress data
                const progressHistory = await srv.run(SELECT.from('GoalProgress')
                    .where({ goal_ID: goal.ID })
                    .orderBy('timestamp desc')
                    .limit(PROGRESS_HISTORY_LIMIT));

                if (progressHistory.length < 2) {
                    return; // Need at least 2 data points for predictions
                }

                // Calculate progress velocity (progress per day)
                const latestProgress = progressHistory[0];
                const previousProgress = progressHistory[1];
                const timeDiff = (new Date(latestProgress.timestamp) - new Date(previousProgress.timestamp)) / (1000 * 60 * 60 * 24);
                const progressDiff = latestProgress.overallProgress - previousProgress.overallProgress;
                const velocity = timeDiff > 0 ? progressDiff / timeDiff : 0;

                // Predict completion date
                const remainingProgress = 100 - latestProgress.overallProgress;
                const predictedDaysToComplete = velocity > 0 ? Math.ceil(remainingProgress / velocity) : null;
                const predictedCompletionDate = predictedDaysToComplete 
                    ? new Date(Date.now() + predictedDaysToComplete * 24 * 60 * 60 * 1000)
                    : null;

                // Generate risk assessment
                const riskFactors = [];
                if (velocity <= 0) riskFactors.push('No progress momentum');
                if (goal.targetDate && predictedCompletionDate && predictedCompletionDate > new Date(goal.targetDate)) {
                    riskFactors.push('Predicted to miss target date');
                }
                if (latestProgress.overallProgress < 25 && new Date() > new Date(goal.startDate || goal.createdAt)) {
                    const daysSinceStart = (new Date() - new Date(goal.startDate || goal.createdAt)) / (1000 * 60 * 60 * 24);
                    if (daysSinceStart > 7) riskFactors.push('Slow initial progress');
                }

                const riskLevel = riskFactors.length === 0 ? 'low' : riskFactors.length === 1 ? 'medium' : 'high';

                // Generate recommendations
                const recommendations = [];
                if (velocity <= 0) {
                    recommendations.push('Consider breaking down goal into smaller, actionable tasks');
                    recommendations.push('Review goal requirements and remove blockers');
                }
                if (riskLevel === 'high') {
                    recommendations.push('Schedule immediate goal review session');
                    recommendations.push('Consider adjusting timeline or scope');
                }
                if (latestProgress.overallProgress > 75) {
                    recommendations.push('Focus on final quality checks and completion');
                }

                // Store AI predictions
                await srv.run(INSERT.into('GoalPredictions').entries({
                    goal_ID: goal.ID,
                    predictionDate: new Date(),
                    predictedCompletionDate: predictedCompletionDate,
                    progressVelocity: velocity,
                    riskLevel: riskLevel,
                    riskFactors: JSON.stringify(riskFactors),
                    recommendations: JSON.stringify(recommendations),
                    confidence: this._calculatePredictionConfidence(progressHistory),
                    metadata: JSON.stringify({
                        dataPoints: progressHistory.length,
                        currentProgress: latestProgress.overallProgress,
                        daysToTarget: goal.targetDate ? Math.ceil((new Date(goal.targetDate) - new Date()) / (1000 * 60 * 60 * 24)) : null
                    })
                }));

                // Send predictions to A2A network for agent optimization
                try {
                    await axios.post(`${A2A_BASE_URL}/api/v1/goals/predictions`, {
                        goal_id: goal.ID,
                        agent_id: goal.agent_agentId,
                        predictions: {
                            completion_date: predictedCompletionDate,
                            velocity: velocity,
                            risk_level: riskLevel,
                            recommendations: recommendations
                        }
                    }, {
                        headers: {
                            'Content-Type': 'application/json',
                            'X-A2A-Service': 'goal-management-ui'
                        }
                    });
                } catch (networkError) {
                    // Failed to send predictions - continue processing
                    LOG.warn('Failed to send AI predictions to network', { goalId: goal.ID, error: networkError.message });
                }

            } catch (error) {
                LOG.error('Failed to generate AI predictions', { goalId: goal.ID, error: error.message });
            }
        };

        /**
         * Calculate confidence level for predictions based on data quality
         */
        this._calculatePredictionConfidence = function(progressHistory) {
            if (progressHistory.length < 3) return PREDICTION_MIN_CONFIDENCE; // Low confidence with few data points
            
            // Check for consistent progress pattern
            let consistentProgress = 0;
            for (let i = 1; i < progressHistory.length; i++) {
                if (progressHistory[i-1].overallProgress >= progressHistory[i].overallProgress) {
                    consistentProgress++;
                }
            }
            
            const consistencyRatio = consistentProgress / (progressHistory.length - 1);
            
            // More data points and consistent progress = higher confidence
            const dataConfidence = Math.min(1.0, progressHistory.length / 10);
            const patternConfidence = consistencyRatio;
            
            return Math.round((dataConfidence * 0.4 + patternConfidence * 0.6) * 100) / 100;
        };

        /**
         * Validate goal data for creation and updates
         */
        this._validateGoalData = function(goalData) {
            const errors = [];

            // Required fields validation
            if (!goalData.agent_agentId) {
                errors.push('Agent ID is required');
            }
            if (!goalData.goalType) {
                errors.push('Goal type is required');
            }
            if (!goalData.specific) {
                errors.push('Specific goal description is required');
            }
            if (!goalData.priority || !['low', 'medium', 'high', 'critical'].includes(goalData.priority)) {
                errors.push('Valid priority (low, medium, high, critical) is required');
            }

            // SMART criteria validation
            if (goalData.specific && goalData.specific.length < MIN_SPECIFIC_LENGTH) {
                errors.push(`Specific description must be at least ${MIN_SPECIFIC_LENGTH} characters`);
            }
            if (goalData.achievable && typeof goalData.achievable !== 'string') {
                errors.push('Achievable criteria must be a string description');
            }
            if (goalData.relevant && typeof goalData.relevant !== 'string') {
                errors.push('Relevant criteria must be a string description');
            }
            if (goalData.timeBound && typeof goalData.timeBound !== 'string') {
                errors.push('Time-bound criteria must be a string description');
            }

            // Date validation
            if (goalData.targetDate) {
                const targetDate = new Date(goalData.targetDate);
                if (isNaN(targetDate.getTime())) {
                    errors.push('Target date must be a valid date');
                } else if (targetDate <= new Date()) {
                    errors.push('Target date must be in the future');
                }
            }

            // Progress validation
            if (goalData.overallProgress !== undefined) {
                const progress = parseFloat(goalData.overallProgress);
                if (isNaN(progress) || progress < 0 || progress > 100) {
                    errors.push('Overall progress must be a number between 0 and 100');
                }
            }

            return errors;
        };

        /**
         * Enhanced error handling with logging and user-friendly messages
         */
        this._handleServiceError = function(error, operation, context = {}) {
            const errorId = Date.now().toString(36) + Math.random().toString(36).substr(2);
            
            // Log detailed error for debugging
            LOG.error(`${operation} failed`, {
                errorId: errorId,
                error: error.message,
                stack: error.stack,
                context: context,
                timestamp: new Date().toISOString()
            });

            // Return user-friendly error message
            const userMessage = this._getUserFriendlyErrorMessage(error, operation);
            return {
                errorId: errorId,
                message: userMessage,
                operation: operation,
                timestamp: new Date().toISOString()
            };
        };

        /**
         * Convert technical errors to user-friendly messages
         */
        this._getUserFriendlyErrorMessage = function(error, operation) {
            const errorMappings = {
                'ECONNREFUSED': 'Unable to connect to A2A network. Please try again later.',
                'ETIMEDOUT': 'Request timed out. Please check your connection and try again.',
                'ENOTFOUND': 'A2A network service is currently unavailable.',
                'ValidationError': 'The provided data is invalid. Please check your input.',
                'AuthenticationError': 'Authentication failed. Please log in again.',
                'PermissionError': 'You do not have permission to perform this action.'
            };

            // Check for specific error types
            for (const [errorType, message] of Object.entries(errorMappings)) {
                if (error.message.includes(errorType) || error.code === errorType) {
                    return message;
                }
            }

            // Default messages by operation
            const operationDefaults = {
                'goal_creation': 'Failed to create goal. Please verify your input and try again.',
                'goal_update': 'Failed to update goal progress. Please try again.',
                'analytics_fetch': 'Unable to load analytics data. Using cached information.',
                'metrics_fetch': 'Unable to load current metrics. Please refresh the page.'
            };

            return operationDefaults[operation] || 'An unexpected error occurred. Please try again or contact support.';
        };

        // Set up real-time data refresh with cleanup
        analyticsRefreshTimer = setInterval(async () => {
            try {
                // Refresh system analytics every 5 minutes
                await this.emit('read', 'GoalAnalytics', {});
            } catch (error) {
                LOG.warn('Failed to refresh analytics automatically', { error: error.message });
            }
        }, ANALYTICS_REFRESH_INTERVAL);

        /**
         * Visualization helper functions
         */
        this._getOverviewVisualization = async function(agentFilter, dateRange) {
            const endDate = new Date();
            const startDate = new Date(endDate.getTime() - dateRange * 24 * 60 * 60 * 1000);
            
            // Get goals with progress history
            let query = SELECT.from(Goals, g => {
                g.ID, g.goalType, g.priority, g.status, g.overallProgress;
                g.specific, g.targetDate, g.agent_agentId;
                g.progress(p => { p.timestamp, p.overallProgress; }).where({ timestamp: { '>=': startDate } });
            });
            
            if (agentFilter) {
                query = query.where({ agent_agentId: agentFilter });
            }
            
            const goals = await srv.run(query);
            
            // Process data for visualization
            return {
                type: 'overview',
                summary: {
                    totalGoals: goals.length,
                    activeGoals: goals.filter(g => g.status === 'active').length,
                    completedGoals: goals.filter(g => g.status === 'completed').length,
                    averageProgress: goals.reduce((sum, g) => sum + (g.overallProgress || 0), 0) / (goals.length || 1),
                    byPriority: this._groupByProperty(goals, 'priority'),
                    byType: this._groupByProperty(goals, 'goalType'),
                    byStatus: this._groupByProperty(goals, 'status')
                },
                charts: {
                    progressDistribution: this._getProgressDistribution(goals),
                    priorityRadar: this._getPriorityRadarData(goals),
                    typeBreakdown: this._getTypeBreakdownData(goals)
                },
                recentActivity: await this._getRecentActivity(agentFilter, 10)
            };
        };
        
        this._getProgressTimelineVisualization = async function(agentFilter, dateRange) {
            const endDate = new Date();
            const startDate = new Date(endDate.getTime() - dateRange * 24 * 60 * 60 * 1000);
            
            // Get all progress updates in date range
            const progressQuery = SELECT.from('GoalProgress', p => {
                p.timestamp, p.overallProgress, p.goal_ID;
                p.goal(g => { g.goalType, g.priority, g.agent_agentId; });
            }).where({ timestamp: { between: [startDate, endDate] } }).orderBy('timestamp');
            
            const progressData = await srv.run(progressQuery);
            
            // Group by day for timeline
            const timelineData = {};
            progressData.forEach(p => {
                if (!agentFilter || p.goal.agent_agentId === agentFilter) {
                    const dateKey = new Date(p.timestamp).toISOString().split('T')[0];
                    if (!timelineData[dateKey]) {
                        timelineData[dateKey] = {
                            date: dateKey,
                            updates: 0,
                            averageProgress: 0,
                            goals: new Set()
                        };
                    }
                    timelineData[dateKey].updates++;
                    timelineData[dateKey].averageProgress += p.overallProgress;
                    timelineData[dateKey].goals.add(p.goal_ID);
                }
            });
            
            // Convert to array and calculate averages
            const timeline = Object.values(timelineData).map(day => ({
                date: day.date,
                updates: day.updates,
                averageProgress: day.averageProgress / day.updates,
                uniqueGoals: day.goals.size
            }));
            
            return {
                type: 'progress_timeline',
                timeline: timeline,
                summary: {
                    totalUpdates: progressData.length,
                    averageUpdatesPerDay: progressData.length / dateRange,
                    daysWithActivity: timeline.length
                }
            };
        };
        
        this._getAgentComparisonVisualization = async function() {
            // Get all agents with their goals
            const agents = await srv.run(SELECT.from(Agents, a => {
                a.agentId, a.agentName, a.status;
                a.goals(g => { g.ID, g.status, g.overallProgress, g.priority; });
            }));
            
            const comparisonData = agents.map(agent => {
                const goals = agent.goals || [];
                return {
                    agentId: agent.agentId,
                    agentName: agent.agentName,
                    status: agent.status,
                    metrics: {
                        totalGoals: goals.length,
                        activeGoals: goals.filter(g => g.status === 'active').length,
                        completedGoals: goals.filter(g => g.status === 'completed').length,
                        averageProgress: goals.length > 0 
                            ? goals.reduce((sum, g) => sum + (g.overallProgress || 0), 0) / goals.length 
                            : 0,
                        highPriorityGoals: goals.filter(g => g.priority === 'high' || g.priority === 'critical').length
                    }
                };
            }).sort((a, b) => b.metrics.averageProgress - a.metrics.averageProgress);
            
            return {
                type: 'agent_comparison',
                agents: comparisonData,
                rankings: {
                    byProgress: comparisonData.slice(0, 10),
                    byActiveGoals: [...comparisonData].sort((a, b) => b.metrics.activeGoals - a.metrics.activeGoals).slice(0, 10),
                    byCompletionRate: [...comparisonData]
                        .filter(a => a.metrics.totalGoals > 0)
                        .map(a => ({
                            ...a,
                            completionRate: (a.metrics.completedGoals / a.metrics.totalGoals) * 100
                        }))
                        .sort((a, b) => b.completionRate - a.completionRate)
                        .slice(0, 10)
                }
            };
        };
        
        this._getGoalHeatmapVisualization = async function() {
            // Get all goals with their progress and dates
            const goals = await srv.run(SELECT.from(Goals, g => {
                g.ID, g.goalType, g.priority, g.status, g.overallProgress;
                g.createdAt, g.modifiedAt, g.targetDate;
                g.agent(a => { a.agentId, a.agentName; });
            }));
            
            // Create heatmap data by week and goal type
            const heatmapData = {};
            const goalTypes = [...new Set(goals.map(g => g.goalType))];
            
            goals.forEach(goal => {
                const weekKey = this._getWeekKey(new Date(goal.modifiedAt));
                if (!heatmapData[weekKey]) {
                    heatmapData[weekKey] = {};
                    goalTypes.forEach(type => {
                        heatmapData[weekKey][type] = { count: 0, totalProgress: 0 };
                    });
                }
                
                heatmapData[weekKey][goal.goalType].count++;
                heatmapData[weekKey][goal.goalType].totalProgress += goal.overallProgress || 0;
            });
            
            // Convert to visualization format
            const heatmap = Object.entries(heatmapData).map(([week, typeData]) => {
                const weekData = { week };
                Object.entries(typeData).forEach(([type, data]) => {
                    weekData[type] = {
                        activity: data.count,
                        averageProgress: data.count > 0 ? data.totalProgress / data.count : 0
                    };
                });
                return weekData;
            }).sort((a, b) => a.week.localeCompare(b.week));
            
            return {
                type: 'goal_heatmap',
                heatmap: heatmap,
                goalTypes: goalTypes,
                intensityScale: {
                    min: 0,
                    max: Math.max(...Object.values(heatmapData).map(week => 
                        Math.max(...Object.values(week).map(type => type.count))
                    ))
                }
            };
        };
        
        this._getDependencyGraphVisualization = async function(agentFilter) {
            // Get goals with dependencies
            let query = SELECT.from(Goals, g => {
                g.ID, g.specific, g.status, g.overallProgress, g.agent_agentId;
                g.dependencies(d => { d.dependsOnGoal_ID, d.dependencyType, d.isBlocking; });
            });
            
            if (agentFilter) {
                query = query.where({ agent_agentId: agentFilter });
            }
            
            const goals = await srv.run(query);
            
            // Build graph nodes and edges
            const nodes = goals.map(g => ({
                id: g.ID,
                label: g.specific.substring(0, 50) + (g.specific.length > 50 ? '...' : ''),
                status: g.status,
                progress: g.overallProgress,
                agentId: g.agent_agentId,
                type: 'goal'
            }));
            
            const edges = [];
            goals.forEach(g => {
                (g.dependencies || []).forEach(dep => {
                    edges.push({
                        source: g.ID,
                        target: dep.dependsOnGoal_ID,
                        type: dep.dependencyType || 'depends_on',
                        isBlocking: dep.isBlocking || false
                    });
                });
            });
            
            // Identify critical paths
            const criticalPaths = this._findCriticalPaths(nodes, edges);
            
            return {
                type: 'dependency_graph',
                graph: {
                    nodes: nodes,
                    edges: edges
                },
                analysis: {
                    totalDependencies: edges.length,
                    blockingDependencies: edges.filter(e => e.isBlocking).length,
                    criticalPaths: criticalPaths,
                    isolatedGoals: nodes.filter(n => 
                        !edges.some(e => e.source === n.id || e.target === n.id)
                    ).length
                }
            };
        };
        
        this._getCollaborativeGoalsVisualization = async function() {
            // Get collaborative goals (goals shared between multiple agents)
            const collaborativeGoals = await srv.run(SELECT.from('CollaborativeGoals', c => {
                c.ID, c.title, c.description, c.status, c.overallProgress;
                c.participants(p => { 
                    p.agent_agentId, p.role, p.contribution;
                    p.agent(a => { a.agentName; });
                });
                c.milestones(m => { m.title, m.achievedDate, m.significance; });
            }));
            
            // Process collaborative network
            const agentConnections = {};
            collaborativeGoals.forEach(goal => {
                const participants = goal.participants || [];
                participants.forEach((p1, i) => {
                    participants.slice(i + 1).forEach(p2 => {
                        const key = [p1.agent_agentId, p2.agent_agentId].sort().join('-');
                        if (!agentConnections[key]) {
                            agentConnections[key] = {
                                agents: [p1.agent_agentId, p2.agent_agentId],
                                collaborations: 0,
                                totalProgress: 0
                            };
                        }
                        agentConnections[key].collaborations++;
                        agentConnections[key].totalProgress += goal.overallProgress || 0;
                    });
                });
            });
            
            // Convert to network format
            const collaborationNetwork = {
                nodes: [],
                links: Object.values(agentConnections).map(conn => ({
                    source: conn.agents[0],
                    target: conn.agents[1],
                    weight: conn.collaborations,
                    averageProgress: conn.totalProgress / conn.collaborations
                }))
            };
            
            // Get unique agents for nodes
            const uniqueAgents = new Set();
            Object.values(agentConnections).forEach(conn => {
                conn.agents.forEach(agent => uniqueAgents.add(agent));
            });
            
            collaborationNetwork.nodes = Array.from(uniqueAgents).map(agentId => ({
                id: agentId,
                collaborations: Object.values(agentConnections)
                    .filter(conn => conn.agents.includes(agentId)).length
            }));
            
            return {
                type: 'collaborative_goals',
                goals: collaborativeGoals.map(g => ({
                    id: g.ID,
                    title: g.title,
                    status: g.status,
                    progress: g.overallProgress,
                    participantCount: (g.participants || []).length,
                    participants: (g.participants || []).map(p => ({
                        agentId: p.agent_agentId,
                        agentName: p.agent?.agentName || p.agent_agentId,
                        role: p.role,
                        contribution: p.contribution
                    })),
                    recentMilestones: (g.milestones || [])
                        .filter(m => m.achievedDate)
                        .sort((a, b) => new Date(b.achievedDate) - new Date(a.achievedDate))
                        .slice(0, 3)
                })),
                network: collaborationNetwork,
                summary: {
                    totalCollaborativeGoals: collaborativeGoals.length,
                    activeCollaborations: collaborativeGoals.filter(g => g.status === 'active').length,
                    averageParticipants: collaborativeGoals.reduce((sum, g) => 
                        sum + (g.participants || []).length, 0) / (collaborativeGoals.length || 1),
                    topCollaborators: this._getTopCollaborators(collaborativeGoals)
                }
            };
        };
        
        // Helper functions for visualizations
        this._groupByProperty = function(items, property) {
            return items.reduce((groups, item) => {
                const key = item[property] || 'unknown';
                groups[key] = (groups[key] || 0) + 1;
                return groups;
            }, {});
        };
        
        this._getProgressDistribution = function(goals) {
            const bins = [0, 25, 50, 75, 100];
            const distribution = {};
            
            bins.forEach((bin, i) => {
                const nextBin = bins[i + 1] || 101;
                const label = i === bins.length - 1 ? `${bin}%` : `${bin}-${nextBin - 1}%`;
                distribution[label] = goals.filter(g => 
                    g.overallProgress >= bin && g.overallProgress < nextBin
                ).length;
            });
            
            return distribution;
        };
        
        this._getPriorityRadarData = function(goals) {
            const priorities = ['low', 'medium', 'high', 'critical'];
            const metrics = ['count', 'averageProgress', 'completionRate'];
            
            const radarData = priorities.map(priority => {
                const priorityGoals = goals.filter(g => g.priority === priority);
                return {
                    priority: priority,
                    count: priorityGoals.length,
                    averageProgress: priorityGoals.length > 0
                        ? priorityGoals.reduce((sum, g) => sum + (g.overallProgress || 0), 0) / priorityGoals.length
                        : 0,
                    completionRate: priorityGoals.length > 0
                        ? (priorityGoals.filter(g => g.status === 'completed').length / priorityGoals.length) * 100
                        : 0
                };
            });
            
            return radarData;
        };
        
        this._getTypeBreakdownData = function(goals) {
            const typeGroups = {};
            
            goals.forEach(goal => {
                const type = goal.goalType || 'unknown';
                if (!typeGroups[type]) {
                    typeGroups[type] = {
                        type: type,
                        count: 0,
                        totalProgress: 0,
                        statuses: {}
                    };
                }
                
                typeGroups[type].count++;
                typeGroups[type].totalProgress += goal.overallProgress || 0;
                typeGroups[type].statuses[goal.status] = (typeGroups[type].statuses[goal.status] || 0) + 1;
            });
            
            return Object.values(typeGroups).map(group => ({
                type: group.type,
                count: group.count,
                averageProgress: group.count > 0 ? group.totalProgress / group.count : 0,
                statuses: group.statuses
            }));
        };
        
        this._getRecentActivity = async function(agentFilter, limit = 10) {
            let query = SELECT.from('GoalActivity', a => {
                a.timestamp, a.activityType, a.description, a.goal_ID, a.agent_agentId;
                a.goal(g => { g.specific; });
            }).orderBy('timestamp desc').limit(limit);
            
            if (agentFilter) {
                query = query.where({ agent_agentId: agentFilter });
            }
            
            const activities = await srv.run(query);
            
            return activities.map(a => ({
                timestamp: a.timestamp,
                type: a.activityType,
                description: a.description,
                goalId: a.goal_ID,
                goalTitle: a.goal?.specific || 'Unknown Goal',
                agentId: a.agent_agentId
            }));
        };
        
        this._getWeekKey = function(date) {
            const year = date.getFullYear();
            const firstDayOfYear = new Date(year, 0, 1);
            const pastDaysOfYear = (date - firstDayOfYear) / 86400000;
            const weekNumber = Math.ceil((pastDaysOfYear + firstDayOfYear.getDay() + 1) / 7);
            return `${year}-W${weekNumber.toString().padStart(2, '0')}`;
        };
        
        this._findCriticalPaths = function(nodes, edges) {
            // Simple critical path detection - find longest dependency chains
            const paths = [];
            const visited = new Set();
            
            function findPath(nodeId, currentPath = []) {
                if (visited.has(nodeId)) return;
                visited.add(nodeId);
                currentPath.push(nodeId);
                
                const dependencies = edges.filter(e => e.source === nodeId);
                if (dependencies.length === 0) {
                    paths.push([...currentPath]);
                } else {
                    dependencies.forEach(dep => {
                        findPath(dep.target, currentPath);
                    });
                }
                
                currentPath.pop();
                visited.delete(nodeId);
            }
            
            // Start from nodes with no incoming edges
            const rootNodes = nodes.filter(n => 
                !edges.some(e => e.target === n.id)
            );
            
            rootNodes.forEach(root => findPath(root.id));
            
            // Return top 5 longest paths
            return paths
                .sort((a, b) => b.length - a.length)
                .slice(0, 5)
                .map(path => ({
                    path: path,
                    length: path.length,
                    isBlocked: edges.some(e => 
                        path.includes(e.source) && path.includes(e.target) && e.isBlocking
                    )
                }));
        };
        
        this._getTopCollaborators = function(collaborativeGoals) {
            const agentStats = {};
            
            collaborativeGoals.forEach(goal => {
                (goal.participants || []).forEach(p => {
                    if (!agentStats[p.agent_agentId]) {
                        agentStats[p.agent_agentId] = {
                            agentId: p.agent_agentId,
                            agentName: p.agent?.agentName || p.agent_agentId,
                            collaborations: 0,
                            roles: new Set(),
                            totalContribution: 0
                        };
                    }
                    
                    agentStats[p.agent_agentId].collaborations++;
                    agentStats[p.agent_agentId].roles.add(p.role);
                    agentStats[p.agent_agentId].totalContribution += p.contribution || 0;
                });
            });
            
            return Object.values(agentStats)
                .map(stats => ({
                    ...stats,
                    roles: Array.from(stats.roles),
                    averageContribution: stats.totalContribution / stats.collaborations
                }))
                .sort((a, b) => b.collaborations - a.collaborations)
                .slice(0, 10);
        };
        
        // Service cleanup handler
        this.on('shutdown', () => {
            if (analyticsRefreshTimer) {
                clearInterval(analyticsRefreshTimer);
                LOG.info('Analytics refresh timer cleared on service shutdown');
            }
        });

    });
};
