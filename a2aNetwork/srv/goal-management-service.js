const cds = require('@sap/cds');
const { SELECT, INSERT, UPDATE, DELETE, UPSERT } = cds.ql;

module.exports = (srv) => {
    srv.service.impl(function() {
        
        const { Agents, Goals } = this.entities;
        
        // A2A Backend Integration
        const A2A_BASE_URL = process.env.A2A_SERVICE_URL || 'http://localhost:8000';
        const axios = require('axios');
        
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
                // Failed to fetch goal analytics
                
                // Fallback to cached data
                const cached = await srv.run(SELECT.from('SystemAnalytics').orderBy('timestamp desc'));
                if (cached) {
                    return JSON.parse(cached.analyticsData);
                }
                
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
                // Failed to fetch goals for agent
                
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
                
                return agent || { agent_id: agentId, goals: [], status: 'unknown' };
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
                // Failed to fetch metrics for agent
                
                // Return cached metrics
                const cached = await srv.run(SELECT.from('AgentMetrics').where({ agent_agentId: agentId }).orderBy('timestamp desc').limit(10));
                
                return cached.reduce((acc, metric) => {
                    acc[metric.metricName] = metric.value;
                    return acc;
                }, {});
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
                // Failed to sync agent goals
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
                            console.warn(`Failed to send milestone notification: ${notificationError.message}`);
                        }
                    }
                }

                // Check for custom milestones based on specific metrics
                if (progressData.metrics) {
                    const metrics = JSON.parse(progressData.metrics);
                    await this._checkCustomMilestones(goalId, metrics, goal);
                }

            } catch (error) {
                console.error(`Failed to detect milestones for goal ${goalId}: ${error.message}`);
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
                console.error(`Failed to check custom milestones for goal ${goalId}: ${error.message}`);
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
                    .limit(10));

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
                    console.warn(`Failed to send AI predictions to network: ${networkError.message}`);
                }

            } catch (error) {
                console.error(`Failed to generate AI predictions for goal ${goal.ID}: ${error.message}`);
            }
        };

        /**
         * Calculate confidence level for predictions based on data quality
         */
        this._calculatePredictionConfidence = function(progressHistory) {
            if (progressHistory.length < 3) return 0.3; // Low confidence with few data points
            
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
            if (goalData.specific && goalData.specific.length < 10) {
                errors.push('Specific description must be at least 10 characters');
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
            const logMessage = `[${errorId}] ${operation} failed: ${error.message}`;
            
            // Log detailed error for debugging
            console.error(logMessage, {
                error: error.stack,
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

        // Set up real-time data refresh
        setInterval(async () => {
            try {
                // Refresh system analytics every 5 minutes
                await this.emit('read', 'GoalAnalytics', {});
            } catch (error) {
                // Failed to refresh analytics
            }
        }, 5 * 60 * 1000);

    });
};
