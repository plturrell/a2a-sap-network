/**
 * @fileoverview Analytics Service Implementation - Pure SAP CAP
 * @module analyticsService
 * 
 * Provides analytics functionality using SAP CAP without external analytics services
 */

const cds = require('@sap/cds');

// Import CAP query builders
const { SELECT, INSERT, UPDATE, DELETE } = cds.ql;

/**
 * Analytics Service Implementation
 */
module.exports = class AnalyticsService extends cds.ApplicationService {
    
    async init() {
        const { TemplatePerformance, AgentUtilization, ExecutionTrends, 
                QualityMetrics, CostAnalysis, UserAdoption } = this.entities;
        
        // Performance analytics function
        this.on('getPerformanceByDateRange', async (req) => {
            const { startDate, endDate, templateId } = req.data;
            
            try {
                let query = SELECT.from('WorkflowTemplateService.ExecutionHistory')
                    .columns([
                        'DATE(startTime) as date',
                        'COUNT(*) as executionCount',
                        'CAST(SUM(CASE WHEN status = "completed" THEN 1 ELSE 0 END) AS DECIMAL) / CAST(COUNT(*) AS DECIMAL) * 100 as successRate',
                        'AVG(duration) as avgDuration',
                        'CAST(SUM(CASE WHEN status = "failed" THEN 1 ELSE 0 END) AS DECIMAL) / CAST(COUNT(*) AS DECIMAL) * 100 as errorRate'
                    ])
                    .where('startTime >= ', startDate)
                    .where('startTime <= ', endDate);
                
                if (templateId) {
                    // Join with WorkflowInstances to filter by template
                    query = SELECT.from('WorkflowTemplateService.ExecutionHistory')
                        .join('WorkflowTemplateService.Instances as instance', 'instance_ID = instance.ID')
                        .columns([
                            'DATE(startTime) as date',
                            'COUNT(*) as executionCount',
                            'CAST(SUM(CASE WHEN status = "completed" THEN 1 ELSE 0 END) AS DECIMAL) / CAST(COUNT(*) AS DECIMAL) * 100 as successRate',
                            'AVG(duration) as avgDuration',
                            'CAST(SUM(CASE WHEN status = "failed" THEN 1 ELSE 0 END) AS DECIMAL) / CAST(COUNT(*) AS DECIMAL) * 100 as errorRate'
                        ])
                        .where('startTime >= ', startDate)
                        .where('startTime <= ', endDate)
                        .where('instance.template_ID = ', templateId)
                        .groupBy('DATE(startTime)')
                        .orderBy('DATE(startTime)');
                }
                
                query = query.groupBy('DATE(startTime)')
                    .orderBy('DATE(startTime)');
                
                const results = await cds.run(query);
                
                return results.map(row => ({
                    date: row.date,
                    executionCount: row.executionCount || 0,
                    successRate: parseFloat(row.successRate) || 0,
                    avgDuration: parseFloat(row.avgDuration) || 0,
                    errorRate: parseFloat(row.errorRate) || 0
                }));
                
            } catch (error) {
                console.error('Error getting performance data:', error);
                return [];
            }
        });
        
        // Agent utilization heatmap function
        this.on('getAgentUtilizationHeatmap', async (req) => {
            const { dateRange } = req.data;
            
            try {
                // Determine date range
                let dateFilter = new Date();
                switch (dateRange) {
                    case 'week':
                        dateFilter.setDate(dateFilter.getDate() - 7);
                        break;
                    case 'month':
                        dateFilter.setMonth(dateFilter.getMonth() - 1);
                        break;
                    case 'quarter':
                        dateFilter.setMonth(dateFilter.getMonth() - 3);
                        break;
                    default:
                        dateFilter.setDate(dateFilter.getDate() - 7);
                }
                
                // Get agent metadata (mock for now)
                const agents = [
                    {id: 0, name: 'Document Extraction Agent'},
                    {id: 1, name: 'Data Standardization Agent'},
                    {id: 2, name: 'AI Preparation Agent'},
                    {id: 3, name: 'Vector Processing Agent'}
                ];
                
                // Get utilization data
                const utilizationData = await SELECT.from('WorkflowTemplateService.ExecutionHistory')
                    .columns([
                        'DATE(startTime) as date',
                        'JSON_VALUE(agentMetrics, "$.agentId") as agentId',
                        'COUNT(*) as taskCount',
                        'AVG(CAST(JSON_VALUE(agentMetrics, "$.responseTime") AS DECIMAL)) as avgResponseTime'
                    ])
                    .where('startTime >= ', dateFilter.toISOString())
                    .where('agentMetrics IS NOT NULL')
                    .groupBy('DATE(startTime)', 'JSON_VALUE(agentMetrics, "$.agentId")');
                
                // Organize data by agent
                const result = agents.map(agent => {
                    const agentData = utilizationData.filter(d => 
                        parseInt(d.agentId) === agent.id
                    );
                    
                    return {
                        agentId: agent.id,
                        agentName: agent.name,
                        utilizationData: agentData.map(d => ({
                            date: d.date,
                            utilizationPercent: Math.min(100, (d.taskCount / 10) * 100), // Normalized to 10 max tasks
                            taskCount: d.taskCount || 0
                        }))
                    };
                });
                
                return result;
                
            } catch (error) {
                console.error('Error getting utilization heatmap:', error);
                return [];
            }
        });
        
        // Cost breakdown function
        this.on('getCostBreakdown', async (req) => {
            const { startDate, endDate } = req.data;
            
            try {
                // Simplified query without complex joins
                const executionData = await SELECT.from('WorkflowTemplateService.ExecutionHistory')
                    .columns([
                        'instance_ID',
                        'COUNT(*) as executionCount',
                        'SUM(duration) as totalDurationSeconds',
                        'SUM(tasksTotal) as totalTasks'
                    ])
                    .where('startTime >= ', startDate)
                    .where('startTime <= ', endDate)
                    .groupBy('instance_ID');
                
                // Get template information separately
                const instanceTemplateMap = {};
                for (const exec of executionData) {
                    if (!instanceTemplateMap[exec.instance_ID]) {
                        const instance = await SELECT.one.from('WorkflowTemplateService.Instances')
                            .where({ ID: exec.instance_ID });
                        if (instance) {
                            const template = await SELECT.one.from('WorkflowTemplateService.Templates')
                                .where({ ID: instance.template_ID });
                            instanceTemplateMap[exec.instance_ID] = template;
                        }
                    }
                }
                
                // Aggregate by template
                const templateCosts = {};
                for (const exec of executionData) {
                    const template = instanceTemplateMap[exec.instance_ID];
                    if (template) {
                        if (!templateCosts[template.ID]) {
                            templateCosts[template.ID] = {
                                templateId: template.ID,
                                templateName: template.name,
                                executionCount: 0,
                                totalDurationSeconds: 0,
                                totalTasks: 0
                            };
                        }
                        templateCosts[template.ID].executionCount += exec.executionCount || 0;
                        templateCosts[template.ID].totalDurationSeconds += exec.totalDurationSeconds || 0;
                        templateCosts[template.ID].totalTasks += exec.totalTasks || 0;
                    }
                }
                
                const costData = Object.values(templateCosts);
                
                const results = costData.map(row => ({
                    templateId: row.templateId,
                    templateName: row.templateName,
                    totalCost: (row.totalDurationSeconds || 0) * 0.001,
                    executionCount: row.executionCount || 0,
                    avgCostPerExecution: row.executionCount ? 
                        ((row.totalDurationSeconds || 0) * 0.001) / row.executionCount : 0,
                    costByAgent: [] // Simplified - would need more complex query for agent breakdown
                }));
                
                return results;
                
            } catch (error) {
                console.error('Error getting cost breakdown:', error);
                return [];
            }
        });
        
        // Predictive analytics function (simplified)
        this.on('getPredictiveAnalytics', async (req) => {
            const { templateId, forecastDays } = req.data;
            
            try {
                // Get historical data for last 30 days
                const historicalDate = new Date();
                historicalDate.setDate(historicalDate.getDate() - 30);
                
                // Get instances for the template first
                const templateInstances = await SELECT.from('WorkflowTemplateService.Instances')
                    .columns(['ID'])
                    .where('template_ID = ', templateId);
                
                const instanceIds = templateInstances.map(i => i.ID);
                
                let historicalData = [];
                if (instanceIds.length > 0) {
                    historicalData = await SELECT.from('WorkflowTemplateService.ExecutionHistory')
                        .columns([
                            'DATE(startTime) as date',
                            'COUNT(*) as value'
                        ])
                        .where('instance_ID in ', instanceIds)
                        .where('startTime >= ', historicalDate.toISOString())
                        .groupBy('DATE(startTime)')
                        .orderBy('DATE(startTime)');
                }
                
                // Simple linear trend calculation
                const historicalTrend = historicalData.map(row => ({
                    date: row.date,
                    value: parseFloat(row.value) || 0
                }));
                
                // Calculate simple moving average for prediction
                const recentValues = historicalTrend.slice(-7); // Last 7 days
                const avgValue = recentValues.reduce((sum, item) => sum + item.value, 0) / recentValues.length;
                
                // Generate predicted trend (simplified)
                const predictedTrend = [];
                for (let i = 1; i <= forecastDays; i++) {
                    const futureDate = new Date();
                    futureDate.setDate(futureDate.getDate() + i);
                    
                    // Simple prediction with some variance
                    const variance = avgValue * 0.2; // 20% variance
                    predictedTrend.push({
                        date: futureDate.toISOString().split('T')[0],
                        predictedValue: Math.max(0, avgValue + (Math.random() - 0.5) * variance),
                        confidenceInterval: {
                            lower: Math.max(0, avgValue - variance),
                            upper: avgValue + variance
                        }
                    });
                }
                
                // Generate insights
                const insights = [];
                if (avgValue > 5) {
                    insights.push("High usage template - consider optimization");
                }
                if (recentValues.length > 0 && recentValues[recentValues.length - 1].value < avgValue * 0.5) {
                    insights.push("Usage declining - may need promotion");
                }
                
                return {
                    historicalTrend,
                    predictedTrend,
                    insights
                };
                
            } catch (error) {
                console.error('Error getting predictive analytics:', error);
                return {
                    historicalTrend: [],
                    predictedTrend: [],
                    insights: ['Error calculating predictions']
                };
            }
        });
        
        // Override read operations for calculated views
        this.on('READ', 'TemplatePerformance', async (req, next) => {
            try {
                // Add custom filtering or calculations if needed
                const results = await next();
                
                // Enhance results with additional calculations
                return results.map(row => ({
                    ...row,
                    performanceGrade: this._calculatePerformanceGrade(row.successRate, row.avgDurationSeconds),
                    trend: this._calculateTrend(row.successRate)
                }));
                
            } catch (error) {
                console.error('Error reading template performance:', error);
                return [];
            }
        });
        
        this.on('READ', 'AgentUtilization', async (req, next) => {
            try {
                const results = await next();
                
                // Add utilization status
                return results.map(row => ({
                    ...row,
                    utilizationStatus: this._getUtilizationStatus(row.executionCount),
                    efficiency: this._calculateEfficiency(row.avgResponseTime, row.successRate)
                }));
                
            } catch (error) {
                console.error('Error reading agent utilization:', error);
                return [];
            }
        });
        
        return super.init();
    }
    
    /**
     * Helper methods for analytics calculations
     */
    
    _calculatePerformanceGrade(successRate, avgDuration) {
        if (successRate >= 95 && avgDuration <= 30) return 'A';
        if (successRate >= 90 && avgDuration <= 60) return 'B';
        if (successRate >= 80 && avgDuration <= 120) return 'C';
        if (successRate >= 70) return 'D';
        return 'F';
    }
    
    _calculateTrend(successRate) {
        // Simplified trend calculation
        if (successRate >= 95) return 'up';
        if (successRate >= 85) return 'stable';
        return 'down';
    }
    
    _getUtilizationStatus(executionCount) {
        if (executionCount >= 100) return 'high';
        if (executionCount >= 50) return 'medium';
        if (executionCount >= 10) return 'low';
        return 'very_low';
    }
    
    _calculateEfficiency(avgResponseTime, successRate) {
        // Combine response time and success rate for efficiency score
        const responseScore = Math.max(0, 100 - (avgResponseTime / 10)); // Penalty for slow response
        const successScore = successRate || 0;
        
        return Math.round((responseScore + successScore) / 2);
    }
};