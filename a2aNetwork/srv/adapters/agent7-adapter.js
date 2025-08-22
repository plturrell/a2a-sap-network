/**
 * Agent 7 Adapter - Agent Management & Orchestration
 * Converts between REST API and OData formats for agent management and coordination operations
 */

const axios = require('axios');
const { v4: uuidv4 } = require('uuid');

class Agent7Adapter {
    constructor() {
        this.baseUrl = process.env.AGENT7_BASE_URL || 'http://localhost:8006';
        this.apiVersion = 'v1';
        this.timeout = 30000;
    }

    // Registered Agents
    async getRegisteredAgents(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/registered-agents`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'RegisteredAgent');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createRegisteredAgent(data) {
        try {
            const restData = this._convertODataAgentToREST(data);
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/registered-agents`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTAgentToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateRegisteredAgent(id, data) {
        try {
            const restData = this._convertODataAgentToREST(data);
            const response = await axios.put(`${this.baseUrl}/api/${this.apiVersion}/registered-agents/${id}`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTAgentToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async deleteRegisteredAgent(id) {
        try {
            await axios.delete(`${this.baseUrl}/api/${this.apiVersion}/registered-agents/${id}`, {
                timeout: this.timeout
            });
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Agent Registration and Management
    async registerAgent(agentData) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/register-agent`, {
                agent_name: agentData.agentName,
                agent_type: agentData.agentType,
                agent_version: agentData.agentVersion,
                endpoint_url: agentData.endpointUrl,
                capabilities: agentData.capabilities,
                configuration: agentData.configuration
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                agentId: response.data.agent_id
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateAgentStatus(agentId, status, reason) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/registered-agents/${agentId}/update-status`, {
                status: status.toLowerCase(),
                reason
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async performHealthCheck(agentId) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/registered-agents/${agentId}/health-check`, {}, {
                timeout: this.timeout
            });
            
            return {
                status: response.data.status?.toUpperCase(),
                responseTime: response.data.response_time,
                statusCode: response.data.status_code,
                details: response.data.details,
                errorDetails: response.data.error_details,
                alertTriggered: response.data.alert_triggered,
                alertLevel: response.data.alert_level,
                recommendations: response.data.recommendations
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateAgentConfiguration(agentId, configuration, restartRequired) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/registered-agents/${agentId}/update-config`, {
                configuration,
                restart_required: restartRequired
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async deactivateAgent(agentId, reason, gracefulShutdown) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/registered-agents/${agentId}/deactivate`, {
                reason,
                graceful_shutdown: gracefulShutdown
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async scheduleTask(agentId, taskData) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/registered-agents/${agentId}/schedule-task`, {
                task_type: taskData.taskType,
                parameters: taskData.parameters,
                scheduled_time: taskData.scheduledTime,
                priority: taskData.priority.toLowerCase()
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                taskId: response.data.task_id
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async assignWorkload(agentId, workloadData) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/registered-agents/${agentId}/assign-workload`, {
                workload_type: workloadData.workloadType,
                parameters: workloadData.parameters,
                priority: workloadData.priority.toLowerCase(),
                expected_duration: workloadData.expectedDuration
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                workloadId: response.data.workload_id
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Management Tasks
    async getManagementTasks(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/management-tasks`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'ManagementTask');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createManagementTask(data) {
        try {
            const restData = this._convertODataTaskToREST(data);
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/management-tasks`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTTaskToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async executeTask(taskId) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/management-tasks/${taskId}/execute`, {}, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async pauseTask(taskId) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/management-tasks/${taskId}/pause`, {}, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async resumeTask(taskId) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/management-tasks/${taskId}/resume`, {}, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async cancelTask(taskId, reason) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/management-tasks/${taskId}/cancel`, {
                reason
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Agent Coordination
    async getAgentCoordinations(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/coordination`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'AgentCoordination');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async activateCoordination(coordinationId) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/coordination/${coordinationId}/activate`, {}, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Bulk Operations
    async getBulkOperations(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/bulk-operations`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'BulkOperation');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async executeBulkOperation(operationId) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/bulk-operations/${operationId}/execute`, {}, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Agent Management Functions
    async getAgentTypes() {
        try {
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/agent-types`, {
                timeout: this.timeout
            });
            
            return response.data.map(type => ({
                type: type.type,
                description: type.description,
                capabilities: JSON.stringify(type.capabilities),
                requirements: type.requirements
            }));
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getDashboardData() {
        try {
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/dashboard`, {
                timeout: this.timeout
            });
            
            return {
                totalAgents: response.data.total_agents,
                activeAgents: response.data.active_agents,
                healthyAgents: response.data.healthy_agents,
                tasksInProgress: response.data.tasks_in_progress,
                averageResponseTime: response.data.average_response_time,
                systemLoad: response.data.system_load,
                alerts: response.data.alerts,
                trends: response.data.trends
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getHealthStatus(agentId) {
        try {
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/health-status/${agentId}`, {
                timeout: this.timeout
            });
            
            return {
                agentId: response.data.agent_id,
                status: response.data.status?.toUpperCase(),
                lastCheck: response.data.last_check,
                responseTime: response.data.response_time,
                errorRate: response.data.error_rate,
                alerts: response.data.alerts,
                recommendations: response.data.recommendations
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getPerformanceAnalysis(agentId, timeRange) {
        try {
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/performance-analysis/${agentId}`, {
                params: { time_range: timeRange },
                timeout: this.timeout
            });
            
            return {
                agentId: response.data.agent_id,
                metrics: response.data.metrics.map(metric => ({
                    metricType: metric.metric_type,
                    value: metric.value,
                    trend: metric.trend?.toUpperCase(),
                    benchmark: metric.benchmark
                })),
                bottlenecks: response.data.bottlenecks,
                recommendations: response.data.recommendations
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getCoordinationStatus() {
        try {
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/coordination-status`, {
                timeout: this.timeout
            });
            
            return {
                activeCoordinations: response.data.active_coordinations,
                totalWorkflows: response.data.total_workflows,
                averageSuccess: response.data.average_success,
                currentLoad: response.data.current_load,
                connections: response.data.connections.map(conn => ({
                    source: conn.source,
                    target: conn.target,
                    status: conn.status?.toUpperCase(),
                    latency: conn.latency
                }))
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getAgentCapabilities(agentType) {
        try {
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/agent-capabilities/${agentType}`, {
                timeout: this.timeout
            });
            
            return {
                type: response.data.type,
                capabilities: response.data.capabilities,
                supportedProtocols: response.data.supported_protocols,
                requirements: response.data.requirements,
                limitations: response.data.limitations
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async validateConfiguration(configuration, agentType) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/validate-configuration`, {
                configuration,
                agent_type: agentType
            }, {
                timeout: this.timeout
            });
            
            return {
                valid: response.data.valid,
                errors: response.data.errors,
                warnings: response.data.warnings,
                suggestions: response.data.suggestions
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getLoadBalancingRecommendations() {
        try {
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/load-balancing-recommendations`, {
                timeout: this.timeout
            });
            
            return {
                strategy: response.data.strategy,
                distribution: response.data.distribution.map(item => ({
                    agentId: item.agent_id,
                    recommendedWeight: item.recommended_weight,
                    currentLoad: item.current_load,
                    capacity: item.capacity
                })),
                expectedImprovement: response.data.expected_improvement
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Health Checks and Performance Metrics
    async getAgentHealthChecks(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/health-checks`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'AgentHealthCheck');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getAgentPerformanceMetrics(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/performance-metrics`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'AgentPerformanceMetric');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Utility methods
    _convertODataToREST(query) {
        const params = {};
        
        if (query.$top) params.limit = query.$top;
        if (query.$skip) params.offset = query.$skip;
        if (query.$orderby) params.sort = query.$orderby.replace(/ desc/gi, '-').replace(/ asc/gi, '');
        if (query.$filter) params.filter = this._parseODataFilter(query.$filter);
        if (query.$select) params.fields = query.$select;
        
        return params;
    }

    _parseODataFilter(filter) {
        // Convert OData filter to REST query parameters
        return filter
            .replace(/ eq /g, '=')
            .replace(/ ne /g, '!=')
            .replace(/ gt /g, '>')
            .replace(/ ge /g, '>=')
            .replace(/ lt /g, '<')
            .replace(/ le /g, '<=')
            .replace(/ and /g, '&')
            .replace(/ or /g, '|');
    }

    _convertRESTToOData(data, entityType) {
        if (Array.isArray(data)) {
            return data.map(item => this._convertRESTItemToOData(item, entityType));
        }
        return this._convertRESTItemToOData(data, entityType);
    }

    _convertRESTItemToOData(item, entityType) {
        switch (entityType) {
            case 'RegisteredAgent':
                return this._convertRESTAgentToOData(item);
            case 'ManagementTask':
                return this._convertRESTTaskToOData(item);
            case 'AgentCoordination':
                return this._convertRESTCoordinationToOData(item);
            case 'BulkOperation':
                return this._convertRESTBulkOpToOData(item);
            case 'AgentHealthCheck':
                return this._convertRESTHealthCheckToOData(item);
            case 'AgentPerformanceMetric':
                return this._convertRESTMetricToOData(item);
            default:
                return item;
        }
    }

    _convertRESTAgentToOData(agent) {
        return {
            ID: agent.id || uuidv4(),
            agentName: agent.agent_name,
            agentType: agent.agent_type?.toUpperCase(),
            agentVersion: agent.agent_version,
            endpointUrl: agent.endpoint_url,
            status: agent.status?.toUpperCase() || 'REGISTERING',
            healthStatus: agent.health_status?.toUpperCase() || 'UNKNOWN',
            capabilities: JSON.stringify(agent.capabilities || {}),
            configuration: JSON.stringify(agent.configuration || {}),
            performanceScore: agent.performance_score || 0,
            responseTime: agent.response_time,
            throughput: agent.throughput,
            errorRate: agent.error_rate || 0,
            lastHealthCheck: agent.last_health_check,
            registrationDate: agent.registration_date,
            deactivationDate: agent.deactivation_date,
            loadBalanceWeight: agent.load_balance_weight || 50,
            priority: agent.priority || 5,
            tags: JSON.stringify(agent.tags || []),
            notes: agent.notes,
            createdAt: agent.created_at,
            createdBy: agent.created_by,
            modifiedAt: agent.modified_at,
            modifiedBy: agent.modified_by
        };
    }

    _convertODataAgentToREST(agent) {
        const restAgent = {
            agent_name: agent.agentName,
            agent_type: agent.agentType?.toLowerCase(),
            agent_version: agent.agentVersion,
            endpoint_url: agent.endpointUrl
        };
        
        if (agent.status) restAgent.status = agent.status.toLowerCase();
        if (agent.healthStatus) restAgent.health_status = agent.healthStatus.toLowerCase();
        if (agent.capabilities) restAgent.capabilities = JSON.parse(agent.capabilities);
        if (agent.configuration) restAgent.configuration = JSON.parse(agent.configuration);
        if (agent.performanceScore !== undefined) restAgent.performance_score = agent.performanceScore;
        if (agent.loadBalanceWeight !== undefined) restAgent.load_balance_weight = agent.loadBalanceWeight;
        if (agent.priority !== undefined) restAgent.priority = agent.priority;
        if (agent.tags) restAgent.tags = JSON.parse(agent.tags);
        if (agent.notes) restAgent.notes = agent.notes;
        
        return restAgent;
    }

    _convertRESTTaskToOData(task) {
        return {
            ID: task.id || uuidv4(),
            taskName: task.task_name,
            taskType: task.task_type?.toUpperCase(),
            status: task.status?.toUpperCase() || 'SCHEDULED',
            priority: task.priority?.toUpperCase() || 'NORMAL',
            targetAgents: JSON.stringify(task.target_agents || []),
            parameters: JSON.stringify(task.parameters || {}),
            scheduleType: task.schedule_type?.toUpperCase() || 'IMMEDIATE',
            scheduledTime: task.scheduled_time,
            recurrencePattern: task.recurrence_pattern,
            startTime: task.start_time,
            endTime: task.end_time,
            duration: task.duration,
            progress: task.progress || 0,
            result: JSON.stringify(task.result || {}),
            errorMessage: task.error_message,
            retryCount: task.retry_count || 0,
            maxRetries: task.max_retries || 3,
            notificationSent: task.notification_sent !== false,
            rollbackAvailable: task.rollback_available !== false,
            createdAt: task.created_at,
            createdBy: task.created_by,
            modifiedAt: task.modified_at,
            modifiedBy: task.modified_by
        };
    }

    _convertODataTaskToREST(task) {
        const restTask = {
            task_name: task.taskName,
            task_type: task.taskType?.toLowerCase(),
            status: task.status?.toLowerCase()
        };
        
        if (task.priority) restTask.priority = task.priority.toLowerCase();
        if (task.targetAgents) restTask.target_agents = JSON.parse(task.targetAgents);
        if (task.parameters) restTask.parameters = JSON.parse(task.parameters);
        if (task.scheduleType) restTask.schedule_type = task.scheduleType.toLowerCase();
        if (task.scheduledTime) restTask.scheduled_time = task.scheduledTime;
        if (task.recurrencePattern) restTask.recurrence_pattern = task.recurrencePattern;
        
        return restTask;
    }

    _convertRESTCoordinationToOData(coordination) {
        return {
            ID: coordination.id || uuidv4(),
            coordinationName: coordination.coordination_name,
            coordinationType: coordination.coordination_type?.toUpperCase(),
            status: coordination.status?.toUpperCase() || 'DRAFT',
            participatingAgents: JSON.stringify(coordination.participating_agents || []),
            coordinationRules: JSON.stringify(coordination.coordination_rules || {}),
            loadBalanceStrategy: coordination.load_balance_strategy,
            failoverConfig: JSON.stringify(coordination.failover_config || {}),
            performanceTarget: coordination.performance_target,
            currentPerformance: coordination.current_performance,
            successRate: coordination.success_rate || 0,
            totalExecutions: coordination.total_executions || 0,
            failedExecutions: coordination.failed_executions || 0,
            averageDuration: coordination.average_duration,
            lastExecution: coordination.last_execution,
            nextScheduled: coordination.next_scheduled,
            createdAt: coordination.created_at,
            createdBy: coordination.created_by,
            modifiedAt: coordination.modified_at,
            modifiedBy: coordination.modified_by
        };
    }

    _convertRESTBulkOpToOData(operation) {
        return {
            ID: operation.id || uuidv4(),
            operationName: operation.operation_name,
            operationType: operation.operation_type?.toUpperCase(),
            status: operation.status?.toUpperCase() || 'PREPARING',
            targetCount: operation.target_count,
            successfulCount: operation.successful_count || 0,
            failedCount: operation.failed_count || 0,
            progress: operation.progress || 0,
            operationDetails: JSON.stringify(operation.operation_details || {}),
            results: JSON.stringify(operation.results || {}),
            rollbackData: JSON.stringify(operation.rollback_data || {}),
            startTime: operation.start_time,
            endTime: operation.end_time,
            duration: operation.duration,
            initiatedBy: operation.initiated_by,
            approvalRequired: operation.approval_required !== false,
            approvedBy: operation.approved_by,
            approvalTime: operation.approval_time,
            createdAt: operation.created_at,
            createdBy: operation.created_by,
            modifiedAt: operation.modified_at,
            modifiedBy: operation.modified_by
        };
    }

    _convertRESTHealthCheckToOData(healthCheck) {
        return {
            ID: healthCheck.id || uuidv4(),
            checkId: healthCheck.check_id,
            checkType: healthCheck.check_type?.toUpperCase(),
            status: healthCheck.status?.toUpperCase(),
            responseTime: healthCheck.response_time,
            statusCode: healthCheck.status_code,
            checkDetails: JSON.stringify(healthCheck.check_details || {}),
            errorDetails: healthCheck.error_details,
            timestamp: healthCheck.timestamp,
            alertTriggered: healthCheck.alert_triggered !== false,
            recommendations: JSON.stringify(healthCheck.recommendations || [])
        };
    }

    _convertRESTMetricToOData(metric) {
        return {
            ID: metric.id || uuidv4(),
            metricId: metric.metric_id,
            metricType: metric.metric_type?.toUpperCase(),
            value: metric.value,
            unit: metric.unit,
            timestamp: metric.timestamp,
            timeWindow: metric.time_window?.toUpperCase() || 'MINUTE',
            minValue: metric.min_value,
            maxValue: metric.max_value,
            averageValue: metric.average_value,
            percentile95: metric.percentile_95,
            percentile99: metric.percentile_99,
            trend: metric.trend?.toUpperCase(),
            anomalyDetected: metric.anomaly_detected !== false,
            benchmarkComparison: metric.benchmark_comparison
        };
    }

    _handleError(error) {
        if (error.response) {
            const status = error.response.status;
            const message = error.response.data?.message || error.message;
            
            switch (status) {
                case 400:
                    return new Error(`Bad Request: ${message}`);
                case 401:
                    return new Error(`Unauthorized: ${message}`);
                case 403:
                    return new Error(`Forbidden: ${message}`);
                case 404:
                    return new Error(`Not Found: ${message}`);
                case 500:
                    return new Error(`Internal Server Error: ${message}`);
                default:
                    return new Error(`HTTP ${status}: ${message}`);
            }
        } else if (error.request) {
            return new Error(`No response from Agent 7 service: ${error.message}`);
        } else {
            return new Error(`Agent 7 adapter error: ${error.message}`);
        }
    }
}

module.exports = Agent7Adapter;