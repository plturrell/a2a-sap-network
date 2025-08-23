/**
 * Agent 13 Adapter - Agent Builder
 * Converts between REST API and OData formats for dynamic agent creation,
 * code generation, template management, build pipelines, and deployments
 */

const axios = require('axios');
const { v4: uuidv4 } = require('uuid');

class Agent13Adapter {
    constructor() {
        this.baseUrl = process.env.AGENT13_BASE_URL || 'http://localhost:8013';
        this.apiVersion = 'v1';
        this.timeout = 45000; // 45 second timeout for builds and deployments
    }

    // ===== AGENT TEMPLATE OPERATIONS =====
    async getAgentTemplates(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/templates`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'AgentTemplate');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createAgentTemplate(data) {
        try {
            const restData = this._convertODataTemplateToREST(data);
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/templates`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTTemplateToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateAgentTemplate(id, data) {
        try {
            const restData = this._convertODataTemplateToREST(data);
            const response = await axios.put(`${this.baseUrl}/api/${this.apiVersion}/templates/${id}`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTTemplateToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async deleteAgentTemplate(id) {
        try {
            await axios.delete(`${this.baseUrl}/api/${this.apiVersion}/templates/${id}`, {
                timeout: this.timeout
            });
            return true;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== AGENT BUILD OPERATIONS =====
    async getAgentBuilds(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/builds`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'AgentBuild');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createAgentBuild(data) {
        try {
            const restData = this._convertODataBuildToREST(data);
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/builds`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTBuildToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateAgentBuild(id, data) {
        try {
            const restData = this._convertODataBuildToREST(data);
            const response = await axios.put(`${this.baseUrl}/api/${this.apiVersion}/builds/${id}`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTBuildToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async deleteAgentBuild(id) {
        try {
            await axios.delete(`${this.baseUrl}/api/${this.apiVersion}/builds/${id}`, {
                timeout: this.timeout
            });
            return true;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async startBuild(buildId) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/builds/${buildId}/start`, {}, {
                timeout: this.timeout
            });
            return response.data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== TEMPLATE COMPONENT OPERATIONS =====
    async getTemplateComponents(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/components`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'TemplateComponent');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createTemplateComponent(data) {
        try {
            const restData = this._convertODataComponentToREST(data);
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/components`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTComponentToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateTemplateComponent(id, data) {
        try {
            const restData = this._convertODataComponentToREST(data);
            const response = await axios.put(`${this.baseUrl}/api/${this.apiVersion}/components/${id}`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTComponentToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async deleteTemplateComponent(id) {
        try {
            await axios.delete(`${this.baseUrl}/api/${this.apiVersion}/components/${id}`, {
                timeout: this.timeout
            });
            return true;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== AGENT DEPLOYMENT OPERATIONS =====
    async getAgentDeployments(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/deployments`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'AgentDeployment');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createAgentDeployment(data) {
        try {
            const restData = this._convertODataDeploymentToREST(data);
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/deployments`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTDeploymentToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateAgentDeployment(id, data) {
        try {
            const restData = this._convertODataDeploymentToREST(data);
            const response = await axios.put(`${this.baseUrl}/api/${this.apiVersion}/deployments/${id}`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTDeploymentToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async deleteAgentDeployment(id) {
        try {
            await axios.delete(`${this.baseUrl}/api/${this.apiVersion}/deployments/${id}`, {
                timeout: this.timeout
            });
            return true;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async startDeployment(deploymentId) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/deployments/${deploymentId}/start`, {}, {
                timeout: this.timeout
            });
            return response.data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== BUILD PIPELINE OPERATIONS =====
    async getBuildPipelines(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/pipelines`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'BuildPipeline');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createBuildPipeline(data) {
        try {
            const restData = this._convertODataPipelineToREST(data);
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/pipelines`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTPipelineToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateBuildPipeline(id, data) {
        try {
            const restData = this._convertODataPipelineToREST(data);
            const response = await axios.put(`${this.baseUrl}/api/${this.apiVersion}/pipelines/${id}`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTPipelineToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async deleteBuildPipeline(id) {
        try {
            await axios.delete(`${this.baseUrl}/api/${this.apiVersion}/pipelines/${id}`, {
                timeout: this.timeout
            });
            return true;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== SPECIALIZED OPERATIONS =====
    async validateTemplate(templateId, validationType) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/validate-template`, {
                template_id: templateId,
                validation_type: validationType
            }, {
                timeout: this.timeout
            });
            
            return {
                isValid: response.data.is_valid,
                errors: response.data.errors || [],
                warnings: response.data.warnings || [],
                suggestions: response.data.suggestions || [],
                issues: response.data.issues || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async testAgent(buildId, testSuite, testConfiguration) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/test-agent`, {
                build_id: buildId,
                test_suite: testSuite,
                test_configuration: testConfiguration
            }, {
                timeout: this.timeout
            });
            
            return {
                passed: response.data.passed || 0,
                failed: response.data.failed || 0,
                duration: response.data.duration || 0,
                results: response.data.results || [],
                coverage: response.data.coverage || 0
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async triggerPipeline(pipelineId, parameters, priority) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/pipelines/${pipelineId}/trigger`, {
                parameters,
                priority
            }, {
                timeout: this.timeout
            });
            
            return {
                executionId: response.data.execution_id,
                pipelineName: response.data.pipeline_name,
                status: response.data.status
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async startBatchBuild(batchId, templateIds) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/batch-build`, {
                batch_id: batchId,
                template_ids: templateIds
            }, {
                timeout: this.timeout
            });
            
            return response.data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== STATISTICS AND INFORMATION =====
    async getBuilderStatistics() {
        try {
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/statistics`, {
                timeout: this.timeout
            });
            
            return {
                totalTemplates: response.data.total_templates || 0,
                totalBuilds: response.data.total_builds || 0,
                successfulBuilds: response.data.successful_builds || 0,
                failedBuilds: response.data.failed_builds || 0,
                activeDeployments: response.data.active_deployments || 0,
                totalPipelines: response.data.total_pipelines || 0,
                buildsToday: response.data.builds_today || 0,
                deploymentsToday: response.data.deployments_today || 0,
                averageBuildTime: response.data.average_build_time || 0,
                templateUsageStats: JSON.stringify(response.data.template_usage_stats || {})
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getTemplateDetails(templateId) {
        try {
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/templates/${templateId}/details`, {
                timeout: this.timeout
            });
            
            const data = response.data;
            return {
                templateId: data.template_id,
                templateName: data.template_name,
                agentType: data.agent_type,
                version: data.version,
                baseTemplate: data.base_template,
                capabilities: data.capabilities,
                configuration: data.configuration,
                components: JSON.stringify(data.components || []),
                lastModified: data.last_modified,
                buildCount: data.build_count || 0,
                deploymentCount: data.deployment_count || 0,
                successRate: data.success_rate || 0.0,
                documentation: data.documentation,
                dependencies: JSON.stringify(data.dependencies || [])
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getDeploymentTargets() {
        try {
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/deployment-targets`, {
                timeout: this.timeout
            });
            
            const data = response.data;
            return {
                environments: JSON.stringify(data.environments || []),
                kubernetesTargets: JSON.stringify(data.kubernetes_targets || []),
                dockerTargets: JSON.stringify(data.docker_targets || []),
                cloudTargets: JSON.stringify(data.cloud_targets || []),
                onPremiseTargets: JSON.stringify(data.on_premise_targets || []),
                availableResources: JSON.stringify(data.available_resources || {})
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getBuildPipelinesData() {
        try {
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/pipelines/info`, {
                timeout: this.timeout
            });
            
            const data = response.data;
            return {
                pipelines: JSON.stringify(data.pipelines || []),
                activeBuilds: data.active_builds || 0,
                queuedBuilds: data.queued_builds || 0,
                buildHistory: JSON.stringify(data.build_history || []),
                pipelineMetrics: JSON.stringify(data.pipeline_metrics || {})
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== CONVERSION HELPERS =====
    _convertODataTemplateToREST(odataTemplate) {
        return {
            id: odataTemplate.ID,
            template_name: odataTemplate.templateName,
            agent_type: odataTemplate.agentType?.toLowerCase(),
            base_template: odataTemplate.baseTemplate,
            capabilities: odataTemplate.capabilities,
            configuration: odataTemplate.configuration,
            description: odataTemplate.description,
            status: odataTemplate.status?.toLowerCase(),
            version: odataTemplate.version,
            is_public: odataTemplate.isPublic,
            tags: odataTemplate.tags,
            framework: odataTemplate.framework,
            language: odataTemplate.language?.toLowerCase(),
            created_at: odataTemplate.createdAt,
            modified_at: odataTemplate.modifiedAt
        };
    }

    _convertRESTTemplateToOData(restTemplate) {
        return {
            ID: restTemplate.id,
            templateName: restTemplate.template_name,
            agentType: restTemplate.agent_type?.toUpperCase(),
            baseTemplate: restTemplate.base_template,
            capabilities: restTemplate.capabilities,
            configuration: restTemplate.configuration,
            description: restTemplate.description,
            status: restTemplate.status?.toUpperCase(),
            version: restTemplate.version,
            isPublic: restTemplate.is_public !== false,
            tags: restTemplate.tags,
            framework: restTemplate.framework,
            language: restTemplate.language?.toUpperCase(),
            buildCount: restTemplate.build_count || 0,
            successRate: restTemplate.success_rate || 0.0,
            lastBuildAt: restTemplate.last_build_at,
            createdBy: restTemplate.created_by,
            createdAt: restTemplate.created_at,
            modifiedAt: restTemplate.modified_at
        };
    }

    _convertODataBuildToREST(odataBuild) {
        return {
            id: odataBuild.ID,
            template_id: odataBuild.templateId,
            build_number: odataBuild.buildNumber,
            agent_name: odataBuild.agentName,
            build_type: odataBuild.buildType?.toLowerCase(),
            status: odataBuild.status?.toLowerCase(),
            target_environment: odataBuild.targetEnvironment?.toLowerCase(),
            build_config: odataBuild.buildConfig,
            artifacts: odataBuild.artifacts,
            build_logs: odataBuild.buildLogs,
            test_results: odataBuild.testResults,
            duration: odataBuild.duration,
            started_at: odataBuild.startedAt,
            completed_at: odataBuild.completedAt,
            created_by: odataBuild.createdBy,
            created_at: odataBuild.createdAt,
            modified_at: odataBuild.modifiedAt
        };
    }

    _convertRESTBuildToOData(restBuild) {
        return {
            ID: restBuild.id,
            templateId: restBuild.template_id,
            buildNumber: restBuild.build_number,
            agentName: restBuild.agent_name,
            buildType: restBuild.build_type?.toUpperCase(),
            status: restBuild.status?.toUpperCase(),
            targetEnvironment: restBuild.target_environment?.toUpperCase(),
            buildConfig: restBuild.build_config,
            artifacts: restBuild.artifacts,
            buildLogs: restBuild.build_logs,
            testResults: restBuild.test_results,
            duration: restBuild.duration || 0,
            startedAt: restBuild.started_at,
            completedAt: restBuild.completed_at,
            createdBy: restBuild.created_by,
            createdAt: restBuild.created_at,
            modifiedAt: restBuild.modified_at
        };
    }

    _convertODataComponentToREST(odataComponent) {
        return {
            id: odataComponent.ID,
            template_id: odataComponent.templateId,
            component_name: odataComponent.componentName,
            component_type: odataComponent.componentType?.toLowerCase(),
            source_code: odataComponent.sourceCode,
            dependencies: odataComponent.dependencies,
            configuration: odataComponent.configuration,
            version: odataComponent.version,
            is_core: odataComponent.isCore,
            is_reusable: odataComponent.isReusable,
            created_at: odataComponent.createdAt,
            modified_at: odataComponent.modifiedAt
        };
    }

    _convertRESTComponentToOData(restComponent) {
        return {
            ID: restComponent.id,
            templateId: restComponent.template_id,
            componentName: restComponent.component_name,
            componentType: restComponent.component_type?.toUpperCase(),
            sourceCode: restComponent.source_code,
            dependencies: restComponent.dependencies,
            configuration: restComponent.configuration,
            version: restComponent.version,
            isCore: restComponent.is_core !== false,
            isReusable: restComponent.is_reusable !== false,
            createdBy: restComponent.created_by,
            createdAt: restComponent.created_at,
            modifiedAt: restComponent.modified_at
        };
    }

    _convertODataDeploymentToREST(odataDeployment) {
        return {
            id: odataDeployment.ID,
            build_id: odataDeployment.buildId,
            deployment_name: odataDeployment.deploymentName,
            target_environment: odataDeployment.targetEnvironment?.toLowerCase(),
            deployment_type: odataDeployment.deploymentType?.toLowerCase(),
            status: odataDeployment.status?.toLowerCase(),
            endpoint: odataDeployment.endpoint,
            replicas: odataDeployment.replicas,
            resources: odataDeployment.resources,
            environment_variables: odataDeployment.environmentVariables,
            health_check_url: odataDeployment.healthCheckUrl,
            is_active: odataDeployment.isActive,
            auto_restart: odataDeployment.autoRestart,
            deployed_at: odataDeployment.deployedAt,
            last_health_check: odataDeployment.lastHealthCheck,
            deployed_by: odataDeployment.deployedBy,
            created_at: odataDeployment.createdAt,
            modified_at: odataDeployment.modifiedAt
        };
    }

    _convertRESTDeploymentToOData(restDeployment) {
        return {
            ID: restDeployment.id,
            buildId: restDeployment.build_id,
            deploymentName: restDeployment.deployment_name,
            targetEnvironment: restDeployment.target_environment?.toUpperCase(),
            deploymentType: restDeployment.deployment_type?.toUpperCase(),
            status: restDeployment.status?.toUpperCase(),
            endpoint: restDeployment.endpoint,
            replicas: restDeployment.replicas || 1,
            resources: restDeployment.resources,
            environmentVariables: restDeployment.environment_variables,
            healthCheckUrl: restDeployment.health_check_url,
            isActive: restDeployment.is_active !== false,
            autoRestart: restDeployment.auto_restart !== false,
            deployedAt: restDeployment.deployed_at,
            lastHealthCheck: restDeployment.last_health_check,
            deployedBy: restDeployment.deployed_by,
            createdAt: restDeployment.created_at,
            modifiedAt: restDeployment.modified_at
        };
    }

    _convertODataPipelineToREST(odataPipeline) {
        return {
            id: odataPipeline.ID,
            pipeline_name: odataPipeline.pipelineName,
            template_ids: odataPipeline.templateIds,
            stages: odataPipeline.stages,
            triggers: odataPipeline.triggers,
            configuration: odataPipeline.configuration,
            status: odataPipeline.status?.toLowerCase(),
            is_active: odataPipeline.isActive,
            last_run_at: odataPipeline.lastRunAt,
            next_run_at: odataPipeline.nextRunAt,
            created_at: odataPipeline.createdAt,
            modified_at: odataPipeline.modifiedAt
        };
    }

    _convertRESTPipelineToOData(restPipeline) {
        return {
            ID: restPipeline.id,
            pipelineName: restPipeline.pipeline_name,
            templateIds: restPipeline.template_ids,
            stages: restPipeline.stages,
            triggers: restPipeline.triggers,
            configuration: restPipeline.configuration,
            status: restPipeline.status?.toUpperCase(),
            isActive: restPipeline.is_active !== false,
            lastRunAt: restPipeline.last_run_at,
            nextRunAt: restPipeline.next_run_at,
            createdBy: restPipeline.created_by,
            createdAt: restPipeline.created_at,
            modifiedAt: restPipeline.modified_at
        };
    }

    _convertODataToREST(query) {
        // Convert OData query parameters to REST API parameters
        const params = {};
        
        if (query.$filter) {
            params.filter = query.$filter;
        }
        
        if (query.$orderby) {
            params.sort = query.$orderby;
        }
        
        if (query.$top) {
            params.limit = query.$top;
        }
        
        if (query.$skip) {
            params.offset = query.$skip;
        }
        
        if (query.$select) {
            params.fields = query.$select;
        }
        
        return params;
    }

    _convertRESTToOData(data, entityType) {
        if (Array.isArray(data)) {
            return data.map(item => this._convertSingleRESTToOData(item, entityType));
        }
        return this._convertSingleRESTToOData(data, entityType);
    }

    _convertSingleRESTToOData(item, entityType) {
        switch (entityType) {
            case 'AgentTemplate':
                return this._convertRESTTemplateToOData(item);
            case 'AgentBuild':
                return this._convertRESTBuildToOData(item);
            case 'TemplateComponent':
                return this._convertRESTComponentToOData(item);
            case 'AgentDeployment':
                return this._convertRESTDeploymentToOData(item);
            case 'BuildPipeline':
                return this._convertRESTPipelineToOData(item);
            default:
                return item;
        }
    }

    _handleError(error) {
        if (error.response) {
            const status = error.response.status;
            const message = error.response.data?.message || error.response.statusText || 'Unknown error';
            
            if (status === 404) {
                return new Error(`Resource not found: ${message}`);
            } else if (status === 400) {
                return new Error(`Bad request: ${message}`);
            } else if (status === 401) {
                return new Error(`Unauthorized: ${message}`);
            } else if (status === 403) {
                return new Error(`Forbidden: ${message}`);
            } else if (status >= 500) {
                return new Error(`Agent 13 service error: ${message}`);
            } else {
                return new Error(`HTTP ${status}: ${message}`);
            }
        } else if (error.request) {
            return new Error('Agent 13 service is unavailable');
        } else {
            return new Error(`Agent 13 adapter error: ${error.message}`);
        }
    }
}

module.exports = Agent13Adapter;