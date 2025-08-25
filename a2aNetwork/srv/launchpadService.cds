/**
 * @fileoverview SAP Fiori Launchpad Service - CDS Definition
 * @since 1.0.0
 * @module launchpadService
 * 
 * CDS service definition for SAP Fiori Launchpad tile data
 * Provides real-time agent health and system metrics
 */

using { cuid, managed } from '@sap/cds/common';

namespace a2a.launchpad;

/**
 * LaunchpadService - Main service for SAP Fiori Launchpad
 * Provides actions to retrieve real-time system metrics
 */
service LaunchpadService {
    
    /**
     * Get Network Statistics for overview dashboard
     * @param id - Dashboard identifier (e.g., 'overview_dashboard')
     */
    action getNetworkStats(id: String) returns {
        d: {
            title: String;
            number: String;
            numberUnit: String;
            numberState: String;
            subtitle: String;
            stateArrow: String;
            info: String;
            real_metrics: {
                healthy_agents: Integer;
                total_agents: Integer;
                agent_health_score: Integer;
                total_active_tasks: Integer;
                total_skills: Integer;
                total_mcp_tools: Integer;
                blockchain_status: String;
                blockchain_score: Integer;
                mcp_status: String;
                mcp_score: Integer;
                overall_system_health: Integer;
            };
            timestamp: String;
        };
    };
    
    /**
     * Get individual agent status
     * @param agentId - Agent identifier (0-15)
     */
    action getAgentStatus(agentId: Integer) returns {
        d: {
            title: String;
            number: String;
            numberUnit: String;
            numberState: String;
            subtitle: String;
            stateArrow: String;
            info: String;
            status: String;
            agent_id: String;
            version: String;
            port: Integer;
            capabilities: {
                skills: Integer;
                handlers: Integer;
                mcp_tools: Integer;
                mcp_resources: Integer;
            };
            performance: {
                cpu_usage: Decimal;
                memory_usage: Decimal;
                uptime_seconds: Integer;
                success_rate: Decimal;
                avg_response_time_ms: Integer;
                processed_today: Integer;
                error_rate: Decimal;
                queue_depth: Integer;
            };
            timestamp: String;
        };
    };
    
    /**
     * Get blockchain statistics
     * @param id - Dashboard identifier (e.g., 'blockchain_dashboard')
     */
    action getBlockchainStats(id: String) returns {
        d: {
            title: String;
            number: String;
            numberUnit: String;
            numberState: String;
            subtitle: String;
            stateArrow: String;
            info: String;
            blockchain_metrics: {
                network: String;
                contracts: {};
                registered_agents_count: Integer;
                contract_count: Integer;
                trust_integration: Boolean;
                avg_trust_score: Decimal;
            };
            timestamp: String;
        };
    };
    
    /**
     * Get services count for marketplace
     */
    action getServicesCount() returns {
        d: {
            title: String;
            number: String;
            numberUnit: String;
            numberState: String;
            subtitle: String;
            stateArrow: String;
            info: String;
            service_breakdown: {
                agent_skills: Integer;
                agent_handlers: Integer;
                mcp_tools: Integer;
                database_services: Integer;
                total_services: Integer;
            };
            provider_health: {
                active_providers: Integer;
                total_providers: Integer;
                provider_health_percentage: Integer;
            };
            timestamp: String;
        };
    };
    
    /**
     * Get system health summary
     */
    action getHealthSummary() returns {
        d: {
            title: String;
            number: String;
            numberUnit: String;
            numberState: String;
            subtitle: String;
            stateArrow: String;
            info: String;
            component_health: {
                agents_health: Integer;
                blockchain_health: Integer;
                mcp_health: Integer;
                api_health: Integer;
            };
            system_performance: {
                avg_cpu_usage: Decimal;
                avg_memory_usage: Decimal;
                network_latency: Integer;
            };
            error_tracking: {
                agent_error_rate: Decimal;
                blockchain_tx_failure_rate: Decimal;
                api_error_rate: Decimal;
            };
            timestamp: String;
        };
    };
    
    /**
     * Get Deployment Status for deployment tile
     * @param id - Tile identifier (e.g., 'deployment_tile')
     */
    action getDeploymentStats(id: String) returns {
        d: {
            title: String;
            number: String;
            numberUnit: String;
            numberState: String;
            subtitle: String;
            stateArrow: String;
            info: String;
            deployment_metrics: {
                active_deployments: Integer;
                total_deployments_today: Integer;
                success_rate: Integer;
                avg_deployment_time: Integer;
                production_health: Integer;
                staging_health: Integer;
                last_deployment_status: String;
                last_deployment_time: String;
                failed_deployments_24h: Integer;
            };
            environments: {
                production: {
                    status: String;
                    last_deployment: String;
                    health_score: Integer;
                };
                staging: {
                    status: String;
                    last_deployment: String;
                    health_score: Integer;
                };
            };
            timestamp: String;
        };
    };
}

/**
 * Entity for agent metadata (read-only reference)
 */
entity AgentMetadata {
    key id: Integer;
    name: String;
    port: Integer;
    type: String;
    icon: String;
}
