/**
 * @fileoverview Launchpad Helper Functions
 * Extracted from launchpadService.js for use in REST endpoints
 */

const cds = require('@sap/cds');
const LOG = cds.log('launchpad-helpers');
const fetch = require('node-fetch');

// Agent metadata configuration
const AGENT_METADATA = {
    0: { name: 'Data Product Agent', port: 8000, type: 'Core Processing', icon: 'product' },
    1: { name: 'Data Standardization', port: 8001, type: 'Core Processing', icon: 'synchronize' },
    2: { name: 'AI Preparation', port: 8002, type: 'Core Processing', icon: 'artificial-intelligence' },
    3: { name: 'Vector Processing', port: 8003, type: 'Core Processing', icon: 'scatter-chart' },
    4: { name: 'Calc Validation', port: 8004, type: 'Core Processing', icon: 'validate' },
    5: { name: 'QA Validation', port: 8005, type: 'Core Processing', icon: 'quality-issue' },
    6: { name: 'Quality Control Manager', port: 8006, type: 'Management', icon: 'process' },
    7: { name: 'Agent Manager', port: 8007, type: 'Management', icon: 'org-chart' },
    8: { name: 'Data Manager', port: 8008, type: 'Management', icon: 'database' },
    9: { name: 'Reasoning Agent', port: 8009, type: 'Management', icon: 'decision' },
    10: { name: 'Calculation Agent', port: 8010, type: 'Specialized', icon: 'sum' },
    11: { name: 'SQL Agent', port: 8011, type: 'Specialized', icon: 'table-view' },
    12: { name: 'Catalog Manager', port: 8012, type: 'Specialized', icon: 'course-book' },
    13: { name: 'Agent Builder', port: 8013, type: 'Specialized', icon: 'build' },
    14: { name: 'Embedding Fine-Tuner', port: 8014, type: 'Specialized', icon: 'machine-learning' },
    15: { name: 'Orchestrator Agent', port: 8015, type: 'Specialized', icon: 'workflow-tasks' }
};

// Helper function to check agent health via real HTTP endpoints
async function checkAgentHealth(port) {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);

        const [healthResponse, metricsResponse] = await Promise.all([
            fetch(`http://localhost:${port}/health`, {
                signal: controller.signal,
                headers: { 'Accept': 'application/json' }
            }).catch(() => null),
            fetch(`http://localhost:${port}/metrics`, {
                signal: controller.signal,
                headers: { 'Accept': 'application/json' }
            }).catch(() => null)
        ]);

        clearTimeout(timeoutId);

        if (healthResponse && healthResponse.ok) {
            const healthData = await healthResponse.json();

            let metricsData = {};
            if (metricsResponse && metricsResponse.ok) {
                try {
                    metricsData = await metricsResponse.json();
                } catch (e) {
                    LOG.warn(`Agent ${port} metrics endpoint error:`, e.message);
                }
            }

            return {
                status: 'healthy',
                agent_id: healthData.agent_id,
                name: healthData.name,
                version: healthData.version,
                active_tasks: healthData.active_tasks || 0,
                total_tasks: healthData.total_tasks || 0,
                skills: healthData.skills || 0,
                handlers: healthData.handlers || 0,
                mcp_tools: healthData.mcp_tools || 0,
                mcp_resources: healthData.mcp_resources || 0,
                cpu_usage: metricsData.cpu_percent || healthData.cpu_percent || null,
                memory_usage: metricsData.memory_percent || healthData.memory_percent || null,
                uptime_seconds: metricsData.uptime_seconds || healthData.uptime_seconds || null,
                success_rate: metricsData.success_rate || null,
                avg_response_time_ms: metricsData.avg_response_time_ms || null,
                processed_today: metricsData.processed_today || null,
                error_rate: metricsData.error_rate || null,
                queue_depth: metricsData.queue_depth || null,
                timestamp: healthData.timestamp,
                port: port
            };
        }
        return { status: 'error', message: `HTTP ${healthResponse?.status || 'connection failed'}`, port: port };
    } catch (error) {
        if (error.name === 'AbortError') {
            return { status: 'timeout', message: 'Health check timeout', port: port };
        }
        return { status: 'offline', message: error.message, port: port };
    }
}

// Helper function to check blockchain status
async function checkBlockchainHealth() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);

        const statusResponse = await fetch('http://localhost:8082/blockchain/status', {
            signal: controller.signal,
            headers: { 'Accept': 'application/json' }
        }).catch(() => null);

        clearTimeout(timeoutId);

        if (statusResponse && statusResponse.ok) {
            const statusData = await statusResponse.json();
            return {
                status: 'healthy',
                network: statusData.network,
                contracts: statusData.contracts,
                agents: statusData.agents,
                trust_integration: statusData.trust_integration,
                total_agents_on_chain: statusData.agents || 0,
                timestamp: statusData.timestamp
            };
        }
        return { status: 'error', message: `Registry HTTP ${statusResponse?.status || 'connection failed'}` };
    } catch (error) {
        if (error.name === 'AbortError') {
            return { status: 'timeout', message: 'Blockchain check timeout' };
        }
        return { status: 'offline', message: error.message };
    }
}

// Helper function to check MCP server status
async function checkMcpHealth() {
    try {
        const healthChecks = await Promise.all(
            Object.entries(AGENT_METADATA).map(async ([_id, agent]) => {
                const health = await checkAgentHealth(agent.port);
                return health.status === 'healthy' ? {
                    mcp_tools: health.mcp_tools || 0,
                    mcp_resources: health.mcp_resources || 0
                } : null;
            })
        );

        const healthyMcpAgents = healthChecks.filter(h => h !== null);
        const totalMcpTools = healthyMcpAgents.reduce((sum, agent) => sum + agent.mcp_tools, 0);
        const totalMcpResources = healthyMcpAgents.reduce((sum, agent) => sum + agent.mcp_resources, 0);

        return {
            status: healthyMcpAgents.length > 0 ? 'healthy' : 'offline',
            total_tools: totalMcpTools,
            total_resources: totalMcpResources,
            active_servers: healthyMcpAgents.length,
            timestamp: new Date().toISOString()
        };
    } catch (error) {
        return { status: 'error', message: error.message };
    }
}

module.exports = {
    checkAgentHealth,
    checkBlockchainHealth,
    checkMcpHealth,
    AGENT_METADATA
};