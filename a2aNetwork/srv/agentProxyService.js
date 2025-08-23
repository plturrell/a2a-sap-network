const cds = require('@sap/cds');
const axios = require('axios');

/**
 * Agent Proxy Service Implementation
 * Handles all agent proxy requests through CAP instead of Express routes
 */
module.exports = cds.service.impl(async function() {
    
    // Agent base URLs configuration
    const AGENT_URLS = {
        agent1: process.env.AGENT1_URL || 'http://localhost:5001',
        agent2: process.env.AGENT2_URL || 'http://localhost:5002',
        agent3: process.env.AGENT3_URL || 'http://localhost:5003',
        agent4: process.env.AGENT4_URL || 'http://localhost:5004',
        agent5: process.env.AGENT5_URL || 'http://localhost:5005',
        agent6: process.env.AGENT6_URL || 'http://localhost:5006',
        agent7: process.env.AGENT7_URL || 'http://localhost:5007',
        agent8: process.env.AGENT8_URL || 'http://localhost:5008',
        agent9: process.env.AGENT9_URL || 'http://localhost:5009',
        agent10: process.env.AGENT10_URL || 'http://localhost:5010',
        agent11: process.env.AGENT11_URL || 'http://localhost:5011',
        agent12: process.env.AGENT12_URL || 'http://localhost:5012',
        agent13: process.env.AGENT13_URL || 'http://localhost:5013',
        agent14: process.env.AGENT14_URL || 'http://localhost:5014',
        agent15: process.env.AGENT15_URL || 'http://localhost:5015'
    };
    
    // Generic proxy handler
    this.on('proxyRequest', async (req) => {
        const { agentId, path, method, body, query } = req.data;
        
        try {
            const agentUrl = AGENT_URLS[agentId];
            if (!agentUrl) {
                throw new Error(`Unknown agent: ${agentId}`);
            }
            
            const config = {
                method: method || 'GET',
                url: `${agentUrl}${path}`,
                headers: {
                    'Content-Type': 'application/json',
                    'X-Forwarded-For': req.headers['x-forwarded-for'] || req.ip,
                    'X-Original-Host': req.headers.host
                }
            };
            
            if (body) {
                config.data = JSON.parse(body);
            }
            
            if (query) {
                config.params = JSON.parse(query);
            }
            
            const response = await axios(config);
            return JSON.stringify(response.data);
            
        } catch (error) {
            console.error(`Proxy error for ${agentId}:`, error.message);
            if (error.response) {
                req.error(error.response.status, error.response.data.message || error.message);
            } else {
                req.error(500, `Agent ${agentId} is not available`);
            }
        }
    });
    
    // Agent health check
    this.on('getAgentHealth', async (req) => {
        const { agentId } = req.data;
        
        try {
            const agentUrl = AGENT_URLS[agentId];
            if (!agentUrl) {
                throw new Error(`Unknown agent: ${agentId}`);
            }
            
            const response = await axios.get(`${agentUrl}/health`, {
                timeout: 5000
            });
            
            return {
                status: 'healthy',
                timestamp: new Date(),
                details: JSON.stringify(response.data)
            };
            
        } catch (error) {
            return {
                status: 'unhealthy',
                timestamp: new Date(),
                details: error.message
            };
        }
    });
    
    // Batch operations
    this.on('executeBatchOperation', async (req) => {
        const { agents, operation, parameters } = req.data;
        
        const results = await Promise.all(
            agents.map(async (agentId) => {
                try {
                    const result = await this.send('proxyRequest', {
                        agentId,
                        path: `/batch/${operation}`,
                        method: 'POST',
                        body: parameters
                    });
                    
                    return {
                        agentId,
                        success: true,
                        result,
                        error: null
                    };
                } catch (error) {
                    return {
                        agentId,
                        success: false,
                        result: null,
                        error: error.message
                    };
                }
            })
        );
        
        return results;
    });
    
    // WebSocket upgrade handler
    this.on('upgradeToWebSocket', async (req) => {
        const { agentId, endpoint } = req.data;
        
        const agentUrl = AGENT_URLS[agentId];
        if (!agentUrl) {
            req.error(404, `Unknown agent: ${agentId}`);
        }
        
        // Generate WebSocket URL
        const wsUrl = agentUrl.replace('http://', 'ws://').replace('https://', 'wss://');
        
        // Generate temporary token for WebSocket authentication
        const token = Buffer.from(`${agentId}:${Date.now()}`).toString('base64');
        
        return {
            wsUrl: `${wsUrl}${endpoint}`,
            token
        };
    });
    
    // OData entity handlers for Agent1
    this.on('READ', 'Agent1Tasks', async (req) => {
        const response = await this.send('proxyRequest', {
            agentId: 'agent1',
            path: '/odata/StandardizationTasks',
            method: 'GET',
            query: JSON.stringify(req.query)
        });
        return JSON.parse(response);
    });
    
    // OData entity handlers for Agent2
    this.on('READ', 'Agent2Tasks', async (req) => {
        const response = await this.send('proxyRequest', {
            agentId: 'agent2',
            path: '/odata/AIPreparationTasks',
            method: 'GET',
            query: JSON.stringify(req.query)
        });
        return JSON.parse(response);
    });
    
    // OData entity handlers for Agent3
    this.on('READ', 'Agent3Tasks', async (req) => {
        const response = await this.send('proxyRequest', {
            agentId: 'agent3',
            path: '/odata/VectorSearchTasks',
            method: 'GET',
            query: JSON.stringify(req.query)
        });
        return JSON.parse(response);
    });
    
    // OData entity handlers for Agent4
    this.on('READ', 'Agent4Tasks', async (req) => {
        const response = await this.send('proxyRequest', {
            agentId: 'agent4',
            path: '/odata/ValidationTasks',
            method: 'GET',
            query: JSON.stringify(req.query)
        });
        return JSON.parse(response);
    });
    
    // OData entity handlers for Agent5
    this.on('READ', 'Agent5Tasks', async (req) => {
        const response = await this.send('proxyRequest', {
            agentId: 'agent5',
            path: '/odata/QaValidationTasks',
            method: 'GET',
            query: JSON.stringify(req.query)
        });
        return JSON.parse(response);
    });
    
    // OData entity handlers for Agent6
    this.on('READ', 'Agent6Tasks', async (req) => {
        const response = await this.send('proxyRequest', {
            agentId: 'agent6',
            path: '/odata/QualityControlTasks',
            method: 'GET',
            query: JSON.stringify(req.query)
        });
        return JSON.parse(response);
    });
    
    // OData entity handlers for Agent7
    this.on('READ', 'Agent7RegisteredAgents', async (req) => {
        const response = await this.send('proxyRequest', {
            agentId: 'agent7',
            path: '/odata/RegisteredAgents',
            method: 'GET',
            query: JSON.stringify(req.query)
        });
        return JSON.parse(response);
    });
    
    this.on('READ', 'Agent7ManagementTasks', async (req) => {
        const response = await this.send('proxyRequest', {
            agentId: 'agent7',
            path: '/odata/ManagementTasks',
            method: 'GET',
            query: JSON.stringify(req.query)
        });
        return JSON.parse(response);
    });
    
    // OData entity handlers for Agent8
    this.on('READ', 'Agent8DataTasks', async (req) => {
        const response = await this.send('proxyRequest', {
            agentId: 'agent8',
            path: '/odata/DataTasks',
            method: 'GET',
            query: JSON.stringify(req.query)
        });
        return JSON.parse(response);
    });
    
    this.on('READ', 'Agent8StorageBackends', async (req) => {
        const response = await this.send('proxyRequest', {
            agentId: 'agent8',
            path: '/odata/StorageBackends',
            method: 'GET',
            query: JSON.stringify(req.query)
        });
        return JSON.parse(response);
    });
});