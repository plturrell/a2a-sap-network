/**
 * A2A Agent Portal API Gateway
 * 
 * REST API gateway for integrating existing A2A Network smart contracts
 * with the Agent Portal frontend.
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const { ethers } = require('ethers');
const WebSocket = require('ws');
const { createServer } = require('http');

const { A2AClient } = require('../../sdk/javascript/src/client/A2AClient');
const { validateAddress, validateAgentParams } = require('../../sdk/javascript/src/utils/validation');

class PortalAPIGateway {
    constructor(config) {
        this.config = {
            port: config.port || 3001,
            wsPort: config.wsPort || 3002,
            corsOrigin: config.corsOrigin || ['http://localhost:4004', 'http://localhost:8080'],
            rateLimit: config.rateLimit || 100, // requests per minute
            network: config.network || 'mainnet',
            privateKey: config.privateKey,
            rpcUrl: config.rpcUrl
        };

        // Initialize Express app
        this.app = express();
        this.server = createServer(this.app);
        
        // Initialize WebSocket server
        this.wsServer = new WebSocket.Server({ port: this.config.wsPort });
        
        // A2A Client instance
        this.a2aClient = null;
        
        // WebSocket connections
        this.wsConnections = new Map();
        
        this.setupMiddleware();
        this.setupRoutes();
        this.setupWebSocket();
    }

    setupMiddleware() {
        // Security
        this.app.use(helmet());
        
        // CORS
        this.app.use(cors({
            origin: this.config.corsOrigin,
            credentials: true
        }));
        
        // Rate limiting
        const limiter = rateLimit({
            windowMs: 60 * 1000, // 1 minute
            max: this.config.rateLimit,
            message: { error: 'Too many requests, please try again later' }
        });
        this.app.use(limiter);
        
        // Body parsing
        this.app.use(express.json({ limit: '10mb' }));
        this.app.use(express.urlencoded({ extended: true }));
        
        // Request logging
        this.app.use((req, res, next) => {
            console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
            next();
        });
        
        // Error handling
        this.app.use((err, req, res, next) => {
            console.error('API Error:', err);
            res.status(err.status || 500).json({
                error: err.message || 'Internal server error',
                timestamp: new Date().toISOString()
            });
        });
    }

    setupRoutes() {
        // Health check
        this.app.get('/health', (req, res) => {
            res.json({
                status: 'healthy',
                timestamp: new Date().toISOString(),
                network: this.config.network,
                connected: this.a2aClient?.getConnectionState() === 'connected'
            });
        });

        // Agent management routes
        this.setupAgentRoutes();
        
        // Message routes
        this.setupMessageRoutes();
        
        // Reputation routes
        this.setupReputationRoutes();
        
        // Token routes
        this.setupTokenRoutes();
        
        // Governance routes
        this.setupGovernanceRoutes();
        
        // Analytics routes
        this.setupAnalyticsRoutes();
    }

    setupAgentRoutes() {
        const router = express.Router();

        // Get all agents (paginated)
        router.get('/agents', async (req, res, next) => {
            try {
                const limit = parseInt(req.query.limit) || 20;
                const offset = parseInt(req.query.offset) || 0;
                const search = req.query.search;
                
                let result;
                if (search) {
                    // Search agents by skills or name
                    result = await this.a2aClient.agents.searchAgents({
                        skills: search.split(',').map(s => s.trim()),
                        limit,
                        offset
                    });
                } else {
                    result = await this.a2aClient.agents.getAllAgents(limit, offset);
                }
                
                res.json({
                    success: true,
                    data: result,
                    pagination: {
                        limit,
                        offset,
                        total: result.total
                    }
                });
            } catch (error) {
                next(error);
            }
        });

        // Get agent by ID
        router.get('/agents/:id', async (req, res, next) => {
            try {
                const agentId = req.params.id;
                const agent = await this.a2aClient.agents.getAgent(agentId);
                
                res.json({
                    success: true,
                    data: agent
                });
            } catch (error) {
                next(error);
            }
        });

        // Get agent profile with reputation
        router.get('/agents/:id/profile', async (req, res, next) => {
            try {
                const agentId = req.params.id;
                const profile = await this.a2aClient.agents.getAgentProfile(agentId);
                
                res.json({
                    success: true,
                    data: profile
                });
            } catch (error) {
                next(error);
            }
        });

        // Get agent statistics
        router.get('/agents/:id/stats', async (req, res, next) => {
            try {
                const agentId = req.params.id;
                const stats = await this.a2aClient.agents.getStatistics(agentId);
                
                res.json({
                    success: true,
                    data: stats
                });
            } catch (error) {
                next(error);
            }
        });

        // Register new agent
        router.post('/agents', async (req, res, next) => {
            try {
                const params = req.body;
                const result = await this.a2aClient.agents.register(params);
                
                res.status(201).json({
                    success: true,
                    data: result
                });
                
                // Broadcast new agent registration
                this.broadcastToPortal('agent_registered', {
                    agentId: result.agentId,
                    transactionHash: result.transactionHash
                });
                
            } catch (error) {
                next(error);
            }
        });

        // Update agent
        router.put('/agents/:id', async (req, res, next) => {
            try {
                const agentId = req.params.id;
                const params = req.body;
                const result = await this.a2aClient.agents.update(agentId, params);
                
                res.json({
                    success: true,
                    data: result
                });
                
                // Broadcast agent update
                this.broadcastToPortal('agent_updated', {
                    agentId,
                    transactionHash: result.transactionHash
                });
                
            } catch (error) {
                next(error);
            }
        });

        // Set agent status
        router.patch('/agents/:id/status', async (req, res, next) => {
            try {
                const agentId = req.params.id;
                const { isActive } = req.body;
                const result = await this.a2aClient.agents.setStatus(agentId, isActive);
                
                res.json({
                    success: true,
                    data: result
                });
                
                // Broadcast status change
                this.broadcastToPortal('agent_status_changed', {
                    agentId,
                    isActive,
                    transactionHash: result.transactionHash
                });
                
            } catch (error) {
                next(error);
            }
        });

        // Search agents by criteria
        router.post('/agents/search', async (req, res, next) => {
            try {
                const criteria = req.body;
                const result = await this.a2aClient.agents.searchAgents(criteria);
                
                res.json({
                    success: true,
                    data: result
                });
            } catch (error) {
                next(error);
            }
        });

        // Get agents by owner
        router.get('/agents/owner/:address', async (req, res, next) => {
            try {
                const ownerAddress = req.params.address;
                if (!validateAddress(ownerAddress)) {
                    return res.status(400).json({
                        error: 'Invalid owner address'
                    });
                }
                
                const agents = await this.a2aClient.agents.getAgentsByOwner(ownerAddress);
                
                res.json({
                    success: true,
                    data: agents
                });
            } catch (error) {
                next(error);
            }
        });

        this.app.use('/api/v1', router);
    }

    setupMessageRoutes() {
        const router = express.Router();

        // Send message
        router.post('/messages', async (req, res, next) => {
            try {
                const { recipientId, content, messageType, metadata } = req.body;
                
                const result = await this.a2aClient.messages.send({
                    recipientId,
                    content,
                    messageType: messageType || 'text',
                    metadata: metadata || {}
                });
                
                res.status(201).json({
                    success: true,
                    data: result
                });
                
                // Broadcast new message
                this.broadcastToPortal('message_sent', {
                    messageId: result.messageId,
                    recipientId,
                    transactionHash: result.transactionHash
                });
                
            } catch (error) {
                next(error);
            }
        });

        // Get message history
        router.get('/messages/:agentId', async (req, res, next) => {
            try {
                const agentId = req.params.agentId;
                const limit = parseInt(req.query.limit) || 50;
                const offset = parseInt(req.query.offset) || 0;
                
                const messages = await this.a2aClient.messages.getHistory(agentId, limit, offset);
                
                res.json({
                    success: true,
                    data: messages
                });
            } catch (error) {
                next(error);
            }
        });

        // Get message by ID
        router.get('/messages/by-id/:messageId', async (req, res, next) => {
            try {
                const messageId = req.params.messageId;
                const message = await this.a2aClient.messages.getMessage(messageId);
                
                res.json({
                    success: true,
                    data: message
                });
            } catch (error) {
                next(error);
            }
        });

        this.app.use('/api/v1', router);
    }

    setupReputationRoutes() {
        const router = express.Router();

        // Get agent reputation
        router.get('/reputation/:agentId', async (req, res, next) => {
            try {
                const agentId = req.params.agentId;
                const reputation = await this.a2aClient.reputation.getReputation(agentId);
                
                res.json({
                    success: true,
                    data: reputation
                });
            } catch (error) {
                next(error);
            }
        });

        // Get reputation leaderboard
        router.get('/reputation/leaderboard', async (req, res, next) => {
            try {
                const limit = parseInt(req.query.limit) || 10;
                const leaderboard = await this.a2aClient.reputation.getLeaderboard(limit);
                
                res.json({
                    success: true,
                    data: leaderboard
                });
            } catch (error) {
                next(error);
            }
        });

        this.app.use('/api/v1', router);
    }

    setupTokenRoutes() {
        const router = express.Router();

        // Get token balance
        router.get('/tokens/balance/:address', async (req, res, next) => {
            try {
                const address = req.params.address;
                if (!validateAddress(address)) {
                    return res.status(400).json({
                        error: 'Invalid address'
                    });
                }
                
                const balance = await this.a2aClient.tokens.getBalance(address);
                
                res.json({
                    success: true,
                    data: { balance }
                });
            } catch (error) {
                next(error);
            }
        });

        // Get token information
        router.get('/tokens/info', async (req, res, next) => {
            try {
                const info = await this.a2aClient.tokens.getTokenInfo();
                
                res.json({
                    success: true,
                    data: info
                });
            } catch (error) {
                next(error);
            }
        });

        this.app.use('/api/v1', router);
    }

    setupGovernanceRoutes() {
        const router = express.Router();

        // Get active proposals
        router.get('/governance/proposals', async (req, res, next) => {
            try {
                const status = req.query.status || 'active';
                const proposals = await this.a2aClient.governance.getProposals(status);
                
                res.json({
                    success: true,
                    data: proposals
                });
            } catch (error) {
                next(error);
            }
        });

        // Get proposal by ID
        router.get('/governance/proposals/:id', async (req, res, next) => {
            try {
                const proposalId = req.params.id;
                const proposal = await this.a2aClient.governance.getProposal(proposalId);
                
                res.json({
                    success: true,
                    data: proposal
                });
            } catch (error) {
                next(error);
            }
        });

        this.app.use('/api/v1', router);
    }

    setupAnalyticsRoutes() {
        const router = express.Router();

        // Get network statistics
        router.get('/analytics/network', async (req, res, next) => {
            try {
                const stats = {
                    totalAgents: 0,
                    activeAgents: 0,
                    totalMessages: 0,
                    averageResponseTime: 0,
                    networkUptime: 99.9
                };
                
                // Get real data from contracts
                try {
                    const registryContract = this.a2aClient.getContract('AgentRegistry');
                    stats.totalAgents = await registryContract.getTotalAgents();
                } catch (e) {
                    console.warn('Could not fetch total agents:', e.message);
                }
                
                res.json({
                    success: true,
                    data: stats
                });
            } catch (error) {
                next(error);
            }
        });

        // Get agent activity metrics
        router.get('/analytics/agents/:id/activity', async (req, res, next) => {
            try {
                const agentId = req.params.id;
                const days = parseInt(req.query.days) || 30;
                
                // This would typically come from an analytics service
                const activity = {
                    messagesReceived: [],
                    messagesSent: [],
                    reputationChanges: [],
                    earnings: []
                };
                
                res.json({
                    success: true,
                    data: activity
                });
            } catch (error) {
                next(error);
            }
        });

        this.app.use('/api/v1', router);
    }

    setupWebSocket() {
        this.wsServer.on('connection', (ws, req) => {
            const connectionId = Math.random().toString(36).substr(2, 9);
            this.wsConnections.set(connectionId, ws);
            
            console.log(`WebSocket connection established: ${connectionId}`);
            
            // Send welcome message
            ws.send(JSON.stringify({
                type: 'connection_established',
                connectionId,
                timestamp: new Date().toISOString()
            }));
            
            // Handle incoming messages
            ws.on('message', async (message) => {
                try {
                    const data = JSON.parse(message);
                    await this.handleWebSocketMessage(connectionId, data);
                } catch (error) {
                    console.error('WebSocket message error:', error);
                    ws.send(JSON.stringify({
                        type: 'error',
                        error: error.message
                    }));
                }
            });
            
            // Handle connection close
            ws.on('close', () => {
                console.log(`WebSocket connection closed: ${connectionId}`);
                this.wsConnections.delete(connectionId);
            });
            
            ws.on('error', (error) => {
                console.error(`WebSocket error for ${connectionId}:`, error);
                this.wsConnections.delete(connectionId);
            });
        });
    }

    async handleWebSocketMessage(connectionId, data) {
        const ws = this.wsConnections.get(connectionId);
        if (!ws) return;
        
        switch (data.type) {
            case 'subscribe_agent':
                // Subscribe to specific agent updates
                const agentId = data.agentId;
                if (agentId) {
                    await this.subscribeToAgent(connectionId, agentId);
                }
                break;
                
            case 'subscribe_messages':
                // Subscribe to message updates for an agent
                const messageAgentId = data.agentId;
                if (messageAgentId) {
                    await this.subscribeToMessages(connectionId, messageAgentId);
                }
                break;
                
            case 'ping':
                // Respond to ping with pong
                ws.send(JSON.stringify({
                    type: 'pong',
                    timestamp: new Date().toISOString()
                }));
                break;
                
            default:
                ws.send(JSON.stringify({
                    type: 'error',
                    error: `Unknown message type: ${data.type}`
                }));
        }
    }

    async subscribeToAgent(connectionId, agentId) {
        try {
            const subscriptionId = await this.a2aClient.agents.subscribeToAgent(
                agentId,
                (event) => {
                    const ws = this.wsConnections.get(connectionId);
                    if (ws) {
                        ws.send(JSON.stringify({
                            type: 'agent_event',
                            agentId,
                            event
                        }));
                    }
                }
            );
            
            const ws = this.wsConnections.get(connectionId);
            if (ws) {
                ws.send(JSON.stringify({
                    type: 'subscription_confirmed',
                    subscriptionType: 'agent',
                    agentId,
                    subscriptionId
                }));
            }
        } catch (error) {
            console.error('Error subscribing to agent:', error);
        }
    }

    broadcastToPortal(eventType, data) {
        const message = JSON.stringify({
            type: eventType,
            data,
            timestamp: new Date().toISOString()
        });
        
        this.wsConnections.forEach((ws) => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(message);
            }
        });
    }

    async start() {
        try {
            // Initialize A2A Client
            this.a2aClient = new A2AClient({
                network: this.config.network,
                privateKey: this.config.privateKey,
                rpcUrl: this.config.rpcUrl
            });
            
            await this.a2aClient.connect();
            console.log('A2A Client connected');
            
            // Start HTTP server
            this.server.listen(this.config.port, () => {
                console.log(`Portal API Gateway listening on port ${this.config.port}`);
            });
            
            console.log(`WebSocket server listening on port ${this.config.wsPort}`);
            
            // Setup event subscriptions
            await this.setupEventSubscriptions();
            
            console.log('Portal API Gateway started successfully');
            
        } catch (error) {
            console.error('Failed to start Portal API Gateway:', error);
            throw error;
        }
    }

    async setupEventSubscriptions() {
        // Subscribe to agent events
        await this.a2aClient.agents.subscribeToEvents((event) => {
            this.broadcastToPortal('agent_event', event);
        });
        
        // Subscribe to message events  
        try {
            await this.a2aClient.messages.subscribeToEvents((event) => {
                this.broadcastToPortal('message_event', event);
            });
        } catch (error) {
            console.warn('Could not subscribe to message events:', error.message);
        }
    }

    async stop() {
        try {
            // Close WebSocket connections
            this.wsConnections.forEach((ws) => {
                ws.close();
            });
            this.wsConnections.clear();
            
            // Close WebSocket server
            this.wsServer.close();
            
            // Close HTTP server
            this.server.close();
            
            // Disconnect A2A client
            if (this.a2aClient) {
                await this.a2aClient.disconnect();
            }
            
            console.log('Portal API Gateway stopped');
        } catch (error) {
            console.error('Error stopping Portal API Gateway:', error);
        }
    }
}

module.exports = { PortalAPIGateway };