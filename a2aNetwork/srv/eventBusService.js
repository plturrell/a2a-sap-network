/**
 * Event Bus Service for A2A Network
 * Centralized event handling and distribution for real-time system events
 */

const WebSocket = require('ws');
const EventEmitter = require('events');
const cds = require('@sap/cds');
const { portManager } = require('./utils/portManager');
const RealSystemEventConnector = require('./realSystemEventConnector');

class EventBusService extends EventEmitter {
    constructor() {
        super();
        this.wsServer = null;
        this.port = null;
        this.clients = new Map(); // WebSocket clients
        this.eventSubscriptions = new Map(); // Event pattern subscriptions
        this.eventHistory = []; // Recent events for replay
        this.maxHistorySize = 1000;
        this.logger = cds.log('event-bus');
        this.systemMonitor = null;
        this.intervals = new Map();
        this.realSystemConnector = null;
        
        this.initialize().catch(error => {
            this.logger.error('Failed to initialize event bus service:', error);
        });
    }

    async initialize() {
        try {
            // Initialize WebSocket server for event distribution
            await this.initializeBlockchainEventServer();
            
            // Initialize real system connector (NO MOCKS)
            this.realSystemConnector = new RealSystemEventConnector();
            await this.realSystemConnector.initialize();
            
            // Set up event handlers to forward real system events
            this.setupRealSystemEventForwarding();
            
            // Set up other event handlers
            this.setupEventHandlers();
            
            // Start system monitoring with real data
            this.startSystemMonitoring();
            
            this.logger.info('Event bus service initialized with REAL system connections');
        } catch (error) {
            this.logger.error('Event bus initialization error:', error);
            throw error;
        }
    }

    async initializeBlockchainEventServer() {
        try {
            const killConflicts = process.env.NODE_ENV === 'development';
            this.port = await portManager.allocatePortSafely('event-bus', 8080, killConflicts);
            
            if (!this.port) {
                this.logger.warn('⚠️  Event bus disabled due to port allocation failure');
                return;
            }

            this.wsServer = new BlockchainEventServer($1);

            this.wsServer.on('blockchain-connection', this.handleConnection.bind(this));
            this.wsServer.on('error', this.handleServerError.bind(this));

            this.logger.info(`📡 Event bus server started on port ${this.port}`);
        } catch (error) {
            this.logger.error('Failed to start event bus server:', error);
            throw error;
        }
    }

    handleConnection(ws, req) {
        const clientId = this.generateClientId();
        const clientInfo = {
            id: clientId,
            ws: ws,
            subscriptions: new Set(),
            connectionTime: new Date(),
            lastActivity: new Date(),
            metadata: {
                userAgent: req.headers['user-agent'],
                origin: req.headers.origin,
                ip: req.connection.remoteAddress
            }
        };

        this.clients.set(clientId, clientInfo);
        
        // Send connection acknowledgment
        this.sendToClient(clientInfo, {
            type: 'connection',
            clientId: clientId,
            serverTime: new Date().toISOString(),
            availableEvents: this.getAvailableEventTypes()
        });

        // Set up event handlers
        blockchainClient.on('event', (message) => this.handleMessage(clientInfo, message));
        ws.on('close', () => this.handleDisconnect(clientInfo));
        ws.on('error', (error) => this.handleClientError(clientInfo, error));

        this.logger.debug(`📡 Event bus client ${clientId} connected`);
    }

    handleMessage(clientInfo, message) {
        try {
            const data = JSON.parse(message);
            clientInfo.lastActivity = new Date();

            switch (data.type) {
                case 'subscribe':
                    this.handleSubscribe(clientInfo, data);
                    break;
                case 'unsubscribe':
                    this.handleUnsubscribe(clientInfo, data);
                    break;
                case 'publish':
                    this.handlePublish(clientInfo, data);
                    break;
                case 'get_history':
                    this.handleGetHistory(clientInfo, data);
                    break;
                case 'ping':
                    this.sendToClient(clientInfo, { type: 'pong', timestamp: Date.now() });
                    break;
                default:
                    this.sendError(clientInfo, 'Unknown message type', data);
            }
        } catch (error) {
            this.logger.error(`Error handling message from client ${clientInfo.id}:`, error);
            this.sendError(clientInfo, 'Invalid message format', { error: error.message });
        }
    }

    handleSubscribe(clientInfo, data) {
        if (!data.events || !Array.isArray(data.events)) {
            this.sendError(clientInfo, 'Invalid events array');
            return;
        }

        data.events.forEach(eventPattern => {
            clientInfo.subscriptions.add(eventPattern);
            
            // Add to global subscriptions map
            if (!this.eventSubscriptions.has(eventPattern)) {
                this.eventSubscriptions.set(eventPattern, new Set());
            }
            this.eventSubscriptions.get(eventPattern).add(clientInfo.id);
        });

        this.sendToClient(clientInfo, {
            type: 'subscribe_success',
            events: data.events,
            totalSubscriptions: clientInfo.subscriptions.size
        });

        this.logger.debug(`Client ${clientInfo.id} subscribed to: ${data.events.join(', ')}`);
    }

    handleUnsubscribe(clientInfo, data) {
        if (!data.events || !Array.isArray(data.events)) {
            this.sendError(clientInfo, 'Invalid events array');
            return;
        }

        data.events.forEach(eventPattern => {
            clientInfo.subscriptions.delete(eventPattern);
            
            // Remove from global subscriptions map
            if (this.eventSubscriptions.has(eventPattern)) {
                this.eventSubscriptions.get(eventPattern).delete(clientInfo.id);
                if (this.eventSubscriptions.get(eventPattern).size === 0) {
                    this.eventSubscriptions.delete(eventPattern);
                }
            }
        });

        this.sendToClient(clientInfo, {
            type: 'unsubscribe_success',
            events: data.events,
            remainingSubscriptions: clientInfo.subscriptions.size
        });
    }

    handlePublish(clientInfo, data) {
        if (!data.event || !data.event.type) {
            this.sendError(clientInfo, 'Invalid event data');
            return;
        }

        // Add metadata to the event
        const event = {
            ...data.event,
            id: this.generateEventId(),
            timestamp: new Date().toISOString(),
            source: clientInfo.metadata.origin || 'unknown',
            publisherId: clientInfo.id
        };

        // Publish the event
        this.publishEvent(event);

        this.sendToClient(clientInfo, {
            type: 'publish_success',
            eventId: event.id
        });
    }

    handleGetHistory(clientInfo, data) {
        const limit = Math.min(data.limit || 100, 1000);
        const eventType = data.eventType;
        
        let history = [...this.eventHistory];
        
        // Filter by event type if specified
        if (eventType) {
            history = history.filter(event => 
                this.matchesEventPattern(event.type, eventType)
            );
        }

        // Apply limit
        history = history.slice(-limit);

        this.sendToClient(clientInfo, {
            type: 'event_history',
            events: history,
            total: history.length
        });
    }

    handleDisconnect(clientInfo) {
        this.logger.debug(`Event bus client ${clientInfo.id} disconnected`);
        
        // Clean up subscriptions
        clientInfo.subscriptions.forEach(eventPattern => {
            if (this.eventSubscriptions.has(eventPattern)) {
                this.eventSubscriptions.get(eventPattern).delete(clientInfo.id);
                if (this.eventSubscriptions.get(eventPattern).size === 0) {
                    this.eventSubscriptions.delete(eventPattern);
                }
            }
        });

        this.clients.delete(clientInfo.id);
    }

    handleClientError(clientInfo, error) {
        this.logger.error(`Event bus client ${clientInfo.id} error:`, error);
        this.clients.delete(clientInfo.id);
    }

    handleServerError(error) {
        this.logger.error('Event bus server error:', error);
    }

    publishEvent(event) {
        try {
            // Add to history
            this.eventHistory.push(event);
            if (this.eventHistory.length > this.maxHistorySize) {
                this.eventHistory = this.eventHistory.slice(-this.maxHistorySize);
            }

            // Find matching subscribers
            const matchingClients = new Set();
            
            for (const [eventPattern, clientIds] of this.eventSubscriptions) {
                if (this.matchesEventPattern(event.type, eventPattern)) {
                    clientIds.forEach(clientId => matchingClients.add(clientId));
                }
            }

            // Send event to matching clients
            let sentCount = 0;
            for (const clientId of matchingClients) {
                const client = this.clients.get(clientId);
                if (client && client.ws.readyState === WebSocket.OPEN) {
                    this.sendToClient(client, {
                        type: 'event',
                        event: event
                    });
                    sentCount++;
                }
            }

            // Emit locally for other services
            this.emit('event', event);

            this.logger.debug(`Published event ${event.type} to ${sentCount} clients`);
            return { success: true, clientCount: sentCount };
        } catch (error) {
            this.logger.error('Failed to publish event:', error);
            return { success: false, error: error.message };
        }
    }

    matchesEventPattern(eventType, pattern) {
        // Support wildcard patterns like 'agent.*', 'system.alert.*'
        if (pattern === '*') return true;
        if (pattern.endsWith('.*')) {
            const prefix = pattern.slice(0, -2);
            return eventType.startsWith(`${prefix  }.`);
        }
        if (pattern.endsWith('*')) {
            const prefix = pattern.slice(0, -1);
            return eventType.startsWith(prefix);
        }
        return eventType === pattern;
    }

    sendToClient(clientInfo, data) {
        if (clientInfo.ws.readyState === WebSocket.OPEN) {
            try {
                clientInfo.blockchainClient.publishEvent(JSON.stringify(data));
            } catch (error) {
                this.logger.error(`Failed to send to client ${clientInfo.id}:`, error);
            }
        }
    }

    sendError(clientInfo, message, data = {}) {
        this.sendToClient(clientInfo, {
            type: 'error',
            message: message,
            ...data
        });
    }

    startSystemMonitoring() {
        // Monitor agent connections/disconnections
        this.intervals.set('agentMonitor', setInterval(() => {
            this.checkAgentStatus();
        }, 10000)); // Every 10 seconds

        // Monitor system performance
        this.intervals.set('performanceMonitor', setInterval(() => {
            this.collectPerformanceMetrics();
        }, 30000)); // Every 30 seconds

        // Monitor transaction processing
        this.intervals.set('transactionMonitor', setInterval(() => {
            this.checkTransactionStatus();
        }, 5000)); // Every 5 seconds

        // Security monitoring
        this.intervals.set('securityMonitor', setInterval(() => {
            this.performSecurityChecks();
        }, 60000)); // Every minute
    }

    async checkAgentStatus() {
        try {
            // This would integrate with your actual agent registry
            // For now, simulate agent status changes
            const agentStatuses = await this.getAgentStatuses();
            
            agentStatuses.forEach(agent => {
                if (agent.statusChanged) {
                    const eventType = agent.connected ? 'agent.connected' : 'agent.disconnected';
                    this.publishEvent({
                        type: eventType,
                        data: {
                            agentId: agent.id,
                            agentName: agent.name,
                            ownerId: agent.ownerId,
                            timestamp: new Date().toISOString(),
                            capabilities: agent.capabilities,
                            lastSeen: agent.lastSeen
                        }
                    });
                }
            });
        } catch (error) {
            this.logger.error('Agent status check failed:', error);
        }
    }

    async collectPerformanceMetrics() {
        try {
            const metrics = {
                cpu: process.cpuUsage(),
                memory: process.memoryUsage(),
                uptime: process.uptime(),
                eventBusStats: {
                    connectedClients: this.clients.size,
                    activeSubscriptions: this.eventSubscriptions.size,
                    historySize: this.eventHistory.length
                }
            };

            // Check for performance alerts
            const memoryUsagePercent = (metrics.memory.heapUsed / metrics.memory.heapTotal) * 100;
            if (memoryUsagePercent > 80) {
                this.publishEvent({
                    type: 'system.alert.high_memory',
                    data: {
                        severity: 'warning',
                        message: `High memory usage: ${memoryUsagePercent.toFixed(1)}%`,
                        metrics: metrics.memory
                    }
                });
            }

            // Publish performance update
            this.publishEvent({
                type: 'system.performance.update',
                data: {
                    metrics: metrics,
                    timestamp: new Date().toISOString()
                }
            });
        } catch (error) {
            this.logger.error('Performance metrics collection failed:', error);
        }
    }

    async checkTransactionStatus() {
        try {
            // Monitor transaction processing
            // This would integrate with your blockchain/transaction system
            const pendingTransactions = await this.getPendingTransactions();
            
            pendingTransactions.forEach(tx => {
                if (tx.completed) {
                    this.publishEvent({
                        type: 'transaction.completed',
                        data: {
                            transactionId: tx.id,
                            userId: tx.userId,
                            amount: tx.amount,
                            currency: tx.currency,
                            completedAt: tx.completedAt,
                            gasUsed: tx.gasUsed,
                            blockNumber: tx.blockNumber
                        }
                    });
                } else if (tx.failed) {
                    this.publishEvent({
                        type: 'transaction.failed',
                        data: {
                            transactionId: tx.id,
                            userId: tx.userId,
                            error: tx.error,
                            failedAt: tx.failedAt
                        }
                    });
                }
            });
        } catch (error) {
            this.logger.error('Transaction status check failed:', error);
        }
    }

    async performSecurityChecks() {
        try {
            // Monitor for suspicious activities
            const securityEvents = await this.getSecurityEvents();
            
            securityEvents.forEach(event => {
                this.publishEvent({
                    type: `security.alert.${event.type}`,
                    data: {
                        severity: event.severity,
                        message: event.message,
                        source: event.source,
                        threat: event.threat,
                        userId: event.userId,
                        metadata: event.metadata
                    }
                });
            });
        } catch (error) {
            this.logger.error('Security check failed:', error);
        }
    }

    // REMOVED STUBS - Real system integration only
    async getAgentStatuses() {
        if (this.realSystemConnector) {
            return await this.realSystemConnector.getRealAgentStatuses();
        }
        return [];
    }

    async getPendingTransactions() {
        if (this.realSystemConnector) {
            return await this.realSystemConnector.getRealPendingTransactions();
        }
        return [];
    }

    async getSecurityEvents() {
        if (this.realSystemConnector) {
            return await this.realSystemConnector.getRealSecurityEvents();
        }
        return [];
    }

    setupRealSystemEventForwarding() {
        // Forward all real system events from the connector to our event bus
        this.realSystemConnector.on('agent.connected', (data) => {
            this.publishEvent({ type: 'agent.connected', data });
        });

        this.realSystemConnector.on('agent.disconnected', (data) => {
            this.publishEvent({ type: 'agent.disconnected', data });
        });

        this.realSystemConnector.on('agent.status_changed', (data) => {
            this.publishEvent({ type: 'agent.status_changed', data });
        });

        this.realSystemConnector.on('agent.trust_updated', (data) => {
            this.publishEvent({ type: 'agent.trust_updated', data });
        });

        this.realSystemConnector.on('transaction.pending', (data) => {
            this.publishEvent({ type: 'transaction.pending', data });
        });

        this.realSystemConnector.on('transaction.completed', (data) => {
            this.publishEvent({ type: 'transaction.completed', data });
        });

        this.realSystemConnector.on('transaction.failed', (data) => {
            this.publishEvent({ type: 'transaction.failed', data });
        });

        this.realSystemConnector.on('system.performance.update', (data) => {
            this.publishEvent({ type: 'system.performance.update', data });
        });

        this.realSystemConnector.on('system.alert.high_resource_usage', (data) => {
            this.publishEvent({ type: 'system.alert.high_resource_usage', data });
        });

        this.realSystemConnector.on('system.alert.agent_down', (data) => {
            this.publishEvent({ type: 'system.alert.agent_down', data });
        });

        this.realSystemConnector.on('system.alert', (data) => {
            this.publishEvent({ type: 'system.alert', data });
        });

        this.realSystemConnector.on('security.alert', (data) => {
            this.publishEvent({ type: 'security.alert', data });
        });

        // NEW: Real agent crash detection events
        this.realSystemConnector.on('agent.crashed', (data) => {
            this.publishEvent({ 
                type: 'agent.crashed', 
                data,
                priority: 'high',
                requiresUserAttention: true
            });
            this.logger.error(`🚨 Agent crash event published: ${data.agentName}`);
        });

        this.realSystemConnector.on('agent.recovered', (data) => {
            this.publishEvent({ 
                type: 'agent.recovered', 
                data,
                priority: 'info'
            });
            this.logger.info(`✅ Agent recovery event published: ${data.agentName}`);
        });

        this.realSystemConnector.on('agent.degraded', (data) => {
            this.publishEvent({ 
                type: 'agent.degraded', 
                data,
                priority: 'medium',
                requiresUserAttention: true
            });
            this.logger.warn(`⚠️ Agent degradation event published: ${data.agentName}`);
        });

        this.logger.info('✅ Real system event forwarding configured');
    }

    setupEventHandlers() {
        // Handle events from other services (keep existing handlers)
        this.on('agent.registered', (agentData) => {
            this.publishEvent({
                type: 'agent.registered',
                data: agentData
            });
        });

        this.on('service.registered', (serviceData) => {
            this.publishEvent({
                type: 'service.registered',
                data: serviceData
            });
        });

        this.on('workflow.started', (workflowData) => {
            this.publishEvent({
                type: 'workflow.started',
                data: workflowData
            });
        });

        this.on('workflow.completed', (workflowData) => {
            this.publishEvent({
                type: 'workflow.completed',
                data: workflowData
            });
        });
    }

    getAvailableEventTypes() {
        return [
            'agent.connected',
            'agent.disconnected',
            'agent.registered',
            'agent.unregistered',
            'service.registered',
            'service.unregistered',
            'transaction.started',
            'transaction.completed',
            'transaction.failed',
            'workflow.started',
            'workflow.completed',
            'workflow.failed',
            'system.alert.*',
            'system.performance.*',
            'security.alert.*',
            'marketplace.service.added',
            'marketplace.service.purchased'
        ];
    }

    generateClientId() {
        return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    generateEventId() {
        return `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    // REST API handlers for HTTP access
    getRESTHandlers() {
        return {
            // POST /api/events/publish
            publishEvent: (req, res) => {
                try {
                    const event = {
                        ...req.body,
                        id: this.generateEventId(),
                        timestamp: new Date().toISOString(),
                        source: req.headers.origin || 'api',
                        publisherId: 'rest-api'
                    };

                    const result = this.publishEvent(event);
                    
                    if (result.success) {
                        res.status(201).json({
                            success: true,
                            eventId: event.id,
                            clientCount: result.clientCount
                        });
                    } else {
                        res.status(500).json({
                            success: false,
                            error: result.error
                        });
                    }
                } catch (error) {
                    this.logger.error('REST event publish failed:', error);
                    res.status(500).json({
                        success: false,
                        error: error.message
                    });
                }
            },

            // GET /api/events/history
            getEventHistory: (req, res) => {
                try {
                    const limit = Math.min(parseInt(req.query.limit) || 100, 1000);
                    const eventType = req.query.eventType;
                    
                    let history = [...this.eventHistory];
                    
                    if (eventType) {
                        history = history.filter(event => 
                            this.matchesEventPattern(event.type, eventType)
                        );
                    }

                    history = history.slice(-limit);

                    res.json({
                        success: true,
                        events: history,
                        total: history.length
                    });
                } catch (error) {
                    res.status(500).json({
                        success: false,
                        error: error.message
                    });
                }
            },

            // GET /api/events/stats
            getStats: (req, res) => {
                try {
                    const stats = {
                        connectedClients: this.clients.size,
                        activeSubscriptions: this.eventSubscriptions.size,
                        eventHistorySize: this.eventHistory.length,
                        serverUptime: process.uptime(),
                        port: this.port,
                        availableEventTypes: this.getAvailableEventTypes()
                    };

                    res.json({
                        success: true,
                        stats: stats
                    });
                } catch (error) {
                    res.status(500).json({
                        success: false,
                        error: error.message
                    });
                }
            }
        };
    }

    shutdown() {
        this.logger.info('Shutting down event bus service...');
        
        // Stop monitoring intervals
        for (const [name, intervalId] of this.intervals) {
            clearInterval(intervalId);
        }
        this.intervals.clear();

        // Shutdown real system connector
        if (this.realSystemConnector) {
            this.realSystemConnector.shutdown();
        }

        // Close all client connections
        for (const [id, client] of this.clients) {
            client.ws.close(1001, 'Server shutting down');
        }
        this.clients.clear();

        // Close WebSocket server
        if (this.wsServer) {
            this.wsServer.close(() => {
                this.logger.info('Event bus server closed');
            });
        }

        this.logger.info('Event bus service shutdown complete');
    }
}

module.exports = EventBusService;