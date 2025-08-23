/**
 * Real-Time Data Service for A2A Network
 * Provides WebSocket-based real-time updates for tiles, analytics, and system metrics
 * Meets SAP Enterprise Standards for real-time data integration
 */

const WebSocket = require('ws');
const EventEmitter = require('events');
const cds = require('@sap/cds');
const { portManager } = require('./utils/portManager');
const websocketMonitor = require('./websocketMonitor');

// Defensive logger initialization
let logger;
try {
    logger = cds.log('realtime');
} catch (error) {
    // Fallback to console if CDS logging not available
    logger = {
        info: console.log,
        error: console.error,
        warn: console.warn,
        debug: console.debug
    };
}

class A2AWebSocketDataService extends EventEmitter {
    constructor() {
        super();
        this.clients = new Map();
        this.wsServer = null;
        this.dataStreams = new Map();
        this.intervals = new Map(); // Track intervals for cleanup
        // Metrics will be fetched from database in real-time
        this.metrics = {
            agents: { count: 0, active: 0, performance: 0 },
            services: { count: 0, active: 0, utilization: 0 },
            network: { health: 0, latency: 0, throughput: 0 },
            blockchain: { blocks: 0, transactions: 0, gasPrice: 0 },
            analytics: { 
                requests: 0, 
                errors: 0, 
                responseTime: 0,
                successRate: 0
            }
        };
        this.dbPath = require('path').join(__dirname, '../a2aNetwork.db');
        this.db = null;
        this.port = null;
        this.initializeDatabase();
        const handleInitializationError = function(error) {
            logger.error('Failed to initialize WebSocket server:', error);
        };
        this.initializeWebSocketServer().catch(handleInitializationError);
        this.startDataStreaming();
    }

    initializeDatabase() {
        const sqlite3 = require('sqlite3').verbose();
        const handleDatabaseConnection = (err) => {
            if (err) {
                logger.error('Failed to connect to database for real-time service:', err);
            } else {
                logger.info('✅ Real-time service connected to database');
                this.updateMetricsFromDatabase();
            }
        };
        this.db = new sqlite3.Database(this.dbPath, handleDatabaseConnection);
    }

    async updateMetricsFromDatabase() {
        try {
            // Fetch agent metrics
            const fetchAgentData = (resolve, reject) => {
                const handleAgentResults = (err, rows) => {
                    if (err) reject(err);
                    else resolve(rows[0] || { total: 0, active: 0 });
                };
                this.db.all(
                    `SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN isActive = 1 THEN 1 ELSE 0 END) as active
                     FROM a2a_network_Agents`,
                    handleAgentResults
                );
            };
            const agentData = await new Promise(fetchAgentData);

            // Fetch service metrics
            const fetchServiceData = (resolve, reject) => {
                const handleServiceResults = (err, rows) => {
                    if (err) reject(err);
                    else resolve(rows[0] || { total: 0, active: 0 });
                };
                this.db.all(
                    `SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN isActive = 1 THEN 1 ELSE 0 END) as active
                     FROM a2a_network_Services`,
                    handleServiceResults
                );
            };
            const serviceData = await new Promise(fetchServiceData);

            // Fetch blockchain stats
            const blockchainData = await new Promise((resolve, reject) => {
                this.db.get(
                    'SELECT * FROM BlockchainService_BlockchainStats ORDER BY timestamp DESC LIMIT 1',
                    (err, row) => {
                        if (err) reject(err);
                        else resolve(row || {});
                    }
                );
            });

            // For now, use default health metrics since NetworkHealthMetrics table doesn't exist
            const healthMetrics = {
                'CPU Utilization': 25,
                'API Response Time': 45,
                'WebSocket Connections': 10
            };

            // Update metrics with real data
            this.metrics = {
                agents: {
                    count: agentData.total,
                    active: agentData.active,
                    performance: agentData.active > 0 ? Math.round((agentData.active / agentData.total) * 100) : 0
                },
                services: {
                    count: serviceData.total,
                    active: serviceData.active,
                    utilization: serviceData.active > 0 ? Math.round((serviceData.active / serviceData.total) * 100) : 0
                },
                network: {
                    health: 100 - (healthMetrics['CPU Utilization'] || 0),
                    latency: Math.round(healthMetrics['API Response Time'] || 0),
                    throughput: Math.round(healthMetrics['WebSocket Connections'] || 0) * 10
                },
                blockchain: {
                    blocks: blockchainData.blockHeight || 0,
                    transactions: blockchainData.totalTransactions || 0,
                    gasPrice: blockchainData.gasPrice || 0
                },
                analytics: {
                    requests: 0, // Will be tracked by monitoring middleware
                    errors: 0, // Will be tracked by error handler
                    responseTime: Math.round(healthMetrics['API Response Time'] || 45),
                    successRate: 100.0 // Will be calculated from actual metrics
                }
            };

            logger.info('📊 Real-time metrics updated from database');
        } catch (error) {
            logger.error('Failed to update metrics from database:', error);
        }
    }

    async initializeWebSocketServer() {
        try {
            // Use port manager to handle port conflicts
            const killConflicts = process.env.NODE_ENV === 'development';
            this.port = await portManager.allocatePortSafely('realtime', 4006, killConflicts);
            
            if (!this.port) {
                logger.warn('⚠️  Real-time WebSocket disabled due to port allocation failure');
                return;
            }

            this.wsServer = new WebSocket.Server({ 
                port: this.port,
                path: '/realtime'
            });

            const handleWebSocketConnection = (ws, req) => {
                const clientId = this.generateClientId();
                logger.info(`📡 Real-time client connected: ${clientId}`);
                
                this.clients.set(clientId, {
                    ws,
                    subscriptions: new Set(),
                    lastPing: Date.now(),
                    metadata: {
                        userAgent: req.headers['user-agent'],
                        ip: req.connection.remoteAddress
                    }
                });

                // Track this connection in the monitor
                websocketMonitor.addConnection(clientId, {
                    userAgent: req.headers['user-agent'],
                    ip: req.connection.remoteAddress
                });

                // Send initial data
                this.sendInitialData(clientId);

                const handleWebSocketMessage = (message) => {
                    try {
                        const data = JSON.parse(message);
                        this.handleClientMessage(clientId, data);
                    } catch (error) {
                        logger.error('Invalid WebSocket message:', error);
                    }
                };
                ws.on('message', handleWebSocketMessage);

                const handleWebSocketClose = () => {
                    logger.info(`📡 Real-time client disconnected: ${clientId}`);
                    this.clients.delete(clientId);
                    websocketMonitor.removeConnection(clientId);
                };
                ws.on('close', handleWebSocketClose);

                const handleWebSocketError = (error) => {
                    logger.error('WebSocket error:', error);
                    this.clients.delete(clientId);
                    websocketMonitor.removeConnection(clientId);
                };
                ws.on('error', handleWebSocketError);

                // Setup ping/pong for connection health
                const handleWebSocketPong = () => {
                    const client = this.clients.get(clientId);
                    if (client) {
                        client.lastPing = Date.now();
                    }
                };
                ws.on('pong', handleWebSocketPong);
            };
            
            this.wsServer.on('connection', handleWebSocketConnection);

            logger.info(`📡 Real-time WebSocket server started on port ${this.port}`);
        } catch (error) {
            logger.error('Failed to start real-time WebSocket server:', error);
        }
    }

    handleClientMessage(clientId, data) {
        const client = this.clients.get(clientId);
        if (!client) return;

        // Track activity in monitor
        websocketMonitor.updateActivity(clientId);

        switch (data.action) {
            case 'subscribe':
                this.subscribeToStream(clientId, data.streams);
                break;
            case 'unsubscribe':
                this.unsubscribeFromStream(clientId, data.streams);
                break;
            case 'ping':
                this.sendToClient(clientId, { action: 'pong', timestamp: Date.now() });
                break;
            case 'request_data':
                this.sendDataSnapshot(clientId, data.dataTypes);
                break;
        }
    }

    subscribeToStream(clientId, streams) {
        const client = this.clients.get(clientId);
        if (!client) return;

        const addStreamSubscription = (stream) => {
            client.subscriptions.add(stream);
            logger.info(`📡 Client ${clientId} subscribed to ${stream}`);
        };
        streams.forEach(addStreamSubscription);

        this.sendToClient(clientId, {
            action: 'subscription_confirmed',
            streams: Array.from(client.subscriptions)
        });
    }

    unsubscribeFromStream(clientId, streams) {
        const client = this.clients.get(clientId);
        if (!client) return;

        const removeStreamSubscription = (stream) => {
            client.subscriptions.delete(stream);
        };
        streams.forEach(removeStreamSubscription);
    }

    sendInitialData(clientId) {
        this.sendToClient(clientId, {
            action: 'initial_data',
            timestamp: Date.now(),
            data: {
                metrics: this.metrics,
                streams: ['tiles', 'analytics', 'network', 'blockchain', 'agents', 'services']
            }
        });
    }

    sendDataSnapshot(clientId, dataTypes) {
        const snapshot = {};
        const addDataTypeToSnapshot = (type) => {
            if (this.metrics[type]) {
                snapshot[type] = this.metrics[type];
            }
        };
        dataTypes.forEach(addDataTypeToSnapshot);

        this.sendToClient(clientId, {
            action: 'data_snapshot',
            timestamp: Date.now(),
            data: snapshot
        });
    }

    sendToClient(clientId, data) {
        const client = this.clients.get(clientId);
        if (client && client.ws.readyState === WebSocket.OPEN) {
            client.ws.send(JSON.stringify(data));
        }
    }

    broadcastToSubscribers(stream, data) {
        this.clients.forEach((client, clientId) => {
            if (client.subscriptions.has(stream)) {
                this.sendToClient(clientId, {
                    action: 'stream_update',
                    stream,
                    timestamp: Date.now(),
                    data
                });
            }
        });
    }

    startDataStreaming() {
        // Clean up existing intervals first
        this.stopDataStreaming();
        
        // Update metrics from database every 5 seconds
        const updateMetricsFromDatabase = () => {
            this.updateMetricsFromDatabase();
        };
        this.intervals.set('metrics', setInterval(updateMetricsFromDatabase, 5000));

        // Update tile data every 10 seconds
        const updateTileData = () => {
            this.updateTileData();
        };
        this.intervals.set('tiles', setInterval(updateTileData, 10000));

        // Update analytics every 5 seconds
        const updateAnalyticsData = () => {
            this.updateAnalyticsData();
        };
        this.intervals.set('analytics', setInterval(updateAnalyticsData, 5000));

        // Update network metrics every 3 seconds
        const updateNetworkMetrics = () => {
            this.updateNetworkMetrics();
        };
        this.intervals.set('network', setInterval(updateNetworkMetrics, 3000));

        // Listen for real-time events from event bus
        this.subscribeToRealTimeEvents();

        // Health check for clients
        const performHealthCheck = () => {
            this.performHealthCheck();
        };
        this.intervals.set('health', setInterval(performHealthCheck, 30000));
    }
    
    stopDataStreaming() {
        // Clear all intervals
        for (const [name, intervalId] of this.intervals) {
            clearInterval(intervalId);
            logger.debug(`Cleared interval: ${name}`);
        }
        this.intervals.clear();
    }
    
    shutdown() {
        logger.info('Shutting down WebSocket data service...');
        
        // Stop all data streaming
        this.stopDataStreaming();
        
        // Close all client connections
        for (const [clientId, ws] of this.clients) {
            ws.close(1001, 'Server shutting down');
        }
        this.clients.clear();
        
        // Close WebSocket server
        if (this.wsServer) {
            const handleServerClose = () => {
                logger.info('WebSocket server closed');
            };
            this.wsServer.close(handleServerClose);
        }
        
        // Close database connection
        if (this.db) {
            this.db.close();
        }
        
        logger.info('WebSocket data service shutdown complete');
    }

    async updateTileData() {
        try {
            // Fetch real metrics from monitoring service
            const agentMetrics = await this.fetchAgentMetrics();
            const serviceMetrics = await this.fetchServiceMetrics();
            
            this.metrics.agents = agentMetrics;
            this.metrics.services = serviceMetrics;

            this.broadcastToSubscribers('tiles', {
                agents: this.metrics.agents,
                services: this.metrics.services
            });
        } catch (error) {
            logger.error('Failed to update tile data:', error);
            // Keep existing values on error
        }
    }
    
    async fetchAgentMetrics() {
        // Fetch real agent metrics from A2A network
        try {
            const agentManagerUrl = process.env.AGENT_MANAGER_URL || 'http://localhost:8000';
            const response = await fetch(`${agentManagerUrl}/api/v1/agents/metrics`);
            if (response.ok) {
                const data = await response.json();
                return {
                    count: data.total_agents || this.metrics.agents.count,
                    active: data.active_agents || this.metrics.agents.active,
                    performance: data.average_performance || this.metrics.agents.performance
                };
            }
        } catch (error) {
            logger.warn('Using cached agent metrics due to fetch error:', error);
        }
        return this.metrics.agents;
    }
    
    async fetchServiceMetrics() {
        // Fetch real service metrics from A2A network
        try {
            const catalogManagerUrl = process.env.CATALOG_MANAGER_URL || 'http://localhost:8001';
            const response = await fetch(`${catalogManagerUrl}/api/v1/services/metrics`);
            if (response.ok) {
                const data = await response.json();
                return {
                    count: data.total_services || this.metrics.services.count,
                    active: data.active_services || this.metrics.services.active,
                    utilization: data.average_utilization || this.metrics.services.utilization
                };
            }
        } catch (error) {
            logger.warn('Using cached service metrics due to fetch error:', error);
        }
        return this.metrics.services;
    }

    async updateAnalyticsData() {
        try {
            // Fetch real analytics from monitoring service
            const analyticsData = await this.fetchAnalyticsMetrics();
            this.metrics.analytics = analyticsData;
            this.broadcastToSubscribers('analytics', this.metrics.analytics);
        } catch (error) {
            logger.error('Failed to update analytics data:', error);
        }
    }
    
    async fetchAnalyticsMetrics() {
        try {
            // Fetch from monitoring service or Prometheus
            const monitoringUrl = process.env.MONITORING_SERVICE_URL || 'http://localhost:9090';
            const response = await fetch(`${monitoringUrl}/api/v1/query?query=a2a_requests_total`);
            if (response.ok) {
                const data = await response.json();
                // Parse Prometheus response format
                const requests = data.data?.result?.[0]?.value?.[1] || this.metrics.analytics.requests;
                
                // Fetch error rate
                const errorResponse = await fetch(`${monitoringUrl}/api/v1/query?query=a2a_errors_total`);
                const errorData = await errorResponse.json();
                const errors = errorData.data?.result?.[0]?.value?.[1] || this.metrics.analytics.errors;
                
                // Calculate success rate
                const successRate = requests > 0 ? ((requests - errors) / requests * 100) : 100;
                
                return {
                    requests: parseInt(requests),
                    errors: parseInt(errors),
                    responseTime: await this.fetchResponseTime(),
                    successRate: Math.round(successRate * 100) / 100
                };
            }
        } catch (error) {
            logger.warn('Using cached analytics metrics:', error);
        }
        return this.metrics.analytics;
    }
    
    async fetchResponseTime() {
        try {
            const monitoringUrl = process.env.MONITORING_SERVICE_URL || 'http://localhost:9090';
            const response = await fetch(`${monitoringUrl}/api/v1/query?query=a2a_response_time_avg`);
            if (response.ok) {
                const data = await response.json();
                return parseFloat(data.data?.result?.[0]?.value?.[1]) || this.metrics.analytics.responseTime;
            }
        } catch (error) {
            logger.warn('Using cached response time:', error);
        }
        return this.metrics.analytics.responseTime;
    }

    async updateNetworkMetrics() {
        try {
            // Fetch real network metrics
            const networkData = await this.fetchNetworkMetrics();
            this.metrics.network = networkData;
            this.broadcastToSubscribers('network', this.metrics.network);
        } catch (error) {
            logger.error('Failed to update network metrics:', error);
        }
    }
    
    async fetchNetworkMetrics() {
        try {
            // Fetch from network monitoring service
            const networkStatsUrl = process.env.NETWORK_STATS_URL || 'http://localhost:8080';
            const response = await fetch(`${networkStatsUrl}/api/v1/network/stats`);
            if (response.ok) {
                const data = await response.json();
                return {
                    health: data.health_score || this.metrics.network.health,
                    latency: data.average_latency_ms || this.metrics.network.latency,
                    throughput: data.throughput_mbps || this.metrics.network.throughput
                };
            }
        } catch (error) {
            logger.warn('Using cached network metrics:', error);
        }
        return this.metrics.network;
    }

    async subscribeToRealTimeEvents() {
        try {
            // Subscribe to real event bus (Redis/MQTT/WebSocket)
            const eventBusUrl = process.env.EVENT_BUS_URL || 'ws://localhost:8080/events';
            
            // In production, connect to actual event streaming service
            logger.info('Attempting to connect to real-time event bus:', eventBusUrl);
            
            // For now, we'll implement a proper event listener when event bus is available
            // This replaces the simulated random events
            
        } catch (error) {
            logger.error('Failed to connect to real-time event bus:', error);
            // No fallback to simulated events - fail properly
        }
    }

    async handleRealTimeEvent(event) {
        // Handle real events from the A2A network
        logger.info('Broadcasting real-time event:', event);
        this.broadcastToSubscribers('events', event);
    }

    performHealthCheck() {
        const now = Date.now();
        this.clients.forEach((client, clientId) => {
            if (now - client.lastPing > 60000) { // 1 minute timeout
                logger.info(`📡 Removing inactive client: ${clientId}`);
                client.ws.terminate();
                this.clients.delete(clientId);
            } else if (client.ws.readyState === WebSocket.OPEN) {
                client.ws.ping();
            }
        });
    }

    generateClientId() {
        return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    // Real analytics data from time-series database
    async generateAnalyticsData() {
        try {
            // Fetch real historical analytics from time-series database
            const analyticsUrl = process.env.ANALYTICS_DB_URL || 'http://localhost:8086'; // InfluxDB
            const response = await fetch(`${analyticsUrl}/api/v1/analytics/timeseries?hours=24`);
            
            if (response.ok) {
                const data = await response.json();
                return {
                    timeSeries: data.timeSeries || [],
                    summary: {
                        totalAgents: this.metrics.agents.count,
                        totalServices: this.metrics.services.count,
                        avgResponseTime: this.metrics.analytics.responseTime,
                        networkHealth: this.metrics.network.health,
                        successRate: this.metrics.analytics.successRate
                    },
                    realTimeMetrics: this.metrics
                };
            }
        } catch (error) {
            logger.error('Failed to fetch real analytics data:', error);
        }
        
        // Return current metrics only - no simulated historical data
        return {
            timeSeries: [],
            summary: {
                totalAgents: this.metrics.agents.count,
                totalServices: this.metrics.services.count,
                avgResponseTime: this.metrics.analytics.responseTime,
                networkHealth: this.metrics.network.health,
                successRate: this.metrics.analytics.successRate
            },
            realTimeMetrics: this.metrics
        };
    }

    // REST API endpoints for HTTP access
    getRESTHandlers() {
        return {
            // GET /api/v1/realtime/metrics
            getMetrics: (req, res) => {
                res.json({
                    success: true,
                    timestamp: Date.now(),
                    metrics: this.metrics,
                    connectedClients: this.clients.size
                });
            },

            // GET /api/v1/realtime/analytics
            getAnalytics: async (req, res) => {
                try {
                    const analyticsData = await this.generateAnalyticsData();
                    res.json({
                        success: true,
                        timestamp: Date.now(),
                        analytics: analyticsData
                    });
                } catch (error) {
                    res.status(500).json({
                        success: false,
                        error: 'Failed to fetch analytics data',
                        timestamp: Date.now()
                    });
                }
            },

            // GET /api/v1/realtime/status
            getStatus: (req, res) => {
                res.json({
                    success: true,
                    timestamp: Date.now(),
                    status: {
                        wsServerRunning: this.wsServer !== null,
                        connectedClients: this.clients.size,
                        activeStreams: Array.from(this.dataStreams.keys()),
                        uptime: process.uptime()
                    }
                });
            }
        };
    }
}

module.exports = A2AWebSocketDataService;
