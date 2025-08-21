/**
 * Real System Event Connector
 * Connects notification system to ACTUAL A2A services - NO MOCKS OR STUBS
 */

const WebSocket = require('ws');
const EventEmitter = require('events');
const cds = require('@sap/cds');
const axios = require('axios');

class RealSystemEventConnector extends EventEmitter {
    constructor() {
        super();
        this.logger = cds.log('real-system-events');
        this.connections = new Map();
        this.isInitialized = false;
        this.retryAttempts = new Map();
        this.maxRetries = 5;
        this.retryDelay = 5000;
        
        // Real service endpoints
        this.services = {
            registry: {
                http: process.env.A2A_REGISTRY_URL || 'http://localhost:8000',
                ws: process.env.A2A_REGISTRY_WS || 'ws://localhost:8000/ws'
            },
            blockchain: {
                http: process.env.BLOCKCHAIN_SERVICE_URL || 'http://localhost:8080/blockchain',
                ws: process.env.BLOCKCHAIN_WS || 'ws://localhost:8080/blockchain/events'
            },
            metrics: {
                ws: process.env.METRICS_WS || 'ws://localhost:3000/ws/metrics'
            },
            security: {
                http: process.env.SECURITY_API_URL || 'http://localhost:8001/security',
                events: process.env.SECURITY_EVENTS_URL || 'http://localhost:8001/security/events'
            }
        };
    }

    async initialize() {
        try {
            this.logger.info('Connecting to real A2A system services...');

            // Connect to agent registry
            await this.connectToAgentRegistry();
            
            // Connect to blockchain service
            await this.connectToBlockchainService();
            
            // Connect to security monitoring
            await this.connectToSecurityMonitoring();
            
            // Connect to metrics service
            await this.connectToMetricsService();

            this.isInitialized = true;
            this.logger.info('✅ Real system event connector initialized');
        } catch (error) {
            this.logger.error('Failed to initialize real system connector:', error);
            throw error;
        }
    }

    async connectToAgentRegistry() {
        try {
            // Test HTTP connection first
            const response = await axios.get(`${this.services.registry.http}/agents`, {
                timeout: 5000,
                headers: { 'Accept': 'application/json' }
            });
            
            this.logger.info(`✅ Connected to agent registry HTTP API (${response.data.agents?.length || 0} agents)`);

            // Connect to registry WebSocket for real-time agent events
            const registryWs = new WebSocket(`${this.services.registry.ws}/agents`);
            
            registryWs.on('open', () => {
                this.logger.info('✅ Connected to agent registry WebSocket');
                this.connections.set('registry', registryWs);
                this.retryAttempts.delete('registry');
            });

            registryWs.on('message', (data) => {
                try {
                    const event = JSON.parse(data);
                    this.handleAgentRegistryEvent(event);
                } catch (error) {
                    this.logger.error('Failed to parse registry event:', error);
                }
            });

            registryWs.on('close', () => {
                this.logger.warn('Agent registry WebSocket closed, attempting reconnect...');
                this.scheduleReconnect('registry', () => this.connectToAgentRegistry());
            });

            registryWs.on('error', (error) => {
                this.logger.error('Agent registry WebSocket error:', error);
            });

        } catch (error) {
            this.logger.error('Failed to connect to agent registry:', error);
            throw error;
        }
    }

    async connectToBlockchainService() {
        try {
            // Test blockchain service availability
            const response = await axios.get(`${this.services.blockchain.http}/status`, {
                timeout: 5000
            });
            
            this.logger.info(`✅ Connected to blockchain service (Network: ${response.data.network}, Block: ${response.data.blockNumber})`);

            // Connect to blockchain WebSocket for transaction events
            const blockchainWs = new WebSocket(this.services.blockchain.ws);
            
            blockchainWs.on('open', () => {
                this.logger.info('✅ Connected to blockchain events WebSocket');
                this.connections.set('blockchain', blockchainWs);
                this.retryAttempts.delete('blockchain');
                
                // Subscribe to transaction events
                blockchainWs.send(JSON.stringify({
                    type: 'subscribe',
                    events: ['transaction.pending', 'transaction.confirmed', 'transaction.failed', 'block.new']
                }));
            });

            blockchainWs.on('message', (data) => {
                try {
                    const event = JSON.parse(data);
                    this.handleBlockchainEvent(event);
                } catch (error) {
                    this.logger.error('Failed to parse blockchain event:', error);
                }
            });

            blockchainWs.on('close', () => {
                this.logger.warn('Blockchain WebSocket closed, attempting reconnect...');
                this.scheduleReconnect('blockchain', () => this.connectToBlockchainService());
            });

            blockchainWs.on('error', (error) => {
                this.logger.error('Blockchain WebSocket error:', error);
            });

        } catch (error) {
            this.logger.error('Failed to connect to blockchain service:', error);
            throw error;
        }
    }

    async connectToSecurityMonitoring() {
        try {
            // Test security API availability
            const response = await axios.get(`${this.services.security.http}/health`, {
                timeout: 5000
            });
            
            this.logger.info('✅ Connected to security monitoring system');

            // Poll security events (since no WebSocket available)
            this.startSecurityEventPolling();

        } catch (error) {
            this.logger.error('Failed to connect to security monitoring:', error);
            // Don't throw - security is optional
        }
    }

    async connectToMetricsService() {
        try {
            // Connect to metrics WebSocket
            const metricsWs = new WebSocket(this.services.metrics.ws);
            
            metricsWs.on('open', () => {
                this.logger.info('✅ Connected to metrics service WebSocket');
                this.connections.set('metrics', metricsWs);
                this.retryAttempts.delete('metrics');
                
                // Subscribe to agent performance events
                metricsWs.send(JSON.stringify({
                    type: 'subscribe',
                    metrics: ['agent.performance', 'agent.status', 'system.alerts']
                }));
            });

            metricsWs.on('message', (data) => {
                try {
                    const event = JSON.parse(data);
                    this.handleMetricsEvent(event);
                } catch (error) {
                    this.logger.error('Failed to parse metrics event:', error);
                }
            });

            metricsWs.on('close', () => {
                this.logger.warn('Metrics WebSocket closed, attempting reconnect...');
                this.scheduleReconnect('metrics', () => this.connectToMetricsService());
            });

            metricsWs.on('error', (error) => {
                this.logger.error('Metrics WebSocket error:', error);
            });

        } catch (error) {
            this.logger.error('Failed to connect to metrics service:', error);
            // Don't throw - metrics is optional for core functionality
        }
    }

    handleAgentRegistryEvent(event) {
        this.logger.debug('Registry event received:', event.type);

        switch (event.type) {
            case 'agent.registered':
                this.emit('agent.connected', {
                    agentId: event.data.agent_id,
                    agentName: event.data.name,
                    ownerId: event.data.registered_by,
                    timestamp: event.timestamp,
                    capabilities: event.data.capabilities,
                    url: event.data.url
                });
                break;

            case 'agent.unregistered':
                this.emit('agent.disconnected', {
                    agentId: event.data.agent_id,
                    agentName: event.data.name,
                    ownerId: event.data.registered_by,
                    timestamp: event.timestamp
                });
                break;

            case 'agent.status_changed':
                this.emit('agent.status_changed', {
                    agentId: event.data.agent_id,
                    oldStatus: event.data.old_status,
                    newStatus: event.data.new_status,
                    timestamp: event.timestamp
                });
                break;

            case 'trust.score_updated':
                this.emit('agent.trust_updated', {
                    agentId: event.data.agent_id,
                    oldScore: event.data.old_score,
                    newScore: event.data.new_score,
                    reason: event.data.reason,
                    timestamp: event.timestamp
                });
                break;
        }
    }

    handleBlockchainEvent(event) {
        this.logger.debug('Blockchain event received:', event.type);

        switch (event.type) {
            case 'transaction.pending':
                this.emit('transaction.pending', {
                    transactionId: event.data.hash,
                    from: event.data.from,
                    to: event.data.to,
                    value: event.data.value,
                    timestamp: event.timestamp
                });
                break;

            case 'transaction.confirmed':
                this.emit('transaction.completed', {
                    transactionId: event.data.hash,
                    userId: event.data.from, // Map blockchain address to user
                    amount: event.data.value,
                    currency: 'ETH',
                    completedAt: event.timestamp,
                    gasUsed: event.data.gasUsed,
                    blockNumber: event.data.blockNumber
                });
                break;

            case 'transaction.failed':
                this.emit('transaction.failed', {
                    transactionId: event.data.hash,
                    userId: event.data.from,
                    error: event.data.error,
                    failedAt: event.timestamp
                });
                break;

            case 'block.new':
                this.emit('system.performance.update', {
                    blockNumber: event.data.number,
                    blockTime: event.data.timestamp,
                    transactionCount: event.data.transactions?.length || 0,
                    gasUsed: event.data.gasUsed,
                    gasLimit: event.data.gasLimit
                });
                break;
        }
    }

    handleMetricsEvent(event) {
        this.logger.debug('Metrics event received:', event.type);

        switch (event.type) {
            case 'agent.performance':
                if (event.data.cpuUsage > 80 || event.data.memoryUsage > 80) {
                    this.emit('system.alert.high_resource_usage', {
                        agentId: event.data.agentId,
                        severity: 'warning',
                        message: `High resource usage: CPU ${event.data.cpuUsage}%, Memory ${event.data.memoryUsage}%`,
                        metrics: event.data
                    });
                }
                break;

            case 'agent.status':
                if (event.data.status === 'offline' || event.data.status === 'error') {
                    this.emit('system.alert.agent_down', {
                        agentId: event.data.agentId,
                        severity: 'high',
                        message: `Agent ${event.data.agentName} is ${event.data.status}`,
                        timestamp: event.timestamp
                    });
                }
                break;

            case 'system.alerts':
                this.emit('system.alert', {
                    title: event.data.title,
                    message: event.data.message,
                    severity: event.data.severity,
                    source: event.data.source,
                    timestamp: event.timestamp,
                    metadata: event.data.metadata
                });
                break;
        }
    }

    startSecurityEventPolling() {
        const pollInterval = 30000; // 30 seconds

        const poll = async () => {
            try {
                const response = await axios.get(`${this.services.security.events}/recent`, {
                    timeout: 5000,
                    params: { since: new Date(Date.now() - pollInterval - 5000).toISOString() }
                });

                if (response.data.events && response.data.events.length > 0) {
                    response.data.events.forEach(event => {
                        this.handleSecurityEvent(event);
                    });
                }
            } catch (error) {
                this.logger.error('Failed to poll security events:', error);
            }
        };

        // Start polling
        setInterval(poll, pollInterval);
        
        // Initial poll
        setTimeout(poll, 1000);
    }

    handleSecurityEvent(event) {
        this.logger.debug('Security event received:', event.event_type);

        this.emit('security.alert', {
            title: event.title || 'Security Alert',
            message: event.description,
            severity: event.threat_level,
            userId: event.user_id,
            source: event.source_ip,
            threat: event.event_type,
            timestamp: event.timestamp,
            metadata: {
                eventId: event.event_id,
                affectedResources: event.affected_resources,
                indicators: event.indicators_of_compromise,
                actions: event.response_actions
            }
        });
    }

    scheduleReconnect(serviceName, reconnectFn) {
        const attempts = this.retryAttempts.get(serviceName) || 0;
        
        if (attempts >= this.maxRetries) {
            this.logger.error(`Max reconnection attempts reached for ${serviceName}`);
            return;
        }

        const delay = this.retryDelay * Math.pow(2, attempts); // Exponential backoff
        this.retryAttempts.set(serviceName, attempts + 1);

        setTimeout(async () => {
            try {
                await reconnectFn();
            } catch (error) {
                this.logger.error(`Reconnection failed for ${serviceName}:`, error);
            }
        }, delay);
    }

    // Real-time data fetchers (replacing stubs)
    async getRealAgentStatuses() {
        try {
            const response = await axios.get(`${this.services.registry.http}/agents`, {
                timeout: 5000
            });

            return response.data.agents.map(agent => ({
                id: agent.agent_id,
                name: agent.name,
                status: agent.status,
                ownerId: agent.registered_by,
                capabilities: agent.capabilities,
                lastSeen: agent.last_seen,
                statusChanged: agent.status !== agent.previous_status,
                connected: agent.status === 'active'
            }));
        } catch (error) {
            this.logger.error('Failed to get real agent statuses:', error);
            return [];
        }
    }

    async getRealPendingTransactions() {
        try {
            const response = await axios.get(`${this.services.blockchain.http}/transactions/pending`, {
                timeout: 5000
            });

            return response.data.transactions.map(tx => ({
                id: tx.hash,
                userId: tx.from,
                amount: tx.value,
                currency: 'ETH',
                status: tx.status,
                completed: tx.status === 'confirmed',
                failed: tx.status === 'failed',
                completedAt: tx.confirmedAt,
                failedAt: tx.failedAt,
                gasUsed: tx.gasUsed,
                blockNumber: tx.blockNumber,
                error: tx.error
            }));
        } catch (error) {
            this.logger.error('Failed to get real pending transactions:', error);
            return [];
        }
    }

    async getRealSecurityEvents() {
        try {
            const response = await axios.get(`${this.services.security.events}/recent`, {
                timeout: 5000,
                params: { limit: 50 }
            });

            return response.data.events.map(event => ({
                type: event.event_type,
                severity: event.threat_level,
                message: event.description,
                userId: event.user_id,
                source: event.source_ip,
                threat: event.event_type,
                timestamp: event.timestamp,
                metadata: {
                    eventId: event.event_id,
                    affectedResources: event.affected_resources,
                    indicators: event.indicators_of_compromise
                }
            }));
        } catch (error) {
            this.logger.error('Failed to get real security events:', error);
            return [];
        }
    }

    // Health check for all connections
    async healthCheck() {
        const health = {
            status: 'healthy',
            services: {},
            timestamp: new Date().toISOString()
        };

        // Check each service
        for (const [serviceName, config] of Object.entries(this.services)) {
            try {
                const connection = this.connections.get(serviceName);
                const wsConnected = connection && connection.readyState === WebSocket.OPEN;
                
                // Try HTTP health check
                let httpHealthy = false;
                try {
                    await axios.get(config.http + '/health', { timeout: 2000 });
                    httpHealthy = true;
                } catch (e) {
                    // HTTP might not be available for all services
                }

                health.services[serviceName] = {
                    websocket: wsConnected,
                    http: httpHealthy,
                    status: wsConnected || httpHealthy ? 'healthy' : 'unhealthy'
                };
            } catch (error) {
                health.services[serviceName] = {
                    websocket: false,
                    http: false,
                    status: 'unhealthy',
                    error: error.message
                };
            }
        }

        const allHealthy = Object.values(health.services).every(s => s.status === 'healthy');
        health.status = allHealthy ? 'healthy' : 'degraded';

        return health;
    }

    shutdown() {
        this.logger.info('Shutting down real system event connector...');

        // Close all WebSocket connections
        for (const [name, connection] of this.connections) {
            if (connection && connection.readyState === WebSocket.OPEN) {
                connection.close();
                this.logger.info(`Closed ${name} connection`);
            }
        }

        this.connections.clear();
        this.retryAttempts.clear();
        this.isInitialized = false;

        this.logger.info('Real system event connector shutdown complete');
    }
}

module.exports = RealSystemEventConnector;