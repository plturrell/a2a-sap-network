/**
 * Enhanced Notification Service with WebSocket Resilience
 * Provides robust real-time notifications with automatic reconnection and error recovery
 */

const WebSocket = require('ws');
const EventEmitter = require('events');
const cds = require('@sap/cds');
const { portManager } = require('./utils/portManager');
const NotificationPersistenceService = require('./notificationPersistenceService');

class EnhancedNotificationService extends EventEmitter {
    constructor() {
        super();
        this.wsServer = null;
        this.port = null;
        this.clients = new Map(); // Track clients with metadata
        this.persistenceService = new NotificationPersistenceService();
        this.logger = cds.log('enhanced-notifications');
        this.heartbeatInterval = 30000; // 30 seconds
        this.reconnectWindow = 5000; // 5 seconds for client to reconnect
        this.eventBusClient = null;
        this.intervals = new Map();

        this.initialize().catch(error => {
            this.logger.error('Failed to initialize enhanced notification service:', error);
        });
    }

    async initialize() {
        try {
            // Initialize persistence service
            await this.persistenceService.init();

            // Initialize WebSocket server
            await this.initializeBlockchainEventServer();

            // Connect to event bus
            await this.connectToEventBus();

            // Start background tasks
            this.startBackgroundTasks();

            this.logger.info('Enhanced notification service initialized successfully');
        } catch (error) {
            this.logger.error('Initialization error:', error);
            throw error;
        }
    }

    async initializeBlockchainEventServer() {
        try {
            const killConflicts = process.env.NODE_ENV === 'development';
            this.port = await portManager.allocatePortSafely('enhanced-notifications', 4006, killConflicts);

            if (!this.port) {
                this.logger.warn('⚠️  WebSocket notifications disabled due to port allocation failure');
                return;
            }

            this.wsServer = new BlockchainEventServer($1);

            this.wsServer.on('blockchain-connection', this.handleConnection.bind(this));
            this.wsServer.on('error', this.handleServerError.bind(this));

            this.logger.info(`🔔 Enhanced WebSocket server started on port ${this.port}`);
        } catch (error) {
            this.logger.error('Failed to start WebSocket server:', error);
            throw error;
        }
    }

    async handleConnection(ws, req) {
        const clientId = this.generateClientId();
        const clientInfo = {
            id: clientId,
            ws: ws,
            userId: null,
            isAlive: true,
            subscribedCategories: [],
            connectionTime: new Date(),
            lastActivity: new Date(),
            reconnectToken: this.generateReconnectToken(),
            messageQueue: [],
            state: 'connected'
        };

        this.clients.set(clientId, clientInfo);

        // Send connection acknowledgment
        this.sendToClient(clientInfo, {
            type: 'connection',
            clientId: clientId,
            reconnectToken: clientInfo.reconnectToken,
            serverTime: new Date().toISOString()
        });

        // Set up event handlers
        blockchainClient.on('event', (message) => this.handleMessage(clientInfo, message));
        ws.on('close', (code, reason) => this.handleDisconnect(clientInfo, code, reason));
        ws.on('error', (error) => this.handleClientError(clientInfo, error));
        ws.on('pong', () => this.handlePong(clientInfo));

        this.logger.debug(`📱 Client ${clientId} connected`);
    }

    async handleMessage(clientInfo, message) {
        try {
            const data = JSON.parse(message);
            clientInfo.lastActivity = new Date();

            switch (data.type) {
                case 'auth':
                    await this.handleAuth(clientInfo, data);
                    break;
                case 'reconnect':
                    await this.handleReconnect(clientInfo, data);
                    break;
                case 'subscribe':
                    await this.handleSubscribe(clientInfo, data);
                    break;
                case 'unsubscribe':
                    await this.handleUnsubscribe(clientInfo, data);
                    break;
                case 'mark_read':
                    await this.handleMarkRead(clientInfo, data);
                    break;
                case 'mark_all_read':
                    await this.handleMarkAllRead(clientInfo);
                    break;
                case 'dismiss':
                    await this.handleDismiss(clientInfo, data);
                    break;
                case 'get_notifications':
                    await this.handleGetNotifications(clientInfo, data);
                    break;
                case 'update_preferences':
                    await this.handleUpdatePreferences(clientInfo, data);
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

    async handleAuth(clientInfo, data) {
        if (!data.userId) {
            this.sendError(clientInfo, 'Missing userId');
            return;
        }

        clientInfo.userId = data.userId;
        clientInfo.state = 'authenticated';

        // Load user preferences
        const preferences = await this.persistenceService.getUserPreferences(data.userId);

        // Send recent notifications
        const notifications = await this.persistenceService.getNotifications(data.userId, {
            limit: 20,
            status: 'unread'
        });

        this.sendToClient(clientInfo, {
            type: 'auth_success',
            userId: data.userId,
            preferences: preferences,
            notifications: notifications,
            unreadCount: notifications.length
        });

        // Deliver any queued messages
        this.deliverQueuedMessages(clientInfo);

        this.logger.info(`Client ${clientInfo.id} authenticated as user ${data.userId}`);
    }

    async handleReconnect(clientInfo, data) {
        if (!data.reconnectToken || !data.userId) {
            this.sendError(clientInfo, 'Invalid reconnect data');
            return;
        }

        // Find previous client session
        let oldClient = null;
        for (const [id, client] of this.clients) {
            if (client.reconnectToken === data.reconnectToken &&
                client.userId === data.userId &&
                client.id !== clientInfo.id) {
                oldClient = client;
                break;
            }
        }

        if (oldClient) {
            // Transfer state from old client
            clientInfo.userId = oldClient.userId;
            clientInfo.subscribedCategories = oldClient.subscribedCategories;
            clientInfo.messageQueue = oldClient.messageQueue;
            clientInfo.state = 'authenticated';

            // Remove old client
            this.clients.delete(oldClient.id);

            // Send reconnect success with queued messages
            this.sendToClient(clientInfo, {
                type: 'reconnect_success',
                queuedMessages: clientInfo.messageQueue
            });

            // Clear the queue after sending
            clientInfo.messageQueue = [];

            this.logger.info(`Client ${clientInfo.id} reconnected as user ${data.userId}`);
        } else {
            // Treat as new auth
            await this.handleAuth(clientInfo, data);
        }
    }

    async handleSubscribe(clientInfo, data) {
        if (!data.categories || !Array.isArray(data.categories)) {
            this.sendError(clientInfo, 'Invalid categories');
            return;
        }

        clientInfo.subscribedCategories = [...new Set([...clientInfo.subscribedCategories, ...data.categories])];

        this.sendToClient(clientInfo, {
            type: 'subscribe_success',
            categories: clientInfo.subscribedCategories
        });
    }

    async handleUnsubscribe(clientInfo, data) {
        if (!data.categories || !Array.isArray(data.categories)) {
            this.sendError(clientInfo, 'Invalid categories');
            return;
        }

        clientInfo.subscribedCategories = clientInfo.subscribedCategories.filter(
            cat => !data.categories.includes(cat)
        );

        this.sendToClient(clientInfo, {
            type: 'unsubscribe_success',
            categories: clientInfo.subscribedCategories
        });
    }

    async handleMarkRead(clientInfo, data) {
        if (!clientInfo.userId || !data.notificationId) {
            this.sendError(clientInfo, 'Invalid request');
            return;
        }

        try {
            const success = await this.persistenceService.markAsRead(clientInfo.userId, data.notificationId);

            if (success) {
                // Broadcast to all user's clients
                this.broadcastToUser(clientInfo.userId, {
                    type: 'notification_read',
                    notificationId: data.notificationId
                });
            }
        } catch (error) {
            this.sendError(clientInfo, 'Failed to mark as read', { error: error.message });
        }
    }

    async handleMarkAllRead(clientInfo) {
        if (!clientInfo.userId) {
            this.sendError(clientInfo, 'Not authenticated');
            return;
        }

        try {
            const count = await this.persistenceService.markAllAsRead(clientInfo.userId);

            // Broadcast to all user's clients
            this.broadcastToUser(clientInfo.userId, {
                type: 'all_notifications_read',
                count: count
            });
        } catch (error) {
            this.sendError(clientInfo, 'Failed to mark all as read', { error: error.message });
        }
    }

    async handleDismiss(clientInfo, data) {
        if (!clientInfo.userId || !data.notificationId) {
            this.sendError(clientInfo, 'Invalid request');
            return;
        }

        try {
            const success = await this.persistenceService.dismissNotification(clientInfo.userId, data.notificationId);

            if (success) {
                // Broadcast to all user's clients
                this.broadcastToUser(clientInfo.userId, {
                    type: 'notification_dismissed',
                    notificationId: data.notificationId
                });
            }
        } catch (error) {
            this.sendError(clientInfo, 'Failed to dismiss notification', { error: error.message });
        }
    }

    async handleGetNotifications(clientInfo, data) {
        if (!clientInfo.userId) {
            this.sendError(clientInfo, 'Not authenticated');
            return;
        }

        try {
            const options = {
                status: data.status,
                type: data.type,
                priority: data.priority,
                category: data.category,
                limit: data.limit || 50,
                offset: data.offset || 0
            };

            const notifications = await this.persistenceService.getNotifications(clientInfo.userId, options);
            const stats = await this.persistenceService.getNotificationStats(clientInfo.userId);

            this.sendToClient(clientInfo, {
                type: 'notifications',
                notifications: notifications,
                stats: stats,
                options: options
            });
        } catch (error) {
            this.sendError(clientInfo, 'Failed to get notifications', { error: error.message });
        }
    }

    async handleUpdatePreferences(clientInfo, data) {
        if (!clientInfo.userId || !data.preferences) {
            this.sendError(clientInfo, 'Invalid request');
            return;
        }

        try {
            const updated = await this.persistenceService.updateUserPreferences(clientInfo.userId, data.preferences);

            this.sendToClient(clientInfo, {
                type: 'preferences_updated',
                preferences: updated
            });
        } catch (error) {
            this.sendError(clientInfo, 'Failed to update preferences', { error: error.message });
        }
    }

    handleDisconnect(clientInfo, code, reason) {
        this.logger.debug(`Client ${clientInfo.id} disconnected: ${code} - ${reason}`);

        if (clientInfo.state === 'authenticated') {
            // Keep client info for reconnection window
            clientInfo.state = 'disconnected';
            clientInfo.disconnectTime = new Date();

            // Schedule cleanup
            setTimeout(() => {
                if (this.clients.has(clientInfo.id) && clientInfo.state === 'disconnected') {
                    this.clients.delete(clientInfo.id);
                    this.logger.debug(`Removed disconnected client ${clientInfo.id}`);
                }
            }, this.reconnectWindow);
        } else {
            // Remove immediately if not authenticated
            this.clients.delete(clientInfo.id);
        }
    }

    handleClientError(clientInfo, error) {
        this.logger.error(`Client ${clientInfo.id} error:`, error);
        clientInfo.ws.terminate();
    }

    handleServerError(error) {
        this.logger.error('WebSocket server error:', error);
    }

    handlePong(clientInfo) {
        clientInfo.isAlive = true;
        clientInfo.lastActivity = new Date();
    }

    sendToClient(clientInfo, data) {
        if (clientInfo.ws.readyState === WebSocket.OPEN) {
            try {
                clientInfo.blockchainClient.publishEvent(JSON.stringify(data));
            } catch (error) {
                this.logger.error(`Failed to send to client ${clientInfo.id}:`, error);
                this.queueMessage(clientInfo, data);
            }
        } else {
            this.queueMessage(clientInfo, data);
        }
    }

    queueMessage(clientInfo, data) {
        if (clientInfo.state === 'authenticated') {
            clientInfo.messageQueue.push({
                ...data,
                queuedAt: new Date().toISOString()
            });

            // Limit queue size
            if (clientInfo.messageQueue.length > 100) {
                clientInfo.messageQueue = clientInfo.messageQueue.slice(-100);
            }
        }
    }

    deliverQueuedMessages(clientInfo) {
        if (clientInfo.messageQueue.length > 0) {
            const messages = [...clientInfo.messageQueue];
            clientInfo.messageQueue = [];

            messages.forEach(msg => {
                this.sendToClient(clientInfo, msg);
            });
        }
    }

    sendError(clientInfo, message, data = {}) {
        this.sendToClient(clientInfo, {
            type: 'error',
            message: message,
            ...data
        });
    }

    broadcastToUser(userId, data) {
        for (const [id, client] of this.clients) {
            if (client.userId === userId && client.state === 'authenticated') {
                this.sendToClient(client, data);
            }
        }
    }

    async broadcastNotification(notification) {
        // Get user preferences to check if in-app is enabled
        const preferences = await this.persistenceService.getUserPreferences(notification.userId);

        if (!preferences.inAppEnabled) {
            return;
        }

        // Check notification type preference
        const typeEnabled = this.isNotificationTypeEnabled(notification.type, preferences);
        if (!typeEnabled) {
            return;
        }

        // Broadcast to all user's clients
        for (const [id, client] of this.clients) {
            if (client.userId === notification.userId &&
                client.state === 'authenticated' &&
                this.shouldSendToClient(client, notification)) {

                this.sendToClient(client, {
                    type: 'new_notification',
                    notification: notification
                });
            }
        }
    }

    isNotificationTypeEnabled(type, preferences) {
        const typeMap = {
            'info': preferences.infoEnabled,
            'warning': preferences.warningEnabled,
            'error': preferences.errorEnabled,
            'success': preferences.successEnabled,
            'system': preferences.systemEnabled
        };

        return typeMap[type] !== false;
    }

    shouldSendToClient(client, notification) {
        // Check if client is subscribed to the notification's category
        if (client.subscribedCategories.length === 0) {
            return true; // No filters, send all
        }

        return client.subscribedCategories.includes(notification.category);
    }

    async connectToEventBus() {
        try {
            const eventBusUrl = process.env.EVENT_BUS_URL || 'blockchain://a2a-events';

            this.eventBusClient = new BlockchainEventClient(eventBusUrl, {
                perMessageDeflate: false
            });

            this.eventBusClient.on('open', () => {
                this.logger.info('Connected to event bus');

                // Subscribe to relevant events
                this.eventBusClient.send(JSON.stringify({
                    type: 'subscribe',
                    events: ['agent.*', 'transaction.*', 'system.*', 'security.*']
                }));
            });

            this.eventBusClient.on('message', async (data) => {
                try {
                    const event = JSON.parse(data);
                    await this.handleSystemEvent(event);
                } catch (error) {
                    this.logger.error('Failed to handle event bus message:', error);
                }
            });

            this.eventBusClient.on('error', (error) => {
                this.logger.error('Event bus connection error:', error);
            });

            this.eventBusClient.on('close', () => {
                this.logger.warn('Event bus connection closed, attempting reconnect...');
                setTimeout(() => this.connectToEventBus(), 5000);
            });

        } catch (error) {
            this.logger.error('Failed to connect to event bus:', error);
            // Retry connection
            setTimeout(() => this.connectToEventBus(), 10000);
        }
    }

    async handleSystemEvent(event) {
        try {
            let notificationData = null;

            switch (event.type) {
                case 'agent.connected':
                    notificationData = {
                        userId: event.data.ownerId || 'system',
                        title: 'Agent Connected',
                        message: `Agent "${event.data.agentName}" has connected to the network`,
                        type: 'success',
                        category: 'agent',
                        metadata: {
                            agentId: event.data.agentId,
                            agentName: event.data.agentName
                        }
                    };
                    break;

                case 'agent.disconnected':
                    notificationData = {
                        userId: event.data.ownerId || 'system',
                        title: 'Agent Disconnected',
                        message: `Agent "${event.data.agentName}" has disconnected from the network`,
                        type: 'warning',
                        category: 'agent',
                        metadata: {
                            agentId: event.data.agentId,
                            agentName: event.data.agentName
                        }
                    };
                    break;

                case 'transaction.completed':
                    notificationData = {
                        userId: event.data.userId,
                        title: 'Transaction Completed',
                        message: `Transaction ${event.data.transactionId} completed successfully`,
                        type: 'success',
                        category: 'transaction',
                        metadata: {
                            transactionId: event.data.transactionId,
                            amount: event.data.amount,
                            currency: event.data.currency
                        }
                    };
                    break;

                case 'system.alert':
                    notificationData = {
                        userId: 'system',
                        title: event.data.title || 'System Alert',
                        message: event.data.message,
                        type: event.data.severity || 'warning',
                        priority: event.data.priority || 'medium',
                        category: 'system',
                        metadata: event.data.metadata
                    };
                    break;

                case 'security.alert':
                    notificationData = {
                        userId: event.data.userId || 'system',
                        title: 'Security Alert',
                        message: event.data.message,
                        type: 'error',
                        priority: 'high',
                        category: 'security',
                        metadata: {
                            threat: event.data.threat,
                            source: event.data.source
                        }
                    };
                    break;
            }

            if (notificationData) {
                const notification = await this.persistenceService.createNotification(notificationData);
                await this.broadcastNotification(notification);
            }

        } catch (error) {
            this.logger.error('Failed to handle system event:', error);
        }
    }

    startBackgroundTasks() {
        // Heartbeat check
        this.intervals.set('heartbeat', setInterval(() => {
            this.checkClientHeartbeats();
        }, this.heartbeatInterval));

        // Cleanup expired notifications
        this.intervals.set('cleanup', setInterval(async () => {
            try {
                const cleaned = await this.persistenceService.cleanupExpiredNotifications();
                if (cleaned > 0) {
                    this.logger.info(`Cleaned up ${cleaned} expired notifications`);
                }
            } catch (error) {
                this.logger.error('Failed to cleanup notifications:', error);
            }
        }, 3600000)); // Every hour

        // Send ping to keep connections alive
        this.intervals.set('ping', setInterval(() => {
            for (const [id, client] of this.clients) {
                if (client.state === 'authenticated' && client.ws.readyState === WebSocket.OPEN) {
                    client.ws.ping();
                }
            }
        }, this.heartbeatInterval / 2));
    }

    checkClientHeartbeats() {
        for (const [id, client] of this.clients) {
            if (client.state === 'authenticated' && !client.isAlive) {
                this.logger.warn(`Client ${id} failed heartbeat check`);
                client.ws.terminate();
                continue;
            }
            client.isAlive = false;
        }
    }

    generateClientId() {
        return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    generateReconnectToken() {
        return `token_${Date.now()}_${Math.random().toString(36).substr(2, 16)}`;
    }

    getRESTHandlers() {
        return {
            createNotification: async (req, res) => {
                try {
                    const notification = await this.persistenceService.createNotification(req.body);
                    await this.broadcastNotification(notification);
                    res.status(201).json({ success: true, notification });
                } catch (error) {
                    this.logger.error('Failed to create notification:', error);
                    res.status(500).json({ success: false, error: error.message });
                }
            },

            getNotifications: async (req, res) => {
                try {
                    const { userId } = req.params;
                    const options = {
                        status: req.query.status,
                        type: req.query.type,
                        priority: req.query.priority,
                        category: req.query.category,
                        limit: parseInt(req.query.limit) || 50,
                        offset: parseInt(req.query.offset) || 0
                    };

                    const notifications = await this.persistenceService.getNotifications(userId, options);
                    const stats = await this.persistenceService.getNotificationStats(userId);

                    res.json({
                        success: true,
                        notifications,
                        stats,
                        options
                    });
                } catch (error) {
                    this.logger.error('Failed to get notifications:', error);
                    res.status(500).json({ success: false, error: error.message });
                }
            },

            markAsRead: async (req, res) => {
                try {
                    const { userId, notificationId } = req.params;
                    const success = await this.persistenceService.markAsRead(userId, notificationId);

                    if (success) {
                        this.broadcastToUser(userId, {
                            type: 'notification_read',
                            notificationId
                        });
                    }

                    res.json({ success });
                } catch (error) {
                    this.logger.error('Failed to mark as read:', error);
                    res.status(500).json({ success: false, error: error.message });
                }
            },

            getPreferences: async (req, res) => {
                try {
                    const { userId } = req.params;
                    const preferences = await this.persistenceService.getUserPreferences(userId);
                    res.json({ success: true, preferences });
                } catch (error) {
                    this.logger.error('Failed to get preferences:', error);
                    res.status(500).json({ success: false, error: error.message });
                }
            },

            updatePreferences: async (req, res) => {
                try {
                    const { userId } = req.params;
                    const preferences = await this.persistenceService.updateUserPreferences(userId, req.body);
                    res.json({ success: true, preferences });
                } catch (error) {
                    this.logger.error('Failed to update preferences:', error);
                    res.status(500).json({ success: false, error: error.message });
                }
            },

            getStats: async (req, res) => {
                try {
                    const stats = {
                        totalClients: this.clients.size,
                        authenticatedClients: Array.from(this.clients.values()).filter(c => c.state === 'authenticated').length,
                        serverUptime: process.uptime(),
                        wsPort: this.port
                    };
                    res.json({ success: true, stats });
                } catch (error) {
                    res.status(500).json({ success: false, error: error.message });
                }
            }
        };
    }

    shutdown() {
        this.logger.info('Shutting down enhanced notification service...');

        // Stop background tasks
        for (const [name, intervalId] of this.intervals) {
            clearInterval(intervalId);
        }
        this.intervals.clear();

        // Close event bus connection
        if (this.eventBusClient) {
            this.eventBusClient.close();
        }

        // Close all client connections
        for (const [id, client] of this.clients) {
            client.ws.close(1001, 'Server shutting down');
        }
        this.clients.clear();

        // Close WebSocket server
        if (this.wsServer) {
            this.wsServer.close(() => {
                this.logger.info('WebSocket server closed');
            });
        }

        this.logger.info('Enhanced notification service shutdown complete');
    }
}

module.exports = EnhancedNotificationService;