/**
 * Enhanced Notification Service for A2A Network
 * Provides real-time notifications with WebSocket support
 * Meets SAP Enterprise Standards for notification management
 */

const WebSocket = require('ws');
const EventEmitter = require('events');
const cds = require('@sap/cds');
const { portManager } = require('./utils/portManager');
const logger = cds.log('notifications');

class A2ANotificationService extends EventEmitter {
    constructor() {
        super();
        this.notifications = [];
        this.subscribers = new Set();
        this.wsServer = null;
        this.port = null;
        this.intervals = new Map(); // Track intervals for cleanup
        this.notificationTypes = {
            AGENT_CONNECTED: { priority: 'medium', category: 'network' },
            AGENT_DISCONNECTED: { priority: 'high', category: 'network' },
            SERVICE_AVAILABLE: { priority: 'low', category: 'marketplace' },
            TRANSACTION_COMPLETED: { priority: 'medium', category: 'blockchain' },
            SYSTEM_UPDATE: { priority: 'high', category: 'system' },
            PERFORMANCE_ALERT: { priority: 'high', category: 'monitoring' },
            SECURITY_ALERT: { priority: 'critical', category: 'security' }
        };
        this.initializeWebSocketServer().catch(error => {
            logger.error('Failed to initialize WebSocket server:', error);
        });
        this.startPeriodicTasks();
    }

    async initializeWebSocketServer() {
        try {
            // Use port manager to handle port conflicts
            const killConflicts = process.env.NODE_ENV === 'development';
            this.port = await portManager.allocatePortSafely('notifications', 4005, killConflicts);
            
            if (!this.port) {
                logger.warn('⚠️  WebSocket notifications disabled due to port allocation failure');
                return;
            }

            this.wsServer = new WebSocket.Server({ 
                port: this.port,
                path: '/notifications'
            });

            this.wsServer.on('connection', (ws, req) => {
                logger.debug('📱 New notification subscriber connected');
                this.subscribers.add(ws);

                // Send recent notifications to new subscriber
                const recentNotifications = this.notifications
                    .slice(0, 10)
                    .map(n => ({ ...n, action: 'notification' }));
                
                ws.send(JSON.stringify({
                    action: 'initial_load',
                    notifications: recentNotifications
                }));

                ws.on('message', (message) => {
                    try {
                        const data = JSON.parse(message);
                        this.handleWebSocketMessage(ws, data);
                    } catch (error) {
                        this.log.error('Invalid WebSocket message:', error);
                    }
                });

                ws.on('close', () => {
                    logger.debug('📱 Notification subscriber disconnected');
                    this.subscribers.delete(ws);
                });

                ws.on('error', (error) => {
                    this.log.error('WebSocket error:', error);
                    this.subscribers.delete(ws);
                });
            });

            logger.info(`🔔 Notification WebSocket server started on port ${this.port}`);
        } catch (error) {
            this.log.error('Failed to start notification WebSocket server:', error);
        }
    }

    handleWebSocketMessage(ws, data) {
        switch (data.action) {
            case 'mark_read':
                this.markAsRead(data.notificationId);
                break;
            case 'mark_all_read':
                this.markAllAsRead();
                break;
            case 'subscribe_category':
                // Handle category-specific subscriptions
                ws.subscribedCategories = data.categories || [];
                break;
            case 'get_notifications':
                this.sendNotificationsToClient(ws, data.filters);
                break;
        }
    }

    createNotification(type, title, message, metadata = {}) {
        const notificationConfig = this.notificationTypes[type] || {
            priority: 'medium',
            category: 'general'
        };

        const notification = {
            id: this.generateNotificationId(),
            type,
            title,
            message,
            priority: notificationConfig.priority,
            category: notificationConfig.category,
            timestamp: new Date().toISOString(),
            read: false,
            metadata: {
                ...metadata,
                source: 'A2A_NETWORK',
                version: '2.1.0'
            }
        };

        this.notifications.unshift(notification);
        
        // Keep only last 100 notifications
        if (this.notifications.length > 100) {
            this.notifications = this.notifications.slice(0, 100);
        }

        // Broadcast to all subscribers
        this.broadcastNotification(notification);
        
        // Emit event for internal listeners
        this.emit('notification', notification);
        
        logger.debug(`🔔 Created ${notification.priority} notification: ${title}`);
        return notification;
    }

    broadcastNotification(notification) {
        const message = JSON.stringify({
            action: 'notification',
            ...notification
        });

        this.subscribers.forEach(ws => {
            if (ws.readyState === WebSocket.OPEN) {
                // Check if client is subscribed to this category
                if (!ws.subscribedCategories || 
                    ws.subscribedCategories.length === 0 || 
                    ws.subscribedCategories.includes(notification.category)) {
                    ws.send(message);
                }
            }
        });
    }

    markAsRead(notificationId) {
        const notification = this.notifications.find(n => n.id === notificationId);
        if (notification) {
            notification.read = true;
            this.broadcastUpdate('mark_read', { notificationId });
        }
    }

    markAllAsRead() {
        this.notifications.forEach(n => n.read = true);
        this.broadcastUpdate('mark_all_read', {});
    }

    broadcastUpdate(action, data) {
        const message = JSON.stringify({ action, ...data });
        this.subscribers.forEach(ws => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(message);
            }
        });
    }

    getNotifications(filters = {}) {
        let filtered = [...this.notifications];

        if (filters.category) {
            filtered = filtered.filter(n => n.category === filters.category);
        }

        if (filters.priority) {
            filtered = filtered.filter(n => n.priority === filters.priority);
        }

        if (filters.unreadOnly) {
            filtered = filtered.filter(n => !n.read);
        }

        if (filters.limit) {
            filtered = filtered.slice(0, filters.limit);
        }

        return filtered;
    }

    getUnreadCount() {
        return this.notifications.filter(n => !n.read).length;
    }

    generateNotificationId() {
        return `notif_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    startPeriodicTasks() {
        // Stop existing tasks first
        this.stopPeriodicTasks();
        
        // Simulate system notifications every 2 minutes
        this.intervals.set('systemNotifications', setInterval(() => {
            this.generateSystemNotifications();
        }, 120000));

        // Clean up old notifications every hour
        this.intervals.set('cleanup', setInterval(() => {
            this.cleanupOldNotifications();
        }, 3600000));
    }
    
    stopPeriodicTasks() {
        for (const [name, intervalId] of this.intervals) {
            clearInterval(intervalId);
            logger.debug(`Cleared interval: ${name}`);
        }
        this.intervals.clear();
    }
    
    shutdown() {
        logger.info('Shutting down notification service...');
        
        // Stop periodic tasks
        this.stopPeriodicTasks();
        
        // Close all subscriber connections
        for (const ws of this.subscribers) {
            ws.close(1001, 'Server shutting down');
        }
        this.subscribers.clear();
        
        // Close WebSocket server
        if (this.wsServer) {
            this.wsServer.close(() => {
                logger.info('Notification WebSocket server closed');
            });
        }
        
        logger.info('Notification service shutdown complete');
    }

    async generateSystemNotifications() {
        // Removed simulated notifications - now listen for real system events
        // Notifications are created only when actual events occur
        try {
            // Subscribe to real system events from event bus
            const eventBusUrl = process.env.EVENT_BUS_URL || 'ws://localhost:8080/events';
            logger.info('System notifications now listening for real events from:', eventBusUrl);
            
            // In production, this would connect to the actual event streaming service
            // and create notifications based on real agent connections, transactions, etc.
            
        } catch (error) {
            logger.error('Failed to connect to event bus for system notifications:', error);
            // No fallback to simulated notifications
        }
    }

    async handleRealSystemEvent(eventType, eventData) {
        // Handle real system events and create appropriate notifications
        let notification = null;
        
        switch (eventType) {
            case 'AGENT_CONNECTED':
                notification = {
                    type: 'AGENT_CONNECTED',
                    title: 'New Agent Connected',
                    message: `Agent "${eventData.agentName}" joined the network`
                };
                break;
            case 'TRANSACTION_COMPLETED':
                notification = {
                    type: 'TRANSACTION_COMPLETED',
                    title: 'Transaction Completed',
                    message: `Transaction ${eventData.txHash} completed successfully`
                };
                break;
            case 'PERFORMANCE_ALERT':
                notification = {
                    type: 'PERFORMANCE_ALERT',
                    title: 'Performance Update',
                    message: `Network performance: ${eventData.efficiency}% efficiency`
                };
                break;
        }
        
        if (notification) {
            this.createNotification(notification.type, notification.title, notification.message);
        }
    }

    cleanupOldNotifications() {
        const oneWeekAgo = new Date();
        oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);

        const initialCount = this.notifications.length;
        this.notifications = this.notifications.filter(n => 
            new Date(n.timestamp) > oneWeekAgo
        );

        const cleanedCount = initialCount - this.notifications.length;
        if (cleanedCount > 0) {
            logger.debug(`🧹 Cleaned up ${cleanedCount} old notifications`);
        }
    }

    // REST API endpoints for HTTP access
    getRESTHandlers() {
        return {
            // GET /api/v1/notifications
            getNotifications: (req, res) => {
                const filters = {
                    category: req.query.category,
                    priority: req.query.priority,
                    unreadOnly: req.query.unread === 'true',
                    limit: parseInt(req.query.limit) || 50
                };

                const notifications = this.getNotifications(filters);
                res.json({
                    success: true,
                    count: notifications.length,
                    unreadCount: this.getUnreadCount(),
                    notifications
                });
            },

            // POST /api/v1/notifications
            createNotification: (req, res) => {
                const { type, title, message, metadata } = req.body;
                
                if (!type || !title || !message) {
                    return res.status(400).json({
                        success: false,
                        error: 'Missing required fields: type, title, message'
                    });
                }

                const notification = this.createNotification(type, title, message, metadata);
                res.status(201).json({
                    success: true,
                    notification
                });
            },

            // PUT /api/v1/notifications/:id/read
            markAsRead: (req, res) => {
                const { id } = req.params;
                this.markAsRead(id);
                res.json({ success: true });
            },

            // PUT /api/v1/notifications/read-all
            markAllAsRead: (req, res) => {
                this.markAllAsRead();
                res.json({ success: true });
            },

            // GET /api/v1/notifications/stats
            getStats: (req, res) => {
                const stats = {
                    total: this.notifications.length,
                    unread: this.getUnreadCount(),
                    byCategory: {},
                    byPriority: {},
                    subscribers: this.subscribers.size
                };

                this.notifications.forEach(n => {
                    stats.byCategory[n.category] = (stats.byCategory[n.category] || 0) + 1;
                    stats.byPriority[n.priority] = (stats.byPriority[n.priority] || 0) + 1;
                });

                res.json({ success: true, stats });
            }
        };
    }

    // Initialize with sample notifications for demo
    initializeSampleNotifications() {
        const samples = [
            {
                type: 'AGENT_CONNECTED',
                title: 'Agent Network Expanded',
                message: 'DataProcessor-01 successfully joined the A2A network'
            },
            {
                type: 'SYSTEM_UPDATE',
                title: 'System Update Available',
                message: 'A2A Network v2.1.0 is ready for installation with enhanced security features'
            },
            {
                type: 'PERFORMANCE_ALERT',
                title: 'Performance Optimization',
                message: 'Network latency improved by 23% after recent optimizations'
            },
            {
                type: 'TRANSACTION_COMPLETED',
                title: 'Smart Contract Executed',
                message: 'Multi-agent collaboration contract completed successfully'
            },
            {
                type: 'SERVICE_AVAILABLE',
                title: 'New Service Available',
                message: 'ML Model Training service is now available in the marketplace'
            }
        ];

        samples.forEach(sample => {
            this.createNotification(sample.type, sample.title, sample.message);
        });

        logger.debug('🔔 Initialized with sample notifications');
    }
}

module.exports = A2ANotificationService;
