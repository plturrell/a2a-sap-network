/**
 * Integrated Notification Service
 * Orchestrates all notification components for a complete solution
 */

const EnhancedNotificationService = require('./enhancedNotificationService');
const NotificationPersistenceService = require('./notificationPersistenceService');
const PushNotificationService = require('./pushNotificationService');
const EventBusService = require('./eventBusService');
const cds = require('@sap/cds');

class IntegratedNotificationService {
    constructor() {
        this.logger = cds.log('integrated-notifications');
        this.isInitialized = false;

        // Initialize all services
        this.persistenceService = new NotificationPersistenceService();
        this.pushService = new PushNotificationService();
        this.eventBusService = new EventBusService();
        this.enhancedService = new EnhancedNotificationService();

        this.setupIntegrations().catch(error => {
            this.logger.error('Failed to setup service integrations:', error);
        });
    }

    async setupIntegrations() {
        try {
            // Wait for persistence service to be ready
            await this.persistenceService.init();

            // Connect enhanced service to persistence
            this.enhancedService.persistenceService = this.persistenceService;

            // Connect push service to enhanced service
            this.enhancedService.pushService = this.pushService;

            // Override delivery methods in enhanced service
            this.enhanceDeliveryMethods();

            // Connect event bus to notification creation
            this.eventBusService.on('event', async (event) => {
                try {
                    await this.enhancedService.handleSystemEvent(event);
                } catch (error) {
                    this.logger.error('Failed to handle system event for notifications:', error);
                }
            });

            // Set up cross-service event handling
            this.setupCrossServiceEvents();

            this.isInitialized = true;
            this.logger.info('Integrated notification service initialized successfully');
        } catch (error) {
            this.logger.error('Failed to setup integrations:', error);
            throw error;
        }
    }

    enhanceDeliveryMethods() {
        // Override the persistence service delivery methods with enhanced functionality
        const originalDeliverEmail = this.persistenceService.deliverEmail.bind(this.persistenceService);
        const originalDeliverPush = this.persistenceService.deliverPush.bind(this.persistenceService);

        // Enhanced email delivery
        this.persistenceService.deliverEmail = async (notification, preferences) => {
            try {
                if (!preferences.emailEnabled || !preferences.emailAddress) {
                    return { success: false, error: 'email_disabled' };
                }

                // TODO: Implement actual email service integration
                // For now, just use the original method
                return await originalDeliverEmail(notification, preferences);
            } catch (error) {
                this.logger.error('Enhanced email delivery failed:', error);
                return { success: false, error: error.message };
            }
        };

        // Enhanced push delivery using PushNotificationService
        this.persistenceService.deliverPush = async (notification, preferences) => {
            try {
                if (!preferences.pushEnabled || !preferences.pushToken) {
                    return { success: false, error: 'push_disabled' };
                }

                const result = await this.pushService.sendNotificationToUser(
                    notification.userId,
                    notification,
                    preferences
                );

                // Log the delivery attempt
                await this.persistenceService.logDelivery(
                    notification.ID,
                    'push',
                    result.success ? 'success' : 'failed',
                    1,
                    result.success ? null : result.message
                );

                return result;
            } catch (error) {
                this.logger.error('Enhanced push delivery failed:', error);
                await this.persistenceService.logDelivery(
                    notification.ID,
                    'push',
                    'failed',
                    1,
                    error.message
                );
                return { success: false, error: error.message };
            }
        };
    }

    setupCrossServiceEvents() {
        // Forward system events to event bus
        this.enhancedService.on('notification', (notification) => {
            this.eventBusService.publishEvent({
                type: 'notification.created',
                data: {
                    notificationId: notification.ID,
                    userId: notification.userId,
                    type: notification.type,
                    priority: notification.priority,
                    title: notification.title
                }
            });
        });

        // Handle agent events from event bus
        this.eventBusService.on('agent.connected', async (agentData) => {
            await this.createSystemNotification('AGENT_CONNECTED', {
                title: 'Agent Connected',
                message: `Agent "${agentData.agentName}" has connected to the network`,
                metadata: agentData
            });
        });

        this.eventBusService.on('agent.disconnected', async (agentData) => {
            await this.createSystemNotification('AGENT_DISCONNECTED', {
                title: 'Agent Disconnected',
                message: `Agent "${agentData.agentName}" has disconnected from the network`,
                metadata: agentData
            });
        });

        // Handle transaction events
        this.eventBusService.on('transaction.completed', async (txData) => {
            await this.createUserNotification(txData.userId, 'TRANSACTION_COMPLETED', {
                title: 'Transaction Completed',
                message: `Your transaction ${txData.transactionId} has been completed successfully`,
                metadata: txData
            });
        });

        // Handle system alerts
        this.eventBusService.on('system.alert', async (alertData) => {
            await this.createSystemNotification('SYSTEM_ALERT', {
                title: alertData.title || 'System Alert',
                message: alertData.message,
                priority: alertData.severity,
                metadata: alertData
            });
        });

        // Handle security alerts
        this.eventBusService.on('security.alert', async (securityData) => {
            await this.createSecurityNotification(securityData);
        });
    }

    async createSystemNotification(type, notificationData) {
        try {
            // Create notification for system administrators
            const systemUsers = await this.getSystemAdminUsers();

            for (const userId of systemUsers) {
                await this.createUserNotification(userId, type, notificationData);
            }
        } catch (error) {
            this.logger.error('Failed to create system notification:', error);
        }
    }

    async createUserNotification(userId, type, notificationData) {
        try {
            const notification = await this.persistenceService.createNotification({
                userId: userId,
                title: notificationData.title,
                message: notificationData.message,
                type: this.mapNotificationType(type),
                priority: notificationData.priority || this.getDefaultPriority(type),
                category: notificationData.category || this.getDefaultCategory(type),
                metadata: notificationData.metadata || {},
                actions: notificationData.actions || [],
                expiresAt: notificationData.expiresAt
            });

            // Broadcast via WebSocket
            await this.enhancedService.broadcastNotification(notification);

            return notification;
        } catch (error) {
            this.logger.error('Failed to create user notification:', error);
            throw error;
        }
    }

    async createSecurityNotification(securityData) {
        try {
            const affectedUsers = securityData.userId ? [securityData.userId] : await this.getSystemAdminUsers();

            for (const userId of affectedUsers) {
                await this.createUserNotification(userId, 'SECURITY_ALERT', {
                    title: 'Security Alert',
                    message: securityData.message,
                    priority: 'critical',
                    category: 'security',
                    metadata: {
                        threat: securityData.threat,
                        source: securityData.source,
                        severity: securityData.severity
                    }
                });
            }
        } catch (error) {
            this.logger.error('Failed to create security notification:', error);
        }
    }

    mapNotificationType(systemType) {
        const typeMap = {
            'AGENT_CONNECTED': 'success',
            'AGENT_DISCONNECTED': 'warning',
            'TRANSACTION_COMPLETED': 'success',
            'SYSTEM_ALERT': 'warning',
            'SECURITY_ALERT': 'error',
            'PERFORMANCE_ALERT': 'warning'
        };

        return typeMap[systemType] || 'info';
    }

    getDefaultPriority(type) {
        const priorityMap = {
            'AGENT_CONNECTED': 'medium',
            'AGENT_DISCONNECTED': 'high',
            'TRANSACTION_COMPLETED': 'medium',
            'SYSTEM_ALERT': 'high',
            'SECURITY_ALERT': 'critical',
            'PERFORMANCE_ALERT': 'high'
        };

        return priorityMap[type] || 'medium';
    }

    getDefaultCategory(type) {
        const categoryMap = {
            'AGENT_CONNECTED': 'agent',
            'AGENT_DISCONNECTED': 'agent',
            'TRANSACTION_COMPLETED': 'transaction',
            'SYSTEM_ALERT': 'system',
            'SECURITY_ALERT': 'security',
            'PERFORMANCE_ALERT': 'system'
        };

        return categoryMap[type] || 'general';
    }

    async getSystemAdminUsers() {
        // TODO: Implement actual system admin user lookup
        // For now, return a default admin user
        return ['system-admin'];
    }

    // Public API methods
    async createNotification(notificationData) {
        if (!this.isInitialized) {
            throw new Error('Service not initialized');
        }

        return await this.createUserNotification(
            notificationData.userId,
            'CUSTOM',
            notificationData
        );
    }

    async getUserNotifications(userId, options = {}) {
        if (!this.isInitialized) {
            throw new Error('Service not initialized');
        }

        return await this.persistenceService.getNotifications(userId, options);
    }

    async markAsRead(userId, notificationId) {
        if (!this.isInitialized) {
            throw new Error('Service not initialized');
        }

        const success = await this.persistenceService.markAsRead(userId, notificationId);

        if (success) {
            // Broadcast update via WebSocket
            this.enhancedService.broadcastToUser(userId, {
                type: 'notification_read',
                notificationId: notificationId
            });
        }

        return success;
    }

    async getUserPreferences(userId) {
        if (!this.isInitialized) {
            throw new Error('Service not initialized');
        }

        return await this.persistenceService.getUserPreferences(userId);
    }

    async updateUserPreferences(userId, preferences) {
        if (!this.isInitialized) {
            throw new Error('Service not initialized');
        }

        return await this.persistenceService.updateUserPreferences(userId, preferences);
    }

    async getNotificationStats(userId) {
        if (!this.isInitialized) {
            throw new Error('Service not initialized');
        }

        return await this.persistenceService.getNotificationStats(userId);
    }

    // Get VAPID public key for push notifications
    getVapidPublicKey() {
        return this.pushService.getVapidPublicKey();
    }

    // REST API handlers
    getRESTHandlers() {
        const handlers = {};

        // Merge all service handlers
        Object.assign(handlers, this.enhancedService.getRESTHandlers());
        Object.assign(handlers, this.pushService.getRESTHandlers());
        Object.assign(handlers, this.eventBusService.getRESTHandlers());

        // Add integrated service specific handlers
        handlers.getServiceStats = (req, res) => {
            try {
                const stats = {
                    initialized: this.isInitialized,
                    services: {
                        websocket: this.enhancedService.clients.size,
                        eventBus: this.eventBusService.clients.size,
                        pushEnabled: this.pushService.isInitialized
                    },
                    uptime: process.uptime()
                };

                res.json({ success: true, stats });
            } catch (error) {
                res.status(500).json({ success: false, error: error.message });
            }
        };

        return handlers;
    }

    // Health check
    async healthCheck() {
        const health = {
            status: 'healthy',
            services: {
                persistence: false,
                websocket: false,
                eventBus: false,
                push: false
            },
            timestamp: new Date().toISOString()
        };

        try {
            // Check persistence service
            health.services.persistence = this.persistenceService.db !== null;

            // Check WebSocket service
            health.services.websocket = this.enhancedService.wsServer !== null;

            // Check event bus service
            health.services.eventBus = this.eventBusService.wsServer !== null;

            // Check push service
            health.services.push = this.pushService.isInitialized;

            const allHealthy = Object.values(health.services).every(status => status);
            health.status = allHealthy ? 'healthy' : 'degraded';

        } catch (error) {
            health.status = 'unhealthy';
            health.error = error.message;
        }

        return health;
    }

    // Graceful shutdown
    async shutdown() {
        this.logger.info('Shutting down integrated notification service...');

        try {
            // Shutdown all services
            if (this.enhancedService) {
                this.enhancedService.shutdown();
            }

            if (this.eventBusService) {
                this.eventBusService.shutdown();
            }

            this.logger.info('Integrated notification service shutdown complete');
        } catch (error) {
            this.logger.error('Error during shutdown:', error);
        }
    }
}

module.exports = IntegratedNotificationService;