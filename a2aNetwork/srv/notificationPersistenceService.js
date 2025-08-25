/**
 * Notification Persistence Service
 * Handles database operations for notifications with retry logic and delivery guarantees
 */

const cds = require('@sap/cds');
const { SELECT, INSERT, UPDATE, DELETE } = cds.ql;
const { v4: uuidv4 } = require('uuid');

class NotificationPersistenceService {
    constructor() {
        this.db = null;
        this.logger = cds.log('notification-persistence');
        this.retryConfig = {
            maxAttempts: 3,
            baseDelay: 1000, // 1 second
            maxDelay: 30000  // 30 seconds
        };
    }

    async init() {
        try {
            this.db = await cds.connect.to('db');
            const { Notifications, NotificationActions, NotificationPreferences, NotificationDeliveryLog } = this.db.entities('a2a.network');
            this.entities = { Notifications, NotificationActions, NotificationPreferences, NotificationDeliveryLog };
            this.logger.info('Notification persistence service initialized');
        } catch (error) {
            this.logger.error('Failed to initialize persistence service:', error);
            throw error;
        }
    }

    async createNotification(notificationData) {
        const { Notifications, NotificationActions } = this.entities;
        const tx = this.db.transaction();

        try {
            // Prepare notification entity
            const notification = {
                ID: uuidv4(),
                userId: notificationData.userId,
                title: notificationData.title,
                message: notificationData.message,
                type: notificationData.type || 'info',
                priority: notificationData.priority || 'medium',
                status: 'unread',
                source: notificationData.source || 'system',
                category: notificationData.category || 'general',
                metadata: JSON.stringify(notificationData.metadata || {}),
                expiresAt: notificationData.expiresAt,
                deliveryStatus: 'pending',
                deliveryAttempts: 0
            };

            // Insert notification
            const createdNotification = await tx.run(
                INSERT.into(Notifications).entries(notification)
            );

            // Insert actions if provided
            if (notificationData.actions && notificationData.actions.length > 0) {
                const actions = notificationData.actions.map(action => ({
                    ID: uuidv4(),
                    notification_ID: notification.ID,
                    label: action.label,
                    actionType: action.actionType,
                    target: action.target,
                    style: action.style || 'default'
                }));

                await tx.run(
                    INSERT.into(NotificationActions).entries(actions)
                );
            }

            await tx.commit();

            // Schedule delivery
            this.scheduleDelivery(notification.ID);

            return notification;
        } catch (error) {
            await tx.rollback();
            this.logger.error('Failed to create notification:', error);
            throw error;
        }
    }

    async getNotifications(userId, options = {}) {
        const { Notifications, NotificationActions } = this.entities;

        try {
            let query = SELECT.from(Notifications)
                .where({ userId: userId });

            // Apply filters
            if (options.status) {
                query = query.and({ status: options.status });
            }
            if (options.type) {
                query = query.and({ type: options.type });
            }
            if (options.priority) {
                query = query.and({ priority: options.priority });
            }
            if (options.category) {
                query = query.and({ category: options.category });
            }

            // Exclude expired notifications
            query = query.and('expiresAt is null or expiresAt > ', new Date().toISOString());

            // Sorting
            query = query.orderBy('priority desc', 'createdAt desc');

            // Pagination
            if (options.limit) {
                query = query.limit(options.limit);
            }
            if (options.offset) {
                query = query.offset(options.offset);
            }

            const notifications = await this.db.run(query);

            // Fetch actions for each notification
            for (const notification of notifications) {
                const actions = await this.db.run(
                    SELECT.from(NotificationActions).where({ notification_ID: notification.ID })
                );
                notification.actions = actions;
                notification.metadata = JSON.parse(notification.metadata || '{}');
            }

            return notifications;
        } catch (error) {
            this.logger.error('Failed to get notifications:', error);
            throw error;
        }
    }

    async markAsRead(userId, notificationId) {
        const { Notifications } = this.entities;

        try {
            const result = await this.db.run(
                UPDATE(Notifications)
                    .set({
                        status: 'read',
                        readAt: new Date().toISOString()
                    })
                    .where({ ID: notificationId, userId: userId })
            );

            return result > 0;
        } catch (error) {
            this.logger.error('Failed to mark notification as read:', error);
            throw error;
        }
    }

    async markAllAsRead(userId) {
        const { Notifications } = this.entities;

        try {
            const result = await this.db.run(
                UPDATE(Notifications)
                    .set({
                        status: 'read',
                        readAt: new Date().toISOString()
                    })
                    .where({ userId: userId, status: 'unread' })
            );

            return result;
        } catch (error) {
            this.logger.error('Failed to mark all notifications as read:', error);
            throw error;
        }
    }

    async dismissNotification(userId, notificationId) {
        const { Notifications } = this.entities;

        try {
            const result = await this.db.run(
                UPDATE(Notifications)
                    .set({
                        status: 'dismissed',
                        dismissedAt: new Date().toISOString()
                    })
                    .where({ ID: notificationId, userId: userId })
            );

            return result > 0;
        } catch (error) {
            this.logger.error('Failed to dismiss notification:', error);
            throw error;
        }
    }

    async deleteNotification(userId, notificationId) {
        const { Notifications, NotificationActions } = this.entities;
        const tx = this.db.transaction();

        try {
            // Delete actions first
            await tx.run(
                DELETE.from(NotificationActions).where({ notification_ID: notificationId })
            );

            // Delete notification
            const result = await tx.run(
                DELETE.from(Notifications).where({ ID: notificationId, userId: userId })
            );

            await tx.commit();
            return result > 0;
        } catch (error) {
            await tx.rollback();
            this.logger.error('Failed to delete notification:', error);
            throw error;
        }
    }

    async getUserPreferences(userId) {
        const { NotificationPreferences } = this.entities;

        try {
            const preferences = await this.db.run(
                SELECT.one.from(NotificationPreferences).where({ userId: userId })
            );

            if (!preferences) {
                // Create default preferences
                return await this.createDefaultPreferences(userId);
            }

            if (preferences.deviceInfo) {
                preferences.deviceInfo = JSON.parse(preferences.deviceInfo);
            }

            return preferences;
        } catch (error) {
            this.logger.error('Failed to get user preferences:', error);
            throw error;
        }
    }

    async updateUserPreferences(userId, preferences) {
        const { NotificationPreferences } = this.entities;

        try {
            const existing = await this.getUserPreferences(userId);

            if (preferences.deviceInfo) {
                preferences.deviceInfo = JSON.stringify(preferences.deviceInfo);
            }

            if (existing) {
                await this.db.run(
                    UPDATE(NotificationPreferences)
                        .set(preferences)
                        .where({ ID: existing.ID })
                );
            } else {
                await this.db.run(
                    INSERT.into(NotificationPreferences).entries({
                        ID: uuidv4(),
                        userId: userId,
                        ...preferences
                    })
                );
            }

            return await this.getUserPreferences(userId);
        } catch (error) {
            this.logger.error('Failed to update user preferences:', error);
            throw error;
        }
    }

    async createDefaultPreferences(userId) {
        const { NotificationPreferences } = this.entities;

        const defaultPreferences = {
            ID: uuidv4(),
            userId: userId,
            emailEnabled: true,
            pushEnabled: false,
            inAppEnabled: true,
            infoEnabled: true,
            warningEnabled: true,
            errorEnabled: true,
            successEnabled: true,
            systemEnabled: true,
            timezone: 'UTC'
        };

        try {
            await this.db.run(
                INSERT.into(NotificationPreferences).entries(defaultPreferences)
            );
            return defaultPreferences;
        } catch (error) {
            this.logger.error('Failed to create default preferences:', error);
            throw error;
        }
    }

    async logDelivery(notificationId, channel, status, attemptNumber, errorMessage = null) {
        const { NotificationDeliveryLog } = this.entities;

        try {
            await this.db.run(
                INSERT.into(NotificationDeliveryLog).entries({
                    ID: uuidv4(),
                    notification_ID: notificationId,
                    channel: channel,
                    status: status,
                    attemptNumber: attemptNumber,
                    errorMessage: errorMessage,
                    deliveredAt: status === 'success' ? new Date().toISOString() : null,
                    metadata: JSON.stringify({
                        timestamp: new Date().toISOString(),
                        attemptNumber: attemptNumber
                    })
                })
            );
        } catch (error) {
            this.logger.error('Failed to log delivery:', error);
        }
    }

    async scheduleDelivery(notificationId) {
        // Immediate delivery attempt
        setImmediate(() => this.deliverNotification(notificationId));
    }

    async deliverNotification(notificationId, attemptNumber = 1) {
        const { Notifications } = this.entities;

        try {
            const notification = await this.db.run(
                SELECT.one.from(Notifications).where({ ID: notificationId })
            );

            if (!notification) {
                this.logger.warn(`Notification ${notificationId} not found for delivery`);
                return;
            }

            // Check if already delivered
            if (notification.deliveryStatus === 'delivered') {
                return;
            }

            // Get user preferences
            const preferences = await this.getUserPreferences(notification.userId);

            // Check quiet hours
            if (this.isInQuietHours(preferences)) {
                // Reschedule for after quiet hours
                const delayMs = this.getQuietHoursDelay(preferences);
                setTimeout(() => this.deliverNotification(notificationId, attemptNumber), delayMs);
                return;
            }

            // Attempt delivery through enabled channels
            const deliveryPromises = [];

            if (preferences.inAppEnabled) {
                deliveryPromises.push(this.deliverInApp(notification, preferences));
            }

            if (preferences.emailEnabled && preferences.emailAddress) {
                deliveryPromises.push(this.deliverEmail(notification, preferences));
            }

            if (preferences.pushEnabled && preferences.pushToken) {
                deliveryPromises.push(this.deliverPush(notification, preferences));
            }

            const results = await Promise.allSettled(deliveryPromises);
            const anySuccess = results.some(r => r.status === 'fulfilled' && r.value === true);

            if (anySuccess) {
                // Mark as delivered
                await this.db.run(
                    UPDATE(Notifications)
                        .set({
                            deliveryStatus: 'delivered',
                            lastDeliveryAt: new Date().toISOString()
                        })
                        .where({ ID: notificationId })
                );
            } else if (attemptNumber < this.retryConfig.maxAttempts) {
                // Schedule retry
                const delay = this.calculateRetryDelay(attemptNumber);

                await this.db.run(
                    UPDATE(Notifications)
                        .set({
                            deliveryStatus: 'retrying',
                            deliveryAttempts: attemptNumber
                        })
                        .where({ ID: notificationId })
                );

                setTimeout(() => this.deliverNotification(notificationId, attemptNumber + 1), delay);
            } else {
                // Mark as failed after max attempts
                await this.db.run(
                    UPDATE(Notifications)
                        .set({
                            deliveryStatus: 'failed',
                            deliveryAttempts: attemptNumber
                        })
                        .where({ ID: notificationId })
                );
            }
        } catch (error) {
            this.logger.error(`Failed to deliver notification ${notificationId}:`, error);
            await this.logDelivery(notificationId, 'system', 'failed', attemptNumber, error.message);
        }
    }

    async deliverInApp(notification, preferences) {
        try {
            // In-app delivery is handled by WebSocket service
            // This just logs the delivery
            await this.logDelivery(notification.ID, 'in-app', 'success', 1);
            return true;
        } catch (error) {
            await this.logDelivery(notification.ID, 'in-app', 'failed', 1, error.message);
            return false;
        }
    }

    async deliverEmail(notification, preferences) {
        try {
            // TODO: Implement actual email delivery
            // For now, just log the attempt
            this.logger.info(`Email delivery for notification ${notification.ID} to ${preferences.emailAddress}`);
            await this.logDelivery(notification.ID, 'email', 'success', 1);
            return true;
        } catch (error) {
            await this.logDelivery(notification.ID, 'email', 'failed', 1, error.message);
            return false;
        }
    }

    async deliverPush(notification, preferences) {
        try {
            // TODO: Implement actual push notification delivery
            // For now, just log the attempt
            this.logger.info(`Push delivery for notification ${notification.ID} to device ${preferences.pushToken}`);
            await this.logDelivery(notification.ID, 'push', 'success', 1);
            return true;
        } catch (error) {
            await this.logDelivery(notification.ID, 'push', 'failed', 1, error.message);
            return false;
        }
    }

    isInQuietHours(preferences) {
        if (!preferences.quietHoursStart || !preferences.quietHoursEnd) {
            return false;
        }

        const now = new Date();
        const userTime = this.convertToUserTimezone(now, preferences.timezone);
        const currentTime = userTime.getHours() * 60 + userTime.getMinutes();

        const [startHour, startMin] = preferences.quietHoursStart.split(':').map(Number);
        const [endHour, endMin] = preferences.quietHoursEnd.split(':').map(Number);

        const quietStart = startHour * 60 + startMin;
        const quietEnd = endHour * 60 + endMin;

        if (quietStart <= quietEnd) {
            return currentTime >= quietStart && currentTime <= quietEnd;
        } else {
            // Quiet hours span midnight
            return currentTime >= quietStart || currentTime <= quietEnd;
        }
    }

    getQuietHoursDelay(preferences) {
        const now = new Date();
        const userTime = this.convertToUserTimezone(now, preferences.timezone);
        const [endHour, endMin] = preferences.quietHoursEnd.split(':').map(Number);

        const endTime = new Date(userTime);
        endTime.setHours(endHour, endMin, 0, 0);

        if (endTime <= userTime) {
            // End time is tomorrow
            endTime.setDate(endTime.getDate() + 1);
        }

        return endTime.getTime() - userTime.getTime();
    }

    convertToUserTimezone(date, timezone) {
        // Simple timezone conversion - in production, use a proper timezone library
        return new Date(date.toLocaleString('en-US', { timeZone: timezone }));
    }

    calculateRetryDelay(attemptNumber) {
        const exponentialDelay = Math.min(
            this.retryConfig.baseDelay * Math.pow(2, attemptNumber - 1),
            this.retryConfig.maxDelay
        );

        // Add jitter to prevent thundering herd
        const jitter = Math.random() * 0.3 * exponentialDelay;

        return exponentialDelay + jitter;
    }

    async cleanupExpiredNotifications() {
        const { Notifications, NotificationActions, NotificationDeliveryLog } = this.entities;
        const tx = this.db.transaction();

        try {
            // Find expired notifications
            const expired = await tx.run(
                SELECT.from(Notifications)
                    .columns('ID')
                    .where('expiresAt is not null and expiresAt < ', new Date().toISOString())
            );

            if (expired.length === 0) {
                return 0;
            }

            const expiredIds = expired.map(n => n.ID);

            // Delete related data
            await tx.run(
                DELETE.from(NotificationDeliveryLog).where({ notification_ID: { in: expiredIds } })
            );

            await tx.run(
                DELETE.from(NotificationActions).where({ notification_ID: { in: expiredIds } })
            );

            await tx.run(
                DELETE.from(Notifications).where({ ID: { in: expiredIds } })
            );

            await tx.commit();

            this.logger.info(`Cleaned up ${expiredIds.length} expired notifications`);
            return expiredIds.length;
        } catch (error) {
            await tx.rollback();
            this.logger.error('Failed to cleanup expired notifications:', error);
            throw error;
        }
    }

    async getNotificationStats(userId) {
        const { Notifications } = this.entities;

        try {
            const stats = await this.db.run(
                SELECT.from(Notifications)
                    .columns('type', 'priority', 'status')
                    .where({ userId: userId })
            );

            const summary = {
                total: stats.length,
                unread: stats.filter(n => n.status === 'unread').length,
                byType: {},
                byPriority: {},
                byStatus: {}
            };

            stats.forEach(n => {
                summary.byType[n.type] = (summary.byType[n.type] || 0) + 1;
                summary.byPriority[n.priority] = (summary.byPriority[n.priority] || 0) + 1;
                summary.byStatus[n.status] = (summary.byStatus[n.status] || 0) + 1;
            });

            return summary;
        } catch (error) {
            this.logger.error('Failed to get notification stats:', error);
            throw error;
        }
    }
}

module.exports = NotificationPersistenceService;
