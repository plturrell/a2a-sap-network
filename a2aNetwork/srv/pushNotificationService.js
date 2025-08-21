/**
 * Push Notification Service
 * Handles browser and mobile push notifications using Web Push Protocol
 */

const webpush = require('web-push');
const cds = require('@sap/cds');

class PushNotificationService {
    constructor() {
        this.logger = cds.log('push-notifications');
        this.isInitialized = false;
        this.vapidKeys = {
            publicKey: process.env.VAPID_PUBLIC_KEY,
            privateKey: process.env.VAPID_PRIVATE_KEY
        };
        this.vapidSubject = process.env.VAPID_SUBJECT || 'mailto:admin@a2a-network.com';
        
        this.initialize();
    }

    initialize() {
        try {
            if (!this.vapidKeys.publicKey || !this.vapidKeys.privateKey) {
                // Generate VAPID keys if not provided
                this.vapidKeys = webpush.generateVAPIDKeys();
                this.logger.warn('VAPID keys not found in environment, using generated keys');
                this.logger.info('Generated VAPID Public Key:', this.vapidKeys.publicKey);
                this.logger.info('Generated VAPID Private Key:', this.vapidKeys.privateKey);
            }

            // Configure web-push
            webpush.setVapidDetails(
                this.vapidSubject,
                this.vapidKeys.publicKey,
                this.vapidKeys.privateKey
            );

            this.isInitialized = true;
            this.logger.info('Push notification service initialized');
        } catch (error) {
            this.logger.error('Failed to initialize push notification service:', error);
        }
    }

    getVapidPublicKey() {
        return this.vapidKeys.publicKey;
    }

    async sendPushNotification(subscription, payload, options = {}) {
        if (!this.isInitialized) {
            throw new Error('Push notification service not initialized');
        }

        try {
            const pushOptions = {
                vapidDetails: {
                    subject: this.vapidSubject,
                    publicKey: this.vapidKeys.publicKey,
                    privateKey: this.vapidKeys.privateKey
                },
                TTL: options.ttl || 3600, // Time to live in seconds
                headers: options.headers || {},
                contentEncoding: options.contentEncoding || 'aes128gcm',
                urgency: options.urgency || 'normal' // very-low, low, normal, high
            };

            const result = await webpush.sendNotification(subscription, payload, pushOptions);
            
            this.logger.debug('Push notification sent successfully:', {
                statusCode: result.statusCode,
                headers: result.headers
            });

            return {
                success: true,
                statusCode: result.statusCode,
                headers: result.headers
            };
        } catch (error) {
            this.logger.error('Failed to send push notification:', error);
            
            // Handle different error scenarios
            if (error.statusCode === 410 || error.statusCode === 404) {
                // Subscription is no longer valid
                return {
                    success: false,
                    error: 'invalid_subscription',
                    statusCode: error.statusCode,
                    message: 'Subscription is no longer valid'
                };
            } else if (error.statusCode === 413) {
                // Payload too large
                return {
                    success: false,
                    error: 'payload_too_large',
                    statusCode: error.statusCode,
                    message: 'Push notification payload too large'
                };
            } else if (error.statusCode === 429) {
                // Rate limited
                return {
                    success: false,
                    error: 'rate_limited',
                    statusCode: error.statusCode,
                    message: 'Push service rate limit exceeded'
                };
            } else {
                // Other error
                return {
                    success: false,
                    error: 'send_failed',
                    statusCode: error.statusCode || 500,
                    message: error.message
                };
            }
        }
    }

    createNotificationPayload(notification, options = {}) {
        const payload = {
            title: notification.title,
            body: notification.message,
            icon: options.icon || '/icons/notification-icon-192.png',
            badge: options.badge || '/icons/badge-icon-72.png',
            image: options.image,
            tag: notification.ID || 'default',
            timestamp: Date.now(),
            requireInteraction: notification.priority === 'critical' || options.requireInteraction,
            silent: options.silent || false,
            renotify: options.renotify || false,
            data: {
                notificationId: notification.ID,
                type: notification.type,
                priority: notification.priority,
                category: notification.category,
                url: options.clickUrl,
                actions: notification.actions || [],
                metadata: notification.metadata || {}
            }
        };

        // Add action buttons for high priority notifications
        if (notification.priority === 'high' || notification.priority === 'critical') {
            payload.actions = [
                {
                    action: 'view',
                    title: 'View',
                    icon: '/icons/view-icon.png'
                },
                {
                    action: 'dismiss',
                    title: 'Dismiss',
                    icon: '/icons/dismiss-icon.png'
                }
            ];
        }

        // Add vibration pattern for mobile devices
        if (notification.priority === 'critical') {
            payload.vibrate = [200, 100, 200, 100, 200];
        } else if (notification.priority === 'high') {
            payload.vibrate = [200, 100, 200];
        }

        // Set urgency for delivery optimization
        const urgencyMap = {
            'low': 'low',
            'medium': 'normal',
            'high': 'high',
            'critical': 'high'
        };
        payload.urgency = urgencyMap[notification.priority] || 'normal';

        return JSON.stringify(payload);
    }

    async sendNotificationToUser(userId, notification, userPreferences) {
        if (!userPreferences.pushEnabled || !userPreferences.pushToken) {
            return { success: false, error: 'push_disabled', message: 'Push notifications disabled for user' };
        }

        try {
            // Parse the subscription from the stored token
            const subscription = this.parseSubscription(userPreferences.pushToken, userPreferences.deviceInfo);
            
            // Create notification payload
            const payload = this.createNotificationPayload(notification, {
                clickUrl: this.generateNotificationUrl(notification),
                icon: this.getNotificationIcon(notification.type),
                requireInteraction: notification.priority === 'critical'
            });

            // Send push notification
            const result = await this.sendPushNotification(subscription, payload, {
                urgency: this.mapPriorityToUrgency(notification.priority),
                ttl: this.getTTL(notification.priority)
            });

            return result;
        } catch (error) {
            this.logger.error(`Failed to send push notification to user ${userId}:`, error);
            return {
                success: false,
                error: 'send_failed',
                message: error.message
            };
        }
    }

    parseSubscription(pushToken, deviceInfo) {
        try {
            // If pushToken is already a subscription object
            if (typeof pushToken === 'object' && pushToken.endpoint) {
                return pushToken;
            }

            // If pushToken is a JSON string
            if (typeof pushToken === 'string' && pushToken.startsWith('{')) {
                return JSON.parse(pushToken);
            }

            // If pushToken is just an endpoint, create basic subscription
            if (typeof pushToken === 'string') {
                const subscription = {
                    endpoint: pushToken
                };

                // Try to extract keys from deviceInfo if available
                if (deviceInfo && typeof deviceInfo === 'object') {
                    if (deviceInfo.keys) {
                        subscription.keys = deviceInfo.keys;
                    }
                } else if (typeof deviceInfo === 'string') {
                    try {
                        const parsedDeviceInfo = JSON.parse(deviceInfo);
                        if (parsedDeviceInfo.keys) {
                            subscription.keys = parsedDeviceInfo.keys;
                        }
                    } catch (e) {
                        // Ignore parsing errors
                    }
                }

                return subscription;
            }

            throw new Error('Invalid push token format');
        } catch (error) {
            this.logger.error('Failed to parse subscription:', error);
            throw new Error('Invalid subscription format');
        }
    }

    generateNotificationUrl(notification) {
        const baseUrl = process.env.APP_BASE_URL || 'https://localhost:3000';
        return `${baseUrl}/notifications?id=${notification.ID}`;
    }

    getNotificationIcon(type) {
        const iconMap = {
            'info': '/icons/info-icon-192.png',
            'warning': '/icons/warning-icon-192.png',
            'error': '/icons/error-icon-192.png',
            'success': '/icons/success-icon-192.png',
            'system': '/icons/system-icon-192.png'
        };
        
        return iconMap[type] || '/icons/notification-icon-192.png';
    }

    mapPriorityToUrgency(priority) {
        const urgencyMap = {
            'low': 'low',
            'medium': 'normal',
            'high': 'high',
            'critical': 'high'
        };
        return urgencyMap[priority] || 'normal';
    }

    getTTL(priority) {
        // Time to live in seconds
        const ttlMap = {
            'low': 86400,    // 24 hours
            'medium': 3600,  // 1 hour
            'high': 1800,    // 30 minutes
            'critical': 300  // 5 minutes
        };
        return ttlMap[priority] || 3600;
    }

    async validateSubscription(subscription) {
        try {
            // Try to send a test notification
            const testPayload = JSON.stringify({
                title: 'Test',
                body: 'Connection test',
                tag: 'test',
                data: { test: true }
            });

            const result = await this.sendPushNotification(subscription, testPayload, {
                ttl: 60,
                urgency: 'low'
            });

            return result.success;
        } catch (error) {
            this.logger.warn('Subscription validation failed:', error);
            return false;
        }
    }

    async cleanupInvalidSubscriptions(subscriptions) {
        const validSubscriptions = [];
        const invalidSubscriptions = [];

        for (const subscription of subscriptions) {
            const isValid = await this.validateSubscription(subscription);
            if (isValid) {
                validSubscriptions.push(subscription);
            } else {
                invalidSubscriptions.push(subscription);
            }
        }

        if (invalidSubscriptions.length > 0) {
            this.logger.info(`Cleaned up ${invalidSubscriptions.length} invalid subscriptions`);
        }

        return {
            valid: validSubscriptions,
            invalid: invalidSubscriptions
        };
    }

    // Batch send notifications
    async sendBatchNotifications(notifications) {
        const results = [];
        const batchSize = 100; // Process in batches to avoid overwhelming the push service
        
        for (let i = 0; i < notifications.length; i += batchSize) {
            const batch = notifications.slice(i, i + batchSize);
            const batchPromises = batch.map(async (notificationData) => {
                try {
                    const result = await this.sendNotificationToUser(
                        notificationData.userId,
                        notificationData.notification,
                        notificationData.preferences
                    );
                    return {
                        userId: notificationData.userId,
                        notificationId: notificationData.notification.ID,
                        result: result
                    };
                } catch (error) {
                    return {
                        userId: notificationData.userId,
                        notificationId: notificationData.notification.ID,
                        result: {
                            success: false,
                            error: 'batch_send_failed',
                            message: error.message
                        }
                    };
                }
            });

            const batchResults = await Promise.allSettled(batchPromises);
            results.push(...batchResults.map(r => r.value || r.reason));

            // Small delay between batches to respect rate limits
            if (i + batchSize < notifications.length) {
                await new Promise(resolve => setTimeout(resolve, 100));
            }
        }

        const successful = results.filter(r => r.result.success).length;
        const failed = results.length - successful;

        this.logger.info(`Batch notification send complete: ${successful} successful, ${failed} failed`);

        return {
            total: results.length,
            successful: successful,
            failed: failed,
            results: results
        };
    }

    // Get service worker registration script
    getServiceWorkerScript() {
        return `
// Service Worker for A2A Notification System
self.addEventListener('install', function(event) {
    console.log('Service Worker installing');
    self.skipWaiting();
});

self.addEventListener('activate', function(event) {
    console.log('Service Worker activating');
    event.waitUntil(self.clients.claim());
});

self.addEventListener('push', function(event) {
    console.log('Push message received:', event);
    
    let notificationData = {};
    
    if (event.data) {
        try {
            notificationData = event.data.json();
        } catch (e) {
            notificationData = {
                title: 'New Notification',
                body: event.data.text() || 'You have a new notification',
                icon: '/icons/notification-icon-192.png',
                badge: '/icons/badge-icon-72.png'
            };
        }
    }
    
    const notificationOptions = {
        body: notificationData.body,
        icon: notificationData.icon || '/icons/notification-icon-192.png',
        badge: notificationData.badge || '/icons/badge-icon-72.png',
        image: notificationData.image,
        tag: notificationData.tag || 'default',
        timestamp: notificationData.timestamp || Date.now(),
        requireInteraction: notificationData.requireInteraction || false,
        silent: notificationData.silent || false,
        renotify: notificationData.renotify || false,
        actions: notificationData.actions || [],
        data: notificationData.data || {},
        vibrate: notificationData.vibrate
    };
    
    event.waitUntil(
        self.registration.showNotification(notificationData.title || 'Notification', notificationOptions)
    );
});

self.addEventListener('notificationclick', function(event) {
    console.log('Notification clicked:', event);
    
    event.notification.close();
    
    const notificationData = event.notification.data || {};
    let targetUrl = '/notifications';
    
    if (event.action === 'view' || !event.action) {
        if (notificationData.url) {
            targetUrl = notificationData.url;
        } else if (notificationData.notificationId) {
            targetUrl = \`/notifications?id=\${notificationData.notificationId}\`;
        }
    } else if (event.action === 'dismiss') {
        // Handle dismiss action
        return;
    }
    
    event.waitUntil(
        clients.matchAll({
            type: 'window',
            includeUncontrolled: true
        }).then(function(clientList) {
            // Check if there's already a window/tab open with the target URL
            for (let i = 0; i < clientList.length; i++) {
                const client = clientList[i];
                if (client.url.includes('/notifications') && 'focus' in client) {
                    return client.focus();
                }
            }
            
            // If no existing window, open a new one
            if (clients.openWindow) {
                return clients.openWindow(targetUrl);
            }
        })
    );
});

self.addEventListener('notificationclose', function(event) {
    console.log('Notification closed:', event);
    // Handle notification close if needed
});
`;
    }

    // REST API handlers
    getRESTHandlers() {
        return {
            // GET /api/push/vapid-public-key
            getVapidPublicKey: (req, res) => {
                res.json({
                    success: true,
                    publicKey: this.getVapidPublicKey()
                });
            },

            // GET /api/push/service-worker.js
            getServiceWorker: (req, res) => {
                res.setHeader('Content-Type', 'application/javascript');
                res.setHeader('Service-Worker-Allowed', '/');
                res.send(this.getServiceWorkerScript());
            },

            // POST /api/push/test
            testPushNotification: async (req, res) => {
                try {
                    const { subscription } = req.body;
                    
                    if (!subscription || !subscription.endpoint) {
                        return res.status(400).json({
                            success: false,
                            error: 'Invalid subscription'
                        });
                    }

                    const testPayload = JSON.stringify({
                        title: 'Test Notification',
                        body: 'This is a test push notification from A2A Network',
                        icon: '/icons/notification-icon-192.png',
                        tag: 'test',
                        data: { test: true }
                    });

                    const result = await this.sendPushNotification(subscription, testPayload);
                    
                    res.json(result);
                } catch (error) {
                    this.logger.error('Test push notification failed:', error);
                    res.status(500).json({
                        success: false,
                        error: 'Test failed',
                        message: error.message
                    });
                }
            },

            // POST /api/push/validate-subscription
            validateSubscription: async (req, res) => {
                try {
                    const { subscription } = req.body;
                    
                    if (!subscription) {
                        return res.status(400).json({
                            success: false,
                            error: 'Subscription required'
                        });
                    }

                    const isValid = await this.validateSubscription(subscription);
                    
                    res.json({
                        success: true,
                        valid: isValid
                    });
                } catch (error) {
                    res.status(500).json({
                        success: false,
                        error: 'Validation failed',
                        message: error.message
                    });
                }
            }
        };
    }
}

module.exports = PushNotificationService;