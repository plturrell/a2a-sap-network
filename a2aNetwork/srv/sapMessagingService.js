/**
 * @fileoverview SAP Enterprise Messaging Service
 * @description High-performance messaging service for event publishing, subscriptions,
 * real-time updates, and enterprise message queue management with SAP Event Mesh integration.
 * @module sapMessagingService
 * @since 1.0.0
 * @author A2A Network Team
 * @namespace a2a.srv
 */

const cds = require('@sap/cds');
const { SELECT, INSERT, UPDATE, DELETE, UPSERT } = cds.ql;
const { v4: uuidv4 } = require('uuid');

/**
 * Implementation of the Messaging Service
 * Handles event publishing, subscriptions, and real-time updates
 */
module.exports = class MessagingService extends cds.Service {

    async init() {
        const { Agents, Services, Workflows, NetworkStats } = cds.entities('a2a.network');

        // Store active subscriptions
        this.subscriptions = new Map();

        // Initialize event handlers
        this._setupEventHandlers();

        // Action implementations
        this.on('publishAgentEvent', this._publishAgentEvent);
        this.on('publishServiceEvent', this._publishServiceEvent);
        this.on('publishWorkflowEvent', this._publishWorkflowEvent);
        this.on('publishNetworkEvent', this._publishNetworkEvent);
        this.on('subscribe', this._subscribe);
        this.on('unsubscribe', this._unsubscribe);
        this.on('getQueueStatus', this._getQueueStatus);
        this.on('retryFailedMessages', this._retryFailedMessages);

        // Connect to Enterprise Messaging if available
        try {
            this.messaging = await cds.connect.to('messaging');
            this.log.info('Connected to Enterprise Messaging Service');
        } catch (e) {
            this.log.warn('Enterprise Messaging not available, using in-memory events');
        }

        return super.init();
    }

    _setupEventHandlers() {
        // Listen to database changes and emit corresponding events
        const { Agents, Services, Workflows, AgentPerformance } = cds.entities('a2a.network');

        // Agent events
        this.after('CREATE', Agents, async (agent) => {
            await this.emit('AgentRegistered', {
                agentId: agent.ID,
                address: agent.address,
                name: agent.name,
                timestamp: new Date()
            });
        });

        this.after('UPDATE', Agents, async (agent) => {
            if (agent.isActive === false) {
                await this.emit('AgentDeactivated', {
                    agentId: agent.ID,
                    reason: 'Manual deactivation',
                    timestamp: new Date()
                });
            }
        });

        // Service events
        this.after('CREATE', Services, async (service) => {
            await this.emit('ServiceCreated', {
                serviceId: service.ID,
                providerId: service.provider_ID,
                name: service.name,
                category: service.category,
                pricePerCall: service.pricePerCall,
                timestamp: new Date()
            });
        });

        // Performance monitoring
        this.after('UPDATE', AgentPerformance, async (performance, req) => {
            const oldData = req._.odataReq?.getPreviousData?.();
            if (oldData && oldData.reputationScore !== performance.reputationScore) {
                await this.emit('ReputationUpdated', {
                    agentId: performance.agent_ID,
                    oldScore: oldData.reputationScore,
                    newScore: performance.reputationScore,
                    reason: 'Performance metrics update',
                    timestamp: new Date()
                });
            }
        });
    }

    async _publishAgentEvent(req) {
        try {
            const { eventType, payload } = req.data;
            const event = JSON.parse(payload);

            await this.emit(eventType, {
                ...event,
                timestamp: new Date()
            });

            // Publish to external messaging if available
            if (this.messaging) {
                await this.messaging.emit(`a2a.agent.${eventType}`, event);
            }

            return `Event ${eventType} published successfully`;
        } catch (error) {
            this.log.error('Failed to publish agent event:', error);
            throw new Error(`Failed to publish agent event: ${error.message}`);
        }
    }

    async _publishServiceEvent(req) {
        const { eventType, payload } = req.data;
        const event = JSON.parse(payload);

        await this.emit(eventType, {
            ...event,
            timestamp: new Date()
        });

        if (this.messaging) {
            await this.messaging.emit(`a2a.service.${eventType}`, event);
        }

        return `Event ${eventType} published successfully`;
    }

    async _publishWorkflowEvent(req) {
        const { eventType, payload } = req.data;
        const event = JSON.parse(payload);

        await this.emit(eventType, {
            ...event,
            timestamp: new Date()
        });

        if (this.messaging) {
            await this.messaging.emit(`a2a.workflow.${eventType}`, event);
        }

        return `Event ${eventType} published successfully`;
    }

    async _publishNetworkEvent(req) {
        const { eventType, payload } = req.data;
        const event = JSON.parse(payload);

        await this.emit(eventType, {
            ...event,
            timestamp: new Date()
        });

        if (this.messaging) {
            await this.messaging.emit(`a2a.network.${eventType}`, event);
        }

        return `Event ${eventType} published successfully`;
    }

    async _subscribe(req) {
        const { topics } = req.data;
        const subscriptionId = uuidv4();

        // Store subscription
        this.subscriptions.set(subscriptionId, {
            topics,
            createdAt: new Date(),
            messageCount: 0
        });

        // Subscribe to external messaging topics if available
        if (this.messaging) {
            for (const topic of topics) {
                await this.messaging.on(`a2a.*.${topic}`, async (msg) => {
                    const subscription = this.subscriptions.get(subscriptionId);
                    if (subscription) {
                        subscription.messageCount++;
                        // Forward to WebSocket if connected
                        if (cds.io) {
                            cds.io.to(subscriptionId).emit(topic, msg);
                        }
                    }
                });
            }
        }

        return {
            subscriptionId,
            topics,
            status: 'active'
        };
    }

    async _unsubscribe(req) {
        const { subscriptionId } = req.data;

        if (this.subscriptions.has(subscriptionId)) {
            this.subscriptions.delete(subscriptionId);

            // Clean up WebSocket room
            if (cds.io) {
                const sockets = await cds.io.in(subscriptionId).fetchSockets();
                for (const socket of sockets) {
                    socket.leave(subscriptionId);
                }
            }

            return true;
        }

        return false;
    }

    async _getQueueStatus(req) {
        try {
            // Get real queue metrics from Redis or message queue service
            const redisClient = this.redisClient || require('../middleware/sapCacheMiddleware').getRedisClient();
            if (!redisClient) {
                throw new Error('Redis client not available for queue metrics');
            }

            // Get real metrics from Redis
            const [pending, processed, failed] = await Promise.all([
                redisClient.llen('message_queue:pending'),
                redisClient.get('queue_stats:processed_today') || '0',
                redisClient.get('queue_stats:failed_today') || '0'
            ]);

            const stats = {
                pendingMessages: parseInt(pending) || 0,
                processedToday: parseInt(processed) || 0,
                failedToday: parseInt(failed) || 0,
                queueHealth: 'healthy'
            };

            // Calculate real health status
            if (stats.failedToday > 50) {
                stats.queueHealth = 'degraded';
            } else if (stats.pendingMessages > 500) {
                stats.queueHealth = 'warning';
            }

            return stats;
        } catch (error) {
            this.logger.error('Failed to retrieve real queue status:', error);
            throw new Error(`Queue status unavailable: ${error.message}`);
        }
    }

    async _retryFailedMessages(req) {
        const { since } = req.data;

        try {
            // Get real failed messages from dead letter queue
            const redisClient = this.redisClient || require('../middleware/sapCacheMiddleware').getRedisClient();
            if (!redisClient) {
                throw new Error('Redis client not available for retry operations');
            }

            // Get failed messages from dead letter queue
            const failedMessages = await redisClient.lrange('message_queue:failed', 0, -1);
            let retriedCount = 0;
            let successCount = 0;
            let failedCount = 0;

            for (const messageStr of failedMessages) {
                try {
                    const message = JSON.parse(messageStr);

                    // Check if message should be retried based on timestamp
                    if (since && new Date(message.failedAt) < new Date(since)) {
                        continue;
                    }

                    // Attempt to reprocess the message
                    retriedCount++;
                    const success = await this._reprocessMessage(message);

                    if (success) {
                        successCount++;
                        // Remove from failed queue
                        await redisClient.lrem('message_queue:failed', 1, messageStr);
                    } else {
                        failedCount++;
                    }
                } catch (error) {
                    this.logger.error('Failed to retry message:', error);
                    failedCount++;
                }
            }

            return {
                retriedCount,
                successCount,
                failedCount
            };
        } catch (error) {
            this.logger.error('Failed to retry failed messages:', error);
            throw new Error(`Message retry failed: ${error.message}`);
        }
    }

    async _reprocessMessage(message) {
        try {
            // Implement actual message reprocessing logic
            // This would depend on your message processing pipeline

            // Example: Send message back to processing queue
            const redisClient = this.redisClient || require('../middleware/sapCacheMiddleware').getRedisClient();
            await redisClient.rpush('message_queue:pending', JSON.stringify(message));

            // Update retry count
            message.retryCount = (message.retryCount || 0) + 1;
            message.retriedAt = new Date().toISOString();

            return true; // Return true if successfully queued for reprocessing
        } catch (error) {
            this.logger.error('Failed to reprocess message:', error);
            return false;
        }
    }

    // Helper method to broadcast events via WebSocket
    async broadcastEvent(eventType, data) {
        if (cds.io) {
            cds.io.emit(eventType, data);
        }

        // Also emit through CAP events
        await this.emit(eventType, data);
    }

    // Monitor network health and emit alerts
    async monitorNetworkHealth() {
        const { NetworkStats } = cds.entities('a2a.network');

        const [latest] = await SELECT.from(NetworkStats)
            .orderBy('validFrom desc')
            .limit(1);

        if (latest && latest.networkLoad > 0.8) {
            await this.emit('NetworkLoadHigh', {
                currentLoad: latest.networkLoad,
                threshold: 0.8,
                recommendedAction: 'Scale up agent resources',
                timestamp: new Date()
            });
        }
    }
};