/**
 * @fileoverview Dead Letter Queue System for A2A Network
 * @description Advanced failed message handling with retry policies,
 * poison message detection, and comprehensive failure analytics
 * @module deadLetterQueue
 * @since 2.0.0
 * @author A2A Network Team
 */

const cds = require('@sap/cds');
const Redis = require('ioredis');
const { v4: uuidv4 } = require('uuid');
const EventEmitter = require('events');

// Track intervals for cleanup
const activeIntervals = new Map();

function stopAllIntervals() {
    for (const [name, intervalId] of activeIntervals) {
        clearInterval(intervalId);
    }
    activeIntervals.clear();
}

function shutdown() {
    stopAllIntervals();
}

// Export cleanup function
module.exports.shutdown = shutdown;


/**
 * Dead Letter Queue Service
 * Handles failed messages with intelligent retry and recovery mechanisms
 */
class DeadLetterQueueService extends cds.Service {
    
    async init() {
        this.log = cds.log('dead-letter-queue');
        this.eventEmitter = new EventEmitter();
        
        // Initialize Redis for DLQ operations
        this.redis = new Redis({
            host: process.env.REDIS_HOST || 'localhost',
            port: process.env.REDIS_PORT || 6379,
            password: process.env.REDIS_PASSWORD,
            db: 2, // Use separate DB for DLQ
            keyPrefix: 'a2a:dlq:',
            retryDelayOnFailover: 100,
            maxRetriesPerRequest: 3
        });

        // Database entities
        const { 
            FailedMessages, 
            RetryPolicies, 
            PoisonMessages,
            FailureAnalytics 
        } = cds.entities('a2a.dlq');
        this.entities = { FailedMessages, RetryPolicies, PoisonMessages, FailureAnalytics };
        
        // DLQ Configuration
        this.config = {
            defaultRetryPolicy: {
                maxRetries: 3,
                initialDelay: 1000,     // 1 second
                maxDelay: 300000,       // 5 minutes
                backoffMultiplier: 2,   // Exponential backoff
                jitterMax: 1000         // Max jitter in ms
            },
            poisonThreshold: 5,         // Mark as poison after 5 consecutive failures
            batchProcessSize: 50,       // Process DLQ in batches
            retryIntervalMs: 60000,     // Check for retries every minute
            analyticsRetentionDays: 30, // Keep analytics for 30 days
            alertThresholds: {
                highFailureRate: 0.1,   // 10% failure rate
                poisonMessageCount: 10  // Alert when 10+ poison messages
            }
        };

        // Initialize retry policies
        await this._initializeRetryPolicies();
        
        // Start background processors
        this._startRetryProcessor();
        this._startAnalyticsProcessor();
        
        // Register handlers
        this._registerHandlers();
        
        this.log.info('Dead Letter Queue Service initialized');
        return super.init();
    }

    _registerHandlers() {
        // Core DLQ operations
        this.on('addToDeadLetter', this._addToDeadLetter.bind(this));
        this.on('retryMessage', this._retryMessage.bind(this));
        this.on('retryBatch', this._retryBatch.bind(this));
        this.on('markAsPoison', this._markAsPoison.bind(this));
        
        // Query operations
        this.on('getFailedMessages', this._getFailedMessages.bind(this));
        this.on('getPoisonMessages', this._getPoisonMessages.bind(this));
        this.on('getFailureAnalytics', this._getFailureAnalytics.bind(this));
        
        // Management operations
        this.on('updateRetryPolicy', this._updateRetryPolicy.bind(this));
        this.on('purgeOldMessages', this._purgeOldMessages.bind(this));
        this.on('reprocessPoisonMessage', this._reprocessPoisonMessage.bind(this));
        
        // Health and monitoring
        this.on('getDLQHealth', this._getDLQHealth.bind(this));
        this.on('getDLQStats', this._getDLQStats.bind(this));
    }

    /**
     * Add message to dead letter queue
     */
    async _addToDeadLetter(req) {
        const { 
            messageId, 
            originalMessage, 
            failureReason, 
            failureDetails,
            retryCount = 0,
            lastAttemptAt,
            agentId,
            messageType,
            priority = 'normal'
        } = req.data;

        try {
            const timestamp = new Date();
            const dlqEntry = {
                dlqId: uuidv4(),
                messageId,
                originalMessage: JSON.stringify(originalMessage),
                failureReason,
                failureDetails: JSON.stringify(failureDetails),
                retryCount,
                maxRetries: await this._getMaxRetries(messageType, agentId),
                nextRetryAt: this._calculateNextRetry(retryCount, messageType),
                addedAt: timestamp,
                lastAttemptAt: lastAttemptAt || timestamp,
                agentId,
                messageType,
                priority,
                status: 'pending_retry',
                poisonScore: this._calculatePoisonScore(failureReason, retryCount)
            };

            // Store in database
            const { FailedMessages } = this.entities;
            await INSERT.into(FailedMessages).entries(dlqEntry);
            
            // Add to Redis retry queue with sorted set for time-based processing
            const retryAt = dlqEntry.nextRetryAt ? dlqEntry.nextRetryAt.getTime() : Date.now() + 60000;
            await this.redis.zadd('retry_queue', retryAt, dlqEntry.dlqId);
            
            // Check if message should be marked as poison
            if (dlqEntry.poisonScore >= this.config.poisonThreshold) {
                await this._markMessageAsPoison(dlqEntry);
            }
            
            // Update analytics
            await this._updateFailureAnalytics(failureReason, agentId, messageType);
            
            // Emit event for monitoring
            this.eventEmitter.emit('messageAddedToDLQ', {
                messageId,
                failureReason,
                retryCount,
                agentId
            });
            
            this.log.warn(`Message added to DLQ: ${messageId}`, {
                reason: failureReason,
                retryCount,
                nextRetry: dlqEntry.nextRetryAt
            });
            
            return {
                success: true,
                dlqId: dlqEntry.dlqId,
                nextRetryAt: dlqEntry.nextRetryAt,
                status: dlqEntry.status
            };
            
        } catch (error) {
            this.log.error(`Failed to add message to DLQ: ${messageId}`, error);
            throw new Error(`DLQ addition failed: ${error.message}`);
        }
    }

    /**
     * Retry a specific message from DLQ
     */
    async _retryMessage(req) {
        const { dlqId, forceRetry = false } = req.data;
        
        try {
            const { FailedMessages } = this.entities;
            const [failedMessage] = await SELECT.from(FailedMessages).where({ dlqId });
            
            if (!failedMessage) {
                return { success: false, error: 'Message not found in DLQ' };
            }
            
            if (failedMessage.status === 'poison' && !forceRetry) {
                return { success: false, error: 'Message marked as poison - use forceRetry to override' };
            }
            
            // Check if ready for retry
            if (!forceRetry && failedMessage.nextRetryAt > new Date()) {
                return { 
                    success: false, 
                    error: 'Message not ready for retry',
                    nextRetryAt: failedMessage.nextRetryAt
                };
            }
            
            // Attempt to reprocess the message
            const originalMessage = JSON.parse(failedMessage.originalMessage);
            const retryResult = await this._attemptMessageReprocessing(
                originalMessage, 
                failedMessage.agentId,
                failedMessage.messageType
            );
            
            if (retryResult.success) {
                // Remove from DLQ on successful retry
                await DELETE.from(FailedMessages).where({ dlqId });
                await this.redis.zrem('retry_queue', dlqId);
                
                this.log.info(`Message successfully retried: ${failedMessage.messageId}`);
                
                // Update analytics
                await this._updateRetryAnalytics(failedMessage.agentId, true);
                
                return {
                    success: true,
                    messageId: failedMessage.messageId,
                    result: retryResult.result
                };
                
            } else {
                // Update retry count and schedule next retry
                const newRetryCount = failedMessage.retryCount + 1;
                const nextRetryAt = this._calculateNextRetry(newRetryCount, failedMessage.messageType);
                
                let newStatus = 'pending_retry';
                if (newRetryCount >= failedMessage.maxRetries) {
                    newStatus = 'exhausted';
                    await this._markMessageAsPoison({
                        ...failedMessage,
                        retryCount: newRetryCount
                    });
                }
                
                await UPDATE(FailedMessages)
                    .set({
                        retryCount: newRetryCount,
                        nextRetryAt,
                        lastAttemptAt: new Date(),
                        failureDetails: JSON.stringify(retryResult.error),
                        status: newStatus
                    })
                    .where({ dlqId });
                
                if (newStatus === 'pending_retry') {
                    await this.redis.zadd('retry_queue', nextRetryAt.getTime(), dlqId);
                } else {
                    await this.redis.zrem('retry_queue', dlqId);
                }
                
                // Update analytics
                await this._updateRetryAnalytics(failedMessage.agentId, false);
                
                return {
                    success: false,
                    messageId: failedMessage.messageId,
                    error: retryResult.error,
                    retryCount: newRetryCount,
                    nextRetryAt,
                    status: newStatus
                };
            }
            
        } catch (error) {
            this.log.error(`Failed to retry message: ${dlqId}`, error);
            throw new Error(`Message retry failed: ${error.message}`);
        }
    }

    /**
     * Get failed messages with filtering and pagination
     */
    async _getFailedMessages(req) {
        const {
            status = 'all',
            agentId,
            messageType,
            startDate,
            endDate,
            limit = 50,
            offset = 0
        } = req.data;
        
        try {
            const { FailedMessages } = this.entities;
            
            let query = SELECT.from(FailedMessages);
            
            if (status !== 'all') {
                query = query.where({ status });
            }
            
            if (agentId) {
                query = query.and({ agentId });
            }
            
            if (messageType) {
                query = query.and({ messageType });
            }
            
            if (startDate) {
                query = query.and({ addedAt: { '>=': new Date(startDate) } });
            }
            
            if (endDate) {
                query = query.and({ addedAt: { '<=': new Date(endDate) } });
            }
            
            const messages = await query
                .orderBy('addedAt desc')
                .limit(limit, offset);
            
            // Get total count
            const [{ total }] = await SELECT.from(FailedMessages).columns('COUNT(*) as total');
            
            return {
                success: true,
                messages: messages.map(msg => ({
                    ...msg,
                    originalMessage: JSON.parse(msg.originalMessage),
                    failureDetails: JSON.parse(msg.failureDetails || '{}')
                })),
                pagination: {
                    total,
                    limit,
                    offset,
                    pages: Math.ceil(total / limit)
                }
            };
            
        } catch (error) {
            this.log.error('Failed to get DLQ messages:', error);
            throw new Error(`Failed to retrieve DLQ messages: ${error.message}`);
        }
    }

    /**
     * Background processor for retrying messages
     */
    _startRetryProcessor() {
        activeIntervals.set('interval_368', setInterval(async () => {
            try {
                // Get messages ready for retry
                const currentTime = Date.now();
                const readyMessages = await this.redis.zrangebyscore(
                    'retry_queue', '-inf', 
                    currentTime, 
                    'LIMIT', 0, this.config.batchProcessSize
                ));
                
                if (readyMessages.length === 0) return;
                
                this.log.debug(`Processing ${readyMessages.length} messages for retry`);
                
                // Process each message
                const retryPromises = readyMessages.map(async (dlqId) => {
                    try {
                        await this._retryMessage({ data: { dlqId } });
                    } catch (error) {
                        this.log.error(`Retry processor failed for ${dlqId}:`, error);
                    }
                });
                
                await Promise.allSettled(retryPromises);
                
            } catch (error) {
                this.log.error('Retry processor error:', error);
            }
        }, this.config.retryIntervalMs);
    }

    /**
     * Background processor for analytics and health monitoring
     */
    _startAnalyticsProcessor() {
        activeIntervals.set('interval_404', setInterval(async () => {
            try {
                // Generate failure rate alerts
                const stats = await this._getDLQStats({ data: {} });
                
                if (stats.failureRate > this.config.alertThresholds.highFailureRate) {
                    this.eventEmitter.emit('highFailureRateAlert', {
                        failureRate: stats.failureRate,
                        threshold: this.config.alertThresholds.highFailureRate,
                        timestamp: new Date())
                    });
                }
                
                if (stats.poisonMessageCount > this.config.alertThresholds.poisonMessageCount) {
                    this.eventEmitter.emit('highPoisonMessageAlert', {
                        count: stats.poisonMessageCount,
                        threshold: this.config.alertThresholds.poisonMessageCount,
                        timestamp: new Date()
                    });
                }
                
            } catch (error) {
                this.log.error('Analytics processor error:', error);
            }
        }, 5 * 60000); // Every 5 minutes
    }

    // Utility methods
    async _attemptMessageReprocessing(originalMessage, agentId, messageType) {
        try {
            // This would integrate with your message processing system
            // For now, we'll simulate the reprocessing
            
            // Get the appropriate service/handler based on messageType
            const processingResult = await this._callMessageHandler(
                originalMessage, 
                agentId, 
                messageType
            );
            
            return { success: true, result: processingResult };
            
        } catch (error) {
            return { 
                success: false, 
                error: {
                    message: error.message,
                    stack: error.stack,
                    timestamp: new Date().toISOString()
                }
            };
        }
    }

    async _callMessageHandler(message, agentId, messageType) {
        // Integration point with your message processing system
        // This should call the appropriate service method based on messageType
        
        switch (messageType) {
            case 'agent_communication':
                return await this._processAgentMessage(message, agentId);
            case 'workflow_execution':
                return await this._processWorkflowMessage(message, agentId);
            case 'data_synchronization':
                return await this._processDataSyncMessage(message, agentId);
            default:
                throw new Error(`Unknown message type: ${messageType}`);
        }
    }

    async _processAgentMessage(message, agentId) {
        // Simulate agent message processing
        // In real implementation, this would call the appropriate agent service
        return { processed: true, agentId, messageId: message.messageId };
    }

    async _processWorkflowMessage(message, agentId) {
        // Simulate workflow processing
        return { processed: true, workflowId: message.workflowId };
    }

    async _processDataSyncMessage(message, agentId) {
        // Simulate data sync processing
        return { processed: true, syncId: message.syncId };
    }

    _calculateNextRetry(retryCount, messageType) {
        const policy = this.config.defaultRetryPolicy;
        
        if (retryCount >= policy.maxRetries) {
            return null; // No more retries
        }
        
        // Exponential backoff with jitter
        let delay = Math.min(
            policy.initialDelay * Math.pow(policy.backoffMultiplier, retryCount),
            policy.maxDelay
        );
        
        // Add jitter to prevent thundering herd
        const jitter = Math.random() * policy.jitterMax;
        delay += jitter;
        
        return new Date(Date.now() + delay);
    }

    _calculatePoisonScore(failureReason, retryCount) {
        let score = retryCount;
        
        // Increase score based on failure type
        if (failureReason.includes('timeout')) score += 1;
        if (failureReason.includes('connection')) score += 1;
        if (failureReason.includes('parse') || failureReason.includes('format')) score += 2;
        if (failureReason.includes('auth') || failureReason.includes('permission')) score += 3;
        
        return score;
    }

    async _markMessageAsPoison(failedMessage) {
        try {
            const { PoisonMessages } = this.entities;
            
            await INSERT.into(PoisonMessages).entries({
                poisonId: uuidv4(),
                originalDlqId: failedMessage.dlqId,
                messageId: failedMessage.messageId,
                originalMessage: failedMessage.originalMessage,
                poisonReason: `Exceeded retry limit with poison score: ${failedMessage.poisonScore}`,
                markedAt: new Date(),
                agentId: failedMessage.agentId,
                messageType: failedMessage.messageType,
                totalRetries: failedMessage.retryCount
            });
            
            // Update original message status
            const { FailedMessages } = this.entities;
            await UPDATE(FailedMessages)
                .set({ status: 'poison' })
                .where({ dlqId: failedMessage.dlqId });
            
            // Remove from retry queue
            await this.redis.zrem('retry_queue', failedMessage.dlqId);
            
            this.log.error(`Message marked as poison: ${failedMessage.messageId}`, {
                retryCount: failedMessage.retryCount,
                poisonScore: failedMessage.poisonScore
            });
        } catch (error) {
            this.log.error('Failed to mark message as poison:', error);
            // Still throw the error to maintain the failure chain
            throw new Error(`Failed to mark message as poison: ${error.message}`);
        }
    }

    async _getMaxRetries(messageType, agentId) {
        // Get custom retry policy or use default
        const { RetryPolicies } = this.entities;
        const policies = await SELECT.from(RetryPolicies)
            .where({ messageType, agentId })
            .or({ messageType, agentId: null })
            .orderBy('agentId desc'); // Prefer agent-specific policies
        
        return policies[0]?.maxRetries || this.config.defaultRetryPolicy.maxRetries;
    }

    async _initializeRetryPolicies() {
        // Initialize default retry policies for different message types
        const { RetryPolicies } = this.entities;
        
        const defaultPolicies = [
            {
                messageType: 'agent_communication',
                maxRetries: 3,
                initialDelay: 1000,
                maxDelay: 60000,
                backoffMultiplier: 2
            },
            {
                messageType: 'workflow_execution',
                maxRetries: 5,
                initialDelay: 5000,
                maxDelay: 300000,
                backoffMultiplier: 1.5
            },
            {
                messageType: 'data_synchronization',
                maxRetries: 2,
                initialDelay: 10000,
                maxDelay: 120000,
                backoffMultiplier: 3
            }
        ];
        
        for (const policy of defaultPolicies) {
            await UPSERT.into(RetryPolicies).entries({
                policyId: uuidv4(),
                ...policy,
                createdAt: new Date()
            });
        }
    }

    async _updateFailureAnalytics(failureReason, agentId, messageType) {
        const { FailureAnalytics } = this.entities;
        const date = new Date().toISOString().split('T')[0];
        
        await UPSERT.into(FailureAnalytics).entries({
            date,
            failureReason,
            agentId,
            messageType,
            count: 1,
            timestamp: new Date()
        });
    }

    async _updateRetryAnalytics(agentId, success) {
        const { FailureAnalytics } = this.entities;
        const date = new Date().toISOString().split('T')[0];
        
        await UPSERT.into(FailureAnalytics).entries({
            date,
            failureReason: success ? 'retry_success' : 'retry_failed',
            agentId,
            messageType: 'retry_attempt',
            count: 1,
            timestamp: new Date()
        });
    }

    /**
     * Get DLQ health and statistics
     */
    async _getDLQStats() {
        try {
            const { FailedMessages, PoisonMessages } = this.entities;
            
            // Count messages by status
            const statusCounts = await SELECT.from(FailedMessages)
                .columns('status', 'COUNT(*) as count')
                .groupBy('status');
            
            // Poison message count
            const [poisonCount] = await SELECT.from(PoisonMessages).columns('COUNT(*) as count');
            
            // Recent failure rate (last 24 hours)
            const yesterday = new Date(Date.now() - 24 * 60 * 60 * 1000);
            const [recentFailures] = await SELECT.from(FailedMessages)
                .columns('COUNT(*) as count')
                .where({ addedAt: { '>=': yesterday } });
            
            // Calculate failure rate (simplified)
            const totalRecentMessages = 0; // Real message metrics should be used here
            const failureRate = totalRecentMessages > 0 ? 
                recentFailures.count / totalRecentMessages : 0;
            
            return {
                success: true,
                stats: {
                    statusCounts: statusCounts.reduce((acc, item) => {
                        acc[item.status] = item.count;
                        return acc;
                    }, {}),
                    poisonMessageCount: poisonCount.count,
                    failureRate,
                    recentFailures: recentFailures.count,
                    lastUpdated: new Date().toISOString()
                }
            };
            
        } catch (error) {
            this.log.error('Failed to get DLQ stats:', error);
            throw new Error(`DLQ stats retrieval failed: ${error.message}`);
        }
    }
}

module.exports = DeadLetterQueueService;