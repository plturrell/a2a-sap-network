/**
 * @fileoverview Message Persistence Layer for A2A Network
 * @description Dedicated message storage with Redis/PostgreSQL persistence,
 * message archiving, search capabilities, and reliable delivery guarantees
 * @module messagePersistence
 * @since 2.0.0
 * @author A2A Network Team
 */

const cds = require('@sap/cds');
const Redis = require('ioredis');
const { v4: uuidv4 } = require('uuid');
const crypto = require('crypto');

/**
 * Message Persistence Service
 * Provides reliable message storage, archiving, and retrieval
 */
class MessagePersistenceService extends cds.Service {
    
    async init() {
        this.log = cds.log('message-persistence');
        
        // Initialize Redis for high-performance message caching
        this.redis = new Redis({
            host: process.env.REDIS_HOST || 'localhost',
            port: process.env.REDIS_PORT || 6379,
            password: process.env.REDIS_PASSWORD,
            db: 1, // Use separate DB for message persistence
            keyPrefix: 'a2a:msg:',
            retryDelayOnFailover: 100,
            maxRetriesPerRequest: 3
        });

        // Database entities
        const { Messages, MessageArchive, MessageMetrics } = cds.entities('a2a.messaging');
        this.entities = { Messages, MessageArchive, MessageMetrics };
        
        // Message retention settings
        this.config = {
            redisRetentionHours: 24,        // Keep in Redis for 24 hours
            dbRetentionDays: 90,            // Keep in DB for 90 days
            archiveRetentionYears: 7,       // Archive for 7 years
            batchSize: 100,                 // Batch processing size
            compressionThreshold: 1024      // Compress messages > 1KB
        };

        // Initialize cleanup tasks
        this._initializeCleanupTasks();
        
        // Register handlers
        this._registerHandlers();
        
        this.log.info('Message Persistence Service initialized');
        return super.init();
    }

    _registerHandlers() {
        // Core persistence operations
        this.on('persistMessage', this._persistMessage.bind(this));
        this.on('retrieveMessage', this._retrieveMessage.bind(this));
        this.on('searchMessages', this._searchMessages.bind(this));
        this.on('archiveMessages', this._archiveMessages.bind(this));
        this.on('getMessageHistory', this._getMessageHistory.bind(this));
        
        // Bulk operations
        this.on('persistBulkMessages', this._persistBulkMessages.bind(this));
        this.on('retrieveBulkMessages', this._retrieveBulkMessages.bind(this));
        
        // Analytics
        this.on('getMessageMetrics', this._getMessageMetrics.bind(this));
        this.on('getStorageStats', this._getStorageStats.bind(this));
    }

    /**
     * Persist a message with multi-layer storage strategy
     */
    async _persistMessage(req) {
        const { messageId, content, metadata, priority = 'normal', ttl } = req.data;
        
        try {
            const timestamp = new Date().toISOString();
            const messageData = {
                messageId,
                content: await this._compressIfNeeded(content),
                metadata: {
                    ...metadata,
                    persistedAt: timestamp,
                    compressed: content.length > this.config.compressionThreshold
                },
                priority,
                ttl: ttl || this._calculateTTL(priority),
                status: 'active',
                createdAt: timestamp,
                updatedAt: timestamp
            };

            // Store in Redis for fast access
            await this._storeInRedis(messageId, messageData);
            
            // Store in database for persistence
            await this._storeInDatabase(messageData);
            
            // Update metrics
            await this._updateMessageMetrics('persisted', priority);
            
            this.log.info(`Message persisted: ${messageId}`, { 
                size: content.length, 
                compressed: messageData.metadata.compressed 
            });
            
            return {
                success: true,
                messageId,
                location: 'redis+db',
                ttl: messageData.ttl
            };
            
        } catch (error) {
            this.log.error(`Failed to persist message ${messageId}:`, error);
            throw new Error(`Message persistence failed: ${error.message}`);
        }
    }

    /**
     * Retrieve message with intelligent caching
     */
    async _retrieveMessage(req) {
        const { messageId, includeMetadata = true } = req.data;
        
        try {
            // Try Redis first (fastest)
            let message = await this._retrieveFromRedis(messageId);
            
            if (!message) {
                // Fallback to database
                message = await this._retrieveFromDatabase(messageId);
                
                if (message) {
                    // Cache back to Redis for future access
                    await this._storeInRedis(messageId, message, 3600); // 1 hour cache
                }
            }

            if (!message) {
                // Final fallback to archive
                message = await this._retrieveFromArchive(messageId);
            }

            if (!message) {
                return { success: false, error: 'Message not found' };
            }

            // Decompress if needed
            if (message.metadata?.compressed) {
                message.content = await this._decompress(message.content);
            }

            // Update access metrics
            await this._updateMessageMetrics('retrieved', message.priority);
            
            return {
                success: true,
                message: includeMetadata ? message : { 
                    messageId: message.messageId, 
                    content: message.content 
                }
            };
            
        } catch (error) {
            this.log.error(`Failed to retrieve message ${messageId}:`, error);
            throw new Error(`Message retrieval failed: ${error.message}`);
        }
    }

    /**
     * Search messages with full-text search and filtering
     */
    async _searchMessages(req) {
        const { 
            query, 
            filters = {}, 
            sortBy = 'createdAt', 
            sortOrder = 'desc',
            limit = 50, 
            offset = 0 
        } = req.data;
        
        try {
            const { Messages } = this.entities;
            
            // Build search query
            let searchQuery = SELECT.from(Messages);
            
            // Add text search
            if (query) {
                searchQuery = searchQuery.where(`content LIKE '%${query}%' OR metadata LIKE '%${query}%'`);
            }
            
            // Add filters
            Object.entries(filters).forEach(([key, value]) => {
                if (value !== undefined) {
                    searchQuery = searchQuery.and({ [key]: value });
                }
            });
            
            // Add sorting and pagination
            searchQuery = searchQuery
                .orderBy(`${sortBy} ${sortOrder}`)
                .limit(limit, offset);
            
            const messages = await searchQuery;
            
            // Get total count for pagination
            const countQuery = SELECT.from(Messages).columns('COUNT(*) as total');
            if (query) {
                countQuery.where(`content LIKE '%${query}%' OR metadata LIKE '%${query}%'`);
            }
            const [{ total }] = await countQuery;
            
            return {
                success: true,
                messages,
                pagination: {
                    total,
                    limit,
                    offset,
                    pages: Math.ceil(total / limit)
                }
            };
            
        } catch (error) {
            this.log.error('Message search failed:', error);
            throw new Error(`Message search failed: ${error.message}`);
        }
    }

    /**
     * Archive old messages to long-term storage
     */
    async _archiveMessages(req) {
        const { olderThanDays = 90, batchSize = 100 } = req.data;
        
        try {
            const { Messages, MessageArchive } = this.entities;
            const cutoffDate = new Date(Date.now() - (olderThanDays * 24 * 60 * 60 * 1000));
            
            let archived = 0;
            let hasMore = true;
            
            while (hasMore) {
                // Get batch of old messages
                const messages = await SELECT.from(Messages)
                    .where({ createdAt: { '<': cutoffDate } })
                    .limit(batchSize);
                
                if (messages.length === 0) {
                    hasMore = false;
                    break;
                }
                
                // Compress and archive messages
                const archiveData = messages.map(msg => ({
                    messageId: msg.messageId,
                    originalContent: msg.content,
                    compressedContent: this._compress(JSON.stringify(msg)),
                    metadata: {
                        ...msg.metadata,
                        archivedAt: new Date().toISOString(),
                        originalSize: JSON.stringify(msg).length
                    },
                    archivedAt: new Date(),
                    originalCreatedAt: msg.createdAt
                }));
                
                // Insert into archive
                await INSERT.into(MessageArchive).entries(archiveData);
                
                // Remove from active storage
                const messageIds = messages.map(m => m.messageId);
                await DELETE.from(Messages).where({ messageId: { in: messageIds } });
                
                // Remove from Redis
                await this._removeFromRedis(messageIds);
                
                archived += messages.length;
                
                this.log.info(`Archived batch: ${messages.length} messages`);
            }
            
            await this._updateMessageMetrics('archived', null, archived);
            
            return {
                success: true,
                archivedCount: archived,
                cutoffDate: cutoffDate.toISOString()
            };
            
        } catch (error) {
            this.log.error('Message archiving failed:', error);
            throw new Error(`Message archiving failed: ${error.message}`);
        }
    }

    /**
     * Get message history for an agent or conversation
     */
    async _getMessageHistory(req) {
        const { 
            agentId, 
            conversationId, 
            startDate, 
            endDate, 
            limit = 100 
        } = req.data;
        
        try {
            const { Messages } = this.entities;
            
            let query = SELECT.from(Messages);
            
            if (agentId) {
                query = query.where({
                    or: [
                        { 'metadata.fromAgent': agentId },
                        { 'metadata.toAgent': agentId }
                    ]
                });
            }
            
            if (conversationId) {
                query = query.and({ 'metadata.conversationId': conversationId });
            }
            
            if (startDate) {
                query = query.and({ createdAt: { '>=': new Date(startDate) } });
            }
            
            if (endDate) {
                query = query.and({ createdAt: { '<=': new Date(endDate) } });
            }
            
            const messages = await query
                .orderBy('createdAt asc')
                .limit(limit);
            
            return {
                success: true,
                messages,
                count: messages.length
            };
            
        } catch (error) {
            this.log.error('Failed to get message history:', error);
            throw new Error(`Message history retrieval failed: ${error.message}`);
        }
    }

    // Redis operations
    async _storeInRedis(messageId, messageData, customTTL = null) {
        const ttl = customTTL || this.config.redisRetentionHours * 3600;
        const key = `message:${messageId}`;
        
        await this.redis.setex(key, ttl, JSON.stringify(messageData));
        
        // Store in priority queue if high priority
        if (messageData.priority === 'high' || messageData.priority === 'critical') {
            await this.redis.zadd('priority_messages', Date.now(), messageId);
        }
    }

    async _retrieveFromRedis(messageId) {
        const key = `message:${messageId}`;
        const data = await this.redis.get(key);
        return data ? JSON.parse(data) : null;
    }

    async _removeFromRedis(messageIds) {
        const keys = messageIds.map(id => `message:${id}`);
        if (keys.length > 0) {
            await this.redis.del(...keys);
            // Remove from priority queue
            await this.redis.zrem('priority_messages', ...messageIds);
        }
    }

    // Database operations
    async _storeInDatabase(messageData) {
        const { Messages } = this.entities;
        await INSERT.into(Messages).entries(messageData);
    }

    async _retrieveFromDatabase(messageId) {
        const { Messages } = this.entities;
        const messages = await SELECT.from(Messages).where({ messageId });
        return messages[0] || null;
    }

    async _retrieveFromArchive(messageId) {
        const { MessageArchive } = this.entities;
        const archived = await SELECT.from(MessageArchive).where({ messageId });
        
        if (archived[0]) {
            return {
                ...JSON.parse(this._decompress(archived[0].compressedContent)),
                archived: true,
                archivedAt: archived[0].archivedAt
            };
        }
        
        return null;
    }

    // Utility methods
    async _compressIfNeeded(content) {
        if (content.length > this.config.compressionThreshold) {
            return this._compress(content);
        }
        return content;
    }

    _compress(data) {
        const zlib = require('zlib');
        return zlib.gzipSync(data).toString('base64');
    }

    _decompress(compressedData) {
        const zlib = require('zlib');
        return zlib.gunzipSync(Buffer.from(compressedData, 'base64')).toString();
    }

    _calculateTTL(priority) {
        switch (priority) {
            case 'critical': return 7 * 24 * 3600; // 7 days
            case 'high': return 3 * 24 * 3600;     // 3 days
            case 'normal': return 24 * 3600;       // 1 day
            case 'low': return 12 * 3600;          // 12 hours
            default: return 24 * 3600;
        }
    }

    async _updateMessageMetrics(operation, priority, count = 1) {
        const { MessageMetrics } = this.entities;
        const date = new Date().toISOString().split('T')[0];
        
        await UPSERT.into(MessageMetrics).entries({
            date,
            operation,
            priority: priority || 'unknown',
            count,
            timestamp: new Date()
        });
    }

    _initializeCleanupTasks() {
        // Cleanup Redis expired keys every hour
        setInterval(async () => {
            try {
                const expiredKeys = await this.redis.keys('message:*');
                const pipeline = this.redis.pipeline();
                
                for (const key of expiredKeys) {
                    const ttl = await this.redis.ttl(key);
                    if (ttl <= 0) {
                        pipeline.del(key);
                    }
                }
                
                await pipeline.exec();
                this.log.debug(`Cleaned up ${expiredKeys.length} expired Redis keys`);
            } catch (error) {
                this.log.error('Redis cleanup failed:', error);
            }
        }, 3600000); // 1 hour
        
        // Archive old messages daily
        setInterval(async () => {
            try {
                await this._archiveMessages({ data: {} });
            } catch (error) {
                this.log.error('Automated archiving failed:', error);
            }
        }, 24 * 3600000); // 24 hours
    }

    /**
     * Get storage statistics and health metrics
     */
    async _getStorageStats() {
        try {
            const { Messages, MessageArchive } = this.entities;
            
            // Active messages count
            const [activeCount] = await SELECT.from(Messages).columns('COUNT(*) as count');
            
            // Archived messages count
            const [archivedCount] = await SELECT.from(MessageArchive).columns('COUNT(*) as count');
            
            // Redis stats
            const redisInfo = await this.redis.info('memory');
            const redisKeyCount = await this.redis.dbsize();
            
            return {
                success: true,
                stats: {
                    activeMessages: activeCount.count,
                    archivedMessages: archivedCount.count,
                    redisKeys: redisKeyCount,
                    redisMemory: this._parseRedisMemory(redisInfo),
                    lastUpdated: new Date().toISOString()
                }
            };
            
        } catch (error) {
            this.log.error('Failed to get storage stats:', error);
            throw new Error(`Storage stats retrieval failed: ${error.message}`);
        }
    }

    _parseRedisMemory(info) {
        const lines = info.split('\n');
        const memoryLine = lines.find(line => line.startsWith('used_memory_human:'));
        return memoryLine ? memoryLine.split(':')[1].trim() : 'Unknown';
    }
}

module.exports = MessagePersistenceService;