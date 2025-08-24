/**
 * Advanced Redis Cache Manager for A2A Platform
 * Provides distributed caching with intelligent invalidation and performance optimization
 */

const Redis = require('redis');
const EventEmitter = require('events');
const crypto = require('crypto');
const { promisify } = require('util');

class A2ARedisCache extends EventEmitter {
    constructor(options = {}) {
        super();
        
        this.config = {
            // Redis connection
            host: options.host || process.env.REDIS_HOST || 'localhost',
            port: options.port || process.env.REDIS_PORT || 6379,
            password: options.password || process.env.REDIS_PASSWORD,
            db: options.db || 0,
            
            // Connection pool
            maxRetriesPerRequest: 3,
            retryDelayOnFailover: 100,
            enableOfflineQueue: false,
            maxmemoryPolicy: 'allkeys-lru',
            
            // Cache behavior
            defaultTTL: options.defaultTTL || 3600, // 1 hour
            keyPrefix: options.keyPrefix || 'a2a:',
            compression: options.compression !== false,
            serialization: options.serialization || 'json',
            
            // Performance settings
            enableMetrics: options.enableMetrics !== false,
            enableDistributedLock: options.enableDistributedLock !== false,
            lockTimeout: options.lockTimeout || 10000,
            
            // Clustering
            cluster: options.cluster || false,
            clusterNodes: options.clusterNodes || [],
            
            ...options
        };
        
        this.client = null;
        this.subscriber = null;
        this.publisher = null;
        this.isConnected = false;
        this.metrics = {
            hits: 0,
            misses: 0,
            sets: 0,
            deletes: 0,
            errors: 0,
            totalRequests: 0
        };
        
        this.lockRegistry = new Map();
        this.tagRegistry = new Map();
    }
    
    async connect() {
        try {
            if (this.config.cluster && this.config.clusterNodes.length > 0) {
                // Redis Cluster mode
                const { Cluster } = require('ioredis');
                this.client = new Cluster(this.config.clusterNodes, {
                    redisOptions: {
                        password: this.config.password,
                        maxRetriesPerRequest: this.config.maxRetriesPerRequest,
                        retryDelayOnFailover: this.config.retryDelayOnFailover
                    }
                });
            } else {
                // Single instance or sentinel
                this.client = Redis.createClient({
                    socket: {
                        host: this.config.host,
                        port: this.config.port
                    },
                    password: this.config.password,
                    database: this.config.db
                });
            }
            
            // Event handlers
            this.client.on('connect', function() {
                this.isConnected = true;
                this.emit('connected');
                console.log(`A2A Redis Cache connected to ${this.config.host}:${this.config.port}`);
            }.bind(this));
            
            this.client.on('error', function(error) {
                this.metrics.errors++;
                this.emit('error', error);
                console.error('A2A Redis Cache error:', error);
            }.bind(this));
            
            this.client.on('end', function() {
                this.isConnected = false;
                this.emit('disconnected');
            }.bind(this));
            
            await this.client.connect();
            
            // Setup pub/sub for cache invalidation
            await this.setupPubSub();
            
            // Setup cache policies
            await this.setupCachePolicies();
            
            return this;
            
        } catch (error) {
            console.error('Failed to connect to Redis:', error);
            throw error;
        }
    }
    
    async setupPubSub() {
        // Create separate connections for pub/sub
        this.subscriber = this.client.duplicate();
        this.publisher = this.client.duplicate();
        
        await this.subscriber.connect();
        await this.publisher.connect();
        
        // Subscribe to cache invalidation events
        await this.subscriber.subscribe('a2a:cache:invalidate', function(message) {
            try {
                const data = JSON.parse(message);
                this.handleCacheInvalidation(data);
            } catch (error) {
                console.error('Error handling cache invalidation:', error);
            }
        }.bind(this));
        
        // Subscribe to distributed lock events
        await this.subscriber.subscribe('a2a:cache:locks', function(message) {
            try {
                const data = JSON.parse(message);
                this.handleLockEvent(data);
            } catch (error) {
                console.error('Error handling lock event:', error);
            }
        });
    }
    
    async setupCachePolicies() {
        // Configure Redis memory policies
        try {
            await this.client.configSet('maxmemory-policy', this.config.maxmemoryPolicy);
            await this.client.configSet('save', '900 1 300 10 60 10000'); // Background saves
        } catch (error) {
            console.warn('Could not set Redis policies (may require admin privileges):', error.message);
        }
    }
    
    // Core caching methods
    async get(key, options = {}) {
        this.metrics.totalRequests++;
        
        try {
            const fullKey = this.buildKey(key);
            const result = await this.client.get(fullKey);
            
            if (result === null) {
                this.metrics.misses++;
                return null;
            }
            
            this.metrics.hits++;
            
            // Deserialize based on configuration
            const value = this.deserialize(result);
            
            // Update access time for LRU if enabled
            if (options.updateAccessTime !== false) {
                await this.touch(key);
            }
            
            this.emit('cache:hit', { key, value });
            return value;
            
        } catch (error) {
            this.metrics.errors++;
            this.emit('cache:error', { operation: 'get', key, error });
            throw error;
        }
    }
    
    async set(key, value, options = {}) {
        this.metrics.totalRequests++;
        
        try {
            const fullKey = this.buildKey(key);
            const serializedValue = this.serialize(value);
            const ttl = options.ttl || this.config.defaultTTL;
            
            // Set with expiration
            if (ttl > 0) {
                await this.client.setEx(fullKey, ttl, serializedValue);
            } else {
                await this.client.set(fullKey, serializedValue);
            }
            
            // Handle tags for group invalidation
            if (options.tags && Array.isArray(options.tags)) {
                await this.addToTags(key, options.tags);
            }
            
            this.metrics.sets++;
            this.emit('cache:set', { key, value, ttl, tags: options.tags });
            
            return true;
            
        } catch (error) {
            this.metrics.errors++;
            this.emit('cache:error', { operation: 'set', key, error });
            throw error;
        }
    }
    
    async delete(key) {
        this.metrics.totalRequests++;
        
        try {
            const fullKey = this.buildKey(key);
            const result = await this.client.del(fullKey);
            
            // Remove from tag registry
            await this.removeFromTags(key);
            
            this.metrics.deletes++;
            this.emit('cache:delete', { key });
            
            return result > 0;
            
        } catch (error) {
            this.metrics.errors++;
            this.emit('cache:error', { operation: 'delete', key, error });
            throw error;
        }
    }
    
    async exists(key) {
        try {
            const fullKey = this.buildKey(key);
            return await this.client.exists(fullKey) > 0;
        } catch (error) {
            this.metrics.errors++;
            throw error;
        }
    }
    
    async touch(key, ttl = null) {
        try {
            const fullKey = this.buildKey(key);
            if (ttl !== null) {
                await this.client.expire(fullKey, ttl);
            } else {
                await this.client.persist(fullKey);
            }
            return true;
        } catch (error) {
            this.metrics.errors++;
            throw error;
        }
    }
    
    // Advanced caching patterns
    async getOrSet(key, fetcher, options = {}) {
        let value = await this.get(key, options);
        
        if (value === null) {
            // Use distributed lock to prevent cache stampede
            const lockKey = `lock:${key}`;
            const lock = await this.acquireLock(lockKey, options.lockTimeout);
            
            try {
                // Double-check after acquiring lock
                value = await this.get(key, options);
                if (value === null) {
                    value = await fetcher();
                    await this.set(key, value, options);
                }
            } finally {
                await this.releaseLock(lockKey, lock);
            }
        }
        
        return value;
    }
    
    async mget(keys) {
        this.metrics.totalRequests++;
        
        try {
            const fullKeys = keys.map(function(key) {
                return this.buildKey(key);
            }.bind(this));
            const results = await this.client.mGet(fullKeys);
            
            const values = {};
            for (let i = 0; i < keys.length; i++) {
                const result = results[i];
                if (result !== null) {
                    values[keys[i]] = this.deserialize(result);
                    this.metrics.hits++;
                } else {
                    this.metrics.misses++;
                }
            }
            
            return values;
            
        } catch (error) {
            this.metrics.errors++;
            throw error;
        }
    }
    
    async mset(keyValuePairs, options = {}) {
        this.metrics.totalRequests++;
        
        try {
            const pipeline = this.client.multi();
            const ttl = options.ttl || this.config.defaultTTL;
            
            for (const [key, value] of Object.entries(keyValuePairs)) {
                const fullKey = this.buildKey(key);
                const serializedValue = this.serialize(value);
                
                if (ttl > 0) {
                    pipeline.setEx(fullKey, ttl, serializedValue);
                } else {
                    pipeline.set(fullKey, serializedValue);
                }
                
                this.metrics.sets++;
            }
            
            await pipeline.exec();
            this.emit('cache:mset', { keys: Object.keys(keyValuePairs), ttl });
            
            return true;
            
        } catch (error) {
            this.metrics.errors++;
            throw error;
        }
    }
    
    // Tag-based invalidation
    async invalidateByTag(tag) {
        try {
            const tagKey = `tags:${tag}`;
            const keys = await this.client.sMembers(this.buildKey(tagKey));
            
            if (keys.length > 0) {
                const pipeline = this.client.multi();
                keys.forEach(function(key) {
                    pipeline.del(this.buildKey(key));
                }.bind(this));
                await pipeline.exec();
                
                // Clean up tag registry
                await this.client.del(this.buildKey(tagKey));
                
                // Broadcast invalidation to other instances
                await this.publisher.publish('a2a:cache:invalidate', JSON.stringify({
                    type: 'tag',
                    tag,
                    keys,
                    timestamp: Date.now()
                }));
                
                this.emit('cache:invalidate:tag', { tag, keys });
            }
            
            return keys.length;
            
        } catch (error) {
            this.metrics.errors++;
            throw error;
        }
    }
    
    async addToTags(key, tags) {
        const pipeline = this.client.multi();
        
        for (const tag of tags) {
            const tagKey = `tags:${tag}`;
            pipeline.sAdd(this.buildKey(tagKey), key);
        }
        
        await pipeline.exec();
    }
    
    async removeFromTags(key) {
        // This would require reverse lookup - simplified for performance
        // In production, maintain a reverse index
    }
    
    // Distributed locking
    async acquireLock(lockKey, timeout = this.config.lockTimeout) {
        const identifier = crypto.randomBytes(16).toString('hex');
        const fullLockKey = this.buildKey(`locks:${lockKey}`);
        const end = Date.now() + timeout;
        
        while (Date.now() < end) {
            const result = await this.client.set(fullLockKey, identifier, {
                PX: timeout,
                NX: true
            });
            
            if (result === 'OK') {
                this.lockRegistry.set(lockKey, identifier);
                return identifier;
            }
            
            // Wait before retry
            await new Promise(function(resolve) { 
                setTimeout(resolve, 10); 
            });
        }
        
        throw new Error(`Failed to acquire lock: ${lockKey}`);
    }
    
    async releaseLock(lockKey, identifier) {
        const fullLockKey = this.buildKey(`locks:${lockKey}`);
        
        // Use Lua script for atomic release
        const script = `
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
        `;
        
        const result = await this.client.eval(script, 1, fullLockKey, identifier);
        
        if (result === 1) {
            this.lockRegistry.delete(lockKey);
            return true;
        }
        
        return false;
    }
    
    // Event handlers
    handleCacheInvalidation(data) {
        switch (data.type) {
            case 'tag':
                this.emit('cache:invalidate:remote', data);
                break;
            case 'key':
                this.emit('cache:invalidate:remote', data);
                break;
        }
    }
    
    handleLockEvent(data) {
        this.emit('cache:lock:event', data);
    }
    
    // Utility methods
    buildKey(key) {
        return `${this.config.keyPrefix}${key}`;
    }
    
    serialize(value) {
        if (this.config.serialization === 'json') {
            return JSON.stringify(value);
        } else if (this.config.serialization === 'msgpack') {
            // Implement MessagePack serialization if needed
            return JSON.stringify(value);
        }
        return String(value);
    }
    
    deserialize(value) {
        if (this.config.serialization === 'json') {
            try {
                return JSON.parse(value);
            } catch {
                return value;
            }
        } else if (this.config.serialization === 'msgpack') {
            // Implement MessagePack deserialization if needed
            try {
                return JSON.parse(value);
            } catch {
                return value;
            }
        }
        return value;
    }
    
    // Analytics and monitoring
    getMetrics() {
        const hitRate = this.metrics.totalRequests > 0 
            ? (this.metrics.hits / this.metrics.totalRequests * 100).toFixed(2)
            : 0;
            
        return {
            ...this.metrics,
            hitRate: `${hitRate}%`,
            isConnected: this.isConnected,
            activeLocks: this.lockRegistry.size
        };
    }
    
    resetMetrics() {
        this.metrics = {
            hits: 0,
            misses: 0,
            sets: 0,
            deletes: 0,
            errors: 0,
            totalRequests: 0
        };
    }
    
    async getInfo() {
        try {
            const info = await this.client.info();
            const memory = await this.client.info('memory');
            return {
                server: info,
                memory,
                metrics: this.getMetrics()
            };
        } catch (error) {
            throw error;
        }
    }
    
    // Cleanup
    async flush() {
        try {
            await this.client.flushDb();
            this.emit('cache:flush');
            return true;
        } catch (error) {
            this.metrics.errors++;
            throw error;
        }
    }
    
    async disconnect() {
        try {
            if (this.client) {
                await this.client.quit();
            }
            if (this.subscriber) {
                await this.subscriber.quit();
            }
            if (this.publisher) {
                await this.publisher.quit();
            }
            
            this.isConnected = false;
            this.emit('disconnected');
            
        } catch (error) {
            console.error('Error disconnecting from Redis:', error);
        }
    }
}

// Factory function for easy instantiation
function createA2ACache(options = {}) {
    return new A2ARedisCache(options);
}

// Singleton instance for application-wide use
let defaultInstance = null;

function getDefaultCache(options = {}) {
    if (!defaultInstance) {
        defaultInstance = new A2ARedisCache(options);
    }
    return defaultInstance;
}

module.exports = {
    A2ARedisCache,
    createA2ACache,
    getDefaultCache
};