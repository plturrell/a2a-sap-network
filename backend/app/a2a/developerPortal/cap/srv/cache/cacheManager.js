/**
 * Enterprise Cache Manager for SAP A2A Developer Portal
 * Implements multi-layer caching strategy with Redis
 */

const redis = require('redis');
const { promisify } = require('util');
const performanceMonitor = require('../monitoring/performance-monitor');

class CacheManager {
    constructor() {
        this.client = null;
        this.connected = false;
        this.memoryCache = new Map();
        this.memoryCacheSize = 1000; // Max items in memory
        this.defaultTTL = 300; // 5 minutes
        
        // Cache configuration by type
        this.cacheConfig = {
            project: { ttl: 600, prefix: 'prj:' },
            agent: { ttl: 300, prefix: 'agt:' },
            workflow: { ttl: 300, prefix: 'wf:' },
            template: { ttl: 1800, prefix: 'tpl:' },
            user: { ttl: 900, prefix: 'usr:' },
            businessPartner: { ttl: 3600, prefix: 'bp:' },
            salesOrder: { ttl: 600, prefix: 'so:' },
            metrics: { ttl: 60, prefix: 'mtr:' },
            config: { ttl: 3600, prefix: 'cfg:' }
        };

        this._initializeRedis();
    }

    /**
     * Initialize Redis connection
     */
    async _initializeRedis() {
        try {
            const redisUrl = process.env.REDIS_URL || 'redis://localhost:6379';
            
            this.client = redis.createClient({
                url: redisUrl,
                socket: {
                    reconnectStrategy: (retries) => {
                        if (retries > 10) {
                            console.error('Redis reconnection limit reached');
                            return new Error('Too many retries');
                        }
                        return Math.min(retries * 100, 3000);
                    }
                },
                // Connection pool settings
                isolationPoolOptions: {
                    min: 2,
                    max: 10
                }
            });

            // Promisify Redis methods
            this.getAsync = promisify(this.client.get).bind(this.client);
            this.setAsync = promisify(this.client.set).bind(this.client);
            this.delAsync = promisify(this.client.del).bind(this.client);
            this.existsAsync = promisify(this.client.exists).bind(this.client);
            this.ttlAsync = promisify(this.client.ttl).bind(this.client);
            this.mgetAsync = promisify(this.client.mget).bind(this.client);
            this.keysAsync = promisify(this.client.keys).bind(this.client);
            this.incrAsync = promisify(this.client.incr).bind(this.client);
            this.expireAsync = promisify(this.client.expire).bind(this.client);

            // Event handlers
            this.client.on('error', (error) => {
                console.error('Redis error:', error);
                performanceMonitor.recordError('redis_connection', 'cache-manager');
                this.connected = false;
            });

            this.client.on('ready', () => {
                console.log('Redis cache connected');
                this.connected = true;
            });

            this.client.on('reconnecting', () => {
                console.log('Redis reconnecting...');
            });

            // Connect to Redis
            await this.client.connect();

            // Start cache cleanup
            this._startCacheCleanup();

        } catch (error) {
            console.error('Failed to initialize Redis:', error);
            this.connected = false;
            // Continue with memory cache only
        }
    }

    /**
     * Get value from cache (multi-layer)
     */
    async get(key, cacheType = 'default') {
        const startTime = Date.now();
        
        try {
            // Check memory cache first (L1)
            const memoryValue = this._getFromMemory(key);
            if (memoryValue !== null) {
                performanceMonitor.recordCacheAccess('memory', 'get', true);
                performanceMonitor.recordDbQuery('cache_get', 'memory', 'hit', Date.now() - startTime);
                return memoryValue;
            }

            // Check Redis cache (L2)
            if (this.connected) {
                const prefixedKey = this._getPrefixedKey(key, cacheType);
                const redisValue = await this.getAsync(prefixedKey);
                
                if (redisValue) {
                    performanceMonitor.recordCacheAccess('redis', 'get', true);
                    performanceMonitor.recordDbQuery('cache_get', 'redis', 'hit', Date.now() - startTime);
                    
                    // Populate memory cache
                    const parsed = JSON.parse(redisValue);
                    this._setToMemory(key, parsed);
                    
                    return parsed;
                }
            }

            performanceMonitor.recordCacheAccess('redis', 'get', false);
            performanceMonitor.recordDbQuery('cache_get', 'redis', 'miss', Date.now() - startTime);
            return null;

        } catch (error) {
            console.error('Cache get error:', error);
            performanceMonitor.recordError('cache_get', 'cache-manager');
            return null;
        }
    }

    /**
     * Set value in cache (multi-layer)
     */
    async set(key, value, cacheType = 'default', customTTL = null) {
        const startTime = Date.now();
        
        try {
            // Set in memory cache (L1)
            this._setToMemory(key, value);

            // Set in Redis cache (L2)
            if (this.connected) {
                const prefixedKey = this._getPrefixedKey(key, cacheType);
                const ttl = customTTL || this.cacheConfig[cacheType]?.ttl || this.defaultTTL;
                const serialized = JSON.stringify(value);
                
                await this.setAsync(prefixedKey, serialized, 'EX', ttl);
                performanceMonitor.recordDbQuery('cache_set', 'redis', 'success', Date.now() - startTime);
            }

            return true;
        } catch (error) {
            console.error('Cache set error:', error);
            performanceMonitor.recordError('cache_set', 'cache-manager');
            return false;
        }
    }

    /**
     * Delete value from cache
     */
    async delete(key, cacheType = 'default') {
        try {
            // Delete from memory cache
            this.memoryCache.delete(key);

            // Delete from Redis
            if (this.connected) {
                const prefixedKey = this._getPrefixedKey(key, cacheType);
                await this.delAsync(prefixedKey);
            }

            return true;
        } catch (error) {
            console.error('Cache delete error:', error);
            return false;
        }
    }

    /**
     * Clear cache by pattern
     */
    async clearPattern(pattern, cacheType = 'default') {
        try {
            const prefix = this.cacheConfig[cacheType]?.prefix || '';
            const fullPattern = `${prefix}${pattern}`;
            
            // Clear from memory cache
            for (const [key] of this.memoryCache) {
                if (key.includes(pattern)) {
                    this.memoryCache.delete(key);
                }
            }

            // Clear from Redis
            if (this.connected) {
                const keys = await this.keysAsync(fullPattern);
                if (keys.length > 0) {
                    await Promise.all(keys.map(key => this.delAsync(key)));
                }
            }

            return true;
        } catch (error) {
            console.error('Cache clear pattern error:', error);
            return false;
        }
    }

    /**
     * Get multiple values from cache
     */
    async mget(keys, cacheType = 'default') {
        try {
            const results = {};
            const missingKeys = [];

            // Check memory cache first
            for (const key of keys) {
                const value = this._getFromMemory(key);
                if (value !== null) {
                    results[key] = value;
                } else {
                    missingKeys.push(key);
                }
            }

            // Get missing from Redis
            if (this.connected && missingKeys.length > 0) {
                const prefixedKeys = missingKeys.map(k => this._getPrefixedKey(k, cacheType));
                const values = await this.mgetAsync(prefixedKeys);
                
                missingKeys.forEach((key, index) => {
                    if (values[index]) {
                        const parsed = JSON.parse(values[index]);
                        results[key] = parsed;
                        this._setToMemory(key, parsed);
                    }
                });
            }

            return results;
        } catch (error) {
            console.error('Cache mget error:', error);
            return {};
        }
    }

    /**
     * Cache-aside pattern implementation
     */
    async cacheAside(key, fetchFunction, cacheType = 'default', customTTL = null) {
        // Try to get from cache
        const cached = await this.get(key, cacheType);
        if (cached !== null) {
            return cached;
        }

        // Fetch from source
        const data = await fetchFunction();
        
        // Cache the result
        if (data !== null && data !== undefined) {
            await this.set(key, data, cacheType, customTTL);
        }

        return data;
    }

    /**
     * Increment counter in cache
     */
    async increment(key, cacheType = 'metrics') {
        try {
            if (this.connected) {
                const prefixedKey = this._getPrefixedKey(key, cacheType);
                const value = await this.incrAsync(prefixedKey);
                
                // Set expiry if it's a new key
                if (value === 1) {
                    const ttl = this.cacheConfig[cacheType]?.ttl || this.defaultTTL;
                    await this.expireAsync(prefixedKey, ttl);
                }
                
                return value;
            }
            return 0;
        } catch (error) {
            console.error('Cache increment error:', error);
            return 0;
        }
    }

    /**
     * Cache warming for frequently accessed data
     */
    async warmCache(warmupFunction, cacheType = 'default') {
        try {
            console.log(`Warming ${cacheType} cache...`);
            const data = await warmupFunction();
            
            for (const [key, value] of Object.entries(data)) {
                await this.set(key, value, cacheType);
            }
            
            console.log(`Warmed ${Object.keys(data).length} entries in ${cacheType} cache`);
        } catch (error) {
            console.error('Cache warming error:', error);
        }
    }

    /**
     * Get cache statistics
     */
    async getStats() {
        const stats = {
            memory: {
                size: this.memoryCache.size,
                maxSize: this.memoryCacheSize
            },
            redis: {
                connected: this.connected
            }
        };

        if (this.connected) {
            try {
                const info = await promisify(this.client.info).bind(this.client)();
                const memoryUsed = info.match(/used_memory_human:(.+)/)?.[1];
                const connectedClients = info.match(/connected_clients:(\d+)/)?.[1];
                
                stats.redis.memoryUsed = memoryUsed;
                stats.redis.connectedClients = parseInt(connectedClients) || 0;
            } catch (error) {
                console.error('Failed to get Redis stats:', error);
            }
        }

        return stats;
    }

    /**
     * Memory cache management
     */
    _getFromMemory(key) {
        const item = this.memoryCache.get(key);
        if (item && item.expiry > Date.now()) {
            return item.value;
        }
        this.memoryCache.delete(key);
        return null;
    }

    _setToMemory(key, value) {
        // Implement LRU eviction if cache is full
        if (this.memoryCache.size >= this.memoryCacheSize) {
            const firstKey = this.memoryCache.keys().next().value;
            this.memoryCache.delete(firstKey);
        }

        this.memoryCache.set(key, {
            value,
            expiry: Date.now() + (this.defaultTTL * 1000)
        });
    }

    /**
     * Get prefixed key for Redis
     */
    _getPrefixedKey(key, cacheType) {
        const prefix = this.cacheConfig[cacheType]?.prefix || '';
        return `${prefix}${key}`;
    }

    /**
     * Start periodic cache cleanup
     */
    _startCacheCleanup() {
        // Clean expired entries from memory cache every minute
        setInterval(() => {
            const now = Date.now();
            for (const [key, item] of this.memoryCache) {
                if (item.expiry < now) {
                    this.memoryCache.delete(key);
                }
            }
        }, 60000);
    }

    /**
     * Close Redis connection
     */
    async close() {
        if (this.client) {
            await this.client.quit();
            this.connected = false;
        }
    }
}

// Export singleton instance
module.exports = new CacheManager();