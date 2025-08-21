/**
 * Blockchain Cache Manager
 * Caches blockchain read operations to improve performance
 */

const cds = require('@sap/cds');
const redis = require('redis');
const log = cds.log('blockchain-cache');

class BlockchainCache {
    constructor() {
        this.cache = new Map(); // In-memory cache as fallback
        this.redisClient = null;
        this.defaultTTL = 30; // 30 seconds
        this.cacheConfig = {
            'agentInfo': { ttl: 300 }, // 5 minutes
            'agentReputation': { ttl: 60 }, // 1 minute
            'serviceInfo': { ttl: 300 }, // 5 minutes
            'networkStats': { ttl: 30 }, // 30 seconds
            'ordDocument': { ttl: 3600 }, // 1 hour
            'agentsByCapability': { ttl: 180 }, // 3 minutes
            'ordDocumentsByTag': { ttl: 300 } // 5 minutes
        };
        
        this.intervals = new Map(); // Track intervals for cleanup
        this.initializeRedis();
    }

    async initializeRedis() {
        try {
            const redisConfig = cds.env.requires?.redis || process.env.REDIS_URL;
            if (redisConfig) {
                this.redisClient = redis.createClient(redisConfig);
                await this.redisClient.connect();
                log.info('Redis cache initialized for blockchain operations');
            } else {
                log.warn('No Redis configuration found, using in-memory cache');
            }
        } catch (error) {
            log.error('Failed to initialize Redis, falling back to in-memory cache', error);
        }
    }

    /**
     * Generate cache key for blockchain operations
     * @param {string} operation - Operation name
     * @param {any} params - Parameters
     * @returns {string} Cache key
     */
    generateKey(operation, params) {
        const paramsStr = typeof params === 'object' ? JSON.stringify(params) : String(params);
        return `blockchain:${operation}:${Buffer.from(paramsStr).toString('base64')}`;
    }

    /**
     * Get cached value
     * @param {string} operation - Operation name
     * @param {any} params - Parameters
     * @returns {Promise<any>} Cached value or null
     */
    async get(operation, params) {
        const key = this.generateKey(operation, params);
        
        try {
            // Try Redis first
            if (this.redisClient) {
                const cached = await this.redisClient.get(key);
                if (cached) {
                    log.debug('Cache hit (Redis)', { operation, key: key.substring(0, 50) + '...' });
                    return JSON.parse(cached);
                }
            }
            
            // Fallback to in-memory cache
            if (this.cache.has(key)) {
                const entry = this.cache.get(key);
                if (entry.expires > Date.now()) {
                    log.debug('Cache hit (memory)', { operation, key: key.substring(0, 50) + '...' });
                    return entry.data;
                } else {
                    // Expired entry
                    this.cache.delete(key);
                }
            }
            
            return null;
        } catch (error) {
            log.error('Cache get error', { operation, error: error.message });
            return null;
        }
    }

    /**
     * Set cached value
     * @param {string} operation - Operation name
     * @param {any} params - Parameters
     * @param {any} data - Data to cache
     * @returns {Promise<void>}
     */
    async set(operation, params, data) {
        const key = this.generateKey(operation, params);
        const config = this.cacheConfig[operation] || { ttl: this.defaultTTL };
        
        try {
            const serialized = JSON.stringify(data);
            
            // Store in Redis
            if (this.redisClient) {
                await this.redisClient.setEx(key, config.ttl, serialized);
            }
            
            // Store in memory cache as backup
            this.cache.set(key, {
                data,
                expires: Date.now() + (config.ttl * 1000)
            });
            
            log.debug('Cache set', { operation, ttl: config.ttl, size: serialized.length });
        } catch (error) {
            log.error('Cache set error', { operation, error: error.message });
        }
    }

    /**
     * Execute operation with caching
     * @param {string} operation - Operation name
     * @param {any} params - Parameters
     * @param {Function} fn - Function to execute if not cached
     * @returns {Promise<any>} Result
     */
    async executeWithCache(operation, params, fn) {
        // Try to get from cache first
        const cached = await this.get(operation, params);
        if (cached !== null) {
            return cached;
        }

        // Execute the function
        const result = await fn();
        
        // Cache the result
        if (result !== null && result !== undefined) {
            await this.set(operation, params, result);
        }
        
        return result;
    }

    /**
     * Invalidate cache for operation
     * @param {string} operation - Operation name
     * @param {any} params - Parameters (optional, clears all if not provided)
     * @returns {Promise<void>}
     */
    async invalidate(operation, params = null) {
        try {
            if (params === null) {
                // Invalidate all entries for operation
                const pattern = `blockchain:${operation}:*`;
                
                if (this.redisClient) {
                    const keys = await this.redisClient.keys(pattern);
                    if (keys.length > 0) {
                        await this.redisClient.del(keys);
                    }
                }
                
                // Clear from memory cache
                for (const key of this.cache.keys()) {
                    if (key.startsWith(`blockchain:${operation}:`)) {
                        this.cache.delete(key);
                    }
                }
                
                log.info('Cache invalidated for operation', { operation });
            } else {
                // Invalidate specific entry
                const key = this.generateKey(operation, params);
                
                if (this.redisClient) {
                    await this.redisClient.del(key);
                }
                
                this.cache.delete(key);
                
                log.debug('Cache entry invalidated', { operation, key: key.substring(0, 50) + '...' });
            }
        } catch (error) {
            log.error('Cache invalidation error', { operation, error: error.message });
        }
    }

    /**
     * Clear all blockchain cache
     * @returns {Promise<void>}
     */
    async clear() {
        try {
            if (this.redisClient) {
                const keys = await this.redisClient.keys('blockchain:*');
                if (keys.length > 0) {
                    await this.redisClient.del(keys);
                }
            }
            
            // Clear memory cache
            for (const key of this.cache.keys()) {
                if (key.startsWith('blockchain:')) {
                    this.cache.delete(key);
                }
            }
            
            log.info('All blockchain cache cleared');
        } catch (error) {
            log.error('Cache clear error', error);
        }
    }

    /**
     * Get cache statistics
     * @returns {Object} Cache stats
     */
    getCacheStats() {
        const memoryEntries = Array.from(this.cache.keys())
            .filter(key => key.startsWith('blockchain:'))
            .length;
        
        return {
            redisConnected: !!this.redisClient?.isReady,
            memoryEntries,
            operations: Object.keys(this.cacheConfig)
        };
    }

    /**
     * Cleanup expired entries from memory cache
     * @returns {void}
     */
    cleanupExpired() {
        const now = Date.now();
        let cleaned = 0;
        
        for (const [key, entry] of this.cache.entries()) {
            if (key.startsWith('blockchain:') && entry.expires <= now) {
                this.cache.delete(key);
                cleaned++;
            }
        }
        
        if (cleaned > 0) {
            log.debug('Cleaned up expired cache entries', { count: cleaned });
        }
    }
}

// Export singleton instance
const blockchainCache = new BlockchainCache();

// Setup cleanup interval
const cleanupInterval = setInterval(() => {
    blockchainCache.cleanupExpired();
}, 60000); // Clean up every minute

// Track interval for cleanup
const activeIntervals = new Map();
activeIntervals.set('blockchain_cleanup', cleanupInterval);

module.exports = blockchainCache;