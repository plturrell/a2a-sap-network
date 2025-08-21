/**
 * @fileoverview High-Performance Caching Service
 * @since 1.0.0
 * @module cacheService
 * 
 * Provides intelligent caching for frequently accessed data with TTL support
 */

const cds = require('@sap/cds');

class CacheService {
    constructor() {
        this.cache = new Map();
        this.cleanupIntervalId = null; // Track cleanup interval for memory leak prevention
        this.stats = {
            hits: 0,
            misses: 0,
            evictions: 0,
            totalRequests: 0
        };
        
        this.intervals = new Map(); // Track intervals for cleanup
        
        // Default cache settings
        this.defaultTTL = 5 * 60 * 1000; // 5 minutes
        this.maxCacheSize = 1000;
        this.cleanupInterval = 60 * 1000; // 1 minute
        
        // Start cleanup timer
        this.startCleanupTimer();
        
        this.log = cds.log('cache-service');
    }

    /**
     * Get item from cache
     * @param {string} key - Cache key
     * @returns {any|null} Cached value or null if not found/expired
     */
    get(key) {
        this.stats.totalRequests++;
        
        const item = this.cache.get(key);
        
        if (!item) {
            this.stats.misses++;
            return null;
        }
        
        // Check if expired
        if (Date.now() > item.expiresAt) {
            this.cache.delete(key);
            this.stats.misses++;
            this.stats.evictions++;
            return null;
        }
        
        // Update access time for LRU
        item.lastAccessed = Date.now();
        this.stats.hits++;
        
        return item.value;
    }

    /**
     * Set item in cache
     * @param {string} key - Cache key
     * @param {any} value - Value to cache
     * @param {number} ttl - Time to live in milliseconds (optional)
     */
    set(key, value, ttl = this.defaultTTL) {
        // Enforce cache size limit
        if (this.cache.size >= this.maxCacheSize) {
            this.evictLRU();
        }
        
        const expiresAt = Date.now() + ttl;
        const item = {
            value,
            expiresAt,
            createdAt: Date.now(),
            lastAccessed: Date.now()
        };
        
        this.cache.set(key, item);
        
        this.log.debug(`Cached item: ${key}, expires at: ${new Date(expiresAt).toISOString()}`);
    }

    /**
     * Check if key exists and is not expired
     * @param {string} key - Cache key
     * @returns {boolean} True if key exists and is valid
     */
    has(key) {
        const item = this.cache.get(key);
        
        if (!item) {
            return false;
        }
        
        // Check if expired
        if (Date.now() > item.expiresAt) {
            this.cache.delete(key);
            this.stats.evictions++;
            return false;
        }
        
        return true;
    }

    /**
     * Delete item from cache
     * @param {string} key - Cache key
     * @returns {boolean} True if item was deleted
     */
    delete(key) {
        return this.cache.delete(key);
    }

    /**
     * Clear all cached items
     */
    clear() {
        const previousSize = this.cache.size;
        this.cache.clear();
        this.log.info(`Cache cleared: ${previousSize} items removed`);
    }

    /**
     * Get or set cached value with function
     * @param {string} key - Cache key
     * @param {Function} fn - Function to call if cache miss
     * @param {number} ttl - Time to live in milliseconds (optional)
     * @returns {Promise<any>} Cached or computed value
     */
    async getOrSet(key, fn, ttl = this.defaultTTL) {
        let value = this.get(key);
        
        if (value !== null) {
            return value;
        }
        
        // Cache miss - compute value
        try {
            value = await fn();
            this.set(key, value, ttl);
            return value;
        } catch (error) {
            this.log.error('Error computing cached value:', error);
            throw error;
        }
    }

    /**
     * Evict least recently used item
     */
    evictLRU() {
        let oldestKey = null;
        let oldestTime = Date.now();
        
        for (const [key, item] of this.cache.entries()) {
            if (item.lastAccessed < oldestTime) {
                oldestTime = item.lastAccessed;
                oldestKey = key;
            }
        }
        
        if (oldestKey) {
            this.cache.delete(oldestKey);
            this.stats.evictions++;
            this.log.debug(`Evicted LRU item: ${oldestKey}`);
        }
    }

    /**
     * Clean up expired items
     */
    cleanup() {
        const now = Date.now();
        const keysToDelete = [];
        
        for (const [key, item] of this.cache.entries()) {
            if (now > item.expiresAt) {
                keysToDelete.push(key);
            }
        }
        
        for (const key of keysToDelete) {
            this.cache.delete(key);
            this.stats.evictions++;
        }
        
        if (keysToDelete.length > 0) {
            this.log.debug(`Cleaned up ${keysToDelete.length} expired cache items`);
        }
    }

    /**
     * Start automatic cleanup timer
     */
    startCleanupTimer() {
        // Stop existing timer first
        this.stopCleanupTimer();
        
        this.cleanupIntervalId = this.intervals.set('interval_204', setInterval(() => {
            setImmediate(() => {
                this.cleanup();
            });
        }, this.cleanupInterval));
    }
    
    stopCleanupTimer() {
        if (this.cleanupIntervalId) {
            clearInterval(this.cleanupIntervalId);
            this.cleanupIntervalId = null;
        }
    }
    
    shutdown() {
        this.stopCleanupTimer();
        this.cache.clear();
        this.log.info('Cache service shutdown complete');
    }

    /**
     * Get cache statistics
     * @returns {object} Cache statistics
     */
    getStats() {
        const hitRate = this.stats.totalRequests > 0 
            ? (this.stats.hits / this.stats.totalRequests * 100).toFixed(2)
            : 0;
            
        return {
            ...this.stats,
            hitRate: `${hitRate}%`,
            cacheSize: this.cache.size,
            maxCacheSize: this.maxCacheSize,
            memoryUsage: this.estimateMemoryUsage()
        };
    }

    /**
     * Estimate memory usage (rough approximation)
     * @returns {string} Memory usage estimate
     */
    estimateMemoryUsage() {
        const itemCount = this.cache.size;
        const estimatedBytesPerItem = 1024; // Rough estimate
        const totalBytes = itemCount * estimatedBytesPerItem;
        
        if (totalBytes < 1024) {
            return `${totalBytes} bytes`;
        } else if (totalBytes < 1024 * 1024) {
            return `${(totalBytes / 1024).toFixed(1)} KB`;
        } else {
            return `${(totalBytes / (1024 * 1024)).toFixed(1)} MB`;
        }
    }

    /**
     * Create cache key for tile data
     * @param {string} tileId - Tile identifier
     * @param {string} userId - User identifier (optional)
     * @returns {string} Cache key
     */
    static createTileKey(tileId, userId = 'default') {
        return `tile:${tileId}:${userId}`;
    }

    /**
     * Create cache key for database query
     * @param {string} entity - Entity name
     * @param {object} params - Query parameters
     * @returns {string} Cache key
     */
    static createQueryKey(entity, params = {}) {
        const paramString = JSON.stringify(params);
        return `query:${entity}:${Buffer.from(paramString).toString('base64')}`;
    }
}

// Global cache instance
const globalCache = new CacheService();

// Export both class and global instance
module.exports = {
    CacheService,
    cache: globalCache,
    
    // Convenience methods
    get: (key) => globalCache.get(key),
    set: (key, value, ttl) => globalCache.set(key, value, ttl),
    has: (key) => globalCache.has(key),
    delete: (key) => globalCache.delete(key),
    clear: () => globalCache.clear(),
    getOrSet: (key, fn, ttl) => globalCache.getOrSet(key, fn, ttl),
    getStats: () => globalCache.getStats(),
    createTileKey: CacheService.createTileKey,
    createQueryKey: CacheService.createQueryKey
};