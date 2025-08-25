/**
 * Real Redis Caching Implementation for A2A Network
 * Provides distributed caching with fallback to in-memory cache
 */
const Redis = require('ioredis');
const EventEmitter = require('events');

// Track intervals for cleanup
const activeIntervals = new Map();

class A2ACacheService extends EventEmitter {
    constructor(options = {}) {
        super();
        
        this.options = {
            host: process.env.REDIS_HOST || 'localhost',
            port: process.env.REDIS_PORT || 6379,
            password: process.env.REDIS_PASSWORD,
            db: process.env.REDIS_DB || 0,
            keyPrefix: process.env.REDIS_KEY_PREFIX || 'a2a:',
            retryDelayOnFailover: 100,
            maxRetriesPerRequest: 3,
            lazyConnect: true,
            ...options
        };
        
        this.redis = null;
        this.fallbackCache = new Map();
        this.isRedisAvailable = false;
        this.cacheStats = {
            hits: 0,
            misses: 0,
            sets: 0,
            deletes: 0,
            errors: 0
        };
        
        this.defaultTTL = 300; // 5 minutes
        this.maxFallbackSize = 1000; // Max items in fallback cache
    }

    async initialize() {
        try {
            // Attempt to connect to Redis
            if (process.env.REDIS_URL) {
                this.redis = new Redis(process.env.REDIS_URL, this.options);
            } else {
                this.redis = new Redis(this.options);
            }

            // Test connection
            await this.redis.ping();
            this.isRedisAvailable = true;
            
            // Set up event handlers
            this.redis.on('connect', () => {
                // console.log('✅ Redis cache connected');
                this.isRedisAvailable = true;
                this.emit('connected');
            });

            this.redis.on('error', (error) => {
                console.warn('⚠️  Redis error, falling back to in-memory cache:', error.message);
                this.isRedisAvailable = false;
                this.cacheStats.errors++;
                this.emit('error', error);
            });

            this.redis.on('reconnecting', () => {
                // console.log('🔄 Redis reconnecting...');
                this.emit('reconnecting');
            });

            // console.log('✅ Cache service initialized with Redis backend');
            
        } catch (error) {
            console.warn('⚠️  Redis not available, using in-memory cache fallback:', error.message);
            this.isRedisAvailable = false;
            this.redis = null;
        }

        // Set up cleanup interval for fallback cache
        activeIntervals.set('interval_80', setInterval(() => {
            this.cleanupFallbackCache();
        }, 60000)); // Clean up every minute

        return this;
    }

    // Generate cache key with namespace
    generateKey(key) {
        return `${this.options.keyPrefix}${key}`;
    }

    // Get value from cache
    async get(key) {
        const fullKey = this.generateKey(key);
        
        try {
            let value;
            
            if (this.isRedisAvailable && this.redis) {
                // Try Redis first
                const result = await this.redis.get(fullKey);
                if (result !== null) {
                    value = JSON.parse(result);
                    this.cacheStats.hits++;
                    return value;
                }
            }
            
            // Fallback to in-memory cache
            const fallbackData = this.fallbackCache.get(fullKey);
            if (fallbackData) {
                // Check if expired
                if (fallbackData.expiry && Date.now() > fallbackData.expiry) {
                    this.fallbackCache.delete(fullKey);
                    this.cacheStats.misses++;
                    return null;
                }
                
                this.cacheStats.hits++;
                return fallbackData.value;
            }
            
            this.cacheStats.misses++;
            return null;
            
        } catch (error) {
            console.error('Cache get error:', error.message);
            this.cacheStats.errors++;
            return null;
        }
    }

    // Set value in cache
    async set(key, value, ttl = this.defaultTTL) {
        const fullKey = this.generateKey(key);
        
        try {
            const serializedValue = JSON.stringify(value);
            
            if (this.isRedisAvailable && this.redis) {
                // Set in Redis
                if (ttl > 0) {
                    await this.redis.setex(fullKey, ttl, serializedValue);
                } else {
                    await this.redis.set(fullKey, serializedValue);
                }
            }
            
            // Also set in fallback cache
            const expiry = ttl > 0 ? Date.now() + (ttl * 1000) : null;
            this.fallbackCache.set(fullKey, {
                value,
                expiry,
                timestamp: Date.now()
            });
            
            this.cacheStats.sets++;
            
            // Cleanup fallback if too large
            if (this.fallbackCache.size > this.maxFallbackSize) {
                this.cleanupFallbackCache();
            }
            
            return true;
            
        } catch (error) {
            console.error('Cache set error:', error.message);
            this.cacheStats.errors++;
            return false;
        }
    }

    // Delete from cache
    async delete(key) {
        const fullKey = this.generateKey(key);
        
        try {
            if (this.isRedisAvailable && this.redis) {
                await this.redis.del(fullKey);
            }
            
            this.fallbackCache.delete(fullKey);
            this.cacheStats.deletes++;
            
            return true;
            
        } catch (error) {
            console.error('Cache delete error:', error.message);
            this.cacheStats.errors++;
            return false;
        }
    }

    // Clear all cache
    async clear(pattern = '*') {
        try {
            if (this.isRedisAvailable && this.redis) {
                const keys = await this.redis.keys(this.generateKey(pattern));
                if (keys.length > 0) {
                    await this.redis.del(...keys);
                }
            }
            
            // Clear fallback cache
            if (pattern === '*') {
                this.fallbackCache.clear();
            } else {
                // Clear matching patterns from fallback
                for (const key of this.fallbackCache.keys()) {
                    if (key.includes(pattern)) {
                        this.fallbackCache.delete(key);
                    }
                }
            }
            
            return true;
            
        } catch (error) {
            console.error('Cache clear error:', error.message);
            this.cacheStats.errors++;
            return false;
        }
    }

    // Get with automatic refresh
    async getOrSet(key, fetchFunction, ttl = this.defaultTTL) {
        try {
            // Try to get from cache first
            let value = await this.get(key);
            
            if (value !== null) {
                return value;
            }
            
            // Cache miss - fetch and cache
            value = await fetchFunction();
            
            if (value !== null && value !== undefined) {
                await this.set(key, value, ttl);
            }
            
            return value;
            
        } catch (error) {
            console.error('Cache getOrSet error:', error.message);
            this.cacheStats.errors++;
            
            // Try to execute fetch function directly
            try {
                return await fetchFunction();
            } catch (fetchError) {
                console.error('Fetch function error:', fetchError.message);
                return null;
            }
        }
    }

    // Cache specific to tile data
    async cacheTileData(tileId, data, ttl = 30) {
        return await this.set(`tile:${tileId}`, data, ttl);
    }

    async getTileData(tileId) {
        return await this.get(`tile:${tileId}`);
    }

    // Cache user sessions
    async cacheUserSession(userId, sessionData, ttl = 3600) {
        return await this.set(`session:${userId}`, sessionData, ttl);
    }

    async getUserSession(userId) {
        return await this.get(`session:${userId}`);
    }

    // Cache user preferences
    async cacheUserPreferences(userId, preferences, ttl = 1800) {
        return await this.set(`preferences:${userId}`, preferences, ttl);
    }

    async getUserPreferences(userId) {
        return await this.get(`preferences:${userId}`);
    }

    // Cache database query results
    async cacheQueryResult(query, params, result, ttl = 300) {
        const cacheKey = `query:${this.hashQuery(query, params)}`;
        return await this.set(cacheKey, result, ttl);
    }

    async getCachedQueryResult(query, params) {
        const cacheKey = `query:${this.hashQuery(query, params)}`;
        return await this.get(cacheKey);
    }

    // Hash query for consistent cache keys
    hashQuery(query, params = []) {
        const crypto = require('crypto');

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

        const queryString = query + JSON.stringify(params);
        return crypto.createHash('sha256').update(queryString).digest('hex').substring(0, 16);
    }

    // Cleanup fallback cache
    cleanupFallbackCache() {
        if (this.fallbackCache.size <= this.maxFallbackSize) {
            return;
        }
        
        const now = Date.now();
        const entries = Array.from(this.fallbackCache.entries());
        
        // Remove expired entries first
        let removedExpired = 0;
        for (const [key, data] of entries) {
            if (data.expiry && now > data.expiry) {
                this.fallbackCache.delete(key);
                removedExpired++;
            }
        }
        
        // If still too large, remove oldest entries
        if (this.fallbackCache.size > this.maxFallbackSize) {
            const sortedEntries = entries
                .filter(([key]) => this.fallbackCache.has(key)) // Still exists after expiry cleanup
                .sort(([,a], [,b]) => a.timestamp - b.timestamp);
            
            const toRemove = this.fallbackCache.size - this.maxFallbackSize;
            for (let i = 0; i < toRemove; i++) {
                if (sortedEntries[i]) {
                    this.fallbackCache.delete(sortedEntries[i][0]);
                }
            }
        }
        
        if (removedExpired > 0) {
            // console.log(`🧹 Cleaned up ${removedExpired} expired cache entries`);
        }
    }

    // Get cache statistics
    getStats() {
        return {
            ...this.cacheStats,
            redisAvailable: this.isRedisAvailable,
            fallbackCacheSize: this.fallbackCache.size,
            hitRate: this.cacheStats.hits / (this.cacheStats.hits + this.cacheStats.misses) || 0
        };
    }

    // Health check
    async healthCheck() {
        try {
            if (this.isRedisAvailable && this.redis) {
                await this.redis.ping();
                return {
                    status: 'healthy',
                    backend: 'redis',
                    stats: this.getStats()
                };
            } else {
                return {
                    status: 'degraded',
                    backend: 'memory',
                    message: 'Using in-memory fallback cache',
                    stats: this.getStats()
                };
            }
        } catch (error) {
            return {
                status: 'unhealthy',
                backend: 'memory',
                error: error.message,
                stats: this.getStats()
            };
        }
    }

    // Express middleware for response caching
    middleware(defaultTTL = 300) {
        return (req, res, next) => {
            // Skip caching for non-GET requests
            if (req.method !== 'GET') {
                return next();
            }

            // Skip caching for authenticated endpoints requiring real-time data
            const skipPaths = ['/health', '/user/info'];
            if (skipPaths.some(path => req.path.startsWith(path))) {
                return next();
            }

            const cacheKey = `http:${req.method}:${req.path}:${JSON.stringify(req.query)}`;
            
            // Try to get from cache
            this.get(cacheKey).then(cachedResponse => {
                if (cachedResponse) {
                    res.set('X-Cache', 'HIT');
                    res.set('Content-Type', cachedResponse.contentType || 'application/json');
                    return res.status(cachedResponse.status || 200).send(cachedResponse.data);
                }

                // Cache miss - capture response
                res.set('X-Cache', 'MISS');
                
                const originalSend = res.send;
                res.send = function(data) {
                    // Cache successful responses
                    if (res.statusCode >= 200 && res.statusCode < 300) {
                        const responseData = {
                            status: res.statusCode,
                            data: data,
                            contentType: res.get('Content-Type')
                        };
                        
                        // Cache with appropriate TTL
                        let ttl = defaultTTL;
                        if (req.path.includes('tile')) {
                            ttl = 30; // Tile data expires quickly
                        } else if (req.path.includes('user')) {
                            ttl = 600; // User data can be cached longer
                        }
                        
                        this.set(cacheKey, responseData, ttl).catch(err => {
                            console.warn('Failed to cache response:', err.message);
                        });
                    }
                    
                    return originalSend.call(this, data);
                }.bind(this);
                
                next();
            }).catch(error => {
                console.error('Cache middleware error:', error.message);
                next();
            });
        };
    }

    // Graceful shutdown
    async shutdown() {
        try {
            if (this.redis) {
                await this.redis.quit();
            }
            this.fallbackCache.clear();
            // console.log('✅ Cache service shut down successfully');
        } catch (error) {
            console.error('❌ Error shutting down cache service:', error.message);
        }
    }
}

module.exports = A2ACacheService;