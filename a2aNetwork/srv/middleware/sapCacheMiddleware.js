/**
 * Enterprise Caching Middleware for A2A Network
 * Implements multi-level caching strategy following SAP performance guidelines
 */

const Redis = require('ioredis');
const cds = require('@sap/cds');
const { performance } = require('perf_hooks');

class CacheMiddleware {
  constructor() {
    this.redis = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: process.env.REDIS_PORT || 6379,
      password: process.env.REDIS_PASSWORD,
      db: process.env.REDIS_DB || 0,
      retryDelayOnFailover: 100,
      enableOfflineQueue: false,
      maxRetriesPerRequest: 3,
      lazyConnect: true
    });

    this.memoryCache = new Map();
    this.maxMemoryCacheSize = 1000;
    
    // Cache configuration by entity type
    this.cacheConfig = {
      'Agents': { ttl: 300, strategy: 'write-through' },           // 5 minutes
      'Capabilities': { ttl: 1800, strategy: 'write-behind' },     // 30 minutes
      'Services': { ttl: 600, strategy: 'write-through' },         // 10 minutes
      'NetworkStats': { ttl: 60, strategy: 'write-behind' },       // 1 minute
      'TopAgents': { ttl: 900, strategy: 'cache-aside' },          // 15 minutes
      'ActiveServices': { ttl: 300, strategy: 'cache-aside' }      // 5 minutes
    };

    this.setupEventListeners();
  }

  /**
   * Initialize cache middleware
   */
  async initialize() {
    try {
      await this.redis.connect();
      cds.log('service').info('Redis cache connected successfully');
    } catch (error) {
      cds.log('service').warn('Redis unavailable, using memory cache only:', error.message);
    }
  }

  /**
   * Cache middleware for Express routes
   */
  middleware() {
    const self = this; // Preserve context for SAP enterprise standard approach
    return async (req, res, next) => {
      const startTime = performance.now();
      
      // Skip caching for non-GET requests
      if (req.method !== 'GET') {
        return next();
      }

      const cacheKey = self.generateCacheKey(req);
      const entityType = self.extractEntityType(req.path);
      const config = self.cacheConfig[entityType] || { ttl: 300, strategy: 'cache-aside' };

      try {
        const cachedData = await self.get(cacheKey);
        
        if (cachedData) {
          const duration = performance.now() - startTime;
          
          // Add cache headers
          res.set({
            'X-Cache-Status': 'HIT',
            'X-Cache-Duration': `${duration.toFixed(2)}ms`,
            'Cache-Control': `public, max-age=${config.ttl}`
          });
          
          return res.json(cachedData);
        }

        // Cache miss - continue to handler
        res.set('X-Cache-Status', 'MISS');
        
        // Use event-based response handling instead of overriding res.json to avoid OpenTelemetry conflicts
        let responseData = null;
        
        // Capture response data using event listeners instead of method override
        res.on('finish', () => {
          // Cache successful responses only
          if (res.statusCode >= 200 && res.statusCode < 300 && responseData) {
            const cachePromise = self.set(cacheKey, responseData, config.ttl);
            
            // Don't wait for cache write in write-behind strategy
            if (config.strategy === 'write-behind') {
              cachePromise.catch(err => 
                cds.log('service').warn('Background cache write failed:', err.message)
              );
            } else {
              // For write-through, we can still respond immediately
              cachePromise.catch(err => 
                cds.log('service').warn('Cache write failed:', err.message)
              );
            }
          }
        });
        
        // Store original json method to capture data without overriding
        const originalJson = res.json;
        res.json = function(data) {
          responseData = data; // Capture data for caching
          return originalJson.call(this, data);
        }.bind(res);

      } catch (error) {
        cds.log('service').warn('Cache middleware error:', error.message);
        res.set('X-Cache-Status', 'ERROR');
      }

      next();
    };
  }

  /**
   * Get data from cache (L1: Memory, L2: Redis)
   */
  async get(key) {
    // L1 Cache: Memory
    if (this.memoryCache.has(key)) {
      const item = this.memoryCache.get(key);
      if (item.expires > Date.now()) {
        return item.data;
      } else {
        this.memoryCache.delete(key);
      }
    }

    // L2 Cache: Redis
    if (this.redis.status === 'ready') {
      try {
        const data = await this.redis.get(key);
        if (data) {
          const parsed = JSON.parse(data);
          
          // Store in memory cache for faster access
          this.setMemoryCache(key, parsed, 60); // 1 minute in memory
          
          return parsed;
        }
      } catch (error) {
        cds.log('service').warn('Redis get error:', error.message);
      }
    }

    return null;
  }

  /**
   * Set data in cache
   */
  async set(key, data, ttl = 300) {
    const promises = [];

    // Memory cache
    this.setMemoryCache(key, data, Math.min(ttl, 300)); // Max 5 minutes in memory

    // Redis cache
    if (this.redis.status === 'ready') {
      promises.push(
        this.redis.setex(key, ttl, JSON.stringify(data))
          .catch(err => cds.log('service').warn('Redis set error:', err.message))
      );
    }

    await Promise.allSettled(promises);
  }

  /**
   * Invalidate cache entries
   */
  async invalidate(pattern) {
    const promises = [];

    // Clear memory cache
    for (const key of this.memoryCache.keys()) {
      if (key.includes(pattern)) {
        this.memoryCache.delete(key);
      }
    }

    // Clear Redis cache
    if (this.redis.status === 'ready') {
      try {
        const keys = await this.redis.keys(`*${pattern}*`);
        if (keys.length > 0) {
          promises.push(this.redis.del(...keys));
        }
      } catch (error) {
        cds.log('service').warn('Redis invalidate error:', error.message);
      }
    }

    await Promise.allSettled(promises);
  }

  /**
   * Memory cache with size limit
   */
  setMemoryCache(key, data, ttl) {
    if (this.memoryCache.size >= this.maxMemoryCacheSize) {
      // Remove oldest entry
      const firstKey = this.memoryCache.keys().next().value;
      this.memoryCache.delete(firstKey);
    }

    this.memoryCache.set(key, {
      data,
      expires: Date.now() + (ttl * 1000)
    });
  }

  /**
   * Generate cache key from request
   */
  generateCacheKey(req) {
    const baseKey = `a2a:${req.path}`;
    const queryParams = new URLSearchParams(req.query);
    queryParams.sort();
    
    const paramsString = queryParams.toString();
    const userContext = req.user?.sub || 'anonymous';
    
    return `${baseKey}:${userContext}:${Buffer.from(paramsString).toString('base64')}`;
  }

  /**
   * Extract entity type from request path
   */
  extractEntityType(path) {
    const matches = path.match(/\/api\/v\d+\/(\w+)/);
    return matches ? matches[1] : 'Unknown';
  }

  /**
   * Setup event listeners for cache invalidation
   */
  setupEventListeners() {
    // Listen for data changes to invalidate cache
    process.on('cache:invalidate', (pattern) => {
      this.invalidate(pattern).catch(err => 
        cds.log('service').warn('Cache invalidation failed:', err.message)
      );
    });

    // Cleanup on shutdown
    process.on('SIGTERM', async () => {
      if (this.redis.status === 'ready') {
        await this.redis.disconnect();
      }
    });
  }

  /**
   * Get cache statistics
   */
  async getStats() {
    const stats = {
      memoryCache: {
        size: this.memoryCache.size,
        maxSize: this.maxMemoryCacheSize
      },
      redis: {
        status: this.redis.status,
        keyCount: 0
      }
    };

    if (this.redis.status === 'ready') {
      try {
        const info = await this.redis.info('keyspace');
        const keyMatch = info.match(/keys=(\d+)/);
        stats.redis.keyCount = keyMatch ? parseInt(keyMatch[1]) : 0;
      } catch (error) {
        cds.log('service').warn('Redis stats error:', error.message);
      }
    }

    return stats;
  }

  /**
   * Warm up cache with frequently accessed data
   */
  async warmUp() {
    cds.log('service').info('Starting cache warm-up...');
    
    const warmUpData = [
      { key: 'a2a:/api/v1/TopAgents', ttl: 900 },
      { key: 'a2a:/api/v1/ActiveServices', ttl: 300 },
      { key: 'a2a:/api/v1/NetworkStats', ttl: 60 }
    ];

    // This would typically fetch from the database and populate cache
    // Implementation depends on your data access layer
    
    cds.log('service').info('Cache warm-up completed');
  }
}

module.exports = new CacheMiddleware();