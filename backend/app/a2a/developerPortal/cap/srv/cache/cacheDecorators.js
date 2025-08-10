/**
 * Cache Decorators for SAP A2A Developer Portal
 * Provides easy-to-use caching decorators for service methods
 */

const cacheManager = require('./cache-manager');
const crypto = require('crypto');

/**
 * Cache decorator for service methods
 * @param {string} cacheType - Type of cache (project, agent, etc.)
 * @param {number} ttl - Custom TTL in seconds (optional)
 * @param {function} keyGenerator - Custom key generator function (optional)
 */
function cache(cacheType = 'default', ttl = null, keyGenerator = null) {
    return function (target, propertyKey, descriptor) {
        const originalMethod = descriptor.value;

        descriptor.value = async function (...args) {
            // Generate cache key
            const key = keyGenerator 
                ? keyGenerator.apply(this, args)
                : generateDefaultKey(propertyKey, args);

            // Try to get from cache
            const cached = await cacheManager.get(key, cacheType);
            if (cached !== null) {
                return cached;
            }

            // Execute original method
            const result = await originalMethod.apply(this, args);

            // Cache the result
            if (result !== null && result !== undefined) {
                await cacheManager.set(key, result, cacheType, ttl);
            }

            return result;
        };

        return descriptor;
    };
}

/**
 * Cache invalidation decorator
 * Clears cache after method execution
 */
function invalidateCache(patterns, cacheTypes = ['default']) {
    return function (target, propertyKey, descriptor) {
        const originalMethod = descriptor.value;

        descriptor.value = async function (...args) {
            // Execute original method
            const result = await originalMethod.apply(this, args);

            // Invalidate cache
            for (const cacheType of cacheTypes) {
                for (const pattern of patterns) {
                    const resolvedPattern = typeof pattern === 'function' 
                        ? pattern.apply(this, args)
                        : pattern;
                    await cacheManager.clearPattern(resolvedPattern, cacheType);
                }
            }

            return result;
        };

        return descriptor;
    };
}

/**
 * Conditional cache decorator
 * Only caches if condition is met
 */
function cacheIf(condition, cacheType = 'default', ttl = null) {
    return function (target, propertyKey, descriptor) {
        const originalMethod = descriptor.value;

        descriptor.value = async function (...args) {
            const shouldCache = typeof condition === 'function'
                ? condition.apply(this, args)
                : condition;

            if (!shouldCache) {
                return originalMethod.apply(this, args);
            }

            // Use regular cache logic
            const key = generateDefaultKey(propertyKey, args);
            const cached = await cacheManager.get(key, cacheType);
            if (cached !== null) {
                return cached;
            }

            const result = await originalMethod.apply(this, args);

            if (result !== null && result !== undefined) {
                await cacheManager.set(key, result, cacheType, ttl);
            }

            return result;
        };

        return descriptor;
    };
}

/**
 * Cache warmup decorator
 * Pre-loads cache on application startup
 */
function warmupCache(cacheType = 'default') {
    return function (target, propertyKey, descriptor) {
        const originalMethod = descriptor.value;

        // Register warmup method
        if (!target.constructor._cacheWarmupMethods) {
            target.constructor._cacheWarmupMethods = [];
        }
        
        target.constructor._cacheWarmupMethods.push({
            method: propertyKey,
            cacheType
        });

        return descriptor;
    };
}

/**
 * Time-based cache decorator
 * Caches with different TTL based on time of day
 */
function timeBasedCache(cacheType = 'default', peakHoursTTL = 60, offPeakTTL = 300) {
    return function (target, propertyKey, descriptor) {
        const originalMethod = descriptor.value;

        descriptor.value = async function (...args) {
            const hour = new Date().getHours();
            const isPeakHours = hour >= 9 && hour <= 17; // 9 AM to 5 PM
            const ttl = isPeakHours ? peakHoursTTL : offPeakTTL;

            const key = generateDefaultKey(propertyKey, args);
            const cached = await cacheManager.get(key, cacheType);
            if (cached !== null) {
                return cached;
            }

            const result = await originalMethod.apply(this, args);

            if (result !== null && result !== undefined) {
                await cacheManager.set(key, result, cacheType, ttl);
            }

            return result;
        };

        return descriptor;
    };
}

/**
 * Refresh cache decorator
 * Updates cache in background after returning stale data
 */
function refreshCache(cacheType = 'default', staleTime = 60) {
    return function (target, propertyKey, descriptor) {
        const originalMethod = descriptor.value;

        descriptor.value = async function (...args) {
            const key = generateDefaultKey(propertyKey, args);
            const cached = await cacheManager.get(key, cacheType);
            
            if (cached !== null) {
                // Check if data is stale
                const cacheAge = await getCacheAge(key, cacheType);
                if (cacheAge > staleTime) {
                    // Return stale data immediately
                    // Refresh in background
                    setImmediate(async () => {
                        try {
                            const fresh = await originalMethod.apply(this, args);
                            await cacheManager.set(key, fresh, cacheType);
                        } catch (error) {
                            console.error('Background cache refresh failed:', error);
                        }
                    });
                }
                return cached;
            }

            // No cache, fetch fresh
            const result = await originalMethod.apply(this, args);
            if (result !== null && result !== undefined) {
                await cacheManager.set(key, result, cacheType);
            }

            return result;
        };

        return descriptor;
    };
}

/**
 * Helper functions
 */
function generateDefaultKey(methodName, args) {
    const hash = crypto.createHash('md5');
    hash.update(methodName);
    hash.update(JSON.stringify(args));
    return hash.digest('hex');
}

async function getCacheAge(key, cacheType) {
    try {
        if (cacheManager.connected) {
            const ttl = await cacheManager.ttlAsync(cacheManager._getPrefixedKey(key, cacheType));
            const maxTTL = cacheManager.cacheConfig[cacheType]?.ttl || cacheManager.defaultTTL;
            return maxTTL - ttl;
        }
    } catch (error) {
        console.error('Failed to get cache age:', error);
    }
    return 0;
}

/**
 * Cache key generators
 */
const keyGenerators = {
    byId: (id) => id.toString(),
    byIds: (...ids) => ids.join(':'),
    byObject: (obj) => JSON.stringify(obj),
    byUserAndId: (userId, id) => `${userId}:${id}`,
    byProjectAndType: (projectId, type) => `${projectId}:${type}`
};

module.exports = {
    cache,
    invalidateCache,
    cacheIf,
    warmupCache,
    timeBasedCache,
    refreshCache,
    keyGenerators
};