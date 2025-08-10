/**
 * Enterprise Caching Service
 * 
 * Provides intelligent caching with offline support, cache invalidation,
 * and storage management following SAP enterprise patterns.
 * 
 * @author SAP SE
 * @since 1.0.0
 * @version 1.0.0
 */
sap.ui.define([
    "sap/base/Log"
], function(Log) {
    "use strict";

    var CacheService = {

        /* =========================================================== */
        /* Constants                                                   */
        /* =========================================================== */

        CACHE_PREFIX: "a2a-cache-",
        METADATA_PREFIX: "a2a-meta-",
        MAX_CACHE_SIZE: 50 * 1024 * 1024, // 50MB
        DEFAULT_TTL: 30 * 60 * 1000, // 30 minutes
        CLEANUP_INTERVAL: 5 * 60 * 1000, // 5 minutes

        /* =========================================================== */
        /* Lifecycle                                                   */
        /* =========================================================== */

        /**
         * Initialize the cache service
         * @public
         * @since 1.0.0
         */
        init: function() {
            this._initializeStorage();
            this._startCleanupTimer();
            this._registerStorageEvents();
            
            Log.info("CacheService initialized", { service: "CacheService" });
        },

        /**
         * Destroy the cache service
         * @public
         * @since 1.0.0
         */
        destroy: function() {
            if (this._cleanupTimer) {
                clearInterval(this._cleanupTimer);
            }
            
            Log.info("CacheService destroyed", { service: "CacheService" });
        },

        /* =========================================================== */
        /* Public API                                                  */
        /* =========================================================== */

        /**
         * Store data in cache with optional TTL
         * @public
         * @param {string} key Cache key
         * @param {*} data Data to cache
         * @param {object} options Caching options
         * @param {number} [options.ttl] Time to live in milliseconds
         * @param {Array<string>} [options.tags] Tags for cache invalidation
         * @param {boolean} [options.compress] Whether to compress the data
         * @param {string} [options.version] Data version for cache invalidation
         * @returns {boolean} Whether caching was successful
         * @since 1.0.0
         */
        set: function(key, data, options) {
            options = options || {};
            
            try {
                var cacheKey = this.CACHE_PREFIX + key;
                var metaKey = this.METADATA_PREFIX + key;
                var now = Date.now();
                
                // Prepare data for storage
                var cacheData = {
                    data: data,
                    timestamp: now,
                    compressed: false
                };
                
                // Compress large objects if requested
                if (options.compress && this._shouldCompress(data)) {
                    cacheData.data = this._compress(data);
                    cacheData.compressed = true;
                }
                
                // Prepare metadata
                var metadata = {
                    key: key,
                    created: now,
                    lastAccessed: now,
                    ttl: options.ttl || this.DEFAULT_TTL,
                    expiresAt: now + (options.ttl || this.DEFAULT_TTL),
                    tags: options.tags || [],
                    version: options.version,
                    size: this._calculateSize(cacheData),
                    accessCount: 0,
                    compressed: cacheData.compressed
                };
                
                // Check storage limits
                if (!this._checkStorageSpace(metadata.size)) {
                    Log.warn("Cache storage limit exceeded", { key: key, size: metadata.size });
                    this._evictOldEntries();
                }
                
                // Store data and metadata
                this._setItem(cacheKey, JSON.stringify(cacheData));
                this._setItem(metaKey, JSON.stringify(metadata));
                
                Log.debug("Data cached successfully", { 
                    key: key, 
                    size: metadata.size,
                    ttl: metadata.ttl,
                    tags: metadata.tags
                });
                
                return true;
            } catch (error) {
                Log.error("Failed to cache data", { key: key, error: error.message });
                return false;
            }
        },

        /**
         * Retrieve data from cache
         * @public
         * @param {string} key Cache key
         * @param {object} options Retrieval options
         * @param {boolean} [options.updateAccess=true] Whether to update access time
         * @returns {*} Cached data or null if not found/expired
         * @since 1.0.0
         */
        get: function(key, options) {
            options = options || {};
            options.updateAccess = options.updateAccess !== false;
            
            try {
                var cacheKey = this.CACHE_PREFIX + key;
                var metaKey = this.METADATA_PREFIX + key;
                
                // Get metadata first
                var metadataStr = this._getItem(metaKey);
                if (!metadataStr) {
                    return null;
                }
                
                var metadata = JSON.parse(metadataStr);
                var now = Date.now();
                
                // Check expiration
                if (now > metadata.expiresAt) {
                    Log.debug("Cache entry expired", { key: key, expiresAt: new Date(metadata.expiresAt) });
                    this.remove(key);
                    return null;
                }
                
                // Get cached data
                var cacheDataStr = this._getItem(cacheKey);
                if (!cacheDataStr) {
                    // Metadata exists but data doesn't - clean up
                    this.remove(key);
                    return null;
                }
                
                var cacheData = JSON.parse(cacheDataStr);
                
                // Decompress if needed
                var data = cacheData.compressed ? 
                    this._decompress(cacheData.data) : cacheData.data;
                
                // Update access statistics
                if (options.updateAccess) {
                    metadata.lastAccessed = now;
                    metadata.accessCount++;
                    this._setItem(metaKey, JSON.stringify(metadata));
                }
                
                Log.debug("Cache hit", { 
                    key: key, 
                    age: now - metadata.created,
                    accessCount: metadata.accessCount 
                });
                
                return data;
            } catch (error) {
                Log.error("Failed to retrieve cached data", { key: key, error: error.message });
                return null;
            }
        },

        /**
         * Remove data from cache
         * @public
         * @param {string} key Cache key
         * @returns {boolean} Whether removal was successful
         * @since 1.0.0
         */
        remove: function(key) {
            try {
                var cacheKey = this.CACHE_PREFIX + key;
                var metaKey = this.METADATA_PREFIX + key;
                
                this._removeItem(cacheKey);
                this._removeItem(metaKey);
                
                Log.debug("Cache entry removed", { key: key });
                return true;
            } catch (error) {
                Log.error("Failed to remove cache entry", { key: key, error: error.message });
                return false;
            }
        },

        /**
         * Clear all cache entries
         * @public
         * @returns {boolean} Whether clearing was successful
         * @since 1.0.0
         */
        clear: function() {
            try {
                var keys = this._getAllKeys();
                var removed = 0;
                
                keys.forEach(function(key) {
                    if (key.startsWith(this.CACHE_PREFIX) || key.startsWith(this.METADATA_PREFIX)) {
                        this._removeItem(key);
                        removed++;
                    }
                }.bind(this));
                
                Log.info("Cache cleared", { entriesRemoved: removed });
                return true;
            } catch (error) {
                Log.error("Failed to clear cache", { error: error.message });
                return false;
            }
        },

        /**
         * Invalidate cache entries by tags
         * @public
         * @param {Array<string>} tags Tags to invalidate
         * @returns {number} Number of entries invalidated
         * @since 1.0.0
         */
        invalidateByTags: function(tags) {
            if (!Array.isArray(tags) || tags.length === 0) {
                return 0;
            }
            
            try {
                var keys = this._getAllKeys();
                var invalidated = 0;
                
                keys.forEach(function(key) {
                    if (key.startsWith(this.METADATA_PREFIX)) {
                        var metadataStr = this._getItem(key);
                        if (metadataStr) {
                            var metadata = JSON.parse(metadataStr);
                            if (metadata.tags && this._hasMatchingTag(metadata.tags, tags)) {
                                var cacheKey = metadata.key;
                                this.remove(cacheKey);
                                invalidated++;
                            }
                        }
                    }
                }.bind(this));
                
                Log.info("Cache invalidated by tags", { tags: tags, entriesInvalidated: invalidated });
                return invalidated;
            } catch (error) {
                Log.error("Failed to invalidate cache by tags", { tags: tags, error: error.message });
                return 0;
            }
        },

        /**
         * Get cache statistics
         * @public
         * @returns {object} Cache statistics
         * @since 1.0.0
         */
        getStats: function() {
            try {
                var keys = this._getAllKeys();
                var stats = {
                    totalEntries: 0,
                    totalSize: 0,
                    oldestEntry: null,
                    newestEntry: null,
                    mostAccessed: null,
                    storageUsed: 0,
                    storageLimit: this.MAX_CACHE_SIZE,
                    hitRate: 0,
                    entries: []
                };
                
                keys.forEach(function(key) {
                    if (key.startsWith(this.METADATA_PREFIX)) {
                        var metadataStr = this._getItem(key);
                        if (metadataStr) {
                            var metadata = JSON.parse(metadataStr);
                            stats.totalEntries++;
                            stats.totalSize += metadata.size;
                            stats.entries.push({
                                key: metadata.key,
                                created: metadata.created,
                                lastAccessed: metadata.lastAccessed,
                                accessCount: metadata.accessCount,
                                size: metadata.size,
                                tags: metadata.tags
                            });
                        }
                    }
                }.bind(this));
                
                // Calculate additional statistics
                if (stats.entries.length > 0) {
                    stats.entries.sort((a, b) => a.created - b.created);
                    stats.oldestEntry = stats.entries[0];
                    stats.newestEntry = stats.entries[stats.entries.length - 1];
                    
                    stats.entries.sort((a, b) => b.accessCount - a.accessCount);
                    stats.mostAccessed = stats.entries[0];
                }
                
                stats.storageUsed = this._getStorageUsage();
                
                return stats;
            } catch (error) {
                Log.error("Failed to get cache stats", { error: error.message });
                return { error: error.message };
            }
        },

        /* =========================================================== */
        /* Private Methods                                             */
        /* =========================================================== */

        /**
         * Initialize storage mechanism
         * @private
         * @since 1.0.0
         */
        _initializeStorage: function() {
            // Test localStorage availability
            try {
                var testKey = "a2a-test";
                localStorage.setItem(testKey, "test");
                localStorage.removeItem(testKey);
                this._storageType = "localStorage";
            } catch (e) {
                // Fallback to sessionStorage
                try {
                    sessionStorage.setItem("a2a-test", "test");
                    sessionStorage.removeItem("a2a-test");
                    this._storageType = "sessionStorage";
                } catch (e2) {
                    // Fallback to memory storage
                    this._storageType = "memory";
                    this._memoryStorage = {};
                }
            }
            
            Log.info("Cache storage initialized", { storageType: this._storageType });
        },

        /**
         * Start cleanup timer
         * @private
         * @since 1.0.0
         */
        _startCleanupTimer: function() {
            this._cleanupTimer = setInterval(function() {
                this._cleanupExpiredEntries();
            }.bind(this), this.CLEANUP_INTERVAL);
        },

        /**
         * Register storage events
         * @private
         * @since 1.0.0
         */
        _registerStorageEvents: function() {
            if (this._storageType === "localStorage") {
                window.addEventListener("storage", function(e) {
                    if (e.key && (e.key.startsWith(this.CACHE_PREFIX) || e.key.startsWith(this.METADATA_PREFIX))) {
                        Log.debug("Storage event detected", { key: e.key, newValue: e.newValue });
                    }
                }.bind(this));
            }
        },

        /**
         * Set item in storage
         * @private
         * @param {string} key Storage key
         * @param {string} value Storage value
         * @since 1.0.0
         */
        _setItem: function(key, value) {
            if (this._storageType === "memory") {
                this._memoryStorage[key] = value;
            } else {
                var storage = this._storageType === "localStorage" ? localStorage : sessionStorage;
                storage.setItem(key, value);
            }
        },

        /**
         * Get item from storage
         * @private
         * @param {string} key Storage key
         * @returns {string} Storage value
         * @since 1.0.0
         */
        _getItem: function(key) {
            if (this._storageType === "memory") {
                return this._memoryStorage[key] || null;
            } else {
                var storage = this._storageType === "localStorage" ? localStorage : sessionStorage;
                return storage.getItem(key);
            }
        },

        /**
         * Remove item from storage
         * @private
         * @param {string} key Storage key
         * @since 1.0.0
         */
        _removeItem: function(key) {
            if (this._storageType === "memory") {
                delete this._memoryStorage[key];
            } else {
                var storage = this._storageType === "localStorage" ? localStorage : sessionStorage;
                storage.removeItem(key);
            }
        },

        /**
         * Get all storage keys
         * @private
         * @returns {Array<string>} Storage keys
         * @since 1.0.0
         */
        _getAllKeys: function() {
            if (this._storageType === "memory") {
                return Object.keys(this._memoryStorage);
            } else {
                var storage = this._storageType === "localStorage" ? localStorage : sessionStorage;
                var keys = [];
                for (var i = 0; i < storage.length; i++) {
                    keys.push(storage.key(i));
                }
                return keys;
            }
        },

        /**
         * Calculate data size
         * @private
         * @param {*} data Data to calculate size for
         * @returns {number} Size in bytes
         * @since 1.0.0
         */
        _calculateSize: function(data) {
            return new Blob([JSON.stringify(data)]).size;
        },

        /**
         * Check if data should be compressed
         * @private
         * @param {*} data Data to check
         * @returns {boolean} Whether to compress
         * @since 1.0.0
         */
        _shouldCompress: function(data) {
            var size = this._calculateSize(data);
            return size > 10240; // Compress if larger than 10KB
        },

        /**
         * Compress data using LZ-string algorithm
         * @private
         * @param {*} data Data to compress
         * @returns {string} Compressed data
         * @since 1.0.0
         */
        _compress: function(data) {
            try {
                var jsonString = JSON.stringify(data);
                
                // Implement LZ77-based compression algorithm
                // This is a simplified version suitable for browser environments
                var compressed = this._lzCompress(jsonString);
                
                Log.debug("Data compressed", {
                    originalSize: jsonString.length,
                    compressedSize: compressed.length,
                    ratio: (compressed.length / jsonString.length).toFixed(2)
                });
                
                return compressed;
            } catch (error) {
                Log.error("Compression failed, using uncompressed data", error);
                return JSON.stringify(data);
            }
        },

        /**
         * Decompress data using LZ-string algorithm
         * @private
         * @param {string} compressedData Compressed data
         * @returns {*} Decompressed data
         * @since 1.0.0
         */
        _decompress: function(compressedData) {
            try {
                // Try to decompress using LZ77 algorithm
                var decompressed = this._lzDecompress(compressedData);
                return JSON.parse(decompressed);
            } catch (error) {
                Log.warn("Decompression failed, trying as uncompressed data", error);
                // Fallback to treating as uncompressed JSON
                try {
                    return JSON.parse(compressedData);
                } catch (parseError) {
                    Log.error("Failed to parse cached data", parseError);
                    return null;
                }
            }
        },

        /**
         * Check storage space availability
         * @private
         * @param {number} requiredSize Required size in bytes
         * @returns {boolean} Whether space is available
         * @since 1.0.0
         */
        _checkStorageSpace: function(requiredSize) {
            var currentUsage = this._getStorageUsage();
            return (currentUsage + requiredSize) <= this.MAX_CACHE_SIZE;
        },

        /**
         * Get current storage usage
         * @private
         * @returns {number} Storage usage in bytes
         * @since 1.0.0
         */
        _getStorageUsage: function() {
            if (this._storageType === "memory") {
                return JSON.stringify(this._memoryStorage).length;
            } else {
                var storage = this._storageType === "localStorage" ? localStorage : sessionStorage;
                var usage = 0;
                for (var key in storage) {
                    if (key.startsWith(this.CACHE_PREFIX) || key.startsWith(this.METADATA_PREFIX)) {
                        usage += storage[key].length;
                    }
                }
                return usage;
            }
        },

        /**
         * Evict old entries to free space
         * @private
         * @since 1.0.0
         */
        _evictOldEntries: function() {
            var keys = this._getAllKeys();
            var metadataEntries = [];
            
            // Collect metadata entries
            keys.forEach(function(key) {
                if (key.startsWith(this.METADATA_PREFIX)) {
                    var metadataStr = this._getItem(key);
                    if (metadataStr) {
                        var metadata = JSON.parse(metadataStr);
                        metadataEntries.push(metadata);
                    }
                }
            }.bind(this));
            
            // Sort by access time (oldest first)
            metadataEntries.sort((a, b) => a.lastAccessed - b.lastAccessed);
            
            // Remove oldest entries until we have enough space
            var removed = 0;
            while (this._getStorageUsage() > (this.MAX_CACHE_SIZE * 0.8) && metadataEntries.length > 0) {
                var entry = metadataEntries.shift();
                this.remove(entry.key);
                removed++;
            }
            
            Log.info("Evicted old cache entries", { entriesRemoved: removed });
        },

        /**
         * Clean up expired entries
         * @private
         * @since 1.0.0
         */
        _cleanupExpiredEntries: function() {
            var keys = this._getAllKeys();
            var now = Date.now();
            var removed = 0;
            
            keys.forEach(function(key) {
                if (key.startsWith(this.METADATA_PREFIX)) {
                    var metadataStr = this._getItem(key);
                    if (metadataStr) {
                        var metadata = JSON.parse(metadataStr);
                        if (now > metadata.expiresAt) {
                            this.remove(metadata.key);
                            removed++;
                        }
                    }
                }
            }.bind(this));
            
            if (removed > 0) {
                Log.debug("Cleaned up expired cache entries", { entriesRemoved: removed });
            }
        },

        /**
         * Check if tags match
         * @private
         * @param {Array<string>} entryTags Entry tags
         * @param {Array<string>} targetTags Target tags
         * @returns {boolean} Whether tags match
         * @since 1.0.0
         */
        _hasMatchingTag: function(entryTags, targetTags) {
            return targetTags.some(function(tag) {
                return entryTags.indexOf(tag) !== -1;
            });
        },

        /* =========================================================== */
        /* LZ77-based Compression Implementation                       */
        /* =========================================================== */

        /**
         * LZ77-based compression implementation
         * @private
         * @param {string} input Input string to compress
         * @returns {string} Compressed string
         * @since 1.0.0
         */
        _lzCompress: function(input) {
            if (!input || typeof input !== 'string') {
                return input;
            }

            var dictionary = {};
            var data = (input + "").split("");
            var out = [];
            var currChar;
            var phrase = data[0];
            var code = 256;
            
            for (var i = 1; i < data.length; i++) {
                currChar = data[i];
                if (dictionary[phrase + currChar] !== undefined) {
                    phrase += currChar;
                } else {
                    out.push(phrase.length > 1 ? dictionary[phrase] : phrase.charCodeAt(0));
                    dictionary[phrase + currChar] = code;
                    code++;
                    phrase = currChar;
                }
            }
            out.push(phrase.length > 1 ? dictionary[phrase] : phrase.charCodeAt(0));
            
            // Convert to base64-like string for storage
            return this._encodeArray(out);
        },

        /**
         * LZ77-based decompression implementation
         * @private
         * @param {string} compressed Compressed string
         * @returns {string} Decompressed string
         * @since 1.0.0
         */
        _lzDecompress: function(compressed) {
            if (!compressed || typeof compressed !== 'string') {
                return compressed;
            }

            var dictionary = {};
            var data = this._decodeArray(compressed);
            var currChar = String.fromCharCode(data[0]);
            var oldPhrase = currChar;
            var out = [currChar];
            var code = 256;
            var phrase;
            
            for (var i = 1; i < data.length; i++) {
                var currCode = data[i];
                if (currCode < 256) {
                    phrase = String.fromCharCode(currCode);
                } else {
                    phrase = dictionary[currCode] ? dictionary[currCode] : (oldPhrase + currChar);
                }
                out.push(phrase);
                currChar = phrase.charAt(0);
                dictionary[code] = oldPhrase + currChar;
                code++;
                oldPhrase = phrase;
            }
            
            return out.join("");
        },

        /**
         * Encode number array to string
         * @private
         * @param {Array<number>} arr Array of numbers
         * @returns {string} Encoded string
         * @since 1.0.0
         */
        _encodeArray: function(arr) {
            var result = "";
            for (var i = 0; i < arr.length; i++) {
                result += String.fromCharCode(arr[i] + 256);
            }
            return btoa(result); // Base64 encode for safe storage
        },

        /**
         * Decode string to number array
         * @private
         * @param {string} str Encoded string
         * @returns {Array<number>} Decoded array
         * @since 1.0.0
         */
        _decodeArray: function(str) {
            var decoded = atob(str); // Base64 decode
            var arr = [];
            for (var i = 0; i < decoded.length; i++) {
                arr.push(decoded.charCodeAt(i) - 256);
            }
            return arr;
        }
    };

    return CacheService;
});