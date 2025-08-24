/**
 * @fileoverview Performance Optimizer for CAP/Glean Analysis
 * @module performanceOptimizer
 * @since 1.0.0
 * 
 * Provides caching, batching, and optimization for large CAP projects
 */

const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');

class PerformanceOptimizer {
    constructor(cacheDir = './cache/glean') {
        this.cacheDir = cacheDir;
        this.memoryCache = new Map();
        this.batchSize = 10; // Process files in batches
        this.cacheTTL = 24 * 60 * 60 * 1000; // 24 hours
        this.performanceMetrics = {
            cacheHits: 0,
            cacheMisses: 0,
            processingTime: {},
            filesSizeProcessed: 0,
            batchesProcessed: 0
        };
    }

    async initialize() {
        try {
            await fs.mkdir(this.cacheDir, { recursive: true });
            // console.log(`✅ Performance optimizer initialized with cache at: ${this.cacheDir}`);
        } catch (error) {
            console.warn(`⚠️ Could not initialize cache directory: ${error.message}`);
        }
    }

    /**
     * Generate cache key for file content
     */
    generateCacheKey(filePath, content) {
        const hash = crypto.createHash('sha256');
        hash.update(filePath);
        hash.update(content);
        return hash.digest('hex');
    }

    /**
     * Check if file has been modified since last cache
     */
    async isFileModified(filePath, cacheKey) {
        try {
            const stats = await fs.stat(filePath);
            const cacheFile = path.join(this.cacheDir, `${cacheKey}.meta`);
            
            if (await this.fileExists(cacheFile)) {
                const cacheStats = JSON.parse(await fs.readFile(cacheFile, 'utf8'));
                return stats.mtime.getTime() > cacheStats.mtime;
            }
            
            return true;
        } catch (error) {
            return true; // If we can't check, assume modified
        }
    }

    /**
     * Get cached result for file
     */
    async getCachedResult(filePath, content) {
        const cacheKey = this.generateCacheKey(filePath, content);
        
        // Check memory cache first
        if (this.memoryCache.has(cacheKey)) {
            this.performanceMetrics.cacheHits++;
            return this.memoryCache.get(cacheKey);
        }
        
        // Check disk cache
        try {
            const cacheFile = path.join(this.cacheDir, `${cacheKey}.json`);
            const metaFile = path.join(this.cacheDir, `${cacheKey}.meta`);
            
            if (await this.fileExists(cacheFile) && await this.fileExists(metaFile)) {
                const meta = JSON.parse(await fs.readFile(metaFile, 'utf8'));
                
                // Check if cache is still valid
                if (Date.now() - meta.timestamp < this.cacheTTL) {
                    const result = JSON.parse(await fs.readFile(cacheFile, 'utf8'));
                    
                    // Store in memory cache for faster access
                    this.memoryCache.set(cacheKey, result);
                    this.performanceMetrics.cacheHits++;
                    
                    return result;
                }
            }
        } catch (error) {
            console.warn(`⚠️ Cache read error for ${filePath}: ${error.message}`);
        }
        
        this.performanceMetrics.cacheMisses++;
        return null;
    }

    /**
     * Store result in cache
     */
    async setCachedResult(filePath, content, result) {
        const cacheKey = this.generateCacheKey(filePath, content);
        
        // Store in memory cache
        this.memoryCache.set(cacheKey, result);
        
        // Store in disk cache
        try {
            const cacheFile = path.join(this.cacheDir, `${cacheKey}.json`);
            const metaFile = path.join(this.cacheDir, `${cacheKey}.meta`);
            
            const stats = await fs.stat(filePath);
            const meta = {
                filePath,
                timestamp: Date.now(),
                mtime: stats.mtime.getTime(),
                size: stats.size
            };
            
            await fs.writeFile(cacheFile, JSON.stringify(result, null, 2));
            await fs.writeFile(metaFile, JSON.stringify(meta, null, 2));
            
        } catch (error) {
            console.warn(`⚠️ Cache write error for ${filePath}: ${error.message}`);
        }
    }

    /**
     * Process files in optimized batches
     */
    async processBatch(files, processor) {
        const startTime = Date.now();
        const results = [];
        
        // console.log(`📦 Processing batch of ${files.length} files...`);
        
        // Process files in parallel but limit concurrency
        const concurrency = Math.min(this.batchSize, files.length);
        const promises = [];
        
        for (let i = 0; i < files.length; i += concurrency) {
            const batch = files.slice(i, i + concurrency);
            const batchPromises = batch.map(async (file) => {
                try {
                    const content = await fs.readFile(file, 'utf8');
                    
                    // Check cache first
                    let result = await this.getCachedResult(file, content);
                    
                    if (!result) {
                        // Process file
                        const processingStart = Date.now();
                        result = await processor(file, content);
                        
                        // Cache result
                        await this.setCachedResult(file, content, result);
                        
                        // Track processing time
                        const processingTime = Date.now() - processingStart;
                        this.performanceMetrics.processingTime[file] = processingTime;
                    }
                    
                    this.performanceMetrics.filesSizeProcessed += content.length;
                    return { file, result, cached: !!result };
                    
                } catch (error) {
                    console.error(`❌ Error processing ${file}: ${error.message}`);
                    return { file, error: error.message };
                }
            });
            
            promises.push(...batchPromises);
        }
        
        const batchResults = await Promise.all(promises);
        results.push(...batchResults);
        
        this.performanceMetrics.batchesProcessed++;
        
        const processingTime = Date.now() - startTime;
        // console.log(`✅ Batch processed in ${processingTime}ms (${files.length} files)`);
        
        return results;
    }

    /**
     * Optimize fact database for faster queries
     */
    optimizeFactDatabase(factBatches) {
        const startTime = Date.now();
        // console.log('🔧 Optimizing fact database...');
        
        // Create indexes for faster lookups
        const indexes = {
            byFile: new Map(),
            byType: new Map(),
            byEntity: new Map(),
            byService: new Map()
        };
        
        // Build indexes
        Object.entries(factBatches).forEach(([predicate, facts]) => {
            facts.forEach((fact, index) => {
                // Index by file
                if (fact.value && fact.value.file) {
                    if (!indexes.byFile.has(fact.value.file)) {
                        indexes.byFile.set(fact.value.file, []);
                    }
                    indexes.byFile.get(fact.value.file).push({ predicate, index, fact });
                }
                
                // Index by type
                if (!indexes.byType.has(predicate)) {
                    indexes.byType.set(predicate, []);
                }
                indexes.byType.get(predicate).push({ index, fact });
                
                // Index by entity name
                if (fact.value && (fact.value.name || fact.value.entity)) {
                    const entityName = fact.value.name || fact.value.entity;
                    if (!indexes.byEntity.has(entityName)) {
                        indexes.byEntity.set(entityName, []);
                    }
                    indexes.byEntity.get(entityName).push({ predicate, index, fact });
                }
                
                // Index by service name
                if (fact.value && fact.value.service) {
                    if (!indexes.byService.has(fact.value.service)) {
                        indexes.byService.set(fact.value.service, []);
                    }
                    indexes.byService.get(fact.value.service).push({ predicate, index, fact });
                }
            });
        });
        
        const optimizationTime = Date.now() - startTime;
        // console.log(`✅ Fact database optimized in ${optimizationTime}ms`);
        
        return {
            factBatches,
            indexes,
            metadata: {
                totalFacts: Object.values(factBatches).reduce((sum, facts) => sum + facts.length, 0),
                predicateCount: Object.keys(factBatches).length,
                optimizationTime
            }
        };
    }

    /**
     * Smart query executor with caching
     */
    createOptimizedQueryExecutor(factDatabase) {
        const queryCache = new Map();
        const indexes = factDatabase.indexes;
        
        return {
            execute: (query) => {
                const queryHash = crypto.createHash('sha256').update(JSON.stringify(query)).digest('hex');
                
                // Check query cache
                if (queryCache.has(queryHash)) {
                    this.performanceMetrics.cacheHits++;
                    return queryCache.get(queryHash);
                }
                
                const startTime = Date.now();
                
                // Use indexes for optimized querying
                const results = this.executeOptimizedQuery(query, factDatabase, indexes);
                
                const executionTime = Date.now() - startTime;
                
                // Cache results for frequently used queries
                queryCache.set(queryHash, results);
                this.performanceMetrics.cacheMisses++;
                
                // console.log(`🔍 Query executed in ${executionTime}ms (${results.length} results)`);
                
                return results;
            },
            
            getStats: () => ({
                cacheSize: queryCache.size,
                ...this.performanceMetrics
            })
        };
    }

    executeOptimizedQuery(query, factDatabase, indexes) {
        // Simple optimization: use indexes when possible
        const { factBatches } = factDatabase;
        
        // If query targets specific predicate, use type index
        if (query.predicate && indexes.byType.has(query.predicate)) {
            return this.filterFacts(indexes.byType.get(query.predicate), query.conditions);
        }
        
        // If query targets specific file, use file index
        if (query.conditions && query.conditions.file && indexes.byFile.has(query.conditions.file)) {
            return this.filterFacts(indexes.byFile.get(query.conditions.file), query.conditions);
        }
        
        // If query targets specific entity, use entity index
        if (query.conditions && query.conditions.entity && indexes.byEntity.has(query.conditions.entity)) {
            return this.filterFacts(indexes.byEntity.get(query.conditions.entity), query.conditions);
        }
        
        // Fallback to full scan
        const allFacts = [];
        Object.entries(factBatches).forEach(([predicate, facts]) => {
            facts.forEach((fact, index) => {
                allFacts.push({ predicate, index, fact });
            });
        });
        
        return this.filterFacts(allFacts, query.conditions || {});
    }

    filterFacts(indexedFacts, conditions) {
        return indexedFacts.filter(({ fact }) => {
            return Object.entries(conditions).every(([key, value]) => {
                if (key === 'file') return fact.value && fact.value.file === value;
                if (key === 'entity') return fact.value && (fact.value.name === value || fact.value.entity === value);
                if (key === 'service') return fact.value && fact.value.service === value;
                return true;
            });
        }).map(({ fact }) => fact);
    }

    /**
     * Clean up old cache files
     */
    async cleanupCache() {
        try {
            const files = await fs.readdir(this.cacheDir);
            let cleanedCount = 0;
            
            for (const file of files) {
                if (file.endsWith('.meta')) {
                    const metaPath = path.join(this.cacheDir, file);
                    const meta = JSON.parse(await fs.readFile(metaPath, 'utf8'));
                    
                    // Remove old cache entries
                    if (Date.now() - meta.timestamp > this.cacheTTL) {
                        const cacheFile = file.replace('.meta', '.json');
                        await fs.unlink(metaPath);
                        await fs.unlink(path.join(this.cacheDir, cacheFile));
                        cleanedCount++;
                    }
                }
            }
            
            // console.log(`🧹 Cleaned up ${cleanedCount} old cache entries`);
        } catch (error) {
            console.warn(`⚠️ Cache cleanup error: ${error.message}`);
        }
    }

    /**
     * Get performance statistics
     */
    getPerformanceStats() {
        const cacheHitRate = this.performanceMetrics.cacheHits / 
            (this.performanceMetrics.cacheHits + this.performanceMetrics.cacheMisses) * 100;
        
        return {
            ...this.performanceMetrics,
            cacheHitRate: `${cacheHitRate.toFixed(2)}%`,
            memoryCache: {
                size: this.memoryCache.size,
                approximateMemoryUsage: this.estimateMemoryUsage()
            }
        };
    }

    estimateMemoryUsage() {
        // Rough estimate of memory cache usage
        let size = 0;
        this.memoryCache.forEach(value => {
            size += JSON.stringify(value).length * 2; // Rough estimate
        });
        return `${(size / 1024 / 1024).toFixed(2)} MB`;
    }

    async fileExists(filePath) {
        try {
            await fs.access(filePath);
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Adaptive batch sizing based on system performance
     */
    adaptBatchSize(averageProcessingTime) {
        if (averageProcessingTime < 100) {
            this.batchSize = Math.min(20, this.batchSize + 2);
        } else if (averageProcessingTime > 500) {
            this.batchSize = Math.max(5, this.batchSize - 2);
        }
        
        // console.log(`📊 Adapted batch size to ${this.batchSize} based on performance`);
    }
}

module.exports = PerformanceOptimizer;