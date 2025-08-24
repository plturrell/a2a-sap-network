/**
 * @fileoverview Enterprise Request/Response Compression Middleware
 * @description Advanced compression middleware with SAP enterprise features,
 * adaptive compression algorithms, performance monitoring, and content-aware optimization
 * @module compression-middleware
 * @since 1.0.0
 * @author A2A Network Team
 */

const compression = require('compression');
const zlib = require('zlib');
const { Transform } = require('stream');

/**
 * Enterprise Compression Manager
 */
class EnterpriseCompressionManager {
    constructor(options = {}) {
        this.options = {
            // Compression thresholds
            threshold: options.threshold || 1024, // 1KB minimum
            level: options.level || 6, // Default compression level
            memLevel: options.memLevel || 8,
            windowBits: options.windowBits || 15,
            
            // Content-aware settings
            jsonLevel: options.jsonLevel || 9, // High compression for JSON
            textLevel: options.textLevel || 7, // Medium-high for text
            binaryLevel: options.binaryLevel || 3, // Low for binary
            
            // Performance settings
            chunkSize: options.chunkSize || 64 * 1024, // 64KB chunks
            maxMemory: options.maxMemory || 256 * 1024 * 1024, // 256MB max
            
            // Enterprise features
            enableMetrics: options.enableMetrics !== false,
            enableBrotli: options.enableBrotli !== false,
            enableContentAware: options.enableContentAware !== false,
            enableStreamingCompression: options.enableStreamingCompression !== false
        };
        
        this.metrics = {
            totalRequests: 0,
            compressedRequests: 0,
            totalBytesIn: 0,
            totalBytesOut: 0,
            compressionRatio: 0,
            avgCompressionTime: 0,
            algorithmUsage: {
                gzip: 0,
                deflate: 0,
                brotli: 0,
                none: 0
            }
        };
        
        this.compressionCache = new Map();
        this.log = require('@sap/cds').log('compression');
    }

    /**
     * Get compression middleware
     */
    getMiddleware() {
        return (req, res, next) => {
            const startTime = Date.now();
            this.metrics.totalRequests++;
            
            // Determine optimal compression strategy
            const strategy = this.determineCompressionStrategy(req, res);
            
            if (strategy.skip) {
                this.metrics.algorithmUsage.none++;
                return next();
            }
            
            // Apply content-aware compression
            const compressionOptions = this.getCompressionOptions(strategy);
            
            // Wrap response to collect metrics
            this.wrapResponse(res, strategy, startTime);
            
            // Apply compression middleware
            const compressionMiddleware = this.createCompressionMiddleware(compressionOptions);
            compressionMiddleware(req, res, next);
        };
    }

    /**
     * Determine optimal compression strategy
     */
    determineCompressionStrategy(req, res) {
        const contentType = res.getHeader('content-type') || '';
        const userAgent = req.headers['user-agent'] || '';
        const acceptEncoding = req.headers['accept-encoding'] || '';
        
        // Skip compression for certain conditions
        if (this.shouldSkipCompression(req, res, contentType)) {
            return { skip: true };
        }
        
        // Determine best algorithm
        let algorithm = 'gzip'; // Default
        if (this.options.enableBrotli && acceptEncoding.includes('br')) {
            algorithm = 'br';
        } else if (acceptEncoding.includes('gzip')) {
            algorithm = 'gzip';
        } else if (acceptEncoding.includes('deflate')) {
            algorithm = 'deflate';
        }
        
        // Determine compression level based on content type
        let level = this.options.level;
        if (this.options.enableContentAware) {
            level = this.getContentAwareLevel(contentType);
        }
        
        return {
            algorithm,
            level,
            contentType,
            userAgent,
            skip: false
        };
    }

    /**
     * Check if compression should be skipped
     */
    shouldSkipCompression(req, res, contentType) {
        // Skip for already compressed content
        if (contentType.includes('gzip') || contentType.includes('compress')) {
            return true;
        }
        
        // Skip for images and videos
        if (contentType.startsWith('image/') || contentType.startsWith('video/')) {
            return true;
        }
        
        // Skip for WebSocket upgrades
        if (req.headers.upgrade) {
            return true;
        }
        
        // Skip for Server-Sent Events
        if (contentType.includes('text/event-stream')) {
            return true;
        }
        
        // Skip for small responses (will be determined later)
        const contentLength = res.getHeader('content-length');
        if (contentLength && parseInt(contentLength) < this.options.threshold) {
            return true;
        }
        
        return false;
    }

    /**
     * Get content-aware compression level
     */
    getContentAwareLevel(contentType) {
        if (contentType.includes('application/json') || contentType.includes('application/xml')) {
            return this.options.jsonLevel;
        }
        
        if (contentType.includes('text/')) {
            return this.options.textLevel;
        }
        
        if (contentType.includes('application/octet-stream')) {
            return this.options.binaryLevel;
        }
        
        return this.options.level;
    }

    /**
     * Get compression options for middleware
     */
    getCompressionOptions(strategy) {
        const options = {
            threshold: this.options.threshold,
            level: strategy.level,
            memLevel: this.options.memLevel,
            windowBits: this.options.windowBits,
            chunkSize: this.options.chunkSize,
            
            // Custom filter function
            filter: (req, res) => {
                return !this.shouldSkipCompression(req, res, res.getHeader('content-type') || '');
            }
        };
        
        // Algorithm-specific options
        if (strategy.algorithm === 'br') {
            options.brotliOptions = {
                [zlib.constants.BROTLI_PARAM_MODE]: zlib.constants.BROTLI_MODE_TEXT,
                [zlib.constants.BROTLI_PARAM_QUALITY]: Math.min(strategy.level, 11),
                [zlib.constants.BROTLI_PARAM_SIZE_HINT]: this.options.chunkSize
            };
        }
        
        return options;
    }

    /**
     * Create compression middleware based on strategy
     */
    createCompressionMiddleware(options) {
        return compression(options);
    }

    /**
     * Wrap response to collect metrics
     */
    wrapResponse(res, strategy, startTime) {
        const originalEnd = res.end;
        const originalWrite = res.write;
        let totalBytes = 0;
        
        // Track written bytes
        res.write = function(chunk, encoding) {
            if (chunk) {
                totalBytes += Buffer.isBuffer(chunk) ? chunk.length : Buffer.byteLength(chunk, encoding);
            }
            return originalWrite.call(this, chunk, encoding);
        };
        
        // Track metrics on response end
        res.end = function(chunk, encoding) {
            if (chunk) {
                totalBytes += Buffer.isBuffer(chunk) ? chunk.length : Buffer.byteLength(chunk, encoding);
            }
            
            const compressionTime = Date.now() - startTime;
            const compressedSize = parseInt(res.getHeader('content-length')) || totalBytes;
            
            // Update metrics
            if (res.getHeader('content-encoding')) {
                strategy.parent.metrics.compressedRequests++;
                strategy.parent.metrics.totalBytesIn += totalBytes;
                strategy.parent.metrics.totalBytesOut += compressedSize;
                strategy.parent.metrics.algorithmUsage[strategy.algorithm]++;
                
                // Update average compression time
                const alpha = 0.1;
                strategy.parent.metrics.avgCompressionTime = 
                    strategy.parent.metrics.avgCompressionTime === 0 
                        ? compressionTime
                        : (alpha * compressionTime) + ((1 - alpha) * strategy.parent.metrics.avgCompressionTime);
                
                // Update compression ratio
                if (strategy.parent.metrics.totalBytesIn > 0) {
                    strategy.parent.metrics.compressionRatio = 
                        (1 - (strategy.parent.metrics.totalBytesOut / strategy.parent.metrics.totalBytesIn)) * 100;
                }
                
                strategy.parent.log.debug('Response compressed', {
                    algorithm: strategy.algorithm,
                    originalSize: totalBytes,
                    compressedSize,
                    ratio: `${((1 - (compressedSize / totalBytes)) * 100).toFixed(2)  }%`,
                    time: `${compressionTime  }ms`
                });
            }
            
            return originalEnd.call(this, chunk, encoding);
        };
        
        // Add parent reference for metrics access
        strategy.parent = this;
    }

    /**
     * Get compression metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            compressionRate: this.metrics.totalRequests > 0 
                ? `${(this.metrics.compressedRequests / this.metrics.totalRequests * 100).toFixed(2)  }%`
                : '0%',
            avgCompressionRatio: `${this.metrics.compressionRatio.toFixed(2)  }%`,
            totalSaved: this.metrics.totalBytesIn - this.metrics.totalBytesOut,
            avgCompressionTime: `${this.metrics.avgCompressionTime.toFixed(2)  }ms`
        };
    }

    /**
     * Reset metrics
     */
    resetMetrics() {
        this.metrics = {
            totalRequests: 0,
            compressedRequests: 0,
            totalBytesIn: 0,
            totalBytesOut: 0,
            compressionRatio: 0,
            avgCompressionTime: 0,
            algorithmUsage: {
                gzip: 0,
                deflate: 0,
                brotli: 0,
                none: 0
            }
        };
    }
}

module.exports = {
    EnterpriseCompressionManager,
    createCompressionMiddleware: (options = {}) => {
        const manager = new EnterpriseCompressionManager(options);
        return manager.getMiddleware();
    }
};