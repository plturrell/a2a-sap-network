/**
 * @fileoverview Enterprise Error Handler for CAP/Glean Analysis
 * @module enterpriseErrorHandler
 * @since 1.0.0
 * 
 * Provides comprehensive error handling, logging, and recovery for enterprise environments
 */

const fs = require('fs').promises;
const path = require('path');

class EnterpriseErrorHandler {
    constructor(logDir = './logs/glean', options = {}) {
        this.logDir = logDir;
        this.options = {
            maxLogFiles: options.maxLogFiles || 30,
            maxLogSize: options.maxLogSize || 100 * 1024 * 1024, // 100MB
            enableDetailedLogging: options.enableDetailedLogging !== false,
            enableRecovery: options.enableRecovery !== false,
            enableMetrics: options.enableMetrics !== false,
            ...options
        };
        
        this.errorTypes = new Map();
        this.recoveryStrategies = new Map();
        this.metrics = {
            totalErrors: 0,
            errorsByType: {},
            errorsByFile: {},
            recoveryAttempts: 0,
            successfulRecoveries: 0,
            criticalErrors: 0
        };
        
        this.initializeErrorTypes();
        this.initializeRecoveryStrategies();
    }

    async initialize() {
        try {
            await fs.mkdir(this.logDir, { recursive: true });
            // console.log(`✅ Enterprise error handler initialized with logs at: ${this.logDir}`);
            
            // Clean up old log files
            await this.cleanupOldLogs();
        } catch (error) {
            console.error(`❌ Failed to initialize error handler: ${error.message}`);
        }
    }

    initializeErrorTypes() {
        this.errorTypes.set('PARSE_ERROR', {
            severity: 'high',
            recoverable: true,
            category: 'syntax',
            description: 'File parsing failed due to syntax errors'
        });

        this.errorTypes.set('CDS_SYNTAX_ERROR', {
            severity: 'high',
            recoverable: true,
            category: 'cds',
            description: 'CDS file contains syntax errors'
        });

        this.errorTypes.set('MEMORY_ERROR', {
            severity: 'critical',
            recoverable: false,
            category: 'system',
            description: 'Insufficient memory to process file'
        });

        this.errorTypes.set('FILE_ACCESS_ERROR', {
            severity: 'medium',
            recoverable: true,
            category: 'io',
            description: 'Unable to read or write file'
        });

        this.errorTypes.set('CACHE_ERROR', {
            severity: 'low',
            recoverable: true,
            category: 'cache',
            description: 'Cache operation failed'
        });

        this.errorTypes.set('QUERY_ERROR', {
            severity: 'medium',
            recoverable: true,
            category: 'query',
            description: 'Query execution failed'
        });

        this.errorTypes.set('VALIDATION_ERROR', {
            severity: 'medium',
            recoverable: true,
            category: 'validation',
            description: 'Data validation failed'
        });

        this.errorTypes.set('NETWORK_ERROR', {
            severity: 'medium',
            recoverable: true,
            category: 'network',
            description: 'Network operation failed'
        });

        this.errorTypes.set('TIMEOUT_ERROR', {
            severity: 'medium',
            recoverable: true,
            category: 'timeout',
            description: 'Operation exceeded timeout limit'
        });

        this.errorTypes.set('CONFIGURATION_ERROR', {
            severity: 'high',
            recoverable: false,
            category: 'config',
            description: 'Invalid configuration detected'
        });
    }

    initializeRecoveryStrategies() {
        this.recoveryStrategies.set('PARSE_ERROR', async (error, context) => {
            // Try alternative parsers or fallback to regex parsing
            // console.log(`🔄 Attempting recovery for parse error in ${context.filePath}`);
            
            try {
                // Fallback to simpler parsing
                if (context.fallbackParser) {
                    const result = await context.fallbackParser(context.content, context.filePath);
                    return { success: true, result, method: 'fallback_parser' };
                }
                
                // Skip complex parts and parse what we can
                const sanitizedContent = this.sanitizeContent(context.content);
                if (sanitizedContent !== context.content) {
                    const result = await context.parser(sanitizedContent, context.filePath);
                    return { success: true, result, method: 'content_sanitization' };
                }
                
            } catch (recoveryError) {
                console.warn(`⚠️ Recovery failed: ${recoveryError.message}`);
            }
            
            return { success: false, method: 'no_recovery' };
        });

        this.recoveryStrategies.set('FILE_ACCESS_ERROR', async (error, context) => {
            // console.log(`🔄 Attempting recovery for file access error: ${context.filePath}`);
            
            try {
                // Wait and retry
                await this.delay(1000);
                const content = await fs.readFile(context.filePath, 'utf8');
                return { success: true, result: content, method: 'retry_after_delay' };
                
            } catch (retryError) {
                // Try alternative file paths
                const alternatives = this.generateAlternativePaths(context.filePath);
                
                for (const altPath of alternatives) {
                    try {
                        const content = await fs.readFile(altPath, 'utf8');
                        return { success: true, result: content, method: 'alternative_path', path: altPath };
                    } catch (altError) {
                        continue;
                    }
                }
            }
            
            return { success: false, method: 'no_recovery' };
        });

        this.recoveryStrategies.set('MEMORY_ERROR', async (error, context) => {
            // console.log(`🔄 Attempting recovery for memory error`);
            
            try {
                // Force garbage collection if available
                if (global.gc) {
                    global.gc();
                }
                
                // Process in smaller chunks
                if (context.content && context.content.length > 1000000) {
                    const chunks = this.splitIntoChunks(context.content, 100000);
                    const results = [];
                    
                    for (const chunk of chunks) {
                        const chunkResult = await context.parser(chunk, `${context.filePath}_chunk`);
                        results.push(chunkResult);
                    }
                    
                    return { success: true, result: this.mergeResults(results), method: 'chunked_processing' };
                }
                
            } catch (recoveryError) {
                console.warn(`⚠️ Memory recovery failed: ${recoveryError.message}`);
            }
            
            return { success: false, method: 'no_recovery' };
        });

        this.recoveryStrategies.set('CACHE_ERROR', async (error, context) => {
            // console.log(`🔄 Attempting recovery for cache error`);
            
            // Simply proceed without cache
            try {
                const result = await context.operation();
                return { success: true, result, method: 'skip_cache' };
            } catch (recoveryError) {
                return { success: false, method: 'no_recovery' };
            }
        });

        this.recoveryStrategies.set('TIMEOUT_ERROR', async (error, context) => {
            // console.log(`🔄 Attempting recovery for timeout error`);
            
            try {
                // Increase timeout and retry
                const extendedTimeout = (context.timeout || 30000) * 2;
                const result = await this.withTimeout(context.operation, extendedTimeout);
                return { success: true, result, method: 'extended_timeout' };
                
            } catch (recoveryError) {
                // Try with simplified processing
                if (context.simplifiedOperation) {
                    try {
                        const result = await context.simplifiedOperation();
                        return { success: true, result, method: 'simplified_operation' };
                    } catch (simplifiedError) {
                        // Continue to failure
                    }
                }
            }
            
            return { success: false, method: 'no_recovery' };
        });
    }

    /**
     * Handle error with comprehensive logging and recovery
     */
    async handleError(error, context = {}) {
        this.metrics.totalErrors++;
        
        const errorType = this.classifyError(error);
        const errorInfo = this.errorTypes.get(errorType);
        
        // Update metrics
        if (!this.metrics.errorsByType[errorType]) {
            this.metrics.errorsByType[errorType] = 0;
        }
        this.metrics.errorsByType[errorType]++;
        
        if (context.filePath) {
            if (!this.metrics.errorsByFile[context.filePath]) {
                this.metrics.errorsByFile[context.filePath] = 0;
            }
            this.metrics.errorsByFile[context.filePath]++;
        }
        
        if (errorInfo && errorInfo.severity === 'critical') {
            this.metrics.criticalErrors++;
        }
        
        // Log error details
        await this.logError(error, errorType, errorInfo, context);
        
        // Attempt recovery if possible
        if (this.options.enableRecovery && errorInfo && errorInfo.recoverable) {
            return await this.attemptRecovery(error, errorType, context);
        }
        
        // Return error information for caller to handle
        return {
            error,
            errorType,
            errorInfo,
            recovered: false
        };
    }

    classifyError(error) {
        const message = error.message.toLowerCase();
        const stack = error.stack ? error.stack.toLowerCase() : '';
        
        if (message.includes('parse') || message.includes('syntax')) {
            if (message.includes('cds') || stack.includes('cds')) {
                return 'CDS_SYNTAX_ERROR';
            }
            return 'PARSE_ERROR';
        }
        
        if (message.includes('memory') || message.includes('heap')) {
            return 'MEMORY_ERROR';
        }
        
        if (message.includes('enoent') || message.includes('permission') || message.includes('eacces')) {
            return 'FILE_ACCESS_ERROR';
        }
        
        if (message.includes('cache')) {
            return 'CACHE_ERROR';
        }
        
        if (message.includes('timeout') || message.includes('timed out')) {
            return 'TIMEOUT_ERROR';
        }
        
        if (message.includes('network') || message.includes('connect')) {
            return 'NETWORK_ERROR';
        }
        
        if (message.includes('config') || message.includes('configuration')) {
            return 'CONFIGURATION_ERROR';
        }
        
        if (message.includes('query') || message.includes('execution')) {
            return 'QUERY_ERROR';
        }
        
        if (message.includes('validation') || message.includes('invalid')) {
            return 'VALIDATION_ERROR';
        }
        
        return 'UNKNOWN_ERROR';
    }

    async attemptRecovery(error, errorType, context) {
        this.metrics.recoveryAttempts++;
        
        const strategy = this.recoveryStrategies.get(errorType);
        if (!strategy) {
            return { error, errorType, recovered: false, reason: 'no_strategy' };
        }
        
        try {
            // console.log(`🔄 Attempting recovery for ${errorType}...`);
            const recovery = await strategy(error, context);
            
            if (recovery.success) {
                this.metrics.successfulRecoveries++;
                // console.log(`✅ Recovery successful using method: ${recovery.method}`);
                
                // Log successful recovery
                await this.logRecovery(errorType, recovery, context);
                
                return {
                    error,
                    errorType,
                    recovered: true,
                    recovery,
                    result: recovery.result
                };
            } else {
                // console.log(`❌ Recovery failed for ${errorType}`);
                return { error, errorType, recovered: false, reason: recovery.method };
            }
            
        } catch (recoveryError) {
            console.error(`❌ Recovery attempt failed: ${recoveryError.message}`);
            return { error, errorType, recovered: false, reason: 'recovery_error', recoveryError };
        }
    }

    async logError(error, errorType, errorInfo, context) {
        if (!this.options.enableDetailedLogging) return;
        
        const timestamp = new Date().toISOString();
        const logEntry = {
            timestamp,
            errorType,
            severity: errorInfo ? errorInfo.severity : 'unknown',
            category: errorInfo ? errorInfo.category : 'unknown',
            message: error.message,
            stack: error.stack,
            context: {
                filePath: context.filePath,
                operation: context.operation ? context.operation.name : 'unknown',
                additionalInfo: context.additionalInfo
            },
            processInfo: {
                nodeVersion: process.version,
                platform: process.platform,
                memoryUsage: process.memoryUsage(),
                uptime: process.uptime()
            }
        };
        
        const logFile = path.join(this.logDir, `error-${timestamp.split('T')[0]}.log`);
        
        try {
            await fs.appendFile(logFile, `${JSON.stringify(logEntry)  }\n`);
        } catch (logError) {
            console.error(`Failed to write error log: ${logError.message}`);
        }
    }

    async logRecovery(errorType, recovery, context) {
        if (!this.options.enableDetailedLogging) return;
        
        const timestamp = new Date().toISOString();
        const logEntry = {
            timestamp,
            type: 'recovery',
            errorType,
            method: recovery.method,
            success: recovery.success,
            context: {
                filePath: context.filePath,
                operation: context.operation ? context.operation.name : 'unknown'
            }
        };
        
        const logFile = path.join(this.logDir, `recovery-${timestamp.split('T')[0]}.log`);
        
        try {
            await fs.appendFile(logFile, `${JSON.stringify(logEntry)  }\n`);
        } catch (logError) {
            console.error(`Failed to write recovery log: ${logError.message}`);
        }
    }

    // Helper methods
    sanitizeContent(content) {
        // Remove problematic characters and patterns
        return content
            .replace(/[^\x20-\x7E\r\n\t]/g, '') // Remove non-printable chars
            .replace(/\/\*[\s\S]*?\*\//g, '') // Remove block comments
            .replace(/\/\/.*$/gm, ''); // Remove line comments
    }

    generateAlternativePaths(filePath) {
        const dir = path.dirname(filePath);
        const basename = path.basename(filePath, path.extname(filePath));
        const ext = path.extname(filePath);
        
        return [
            path.join(dir, `${basename}.backup${ext}`),
            path.join(dir, `${basename}_backup${ext}`),
            path.join(dir, `${basename}.bak${ext}`),
            filePath.replace(/\\/g, '/'), // Windows to Unix path
            filePath.replace(/\//g, '\\')  // Unix to Windows path
        ];
    }

    splitIntoChunks(content, chunkSize) {
        const chunks = [];
        for (let i = 0; i < content.length; i += chunkSize) {
            chunks.push(content.slice(i, i + chunkSize));
        }
        return chunks;
    }

    mergeResults(results) {
        // Simple merge strategy - combine arrays and objects
        if (!results || results.length === 0) return {};
        
        const merged = {
            scip: { symbols: [], occurrences: [] },
            glean: []
        };
        
        results.forEach(result => {
            if (result.scip) {
                if (result.scip.symbols) merged.scip.symbols.push(...result.scip.symbols);
                if (result.scip.occurrences) merged.scip.occurrences.push(...result.scip.occurrences);
            }
            if (result.glean) {
                merged.glean.push(...result.glean);
            }
        });
        
        return merged;
    }

    async withTimeout(operation, timeout) {
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                reject(new Error(`Operation timed out after ${timeout}ms`));
            }, timeout);
            
            Promise.resolve(operation()).then(
                result => {
                    clearTimeout(timer);
                    resolve(result);
                },
                error => {
                    clearTimeout(timer);
                    reject(error);
                }
            );
        });
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async cleanupOldLogs() {
        try {
            const files = await fs.readdir(this.logDir);
            const logFiles = files.filter(f => f.endsWith('.log')).sort();
            
            if (logFiles.length > this.options.maxLogFiles) {
                const filesToDelete = logFiles.slice(0, logFiles.length - this.options.maxLogFiles);
                
                for (const file of filesToDelete) {
                    await fs.unlink(path.join(this.logDir, file));
                }
                
                // console.log(`🧹 Cleaned up ${filesToDelete.length} old log files`);
            }
        } catch (error) {
            console.warn(`⚠️ Log cleanup failed: ${error.message}`);
        }
    }

    getMetrics() {
        const successRate = this.metrics.recoveryAttempts > 0 
            ? (this.metrics.successfulRecoveries / this.metrics.recoveryAttempts * 100).toFixed(2)
            : 0;
        
        return {
            ...this.metrics,
            recoverySuccessRate: `${successRate}%`,
            errorRate: this.metrics.totalErrors > 0 
                ? `${(this.metrics.criticalErrors / this.metrics.totalErrors * 100).toFixed(2)  }%`
                : '0%'
        };
    }

    /**
     * Generate error report for debugging
     */
    generateErrorReport() {
        const metrics = this.getMetrics();
        const topErrors = Object.entries(metrics.errorsByType)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 10);
        
        const topProblematicFiles = Object.entries(metrics.errorsByFile)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 10);
        
        return {
            summary: {
                totalErrors: metrics.totalErrors,
                criticalErrors: metrics.criticalErrors,
                recoverySuccessRate: metrics.recoverySuccessRate,
                errorRate: metrics.errorRate
            },
            topErrorTypes: topErrors,
            topProblematicFiles: topProblematicFiles,
            recommendations: this.generateRecommendations(metrics, topErrors, topProblematicFiles)
        };
    }

    generateRecommendations(metrics, topErrors, topProblematicFiles) {
        const recommendations = [];
        
        if (metrics.criticalErrors > 0) {
            recommendations.push({
                type: 'critical',
                message: 'Critical errors detected. Review system resources and configuration.',
                action: 'Check memory usage and system limits'
            });
        }
        
        if (topErrors.length > 0 && topErrors[0][1] > 10) {
            recommendations.push({
                type: 'error_pattern',
                message: `High frequency of ${topErrors[0][0]} errors detected.`,
                action: 'Review error handling for this specific error type'
            });
        }
        
        if (topProblematicFiles.length > 0) {
            recommendations.push({
                type: 'file_issue',
                message: `File ${topProblematicFiles[0][0]} has recurring issues.`,
                action: 'Investigate file content and structure'
            });
        }
        
        if (parseFloat(metrics.recoverySuccessRate) < 50) {
            recommendations.push({
                type: 'recovery',
                message: 'Low recovery success rate detected.',
                action: 'Review and improve recovery strategies'
            });
        }
        
        return recommendations;
    }
}

module.exports = EnterpriseErrorHandler;