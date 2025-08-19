/**
 * Blockchain Transaction Retry Utility
 * Implements exponential backoff and retry logic for blockchain operations
 */

const cds = require('@sap/cds');
const log = cds.log('blockchain-retry');

class BlockchainRetry {
    constructor(options = {}) {
        this.maxRetries = options.maxRetries || 3;
        this.initialDelay = options.initialDelay || 1000; // 1 second
        this.maxDelay = options.maxDelay || 30000; // 30 seconds
        this.backoffMultiplier = options.backoffMultiplier || 2;
        this.retryableErrors = [
            'ETIMEDOUT',
            'ECONNREFUSED',
            'ECONNRESET',
            'EPIPE',
            'ENOTFOUND',
            'ENETUNREACH',
            'EAI_AGAIN',
            'replacement fee too low',
            'nonce too low',
            'insufficient funds',
            'transaction underpriced'
        ];
    }

    /**
     * Execute blockchain operation with retry logic
     * @param {Function} operation - Async function to execute
     * @param {Object} context - Context for logging
     * @returns {Promise<any>} Result of the operation
     */
    async executeWithRetry(operation, context = {}) {
        let lastError;
        
        for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
            try {
                log.debug('Executing blockchain operation', { 
                    attempt, 
                    maxRetries: this.maxRetries,
                    ...context 
                });
                
                const result = await operation();
                
                if (attempt > 1) {
                    log.info('Blockchain operation succeeded after retry', { 
                        attempt,
                        ...context 
                    });
                }
                
                return result;
            } catch (error) {
                lastError = error;
                
                if (!this.isRetryableError(error) || attempt === this.maxRetries) {
                    log.error('Blockchain operation failed', { 
                        attempt,
                        error: error.message,
                        code: error.code,
                        ...context 
                    });
                    throw error;
                }
                
                const delay = this.calculateDelay(attempt);
                log.warn('Retrying blockchain operation', { 
                    attempt,
                    nextAttemptIn: delay,
                    error: error.message,
                    ...context 
                });
                
                await this.sleep(delay);
                
                // For nonce-related errors, increment nonce
                if (this.isNonceError(error)) {
                    context.incrementNonce = true;
                }
            }
        }
        
        throw lastError;
    }

    /**
     * Check if error is retryable
     * @param {Error} error - Error to check
     * @returns {boolean} True if retryable
     */
    isRetryableError(error) {
        const errorMessage = error.message?.toLowerCase() || '';
        const errorCode = error.code || '';
        
        return this.retryableErrors.some(retryable => {
            const retryableLower = retryable.toLowerCase();
            return errorMessage.includes(retryableLower) || 
                   errorCode === retryable ||
                   (error.reason && error.reason.toLowerCase().includes(retryableLower));
        });
    }

    /**
     * Check if error is nonce-related
     * @param {Error} error - Error to check
     * @returns {boolean} True if nonce error
     */
    isNonceError(error) {
        const errorMessage = error.message?.toLowerCase() || '';
        return errorMessage.includes('nonce too low') || 
               errorMessage.includes('replacement fee too low');
    }

    /**
     * Calculate delay for next retry with exponential backoff
     * @param {number} attempt - Current attempt number
     * @returns {number} Delay in milliseconds
     */
    calculateDelay(attempt) {
        const exponentialDelay = this.initialDelay * Math.pow(this.backoffMultiplier, attempt - 1);
        // Use deterministic jitter based on attempt number for consistent retry patterns
        const jitterFactor = 0.5 + (0.5 * ((attempt * 13) % 100) / 100); // Deterministic jitter between 0.5-1.0
        const jitteredDelay = exponentialDelay * jitterFactor;
        return Math.min(jitteredDelay, this.maxDelay);
    }

    /**
     * Sleep for specified milliseconds
     * @param {number} ms - Milliseconds to sleep
     * @returns {Promise<void>}
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Wrap a blockchain method with retry logic
     * @param {Function} method - Method to wrap
     * @param {string} methodName - Name for logging
     * @returns {Function} Wrapped method
     */
    wrapMethod(method, methodName) {
        return async (...args) => {
            return this.executeWithRetry(
                () => method.apply(this, args),
                { method: methodName, args: args.length }
            );
        };
    }
}

// Export singleton instance
module.exports = new BlockchainRetry();