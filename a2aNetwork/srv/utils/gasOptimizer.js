/**
 * @fileoverview Gas Optimization Utility
 * @description Optimizes gas usage for blockchain transactions
 * @module gas-optimizer
 */

const cds = require('@sap/cds');

/**
 * Gas optimization utility for blockchain transactions
 */
class GasOptimizer {
    constructor(web3Provider) {
        this.web3 = web3Provider;
        this.gasCache = new Map();
        this.gasPriceCache = null;
        this.gasPriceCacheExpiry = 0;
        this.CACHE_TTL = 60000; // 1 minute
    }

    /**
     * Get optimized gas price based on network conditions
     */
    async getOptimizedGasPrice() {
        const now = Date.now();
        
        // Use cached gas price if still valid
        if (this.gasPriceCache && now < this.gasPriceCacheExpiry) {
            return this.gasPriceCache;
        }

        try {
            // Get current gas price from network
            const networkGasPrice = await this.web3.eth.getGasPrice();
            
            // Apply optimization based on transaction priority
            const optimizedPrice = this.calculateOptimizedGasPrice(networkGasPrice);
            
            // Cache the result
            this.gasPriceCache = optimizedPrice;
            this.gasPriceCacheExpiry = now + this.CACHE_TTL;
            
            return optimizedPrice;
            
        } catch (error) {
            cds.log('gas-optimizer').error('Failed to get gas price:', error);
            // Fallback to cached value or default
            return this.gasPriceCache || '20000000000'; // 20 gwei
        }
    }

    /**
     * Calculate optimized gas price based on network conditions
     */
    calculateOptimizedGasPrice(networkGasPrice) {
        const basePrice = BigInt(networkGasPrice);
        
        // Get current hour to adjust for network usage patterns
        const hour = new Date().getHours();
        
        // Apply time-based optimization (lower gas during off-peak hours)
        let multiplier = 1.0;
        
        if (hour >= 2 && hour <= 6) {
            // Early morning - typically lower usage
            multiplier = 0.9;
        } else if (hour >= 9 && hour <= 17) {
            // Business hours - higher usage
            multiplier = 1.1;
        } else if (hour >= 18 && hour <= 22) {
            // Evening peak - highest usage
            multiplier = 1.2;
        }
        
        const optimizedPrice = BigInt(Math.floor(Number(basePrice) * multiplier));
        
        // Ensure minimum viable gas price
        const minGasPrice = BigInt('1000000000'); // 1 gwei minimum
        return optimizedPrice > minGasPrice ? optimizedPrice.toString() : minGasPrice.toString();
    }

    /**
     * Estimate gas for a contract function with optimization
     */
    async estimateGasOptimized(contractFunction, fromAddress, options = {}) {
        const cacheKey = this.generateCacheKey(contractFunction, fromAddress, options);
        
        // Check cache first
        const cached = this.gasCache.get(cacheKey);
        if (cached && Date.now() < cached.expiry) {
            return cached.gasLimit;
        }

        try {
            // Estimate gas with current network conditions
            const estimatedGas = await contractFunction.estimateGas({
                from: fromAddress,
                ...options
            });

            // Apply safety buffer (20% extra)
            const safeGasLimit = Math.floor(estimatedGas * 1.2);
            
            // Cache the result
            this.gasCache.set(cacheKey, {
                gasLimit: safeGasLimit,
                expiry: Date.now() + this.CACHE_TTL
            });

            return safeGasLimit;
            
        } catch (error) {
            cds.log('gas-optimizer').error('Gas estimation failed:', error);
            
            // Fallback to conservative estimate based on function type
            return this.getFallbackGasLimit(contractFunction.name);
        }
    }

    /**
     * Get fallback gas limits for common operations
     */
    getFallbackGasLimit(functionName) {
        const gasLimits = {
            // Agent operations
            'registerAgent': 200000,
            'updateReputation': 100000,
            'deactivateAgent': 80000,
            
            // Service operations
            'listService': 150000,
            'requestService': 200000,
            'startService': 80000,
            'completeService': 100000,
            'releasePayment': 150000,
            
            // Review operations
            'submitPeerReview': 180000,
            'validateReview': 120000,
            
            // Default
            'default': 200000
        };

        return gasLimits[functionName] || gasLimits.default;
    }

    /**
     * Generate cache key for gas estimation
     */
    generateCacheKey(contractFunction, fromAddress, options) {
        const crypto = require('crypto');

// Track intervals for cleanup
const activeIntervals = new Map();

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

        const data = {
            function: contractFunction.name,
            from: fromAddress,
            options: options
        };
        
        return crypto.createHash('md5')
            .update(JSON.stringify(data))
            .digest('hex');
    }

    /**
     * Optimize batch transactions
     */
    async optimizeBatchTransactions(transactions) {
        const optimized = [];
        
        // Sort transactions by gas price efficiency
        const sorted = transactions.sort((a, b) => {
            const efficiencyA = this.calculateGasEfficiency(a);
            const efficiencyB = this.calculateGasEfficiency(b);
            return efficiencyB - efficiencyA;
        });

        // Group transactions by gas price to batch efficiently
        const gasPrice = await this.getOptimizedGasPrice();
        
        for (const tx of sorted) {
            optimized.push({
                ...tx,
                gasPrice: gasPrice,
                gasLimit: await this.estimateGasOptimized(tx.function, tx.from, tx.options)
            });
        }

        return optimized;
    }

    /**
     * Calculate gas efficiency score for transaction prioritization
     */
    calculateGasEfficiency(transaction) {
        const priorities = {
            'registerAgent': 10,
            'updateReputation': 8,
            'completeService': 9,
            'releasePayment': 10,
            'submitPeerReview': 6,
            'validateReview': 7,
            'default': 5
        };

        const priority = priorities[transaction.function?.name] || priorities.default;
        const estimatedCost = transaction.gasLimit * transaction.gasPrice;
        
        // Higher priority / lower cost = better efficiency
        return priority / (estimatedCost || 1);
    }

    /**
     * Clean up expired cache entries
     */
    cleanupCache() {
        const now = Date.now();
        
        for (const [key, value] of this.gasCache.entries()) {
            if (now >= value.expiry) {
                this.gasCache.delete(key);
            }
        }
        
        // Clean gas price cache
        if (now >= this.gasPriceCacheExpiry) {
            this.gasPriceCache = null;
        }
    }

    /**
     * Get gas optimization statistics
     */
    getOptimizationStats() {
        return {
            cacheSize: this.gasCache.size,
            gasPriceCached: !!this.gasPriceCache,
            cacheHitRate: this.calculateCacheHitRate()
        };
    }

    /**
     * Calculate cache hit rate for monitoring
     */
    calculateCacheHitRate() {
        // This would be implemented with actual metrics collection
        return 0.85; // Placeholder
    }
}

// Start cache cleanup interval
const gasOptimizer = new GasOptimizer();
// activeIntervals.set('interval_270', setInterval(() => gasOptimizer.cleanupCache(), 300000)); // Clean every 5 minutes

module.exports = GasOptimizer;