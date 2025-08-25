/**
 * @fileoverview Blockchain Transaction Security Validator
 * @description Comprehensive security validation for blockchain transactions including
 * signature verification, transaction integrity checks, and smart contract security
 * @module BlockchainSecurityValidator
 * @since 1.0.0
 * @author A2A Network Security Team
 */

const crypto = require('crypto');
const { ethers } = require('ethers');
const cds = require('@sap/cds');

/**
 * Blockchain Security Configuration
 */
const BLOCKCHAIN_SECURITY_CONFIG = {
    // Transaction validation rules
    validation: {
        enabled: true,
        strictMode: true,
        maxGasLimit: 5000000,
        maxGasPrice: '100000000000', // 100 Gwei
        minConfirmations: 3,
        transactionTimeout: 300000, // 5 minutes
        maxTransactionValue: '10', // 10 ETH
        whitelistedContracts: []
    },

    // Signature verification
    signatures: {
        requireSignature: true,
        verifyChain: true,
        allowedSignatureTypes: ['secp256k1', 'ed25519'],
        messagePrefix: '\x19Ethereum Signed Message:\n'
    },

    // Smart contract security
    contracts: {
        verifyBytecode: true,
        checkContractVerification: true,
        validateABI: true,
        preventUnknownContracts: true,
        allowedContractPatterns: [
            'Agent*',
            'Service*',
            'Reputation*',
            'Capability*'
        ]
    },

    // Anti-fraud measures
    fraud: {
        enabled: true,
        detectDuplicateTransactions: true,
        checkTransactionOrigin: true,
        validateTimestamps: true,
        maxTransactionsPerMinute: 100,
        suspiciousGasLimits: [21000, 22000] // Common scam gas limits
    },

    // Monitoring and alerting
    monitoring: {
        logAllTransactions: true,
        alertOnSuspicious: true,
        trackFailedValidations: true,
        reportMetrics: true
    }
};

/**
 * Blockchain Security Validator
 * Main class for validating blockchain transactions and smart contract interactions
 */
class BlockchainSecurityValidator {
    constructor() {
        this.log = cds.log('blockchain-security');
        this.transactionCache = new Map();
        this.validationMetrics = {
            totalValidations: 0,
            successfulValidations: 0,
            failedValidations: 0,
            suspiciousTransactions: 0,
            blockedTransactions: 0
        };

        this.suspiciousAddresses = new Set();
        this.contractVerifications = new Map();

        this._initializeValidator();
    }

    /**
     * Initialize validator with security checks
     */
    _initializeValidator() {
        // Load whitelisted contracts from environment
        const whitelistedContracts = process.env.WHITELISTED_CONTRACTS?.split(',') || [];
        BLOCKCHAIN_SECURITY_CONFIG.validation.whitelistedContracts = whitelistedContracts;

        this.log.info('Blockchain security validator initialized', {
            strictMode: BLOCKCHAIN_SECURITY_CONFIG.validation.strictMode,
            whitelistedContracts: whitelistedContracts.length
        });

        // Start monitoring
        this._startSecurityMonitoring();
    }

    /**
     * Validate blockchain transaction before execution
     */
    async validateTransaction(transaction, context = {}) {
        const validationId = this._generateValidationId();
        const startTime = Date.now();

        try {
            this.validationMetrics.totalValidations++;

            this.log.info('Starting transaction validation', {
                validationId,
                transactionHash: transaction.hash,
                to: transaction.to,
                value: transaction.value
            });

            // 1. Basic transaction structure validation
            await this._validateTransactionStructure(transaction);

            // 2. Signature verification
            await this._validateSignature(transaction, context);

            // 3. Gas and value limits
            await this._validateGasAndValue(transaction);

            // 4. Contract security validation
            if (transaction.to && await this._isContract(transaction.to)) {
                await this._validateContractInteraction(transaction);
            }

            // 5. Fraud detection
            await this._detectFraudPatterns(transaction, context);

            // 6. Business logic validation
            await this._validateBusinessRules(transaction, context);

            // 7. Rate limiting check
            await this._checkRateLimits(transaction, context);

            const validationTime = Date.now() - startTime;
            this.validationMetrics.successfulValidations++;

            this.log.info('Transaction validation successful', {
                validationId,
                transactionHash: transaction.hash,
                validationTime: `${validationTime}ms`
            });

            return {
                valid: true,
                validationId,
                validationTime,
                warnings: []
            };

        } catch (error) {
            this.validationMetrics.failedValidations++;

            // Determine if this is suspicious activity
            if (this._isSuspiciousError(error)) {
                this.validationMetrics.suspiciousTransactions++;
                await this._handleSuspiciousActivity(transaction, error, context);
            }

            this.log.error('Transaction validation failed', {
                validationId,
                transactionHash: transaction.hash,
                error: error.message,
                errorCode: error.code
            });

            throw new BlockchainSecurityError(
                'TRANSACTION_VALIDATION_FAILED',
                `Transaction validation failed: ${error.message}`,
                { validationId, originalError: error }
            );
        }
    }

    /**
     * Validate transaction structure and required fields
     */
    async _validateTransactionStructure(transaction) {
        if (!transaction || typeof transaction !== 'object') {
            throw new BlockchainSecurityError('INVALID_TRANSACTION_STRUCTURE', 'Transaction must be a valid object');
        }

        // Required fields validation
        const requiredFields = ['to', 'value', 'gasLimit', 'gasPrice', 'nonce'];
        for (const field of requiredFields) {
            if (transaction[field] === undefined || transaction[field] === null) {
                throw new BlockchainSecurityError('MISSING_REQUIRED_FIELD', `Transaction missing required field: ${field}`);
            }
        }

        // Validate addresses format
        if (transaction.to && !ethers.isAddress(transaction.to)) {
            throw new BlockchainSecurityError('INVALID_TO_ADDRESS', 'Invalid recipient address format');
        }

        if (transaction.from && !ethers.isAddress(transaction.from)) {
            throw new BlockchainSecurityError('INVALID_FROM_ADDRESS', 'Invalid sender address format');
        }

        // Validate numeric fields
        const numericFields = ['value', 'gasLimit', 'gasPrice', 'nonce'];
        for (const field of numericFields) {
            if (transaction[field] && isNaN(transaction[field])) {
                throw new BlockchainSecurityError('INVALID_NUMERIC_FIELD', `Field ${field} must be numeric`);
            }
        }

        // Check for zero address (common attack vector)
        if (transaction.to === '0x0000000000000000000000000000000000000000') {
            throw new BlockchainSecurityError('ZERO_ADDRESS_TRANSACTION', 'Cannot send transaction to zero address');
        }
    }

    /**
     * Validate transaction signature
     */
    async _validateSignature(transaction, context) {
        if (!BLOCKCHAIN_SECURITY_CONFIG.signatures.requireSignature) {
            return;
        }

        if (!transaction.signature && !transaction.r && !transaction.s && !transaction.v) {
            throw new BlockchainSecurityError('MISSING_SIGNATURE', 'Transaction signature is required');
        }

        try {
            let recoveredAddress;

            // Recover address from signature
            if (transaction.signature) {
                // Extract r, s, v from signature
                const signature = transaction.signature.startsWith('0x')
                    ? transaction.signature.slice(2)
                    : transaction.signature;

                if (signature.length !== 130) {
                    throw new BlockchainSecurityError('INVALID_SIGNATURE_LENGTH', 'Signature must be 130 characters (65 bytes)');
                }

                const r = `0x${  signature.slice(0, 64)}`;
                const s = `0x${  signature.slice(64, 128)}`;
                const v = parseInt(signature.slice(128, 130), 16);

                // Create transaction hash for signature verification
                const transactionHash = this._createTransactionHash(transaction);
                const messageHash = ethers.hashMessage(transactionHash);

                recoveredAddress = ethers.recoverAddress(messageHash, { r, s, v });

            } else if (transaction.r && transaction.s && transaction.v) {
                // Use provided r, s, v values
                const transactionHash = this._createTransactionHash(transaction);
                const messageHash = ethers.hashMessage(transactionHash);

                recoveredAddress = ethers.recoverAddress(messageHash, {
                    r: transaction.r,
                    s: transaction.s,
                    v: transaction.v
                });
            }

            // Verify recovered address matches sender
            if (recoveredAddress && transaction.from) {
                if (recoveredAddress.toLowerCase() !== transaction.from.toLowerCase()) {
                    throw new BlockchainSecurityError('SIGNATURE_MISMATCH', 'Transaction signature does not match sender address');
                }
            }

            // Additional signature security checks
            await this._validateSignatureSecurity(transaction, recoveredAddress);

        } catch (error) {
            if (error instanceof BlockchainSecurityError) {
                throw error;
            }

            throw new BlockchainSecurityError('SIGNATURE_VERIFICATION_FAILED',
                `Failed to verify transaction signature: ${error.message}`);
        }
    }

    /**
     * Validate gas limits and transaction value
     */
    async _validateGasAndValue(transaction) {
        // Gas limit validation
        const gasLimit = BigInt(transaction.gasLimit || 0);
        const maxGasLimit = BigInt(BLOCKCHAIN_SECURITY_CONFIG.validation.maxGasLimit);

        if (gasLimit > maxGasLimit) {
            throw new BlockchainSecurityError('EXCESSIVE_GAS_LIMIT',
                `Gas limit ${gasLimit} exceeds maximum allowed ${maxGasLimit}`);
        }

        // Gas price validation
        const gasPrice = BigInt(transaction.gasPrice || 0);
        const maxGasPrice = BigInt(BLOCKCHAIN_SECURITY_CONFIG.validation.maxGasPrice);

        if (gasPrice > maxGasPrice) {
            throw new BlockchainSecurityError('EXCESSIVE_GAS_PRICE',
                `Gas price ${gasPrice} exceeds maximum allowed ${maxGasPrice}`);
        }

        // Transaction value validation
        if (transaction.value) {
            const value = ethers.parseEther(transaction.value.toString());
            const maxValue = ethers.parseEther(BLOCKCHAIN_SECURITY_CONFIG.validation.maxTransactionValue);

            if (value > maxValue) {
                throw new BlockchainSecurityError('EXCESSIVE_TRANSACTION_VALUE',
                    `Transaction value ${transaction.value} ETH exceeds maximum allowed ${BLOCKCHAIN_SECURITY_CONFIG.validation.maxTransactionValue} ETH`);
            }
        }

        // Check for suspicious gas limit patterns
        const gasLimitNum = Number(gasLimit);
        if (BLOCKCHAIN_SECURITY_CONFIG.fraud.suspiciousGasLimits.includes(gasLimitNum)) {
            throw new BlockchainSecurityError('SUSPICIOUS_GAS_LIMIT',
                `Gas limit ${gasLimitNum} matches known scam patterns`);
        }
    }

    /**
     * Validate smart contract interaction
     */
    async _validateContractInteraction(transaction) {
        const contractAddress = transaction.to;

        // Check if contract is whitelisted
        const isWhitelisted = BLOCKCHAIN_SECURITY_CONFIG.validation.whitelistedContracts.includes(contractAddress);

        if (BLOCKCHAIN_SECURITY_CONFIG.contracts.preventUnknownContracts && !isWhitelisted) {
            // Check if contract name matches allowed patterns
            const contractInfo = await this._getContractInfo(contractAddress);
            const isPatternAllowed = BLOCKCHAIN_SECURITY_CONFIG.contracts.allowedContractPatterns.some(pattern => {
                const regex = new RegExp(pattern.replace('*', '.*'), 'i');
                return regex.test(contractInfo.name || '');
            });

            if (!isPatternAllowed) {
                throw new BlockchainSecurityError('UNKNOWN_CONTRACT_INTERACTION',
                    `Interaction with unknown contract ${contractAddress} not allowed`);
            }
        }

        // Validate contract bytecode if required
        if (BLOCKCHAIN_SECURITY_CONFIG.contracts.verifyBytecode) {
            await this._validateContractBytecode(contractAddress);
        }

        // Validate function call data
        if (transaction.data) {
            await this._validateContractCallData(transaction.data, contractAddress);
        }
    }

    /**
     * Detect fraud patterns in transaction
     */
    async _detectFraudPatterns(transaction, context) {
        if (!BLOCKCHAIN_SECURITY_CONFIG.fraud.enabled) {
            return;
        }

        // Check for duplicate transactions
        if (BLOCKCHAIN_SECURITY_CONFIG.fraud.detectDuplicateTransactions) {
            const txKey = this._createTransactionKey(transaction);
            if (this.transactionCache.has(txKey)) {
                const previousTx = this.transactionCache.get(txKey);
                const timeDiff = Date.now() - previousTx.timestamp;

                // Allow duplicate if enough time has passed (5 minutes)
                if (timeDiff < 300000) {
                    throw new BlockchainSecurityError('DUPLICATE_TRANSACTION',
                        'Duplicate transaction detected within 5 minutes');
                }
            }

            // Cache transaction
            this.transactionCache.set(txKey, {
                timestamp: Date.now(),
                transaction: { ...transaction }
            });
        }

        // Validate transaction timestamp if available
        if (BLOCKCHAIN_SECURITY_CONFIG.fraud.validateTimestamps && transaction.timestamp) {
            const now = Date.now();
            const txTime = new Date(transaction.timestamp).getTime();
            const timeDiff = Math.abs(now - txTime);

            // Allow 10 minutes clock drift
            if (timeDiff > 600000) {
                throw new BlockchainSecurityError('INVALID_TIMESTAMP',
                    'Transaction timestamp too far from current time');
            }
        }

        // Check transaction origin
        if (BLOCKCHAIN_SECURITY_CONFIG.fraud.checkTransactionOrigin && context.origin) {
            if (this.suspiciousAddresses.has(transaction.from)) {
                throw new BlockchainSecurityError('SUSPICIOUS_SENDER',
                    'Transaction from flagged suspicious address');
            }
        }
    }

    /**
     * Validate business rules for A2A transactions
     */
    async _validateBusinessRules(transaction, context) {
        // Agent registration validations
        if (context.operationType === 'AGENT_REGISTRATION') {
            if (!transaction.data || transaction.data.length < 10) {
                throw new BlockchainSecurityError('INVALID_AGENT_DATA',
                    'Agent registration requires valid data payload');
            }
        }

        // Service marketplace validations
        if (context.operationType === 'SERVICE_LISTING') {
            if (!transaction.value || Number(transaction.value) <= 0) {
                throw new BlockchainSecurityError('INVALID_SERVICE_PRICE',
                    'Service listing requires valid price');
            }

            // Check service provider reputation
            if (context.providerReputation < 50) {
                throw new BlockchainSecurityError('LOW_PROVIDER_REPUTATION',
                    'Provider reputation too low for service listing');
            }
        }

        // Capability registration validations
        if (context.operationType === 'CAPABILITY_REGISTRATION') {
            if (!context.agentId || !ethers.isAddress(context.agentId)) {
                throw new BlockchainSecurityError('INVALID_AGENT_ID',
                    'Valid agent ID required for capability registration');
            }
        }
    }

    /**
     * Check rate limits for transactions
     */
    async _checkRateLimits(transaction, context) {
        const sender = transaction.from;
        const now = Date.now();
        const windowSize = 60000; // 1 minute

        // Get transaction history for sender
        const txHistory = this._getTransactionHistory(sender);

        // Count transactions in current window
        const recentTxs = txHistory.filter(tx => now - tx.timestamp < windowSize);

        if (recentTxs.length >= BLOCKCHAIN_SECURITY_CONFIG.fraud.maxTransactionsPerMinute) {
            throw new BlockchainSecurityError('RATE_LIMIT_EXCEEDED',
                `Transaction rate limit exceeded: ${recentTxs.length} transactions in the last minute`);
        }

        // Add current transaction to history
        txHistory.push({
            timestamp: now,
            hash: transaction.hash,
            to: transaction.to,
            value: transaction.value
        });

        // Keep only last 1000 transactions
        if (txHistory.length > 1000) {
            txHistory.splice(0, txHistory.length - 1000);
        }
    }

    /**
     * Validate signature security parameters
     */
    async _validateSignatureSecurity(transaction, recoveredAddress) {
        // Check for signature malleability
        if (transaction.s) {
            const s = BigInt(transaction.s);
            const secp256k1N = BigInt('0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141');
            const halfN = secp256k1N / 2n;

            if (s > halfN) {
                throw new BlockchainSecurityError('SIGNATURE_MALLEABILITY',
                    'Signature s value must be in low form to prevent malleability');
            }
        }

        // Additional signature checks could be added here
    }

    /**
     * Validate contract bytecode
     */
    async _validateContractBytecode(contractAddress) {
        // In a real implementation, this would check the contract bytecode
        // against known patterns or verify it against a source

        // For now, just check if we have cached verification
        if (this.contractVerifications.has(contractAddress)) {
            const verification = this.contractVerifications.get(contractAddress);
            if (!verification.verified) {
                throw new BlockchainSecurityError('UNVERIFIED_CONTRACT',
                    `Contract ${contractAddress} bytecode not verified`);
            }
        }
    }

    /**
     * Validate contract call data
     */
    async _validateContractCallData(data, contractAddress) {
        if (!data || data === '0x') {
            return; // No data to validate
        }

        // Validate data is properly formatted hex
        if (!data.startsWith('0x') || data.length % 2 !== 0) {
            throw new BlockchainSecurityError('INVALID_CALL_DATA',
                'Contract call data must be properly formatted hex');
        }

        // Extract function selector (first 4 bytes)
        const selector = data.slice(0, 10);

        // Check against known dangerous selectors
        const dangerousSelectors = [
            '0xa9059cbb', // transfer(address,uint256) - often used in scams
            '0x23b872dd', // transferFrom(address,address,uint256)
        ];

        if (dangerousSelectors.includes(selector)) {
            this.log.warn('Potentially dangerous function call detected', {
                contractAddress,
                selector,
                data: data.slice(0, 100) // First 50 bytes
            });
        }
    }

    /**
     * Check if address is a contract
     */
    async _isContract(address) {
        // In a real implementation, this would check if the address has contract bytecode
        // For now, return true for known contract addresses
        return BLOCKCHAIN_SECURITY_CONFIG.validation.whitelistedContracts.includes(address);
    }

    /**
     * Get contract information
     */
    async _getContractInfo(contractAddress) {
        // In a real implementation, this would fetch contract info from blockchain or database
        return {
            name: 'Unknown Contract',
            verified: false,
            createdAt: null
        };
    }

    /**
     * Create transaction hash for signature verification
     */
    _createTransactionHash(transaction) {
        const txData = {
            to: transaction.to,
            value: transaction.value,
            gasLimit: transaction.gasLimit,
            gasPrice: transaction.gasPrice,
            nonce: transaction.nonce,
            data: transaction.data || '0x'
        };

        return crypto.createHash('sha256')
            .update(JSON.stringify(txData))
            .digest('hex');
    }

    /**
     * Create unique transaction key for duplicate detection
     */
    _createTransactionKey(transaction) {
        return crypto.createHash('sha256')
            .update(`${transaction.from}:${transaction.to}:${transaction.value}:${transaction.nonce}`)
            .digest('hex');
    }

    /**
     * Get transaction history for address
     */
    _getTransactionHistory(address) {
        if (!this.transactionHistories) {
            this.transactionHistories = new Map();
        }

        if (!this.transactionHistories.has(address)) {
            this.transactionHistories.set(address, []);
        }

        return this.transactionHistories.get(address);
    }

    /**
     * Check if error indicates suspicious activity
     */
    _isSuspiciousError(error) {
        const suspiciousCodes = [
            'DUPLICATE_TRANSACTION',
            'SUSPICIOUS_GAS_LIMIT',
            'EXCESSIVE_TRANSACTION_VALUE',
            'SIGNATURE_MISMATCH',
            'UNKNOWN_CONTRACT_INTERACTION',
            'RATE_LIMIT_EXCEEDED'
        ];

        return suspiciousCodes.includes(error.code);
    }

    /**
     * Handle suspicious activity
     */
    async _handleSuspiciousActivity(transaction, error, context) {
        const suspiciousActivity = {
            timestamp: new Date().toISOString(),
            transactionHash: transaction.hash,
            fromAddress: transaction.from,
            toAddress: transaction.to,
            errorCode: error.code,
            errorMessage: error.message,
            severity: 'HIGH',
            blocked: true
        };

        // Add sender to suspicious addresses
        if (transaction.from) {
            this.suspiciousAddresses.add(transaction.from);
        }

        // Log security incident
        this.log.error('Suspicious blockchain activity detected', suspiciousActivity);

        // Send security alert
        await this._sendSecurityAlert(suspiciousActivity);

        this.validationMetrics.blockedTransactions++;
    }

    /**
     * Send security alert
     */
    async _sendSecurityAlert(activity) {
        try {
            // In a real implementation, this would integrate with your alerting system
            const alert = {
                type: 'BLOCKCHAIN_SECURITY_VIOLATION',
                severity: activity.severity,
                message: `Suspicious blockchain transaction blocked: ${activity.errorCode}`,
                details: activity,
                timestamp: new Date().toISOString()
            };

            // Emit to security monitoring service
            const securityService = await cds.connect.to('SecurityMonitoringService');
            await securityService.emit('securityAlert', alert);

        } catch (error) {
            this.log.error('Failed to send blockchain security alert:', error);
        }
    }

    /**
     * Generate validation ID
     */
    _generateValidationId() {
        return crypto.randomBytes(8).toString('hex');
    }

    /**
     * Start security monitoring
     */
    _startSecurityMonitoring() {
        // Clean up old transaction cache every 10 minutes
        setInterval(() => {
            const now = Date.now();
            const cleanupThreshold = 600000; // 10 minutes

            for (const [key, value] of this.transactionCache.entries()) {
                if (now - value.timestamp > cleanupThreshold) {
                    this.transactionCache.delete(key);
                }
            }
        }, 600000);

        // Report metrics every 5 minutes
        setInterval(() => {
            this._reportSecurityMetrics();
        }, 300000);
    }

    /**
     * Report security metrics
     */
    _reportSecurityMetrics() {
        const metrics = {
            ...this.validationMetrics,
            successRate: this.validationMetrics.totalValidations > 0
                ? `${(this.validationMetrics.successfulValidations / this.validationMetrics.totalValidations * 100).toFixed(2)  }%`
                : '0%',
            suspiciousAddressCount: this.suspiciousAddresses.size,
            cachedTransactions: this.transactionCache.size
        };

        this.log.info('Blockchain Security Metrics', metrics);
    }

    /**
     * Get security status
     */
    getSecurityStatus() {
        return {
            enabled: BLOCKCHAIN_SECURITY_CONFIG.validation.enabled,
            strictMode: BLOCKCHAIN_SECURITY_CONFIG.validation.strictMode,
            metrics: this.validationMetrics,
            suspiciousAddresses: this.suspiciousAddresses.size,
            whitelistedContracts: BLOCKCHAIN_SECURITY_CONFIG.validation.whitelistedContracts.length,
            lastUpdate: new Date().toISOString()
        };
    }

    /**
     * Add address to suspicious list
     */
    flagAddressAsSuspicious(address, reason) {
        this.suspiciousAddresses.add(address);

        this.log.warn('Address flagged as suspicious', {
            address,
            reason,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Remove address from suspicious list
     */
    clearSuspiciousFlag(address) {
        this.suspiciousAddresses.delete(address);

        this.log.info('Suspicious flag cleared for address', {
            address,
            timestamp: new Date().toISOString()
        });
    }
}

/**
 * Blockchain Security Error Class
 */
class BlockchainSecurityError extends Error {
    constructor(code, message, details = {}) {
        super(message);
        this.name = 'BlockchainSecurityError';
        this.code = code;
        this.details = details;
        this.timestamp = new Date().toISOString();
    }
}

/**
 * Transaction Security Middleware
 * Express/CDS middleware for validating blockchain transactions
 */
function createBlockchainSecurityMiddleware() {
    const validator = new BlockchainSecurityValidator();

    return async (req, res, next) => {
        try {
            // Only validate blockchain-related requests
            if (!req.path.includes('/blockchain') && !req.body?.transaction) {
                return next();
            }

            const transaction = req.body.transaction;
            if (!transaction) {
                return next();
            }

            // Validate transaction
            const result = await validator.validateTransaction(transaction, {
                userAgent: req.get('User-Agent'),
                ip: req.ip,
                origin: req.get('Origin'),
                operationType: req.body.operationType,
                agentId: req.body.agentId,
                providerReputation: req.body.providerReputation
            });

            // Add validation result to request
            req.blockchainValidation = result;

            next();

        } catch (error) {
            if (error instanceof BlockchainSecurityError) {
                res.status(400).json({
                    error: 'BLOCKCHAIN_SECURITY_ERROR',
                    code: error.code,
                    message: error.message,
                    timestamp: error.timestamp
                });
            } else {
                res.status(500).json({
                    error: 'VALIDATION_ERROR',
                    message: 'Failed to validate blockchain transaction'
                });
            }
        }
    };
}

// Initialize global validator
const blockchainSecurityValidator = new BlockchainSecurityValidator();

module.exports = {
    BlockchainSecurityValidator,
    BlockchainSecurityError,
    blockchainSecurityValidator,
    createBlockchainSecurityMiddleware,
    BLOCKCHAIN_SECURITY_CONFIG
};