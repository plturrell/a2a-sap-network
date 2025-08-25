/**
 * @fileoverview Blockchain Authentication Middleware
 * @description Security middleware for blockchain operations
 * @module blockchain-auth-middleware
 */

const cds = require('@sap/cds');
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


// Rate limiting storage (in production, use Redis)
const rateLimitStore = new Map();
const RATE_LIMIT_WINDOW = 60000; // 1 minute
const MAX_OPERATIONS_PER_MINUTE = 10;

/**
 * Clean up expired rate limit entries
 */
function cleanupRateLimits() {
    const now = Date.now();
    for (const [key, data] of rateLimitStore.entries()) {
        if (now - data.windowStart > RATE_LIMIT_WINDOW) {
            rateLimitStore.delete(key);
        }
    }
}

// Run cleanup every minute
activeIntervals.set('interval_46', setInterval(cleanupRateLimits, 60000));

/**
 * Check and update rate limit
 */
async function checkRateLimit(userId, operation) {
    const key = `${userId}:${operation}`;
    const now = Date.now();

    let rateLimitData = rateLimitStore.get(key);

    if (!rateLimitData || now - rateLimitData.windowStart > RATE_LIMIT_WINDOW) {
        rateLimitData = {
            count: 0,
            windowStart: now
        };
    }

    rateLimitData.count++;
    rateLimitStore.set(key, rateLimitData);

    if (rateLimitData.count > MAX_OPERATIONS_PER_MINUTE) {
        throw new Error(`Rate limit exceeded. Maximum ${MAX_OPERATIONS_PER_MINUTE} operations per minute.`);
    }

    return rateLimitData.count;
}

/**
 * Get user permissions from database or cache
 */
async function getUserPermissions(userId) {
    try {
        // In production, this should query from database or permission service
        // For now, we'll implement a basic permission check

        if (!userId) {
            return [];
        }

        // Check if user has blockchain permissions
        // This is a placeholder - implement actual permission logic
        const { Users } = cds.entities;
        const user = await SELECT.one.from(Users).where({ ID: userId });

        if (!user) {
            return [];
        }

        // Return permissions based on user role
        if (user.role === 'admin' || user.role === 'blockchain_operator') {
            return ['blockchain.read', 'blockchain.write', 'blockchain.admin'];
        } else if (user.role === 'agent') {
            return ['blockchain.read', 'blockchain.write'];
        } else {
            return ['blockchain.read'];
        }
    } catch (error) {
        cds.log('blockchain-auth').error('Failed to get user permissions:', error);
        return [];
    }
}

/**
 * Validate request signature for high-value operations
 */
function validateRequestSignature(req) {
    const signature = req.headers['x-blockchain-signature'];
    const timestamp = req.headers['x-blockchain-timestamp'];

    if (!signature || !timestamp) {
        return false;
    }

    // Check timestamp is within 5 minutes
    const requestTime = parseInt(timestamp);
    const now = Date.now();
    if (Math.abs(now - requestTime) > 300000) { // 5 minutes
        return false;
    }

    // Verify signature
    const payload = JSON.stringify({
        data: req.data,
        timestamp: timestamp,
        user: req.user.id
    });

    const expectedSignature = crypto
        .createHmac('sha256', process.env.BLOCKCHAIN_SIGNING_KEY || 'development-key')
        .update(payload)
        .digest('hex');

    return signature === expectedSignature;
}

/**
 * Main authentication middleware for blockchain operations
 */
async function authenticateBlockchainOperation(req) {
    try {
        // Check if user is authenticated
        if (!req.user || !req.user.id) {
            req.error(401, 'Authentication required for blockchain operations');
            return;
        }

        // Get operation type
        const operation = req.event;
        const isWriteOperation = [
            'registerAgent',
            'updateAgentReputation',
            'deactivateAgent',
            'listService',
            'createServiceOrder',
            'completeServiceOrder',
            'sendMessage',
            'deployWorkflow',
            'executeWorkflow',
            'endorsePeer'
        ].includes(operation);

        // Check permissions
        const permissions = await getUserPermissions(req.user.id);
        const requiredPermission = isWriteOperation ? 'blockchain.write' : 'blockchain.read';

        if (!permissions.includes(requiredPermission)) {
            req.error(403, `Insufficient permissions. Required: ${requiredPermission}`);
            return;
        }

        // Rate limiting for write operations
        if (isWriteOperation) {
            try {
                await checkRateLimit(req.user.id, operation);
            } catch (rateLimitError) {
                req.error(429, rateLimitError.message);
                return;
            }
        }

        // For high-value operations, require signature
        const highValueOperations = [
            'updateAgentReputation',
            'completeServiceOrder',
            'endorsePeer',
            'deployWorkflow'
        ];

        if (highValueOperations.includes(operation)) {
            if (!validateRequestSignature(req)) {
                req.error(403, 'Invalid or missing request signature for high-value operation');
                return;
            }
        }

        // Log blockchain operation for audit trail
        cds.log('blockchain-audit').info('Blockchain operation', {
            user: req.user.id,
            operation: operation,
            data: req.data,
            timestamp: new Date().toISOString()
        });

        // Add security headers to context
        req.context = req.context || {};
        req.context.blockchainAuth = {
            authenticated: true,
            userId: req.user.id,
            permissions: permissions,
            timestamp: Date.now()
        };

    } catch (error) {
        cds.log('blockchain-auth').error('Authentication error:', error);
        req.error(500, 'Authentication failed');
    }
}

/**
 * Middleware for validating blockchain addresses
 */
function validateBlockchainAddress(req) {
    const addressFields = ['address', 'agentAddress', 'from', 'to', 'provider', 'consumer'];

    for (const field of addressFields) {
        if (req.data[field]) {
            if (!isValidEthereumAddress(req.data[field])) {
                req.error(400, `Invalid Ethereum address in field: ${field}`);
                return;
            }
        }
    }
}

/**
 * Validate Ethereum address format
 */
function isValidEthereumAddress(address) {
    if (!address || typeof address !== 'string') {
        return false;
    }
    return /^0x[a-fA-F0-9]{40}$/.test(address);
}

module.exports = {
    authenticateBlockchainOperation,
    validateBlockchainAddress,
    checkRateLimit,
    getUserPermissions,
    validateRequestSignature
};