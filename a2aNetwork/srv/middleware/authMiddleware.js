/**
 * Simple authentication middleware for development
 * In production, this should be replaced with proper authentication
 */

const logger = require('../utils/logger');

const authMiddleware = (req, res, next) => {
    // For development, we'll accept any request with an address header
    // In production, this should verify JWT tokens or similar
    
    const userAddress = req.headers['x-user-address'] || req.headers['authorization'];
    
    if (!userAddress) {
        // For development, generate a mock address
        if (process.env.NODE_ENV === 'development') {
            req.user = {
                address: '0x1234567890123456789012345678901234567890',
                authenticated: false
            };
            return next();
        }
        
        return res.status(401).json({ 
            error: 'Authentication required',
            message: 'Please provide user address in x-user-address header' 
        });
    }
    
    // Basic address validation
    if (!/^0x[a-fA-F0-9]{40}$/.test(userAddress)) {
        return res.status(400).json({ 
            error: 'Invalid address format',
            message: 'Address must be a valid Ethereum address' 
        });
    }
    
    // Set user in request
    req.user = {
        address: userAddress,
        authenticated: true
    };
    
    logger.debug(`Authenticated request from ${userAddress}`);
    next();
};

module.exports = authMiddleware;