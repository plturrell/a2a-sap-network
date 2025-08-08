const express = require('express');
const router = express.Router();

/**
 * User API for BTP Integration
 * Provides user information and authentication tokens for the frontend
 */

// Get current user information
router.get('/currentUser', async (req, res) => {
    if (!req.user) {
        return res.status(401).json({ error: 'Authentication required' });
    }

    // Extract user information from request context
    const user = {
        id: req.user.id,
        name: req.user.givenName && req.user.familyName 
            ? `${req.user.givenName} ${req.user.familyName}`
            : req.user.id,
        email: req.user.email,
        roles: req.user.sapRoles || req.user.roles || [],
        scopes: req.user.scopes || [],
        tenant: req.user.tenant,
        zoneId: req.user.zoneId
    };

    res.json(user);
});

// Get authentication token for API calls
router.get('/token', async (req, res) => {
    if (!req.user) {
        return res.status(401).json({ error: 'Authentication required' });
    }

    // In BTP, extract token from headers or security context
    const authHeader = req.headers.authorization;
    
    if (authHeader && authHeader.startsWith('Bearer ')) {
        return res.json({
            token: authHeader.substring(7),
            type: 'Bearer',
            expiresIn: 3600 // Default 1 hour, should be extracted from actual token
        });
    }

    res.status(400).json({ error: 'No authentication token available' });
});

// Get user permissions for specific resources
router.get('/permissions', async (req, res) => {
    if (!req.user) {
        return res.status(401).json({ error: 'Authentication required' });
    }

    const userRoles = req.user.sapRoles || req.user.roles || [];
    const userScopes = req.user.scopes || [];

    // Helper method to check permissions
    const hasPermission = (requiredRoles, requiredScopes) => {
        // Check roles
        const hasRole = requiredRoles.some(role => userRoles.includes(role));
        
        // Check scopes
        const hasScope = requiredScopes.some(scope => userScopes.includes(scope));
        
        // User needs either the role OR the scope
        return hasRole || hasScope;
    };

    // Define permission matrix
    const permissions = {
        // Agent management
        'agent:create': hasPermission(['Admin', 'Developer'], ['agent.write']),
        'agent:read': hasPermission(['Admin', 'Developer', 'User'], ['agent.read']),
        'agent:update': hasPermission(['Admin', 'Developer'], ['agent.write']),
        'agent:delete': hasPermission(['Admin'], ['agent.delete']),

        // Service management
        'service:create': hasPermission(['Admin', 'Developer'], ['service.write']),
        'service:read': hasPermission(['Admin', 'Developer', 'User'], ['service.read']),
        'service:update': hasPermission(['Admin', 'Developer'], ['service.write']),
        'service:delete': hasPermission(['Admin'], ['service.delete']),

        // Workflow management
        'workflow:create': hasPermission(['Admin', 'Developer'], ['workflow.write']),
        'workflow:execute': hasPermission(['Admin', 'Developer', 'User'], ['workflow.execute']),
        'workflow:read': hasPermission(['Admin', 'Developer', 'User'], ['workflow.read']),

        // Administration
        'admin:network': hasPermission(['Admin'], ['admin.network']),
        'admin:users': hasPermission(['Admin'], ['admin.users']),
        'admin:blockchain': hasPermission(['Admin'], ['admin.blockchain']),

        // Monitoring
        'monitor:view': hasPermission(['Admin', 'Developer'], ['monitor.read']),
        'monitor:alerts': hasPermission(['Admin'], ['monitor.alerts'])
    };

    res.json({
        userId: req.user.id,
        permissions: permissions,
        roles: userRoles,
        scopes: userScopes
    });
});

module.exports = router;