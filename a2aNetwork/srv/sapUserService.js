/**
 * @fileoverview SAP User Service - BTP Integration
 * @description Handles user authentication and profile management for BTP environments
 * @module sapUserService
 * @since 1.0.0
 * @author A2A Network Team
 * @namespace a2a.srv.user
 */

const express = require('express');
const cds = require('@sap/cds');
const router = express.Router();

/**
 * Get current user information
 * @route GET /currentUser
 */
router.get('/currentUser', async (req, res) => {
    try {
        // In BTP environment with XSUAA
        if (req.user) {
            const user = {
                id: req.user.id || req.user.sub || 'anonymous',
                name: req.user.given_name && req.user.family_name 
                    ? `${req.user.given_name} ${req.user.family_name}`
                    : req.user.name || req.user.id || 'Unknown User',
                email: req.user.email || null,
                scopes: req.user.scopes || [],
                roles: req.user.roles || [],
                tenant: req.user.tenant || 'default'
            };
            
            res.json(user);
            cds.log('user').info('User info retrieved', { userId: user.id });
        } else {
            // Development environment fallback
            const devUser = {
                id: 'dev-user',
                name: 'Development User',
                email: 'dev@a2a.network',
                scopes: ['Admin', 'Developer'],
                roles: ['Admin', 'Developer'],
                tenant: 'development'
            };
            
            res.json(devUser);
            cds.log('user').warn('Using development user - ensure XSUAA is configured for production');
        }
    } catch (error) {
        cds.log('user').error('Failed to get user info:', error);
        res.status(500).json({ 
            error: 'Failed to retrieve user information',
            message: error.message 
        });
    }
});

/**
 * Get user profile
 * @route GET /profile
 */
router.get('/profile', async (req, res) => {
    try {
        const userId = req.user?.id || req.user?.sub;
        
        if (!userId) {
            return res.status(401).json({ error: 'User not authenticated' });
        }
        
        // In a real implementation, this would fetch from user database
        const profile = {
            id: userId,
            preferences: {
                theme: 'sap_fiori_3',
                language: 'en',
                dateFormat: 'MM/DD/YYYY',
                timezone: 'UTC'
            },
            lastLogin: new Date().toISOString(),
            permissions: req.user?.scopes || []
        };
        
        res.json(profile);
    } catch (error) {
        cds.log('user').error('Failed to get user profile:', error);
        res.status(500).json({ 
            error: 'Failed to retrieve user profile',
            message: error.message 
        });
    }
});

/**
 * Update user preferences
 * @route PUT /profile/preferences
 */
router.put('/profile/preferences', async (req, res) => {
    try {
        const userId = req.user?.id || req.user?.sub;
        
        if (!userId) {
            return res.status(401).json({ error: 'User not authenticated' });
        }
        
        const { preferences } = req.body;
        
        // Validate preferences
        if (!preferences || typeof preferences !== 'object') {
            return res.status(400).json({ error: 'Invalid preferences format' });
        }
        
        // In a real implementation, this would update the user database
        cds.log('user').info('User preferences updated', { userId, preferences });
        
        res.json({ 
            success: true, 
            message: 'Preferences updated successfully',
            preferences 
        });
        
    } catch (error) {
        cds.log('user').error('Failed to update user preferences:', error);
        res.status(500).json({ 
            error: 'Failed to update preferences',
            message: error.message 
        });
    }
});

/**
 * Get user permissions/scopes
 * @route GET /permissions
 */
router.get('/permissions', async (req, res) => {
    try {
        const permissions = {
            scopes: req.user?.scopes || [],
            roles: req.user?.roles || [],
            tenant: req.user?.tenant || 'default',
            hasAdminAccess: (req.user?.scopes || []).includes('Admin'),
            hasDeveloperAccess: (req.user?.scopes || []).includes('Developer')
        };
        
        res.json(permissions);
    } catch (error) {
        cds.log('user').error('Failed to get permissions:', error);
        res.status(500).json({ 
            error: 'Failed to retrieve permissions',
            message: error.message 
        });
    }
});

module.exports = router;