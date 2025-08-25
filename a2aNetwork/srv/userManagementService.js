/**
 * @fileoverview User Management Service - CAP Implementation
 * @since 1.0.0
 * @module userManagementService
 *
 * CAP service handlers for user profile and authentication management
 * Replaces Express router with proper SAP CAP architecture
 */

const cds = require('@sap/cds');
const { SELECT, INSERT, UPDATE, DELETE } = cds.ql;
const LOG = cds.log('user-management');

/**
 * CAP Service Handler for User Management Actions
 */
module.exports = function() {

    // Get current authenticated user information
    this.on('getCurrentUser', async (req) => {
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

                LOG.info('User info retrieved', { userId: user.id });
                return user;
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

                LOG.warn('Using development user - ensure XSUAA is configured for production');
                return devUser;
            }
        } catch (error) {
            LOG.error('Failed to get user info:', error);
            req.error(500, 'USER_INFO_ERROR', `Failed to retrieve user information: ${error.message}`);
        }
    });

    // Get user profile with preferences
    this.on('getProfile', async (req) => {
        try {
            const userId = req.user?.id || req.user?.sub;

            if (!userId) {
                req.error(401, 'UNAUTHORIZED', 'User not authenticated');
                return;
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

            return profile;
        } catch (error) {
            LOG.error('Failed to get user profile:', error);
            req.error(500, 'PROFILE_ERROR', `Failed to retrieve user profile: ${error.message}`);
        }
    });

    // Update user preferences
    this.on('updatePreferences', async (req) => {
        try {
            const userId = req.user?.id || req.user?.sub;

            if (!userId) {
                req.error(401, 'UNAUTHORIZED', 'User not authenticated');
                return;
            }

            const { preferences } = req.data;

            // Validate preferences
            if (!preferences || typeof preferences !== 'object') {
                req.error(400, 'INVALID_PREFERENCES', 'Invalid preferences format');
                return;
            }

            // In a real implementation, this would update the user database
            LOG.info('User preferences updated', { userId, preferences });

            return {
                success: true,
                message: 'Preferences updated successfully',
                preferences
            };

        } catch (error) {
            LOG.error('Failed to update user preferences:', error);
            req.error(500, 'UPDATE_ERROR', `Failed to update preferences: ${error.message}`);
        }
    });

    // Get user permissions and access levels
    this.on('getPermissions', async (req) => {
        try {
            const permissions = {
                scopes: req.user?.scopes || [],
                roles: req.user?.roles || [],
                tenant: req.user?.tenant || 'default',
                hasAdminAccess: (req.user?.scopes || []).includes('Admin'),
                hasDeveloperAccess: (req.user?.scopes || []).includes('Developer')
            };

            return permissions;
        } catch (error) {
            LOG.error('Failed to get permissions:', error);
            req.error(500, 'PERMISSIONS_ERROR', `Failed to retrieve permissions: ${error.message}`);
        }
    });

    LOG.info('User Management service handlers registered');
};
