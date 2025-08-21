/**
 * @fileoverview User Management Service - CDS Definition
 * @since 1.0.0
 * @module userManagementService
 * 
 * CDS service definition for user profile and authentication management
 * Replaces Express router with proper CAP service actions
 */

namespace a2a.user;

/**
 * UserManagementService - User profile and authentication management
 * Provides user info, profile, preferences, and permissions
 */
service UserManagementService @(path: '/api/v1/user') {
    
    /**
     * Get current authenticated user information
     */
    action getCurrentUser() returns {
        id: String;
        name: String;
        email: String;
        scopes: array of String;
        roles: array of String;
        tenant: String;
    };
    
    /**
     * Get user profile with preferences
     */
    action getProfile() returns {
        id: String;
        preferences: {
            theme: String;
            language: String;
            dateFormat: String;
            timezone: String;
        };
        lastLogin: String;
        permissions: array of String;
    };
    
    /**
     * Update user preferences
     */
    action updatePreferences(preferences: {
        theme: String;
        language: String;
        dateFormat: String;
        timezone: String;
    }) returns {
        success: Boolean;
        message: String;
        preferences: {
            theme: String;
            language: String;
            dateFormat: String;
            timezone: String;
        };
    };
    
    /**
     * Get user permissions and access levels
     */
    action getPermissions() returns {
        scopes: array of String;
        roles: array of String;
        tenant: String;
        hasAdminAccess: Boolean;
        hasDeveloperAccess: Boolean;
    };
}