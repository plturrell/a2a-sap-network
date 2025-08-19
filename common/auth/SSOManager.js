/**
 * SAP Enterprise SSO Manager
 * Implements unified authentication across A2A platform components
 * 
 * @module SSOManager
 * @implements {TC-COM-LPD-001}
 */

const cds = require('@sap/cds');
const jwt = require('jsonwebtoken');
const crypto = require('crypto');

class SSOManager {
    constructor() {
        this.tokenStore = new Map();
        this.sessionStore = new Map();
        this.config = {
            tokenExpiry: 3600, // 1 hour
            refreshExpiry: 86400, // 24 hours
            maxSessions: 5,
            enableMFA: true
        };
    }

    /**
     * Authenticate user via SSO
     * @param {Object} credentials - User credentials
     * @returns {Promise<Object>} Authentication result
     */
    async authenticateUser(credentials) {
        try {
            // Validate input
            if (!credentials || !credentials.username) {
                throw new Error('Invalid credentials');
            }

            // Check if user is already authenticated
            const existingSession = this.sessionStore.get(credentials.username);
            if (existingSession && this.isSessionValid(existingSession)) {
                return {
                    success: true,
                    token: existingSession.token,
                    refreshToken: existingSession.refreshToken,
                    cached: true
                };
            }

            // Perform authentication (SAML/OAuth2)
            const authResult = await this._performAuthentication(credentials);
            
            if (!authResult.success) {
                throw new Error(authResult.error || 'Authentication failed');
            }

            // Generate tokens
            const token = this._generateToken(authResult.user);
            const refreshToken = this._generateRefreshToken(authResult.user);

            // Store session
            this._storeSession(authResult.user.id, {
                token,
                refreshToken,
                user: authResult.user,
                createdAt: new Date(),
                lastAccess: new Date()
            });

            // Emit authentication event
            await cds.emit('authentication.success', {
                userId: authResult.user.id,
                timestamp: new Date()
            });

            return {
                success: true,
                token,
                refreshToken,
                user: authResult.user
            };

        } catch (error) {
            console.error('SSO authentication error:', error);
            await cds.emit('authentication.failed', {
                username: credentials.username,
                error: error.message,
                timestamp: new Date()
            });
            throw error;
        }
    }

    /**
     * Validate authentication token
     * @param {string} token - JWT token
     * @returns {Promise<Object>} Validation result
     */
    async validateToken(token) {
        try {
            if (!token) {
                throw new Error('Token required');
            }

            // Check token blacklist
            if (this._isTokenBlacklisted(token)) {
                throw new Error('Token has been revoked');
            }

            // Verify JWT
            const decoded = jwt.verify(token, this._getSecretKey());
            
            // Check token expiry
            if (decoded.exp && decoded.exp < Date.now() / 1000) {
                throw new Error('Token expired');
            }

            // Validate token claims
            if (!decoded.sub || !decoded.iat) {
                throw new Error('Invalid token claims');
            }

            // Update last access
            const session = this._getSessionByToken(token);
            if (session) {
                session.lastAccess = new Date();
            }

            return {
                valid: true,
                userId: decoded.sub,
                permissions: decoded.permissions || [],
                expiresAt: new Date(decoded.exp * 1000)
            };

        } catch (error) {
            console.error('Token validation error:', error);
            return {
                valid: false,
                error: error.message
            };
        }
    }

    /**
     * Refresh user session
     * @param {string} refreshToken - Refresh token
     * @returns {Promise<Object>} New tokens
     */
    async refreshSession(refreshToken) {
        try {
            if (!refreshToken) {
                throw new Error('Refresh token required');
            }

            // Validate refresh token
            const decoded = jwt.verify(refreshToken, this._getRefreshSecretKey());
            
            // Get user session
            const session = this._getSessionByUserId(decoded.sub);
            if (!session || session.refreshToken !== refreshToken) {
                throw new Error('Invalid refresh token');
            }

            // Generate new tokens
            const newToken = this._generateToken(session.user);
            const newRefreshToken = this._generateRefreshToken(session.user);

            // Update session
            session.token = newToken;
            session.refreshToken = newRefreshToken;
            session.lastAccess = new Date();

            return {
                success: true,
                token: newToken,
                refreshToken: newRefreshToken
            };

        } catch (error) {
            console.error('Session refresh error:', error);
            throw error;
        }
    }

    /**
     * Logout user and invalidate tokens
     * @param {string} token - User token
     * @returns {Promise<boolean>} Logout success
     */
    async logout(token) {
        try {
            const validation = await this.validateToken(token);
            if (!validation.valid) {
                return false;
            }

            // Remove session
            this.sessionStore.delete(validation.userId);
            
            // Blacklist token
            this._blacklistToken(token);

            // Emit logout event
            await cds.emit('authentication.logout', {
                userId: validation.userId,
                timestamp: new Date()
            });

            return true;

        } catch (error) {
            console.error('Logout error:', error);
            return false;
        }
    }

    /**
     * Perform actual authentication (SAML/OAuth2)
     * @private
     */
    async _performAuthentication(credentials) {
        // Simulate authentication for testing
        // In production, this would integrate with SAML/OAuth2 providers
        
        if (credentials.username === 'testuser' && credentials.password === 'testpass') {
            return {
                success: true,
                user: {
                    id: 'user-123',
                    username: credentials.username,
                    email: 'testuser@example.com',
                    permissions: ['read', 'write'],
                    roles: ['user']
                }
            };
        }

        return {
            success: false,
            error: 'Invalid credentials'
        };
    }

    /**
     * Generate JWT token
     * @private
     */
    _generateToken(user) {
        const payload = {
            sub: user.id,
            username: user.username,
            email: user.email,
            permissions: user.permissions,
            iat: Math.floor(Date.now() / 1000),
            exp: Math.floor(Date.now() / 1000) + this.config.tokenExpiry
        };

        return jwt.sign(payload, this._getSecretKey());
    }

    /**
     * Generate refresh token
     * @private
     */
    _generateRefreshToken(user) {
        const payload = {
            sub: user.id,
            type: 'refresh',
            iat: Math.floor(Date.now() / 1000),
            exp: Math.floor(Date.now() / 1000) + this.config.refreshExpiry
        };

        return jwt.sign(payload, this._getRefreshSecretKey());
    }

    /**
     * Get secret key for JWT
     * @private
     */
    _getSecretKey() {
        return process.env.JWT_SECRET || 'development-secret-key';
    }

    /**
     * Get refresh secret key
     * @private
     */
    _getRefreshSecretKey() {
        return process.env.JWT_REFRESH_SECRET || 'development-refresh-secret';
    }

    /**
     * Store user session
     * @private
     */
    _storeSession(userId, session) {
        this.sessionStore.set(userId, session);
        this.tokenStore.set(session.token, userId);
    }

    /**
     * Get session by token
     * @private
     */
    _getSessionByToken(token) {
        const userId = this.tokenStore.get(token);
        return userId ? this.sessionStore.get(userId) : null;
    }

    /**
     * Get session by user ID
     * @private
     */
    _getSessionByUserId(userId) {
        return this.sessionStore.get(userId);
    }

    /**
     * Check if session is valid
     * @private
     */
    isSessionValid(session) {
        if (!session) return false;
        
        const now = new Date();
        const lastAccess = new Date(session.lastAccess);
        const maxInactivity = 30 * 60 * 1000; // 30 minutes
        
        return (now - lastAccess) < maxInactivity;
    }

    /**
     * Blacklist token
     * @private
     */
    _blacklistToken(token) {
        // In production, this would use Redis or similar
        this.blacklist = this.blacklist || new Set();
        this.blacklist.add(token);
    }

    /**
     * Check if token is blacklisted
     * @private
     */
    _isTokenBlacklisted(token) {
        return this.blacklist && this.blacklist.has(token);
    }
}

// Export singleton instance
module.exports = new SSOManager();