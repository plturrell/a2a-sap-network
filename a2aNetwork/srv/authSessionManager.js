const jwt = require('jsonwebtoken');

const { LoggerFactory } = require('../../shared/logging/structured-logger');
const logger = LoggerFactory.createLogger('authSessionManager');
const bcrypt = require('bcrypt');
const crypto = require('crypto');
const Redis = require('ioredis');

/**
 * Production-ready Authentication and Session Management
 * Handles JWT tokens, session persistence, and security
 */
class AuthSessionManager {
    constructor() {
        // Redis for session storage
        this.redis = new Redis({
            host: process.env.REDIS_HOST || 'localhost',
            port: process.env.REDIS_PORT || 6379,
            password: process.env.REDIS_PASSWORD,
            retryStrategy: (times) => {
                const delay = Math.min(times * 50, 2000);
                return delay;
            }
        });

        // JWT configuration
        this.jwtSecret = process.env.JWT_SECRET || this.generateSecureSecret();
        this.jwtRefreshSecret = process.env.JWT_REFRESH_SECRET || this.generateSecureSecret();
        this.accessTokenExpiry = process.env.ACCESS_TOKEN_EXPIRY || '15m';
        this.refreshTokenExpiry = process.env.REFRESH_TOKEN_EXPIRY || '7d';

        // Session configuration
        this.sessionTimeout = parseInt(process.env.SESSION_TIMEOUT || '3600'); // 1 hour
        this.maxConcurrentSessions = parseInt(process.env.MAX_CONCURRENT_SESSIONS || '5');

        // Security settings
        this.saltRounds = 12;
        this.sessionPrefix = 'session:';
        this.userSessionPrefix = 'user:sessions:';
        this.refreshTokenPrefix = 'refresh:';

        logger.info('🔒 AuthSessionManager initialized with Redis session store');
    }

    /**
     * Generate secure random secret
     */
    generateSecureSecret() {
        return crypto.randomBytes(64).toString('hex');
    }

    /**
     * Authenticate user and create session
     */
    async authenticateUser(username, password, metadata = {}) {
        try {
            // Fetch user from database (example implementation)
            const user = await this.getUserFromDatabase(username);

            if (!user) {
                return { success: false, error: 'Invalid credentials' };
            }

            // Verify password
            const isValid = await bcrypt.compare(password, user.passwordHash);

            if (!isValid) {
                return { success: false, error: 'Invalid credentials' };
            }

            // Check concurrent session limit
            const activeSessions = await this.getUserActiveSessions(user.id);
            if (activeSessions.length >= this.maxConcurrentSessions) {
                // Optionally remove oldest session
                await this.removeOldestSession(user.id);
            }

            // Create session
            const session = await this.createSession(user, metadata);

            return {
                success: true,
                session,
                user: {
                    id: user.id,
                    username: user.username,
                    roles: user.roles
                }
            };

        } catch (error) {
            logger.error('Authentication error:', { error: error });
            return { success: false, error: 'Authentication failed' };
        }
    }

    /**
     * Create new session with tokens
     */
    async createSession(user, metadata = {}) {
        const sessionId = this.generateSessionId();
        const now = new Date();

        // Create tokens
        const accessToken = this.generateAccessToken(user, sessionId);
        const refreshToken = this.generateRefreshToken(user, sessionId);

        // Session data
        const sessionData = {
            sessionId,
            userId: user.id,
            username: user.username,
            roles: user.roles || [],
            createdAt: now.toISOString(),
            lastActivity: now.toISOString(),
            metadata: {
                ...metadata,
                ip: metadata.ip,
                userAgent: metadata.userAgent,
                device: this.detectDevice(metadata.userAgent)
            },
            accessToken,
            refreshToken
        };

        // Store session in Redis
        await this.redis.setex(
            `${this.sessionPrefix}${sessionId}`,
            this.sessionTimeout,
            JSON.stringify(sessionData)
        );

        // Track user sessions
        await this.redis.sadd(`${this.userSessionPrefix}${user.id}`, sessionId);

        // Store refresh token
        await this.redis.setex(
            `${this.refreshTokenPrefix}${refreshToken}`,
            7 * 24 * 60 * 60, // 7 days
            JSON.stringify({ userId: user.id, sessionId })
        );

        return {
            sessionId,
            accessToken,
            refreshToken,
            expiresIn: this.sessionTimeout
        };
    }

    /**
     * Generate JWT access token
     */
    generateAccessToken(user, sessionId) {
        return jwt.sign(
            {
                userId: user.id,
                username: user.username,
                roles: user.roles || [],
                sessionId,
                type: 'access'
            },
            this.jwtSecret,
            { expiresIn: this.accessTokenExpiry }
        );
    }

    /**
     * Generate JWT refresh token
     */
    generateRefreshToken(user, sessionId) {
        return jwt.sign(
            {
                userId: user.id,
                sessionId,
                type: 'refresh'
            },
            this.jwtRefreshSecret,
            { expiresIn: this.refreshTokenExpiry }
        );
    }

    /**
     * Validate access token
     */
    async validateAccessToken(token) {
        try {
            const decoded = jwt.verify(token, this.jwtSecret);

            if (decoded.type !== 'access') {
                throw new Error('Invalid token type');
            }

            // Check if session exists
            const session = await this.getSession(decoded.sessionId);

            if (!session) {
                throw new Error('Session not found');
            }

            // Update last activity
            await this.updateSessionActivity(decoded.sessionId);

            return {
                valid: true,
                userId: decoded.userId,
                sessionId: decoded.sessionId,
                roles: decoded.roles
            };

        } catch (error) {
            return {
                valid: false,
                error: error.message
            };
        }
    }

    /**
     * Refresh access token
     */
    async refreshAccessToken(refreshToken) {
        try {
            const decoded = jwt.verify(refreshToken, this.jwtRefreshSecret);

            if (decoded.type !== 'refresh') {
                throw new Error('Invalid token type');
            }

            // Check if refresh token exists in Redis
            const tokenData = await this.redis.get(`${this.refreshTokenPrefix}${refreshToken}`);

            if (!tokenData) {
                throw new Error('Refresh token not found');
            }

            const { userId, sessionId } = JSON.parse(tokenData);

            // Get user data
            const user = await this.getUserById(userId);

            if (!user) {
                throw new Error('User not found');
            }

            // Generate new access token
            const newAccessToken = this.generateAccessToken(user, sessionId);

            // Update session
            await this.updateSessionToken(sessionId, newAccessToken);

            return {
                success: true,
                accessToken: newAccessToken,
                expiresIn: this.sessionTimeout
            };

        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Get session data
     */
    async getSession(sessionId) {
        const data = await this.redis.get(`${this.sessionPrefix}${sessionId}`);
        return data ? JSON.parse(data) : null;
    }

    /**
     * Update session activity timestamp
     */
    async updateSessionActivity(sessionId) {
        const session = await this.getSession(sessionId);

        if (session) {
            session.lastActivity = new Date().toISOString();

            await this.redis.setex(
                `${this.sessionPrefix}${sessionId}`,
                this.sessionTimeout,
                JSON.stringify(session)
            );
        }
    }

    /**
     * Get user's active sessions
     */
    async getUserActiveSessions(userId) {
        const sessionIds = await this.redis.smembers(`${this.userSessionPrefix}${userId}`);
        const sessions = [];

        for (const sessionId of sessionIds) {
            const session = await this.getSession(sessionId);
            if (session) {
                sessions.push(session);
            } else {
                // Clean up invalid session reference
                await this.redis.srem(`${this.userSessionPrefix}${userId}`, sessionId);
            }
        }

        return sessions;
    }

    /**
     * Remove session
     */
    async removeSession(sessionId) {
        const session = await this.getSession(sessionId);

        if (session) {
            // Remove session
            await this.redis.del(`${this.sessionPrefix}${sessionId}`);

            // Remove from user sessions
            await this.redis.srem(`${this.userSessionPrefix}${session.userId}`, sessionId);

            // Remove refresh token
            if (session.refreshToken) {
                await this.redis.del(`${this.refreshTokenPrefix}${session.refreshToken}`);
            }
        }
    }

    /**
     * Remove oldest session for user
     */
    async removeOldestSession(userId) {
        const sessions = await this.getUserActiveSessions(userId);

        if (sessions.length > 0) {
            // Sort by creation date
            sessions.sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt));

            // Remove oldest
            await this.removeSession(sessions[0].sessionId);
        }
    }

    /**
     * Logout user
     */
    async logout(sessionId) {
        await this.removeSession(sessionId);
        return { success: true };
    }

    /**
     * Generate secure session ID
     */
    generateSessionId() {
        return crypto.randomBytes(32).toString('hex');
    }

    /**
     * Detect device type from user agent
     */
    detectDevice(userAgent) {
        if (!userAgent) return 'unknown';

        if (/mobile/i.test(userAgent)) return 'mobile';
        if (/tablet/i.test(userAgent)) return 'tablet';
        if (/bot/i.test(userAgent)) return 'bot';

        return 'desktop';
    }

    /**
     * Example: Get user from database
     */
    async getUserFromDatabase(username) {
        // In production, this would query your actual database
        // This is a mock implementation
        const mockUsers = {
            'admin': {
                id: '1',
                username: 'admin',
                passwordHash: await bcrypt.hash('admin123', this.saltRounds),
                roles: ['admin', 'user']
            },
            'user': {
                id: '2',
                username: 'user',
                passwordHash: await bcrypt.hash('user123', this.saltRounds),
                roles: ['user']
            }
        };

        return mockUsers[username];
    }

    /**
     * Example: Get user by ID
     */
    async getUserById(userId) {
        // Mock implementation
        const mockUsers = {
            '1': { id: '1', username: 'admin', roles: ['admin', 'user'] },
            '2': { id: '2', username: 'user', roles: ['user'] }
        };

        return mockUsers[userId];
    }

    /**
     * Middleware for Express
     */
    middleware() {
        return async (req, res, next) => {
            const token = this.extractToken(req);

            if (!token) {
                req.auth = { authenticated: false };
                return next();
            }

            const validation = await this.validateAccessToken(token);

            if (validation.valid) {
                req.auth = {
                    authenticated: true,
                    userId: validation.userId,
                    sessionId: validation.sessionId,
                    roles: validation.roles
                };
            } else {
                req.auth = { authenticated: false };
            }

            next();
        };
    }

    /**
     * Extract token from request
     */
    extractToken(req) {
        // Check Authorization header
        const authHeader = req.headers.authorization;
        if (authHeader && authHeader.startsWith('Bearer ')) {
            return authHeader.substring(7);
        }

        // Check cookie
        if (req.cookies && req.cookies.accessToken) {
            return req.cookies.accessToken;
        }

        return null;
    }

    /**
     * Require authentication middleware
     */
    requireAuth() {
        return (req, res, next) => {
            if (!req.auth || !req.auth.authenticated) {
                return res.status(401).json({
                    error: 'Authentication required'
                });
            }
            next();
        };
    }

    /**
     * Require specific role middleware
     */
    requireRole(role) {
        return (req, res, next) => {
            if (!req.auth || !req.auth.authenticated) {
                return res.status(401).json({
                    error: 'Authentication required'
                });
            }

            if (!req.auth.roles || !req.auth.roles.includes(role)) {
                return res.status(403).json({
                    error: 'Insufficient permissions'
                });
            }

            next();
        };
    }

    /**
     * Clean up expired sessions
     */
    async cleanupExpiredSessions() {
        // This would run as a periodic job
        logger.info('🧹 Cleaning up expired sessions...');

        // Redis automatically expires keys, but we can do additional cleanup
        // of user session references here if needed
    }
}

// Export singleton instance
const authManager = new AuthSessionManager();

module.exports = {
    AuthSessionManager,
    getInstance: () => authManager
};