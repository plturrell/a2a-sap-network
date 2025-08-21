/**
 * @fileoverview SAP User Service - CAP Service Implementation
 * @since 1.0.0
 * @module sapUserService
 * 
 * SAP-compliant user management service for A2A Network
 */

const cds = require('@sap/cds');

class SAPUserService {
    constructor() {
        this.users = new Map();
        this.sessions = new Map();
        this.log = cds.log('sapUserService');
    }

    async getUserById(userId) {
        return this.users.get(userId) || null;
    }

    async createUser(userData) {
        const user = {
            id: userData.id || this.generateUserId(),
            username: userData.username,
            email: userData.email,
            roles: userData.roles || ['authenticated-user'],
            created: new Date().toISOString(),
            ...userData
        };
        
        this.users.set(user.id, user);
        this.log.info(`User created: ${user.id}`);
        return user;
    }

    async updateUser(userId, updates) {
        const user = this.users.get(userId);
        if (!user) {
            throw new Error(`User not found: ${userId}`);
        }
        
        const updatedUser = { ...user, ...updates, updated: new Date().toISOString() };
        this.users.set(userId, updatedUser);
        this.log.info(`User updated: ${userId}`);
        return updatedUser;
    }

    async deleteUser(userId) {
        const deleted = this.users.delete(userId);
        if (deleted) {
            this.log.info(`User deleted: ${userId}`);
        }
        return deleted;
    }

    async validateSession(sessionId) {
        return this.sessions.get(sessionId) || null;
    }

    async createSession(userId) {
        const sessionId = this.generateSessionId();
        const session = {
            id: sessionId,
            userId: userId,
            created: new Date().toISOString(),
            expires: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString() // 24 hours
        };
        
        this.sessions.set(sessionId, session);
        this.log.info(`Session created: ${sessionId} for user: ${userId}`);
        return session;
    }

    generateUserId() {
        return 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    // Development mode helpers
    async initializeDevelopmentUsers() {
        if (process.env.NODE_ENV === 'development' || process.env.USE_DEVELOPMENT_AUTH === 'true') {
            const devUser = {
                id: 'dev-user',
                username: 'developer',
                email: 'developer@a2a.local',
                roles: ['authenticated-user', 'Admin'],
                isDevelopment: true
            };
            
            await this.createUser(devUser);
            this.log.info('Development user initialized');
        }
    }
}

// Export singleton instance
const sapUserService = new SAPUserService();

// Initialize development users if in development mode
if (process.env.NODE_ENV === 'development' || process.env.USE_DEVELOPMENT_AUTH === 'true') {
    sapUserService.initializeDevelopmentUsers().catch(err => {
        cds.log('sapUserService').warn('Failed to initialize development users:', err);
    });
}

module.exports = sapUserService;