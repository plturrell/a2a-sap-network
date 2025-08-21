/**
 * @fileoverview Secure WebSocket Authentication Middleware
 * @description Enforces authentication and security for all WebSocket connections
 * @module secureWebSocketAuth
 * @since 4.0.0
 */

const jwt = require('jsonwebtoken');
const crypto = require('crypto');
const { RateLimiterMemory } = require('rate-limiter-flexible');
const cds = require('@sap/cds');

/**
 * WebSocket Security Configuration
 */
const WS_SECURITY_CONFIG = {
    // Connection limits
    MAX_CONNECTIONS_PER_IP: 10,
    MAX_CONNECTIONS_PER_USER: 5,
    CONNECTION_TIMEOUT: 300000, // 5 minutes
    
    // Message limits
    MAX_MESSAGE_SIZE: 1048576, // 1MB
    MAX_MESSAGES_PER_MINUTE: 100,
    MAX_SUBSCRIPTIONS: 20,
    
    // Authentication
    TOKEN_REFRESH_INTERVAL: 1800000, // 30 minutes
    HEARTBEAT_INTERVAL: 30000, // 30 seconds
    HEARTBEAT_TIMEOUT: 60000, // 60 seconds
    
    // Security
    ALLOWED_EVENTS: [
        'subscribe',
        'unsubscribe',
        'ping',
        'refresh-token',
        'get-status'
    ],
    
    BLOCKED_PAYLOAD_PATTERNS: [
        /<script/i,
        /javascript:/i,
        /on\w+=/i,
        /__proto__/,
        /constructor/,
        /prototype/
    ]
};

/**
 * WebSocket Security Manager
 */
class WebSocketSecurityManager {
    constructor() {
        this.connections = new Map();
        this.ipConnections = new Map();
        this.userConnections = new Map();
        this.suspiciousActivity = new Map();
        
        // Rate limiters
        this.connectionLimiter = new RateLimiterMemory({
            points: 5, // 5 connections
            duration: 60 // per minute
        });
        
        this.intervals = new Map(); // Track intervals for cleanup
        
        this.messageLimiter = new RateLimiterMemory({
            points: WS_SECURITY_CONFIG.MAX_MESSAGES_PER_MINUTE,
            duration: 60
        });
        
        // Start cleanup interval
        this.intervals.set('interval_73', setInterval(() => this.cleanup(), 60000)); // Every minute
    }
    
    /**
     * Main authentication middleware for Socket.IO
     */
    createAuthMiddleware(jwtPublicKey) {
        return async (socket, next) => {
            try {
                // Extract token
                const token = socket.handshake.auth.token || 
                             socket.handshake.headers.authorization?.substring(7);
                
                if (!token) {
                    return next(new Error('Authentication required'));
                }
                
                // Validate origin
                const origin = socket.handshake.headers.origin;
                if (!this.isValidOrigin(origin)) {
                    this.logSecurityEvent('INVALID_WS_ORIGIN', {
                        origin,
                        ip: socket.handshake.address
                    });
                    return next(new Error('Invalid origin'));
                }
                
                // Check connection limits
                const ip = this.getClientIP(socket);
                if (!await this.checkConnectionLimits(ip)) {
                    return next(new Error('Connection limit exceeded'));
                }
                
                // Verify JWT (RS256 only)
                const payload = await this.verifyToken(token, jwtPublicKey);
                if (!payload) {
                    return next(new Error('Invalid token'));
                }
                
                // Check user connection limits
                if (!this.checkUserConnectionLimit(payload.user.id)) {
                    return next(new Error('User connection limit exceeded'));
                }
                
                // Setup connection tracking
                this.trackConnection(socket, payload.user, ip);
                
                // Attach user and security info
                socket.user = payload.user;
                socket.authTime = Date.now();
                socket.lastActivity = Date.now();
                socket.ip = ip;
                
                // Setup security handlers
                this.setupSecurityHandlers(socket);
                
                next();
            } catch (error) {
                cds.log('ws-security').error('WebSocket auth failed', error);
                next(new Error('Authentication failed'));
            }
        };
    }
    
    /**
     * Verify JWT token with RS256
     */
    async verifyToken(token, publicKey) {
        try {
            // Decode to check algorithm
            const decoded = jwt.decode(token, { complete: true });
            if (!decoded || decoded.header.alg !== 'RS256') {
                throw new Error('Invalid token algorithm');
            }
            
            // Verify with public key
            const payload = jwt.verify(token, publicKey, {
                algorithms: ['RS256'],
                clockTolerance: 30
            });
            
            // Additional checks
            if (!payload.user || !payload.user.id) {
                throw new Error('Invalid token payload');
            }
            
            return payload;
        } catch (error) {
            return null;
        }
    }
    
    /**
     * Setup security handlers for socket
     */
    setupSecurityHandlers(socket) {
        const originalEmit = socket.emit;
        const originalOn = socket.on;
        
        // Wrap emit to validate outgoing messages
        socket.emit = function(...args) {
            const [event, data] = args;
            
            // Validate event
            if (!this.validateOutgoingEvent(event, data)) {
                cds.log('ws-security').warn('Blocked outgoing event', {
                    event,
                    userId: socket.user.id
                });
                return;
            }
            
            originalEmit.apply(socket, args);
        }.bind(this);
        
        // Wrap on to validate incoming messages
        socket.on = function(event, handler) {
            const secureHandler = async (...args) => {
                try {
                    // Rate limiting
                    await this.checkMessageRateLimit(socket);
                    
                    // Validate event
                    if (!this.validateIncomingEvent(event, args[0], socket)) {
                        socket.emit('error', {
                            message: 'Invalid event',
                            code: 'INVALID_EVENT'
                        });
                        return;
                    }
                    
                    // Update activity
                    socket.lastActivity = Date.now();
                    
                    // Call original handler
                    handler(...args);
                } catch (error) {
                    if (error.message === 'Rate limit exceeded') {
                        socket.emit('error', {
                            message: 'Too many messages',
                            code: 'RATE_LIMITED'
                        });
                    } else {
                        cds.log('ws-security').error('Handler error', error);
                    }
                }
            };
            
            originalOn.call(socket, event, secureHandler);
        }.bind(this);
        
        // Setup heartbeat
        this.setupHeartbeat(socket);
        
        // Setup token refresh
        socket.on('refresh-token', async (newToken) => {
            try {
                const payload = await this.verifyToken(newToken, socket.publicKey);
                if (payload && payload.user.id === socket.user.id) {
                    socket.authTime = Date.now();
                    socket.emit('token-refreshed', { success: true });
                } else {
                    socket.disconnect(true);
                }
            } catch (error) {
                socket.disconnect(true);
            }
        });
    }
    
    /**
     * Setup heartbeat monitoring
     */
    setupHeartbeat(socket) {
        let heartbeatTimeout;
        
        const resetHeartbeat = () => {
            clearTimeout(heartbeatTimeout);
            heartbeatTimeout = setTimeout(() => {
                cds.log('ws-security').warn('Heartbeat timeout', {
                    userId: socket.user.id,
                    socketId: socket.id
                });
                socket.disconnect(true);
            }, WS_SECURITY_CONFIG.HEARTBEAT_TIMEOUT);
        };
        
        socket.on('ping', () => {
            socket.emit('pong', { timestamp: Date.now() });
            resetHeartbeat();
        });
        
        resetHeartbeat();
        
        // Start heartbeat interval
        const heartbeatInterval = this.intervals.set('interval_268', setInterval(() => {
            if (socket.connected) {
                socket.emit('ping', { timestamp: Date.now()) });
            } else {
                clearInterval(heartbeatInterval);
                clearTimeout(heartbeatTimeout);
            }
        }, WS_SECURITY_CONFIG.HEARTBEAT_INTERVAL);
        
        socket.on('disconnect', () => {
            clearInterval(heartbeatInterval);
            clearTimeout(heartbeatTimeout);
        });
    }
    
    /**
     * Validate incoming event
     */
    validateIncomingEvent(event, data, socket) {
        // Check if event is allowed
        if (!WS_SECURITY_CONFIG.ALLOWED_EVENTS.includes(event) && 
            !event.startsWith('authenticated:')) {
            this.recordSuspiciousActivity(socket, 'UNKNOWN_EVENT', { event });
            return false;
        }
        
        // Check payload size
        const payloadSize = JSON.stringify(data || {}).length;
        if (payloadSize > WS_SECURITY_CONFIG.MAX_MESSAGE_SIZE) {
            this.recordSuspiciousActivity(socket, 'OVERSIZED_PAYLOAD', { 
                size: payloadSize 
            });
            return false;
        }
        
        // Check for malicious patterns
        const payloadStr = JSON.stringify(data);
        for (const pattern of WS_SECURITY_CONFIG.BLOCKED_PAYLOAD_PATTERNS) {
            if (pattern.test(payloadStr)) {
                this.recordSuspiciousActivity(socket, 'MALICIOUS_PAYLOAD', { 
                    pattern: pattern.source 
                });
                return false;
            }
        }
        
        // Validate specific events
        switch (event) {
            case 'subscribe':
                return this.validateSubscription(data, socket);
            case 'unsubscribe':
                return this.validateUnsubscription(data, socket);
            default:
                return true;
        }
    }
    
    /**
     * Validate subscription request
     */
    validateSubscription(topics, socket) {
        if (!Array.isArray(topics)) {
            return false;
        }
        
        // Check subscription limit
        const currentSubs = socket.subscriptions || new Set();
        if (currentSubs.size + topics.length > WS_SECURITY_CONFIG.MAX_SUBSCRIPTIONS) {
            this.recordSuspiciousActivity(socket, 'SUBSCRIPTION_LIMIT', { 
                requested: topics.length,
                current: currentSubs.size
            });
            return false;
        }
        
        // Validate topic names
        const validTopicPattern = /^[a-zA-Z0-9._-]+$/;
        for (const topic of topics) {
            if (!validTopicPattern.test(topic)) {
                this.recordSuspiciousActivity(socket, 'INVALID_TOPIC', { topic });
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Track connection
     */
    trackConnection(socket, user, ip) {
        // Track by socket ID
        this.connections.set(socket.id, {
            socket,
            user,
            ip,
            connectedAt: Date.now(),
            lastActivity: Date.now()
        });
        
        // Track by IP
        const ipSockets = this.ipConnections.get(ip) || new Set();
        ipSockets.add(socket.id);
        this.ipConnections.set(ip, ipSockets);
        
        // Track by user
        const userSockets = this.userConnections.get(user.id) || new Set();
        userSockets.add(socket.id);
        this.userConnections.set(user.id, userSockets);
        
        // Setup disconnect handler
        socket.on('disconnect', () => {
            this.removeConnection(socket.id, ip, user.id);
        });
    }
    
    /**
     * Remove connection tracking
     */
    removeConnection(socketId, ip, userId) {
        this.connections.delete(socketId);
        
        // Remove from IP tracking
        const ipSockets = this.ipConnections.get(ip);
        if (ipSockets) {
            ipSockets.delete(socketId);
            if (ipSockets.size === 0) {
                this.ipConnections.delete(ip);
            }
        }
        
        // Remove from user tracking
        const userSockets = this.userConnections.get(userId);
        if (userSockets) {
            userSockets.delete(socketId);
            if (userSockets.size === 0) {
                this.userConnections.delete(userId);
            }
        }
    }
    
    /**
     * Check connection limits
     */
    async checkConnectionLimits(ip) {
        try {
            // Rate limit check
            await this.connectionLimiter.consume(ip);
            
            // Absolute limit check
            const ipSockets = this.ipConnections.get(ip) || new Set();
            if (ipSockets.size >= WS_SECURITY_CONFIG.MAX_CONNECTIONS_PER_IP) {
                return false;
            }
            
            return true;
        } catch (error) {
            return false;
        }
    }
    
    /**
     * Check message rate limit
     */
    async checkMessageRateLimit(socket) {
        const key = `${socket.ip}:${socket.user.id}`;
        await this.messageLimiter.consume(key);
    }
    
    /**
     * Record suspicious activity
     */
    recordSuspiciousActivity(socket, type, details) {
        const key = `${socket.ip}:${socket.user.id}`;
        const activities = this.suspiciousActivity.get(key) || [];
        
        activities.push({
            type,
            details,
            timestamp: Date.now(),
            socketId: socket.id
        });
        
        this.suspiciousActivity.set(key, activities);
        
        // Check if should disconnect
        const recentActivities = activities.filter(
            a => Date.now() - a.timestamp < 300000 // Last 5 minutes
        );
        
        if (recentActivities.length > 10) {
            cds.log('ws-security').error('Disconnecting suspicious client', {
                ip: socket.ip,
                userId: socket.user.id,
                activities: recentActivities.length
            });
            socket.disconnect(true);
        }
    }
    
    /**
     * Cleanup old connections and data
     */
    cleanup() {
        const now = Date.now();
        
        // Clean up stale connections
        for (const [socketId, conn] of this.connections.entries()) {
            if (now - conn.lastActivity > WS_SECURITY_CONFIG.CONNECTION_TIMEOUT) {
                conn.socket.disconnect(true);
            }
        }
        
        // Clean up old suspicious activity records
        for (const [key, activities] of this.suspiciousActivity.entries()) {
            const filtered = activities.filter(
                a => now - a.timestamp < 3600000 // Keep last hour
            );
            
            if (filtered.length === 0) {
                this.suspiciousActivity.delete(key);
            } else {
                this.suspiciousActivity.set(key, filtered);
            }
        }
    }
    
    /**
     * Get client IP address
     */
    getClientIP(socket) {
        return socket.handshake.headers['x-forwarded-for']?.split(',')[0] || 
               socket.handshake.address;
    }
    
    /**
     * Validate origin
     */
    isValidOrigin(origin) {
        if (process.env.NODE_ENV === 'development') {
            return true; // Allow all in development
        }
        
        const allowedOrigins = process.env.WS_ALLOWED_ORIGINS?.split(',') || [];
        return allowedOrigins.includes(origin);
    }
    
    /**
     * Log security event
     */
    logSecurityEvent(type, details) {
        cds.log('ws-security').warn('Security event', {
            type,
            details,
            timestamp: new Date()
        });
    }
    
    /**
     * Get connection statistics
     */
    getStats() {
        return {
            totalConnections: this.connections.size,
            connectionsByIP: Object.fromEntries(
                Array.from(this.ipConnections.entries())
                    .map(([ip, sockets]) => [ip, sockets.size])
            ),
            connectionsByUser: Object.fromEntries(
                Array.from(this.userConnections.entries())
                    .map(([userId, sockets]) => [userId, sockets.size])
            ),
            suspiciousActivities: this.suspiciousActivity.size
        };
    }
}

// Export singleton instance
const wsSecurityManager = new WebSocketSecurityManager();

module.exports = {
    wsSecurityManager,
    createAuthMiddleware: (publicKey) => wsSecurityManager.createAuthMiddleware(publicKey),
    getStats: () => wsSecurityManager.getStats(),
    WS_SECURITY_CONFIG
};