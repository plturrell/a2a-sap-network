"use strict";

/**
 * SAP API Gateway Implementation
 * Enterprise-grade API Gateway pattern for A2A Platform
 */

const express = require('express');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const { createProxyMiddleware } = require('http-proxy-middleware');
const jwt = require('jsonwebtoken');
const { OpenTelemetryInstrumentation: _OpenTelemetryInstrumentation } = require('@opentelemetry/instrumentation-express');
const winston = require('winston');
const prometheus = require('prom-client');

class A2AAPIGateway {
    constructor(config) {
        this.app = express();
        this.config = config;
        this.logger = this.setupLogging();
        this.metrics = this.setupMetrics();
        this.setupMiddleware();
        this.setupRoutes();
    }

    setupLogging() {
        return winston.createLogger({
            level: 'info',
            format: winston.format.json(),
            transports: [
                new winston.transports.Console(),
                new winston.transports.File({ filename: 'api-gateway.log' })
            ]
        });
    }

    setupMetrics() {
        const register = new prometheus.Registry();
        
        // Define metrics
        const httpRequestDuration = new prometheus.Histogram({
            name: 'http_request_duration_seconds',
            help: 'Duration of HTTP requests in seconds',
            labelNames: ['method', 'route', 'status_code'],
            registers: [register]
        });

        const httpRequestTotal = new prometheus.Counter({
            name: 'http_requests_total',
            help: 'Total number of HTTP requests',
            labelNames: ['method', 'route', 'status_code'],
            registers: [register]
        });

        prometheus.collectDefaultMetrics({ register });

        return { register, httpRequestDuration, httpRequestTotal };
    }

    setupMiddleware() {
        // Security headers
        this.app.use(helmet({
            contentSecurityPolicy: {
                directives: {
                    defaultSrc: ["'self'"],
                    styleSrc: ["'self'", "'unsafe-inline'"],
                    scriptSrc: ["'self'"],
                    imgSrc: ["'self'", "data:", "https:"],
                }
            },
            hsts: {
                maxAge: 31536000,
                includeSubDomains: true,
                preload: true
            }
        }));

        // Request ID generation
        this.app.use((req, res, next) => {
            req.id = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            res.setHeader('X-Request-ID', req.id);
            next();
        });

        // Logging middleware
        this.app.use((req, res, next) => {
            const start = Date.now();
            
            res.on('finish', () => {
                const duration = Date.now() - start;
                
                this.logger.info({
                    request_id: req.id,
                    method: req.method,
                    path: req.path,
                    status: res.statusCode,
                    duration_ms: duration,
                    user_agent: req.get('user-agent'),
                    ip: req.ip
                });

                // Update metrics
                this.metrics.httpRequestDuration
                    .labels(req.method, req.route?.path || req.path, res.statusCode)
                    .observe(duration / 1000);
                    
                this.metrics.httpRequestTotal
                    .labels(req.method, req.route?.path || req.path, res.statusCode)
                    .inc();
            });
            
            next();
        });

        // Rate limiting
        const createRateLimiter = (requests, windowMs) => {
            return rateLimit({
                windowMs,
                max: requests,
                message: 'Too many requests from this IP, please try again later.',
                standardHeaders: true,
                legacyHeaders: false,
            });
        };

        // Apply different rate limits for different APIs
        this.app.use('/api/v1/registry', createRateLimiter(1000, 60 * 1000));
        this.app.use('/api/v1/agents', createRateLimiter(500, 60 * 1000));
        this.app.use('/api/v1/orchestration', createRateLimiter(200, 60 * 1000));

        // Authentication middleware
        this.app.use('/api', this.authenticateRequest.bind(this));

        // Request transformation
        this.app.use(this.transformRequest.bind(this));
    }

    authenticateRequest(req, res, next) {
        const token = req.headers.authorization?.split(' ')[1];
        
        if (!token) {
            return res.status(401).json({
                error: 'Authentication required',
                request_id: req.id
            });
        }

        try {
            // Verify JWT token with XSUAA
            const decoded = jwt.verify(token, process.env.XSUAA_PUBLIC_KEY);
            req.user = decoded;
            next();
        } catch (error) {
            this.logger.error({
                request_id: req.id,
                error: 'Authentication failed',
                details: error.message
            });
            
            return res.status(401).json({
                error: 'Invalid or expired token',
                request_id: req.id
            });
        }
    }

    transformRequest(req, _res, next) {
        // Add gateway headers
        req.headers['x-gateway-request-id'] = req.id;
        req.headers['x-gateway-timestamp'] = new Date().toISOString();
        req.headers['x-forwarded-for'] = req.ip;
        
        // Remove internal headers
        delete req.headers['x-internal-token'];
        
        next();
    }

    setupRoutes() {
        // Health check endpoint
        this.app.get('/health', (req, res) => {
            res.json({
                status: 'healthy',
                timestamp: new Date().toISOString(),
                uptime: process.uptime()
            });
        });

        // Metrics endpoint
        this.app.get('/metrics', async (req, res) => {
            res.set('Content-Type', this.metrics.register.contentType);
            res.end(await this.metrics.register.metrics());
        });

        // API Documentation
        this.app.get('/api-docs', (req, res) => {
            res.json({
                openapi: '3.0.0',
                info: {
                    title: 'A2A Platform API',
                    version: '1.0.0',
                    description: 'Enterprise API Gateway for A2A Platform'
                },
                servers: [
                    {
                        url: 'https://api.a2a-platform.sap.com',
                        description: 'Production API Gateway'
                    }
                ],
                paths: {
                    '/api/v1/registry': {
                        get: { summary: 'Registry API endpoints' }
                    },
                    '/api/v1/agents': {
                        get: { summary: 'Agent management endpoints' }
                    },
                    '/api/v1/orchestration': {
                        get: { summary: 'Orchestration endpoints' }
                    }
                }
            });
        });

        // Proxy configuration for backend services
        const proxyOptions = {
            changeOrigin: true,
            onProxyReq: (proxyReq, req, _res) => {
                // Add correlation headers
                proxyReq.setHeader('X-Correlation-ID', req.id);
            },
            onProxyRes: (proxyRes, req, _res) => {
                // Transform response
                delete proxyRes.headers['x-powered-by'];
                proxyRes.headers['x-response-time'] = Date.now() - req.startTime;
            },
            onError: (err, req, res) => {
                this.logger.error({
                    request_id: req.id,
                    error: 'Proxy error',
                    details: err.message
                });
                
                res.status(502).json({
                    error: 'Bad Gateway',
                    request_id: req.id
                });
            }
        };

        // Registry API proxy
        this.app.use('/api/v1/registry', createProxyMiddleware({
            target: 'http://a2a-registry-service:8080',
            ...proxyOptions
        }));

        // Agent API proxy
        this.app.use('/api/v1/agents', createProxyMiddleware({
            target: 'http://a2a-agent-service:8081',
            ...proxyOptions
        }));

        // Orchestration API proxy
        this.app.use('/api/v1/orchestration', createProxyMiddleware({
            target: 'http://a2a-orchestration-service:8082',
            ...proxyOptions
        }));

        // Error handling
        this.app.use((err, req, res, _next) => {
            this.logger.error({
                request_id: req.id,
                error: err.message,
                stack: err.stack
            });

            res.status(err.status || 500).json({
                error: 'Internal Server Error',
                request_id: req.id,
                message: process.env.NODE_ENV === 'development' ? err.message : undefined
            });
        });
    }

    start(port = 3000) {
        this.app.listen(port, () => {
            this.logger.info(`API Gateway running on port ${port}`);
        });
    }
}

// Circuit breaker implementation
class CircuitBreaker {
    constructor(options = {}) {
        this.threshold = options.threshold || 5;
        this.timeout = options.timeout || 60000;
        this.resetTimeout = options.resetTimeout || 60000;
        
        this.failureCount = 0;
        this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
        this.nextAttempt = Date.now();
    }

    async execute(fn) {
        if (this.state === 'OPEN') {
            if (Date.now() < this.nextAttempt) {
                throw new Error('Circuit breaker is OPEN');
            }
            this.state = 'HALF_OPEN';
        }

        try {
            const result = await fn();
            this.onSuccess();
            return result;
        } catch (error) {
            this.onFailure();
            throw error;
        }
    }

    onSuccess() {
        this.failureCount = 0;
        this.state = 'CLOSED';
    }

    onFailure() {
        this.failureCount++;
        if (this.failureCount >= this.threshold) {
            this.state = 'OPEN';
            this.nextAttempt = Date.now() + this.resetTimeout;
        }
    }
}

// Export for use
module.exports = { A2AAPIGateway, CircuitBreaker };

// Start the gateway if run directly
if (require.main === module) {
    const gateway = new A2AAPIGateway({
        port: process.env.GATEWAY_PORT || 3000
    });
    gateway.start();
}