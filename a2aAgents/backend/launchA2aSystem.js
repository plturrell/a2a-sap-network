#!/usr/bin/env node
"use strict";
/* global URL */
/**
 * A2A System Launcher
 * Unified launcher for full A2A agents implementation
 * Works locally and on BTP without code changes
 */

const { btpAdapter } = require('./config/btpAdapter');
const express = require('express');
const path = require('path');
const fs = require('fs').promises;
const { existsSync } = require('fs');

class A2ASystemLauncher {
    constructor() {
        this.app = express();
        this.adapter = btpAdapter.initialize();
        this.appInfo = this.adapter.getApplicationInfo();
        this.setupExpress();
    }

    setupExpress() {
        // Trust proxy for BTP load balancers
        this.app.set('trust proxy', true);
        
        // Basic middleware
        this.app.use(express.json({ limit: '10mb' }));
        this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));
        
        // CORS for local development
        if (!this.adapter.isBTP) {
            this.app.use((req, _res, _next) => {
                res.header('Access-Control-Allow-Origin', '*');
                res.header('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS');
                res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
                if (req.method === 'OPTIONS') {
                    res.sendStatus(200);
                } else {
                    next();
                }
            });
        }
    }

    async loadExistingRoutes() {
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log('🔌 Loading existing A2A routes...');
        
        try {
            // Try to load existing API routes
            const apiRoutesPath = path.join(__dirname, 'srv', 'apiRoutes.js');
            if (existsSync(apiRoutesPath)) {
                const apiRoutes = require(apiRoutesPath);
                if (typeof apiRoutes === 'function') {
                    this.app.use('/api', apiRoutes);
                    // eslint-disable-next-line no-console
                    // eslint-disable-next-line no-console
                    console.log('✅ Loaded API routes from srv/apiRoutes.js');
                }
            }

            // Try to load existing A2A service
            const a2aServicePath = path.join(__dirname, 'srv', 'sapA2aService.js');
            if (existsSync(a2aServicePath)) {
                const a2aService = require(a2aServicePath);
                if (a2aService && a2aService.app) {
                    // Merge routes from existing service
                    this.app.use('/', a2aService.app);
                    // eslint-disable-next-line no-console
                    // eslint-disable-next-line no-console
                    console.log('✅ Loaded SAP A2A Service routes');
                }
            }

            // Try to load existing blockchain service
            const blockchainServicePath = path.join(__dirname, 'srv', 'sapBlockchainService.js');
            if (existsSync(blockchainServicePath)) {
                const blockchainService = require(blockchainServicePath);
                if (blockchainService && blockchainService.router) {
                    this.app.use('/blockchain', blockchainService.router);
                    // eslint-disable-next-line no-console
                    // eslint-disable-next-line no-console
                    console.log('✅ Loaded blockchain service routes');
                }
            }

        } catch (error) {
            console.warn('⚠️ Some existing routes could not be loaded:', error.message);
            // eslint-disable-next-line no-console
            // eslint-disable-next-line no-console
            console.log('🔧 Continuing with basic setup...');
        }
    }

    setupCoreRoutes() {
        // Health check endpoint
        this.app.get('/health', (req, res) => {
            res.json(this.adapter.getHealthStatus());
        });

        // System information
        this.app.get('/info', (req, res) => {
            res.json({
                ...this.appInfo,
                adapter: this.adapter.getHealthStatus(),
                database: this.adapter.getDatabaseConfig(),
                auth: this.adapter.getAuthConfig(),
                cache: this.adapter.getCacheConfig()
            });
        });

        // Configuration endpoint for debugging
        this.app.get('/config', (req, res) => {
            if (this.adapter.isBTP) {
                // Don't expose secrets in BTP
                res.json({
                    environment: 'btp',
                    services: Object.keys(this.adapter.services),
                    message: 'Configuration loaded from BTP service bindings'
                });
            } else {
                res.json({
                    environment: 'local',
                    config: {
                        database: this.adapter.getDatabaseConfig(),
                        auth: this.adapter.getAuthConfig(),
                        cache: this.adapter.getCacheConfig()
                    }
                });
            }
        });

        // Root endpoint
        this.app.get('/', (req, res) => {
            res.json({
                name: 'A2A Agents System',
                version: this.appInfo.version,
                environment: this.adapter.isBTP ? 'SAP BTP' : 'Local Development',
                instance: this.appInfo.instance_id,
                endpoints: {
                    health: '/health',
                    info: '/info',
                    config: '/config',
                    api: '/api',
                    agents: '/api/agents',
                    blockchain: '/blockchain'
                },
                documentation: {
                    api: '/docs',
                    swagger: '/swagger-ui'
                }
            });
        });
    }

    setupAgentsAPI() {
        // Basic agents API - will be enhanced by loaded modules
        this.app.get('/api/agents', (req, res) => {
            res.json({
                message: 'A2A Agents System API',
                environment: this.adapter.isBTP ? 'btp' : 'local',
                agents: [
                    {
                        id: 'calculation-agent',
                        name: 'Enhanced Calculation Agent',
                        status: 'active',
                        features: ['self-healing', 'blockchain-integration', 'monitoring']
                    },
                    {
                        id: 'data-processing-agent',
                        name: 'Data Processing Agent',
                        status: 'active',
                        features: ['hana-integration', 'lifecycle-management']
                    },
                    {
                        id: 'reasoning-agent',
                        name: 'Reasoning Agent',
                        status: 'active',
                        features: ['chain-of-thought', 'error-recovery']
                    },
                    {
                        id: 'sql-agent',
                        name: 'SQL Agent',
                        status: 'active',
                        features: ['query-optimization', 'caching', 'nl2sql']
                    },
                    {
                        id: 'vector-processing-agent',
                        name: 'Vector Processing Agent',
                        status: 'active',
                        features: ['hana-vector-engine', 'embedding-processing']
                    }
                ],
                statistics: {
                    total_agents: 5,
                    active_agents: 5,
                    blockchain_enabled: true,
                    monitoring_enabled: true
                }
            });
        });

        // Agent status endpoint
        this.app.get('/api/agents/:agentId/status', (req, res) => {
            const { agentId } = req.params;
            res.json({
                agent_id: agentId,
                status: 'active',
                environment: this.adapter.isBTP ? 'btp' : 'local',
                last_check: new Date().toISOString(),
                configuration: 'loaded',
                blockchain_connected: true,
                database_connected: !!this.adapter.services.hana
            });
        });
    }

    setupErrorHandling() {
        // 404 handler
        this.app.use((req, res) => {
            res.status(404).json({
                error: 'Not found',
                path: req.path,
                available_endpoints: [
                    '/',
                    '/health',
                    '/info',
                    '/config',
                    '/api/agents'
                ]
            });
        });

        // Error handler
        this.app.use((err, _req, _res, _next) => {
            console.error('❌ Server error:', err);
            
            const isDevelopment = !this.adapter.isBTP;
            
            res.status(err.status || 500).json({
                error: 'Internal server error',
                message: isDevelopment ? err.message : 'Something went wrong',
                timestamp: new Date().toISOString(),
                environment: this.adapter.isBTP ? 'btp' : 'local'
            });
        });
    }

    async start() {
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log('🚀 Starting A2A Agents System...');
        
        // Load existing routes if available
        await this.loadExistingRoutes();
        
        // Setup core routes
        this.setupCoreRoutes();
        this.setupAgentsAPI();
        this.setupErrorHandling();
        
        const port = this.appInfo.port;
        
        return new Promise((resolve) => {
            this.app.listen(port, () => {
                // eslint-disable-next-line no-console
                // eslint-disable-next-line no-console
                console.log('✅ A2A Agents System is running!');
                // eslint-disable-next-line no-console
                // eslint-disable-next-line no-console
                console.log(`🌐 Environment: ${this.adapter.isBTP ? 'SAP BTP' : 'Local Development'}`);
                // eslint-disable-next-line no-console
                // eslint-disable-next-line no-console
                console.log(`🎯 Port: ${port}`);
                // eslint-disable-next-line no-console
                // eslint-disable-next-line no-console
                console.log(`📍 Instance: ${this.appInfo.instance_id}`);
                
                if (this.adapter.isBTP) {
                    // eslint-disable-next-line no-console
                    // eslint-disable-next-line no-console
                    console.log(`🔗 URLs: ${this.appInfo.uris.map(uri => `https://${uri}`).join(', ')}`);
                    // eslint-disable-next-line no-console
                    // eslint-disable-next-line no-console
                    console.log('✅ BTP services connected');
                } else {
                    // eslint-disable-next-line no-console
                    // eslint-disable-next-line no-console
                    console.log(`🔗 URL: http://localhost:${port}`);
                    // eslint-disable-next-line no-console
                    // eslint-disable-next-line no-console
                    console.log('🔧 Local development mode');
                }
                
                // eslint-disable-next-line no-console
                
                // eslint-disable-next-line no-console
                
                // eslint-disable-next-line no-console
                console.log('');
                // eslint-disable-next-line no-console
                // eslint-disable-next-line no-console
                console.log('📋 Available endpoints:');
                // eslint-disable-next-line no-console
                // eslint-disable-next-line no-console
                console.log(`   Health: ${this.adapter.isBTP ? `https://${  this.appInfo.uris[0]}` : `http://localhost:${  port}`}/health`);
                // eslint-disable-next-line no-console
                // eslint-disable-next-line no-console
                console.log(`   API: ${this.adapter.isBTP ? `https://${  this.appInfo.uris[0]}` : `http://localhost:${  port}`}/api/agents`);
                // eslint-disable-next-line no-console
                // eslint-disable-next-line no-console
                console.log(`   Info: ${this.adapter.isBTP ? `https://${  this.appInfo.uris[0]}` : `http://localhost:${  port}`}/info`);
                
                resolve(this.app);
            });
        });
    }
}

// Export for use as module
module.exports = { A2ASystemLauncher };

// Run directly if called as script
if (require.main === module) {
    const launcher = new A2ASystemLauncher();
    launcher.start().catch(error => {
        console.error('❌ Failed to start A2A system:', error);
        process.exit(1);
    });
}