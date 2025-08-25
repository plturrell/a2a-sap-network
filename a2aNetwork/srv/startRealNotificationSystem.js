#!/usr/bin/env node

/**
 * A2A Protocol Compliance: WebSocket replaced with blockchain event streaming
 * All real-time communication now uses blockchain events instead of WebSockets
 */

/**
 * Startup Script for Real A2A Notification System
 * Connects to actual A2A services with proper environment configuration
 */

const dotenv = require('dotenv');
const path = require('path');
const { exec } = require('child_process');
const fs = require('fs').promises;

// Load environment variables
dotenv.config();

class RealNotificationSystemStarter {
    constructor() {
        this.logger = console;
        this.services = {};
        this.requiredServices = [
            'agent_registry',
            'blockchain_service',
            'security_monitoring'
        ];
    }

    async start() {
        try {
            this.logger.info('🚀 Starting Real A2A Notification System...');

            // Check environment configuration
            await this.checkEnvironment();

            // Verify real services are running
            await this.verifyServices();

            // Start notification system components
            await this.startComponents();

            this.logger.info('✅ Real A2A Notification System is LIVE!');
            this.printServiceStatus();

        } catch (error) {
            this.logger.error('❌ Failed to start notification system:', error.message);
            process.exit(1);
        }
    }

    async checkEnvironment() {
        this.logger.info('🔧 Checking environment configuration...');

        const requiredVars = {
            'A2A_REGISTRY_URL': process.env.A2A_REGISTRY_URL || 'http://localhost:8000',
            'A2A_REGISTRY_WS': process.env.A2A_REGISTRY_WS || 'blockchain://a2a-events',
            'BLOCKCHAIN_SERVICE_URL': process.env.BLOCKCHAIN_SERVICE_URL || 'http://localhost:8080/blockchain',
            'BLOCKCHAIN_WS': process.env.BLOCKCHAIN_WS || 'blockchain://a2a-events',
            'SECURITY_API_URL': process.env.SECURITY_API_URL || 'http://localhost:8001/security',
            'SECURITY_EVENTS_URL': process.env.SECURITY_EVENTS_URL || 'http://localhost:8001/security/events',
            'METRICS_WS': process.env.METRICS_WS || 'blockchain://a2a-events'
        };

        // Set environment variables if not already set
        for (const [key, defaultValue] of Object.entries(requiredVars)) {
            if (!process.env[key]) {
                process.env[key] = defaultValue;
                this.logger.warn(`⚠️  Using default for ${key}: ${defaultValue}`);
            } else {
                this.logger.info(`✅ ${key}: ${process.env[key]}`);
            }
        }

        // Create .env file if it doesn't exist
        await this.createEnvironmentFile(requiredVars);
    }

    async createEnvironmentFile(vars) {
        const envFile = path.join(__dirname, '..', '.env');

        try {
            await fs.access(envFile);
            this.logger.info('✅ .env file exists');
        } catch {
            this.logger.info('📝 Creating .env file with default configuration...');

            const envContent = Object.entries(vars)
                .map(([key, value]) => `${key}=${value}`)
                .join('\n');

            await fs.writeFile(envFile, envContent + '\n');
            this.logger.info('✅ .env file created');
        }
    }

    async verifyServices() {
        this.logger.info('🔍 Verifying real A2A services...');

        const { BlockchainClient } = require('../core/blockchain-client');
        const { BlockchainEventServer, BlockchainEventClient } = require('./blockchain-event-adapter');

        const serviceChecks = [
            {
                name: 'Agent Registry',
                type: 'http',
                url: process.env.A2A_REGISTRY_URL + '/agents',
                required: true
            },
            {
                name: 'Agent Registry WebSocket',
                type: 'ws',
                url: process.env.A2A_REGISTRY_WS + '/agents',
                required: true
            },
            {
                name: 'Blockchain Service',
                type: 'http',
                url: process.env.BLOCKCHAIN_SERVICE_URL + '/status',
                required: true
            },
            {
                name: 'Blockchain WebSocket',
                type: 'ws',
                url: process.env.BLOCKCHAIN_WS,
                required: false // Optional for initial startup
            },
            {
                name: 'Security API',
                type: 'http',
                url: process.env.SECURITY_API_URL + '/health',
                required: false // Optional
            },
            {
                name: 'Metrics WebSocket',
                type: 'ws',
                url: process.env.METRICS_WS,
                required: false // Optional
            }
        ];

        for (const service of serviceChecks) {
            try {
                if (service.type === 'http') {
                    const response = await blockchainClient.sendMessage(service.url, {
                        timeout: 5000,
                        validateStatus: status => status < 500
                    });
                    this.logger.info(`✅ ${service.name}: Connected (${response.status})`);
                    this.services[service.name] = 'connected';
                } else if (service.type === 'ws') {
                    await this.testWebSocketConnection(service.url);
                    this.logger.info(`✅ ${service.name}: WebSocket accessible`);
                    this.services[service.name] = 'accessible';
                }
            } catch (error) {
                if (service.required) {
                    this.logger.error(`❌ ${service.name}: ${error.message}`);
                    throw new Error(`Required service ${service.name} is not available`);
                } else {
                    this.logger.warn(`⚠️  ${service.name}: Not available (${error.message}) - will continue without it`);
                    this.services[service.name] = 'unavailable';
                }
            }
        }
    }

    async testWebSocketConnection(url) {
        return new Promise((resolve, reject) => {
            const ws = new BlockchainEventClient(url);
            const timeout = setTimeout(() => {
                ws.terminate();
                reject(new Error('WebSocket connection timeout'));
            }, 5000);

            ws.on('open', () => {
                clearTimeout(timeout);
                ws.close();
                resolve();
            });

            ws.on('error', (error) => {
                clearTimeout(timeout);
                reject(error);
            });
        });
    }

    async startComponents() {
        this.logger.info('🎯 Starting notification system components...');

        // Start the integrated notification service
        const IntegratedNotificationService = require('./integratedNotificationService');
        this.notificationService = new IntegratedNotificationService();

        // Wait for initialization
        let attempts = 0;
        const maxAttempts = 30;

        while (!this.notificationService.isInitialized && attempts < maxAttempts) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            attempts++;
        }

        if (!this.notificationService.isInitialized) {
            throw new Error('Notification service failed to initialize within timeout');
        }

        this.logger.info('✅ Integrated notification service started');

        // Set up graceful shutdown
        this.setupGracefulShutdown();
    }

    setupGracefulShutdown() {
        const shutdown = async (signal) => {
            this.logger.info(`\n🛑 Received ${signal}, shutting down gracefully...`);

            if (this.notificationService) {
                await this.notificationService.shutdown();
            }

            this.logger.info('✅ Shutdown complete');
            process.exit(0);
        };

        process.on('SIGINT', () => shutdown('SIGINT'));
        process.on('SIGTERM', () => shutdown('SIGTERM'));
        process.on('uncaughtException', (error) => {
            this.logger.error('Uncaught Exception:', error);
            shutdown('UNCAUGHT_EXCEPTION');
        });
        process.on('unhandledRejection', (reason, promise) => {
            this.logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
            shutdown('UNHANDLED_REJECTION');
        });
    }

    printServiceStatus() {
        this.logger.info('\n📊 Service Status:');
        this.logger.info('=====================================');

        for (const [service, status] of Object.entries(this.services)) {
            const emoji = status === 'connected' ? '✅' :
                         status === 'accessible' ? '🔗' : '⚠️';
            this.logger.info(`${emoji} ${service}: ${status}`);
        }

        this.logger.info('=====================================');
        this.logger.info('🎉 Real notification system is running!');
        this.logger.info('📱 WebSocket endpoints:');
        this.logger.info(`   • Enhanced Notifications: blockchain://localhost:4006/notifications/v2`);
        this.logger.info(`   • Event Bus: blockchain://localhost:8080/events`);
        this.logger.info('🌐 REST API endpoints:');
        this.logger.info('   • Notifications: http://localhost:4000/api/notifications');
        this.logger.info('   • Push VAPID Key: http://localhost:4000/api/push/vapid-public-key');
        this.logger.info('   • Service Worker: http://localhost:4000/api/push/service-worker.js');
        this.logger.info('\n💡 Connect your frontend to these endpoints for real-time notifications!');
    }
}

// Start the system
if (require.main === module) {
    const starter = new RealNotificationSystemStarter();
    starter.start().catch(error => {
        console.error('Failed to start:', error);
        process.exit(1);
    });
}

module.exports = RealNotificationSystemStarter;