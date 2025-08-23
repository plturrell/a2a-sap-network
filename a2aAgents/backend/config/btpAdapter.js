"use strict";

/**
 * BTP Adapter Layer - Smart Integration Without Code Changes
 * Adapts existing A2A system to work with BTP services when available
 */

const os = require('os');

class BTPAdapter {
    constructor() {
        this.isBTP = this._detectBTPEnvironment();
        this.services = this._loadServices();
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log(`🎯 BTP Adapter initialized - Environment: ${this.isBTP ? 'BTP' : 'Local'}`);
    }

    _detectBTPEnvironment() {
        return !!(process.env.VCAP_SERVICES || process.env.VCAP_APPLICATION);
    }

    _loadServices() {
        if (this.isBTP) {
            return this._loadBTPServices();
        } else {
            return this._loadLocalServices();
        }
    }

    _loadBTPServices() {
        try {
            const vcapServices = JSON.parse(process.env.VCAP_SERVICES || '{}');
            // eslint-disable-next-line no-console
            // eslint-disable-next-line no-console
            console.log('✅ Loading BTP service bindings');
            
            const services = {};

            // HANA Database
            const hanaServices = vcapServices.hana || vcapServices.hanatrial || [];
            if (hanaServices.length > 0) {
                const creds = hanaServices[0].credentials;
                services.hana = {
                    host: creds.host,
                    port: creds.port || 443,
                    user: creds.user,
                    password: creds.password,
                    database: creds.database,
                    schema: creds.schema || 'A2A_AGENTS',
                    encrypt: true,
                    sslValidateCertificate: true
                };
                // eslint-disable-next-line no-console
                // eslint-disable-next-line no-console
                console.log('✅ HANA service bound');
            }

            // XSUAA Authentication
            const xsuaaServices = vcapServices.xsuaa || [];
            if (xsuaaServices.length > 0) {
                const creds = xsuaaServices[0].credentials;
                services.xsuaa = {
                    url: creds.url,
                    clientId: creds.clientid,
                    clientSecret: creds.clientsecret,
                    xsAppName: creds.xsappname
                };
                // eslint-disable-next-line no-console
                // eslint-disable-next-line no-console
                console.log('✅ XSUAA service bound');
            }

            // Redis Cache (optional)
            const redisServices = vcapServices['redis-cache'] || [];
            if (redisServices.length > 0) {
                const creds = redisServices[0].credentials;
                services.redis = {
                    host: creds.hostname,
                    port: creds.port || 6379,
                    password: creds.password
                };
                // eslint-disable-next-line no-console
                // eslint-disable-next-line no-console
                console.log('✅ Redis service bound');
            }

            return services;

        } catch (error) {
            console.warn('⚠️ Failed to load BTP services, falling back to local:', error.message);
            return this._loadLocalServices();
        }
    }

    _loadLocalServices() {
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log('🔧 Loading local development configuration');
        
        return {
            hana: {
                host: process.env.HANA_HOST || 'localhost',
                port: parseInt(process.env.HANA_PORT) || 30015,
                user: process.env.HANA_USER || 'SYSTEM',
                password: process.env.HANA_PASSWORD || '',
                database: process.env.HANA_DATABASE || 'A2A',
                schema: process.env.HANA_SCHEMA || 'A2A_AGENTS',
                encrypt: process.env.HANA_ENCRYPT === 'true',
                sslValidateCertificate: false
            },
            xsuaa: {
                url: process.env.XSUAA_URL || 'http://localhost:8080/uaa',
                clientId: process.env.XSUAA_CLIENT_ID || 'local-client',
                clientSecret: process.env.XSUAA_CLIENT_SECRET || 'local-secret',
                xsAppName: process.env.XSUAA_XSAPPNAME || 'a2a-agents-local'
            },
            redis: process.env.REDIS_HOST ? {
                host: process.env.REDIS_HOST,
                port: parseInt(process.env.REDIS_PORT) || 6379,
                password: process.env.REDIS_PASSWORD
            } : null
        };
    }

    /**
     * Get database configuration in format expected by existing A2A code
     */
    getDatabaseConfig() {
        const hana = this.services.hana;
        if (!hana) {
            throw new Error('HANA configuration not available');
        }

        // Return in format that existing A2A code expects
        return {
            // Standard SAP format
            ...hana,
            // Add aliases for different naming conventions used in A2A code
            hostname: hana.host,
            sql_port: hana.port,
            hdi_user: hana.user,
            hdi_password: hana.password
        };
    }

    /**
     * Get authentication configuration for existing A2A auth middleware
     */
    getAuthConfig() {
        if (this.isBTP) {
            return {
                type: 'xsuaa',
                ...this.services.xsuaa,
                // Add flags for existing auth code
                enabled: true,
                development: false
            };
        } else {
            return {
                type: 'local',
                enabled: process.env.BYPASS_AUTH !== 'false',
                development: true,
                // Mock XSUAA for local development
                ...this.services.xsuaa
            };
        }
    }

    /**
     * Get Redis configuration for existing caching code
     */
    getCacheConfig() {
        const redis = this.services.redis;
        if (!redis) {
            return null;
        }

        return {
            enabled: true,
            ...redis,
            // Add options for existing Redis code
            retry_on_timeout: true,
            socket_keepalive: true,
            socket_keepalive_options: {}
        };
    }

    /**
     * Create environment variables that existing A2A code expects
     * This allows existing code to work without changes
     */
    injectEnvironmentVariables() {
        const hana = this.services.hana;
        const xsuaa = this.services.xsuaa;
        const redis = this.services.redis;

        if (hana) {
            process.env.HANA_HOST = hana.host;
            process.env.HANA_PORT = hana.port.toString();
            process.env.HANA_USER = hana.user;
            process.env.HANA_PASSWORD = hana.password;
            process.env.HANA_DATABASE = hana.database || 'A2A';
            process.env.HANA_SCHEMA = hana.schema || 'A2A_AGENTS';
            process.env.HANA_ENCRYPT = hana.encrypt ? 'true' : 'false';
        }

        if (xsuaa) {
            process.env.XSUAA_URL = xsuaa.url;
            process.env.XSUAA_CLIENT_ID = xsuaa.clientId;
            process.env.XSUAA_CLIENT_SECRET = xsuaa.clientSecret;
            process.env.XSUAA_XSAPPNAME = xsuaa.xsAppName;
        }

        if (redis) {
            process.env.REDIS_HOST = redis.host;
            process.env.REDIS_PORT = redis.port.toString();
            if (redis.password) {
                process.env.REDIS_PASSWORD = redis.password;
            }
        }

        // Set environment indicators
        process.env.BTP_ENVIRONMENT = this.isBTP ? 'true' : 'false';
        process.env.DEVELOPMENT_MODE = this.isBTP ? 'false' : 'true';

         

        // eslint-disable-next-line no-console

         

        // eslint-disable-next-line no-console
        console.log('✅ Environment variables injected for existing A2A code compatibility');
    }

    /**
     * Get application information from BTP or local environment
     */
    getApplicationInfo() {
        if (this.isBTP && process.env.VCAP_APPLICATION) {
            const vcapApp = JSON.parse(process.env.VCAP_APPLICATION);
            return {
                name: vcapApp.name || 'a2a-agents',
                version: vcapApp.version || '1.0.0',
                uris: vcapApp.uris || [],
                space_name: vcapApp.space_name,
                organization_name: vcapApp.organization_name,
                instance_id: vcapApp.instance_id,
                instance_index: vcapApp.instance_index || 0,
                port: vcapApp.port || process.env.PORT || 8080
            };
        } else {
            return {
                name: 'a2a-agents-local',
                version: '1.0.0',
                uris: [`localhost:${process.env.PORT || 8080}`],
                space_name: 'local',
                organization_name: 'development',
                instance_id: `local-${  os.hostname()}`,
                instance_index: 0,
                port: process.env.PORT || 8080
            };
        }
    }

    /**
     * Health check that works with existing A2A monitoring
     */
    getHealthStatus() {
        return {
            status: 'healthy',
            environment: this.isBTP ? 'btp' : 'local',
            timestamp: new Date().toISOString(),
            services: {
                database: !!this.services.hana,
                authentication: !!this.services.xsuaa,
                cache: !!this.services.redis
            },
            btp: {
                detected: this.isBTP,
                vcap_services: !!process.env.VCAP_SERVICES,
                vcap_application: !!process.env.VCAP_APPLICATION
            }
        };
    }

    /**
     * Initialize the adapter and inject variables for existing code
     */
    initialize() {
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log('🚀 Initializing BTP Adapter');
        
        // Inject environment variables so existing A2A code works unchanged
        this.injectEnvironmentVariables();
        
        // Log configuration summary
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log('📋 Configuration Summary:');
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log(`   Database: ${this.services.hana ? '✅' : '❌'} ${this.services.hana?.host || 'N/A'}`);
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log(`   Auth: ${this.services.xsuaa ? '✅' : '❌'} ${this.services.xsuaa?.url || 'N/A'}`);
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log(`   Cache: ${this.services.redis ? '✅' : '❌'} ${this.services.redis?.host || 'N/A'}`);
        
        return this;
    }
}

// Export singleton instance
const btpAdapter = new BTPAdapter();

module.exports = {
    BTPAdapter,
    btpAdapter
};