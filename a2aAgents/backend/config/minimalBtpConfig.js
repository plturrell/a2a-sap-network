"use strict";

/**
 * Minimal BTP Configuration - JavaScript Version
 * Local First, BTP Optional - Simple approach that works locally and adds BTP services when available
 */

class MinimalBTPConfig {
    constructor() {
        this.is_btp = this._detectBTPEnvironment();
        this.services = this._loadServices();
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log(`🎯 Minimal BTP Config initialized - Environment: ${this.is_btp ? 'BTP' : 'Local'}`);
    }

    _detectBTPEnvironment() {
        return !!(process.env.VCAP_SERVICES || process.env.VCAP_APPLICATION);
    }

    _loadServices() {
        if (this.is_btp) {
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
            console.log('✅ Running on BTP - using service bindings');
            
            const services = {};

            // HANA - get first available
            const hanaServices = vcapServices.hana || vcapServices.hanatrial || [];
            if (hanaServices.length > 0) {
                const creds = hanaServices[0].credentials;
                services.hana = {
                    host: creds.host,
                    port: creds.port || 443,
                    user: creds.user,
                    password: creds.password,
                    encrypt: true,
                    schema: creds.schema || 'A2A_AGENTS'
                };
            }

            // XSUAA - get first available
            const xsuaaServices = vcapServices.xsuaa || [];
            if (xsuaaServices.length > 0) {
                const creds = xsuaaServices[0].credentials;
                services.xsuaa = {
                    url: creds.url,
                    client_id: creds.clientid,
                    client_secret: creds.clientsecret
                };
            }

            // Redis - optional
            const redisServices = vcapServices['redis-cache'] || [];
            if (redisServices.length > 0) {
                const creds = redisServices[0].credentials;
                services.redis = {
                    host: creds.hostname,
                    port: creds.port || 6379,
                    password: creds.password
                };
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
        console.log('🔧 Running locally - using environment variables');
        
        const services = {
            hana: {
                host: process.env.HANA_HOST || 'localhost',
                port: parseInt(process.env.HANA_PORT) || 30015,
                user: process.env.HANA_USER || 'SYSTEM',
                password: process.env.HANA_PASSWORD || '',
                encrypt: false, // Local development
                schema: process.env.HANA_SCHEMA || 'A2A_AGENTS'
            },
            xsuaa: {
                url: process.env.XSUAA_URL || 'http://localhost:8080',
                client_id: process.env.XSUAA_CLIENT_ID || 'local-client',
                client_secret: process.env.XSUAA_CLIENT_SECRET || 'local-secret'
            }
        };

        // Redis - only if configured
        if (process.env.REDIS_HOST) {
            services.redis = {
                host: process.env.REDIS_HOST,
                port: parseInt(process.env.REDIS_PORT) || 6379,
                password: process.env.REDIS_PASSWORD || ''
            };
        }

        return services;
    }

    getHanaConfig() {
        const hana = this.services.hana;
        if (!hana) {
            throw new Error('HANA configuration not available');
        }
        return hana;
    }

    getAuthConfig() {
        if (this.is_btp) {
            return this.services.xsuaa || {};
        } else {
            // Local development - simple auth
            return {
                local_mode: true,
                bypass_auth: (process.env.BYPASS_AUTH || 'true').toLowerCase() === 'true'
            };
        }
    }

    getCacheConfig() {
        return this.services.redis || null;
    }

    is_service_available(serviceName) {
        return serviceName in this.services && this.services[serviceName] !== null;
    }
}

module.exports = { MinimalBTPConfig };