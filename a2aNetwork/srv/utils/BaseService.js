const { v4: uuidv4 } = require('uuid');

class BaseService {
    constructor(options = {}) {
        this.serviceId = options.serviceId || uuidv4();
        this.serviceName = options.serviceName || this.constructor.name;
        this.logger = options.logger || console;
        this.config = options.config || {};
        this.initialized = false;
        this.startupTime = null;
    }

    async initializeService() {
        if (this.initialized) {
            return this;
        }

        this.startupTime = new Date();
        
        if (this._validateConfiguration) {
            await this._validateConfiguration();
        }
        
        if (this._initializeComponents) {
            await this._initializeComponents();
        }
        
        this.initialized = true;
        this.logger.info(`${this.serviceName} initialized successfully`);
        return this;
    }

    async shutdown() {
        if (this._cleanup) {
            await this._cleanup();
        }
        this.initialized = false;
        this.logger.info(`${this.serviceName} shut down successfully`);
    }

    getServiceInfo() {
        return {
            serviceId: this.serviceId,
            serviceName: this.serviceName,
            initialized: this.initialized,
            startupTime: this.startupTime,
            uptime: this.startupTime ? Date.now() - this.startupTime.getTime() : 0
        };
    }

    validateInput(input, schema) {
        if (!input) {
            throw new Error('Input is required');
        }
        return true;
    }

    handleError(error, context = {}) {
        const errorInfo = {
            service: this.serviceName,
            serviceId: this.serviceId,
            error: error.message,
            context,
            timestamp: new Date().toISOString()
        };
        
        this.logger.error('Service error:', errorInfo);
        throw error;
    }

    async healthCheck() {
        return {
            status: this.initialized ? 'healthy' : 'not_initialized',
            service: this.serviceName,
            serviceId: this.serviceId,
            uptime: this.getServiceInfo().uptime,
            timestamp: new Date().toISOString()
        };
    }
}

module.exports = BaseService;