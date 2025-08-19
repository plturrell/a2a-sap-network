const EventEmitter = require('eventemitter3');
const axios = require('axios');
const WebSocket = require('ws');
const { v4: uuidv4 } = require('uuid');
const Joi = require('joi');
const logger = require('pino')({ name: 'a2a-agent' });

const { Registry } = require('./Registry');
const { Blockchain } = require('./Blockchain');
const { MessageRouter } = require('./MessageRouter');
const { HealthMonitor } = require('./HealthMonitor');
const { SecurityManager } = require('./SecurityManager');

/**
 * A2A Agent - Core agent class for the A2A Framework
 */
class Agent extends EventEmitter {
  constructor(config) {
    super();
    
    this.config = this.validateConfig(config);
    this.id = this.config.id || uuidv4();
    this.services = new Map();
    this.middleware = [];
    this.status = 'created';
    this.metrics = {
      messagesReceived: 0,
      messagesSent: 0,
      servicesExecuted: 0,
      uptime: 0,
      lastActivity: null
    };
    
    // Initialize components
    this.registry = new Registry(this.config.registry);
    this.blockchain = this.config.blockchain?.enabled ? new Blockchain(this.config.blockchain) : null;
    this.messageRouter = new MessageRouter(this);
    this.healthMonitor = new HealthMonitor(this);
    this.security = new SecurityManager(this.config.security);
    
    this.startTime = Date.now();
    this.setupEventHandlers();
  }

  validateConfig(config) {
    const schema = Joi.object({
      name: Joi.string().required(),
      type: Joi.string().required(),
      capabilities: Joi.array().items(Joi.string()).default([]),
      id: Joi.string().optional(),
      port: Joi.number().min(1024).max(65535).default(3000),
      registry: Joi.object({
        url: Joi.string().uri().required(),
        apiKey: Joi.string().optional(),
        heartbeatInterval: Joi.number().default(30000)
      }).required(),
      blockchain: Joi.object({
        enabled: Joi.boolean().default(false),
        network: Joi.string().default('localhost'),
        rpcUrl: Joi.string().uri().when('enabled', { is: true, then: Joi.required() }),
        privateKey: Joi.string().when('enabled', { is: true, then: Joi.required() }),
        contractAddresses: Joi.object().default({})
      }).default({ enabled: false }),
      security: Joi.object({
        encryption: Joi.boolean().default(true),
        authentication: Joi.boolean().default(true),
        allowedOrigins: Joi.array().items(Joi.string()).default(['*'])
      }).default({}),
      logging: Joi.object({
        level: Joi.string().valid('trace', 'debug', 'info', 'warn', 'error').default('info'),
        pretty: Joi.boolean().default(false)
      }).default({})
    });

    const { error, value } = schema.validate(config);
    if (error) {
      throw new Error(`Invalid agent configuration: ${error.message}`);
    }

    return value;
  }

  setupEventHandlers() {
    this.on('message', this.handleMessage.bind(this));
    this.on('error', this.handleError.bind(this));
    
    // Metrics tracking
    this.on('service:called', () => {
      this.metrics.servicesExecuted++;
      this.metrics.lastActivity = Date.now();
    });
    
    this.on('message:received', () => {
      this.metrics.messagesReceived++;
      this.metrics.lastActivity = Date.now();
    });
    
    this.on('message:sent', () => {
      this.metrics.messagesSent++;
      this.metrics.lastActivity = Date.now();
    });
  }

  /**
   * Add a service to the agent
   */
  addService(name, handler, options = {}) {
    if (typeof handler !== 'function') {
      throw new Error('Service handler must be a function');
    }

    const serviceConfig = {
      name,
      handler,
      timeout: options.timeout || 30000,
      retries: options.retries || 0,
      description: options.description || '',
      schema: options.schema || null,
      middleware: options.middleware || []
    };

    this.services.set(name, serviceConfig);
    logger.info(`Service '${name}' added to agent '${this.config.name}'`);
    
    return this;
  }

  /**
   * Add middleware to the agent
   */
  use(middleware) {
    if (typeof middleware !== 'function') {
      throw new Error('Middleware must be a function');
    }
    
    this.middleware.push(middleware);
    return this;
  }

  /**
   * Start the agent
   */
  async start() {
    try {
      this.status = 'starting';
      this.emit('status:changed', 'starting');

      // Start health monitoring
      await this.healthMonitor.start();

      // Register with blockchain if enabled
      if (this.blockchain) {
        await this.blockchain.initialize();
        await this.blockchain.registerAgent({
          id: this.id,
          name: this.config.name,
          type: this.config.type,
          capabilities: this.config.capabilities,
          endpoint: `http://localhost:${this.config.port}`
        });
        logger.info(`Agent '${this.config.name}' registered on blockchain`);
      }

      // Start message router
      await this.messageRouter.start(this.config.port);

      // Register with registry
      await this.registry.register({
        id: this.id,
        name: this.config.name,
        type: this.config.type,
        capabilities: this.config.capabilities,
        services: Array.from(this.services.keys()),
        endpoint: `http://localhost:${this.config.port}`,
        status: 'running'
      });

      this.status = 'running';
      this.emit('status:changed', 'running');
      this.emit('started');
      
      logger.info(`Agent '${this.config.name}' started successfully on port ${this.config.port}`);
      
      return this;
    } catch (error) {
      this.status = 'error';
      this.emit('status:changed', 'error');
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Stop the agent
   */
  async stop() {
    try {
      this.status = 'stopping';
      this.emit('status:changed', 'stopping');

      // Unregister from registry
      await this.registry.unregister(this.id);

      // Stop message router
      await this.messageRouter.stop();

      // Stop health monitoring
      await this.healthMonitor.stop();

      // Unregister from blockchain if enabled
      if (this.blockchain) {
        await this.blockchain.unregisterAgent(this.id);
      }

      this.status = 'stopped';
      this.emit('status:changed', 'stopped');
      this.emit('stopped');
      
      logger.info(`Agent '${this.config.name}' stopped successfully`);
      
      return this;
    } catch (error) {
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Call a service on this agent
   */
  async call(serviceName, data, options = {}) {
    const service = this.services.get(serviceName);
    if (!service) {
      throw new Error(`Service '${serviceName}' not found`);
    }

    try {
      this.emit('service:called', { service: serviceName, data });

      // Validate input data if schema provided
      if (service.schema) {
        const { error } = service.schema.validate(data);
        if (error) {
          throw new Error(`Invalid input data: ${error.message}`);
        }
      }

      // Apply middleware
      let context = { data, options, agent: this };
      for (const middleware of [...this.middleware, ...service.middleware]) {
        context = await middleware(context) || context;
      }

      // Execute service with timeout
      const result = await this.executeWithTimeout(
        service.handler, 
        context.data, 
        service.timeout
      );

      this.emit('service:completed', { service: serviceName, result });
      return result;

    } catch (error) {
      this.emit('service:error', { service: serviceName, error });
      throw error;
    }
  }

  /**
   * Send a message to another agent
   */
  async sendMessage(targetAgentId, message, options = {}) {
    try {
      // Discover target agent
      const targetAgent = await this.registry.getAgent(targetAgentId);
      if (!targetAgent) {
        throw new Error(`Agent '${targetAgentId}' not found`);
      }

      // Send message via message router
      const response = await this.messageRouter.sendMessage(targetAgent, message, options);
      
      this.emit('message:sent', { target: targetAgentId, message, response });
      return response;

    } catch (error) {
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Discover agents by capability
   */
  async discover(capability, options = {}) {
    try {
      const agents = await this.registry.discover(capability, options);
      return agents;
    } catch (error) {
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Get agent status and metrics
   */
  getStatus() {
    return {
      id: this.id,
      name: this.config.name,
      type: this.config.type,
      status: this.status,
      capabilities: this.config.capabilities,
      services: Array.from(this.services.keys()),
      metrics: {
        ...this.metrics,
        uptime: Date.now() - this.startTime
      },
      health: this.healthMonitor.getHealth(),
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Execute function with timeout
   */
  async executeWithTimeout(fn, data, timeout) {
    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        reject(new Error(`Service execution timed out after ${timeout}ms`));
      }, timeout);

      Promise.resolve(fn(data))
        .then(result => {
          clearTimeout(timeoutId);
          resolve(result);
        })
        .catch(error => {
          clearTimeout(timeoutId);
          reject(error);
        });
    });
  }

  /**
   * Handle incoming messages
   */
  async handleMessage(message) {
    try {
      this.emit('message:received', message);

      // Security check
      if (!await this.security.validateMessage(message)) {
        throw new Error('Message security validation failed');
      }

      // Route message based on type
      switch (message.type) {
        case 'service_call':
          return await this.handleServiceCall(message);
        case 'health_check':
          return this.getStatus();
        case 'ping':
          return { type: 'pong', timestamp: Date.now() };
        default:
          throw new Error(`Unknown message type: ${message.type}`);
      }
    } catch (error) {
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Handle service call messages
   */
  async handleServiceCall(message) {
    const { service, data, options } = message.payload;
    return await this.call(service, data, options);
  }

  /**
   * Handle errors
   */
  handleError(error) {
    logger.error(`Agent '${this.config.name}' error:`, error);
    
    // Update status if critical error
    if (this.status === 'running') {
      this.status = 'degraded';
      this.emit('status:changed', 'degraded');
    }
  }
}

module.exports = { Agent };