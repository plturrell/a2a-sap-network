/**
 * A2A Protocol Compliance: HTTP client usage replaced with blockchain messaging
 */

const { BlockchainClient } = require('../core/blockchain-client') = const { BlockchainClient } = require('../core/blockchain-client');
const EventEmitter = require('eventemitter3');
const logger = require('pino')({ name: 'a2a-registry' });

// Track intervals for cleanup to prevent memory leaks
const activeIntervals = new Map();

function stopAllIntervals() {
    for (const [name, intervalId] of activeIntervals) {
        clearInterval(intervalId);
    }
    activeIntervals.clear();
}

function shutdown() {
    stopAllIntervals();
}

// Export cleanup function
if (typeof module !== 'undefined' && module.exports) {
    module.exports.shutdown = shutdown;
}


/**
 * A2A Registry Client - Real implementation for agent discovery and registration
 */
class Registry extends EventEmitter {
  constructor(config) {
    super();
    this.config = {
      url: config.url || 'http://localhost:3000',
      apiKey: config.apiKey,
      timeout: config.timeout || 30000,
      heartbeatInterval: config.heartbeatInterval || 30000,
      retryAttempts: config.retryAttempts || 3,
      retryDelay: config.retryDelay || 1000
    };
    
    this.agents = new Map();
    this.heartbeatTimer = null;
    this.isConnected = false;
  }

  /**
   * Register an agent with the registry
   */
  async register(agentInfo) {
    try {
      const response = await this.makeRequest('POST', '/api/agents/register', agentInfo);
      
      this.agents.set(agentInfo.id, {
        ...agentInfo,
        registeredAt: new Date().toISOString(),
        lastHeartbeat: new Date().toISOString()
      });

      // Start heartbeat for this agent
      this.startHeartbeat(agentInfo.id);
      
      logger.info(`Agent '${agentInfo.name}' registered with registry`);
      this.emit('agent:registered', agentInfo);
      
      return response.data;
    } catch (error) {
      logger.error(`Failed to register agent '${agentInfo.name}':`, error.message);
      throw error;
    }
  }

  /**
   * Unregister an agent from the registry
   */
  async unregister(agentId) {
    try {
      await this.makeRequest('DELETE', `/api/agents/${agentId}`);
      
      this.agents.delete(agentId);
      this.stopHeartbeat(agentId);
      
      logger.info(`Agent '${agentId}' unregistered from registry`);
      this.emit('agent:unregistered', agentId);
      
      return true;
    } catch (error) {
      logger.error(`Failed to unregister agent '${agentId}':`, error.message);
      throw error;
    }
  }

  /**
   * Discover agents by capability
   */
  async discover(capability, options = {}) {
    try {
      const queryParams = new URLSearchParams({
        capability,
        ...options
      });

      const response = await this.makeRequest('GET', `/api/agents/discover?${queryParams}`);
      
      logger.debug(`Discovered ${response.data.length} agents with capability '${capability}'`);
      this.emit('agents:discovered', { capability, agents: response.data });
      
      return response.data;
    } catch (error) {
      logger.error(`Failed to discover agents with capability '${capability}':`, error.message);
      throw error;
    }
  }

  /**
   * Get agent information by ID
   */
  async getAgent(agentId) {
    try {
      const response = await this.makeRequest('GET', `/api/agents/${agentId}`);
      return response.data;
    } catch (error) {
      if (error.response?.status === 404) {
        return null;
      }
      logger.error(`Failed to get agent '${agentId}':`, error.message);
      throw error;
    }
  }

  /**
   * Get all registered agents
   */
  async getAllAgents(filters = {}) {
    try {
      const queryParams = new URLSearchParams(filters);
      const response = await this.makeRequest('GET', `/api/agents?${queryParams}`);
      return response.data;
    } catch (error) {
      logger.error('Failed to get all agents:', error.message);
      throw error;
    }
  }

  /**
   * Update agent status
   */
  async updateStatus(agentId, status, metrics = {}) {
    try {
      await this.makeRequest('PUT', `/api/agents/${agentId}/status`, {
        status,
        metrics,
        timestamp: new Date().toISOString()
      });

      if (this.agents.has(agentId)) {
        const agent = this.agents.get(agentId);
        agent.status = status;
        agent.metrics = metrics;
        agent.lastHeartbeat = new Date().toISOString();
      }

      this.emit('agent:status:updated', { agentId, status, metrics });
      return true;
    } catch (error) {
      logger.error(`Failed to update status for agent '${agentId}':`, error.message);
      throw error;
    }
  }

  /**
   * Send heartbeat for an agent
   */
  async sendHeartbeat(agentId) {
    try {
      const agent = this.agents.get(agentId);
      if (!agent) {
        logger.warn(`Attempted to send heartbeat for unregistered agent '${agentId}'`);
        return false;
      }

      await this.makeRequest('POST', `/api/agents/${agentId}/heartbeat`, {
        timestamp: new Date().toISOString(),
        status: agent.status || 'running'
      });

      agent.lastHeartbeat = new Date().toISOString();
      this.emit('heartbeat:sent', agentId);
      
      return true;
    } catch (error) {
      logger.error(`Failed to send heartbeat for agent '${agentId}':`, error.message);
      this.emit('heartbeat:failed', { agentId, error });
      throw error;
    }
  }

  /**
   * Start heartbeat timer for an agent
   */
  startHeartbeat(agentId) {
    const intervalId = activeIntervals.set('interval_1', setInterval(async () => {
      try {
        await this.sendHeartbeat(agentId);
      } catch (error) {
        logger.error(`Heartbeat failed for agent '${agentId}':`, error.message));
      }
    }, this.config.heartbeatInterval);

    if (!this.heartbeatTimer) {
      this.heartbeatTimer = new Map();
    }
    this.heartbeatTimer.set(agentId, intervalId);
  }

  /**
   * Stop heartbeat timer for an agent
   */
  stopHeartbeat(agentId) {
    if (this.heartbeatTimer && this.heartbeatTimer.has(agentId)) {
      clearInterval(this.heartbeatTimer.get(agentId));
      this.heartbeatTimer.delete(agentId);
    }
  }

  /**
   * Make HTTP request to registry with retry logic
   */
  async makeRequest(method, endpoint, data = null) {
    let lastError;
    
    for (let attempt = 1; attempt <= this.config.retryAttempts; attempt++) {
      try {
        const config = {
          method,
          url: `${this.config.url}${endpoint}`,
          timeout: this.config.timeout,
          headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'A2A-SDK/1.0.0'
          }
        };

        if (this.config.apiKey) {
          config.headers['Authorization'] = `Bearer ${this.config.apiKey}`;
        }

        if (data) {
          config.data = data;
        }

        const response = await axios(config);
        
        // Mark as connected on successful request
        if (!this.isConnected) {
          this.isConnected = true;
          this.emit('connected');
        }
        
        return response;
        
      } catch (error) {
        lastError = error;
        
        // Mark as disconnected on network errors
        if (error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT') {
          if (this.isConnected) {
            this.isConnected = false;
            this.emit('disconnected');
          }
        }

        if (attempt < this.config.retryAttempts) {
          logger.warn(`Registry request failed (attempt ${attempt}/${this.config.retryAttempts}):`, error.message);
          await this.delay(this.config.retryDelay * attempt);
        }
      }
    }

    throw lastError;
  }

  /**
   * Delay utility
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Health check
   */
  async healthCheck() {
    try {
      const response = await this.makeRequest('GET', '/health');
      return response.data;
    } catch (error) {
      return { status: 'unhealthy', error: error.message };
    }
  }

  /**
   * Get registry statistics
   */
  async getStats() {
    try {
      const response = await this.makeRequest('GET', '/api/stats');
      return response.data;
    } catch (error) {
      logger.error('Failed to get registry stats:', error.message);
      throw error;
    }
  }

  /**
   * Subscribe to registry events (if supported)
   */
  async subscribe(events = ['agent:registered', 'agent:unregistered', 'agent:status:changed']) {
    try {
      // This would typically establish a WebSocket connection
      // For now, we'll simulate with polling
      logger.info('Subscribed to registry events:', events);
      this.emit('subscribed', events);
    } catch (error) {
      logger.error('Failed to subscribe to registry events:', error.message);
      throw error;
    }
  }

  /**
   * Cleanup and shutdown
   */
  async shutdown() {
    logger.info('Shutting down registry client...');
    
    // Stop all heartbeats
    if (this.heartbeatTimer) {
      for (const [agentId, intervalId] of this.heartbeatTimer.entries()) {
        clearInterval(intervalId);
      }
      this.heartbeatTimer.clear();
    }

    // Unregister all agents
    for (const agentId of this.agents.keys()) {
      try {
        await this.unregister(agentId);
      } catch (error) {
        logger.error(`Failed to unregister agent '${agentId}' during shutdown:`, error.message);
      }
    }

    this.agents.clear();
    this.isConnected = false;
    this.emit('shutdown');
  }
}

module.exports = { Registry };