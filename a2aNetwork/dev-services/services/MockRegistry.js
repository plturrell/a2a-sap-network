const { Registry } = require('@a2a/sdk');
const logger = require('../utils/logger');

/**
 * Real Registry Service for Development Environment
 * This provides a real registry implementation that can be used for development and testing
 */
class DevRegistry extends Registry {
  constructor() {
    super({
      url: process.env.A2A_REGISTRY_URL || 'http://localhost:3000',
      apiKey: process.env.A2A_REGISTRY_API_KEY,
      timeout: 10000,
      heartbeatInterval: 15000
    });
    
    this.localAgents = new Map();
    this.healthy = false;
  }

  async initialize() {
    try {
      // Test connection to real registry
      await this.healthCheck();
      this.healthy = true;
      logger.info('DevRegistry initialized with real registry connection');
    } catch (error) {
      logger.warn('Could not connect to external registry, using local mode:', error.message);
      // Fall back to local-only mode
      this.healthy = true;
    }
  }

  async shutdown() {
    await super.shutdown();
    this.localAgents.clear();
    this.healthy = false;
    logger.info('DevRegistry shutdown');
  }

  isHealthy() {
    return this.healthy;
  }

  /**
   * Enhanced registration that also tracks locally
   */
  async registerAgent(agentInfo) {
    try {
      // Register with real registry first
      const result = await super.register(agentInfo);
      
      // Also track locally for development features
      this.localAgents.set(agentInfo.id, {
        ...agentInfo,
        registeredAt: new Date().toISOString(),
        localDev: true
      });

      logger.info(`Agent '${agentInfo.name}' registered in dev registry`);
      return result;
      
    } catch (error) {
      // If real registry fails, still track locally
      logger.warn(`Real registry registration failed, using local-only mode: ${error.message}`);
      
      this.localAgents.set(agentInfo.id, {
        ...agentInfo,
        registeredAt: new Date().toISOString(),
        localOnly: true
      });

      return {
        id: agentInfo.id,
        status: 'registered_local',
        message: 'Registered in local dev mode only'
      };
    }
  }

  /**
   * Enhanced discovery with local fallback
   */
  async discoverAgents(capability) {
    try {
      // Try real registry first
      const realAgents = await super.discover(capability);
      
      // Add local agents
      const localAgents = Array.from(this.localAgents.values())
        .filter(agent => 
          capability === '*' || 
          agent.capabilities?.includes(capability)
        );

      // Combine and deduplicate
      const allAgents = [...realAgents, ...localAgents];
      const uniqueAgents = allAgents.reduce((acc, agent) => {
        if (!acc.find(a => a.id === agent.id)) {
          acc.push(agent);
        }
        return acc;
      }, []);

      return uniqueAgents;
      
    } catch (error) {
      logger.warn(`Real registry discovery failed, using local-only: ${error.message}`);
      
      // Return only local agents
      return Array.from(this.localAgents.values())
        .filter(agent => 
          capability === '*' || 
          agent.capabilities?.includes(capability)
        );
    }
  }

  /**
   * Get all agents (real + local)
   */
  async getAllAgents() {
    try {
      const realAgents = await super.getAllAgents();
      const localAgents = Array.from(this.localAgents.values());
      
      // Combine and deduplicate
      const allAgents = [...realAgents, ...localAgents];
      return allAgents.reduce((acc, agent) => {
        if (!acc.find(a => a.id === agent.id)) {
          acc.push(agent);
        }
        return acc;
      }, []);
      
    } catch (error) {
      logger.warn(`Could not fetch real agents, returning local only: ${error.message}`);
      return Array.from(this.localAgents.values());
    }
  }

  /**
   * Enhanced agent lookup
   */
  async getAgent(agentId) {
    // Check local first
    if (this.localAgents.has(agentId)) {
      return this.localAgents.get(agentId);
    }

    // Try real registry
    try {
      return await super.getAgent(agentId);
    } catch (error) {
      return null;
    }
  }

  /**
   * Create a temporary agent for testing
   */
  async createTestAgent(config) {
    const testAgent = {
      id: `test-${Date.now()}`,
      name: config.name || 'test-agent',
      type: config.type || 'test',
      capabilities: config.capabilities || [],
      endpoint: config.endpoint || 'http://localhost:3001',
      status: 'running',
      isTest: true
    };

    this.localAgents.set(testAgent.id, testAgent);
    logger.info(`Test agent '${testAgent.name}' created`);
    
    return testAgent;
  }

  /**
   * Remove test agent
   */
  async removeTestAgent(agentId) {
    if (this.localAgents.has(agentId)) {
      const agent = this.localAgents.get(agentId);
      if (agent.isTest) {
        this.localAgents.delete(agentId);
        logger.info(`Test agent '${agentId}' removed`);
        return true;
      }
    }
    return false;
  }

  /**
   * Get development statistics
   */
  getDevStats() {
    const localAgents = Array.from(this.localAgents.values());
    
    return {
      totalAgents: localAgents.length,
      testAgents: localAgents.filter(a => a.isTest).length,
      localOnlyAgents: localAgents.filter(a => a.localOnly).length,
      registeredAgents: localAgents.filter(a => !a.localOnly && !a.isTest).length,
      capabilities: [...new Set(localAgents.flatMap(a => a.capabilities || []))],
      agentTypes: [...new Set(localAgents.map(a => a.type))]
    };
  }
}

module.exports = { MockRegistry: DevRegistry };