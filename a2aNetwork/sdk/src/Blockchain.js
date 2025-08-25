const { ethers } = require('ethers');
const logger = require('pino')({ name: 'a2a-blockchain' });

/**
 * A2A Blockchain Client - Real implementation for blockchain interactions
 */
class Blockchain {
  constructor(config) {
    this.config = {
      network: config.network || 'localhost',
      rpcUrl: config.rpcUrl,
      privateKey: config.privateKey,
      chainId: config.chainId || 31337,
      gasLimit: config.gasLimit || 3000000,
      gasPrice: config.gasPrice || '20000000000', // 20 gwei
      contractAddresses: config.contractAddresses || {},
      confirmations: config.confirmations || 1,
      timeout: config.timeout || 30000
    };

    this.provider = null;
    this.wallet = null;
    this.contracts = new Map();
    this.isConnected = false;
  }

  /**
   * Initialize blockchain connection
   */
  async initialize() {
    try {
      // Create provider
      this.provider = new ethers.JsonRpcProvider(this.config.rpcUrl);

      // Create wallet
      this.wallet = new ethers.Wallet(this.config.privateKey, this.provider);

      // Test connection
      await this.testConnection();

      // Load contracts
      await this.loadContracts();

      this.isConnected = true;
      logger.info(`Connected to blockchain network: ${this.config.network}`);

    } catch (error) {
      logger.error('Failed to initialize blockchain connection:', error.message);
      throw error;
    }
  }

  /**
   * Test blockchain connection
   */
  async testConnection() {
    try {
      const network = await this.provider.getNetwork();
      const balance = await this.provider.getBalance(this.wallet.address);

      logger.info('Blockchain connection established:', {
        network: network.name,
        chainId: network.chainId.toString(),
        address: this.wallet.address,
        balance: ethers.formatEther(balance)
      });

      return true;
    } catch (error) {
      throw new Error(`Blockchain connection test failed: ${error.message}`);
    }
  }

  /**
   * Load smart contracts
   */
  async loadContracts() {
    const contractConfigs = [
      {
        name: 'AgentRegistry',
        address: this.config.contractAddresses.agentRegistry,
        abi: require('./contracts/AgentRegistry.json').abi
      },
      {
        name: 'ServiceMarketplace',
        address: this.config.contractAddresses.serviceMarketplace,
        abi: require('./contracts/ServiceMarketplace.json').abi
      },
      {
        name: 'ReputationSystem',
        address: this.config.contractAddresses.reputationSystem,
        abi: require('./contracts/ReputationSystem.json').abi
      }
    ];

    for (const contractConfig of contractConfigs) {
      if (contractConfig.address) {
        try {
          const contract = new ethers.Contract(
            contractConfig.address,
            contractConfig.abi,
            this.wallet
          );

          this.contracts.set(contractConfig.name, contract);
          logger.info(`Contract '${contractConfig.name}' loaded at ${contractConfig.address}`);
        } catch (error) {
          logger.warn(`Failed to load contract '${contractConfig.name}':`, error.message);
        }
      }
    }
  }

  /**
   * Register an agent on the blockchain
   */
  async registerAgent(agentInfo) {
    try {
      const agentRegistry = this.contracts.get('AgentRegistry');
      if (!agentRegistry) {
        throw new Error('AgentRegistry contract not available');
      }

      const tx = await agentRegistry.registerAgent(
        agentInfo.name,
        agentInfo.type,
        JSON.stringify({
          capabilities: agentInfo.capabilities,
          endpoint: agentInfo.endpoint,
          metadata: agentInfo.metadata || {}
        }),
        {
          gasLimit: this.config.gasLimit,
          gasPrice: this.config.gasPrice
        }
      );

      const receipt = await tx.wait(this.config.confirmations);

      logger.info(`Agent '${agentInfo.name}' registered on blockchain:`, {
        txHash: receipt.hash,
        blockNumber: receipt.blockNumber,
        gasUsed: receipt.gasUsed.toString()
      });

      return {
        txHash: receipt.hash,
        blockNumber: receipt.blockNumber,
        gasUsed: receipt.gasUsed.toString(),
        agentId: agentInfo.id
      };

    } catch (error) {
      logger.error(`Failed to register agent '${agentInfo.name}' on blockchain:`, error.message);
      throw error;
    }
  }

  /**
   * Unregister an agent from the blockchain
   */
  async unregisterAgent(agentId) {
    try {
      const agentRegistry = this.contracts.get('AgentRegistry');
      if (!agentRegistry) {
        throw new Error('AgentRegistry contract not available');
      }

      const tx = await agentRegistry.unregisterAgent(agentId, {
        gasLimit: this.config.gasLimit,
        gasPrice: this.config.gasPrice
      });

      const receipt = await tx.wait(this.config.confirmations);

      logger.info(`Agent '${agentId}' unregistered from blockchain:`, {
        txHash: receipt.hash,
        gasUsed: receipt.gasUsed.toString()
      });

      return {
        txHash: receipt.hash,
        blockNumber: receipt.blockNumber,
        gasUsed: receipt.gasUsed.toString()
      };

    } catch (error) {
      logger.error(`Failed to unregister agent '${agentId}' from blockchain:`, error.message);
      throw error;
    }
  }

  /**
   * Update agent reputation
   */
  async updateReputation(agentId, score, feedback = '') {
    try {
      const reputationSystem = this.contracts.get('ReputationSystem');
      if (!reputationSystem) {
        throw new Error('ReputationSystem contract not available');
      }

      const tx = await reputationSystem.updateReputation(
        agentId,
        Math.floor(score * 100), // Convert to integer (basis points)
        feedback,
        {
          gasLimit: this.config.gasLimit,
          gasPrice: this.config.gasPrice
        }
      );

      const receipt = await tx.wait(this.config.confirmations);

      logger.info(`Reputation updated for agent '${agentId}':`, {
        score,
        txHash: receipt.hash,
        gasUsed: receipt.gasUsed.toString()
      });

      return {
        txHash: receipt.hash,
        blockNumber: receipt.blockNumber,
        gasUsed: receipt.gasUsed.toString()
      };

    } catch (error) {
      logger.error(`Failed to update reputation for agent '${agentId}':`, error.message);
      throw error;
    }
  }

  /**
   * Get agent reputation
   */
  async getReputation(agentId) {
    try {
      const reputationSystem = this.contracts.get('ReputationSystem');
      if (!reputationSystem) {
        throw new Error('ReputationSystem contract not available');
      }

      const reputation = await reputationSystem.getReputation(agentId);

      return {
        agentId,
        score: Number(reputation.score) / 100, // Convert from basis points
        totalRatings: Number(reputation.totalRatings),
        lastUpdated: Number(reputation.lastUpdated),
        isActive: reputation.isActive
      };

    } catch (error) {
      logger.error(`Failed to get reputation for agent '${agentId}':`, error.message);
      throw error;
    }
  }

  /**
   * Create service offering
   */
  async createServiceOffering(serviceInfo) {
    try {
      const marketplace = this.contracts.get('ServiceMarketplace');
      if (!marketplace) {
        throw new Error('ServiceMarketplace contract not available');
      }

      const tx = await marketplace.createServiceOffering(
        serviceInfo.name,
        serviceInfo.description,
        ethers.parseEther(serviceInfo.price.toString()),
        serviceInfo.duration || 3600, // 1 hour default
        JSON.stringify(serviceInfo.metadata || {}),
        {
          gasLimit: this.config.gasLimit,
          gasPrice: this.config.gasPrice
        }
      );

      const receipt = await tx.wait(this.config.confirmations);

      // Get service ID from event logs
      const serviceCreatedEvent = receipt.logs.find(
        log => log.fragment?.name === 'ServiceCreated'
      );

      const serviceId = serviceCreatedEvent ? serviceCreatedEvent.args[0] : null;

      logger.info(`Service offering '${serviceInfo.name}' created:`, {
        serviceId: serviceId?.toString(),
        txHash: receipt.hash,
        gasUsed: receipt.gasUsed.toString()
      });

      return {
        serviceId: serviceId?.toString(),
        txHash: receipt.hash,
        blockNumber: receipt.blockNumber,
        gasUsed: receipt.gasUsed.toString()
      };

    } catch (error) {
      logger.error(`Failed to create service offering '${serviceInfo.name}':`, error.message);
      throw error;
    }
  }

  /**
   * Purchase service
   */
  async purchaseService(serviceId, agentId) {
    try {
      const marketplace = this.contracts.get('ServiceMarketplace');
      if (!marketplace) {
        throw new Error('ServiceMarketplace contract not available');
      }

      // Get service details to determine price
      const service = await marketplace.getService(serviceId);

      const tx = await marketplace.purchaseService(serviceId, agentId, {
        value: service.price,
        gasLimit: this.config.gasLimit,
        gasPrice: this.config.gasPrice
      });

      const receipt = await tx.wait(this.config.confirmations);

      logger.info(`Service '${serviceId}' purchased:`, {
        buyer: agentId,
        txHash: receipt.hash,
        gasUsed: receipt.gasUsed.toString()
      });

      return {
        txHash: receipt.hash,
        blockNumber: receipt.blockNumber,
        gasUsed: receipt.gasUsed.toString()
      };

    } catch (error) {
      logger.error(`Failed to purchase service '${serviceId}':`, error.message);
      throw error;
    }
  }

  /**
   * Get blockchain stats
   */
  async getStats() {
    try {
      const network = await this.provider.getNetwork();
      const blockNumber = await this.provider.getBlockNumber();
      const gasPrice = await this.provider.getFeeData();
      const balance = await this.provider.getBalance(this.wallet.address);

      return {
        network: {
          name: network.name,
          chainId: network.chainId.toString()
        },
        blockNumber,
        gasPrice: {
          gasPrice: gasPrice.gasPrice?.toString(),
          maxFeePerGas: gasPrice.maxFeePerGas?.toString(),
          maxPriorityFeePerGas: gasPrice.maxPriorityFeePerGas?.toString()
        },
        wallet: {
          address: this.wallet.address,
          balance: ethers.formatEther(balance)
        },
        contracts: Array.from(this.contracts.keys())
      };

    } catch (error) {
      logger.error('Failed to get blockchain stats:', error.message);
      throw error;
    }
  }

  /**
   * Listen for blockchain events
   */
  async listenForEvents(contractName, eventName, callback) {
    try {
      const contract = this.contracts.get(contractName);
      if (!contract) {
        throw new Error(`Contract '${contractName}' not available`);
      }

      contract.on(eventName, (...args) => {
        logger.debug(`Blockchain event '${eventName}' received:`, args);
        callback(...args);
      });

      logger.info(`Listening for '${eventName}' events on '${contractName}' contract`);

    } catch (error) {
      logger.error(`Failed to listen for events on '${contractName}':`, error.message);
      throw error;
    }
  }

  /**
   * Stop listening for events
   */
  stopListening(contractName, eventName) {
    try {
      const contract = this.contracts.get(contractName);
      if (contract) {
        contract.removeAllListeners(eventName);
        logger.info(`Stopped listening for '${eventName}' events on '${contractName}' contract`);
      }
    } catch (error) {
      logger.error('Failed to stop listening for events:', error.message);
    }
  }

  /**
   * Get transaction receipt
   */
  async getTransactionReceipt(txHash) {
    try {
      const receipt = await this.provider.getTransactionReceipt(txHash);
      return receipt;
    } catch (error) {
      logger.error(`Failed to get transaction receipt for '${txHash}':`, error.message);
      throw error;
    }
  }

  /**
   * Estimate gas for transaction
   */
  async estimateGas(contractName, methodName, ...args) {
    try {
      const contract = this.contracts.get(contractName);
      if (!contract) {
        throw new Error(`Contract '${contractName}' not available`);
      }

      const gasEstimate = await contract[methodName].estimateGas(...args);
      return gasEstimate.toString();

    } catch (error) {
      logger.error(`Failed to estimate gas for '${contractName}.${methodName}':`, error.message);
      throw error;
    }
  }

  /**
   * Check if connected to blockchain
   */
  isHealthy() {
    return this.isConnected && this.provider && this.wallet;
  }

  /**
   * Disconnect from blockchain
   */
  async disconnect() {
    try {
      // Stop listening to all events
      for (const contract of this.contracts.values()) {
        contract.removeAllListeners();
      }

      this.contracts.clear();
      this.provider = null;
      this.wallet = null;
      this.isConnected = false;

      logger.info('Disconnected from blockchain');

    } catch (error) {
      logger.error('Error disconnecting from blockchain:', error.message);
      throw error;
    }
  }
}

module.exports = { Blockchain };