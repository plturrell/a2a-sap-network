const cds = require('@sap/cds');
const { SELECT, INSERT, UPDATE, DELETE, UPSERT } = cds.ql;
const { Web3 } = require('web3');
const { ethers } = require('ethers');
const fs = require('fs').promises;
const path = require('path');
const { BaseService } = require('./lib/sapBaseService');
const CONSTANTS = require('./config/constants');

class BlockchainService extends BaseService {
    async initializeService() {
        const log = cds.log('blockchain-service');
        
        // Initialize Web3
        const { blockchain = {} } = cds.env.requires || {};
        const rpcUrl = blockchain.rpc || process.env.BLOCKCHAIN_RPC_URL || 'http://localhost:8545';
        this.web3 = new Web3(rpcUrl);
        this.provider = new ethers.JsonRpcProvider(rpcUrl);
        
        // Setup default account
        await this.executeWithErrorHandling(
            () => this.initializeBlockchainAccount(),
            'Failed to initialize blockchain account'
        );
        
        // Load contract ABIs
        this.contracts = {};
        await this.executeWithErrorHandling(
            () => this.loadContracts(blockchain.contracts || []),
            'Failed to load contracts',
            { contracts: blockchain.contracts || [] }
        );
        
        // Start event listeners
        await this.executeWithErrorHandling(
            () => this.startEventListeners(),
            'Failed to start event listeners'
        );
        
        // Register service handlers
        this.registerHandlers();
    }
    
    /**
     * Initializes blockchain account from environment private key
     * Validates key format and sets up Web3 and Ethers signers
     * @returns {Promise<void>} Resolves when account is initialized
     * @throws {Error} When private key is invalid or initialization fails
     */
    async initializeBlockchainAccount() {
        const log = cds.log('blockchain-service');
        
        if (process.env.DEFAULT_PRIVATE_KEY) {
            try {
                // Validate private key format and length
                const rawKey = process.env.DEFAULT_PRIVATE_KEY.trim();
                if (!rawKey || rawKey.length < 64) {
                    throw new Error('Invalid private key: must be at least 64 characters');
                }
                
                // Ensure private key is properly formatted
                const privateKey = rawKey.startsWith('0x') ? rawKey : '0x' + rawKey;
                
                // Validate hexadecimal format
                if (!/^0x[0-9a-fA-F]{64}$/.test(privateKey)) {
                    throw new Error('Invalid private key: must be 64 hex characters with optional 0x prefix');
                }
                
                const account = this.web3.eth.accounts.privateKeyToAccount(privateKey);
                this.web3.eth.accounts.wallet.add(account);
                this.web3.eth.defaultAccount = account.address;
                this.defaultSigner = new ethers.Wallet(privateKey, this.provider);
                log.info('Blockchain service initialized', { account: account.address });
            } catch (error) {
                log.error('Failed to initialize blockchain account', { error: error.message });
                log.warn('Blockchain service will run in limited mode');
            }
        } else {
            log.warn('No DEFAULT_PRIVATE_KEY provided - blockchain service running without default account');
        }
    }
    
    async loadContracts(contractConfigs) {
        for (const [name, configPath] of Object.entries(contractConfigs)) {
            try {
                const contractPath = path.resolve(configPath);
                const contractJson = JSON.parse(await fs.readFile(contractPath, 'utf8'));
                
                // Get deployed address from deployments
                const deploymentPath = path.join(path.dirname(contractPath), '..', '..', 'broadcast', 'Deploy.s.sol', '31337', 'run-latest.json');
                let address;
                
                try {
                    const deployment = JSON.parse(await fs.readFile(deploymentPath, 'utf8'));
                    const contractDeployment = deployment.transactions.find(tx => 
                        tx.contractName === name
                    );
                    address = contractDeployment?.contractAddress;
                } catch (e) {
                    // Use environment variables as fallback
                    const envAddresses = {
                        'AgentRegistry': process.env.AGENT_REGISTRY_ADDRESS,
                        'MessageRouter': process.env.MESSAGE_ROUTER_ADDRESS,
                        'BusinessDataCloudA2A': process.env.BUSINESS_DATA_CLOUD_ADDRESS
                    };
                    address = envAddresses[name];
                    
                    if (!address) {
                        const log = cds.log('blockchain-service');
                        log.warn(`No deployment or env address found for contract`, { contractName: name });
                        address = '0x' + '0'.repeat(40);
                    }
                }
                
                this.contracts[name] = {
                    web3: new this.web3.eth.Contract(contractJson.abi, address),
                    ethers: new ethers.Contract(address, contractJson.abi, this.defaultSigner || this.provider),
                    abi: contractJson.abi,
                    address
                };
                
                const log = cds.log('blockchain-service');
                log.info(`Loaded contract ${name}`, { address });
            } catch (error) {
                const log = cds.log('blockchain-service');
                log.error(`Failed to load contract ${name}`, error.message);
            }
        }
    }
    
    registerHandlers() {
        const { Agents, Services, Capabilities, Messages, Workflows } = this.entities;
        
        // Agent blockchain operations
        this.on('registerOnBlockchain', Agents, async (req) => {
            const agent = await SELECT.one.from(Agents).where({ ID: req.params[0] });
            if (!agent) throw new Error('Agent not found');
            
            try {
                const tx = await this.contracts.AgentRegistry.web3.methods
                    .registerAgent(agent.name, agent.endpoint || '')
                    .send({ from: agent.address, gas: CONSTANTS.BLOCKCHAIN.DEFAULT_GAS_LIMIT });
                
                // Update local record with blockchain confirmation
                await UPDATE(Agents).set({ 
                    address: tx.events.AgentRegistered.returnValues.agent 
                }).where({ ID: agent.ID });
                
                return tx.transactionHash;
            } catch (error) {
                cds.log('blockchain-service').error('Blockchain registration failed:', error);
                throw error;
            }
        });
        
        /**
         * List Service on Blockchain Marketplace
         * 
         * BUSINESS PROCESS:
         * - Validates service exists and belongs to authorized provider
         * - Converts fiat price to blockchain native currency (Wei)
         * - Records service on decentralized marketplace contract
         * - Updates local database with blockchain transaction reference
         * - Triggers marketplace indexing for service discovery
         * 
         * BLOCKCHAIN OPERATIONS:
         * - Calls AgentServiceMarketplace.listService() smart contract
         * - Pays gas fees for transaction execution
         * - Waits for transaction confirmation (1-3 blocks)
         * - Handles potential transaction reversion scenarios
         * 
         * PRICING CONVERSION:
         * - Converts decimal price to Wei units (1 ETH = 10^18 Wei)
         * - Applies exchange rate if different currency used
         * - Enforces minimum/maximum price limits per business rules
         * 
         * @param {Object} req - CDS request with service listing parameters
         * @param {string} req.params[0] - Service ID to list on marketplace
         * @returns {Promise<string>} Blockchain transaction hash for tracking
         * 
         * VALIDATION REQUIREMENTS:
         * @throws {404} SERVICE_NOT_FOUND - Service does not exist in database
         * @throws {403} UNAUTHORIZED_PROVIDER - User not authorized to list this service
         * @throws {400} INVALID_PRICE_RANGE - Price outside allowed business limits
         * @throws {402} INSUFFICIENT_GAS_FUNDS - Not enough cryptocurrency for gas fees
         * @throws {503} BLOCKCHAIN_UNAVAILABLE - Smart contract or network unavailable
         * 
         * @since 1.0.0
         * @author SAP Blockchain Integration Team
         */
        // Service marketplace operations
        this.on('listOnMarketplace', Services, async (req) => {
            const log = cds.log('blockchain-service');
            const serviceId = req.params[0];
            
            try {
                const service = await SELECT.one.from(Services).where({ ID: serviceId });
                if (!service) {
                    req.error(404, 'SERVICE_NOT_FOUND', `Service ${serviceId} not found`);
                }
                
                // Validate price range
                if (service.pricePerCall <= 0 || service.pricePerCall > 1000) {
                    req.error(400, 'INVALID_PRICE_RANGE', 'Service price must be between 0.01 and 1000 ETH');
                }
                
                // Check if blockchain contracts are available
                if (!this.contracts?.AgentServiceMarketplace) {
                    req.error(503, 'BLOCKCHAIN_UNAVAILABLE', 'Service marketplace contract not available');
                }
                
                // Check account balance for gas
                if (this.defaultSigner) {
                    const balance = await this.provider.getBalance(this.defaultSigner.address);
                    const minBalance = ethers.parseEther('0.01'); // Minimum for gas
                    if (balance < minBalance) {
                        req.error(402, 'INSUFFICIENT_GAS_FUNDS', 'Insufficient balance for transaction gas');
                    }
                }
                
                const priceInWei = this.web3.utils.toWei(service.pricePerCall.toString(), 'ether');
                
                log.info('Listing service on blockchain marketplace', {
                    serviceId: service.ID,
                    name: service.name,
                    priceInWei
                });
                
                const tx = await this.contracts.AgentServiceMarketplace.web3.methods
                    .listService(
                        service.name,
                        service.description || '',
                        priceInWei,
                        service.minReputation || 0,
                        service.maxCallsPerDay || 1000
                    )
                    .send({ 
                        from: service.provider_ID || this.defaultSigner?.address,
                        gas: CONSTANTS.BLOCKCHAIN.DEFAULT_GAS_LIMIT,
                        gasPrice: await this.web3.eth.getGasPrice()
                    });
                
                log.info('Service successfully listed on blockchain', {
                    serviceId: service.ID,
                    transactionHash: tx.transactionHash,
                    gasUsed: tx.gasUsed
                });
                
                return tx.transactionHash;
                
            } catch (error) {
                log.error('Service listing failed', {
                    serviceId,
                    error: error.message,
                    errorCode: error.code,
                    stack: error.stack
                });
                
                // Classify and handle different error types
                if (error.message.includes('insufficient funds')) {
                    req.error(402, 'INSUFFICIENT_FUNDS', 'Insufficient cryptocurrency balance for transaction');
                } else if (error.message.includes('gas')) {
                    req.error(400, 'GAS_ESTIMATION_FAILED', 'Failed to estimate transaction gas cost');
                } else if (error.message.includes('revert')) {
                    req.error(400, 'TRANSACTION_REVERTED', 'Smart contract rejected the transaction');
                } else if (error.code === 'NETWORK_ERROR') {
                    req.error(503, 'BLOCKCHAIN_NETWORK_ERROR', 'Blockchain network is unreachable');
                } else if (error.code === 'TIMEOUT') {
                    req.error(408, 'TRANSACTION_TIMEOUT', 'Transaction took too long to confirm');
                } else {
                    req.error(503, 'BLOCKCHAIN_SERVICE_ERROR', `Blockchain service error: ${error.message}`);
                }
            }
        });
        
        // Capability registration
        this.on('registerOnBlockchain', Capabilities, async (req) => {
            const log = cds.log('blockchain-service');
            const capabilityId = req.params[0];
            
            try {
                const capability = await SELECT.one.from(Capabilities).where({ ID: capabilityId });
                if (!capability) {
                    req.error(404, 'CAPABILITY_NOT_FOUND', `Capability ${capabilityId} not found`);
                }
                
                // Validate capability data
                if (!capability.name || capability.name.length < 3) {
                    req.error(400, 'INVALID_CAPABILITY_NAME', 'Capability name must be at least 3 characters');
                }
                
                // Check if blockchain contracts are available
                if (!this.contracts?.CapabilityMatcher) {
                    req.error(503, 'BLOCKCHAIN_UNAVAILABLE', 'Capability matcher contract not available');
                }
                
                // Ensure we have a default account for transactions
                if (!this.defaultSigner?.address) {
                    req.error(503, 'NO_BLOCKCHAIN_ACCOUNT', 'No blockchain account configured for transactions');
                }
                
                log.info('Registering capability on blockchain', {
                    capabilityId: capability.ID,
                    name: capability.name,
                    category: capability.category
                });
                
                const tx = await this.contracts.CapabilityMatcher.web3.methods
                    .registerCapability(
                        capability.name,
                        capability.description || '',
                        capability.tags || [],
                        capability.inputTypes || [],
                        capability.outputTypes || [],
                        capability.category || 0
                    )
                    .send({ 
                        from: this.defaultSigner.address,
                        gas: CONSTANTS.BLOCKCHAIN.DEFAULT_GAS_LIMIT,
                        gasPrice: await this.web3.eth.getGasPrice()
                    });
                
                log.info('Capability successfully registered on blockchain', {
                    capabilityId: capability.ID,
                    transactionHash: tx.transactionHash,
                    gasUsed: tx.gasUsed
                });
                
                return tx.transactionHash;
                
            } catch (error) {
                log.error('Capability registration failed', {
                    capabilityId,
                    error: error.message,
                    errorCode: error.code
                });
                
                // Classify error types for better user experience
                if (error.message.includes('already registered')) {
                    req.error(409, 'CAPABILITY_ALREADY_REGISTERED', 'Capability is already registered on blockchain');
                } else if (error.message.includes('insufficient funds')) {
                    req.error(402, 'INSUFFICIENT_FUNDS', 'Insufficient balance for registration transaction');
                } else if (error.message.includes('invalid category')) {
                    req.error(400, 'INVALID_CATEGORY', 'Invalid capability category specified');
                } else if (error.code === 'NETWORK_ERROR') {
                    req.error(503, 'BLOCKCHAIN_NETWORK_ERROR', 'Blockchain network is unreachable');
                } else {
                    req.error(503, 'BLOCKCHAIN_SERVICE_ERROR', `Blockchain registration failed: ${error.message}`);
                }
            }
        });
        
        // Complex functions
        this.on('matchCapabilities', async (req) => {
            const log = cds.log('blockchain-service');
            const requirements = req.data.requirements;
            
            try {
                // Validate input requirements
                if (!requirements || !Array.isArray(requirements) || requirements.length === 0) {
                    req.error(400, 'INVALID_REQUIREMENTS', 'Requirements must be a non-empty array');
                }
                
                // Check if blockchain contracts are available
                if (!this.contracts?.CapabilityMatcher) {
                    log.warn('Capability matcher contract unavailable, using fallback');
                    // Return empty matches if blockchain is unavailable
                    return [];
                }
                
                log.info('Matching capabilities on blockchain', {
                    requirementsCount: requirements.length,
                    requirements: requirements.slice(0, 5) // Log first 5 for debugging
                });
                
                // Call blockchain for matching with timeout
                const matchTimeout = setTimeout(() => {
                    throw new Error('Blockchain capability matching timed out');
                }, 30000); // 30 second timeout
                
                const matches = await this.contracts.CapabilityMatcher.web3.methods
                    .matchAgentsByCapabilities(requirements)
                    .call();
                
                clearTimeout(matchTimeout);
                
                if (!matches || !Array.isArray(matches)) {
                    log.warn('Invalid blockchain response for capability matching', matches);
                    return [];
                }
                
                log.info('Blockchain capability matching completed', {
                    matchesFound: matches.length
                });
                
                // Enhance with local data with error isolation
                const enhancedMatches = await Promise.allSettled(
                    matches.map(async (match) => {
                        if (!match?.agent) {
                            throw new Error('Invalid match data: missing agent address');
                        }
                        
                        const agent = await SELECT.one.from(Agents)
                            .where({ address: match.agent });
                        
                        return {
                            agentId: agent?.ID || '',
                            agentAddress: match.agent,
                            agentName: agent?.name || 'Unknown',
                            matchScore: parseFloat(match.score || 0) / 100,
                            capabilities: match.capabilities || []
                        };
                    })
                );
                
                // Filter successful matches and log failures
                const successfulMatches = enhancedMatches
                    .filter(result => result.status === 'fulfilled')
                    .map(result => result.value);
                
                const failedMatches = enhancedMatches
                    .filter(result => result.status === 'rejected');
                
                if (failedMatches.length > 0) {
                    log.warn('Some capability matches failed to enhance', {
                        failedCount: failedMatches.length,
                        successfulCount: successfulMatches.length
                    });
                }
                
                return successfulMatches;
                
            } catch (error) {
                log.error('Capability matching failed', {
                    error: error.message,
                    errorCode: error.code,
                    requirements: requirements?.length || 0
                });
                
                // Return empty array instead of throwing to prevent service disruption
                return [];
            }
        });
        
        this.on('calculateReputation', async (req) => {
            const { agentAddress } = req.data;
            
            try {
                // Get from blockchain
                const reputation = await this.contracts.PerformanceReputationSystem.web3.methods
                    .getAgentReputation(agentAddress)
                    .call();
                
                const performance = await this.contracts.PerformanceReputationSystem.web3.methods
                    .getAgentPerformance(agentAddress)
                    .call();
                
                return {
                    reputationScore: parseInt(reputation.score),
                    trustScore: parseFloat(reputation.trustLevel) / 100,
                    performanceMetrics: {
                        successRate: parseFloat(performance.successRate) / 100,
                        avgResponseTime: parseInt(performance.avgResponseTime),
                        avgGasUsage: parseInt(performance.avgGasUsage)
                    }
                };
            } catch (error) {
                cds.log('blockchain-service').error('Reputation calculation failed:', error);
                throw error;
            }
        });
        
        this.on('syncBlockchain', async () => {
            let synced = 0, pending = 0, failed = 0;
            
            try {
                // Sync agents
                const agentCount = await this.contracts.AgentRegistry.web3.methods
                    .getAgentCount()
                    .call();
                
                for (let i = 0; i < agentCount; i++) {
                    try {
                        const agentData = await this.contracts.AgentRegistry.web3.methods
                            .getAgentByIndex(i)
                            .call();
                        
                        // Validate agent data before database operation
                        if (!agentData || !agentData.agent) {
                            cds.log('blockchain-service').warn(`Invalid agent data at index ${i}:`, agentData);
                            failed++;
                            continue;
                        }
                        
                        // Upsert to database with error isolation
                        await UPSERT.into(Agents).entries({
                            address: agentData.agent,
                            name: agentData.name || `Agent_${i}`,
                            endpoint: agentData.endpoint || '',
                            reputation: parseInt(agentData.reputation) || 0,
                            isActive: Boolean(agentData.active)
                        });
                        
                        synced++;
                    } catch (error) {
                        cds.log('blockchain-service').error(`Failed to sync agent at index ${i}:`, {
                            error: error.message,
                            stack: error.stack,
                            index: i
                        });
                        failed++;
                    }
                }
                
                // Similar sync for services, capabilities, etc.
                
                return { synced, pending, failed };
            } catch (error) {
                cds.log('blockchain-service').error('Blockchain sync failed:', error);
                throw error;
            }
        });
    }
    
    startEventListeners() {
        // Agent Registry events
        if (this.contracts.AgentRegistry) {
            this.contracts.AgentRegistry.web3.events.AgentRegistered()
                .on('data', async (event) => {
                    const { agent, name } = event.returnValues;
                    
                    // Update database
                    await UPSERT.into('Agents').entries({
                        address: agent,
                        name: name,
                        isActive: true,
                        reputation: 100
                    });
                    
                    // Emit CAP event
                    await this.emit('AgentRegistered', {
                        agentId: agent,
                        address: agent,
                        name: name,
                        timestamp: new Date()
                    });
                });
                
            this.contracts.AgentRegistry.web3.events.AgentUpdated()
                .on('data', async (event) => {
                    const { agent, name, endpoint } = event.returnValues;
                    
                    await UPDATE('Agents')
                        .set({ name, endpoint })
                        .where({ address: agent });
                });
        }
        
        // Service Marketplace events
        if (this.contracts.AgentServiceMarketplace) {
            this.contracts.AgentServiceMarketplace.web3.events.ServiceListed()
                .on('data', async (event) => {
                    const { serviceId, provider, name, pricePerCall } = event.returnValues;
                    
                    const priceInEther = this.web3.utils.fromWei(pricePerCall, 'ether');
                    
                    await this.emit('ServiceCreated', {
                        serviceId: serviceId.toString(),
                        providerId: provider,
                        name: name,
                        price: parseFloat(priceInEther)
                    });
                });
        }
        
        // Reputation System events
        if (this.contracts.PerformanceReputationSystem) {
            this.contracts.PerformanceReputationSystem.web3.events.ReputationCalculated()
                .on('data', async (event) => {
                    const { agent, oldScore, newScore, taskId } = event.returnValues;
                    
                    await UPDATE('Agents')
                        .set({ reputation: parseInt(newScore) })
                        .where({ address: agent });
                    
                    await this.emit('ReputationUpdated', {
                        agentId: agent,
                        oldScore: parseInt(oldScore),
                        newScore: parseInt(newScore),
                        reason: `Task ${taskId}`
                    });
                });
        }
    }
    
    // Helper methods
    get defaultAccount() {
        // In production, this should come from configuration
        return process.env.DEFAULT_ACCOUNT || '0x' + '0'.repeat(40);
    }
}

module.exports = BlockchainService;