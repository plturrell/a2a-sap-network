const cds = require('@sap/cds');
const { Web3 } = require('web3');
const { ethers } = require('ethers');
const fs = require('fs').promises;
const path = require('path');
const BaseService = require('./lib/base-service');

class BlockchainService extends BaseService {
    async initializeService() {
        const log = cds.log('blockchain-service');
        
        // Initialize Web3
        const { blockchain } = cds.env.requires;
        this.web3 = new Web3(blockchain.rpc || process.env.BLOCKCHAIN_RPC_URL);
        this.provider = new ethers.JsonRpcProvider(blockchain.rpc || process.env.BLOCKCHAIN_RPC_URL);
        
        // Setup default account
        await this.executeWithErrorHandling(
            () => this.initializeBlockchainAccount(),
            'Failed to initialize blockchain account'
        );
        
        // Load contract ABIs
        this.contracts = {};
        await this.executeWithErrorHandling(
            () => this.loadContracts(blockchain.contracts),
            'Failed to load contracts',
            { contracts: blockchain.contracts }
        );
        
        // Start event listeners
        await this.executeWithErrorHandling(
            () => this.startEventListeners(),
            'Failed to start event listeners'
        );
        
        // Register service handlers
        this.registerHandlers();
    }
    
    async initializeBlockchainAccount() {
        const log = cds.log('blockchain-service');
        
        if (process.env.DEFAULT_PRIVATE_KEY) {
            const account = this.web3.eth.accounts.privateKeyToAccount(process.env.DEFAULT_PRIVATE_KEY);
            this.web3.eth.accounts.wallet.add(account);
            this.web3.eth.defaultAccount = account.address;
            this.defaultSigner = new ethers.Wallet(process.env.DEFAULT_PRIVATE_KEY, this.provider);
            log.info('Blockchain service initialized', { account: account.address });
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
                    .send({ from: agent.address, gas: 500000 });
                
                // Update local record with blockchain confirmation
                await UPDATE(Agents).set({ 
                    address: tx.events.AgentRegistered.returnValues.agent 
                }).where({ ID: agent.ID });
                
                return tx.transactionHash;
            } catch (error) {
                console.error('Blockchain registration failed:', error);
                throw error;
            }
        });
        
        // Service marketplace operations
        this.on('listOnMarketplace', Services, async (req) => {
            const service = await SELECT.one.from(Services).where({ ID: req.params[0] });
            if (!service) throw new Error('Service not found');
            
            const priceInWei = this.web3.utils.toWei(service.pricePerCall.toString(), 'ether');
            
            try {
                const tx = await this.contracts.AgentServiceMarketplace.web3.methods
                    .listService(
                        service.name,
                        service.description || '',
                        priceInWei,
                        service.minReputation,
                        service.maxCallsPerDay
                    )
                    .send({ from: service.provider_ID, gas: 500000 });
                
                return tx.transactionHash;
            } catch (error) {
                console.error('Service listing failed:', error);
                throw error;
            }
        });
        
        // Capability registration
        this.on('registerOnBlockchain', Capabilities, async (req) => {
            const capability = await SELECT.one.from(Capabilities).where({ ID: req.params[0] });
            if (!capability) throw new Error('Capability not found');
            
            try {
                const tx = await this.contracts.CapabilityMatcher.web3.methods
                    .registerCapability(
                        capability.name,
                        capability.description || '',
                        capability.tags || [],
                        capability.inputTypes || [],
                        capability.outputTypes || [],
                        capability.category || 0
                    )
                    .send({ from: this.defaultAccount, gas: 500000 });
                
                return tx.transactionHash;
            } catch (error) {
                console.error('Capability registration failed:', error);
                throw error;
            }
        });
        
        // Complex functions
        this.on('matchCapabilities', async (req) => {
            const requirements = req.data.requirements;
            
            try {
                // Call blockchain for matching
                const matches = await this.contracts.CapabilityMatcher.web3.methods
                    .matchAgentsByCapabilities(requirements)
                    .call();
                
                // Enhance with local data
                const enhancedMatches = await Promise.all(
                    matches.map(async (match) => {
                        const agent = await SELECT.one.from(Agents)
                            .where({ address: match.agent });
                        
                        return {
                            agentId: agent?.ID || '',
                            agentAddress: match.agent,
                            agentName: agent?.name || 'Unknown',
                            matchScore: parseFloat(match.score) / 100,
                            capabilities: match.capabilities
                        };
                    })
                );
                
                return enhancedMatches;
            } catch (error) {
                console.error('Capability matching failed:', error);
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
                console.error('Reputation calculation failed:', error);
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
                        
                        // Upsert to database
                        await UPSERT.into(Agents).entries({
                            address: agentData.agent,
                            name: agentData.name,
                            endpoint: agentData.endpoint,
                            reputation: parseInt(agentData.reputation),
                            isActive: agentData.active
                        });
                        
                        synced++;
                    } catch (e) {
                        failed++;
                    }
                }
                
                // Similar sync for services, capabilities, etc.
                
                return { synced, pending, failed };
            } catch (error) {
                console.error('Blockchain sync failed:', error);
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