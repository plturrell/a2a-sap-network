/**
 * @fileoverview Blockchain Service Implementation
 * @description Handles all blockchain interactions for A2A Network
 * @module blockchain-service
 * @since 1.0.0
 * @author A2A Network Team
 * @namespace a2a.srv
 */

const cds = require('@sap/cds');
const { SELECT, INSERT, UPDATE } = cds.ql;

/**
 * Blockchain Service Implementation
 * Handles agent registration, service marketplace, and blockchain operations
 */
module.exports = cds.service.impl(async function () {
    
    const { BlockchainStats } = this.entities;
    
    // Import blockchain client
    const blockchainService = await cds.connect.to('sapBlockchainService');
    
    // SECURITY FIX: Import authentication middleware
    const { authenticateBlockchainOperation, validateBlockchainAddress } = require('./middleware/blockchainAuth');
    
    // ENHANCEMENT: Import error handling and security logging
    const { handleRequestError } = require('./utils/errorHandler');
    const { logSecurityEvent, SecurityEventTypes, RiskLevels } = require('./utils/securityLogger');

    /**
     * Initialize blockchain statistics
     */
    this.on('served', async () => {
        try {
            // Initialize blockchain stats if not exists
            const existingStats = await SELECT.one.from(BlockchainStats);
            
            if (!existingStats) {
                await INSERT.into(BlockchainStats).entries({
                    ID: cds.utils.uuid(),
                    blockHeight: 0,
                    gasPrice: 20.0,
                    networkStatus: 'connecting',
                    totalTransactions: 0,
                    averageBlockTime: 13.5,
                    timestamp: new Date()
                });
                
                cds.log('blockchain').info('Blockchain stats initialized');
            }
        } catch (error) {
            cds.log('blockchain').error('Failed to initialize blockchain stats:', error);
        }
    });

    // Import retry utility and cache
    const blockchainRetry = require('./utils/blockchainRetry');
    const blockchainCache = require('./utils/blockchainCache');
    
    // ENHANCEMENT: Import circuit breaker and gas optimizer
    const { getBreaker } = require('./utils/circuitBreaker');
    const GasOptimizer = require('./utils/gasOptimizer');
    
    // Initialize circuit breakers for different operation types
    const agentOpsBreaker = getBreaker('agent-operations', {
        failureThreshold: 5,
        resetTimeout: 60000
    });
    
    const serviceOpsBreaker = getBreaker('service-operations', {
        failureThreshold: 3,
        resetTimeout: 45000
    });
    
    const queryOpsBreaker = getBreaker('query-operations', {
        failureThreshold: 10,
        resetTimeout: 30000
    });
    
    // SECURITY FIX: Apply authentication middleware to all write operations
    this.before([
        'registerAgent',
        'updateAgentReputation',
        'deactivateAgent',
        'listService',
        'createServiceOrder',
        'completeServiceOrder',
        'sendMessage',
        'confirmMessageDelivery',
        'deployWorkflow',
        'executeWorkflow',
        'endorsePeer',
        'registerCapability',
        'addAgentCapability'
    ], authenticateBlockchainOperation);
    
    // SECURITY FIX: Apply address validation to relevant operations
    this.before([
        'registerAgent',
        'updateAgentReputation',
        'deactivateAgent',
        'sendMessage',
        'getAgentInfo',
        'getAgentReputation',
        'addAgentCapability',
        'endorsePeer'
    ], validateBlockchainAddress);

    // Agent Operations
    this.on('registerAgent', async (req) => {
        try {
            const { agentId, address, name, endpoint } = req.data;
            
            // Validate inputs
            if (!agentId || !address || !name || !endpoint) {
                req.error(400, 'Missing required fields for agent registration');
                return; // SECURITY FIX: Return after error
            }
            
            // SECURITY FIX: Validate Ethereum address format
            if (!this.isValidEthereumAddress(address)) {
                req.error(400, 'Invalid Ethereum address format');
                return;
            }
            
            // ENHANCEMENT: Use circuit breaker with retry logic
            const result = await agentOpsBreaker.execute(
                () => blockchainRetry.executeWithRetry(
                    () => blockchainService.registerAgent(address, name, endpoint),
                    { operation: 'registerAgent', agentId, address }
                ),
                'registerAgent'
            );
            
            // Emit event
            this.emit('AgentRegistered', {
                agentId,
                address,
                name,
                timestamp: new Date()
            });
            
            // ENHANCEMENT: Log successful security event
            logSecurityEvent(SecurityEventTypes.BLOCKCHAIN_TRANSACTION, RiskLevels.INFO, {
                operation: 'registerAgent',
                agentId: agentId,
                address: address,
                userId: req.user?.id,
                ipAddress: req._.req?.ip,
                userAgent: req._.req?.get('User-Agent'),
                success: true
            });
            
            cds.log('blockchain').info('Agent registered', { agentId, address });
            return `Agent ${agentId} registered successfully`;
            
        } catch (error) {
            // ENHANCEMENT: Use structured error handling and security logging
            logSecurityEvent(SecurityEventTypes.ERROR_OCCURRED, RiskLevels.MEDIUM, {
                operation: 'registerAgent',
                agentId: agentId,
                address: address,
                userId: req.user?.id,
                ipAddress: req._.req?.ip,
                error: error.message,
                success: false
            });
            
            handleRequestError(req, error, {
                operation: 'registerAgent',
                agentId: agentId
            });
        }
    });

    this.on('updateAgentReputation', async (req) => {
        try {
            const { agentAddress, newScore } = req.data;
            
            if (!agentAddress || newScore === undefined) {
                req.error(400, 'Missing agent address or reputation score');
                return; // SECURITY FIX: Return after error
            }
            
            // SECURITY FIX: Validate score range
            if (newScore < 0 || newScore > 1000) {
                req.error(400, 'Reputation score must be between 0 and 1000');
                return;
            }
            
            const result = await blockchainService.updateReputation(agentAddress, newScore);
            
            // Emit event
            this.emit('ReputationUpdated', {
                agentId: agentAddress,
                oldScore: 0, // Would need to fetch from blockchain
                newScore,
                reason: 'Manual update',
                timestamp: new Date()
            });
            
            return `Reputation updated for agent ${agentAddress}`;
            
        } catch (error) {
            cds.log('blockchain').error('Failed to update reputation:', error);
            req.error(500, 'Failed to update reputation', error.message);
        }
    });

    this.on('deactivateAgent', async (req) => {
        try {
            const { agentAddress } = req.data;
            
            if (!agentAddress) {
                req.error(400, 'Missing agent address');
                return; // SECURITY FIX: Return after error
            }
            
            const result = await blockchainService.deactivateAgent(agentAddress);
            
            cds.log('blockchain').info('Agent deactivated', { agentAddress });
            return `Agent ${agentAddress} deactivated successfully`;
            
        } catch (error) {
            cds.log('blockchain').error('Failed to deactivate agent:', error);
            req.error(500, 'Failed to deactivate agent', error.message);
        }
    });

    // Service Marketplace Operations
    this.on('listService', async (req) => {
        try {
            const { serviceId, name, description, pricePerCall, minReputation } = req.data;
            
            if (!serviceId || !name || !description || !pricePerCall) {
                req.error(400, 'Missing required service fields');
            }
            
            const result = await blockchainService.listService(serviceId, {
                name,
                description,
                pricePerCall,
                minReputation: minReputation || 0
            });
            
            // Emit event
            this.emit('ServiceCreated', {
                serviceId,
                providerId: req.user?.id || 'system',
                name,
                price: pricePerCall
            });
            
            return `Service ${serviceId} listed successfully`;
            
        } catch (error) {
            cds.log('blockchain').error('Failed to list service:', error);
            req.error(500, 'Failed to list service', error.message);
        }
    });

    this.on('createServiceOrder', async (req) => {
        try {
            const { serviceId, consumer, amount } = req.data;
            
            if (!serviceId || !consumer || !amount) {
                req.error(400, 'Missing required order fields');
            }
            
            const result = await blockchainService.createServiceOrder(serviceId, consumer, amount);
            
            return `Service order created for ${serviceId}`;
            
        } catch (error) {
            cds.log('blockchain').error('Failed to create service order:', error);
            req.error(500, 'Failed to create service order', error.message);
        }
    });

    this.on('completeServiceOrder', async (req) => {
        try {
            const { orderId, rating } = req.data;
            
            if (!orderId) {
                req.error(400, 'Missing order ID');
            }
            
            const result = await blockchainService.completeServiceOrder(orderId, rating || 0);
            
            return `Service order ${orderId} completed`;
            
        } catch (error) {
            cds.log('blockchain').error('Failed to complete service order:', error);
            req.error(500, 'Failed to complete service order', error.message);
        }
    });

    // Capability Operations
    this.on('registerCapability', async (req) => {
        try {
            const { name, description, category } = req.data;
            
            if (!name || !description) {
                req.error(400, 'Missing capability name or description');
            }
            
            const result = await blockchainService.registerCapability({
                name,
                description,
                category: category || 0
            });
            
            return `Capability ${name} registered successfully`;
            
        } catch (error) {
            cds.log('blockchain').error('Failed to register capability:', error);
            req.error(500, 'Failed to register capability', error.message);
        }
    });

    this.on('addAgentCapability', async (req) => {
        try {
            const { agentAddress, capabilityId } = req.data;
            
            if (!agentAddress || !capabilityId) {
                req.error(400, 'Missing agent address or capability ID');
            }
            
            const result = await blockchainService.addAgentCapability(agentAddress, capabilityId);
            
            return `Capability ${capabilityId} added to agent ${agentAddress}`;
            
        } catch (error) {
            cds.log('blockchain').error('Failed to add agent capability:', error);
            req.error(500, 'Failed to add agent capability', error.message);
        }
    });

    // Message Operations
    this.on('sendMessage', async (req) => {
        try {
            const { from, to, messageHash, protocol } = req.data;
            
            if (!from || !to || !messageHash) {
                req.error(400, 'Missing required message fields');
            }
            
            const result = await blockchainService.sendMessage({
                from,
                to,
                messageHash,
                protocol: protocol || 'default'
            });
            
            return `Message sent from ${from} to ${to}`;
            
        } catch (error) {
            cds.log('blockchain').error('Failed to send message:', error);
            req.error(500, 'Failed to send message', error.message);
        }
    });

    this.on('confirmMessageDelivery', async (req) => {
        try {
            const { messageHash } = req.data;
            
            if (!messageHash) {
                req.error(400, 'Missing message hash');
            }
            
            const result = await blockchainService.confirmMessageDelivery(messageHash);
            
            return `Message delivery confirmed for ${messageHash}`;
            
        } catch (error) {
            cds.log('blockchain').error('Failed to confirm message delivery:', error);
            req.error(500, 'Failed to confirm message delivery', error.message);
        }
    });

    // Workflow Operations
    this.on('deployWorkflow', async (req) => {
        try {
            const { workflowDefinition } = req.data;
            
            if (!workflowDefinition) {
                req.error(400, 'Missing workflow definition');
            }
            
            const result = await blockchainService.deployWorkflow(workflowDefinition);
            
            return `Workflow deployed successfully`;
            
        } catch (error) {
            cds.log('blockchain').error('Failed to deploy workflow:', error);
            req.error(500, 'Failed to deploy workflow', error.message);
        }
    });

    this.on('executeWorkflow', async (req) => {
        try {
            const { workflowId, parameters } = req.data;
            
            if (!workflowId) {
                req.error(400, 'Missing workflow ID');
            }
            
            const result = await blockchainService.executeWorkflow(workflowId, parameters || '{}');
            
            return `Workflow ${workflowId} executed successfully`;
            
        } catch (error) {
            cds.log('blockchain').error('Failed to execute workflow:', error);
            req.error(500, 'Failed to execute workflow', error.message);
        }
    });

    // Query Functions with caching
    this.on('getAgentInfo', async (req) => {
        try {
            const { agentAddress } = req.data;
            
            if (!agentAddress) {
                req.error(400, 'Missing agent address');
            }
            
            const result = await blockchainCache.executeWithCache(
                'agentInfo',
                { agentAddress },
                () => blockchainService.getAgentInfo(agentAddress)
            );
            
            return JSON.stringify(result);
            
        } catch (error) {
            cds.log('blockchain').error('Failed to get agent info:', error);
            req.error(500, 'Failed to get agent info', error.message);
        }
    });

    this.on('getAgentReputation', async (req) => {
        try {
            const { agentAddress } = req.data;
            
            if (!agentAddress) {
                req.error(400, 'Missing agent address');
            }
            
            const result = await blockchainCache.executeWithCache(
                'agentReputation',
                { agentAddress },
                () => blockchainService.getAgentReputation(agentAddress)
            );
            
            return result;
            
        } catch (error) {
            cds.log('blockchain').error('Failed to get agent reputation:', error);
            req.error(500, 'Failed to get agent reputation', error.message);
        }
    });

    this.on('getServiceInfo', async (req) => {
        try {
            const { serviceId } = req.data;
            
            if (!serviceId) {
                req.error(400, 'Missing service ID');
            }
            
            const result = await blockchainCache.executeWithCache(
                'serviceInfo',
                { serviceId },
                () => blockchainService.getServiceInfo(serviceId)
            );
            
            return JSON.stringify(result);
            
        } catch (error) {
            cds.log('blockchain').error('Failed to get service info:', error);
            req.error(500, 'Failed to get service info', error.message);
        }
    });

    this.on('getNetworkStats', async (req) => {
        try {
            const result = await blockchainService.getNetworkStats();
            
            // Update local stats
            const stats = await SELECT.one.from(BlockchainStats).orderBy({ timestamp: 'desc' });
            if (stats) {
                await UPDATE(BlockchainStats)
                    .set({
                        blockHeight: result.blockHeight || stats.blockHeight,
                        gasPrice: result.gasPrice || stats.gasPrice,
                        networkStatus: result.status || 'connected',
                        totalTransactions: result.totalTransactions || stats.totalTransactions,
                        timestamp: new Date()
                    })
                    .where({ ID: stats.ID });
            }
            
            return JSON.stringify(result);
            
        } catch (error) {
            cds.log('blockchain').error('Failed to get network stats:', error);
            req.error(500, 'Failed to get network stats', error.message);
        }
    });

    // Synchronization
    this.on('syncBlockchain', async (req) => {
        try {
            const result = await blockchainService.syncBlockchain();
            
            cds.log('blockchain').info('Blockchain synchronization completed', result);
            
            return {
                synced: result.synced || 0,
                pending: result.pending || 0,
                failed: result.failed || 0
            };
            
        } catch (error) {
            cds.log('blockchain').error('Failed to sync blockchain:', error);
            req.error(500, 'Failed to sync blockchain', error.message);
        }
    });

    // Reputation Operations
    this.on('endorsePeer', async (req) => {
        try {
            const { fromAgent, toAgent, amount, reason, context } = req.data;
            
            if (!fromAgent || !toAgent || !amount || !reason) {
                req.error(400, 'Missing required fields for endorsement');
            }
            
            // Call reputation exchange smart contract
            const result = await blockchainService.endorsePeer({
                fromAgent,
                toAgent,
                amount,
                reason,
                contextHash: this.hashContext(context)
            });
            
            return `Endorsement recorded on blockchain: ${result.transactionHash}`;
            
        } catch (error) {
            cds.log('blockchain').error('Failed to record endorsement:', error);
            req.error(500, 'Failed to record endorsement', error.message);
        }
    });
    
    this.on('updateAgentReputation', async (req) => {
        try {
            const { agentAddress, change, reason } = req.data;
            
            if (!agentAddress || change === undefined || !reason) {
                req.error(400, 'Missing required fields for reputation update');
            }
            
            const result = await blockchainService.applyReputationChange({
                agentAddress,
                change,
                reason
            });
            
            return `Reputation updated on blockchain: ${result.newReputation}`;
            
        } catch (error) {
            cds.log('blockchain').error('Failed to update reputation:', error);
            req.error(500, 'Failed to update reputation', error.message);
        }
    });
    
    this.on('getReputationHistory', async (req) => {
        try {
            const { agentAddress, fromBlock, toBlock } = req.data;
            
            if (!agentAddress) {
                req.error(400, 'Missing agent address');
            }
            
            const result = await blockchainService.getReputationHistory(
                agentAddress, 
                fromBlock || 0, 
                toBlock || 'latest'
            );
            
            return JSON.stringify(result);
            
        } catch (error) {
            cds.log('blockchain').error('Failed to get reputation history:', error);
            req.error(500, 'Failed to get reputation history', error.message);
        }
    });
    
    // Utility method for context hashing
    this.hashContext = function(context) {
        const crypto = require('crypto');
        return crypto.createHash('sha256').update(JSON.stringify(context)).digest('hex');
    }
    
    // SECURITY FIX: Ethereum address validation helper
    this.isValidEthereumAddress = function(address) {
        if (!address || typeof address !== 'string') {
            return false;
        }
        // Check if it matches Ethereum address format (0x followed by 40 hex characters)
        return /^0x[a-fA-F0-9]{40}$/.test(address);
    }

    // READ handlers for entities
    this.on('READ', 'BlockchainStats', async (req) => {
        try {
            // Refresh stats before reading
            await this.getNetworkStats();
            
            const stats = await SELECT.from(BlockchainStats).orderBy({ timestamp: 'desc' }).limit(1);
            
            return stats;
            
        } catch (error) {
            cds.log('blockchain').error('Failed to read blockchain stats:', error);
            return [];
        }
    });

    cds.log('blockchain').info('Blockchain service handlers initialized');

});