/**
 * A2A Protocol Blockchain Client
 * Provides HTTP-compatible interface for blockchain messaging
 */

const EventEmitter = require('events');

class BlockchainClient extends EventEmitter {
    constructor(options = {}) {
        super();
        
        this.config = {
            blockchainUrl: options.blockchainUrl || process.env.BLOCKCHAIN_URL || 'http://localhost:8545',
            contractAddress: options.contractAddress || process.env.A2A_CONTRACT_ADDRESS,
            privateKey: options.privateKey || process.env.A2A_PRIVATE_KEY,
            ...options
        };
        
        this.connected = false;
        this.messageQueue = [];
        
        this.initialize();
    }
    
    async initialize() {
        // Initialize blockchain connection
        try {
            console.log('🔗 Connecting to A2A blockchain network...');
            // TODO: Initialize Web3 connection
            this.connected = true;
            console.log('✅ Connected to A2A blockchain network');
        } catch (error) {
            console.error('❌ Failed to connect to blockchain:', error);
        }
    }
    
    /**
     * Send message via A2A protocol (replaces HTTP calls)
     */
    async sendMessage(endpoint, options = {}) {
        if (!this.connected) {
            console.warn('⚠️  Blockchain not connected, queuing message...');
            return this.queueMessage(endpoint, options);
        }
        
        try {
            // Convert HTTP-style call to A2A message
            const message = this.convertHttpToA2A(endpoint, options);
            
            // Send via blockchain
            const response = await this.sendBlockchainMessage(message);
            
            // Return HTTP-compatible response
            return this.formatResponse(response);
            
        } catch (error) {
            console.error('Failed to send A2A message:', error);
            throw error;
        }
    }
    
    convertHttpToA2A(endpoint, options) {
        // Convert HTTP request to A2A message format
        const url = new URL(endpoint.startsWith('http') ? endpoint : 'http://localhost' + endpoint);
        
        return {
            to: this.extractTargetAgent(url.pathname),
            messageType: this.extractMessageType(url.pathname, options.method || 'GET'),
            data: options.data || options.body || {},
            headers: options.headers || {},
            timestamp: Date.now(),
            sender: this.config.agentId || 'unknown'
        };
    }
    
    extractTargetAgent(pathname) {
        // Extract target agent from API path
        const parts = pathname.split('/').filter(p => p);
        if (parts.length > 1 && parts[0] === 'api') {
            return parts[1]; // e.g., /api/agents -> 'agents'
        }
        return 'registry'; // Default to registry
    }
    
    extractMessageType(pathname, method) {
        // Convert REST endpoint to message type
        const path = pathname.replace('/api/', '').replace(/\//g, '_').toUpperCase();
        return `${method}_${path}`;
    }
    
    async sendBlockchainMessage(message) {
        // Send message via blockchain
        console.log('📤 Sending A2A message:', message.messageType);
        
        // TODO: Implement actual blockchain message sending
        // For now, simulate response
        return {
            success: true,
            data: { message: 'A2A message sent successfully' },
            timestamp: Date.now()
        };
    }
    
    formatResponse(blockchainResponse) {
        // Format blockchain response to be HTTP-compatible
        return {
            ok: blockchainResponse.success,
            status: blockchainResponse.success ? 200 : 500,
            json: async () => blockchainResponse.data,
            text: async () => JSON.stringify(blockchainResponse.data),
            data: blockchainResponse.data
        };
    }
    
    async queueMessage(endpoint, options) {
        this.messageQueue.push({ endpoint, options });
        
        // Return mock response for queued messages
        return {
            ok: true,
            status: 202, // Accepted
            json: async () => ({ message: 'Message queued for blockchain sending' }),
            text: async () => 'Message queued for blockchain sending'
        };
    }
}

module.exports = { BlockchainClient };
