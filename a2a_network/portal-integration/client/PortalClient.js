/**
 * A2A Portal Client Library
 * 
 * JavaScript client library for integrating with the A2A Agent Portal.
 * Provides easy-to-use methods for portal developers.
 */

class PortalClient {
    constructor(config = {}) {
        this.config = {
            apiBaseUrl: config.apiBaseUrl || 'http://localhost:3001/api/v1',
            wsUrl: config.wsUrl || 'ws://localhost:3002',
            timeout: config.timeout || 30000,
            retries: config.retries || 3,
            apiKey: config.apiKey,
            ...config
        };

        this.ws = null;
        this.eventCallbacks = new Map();
        this.subscriptions = new Map();
        this.connectionState = 'disconnected';
        
        // Auto-reconnect settings
        this.autoReconnect = config.autoReconnect !== false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = config.maxReconnectAttempts || 5;
        this.reconnectDelay = config.reconnectDelay || 1000;
    }

    // Connection Management
    async connect() {
        return new Promise((resolve, reject) => {
            if (this.connectionState === 'connected') {
                resolve();
                return;
            }

            this.connectionState = 'connecting';
            this.ws = new WebSocket(this.config.wsUrl);

            this.ws.onopen = () => {
                console.log('Connected to A2A Portal');
                this.connectionState = 'connected';
                this.reconnectAttempts = 0;
                resolve();
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.connectionState = 'error';
                if (this.connectionState === 'connecting') {
                    reject(error);
                }
            };

            this.ws.onclose = () => {
                console.log('Disconnected from A2A Portal');
                this.connectionState = 'disconnected';
                
                if (this.autoReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
                    setTimeout(() => {
                        this.reconnectAttempts++;
                        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                        this.connect().catch(console.error);
                    }, this.reconnectDelay * Math.pow(2, this.reconnectAttempts));
                }
            };

            setTimeout(() => {
                if (this.connectionState === 'connecting') {
                    reject(new Error('Connection timeout'));
                }
            }, this.config.timeout);
        });
    }

    disconnect() {
        this.autoReconnect = false;
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.connectionState = 'disconnected';
    }

    handleMessage(data) {
        const { type, ...messageData } = data;
        
        // Handle system messages
        switch (type) {
            case 'connection_established':
                this.connectionId = data.connectionId;
                break;
            case 'pong':
                // Handle ping response
                break;
            case 'subscription_confirmed':
                console.log('Subscription confirmed:', data);
                break;
            case 'error':
                console.error('Server error:', data.error);
                break;
            default:
                // Broadcast to event callbacks
                if (this.eventCallbacks.has(type)) {
                    this.eventCallbacks.get(type).forEach(callback => {
                        try {
                            callback(messageData);
                        } catch (error) {
                            console.error('Error in event callback:', error);
                        }
                    });
                }
        }
    }

    // Event Management
    on(eventType, callback) {
        if (!this.eventCallbacks.has(eventType)) {
            this.eventCallbacks.set(eventType, new Set());
        }
        this.eventCallbacks.get(eventType).add(callback);
    }

    off(eventType, callback) {
        if (this.eventCallbacks.has(eventType)) {
            this.eventCallbacks.get(eventType).delete(callback);
        }
    }

    // WebSocket Communication
    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        } else {
            throw new Error('WebSocket not connected');
        }
    }

    // HTTP API Methods
    async apiCall(method, endpoint, data = null) {
        const url = `${this.config.apiBaseUrl}${endpoint}`;
        const headers = {
            'Content-Type': 'application/json'
        };

        if (this.config.apiKey) {
            headers['Authorization'] = `Bearer ${this.config.apiKey}`;
        }

        const options = {
            method: method.toUpperCase(),
            headers
        };

        if (data && (method === 'POST' || method === 'PUT' || method === 'PATCH')) {
            options.body = JSON.stringify(data);
        }

        let lastError;
        for (let attempt = 0; attempt < this.config.retries; attempt++) {
            try {
                const response = await fetch(url, options);
                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.error || `HTTP ${response.status}`);
                }
                
                return result;
            } catch (error) {
                lastError = error;
                if (attempt < this.config.retries - 1) {
                    await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
                }
            }
        }
        throw lastError;
    }

    // Agent Management
    async getAgents(options = {}) {
        const params = new URLSearchParams();
        if (options.limit) params.append('limit', options.limit);
        if (options.offset) params.append('offset', options.offset);
        if (options.search) params.append('search', options.search);
        
        const query = params.toString();
        const endpoint = `/agents${query ? `?${query}` : ''}`;
        
        return this.apiCall('GET', endpoint);
    }

    async getAgent(agentId) {
        return this.apiCall('GET', `/agents/${agentId}`);
    }

    async getAgentProfile(agentId) {
        return this.apiCall('GET', `/agents/${agentId}/profile`);
    }

    async getAgentStats(agentId) {
        return this.apiCall('GET', `/agents/${agentId}/stats`);
    }

    async registerAgent(agentParams) {
        return this.apiCall('POST', '/agents', agentParams);
    }

    async updateAgent(agentId, updates) {
        return this.apiCall('PUT', `/agents/${agentId}`, updates);
    }

    async setAgentStatus(agentId, isActive) {
        return this.apiCall('PATCH', `/agents/${agentId}/status`, { isActive });
    }

    async searchAgents(criteria) {
        return this.apiCall('POST', '/agents/search', criteria);
    }

    async getAgentsByOwner(ownerAddress) {
        return this.apiCall('GET', `/agents/owner/${ownerAddress}`);
    }

    // Message Management
    async sendMessage(messageParams) {
        return this.apiCall('POST', '/messages', messageParams);
    }

    async getMessageHistory(agentId, options = {}) {
        const params = new URLSearchParams();
        if (options.limit) params.append('limit', options.limit);
        if (options.offset) params.append('offset', options.offset);
        
        const query = params.toString();
        const endpoint = `/messages/${agentId}${query ? `?${query}` : ''}`;
        
        return this.apiCall('GET', endpoint);
    }

    async getMessage(messageId) {
        return this.apiCall('GET', `/messages/by-id/${messageId}`);
    }

    // Reputation Management
    async getReputation(agentId) {
        return this.apiCall('GET', `/reputation/${agentId}`);
    }

    async getReputationLeaderboard(limit = 10) {
        return this.apiCall('GET', `/reputation/leaderboard?limit=${limit}`);
    }

    // Token Management
    async getTokenBalance(address) {
        return this.apiCall('GET', `/tokens/balance/${address}`);
    }

    async getTokenInfo() {
        return this.apiCall('GET', '/tokens/info');
    }

    // Governance
    async getProposals(status = 'active') {
        return this.apiCall('GET', `/governance/proposals?status=${status}`);
    }

    async getProposal(proposalId) {
        return this.apiCall('GET', `/governance/proposals/${proposalId}`);
    }

    // Analytics
    async getNetworkStats() {
        return this.apiCall('GET', '/analytics/network');
    }

    async getAgentActivity(agentId, days = 30) {
        return this.apiCall('GET', `/analytics/agents/${agentId}/activity?days=${days}`);
    }

    // Subscriptions
    subscribeToAgent(agentId) {
        this.send({
            type: 'subscribe_agent',
            agentId
        });
        
        const subscriptionKey = `agent_${agentId}`;
        this.subscriptions.set(subscriptionKey, { type: 'agent', agentId });
    }

    subscribeToMessages(agentId) {
        this.send({
            type: 'subscribe_messages',
            agentId
        });
        
        const subscriptionKey = `messages_${agentId}`;
        this.subscriptions.set(subscriptionKey, { type: 'messages', agentId });
    }

    unsubscribe(subscriptionKey) {
        this.subscriptions.delete(subscriptionKey);
        // Note: Server-side unsubscribe would need to be implemented
    }

    // Utility Methods
    ping() {
        this.send({ type: 'ping' });
    }

    getConnectionState() {
        return this.connectionState;
    }

    isConnected() {
        return this.connectionState === 'connected';
    }

    // Health Check
    async healthCheck() {
        return this.apiCall('GET', '/health');
    }

    // Batch Operations
    async batchGetAgents(agentIds) {
        const promises = agentIds.map(id => this.getAgent(id).catch(err => ({ error: err.message, id })));
        return Promise.all(promises);
    }

    async batchGetProfiles(agentIds) {
        const promises = agentIds.map(id => this.getAgentProfile(id).catch(err => ({ error: err.message, id })));
        return Promise.all(promises);
    }

    // Convenience Methods for Portal Developers
    async getAgentDashboardData(agentId) {
        try {
            const [profile, stats, reputation] = await Promise.all([
                this.getAgentProfile(agentId),
                this.getAgentStats(agentId),
                this.getReputation(agentId)
            ]);

            return {
                success: true,
                data: {
                    profile: profile.data,
                    stats: stats.data,
                    reputation: reputation.data
                }
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    async getOwnerDashboard(ownerAddress) {
        try {
            const [agents, balance] = await Promise.all([
                this.getAgentsByOwner(ownerAddress),
                this.getTokenBalance(ownerAddress)
            ]);

            return {
                success: true,
                data: {
                    agents: agents.data,
                    tokenBalance: balance.data.balance,
                    agentCount: agents.data.length
                }
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    async getNetworkOverview() {
        try {
            const [networkStats, leaderboard, proposals] = await Promise.all([
                this.getNetworkStats(),
                this.getReputationLeaderboard(5),
                this.getProposals('active')
            ]);

            return {
                success: true,
                data: {
                    network: networkStats.data,
                    topAgents: leaderboard.data,
                    activeProposals: proposals.data
                }
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    // Error Handling Helper
    handleError(error, context = '') {
        const errorMsg = `${context ? context + ': ' : ''}${error.message}`;
        console.error(errorMsg, error);
        
        // Emit error event for portal developers to handle
        if (this.eventCallbacks.has('error')) {
            this.eventCallbacks.get('error').forEach(callback => {
                callback({ message: errorMsg, error, context });
            });
        }
        
        return { success: false, error: errorMsg };
    }
}

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PortalClient };
} else if (typeof window !== 'undefined') {
    window.PortalClient = PortalClient;
}