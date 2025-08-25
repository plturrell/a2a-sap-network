const { ethers } = require('ethers');
const EventEmitter = require('events');
const WebSocket = require('ws');

const AgentManager = require('../services/AgentManager');
const MessageManager = require('../services/MessageManager');
const TokenManager = require('../services/TokenManager');
const GovernanceManager = require('../services/GovernanceManager');
const ScalabilityManager = require('../services/ScalabilityManager');
const ReputationManager = require('../services/ReputationManager');

const { A2AError, ErrorCode } = require('../utils/errors');
const { validateAddress, validateConfig } = require('../utils/validation');
const { NETWORKS, DEFAULT_NETWORK } = require('../constants/networks');
const { CONTRACT_ADDRESSES } = require('../constants/contracts');

/**
 * Connection states
 */
const ConnectionState = {
    DISCONNECTED: 'disconnected',
    CONNECTING: 'connecting',
    CONNECTED: 'connected',
    ERROR: 'error'
};

/**
 * Main A2A Network client for all SDK interactions
 */
class A2AClient extends EventEmitter {
    constructor(config) {
        super();

        // Validate configuration
        const validationResult = validateConfig(config);
        if (!validationResult.isValid) {
            throw new A2AError(
                ErrorCode.INVALID_CONFIG,
                `Invalid configuration: ${validationResult.errors?.join(', ')}`
            );
        }

        this.config = {
            ...config,
            network: config.network || DEFAULT_NETWORK,
            apiTimeout: config.apiTimeout || 30000,
            retryAttempts: config.retryAttempts || 3,
            autoReconnect: config.autoReconnect !== false
        };

        this.provider = null;
        this.signer = null;
        this.websocket = null;
        this.connectionState = ConnectionState.DISCONNECTED;

        // Contract instances
        this.contracts = new Map();

        // Event subscriptions
        this.subscriptions = new Map();

        // Initialize provider
        this.initializeProvider();

        // Initialize service managers
        this.agents = new AgentManager(this);
        this.messages = new MessageManager(this);
        this.tokens = new TokenManager(this);
        this.governance = new GovernanceManager(this);
        this.scalability = new ScalabilityManager(this);
        this.reputation = new ReputationManager(this);

        // Setup event handlers
        this.setupEventHandlers();
    }

    /**
     * Initialize provider based on configuration
     */
    initializeProvider() {
        const networkConfig = NETWORKS[this.config.network];
        if (!networkConfig) {
            throw new A2AError(
                ErrorCode.UNSUPPORTED_NETWORK,
                `Unsupported network: ${this.config.network}`
            );
        }

        if (this.config.provider) {
            this.provider = this.config.provider;
        } else if (this.config.rpcUrl) {
            this.provider = new ethers.providers.JsonRpcProvider(this.config.rpcUrl);
        } else if (networkConfig.rpcUrls?.length > 0) {
            this.provider = new ethers.providers.JsonRpcProvider(networkConfig.rpcUrls[0]);
        } else {
            throw new A2AError(
                ErrorCode.NO_PROVIDER,
                'No provider configuration found'
            );
        }

        // Set signer if private key provided
        if (this.config.privateKey) {
            this.signer = new ethers.Wallet(this.config.privateKey, this.provider);
        }
    }

    /**
     * Setup event handlers for client
     */
    setupEventHandlers() {
        // Provider event handlers
        this.provider.on('error', (error) => {
            this.emit('error', new A2AError(ErrorCode.PROVIDER_ERROR, error.message));
        });

        this.provider.on('network', (network) => {
            this.emit('networkChanged', network);
        });

        // WebSocket event handlers
        this.on('websocketConnected', () => {
            this.connectionState = ConnectionState.CONNECTED;
        });

        this.on('websocketDisconnected', () => {
            this.connectionState = ConnectionState.DISCONNECTED;
            if (this.config.autoReconnect) {
                this.reconnectWebSocket();
            }
        });
    }

    /**
     * Connect to A2A Network
     */
    async connect() {
        try {
            this.connectionState = ConnectionState.CONNECTING;
            this.emit('connecting');

            // Verify network connection
            const network = await this.provider.getNetwork();
            const expectedChainId = NETWORKS[this.config.network].chainId;

            if (network.chainId !== expectedChainId) {
                throw new A2AError(
                    ErrorCode.WRONG_NETWORK,
                    `Connected to wrong network. Expected ${expectedChainId}, got ${network.chainId}`
                );
            }

            // Initialize contracts
            await this.initializeContracts();

            // Connect WebSocket if configured
            if (this.config.websocketUrl) {
                await this.connectWebSocket();
            }

            this.connectionState = ConnectionState.CONNECTED;
            this.emit('connected', {
                network: this.config.network,
                chainId: network.chainId,
                address: await this.getAddress()
            });

        } catch (error) {
            this.connectionState = ConnectionState.ERROR;
            const a2aError = error instanceof A2AError
                ? error
                : new A2AError(ErrorCode.CONNECTION_FAILED, error.message);
            this.emit('error', a2aError);
            throw a2aError;
        }
    }

    /**
     * Disconnect from A2A Network
     */
    async disconnect() {
        try {
            // Close WebSocket connection
            if (this.websocket) {
                this.websocket.close();
                this.websocket = null;
            }

            // Clear subscriptions
            this.subscriptions.clear();

            // Clear contracts
            this.contracts.clear();

            this.connectionState = ConnectionState.DISCONNECTED;
            this.emit('disconnected');

        } catch (error) {
            this.emit('error', new A2AError(ErrorCode.DISCONNECTION_FAILED, error.message));
        }
    }

    /**
     * Initialize smart contracts
     */
    async initializeContracts() {
        const contractAddresses = CONTRACT_ADDRESSES[this.config.network];
        if (!contractAddresses) {
            throw new A2AError(
                ErrorCode.NO_CONTRACTS,
                `No contract addresses found for network: ${this.config.network}`
            );
        }

        // Import contract ABIs
        const abis = require('../constants/abis');

        // Initialize core contracts
        const contracts = [
            { name: 'AgentRegistry', address: contractAddresses.AGENT_REGISTRY, abi: abis.AGENT_REGISTRY_ABI },
            { name: 'MessageRouter', address: contractAddresses.MESSAGE_ROUTER, abi: abis.MESSAGE_ROUTER_ABI },
            { name: 'A2AToken', address: contractAddresses.A2A_TOKEN, abi: abis.A2A_TOKEN_ABI },
            { name: 'TimelockGovernor', address: contractAddresses.TIMELOCK_GOVERNOR, abi: abis.GOVERNANCE_ABI },
            { name: 'LoadBalancer', address: contractAddresses.LOAD_BALANCER, abi: abis.LOAD_BALANCER_ABI },
            { name: 'AIAgentMatcher', address: contractAddresses.AI_AGENT_MATCHER, abi: abis.AI_AGENT_MATCHER_ABI }
        ];

        for (const contract of contracts) {
            if (!validateAddress(contract.address)) {
                throw new A2AError(
                    ErrorCode.INVALID_CONTRACT_ADDRESS,
                    `Invalid contract address for ${contract.name}: ${contract.address}`
                );
            }

            const contractInstance = new ethers.Contract(
                contract.address,
                contract.abi,
                this.signer || this.provider
            );

            this.contracts.set(contract.name, contractInstance);
        }
    }

    /**
     * Connect WebSocket for real-time updates
     */
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            try {
                this.websocket = new WebSocket(this.config.websocketUrl);

                this.websocket.on('open', () => {
                    this.emit('websocketConnected');
                    resolve();
                });

                this.websocket.on('close', () => {
                    this.emit('websocketDisconnected');
                });

                this.websocket.on('error', (error) => {
                    this.emit('error', new A2AError(ErrorCode.WEBSOCKET_ERROR, error.message));
                    reject(error);
                });

                this.websocket.on('message', (data) => {
                    try {
                        const message = JSON.parse(data.toString());
                        this.handleWebSocketMessage(message);
                    } catch (error) {
                        this.emit('error', new A2AError(ErrorCode.INVALID_MESSAGE, 'Invalid WebSocket message'));
                    }
                });

                // Set connection timeout
                setTimeout(() => {
                    if (this.websocket.readyState !== WebSocket.OPEN) {
                        reject(new A2AError(ErrorCode.CONNECTION_TIMEOUT, 'WebSocket connection timeout'));
                    }
                }, this.config.apiTimeout);

            } catch (error) {
                reject(new A2AError(ErrorCode.WEBSOCKET_ERROR, error.message));
            }
        });
    }

    /**
     * Reconnect WebSocket with exponential backoff
     */
    async reconnectWebSocket() {
        let attempts = 0;
        const maxAttempts = this.config.retryAttempts || 3;

        while (attempts < maxAttempts) {
            try {
                const delay = Math.pow(2, attempts) * 1000; // Exponential backoff
                await new Promise(resolve => setTimeout(resolve, delay));

                await this.connectWebSocket();
                return;

            } catch (error) {
                attempts++;
                if (attempts >= maxAttempts) {
                    this.emit('error', new A2AError(
                        ErrorCode.RECONNECTION_FAILED,
                        `Failed to reconnect after ${maxAttempts} attempts`
                    ));
                }
            }
        }
    }

    /**
     * Handle incoming WebSocket messages
     */
    handleWebSocketMessage(message) {
        const { type, data } = message;

        switch (type) {
            case 'agent_update':
                this.emit('agentUpdate', data);
                break;
            case 'message_received':
                this.emit('messageReceived', data);
                break;
            case 'reputation_change':
                this.emit('reputationChange', data);
                break;
            case 'governance_proposal':
                this.emit('governanceProposal', data);
                break;
            case 'network_stats':
                this.emit('networkStats', data);
                break;
            default:
                this.emit('unknownMessage', message);
        }
    }

    /**
     * Get contract instance by name
     */
    getContract(name) {
        const contract = this.contracts.get(name);
        if (!contract) {
            throw new A2AError(ErrorCode.CONTRACT_NOT_FOUND, `Contract ${name} not found`);
        }
        return contract;
    }

    /**
     * Get current provider
     */
    getProvider() {
        return this.provider;
    }

    /**
     * Get current signer
     */
    getSigner() {
        return this.signer;
    }

    /**
     * Set signer (for wallet connections)
     */
    setSigner(signer) {
        this.signer = signer;

        // Update contract instances with new signer
        for (const [name, contract] of this.contracts) {
            this.contracts.set(name, contract.connect(signer));
        }
    }

    /**
     * Get current wallet address
     */
    async getAddress() {
        if (!this.signer) return null;
        return await this.signer.getAddress();
    }

    /**
     * Get current connection state
     */
    getConnectionState() {
        return this.connectionState;
    }

    /**
     * Get client configuration
     */
    getConfig() {
        return { ...this.config };
    }

    /**
     * Subscribe to contract events
     */
    async subscribe(contractName, eventName, callback) {
        const contract = this.getContract(contractName);
        const subscriptionId = `${contractName}_${eventName}_${Date.now()}`;

        const listener = (...args) => {
            callback(...args);
        };

        contract.on(eventName, listener);
        this.subscriptions.set(subscriptionId, { contract, eventName, listener });

        return subscriptionId;
    }

    /**
     * Unsubscribe from contract events
     */
    unsubscribe(subscriptionId) {
        const subscription = this.subscriptions.get(subscriptionId);
        if (subscription) {
            subscription.contract.off(subscription.eventName, subscription.listener);
            this.subscriptions.delete(subscriptionId);
        }
    }

    /**
     * Send WebSocket message
     */
    sendWebSocketMessage(message) {
        if (this.websocket?.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(message));
        } else {
            throw new A2AError(ErrorCode.WEBSOCKET_NOT_CONNECTED, 'WebSocket not connected');
        }
    }

    /**
     * Get network information
     */
    async getNetworkInfo() {
        const network = await this.provider.getNetwork();
        const blockNumber = await this.provider.getBlockNumber();
        const gasPrice = await this.provider.getGasPrice();

        return {
            name: network.name,
            chainId: network.chainId,
            blockNumber,
            gasPrice: gasPrice.toString(),
            contracts: Object.fromEntries(this.contracts.entries())
        };
    }

    /**
     * Health check for the client
     */
    async healthCheck() {
        try {
            const networkInfo = await this.getNetworkInfo();
            const address = await this.getAddress();

            return {
                status: 'healthy',
                details: {
                    connectionState: this.connectionState,
                    network: networkInfo,
                    address,
                    contractsLoaded: this.contracts.size,
                    subscriptions: this.subscriptions.size,
                    websocketConnected: this.websocket?.readyState === WebSocket.OPEN
                }
            };
        } catch (error) {
            return {
                status: 'unhealthy',
                details: {
                    error: error.message,
                    connectionState: this.connectionState
                }
            };
        }
    }
}

module.exports = A2AClient;