import { ethers } from 'ethers';
import { EventEmitter } from 'events';
import WebSocket from 'ws';

import { AgentManager } from '../services/agentManager';
import { MessageManager } from '../services/messageManager';
import { TokenManager } from '../services/tokenManager';
import { GovernanceManager } from '../services/governanceManager';
import { ScalabilityManager } from '../services/scalabilityManager';
import { ReputationManager } from '../services/reputationManager';

import { A2AClientConfig, ConnectionState, ContractEvent, WebSocketEventData, ProviderErrorEvent, NetworkChangedEvent } from '../types/common';
import { A2AError, ErrorCode } from '../utils/errors';
import { isValidAddress, validateConfig } from '../utils/validation';
import { NETWORKS, DEFAULT_NETWORK } from '../constants/networks';
import { CONTRACT_ADDRESSES } from '../constants/contracts';

/**
 * Main A2A Network client for all SDK interactions
 */
export class A2AClient extends EventEmitter {
    private config: A2AClientConfig;
    private provider!: ethers.Provider;
    private signer?: ethers.Signer;
    private websocket?: WebSocket;
    private connectionState: ConnectionState = ConnectionState.DISCONNECTED;
    
    // Service managers
    public readonly agents: AgentManager;
    public readonly messages: MessageManager;
    public readonly tokens: TokenManager;
    public readonly governance: GovernanceManager;
    public readonly scalability: ScalabilityManager;
    public readonly reputation: ReputationManager;

    // Contract instances
    private contracts: Map<string, ethers.Contract> = new Map();
    
    // Event subscriptions
    private subscriptions: Map<string, {
        contract: ethers.Contract;
        eventName: string;
        listener: (...args: unknown[]) => void;
    }> = new Map();
    
    constructor(config: A2AClientConfig) {
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
    private initializeProvider(): void {
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
            this.provider = new ethers.JsonRpcProvider(this.config.rpcUrl);
        } else if (networkConfig.rpcUrls?.length > 0) {
            this.provider = new ethers.JsonRpcProvider(networkConfig.rpcUrls[0]);
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
    private setupEventHandlers(): void {
        // Provider event handlers
        this.provider.on('error', (error: ProviderErrorEvent) => {
            this.emit('error', new A2AError(ErrorCode.PROVIDER_ERROR, error.message));
        });

        this.provider.on('network', (network: NetworkChangedEvent) => {
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
    async connect(): Promise<void> {
        try {
            this.connectionState = ConnectionState.CONNECTING;
            this.emit('connecting');

            // Verify network connection
            const network = await this.provider.getNetwork();
            const expectedChainId = NETWORKS[this.config.network].chainId;
            
            if (Number(network.chainId) !== expectedChainId) {
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

        } catch (error: unknown) {
            this.connectionState = ConnectionState.ERROR;
            const a2aError = error instanceof A2AError 
                ? error 
                : new A2AError(ErrorCode.CONNECTION_FAILED, error instanceof Error ? error.message : 'Connection failed');
            this.emit('error', a2aError);
            throw a2aError;
        }
    }

    /**
     * Disconnect from A2A Network
     */
    async disconnect(): Promise<void> {
        try {
            // Close WebSocket connection
            if (this.websocket) {
                this.websocket.close();
                this.websocket = undefined;
            }

            // Clear subscriptions
            this.subscriptions.clear();
            
            // Clear contracts
            this.contracts.clear();

            this.connectionState = ConnectionState.DISCONNECTED;
            this.emit('disconnected');

        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : 'Disconnection failed';
            this.emit('error', new A2AError(ErrorCode.DISCONNECTION_FAILED, message));
        }
    }

    /**
     * Initialize smart contracts
     */
    private async initializeContracts(): Promise<void> {
        const contractAddresses = CONTRACT_ADDRESSES[this.config.network];
        if (!contractAddresses) {
            throw new A2AError(
                ErrorCode.NO_CONTRACTS,
                `No contract addresses found for network: ${this.config.network}`
            );
        }

        // Import contract ABIs
        const { 
            AGENT_REGISTRY_ABI,
            MESSAGE_ROUTER_ABI,
            A2A_TOKEN_ABI,
            GOVERNANCE_ABI,
            LOAD_BALANCER_ABI,
            AI_AGENT_MATCHER_ABI
        } = await import('../constants/abis');

        // Initialize core contracts
        const contracts = [
            { name: 'AgentRegistry', address: contractAddresses.AGENT_REGISTRY, abi: AGENT_REGISTRY_ABI },
            { name: 'MessageRouter', address: contractAddresses.MESSAGE_ROUTER, abi: MESSAGE_ROUTER_ABI },
            { name: 'A2AToken', address: contractAddresses.A2A_TOKEN, abi: A2A_TOKEN_ABI },
            { name: 'TimelockGovernor', address: contractAddresses.TIMELOCK_GOVERNOR, abi: GOVERNANCE_ABI },
            { name: 'LoadBalancer', address: contractAddresses.LOAD_BALANCER, abi: LOAD_BALANCER_ABI },
            { name: 'AIAgentMatcher', address: contractAddresses.AI_AGENT_MATCHER, abi: AI_AGENT_MATCHER_ABI }
        ];

        for (const contract of contracts) {
            if (!contract.address || !isValidAddress(contract.address)) {
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
    private async connectWebSocket(): Promise<void> {
        return new Promise((resolve, reject) => {
            try {
                this.websocket = new WebSocket(this.config.websocketUrl!);

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
                    if (this.websocket?.readyState !== WebSocket.OPEN) {
                        reject(new A2AError(ErrorCode.CONNECTION_TIMEOUT, 'WebSocket connection timeout'));
                    }
                }, this.config.apiTimeout);

            } catch (error: unknown) {
                const message = error instanceof Error ? error.message : 'WebSocket error';
                reject(new A2AError(ErrorCode.WEBSOCKET_ERROR, message));
            }
        });
    }

    /**
     * Reconnect WebSocket with exponential backoff
     */
    private async reconnectWebSocket(): Promise<void> {
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
    private handleWebSocketMessage(message: WebSocketEventData): void {
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
    getContract(name: string): ethers.Contract {
        const contract = this.contracts.get(name);
        if (!contract) {
            throw new A2AError(ErrorCode.CONTRACT_NOT_FOUND, `Contract ${name} not found`);
        }
        return contract;
    }

    /**
     * Get current provider
     */
    getProvider(): ethers.Provider {
        return this.provider;
    }

    /**
     * Get current signer
     */
    getSigner(): ethers.Signer | undefined {
        return this.signer;
    }

    /**
     * Set signer (for wallet connections)
     */
    setSigner(signer: ethers.Signer): void {
        this.signer = signer;
        
        // Update contract instances with new signer
        for (const [name, contract] of this.contracts) {
            this.contracts.set(name, contract.connect(signer));
        }
    }

    /**
     * Get current wallet address
     */
    async getAddress(): Promise<string | null> {
        if (!this.signer) return null;
        return await this.signer.getAddress();
    }

    /**
     * Get current connection state
     */
    getConnectionState(): ConnectionState {
        return this.connectionState;
    }

    /**
     * Get client configuration
     */
    getConfig(): A2AClientConfig {
        return { ...this.config };
    }

    /**
     * Subscribe to contract events
     */
    async subscribe(contractName: string, eventName: string, callback: (...args: unknown[]) => void): Promise<string> {
        const contract = this.getContract(contractName);
        const subscriptionId = `${contractName}_${eventName}_${Date.now()}`;
        
        const listener = (...args: unknown[]) => {
            callback(...args);
        };

        contract.on(eventName, listener);
        this.subscriptions.set(subscriptionId, { contract, eventName, listener });
        
        return subscriptionId;
    }

    /**
     * Unsubscribe from contract events
     */
    unsubscribe(subscriptionId: string): void {
        const subscription = this.subscriptions.get(subscriptionId);
        if (subscription) {
            subscription.contract.off(subscription.eventName, subscription.listener);
            this.subscriptions.delete(subscriptionId);
        }
    }

    /**
     * Send WebSocket message
     */
    sendWebSocketMessage(message: Record<string, unknown>): void {
        if (this.websocket?.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(message));
        } else {
            throw new A2AError(ErrorCode.WEBSOCKET_NOT_CONNECTED, 'WebSocket not connected');
        }
    }

    /**
     * Get network information
     */
    async getNetworkInfo(): Promise<{
        name: string;
        chainId: bigint;
        blockNumber: number;
        gasPrice: string;
        contracts: Record<string, ethers.Contract>;
    }> {
        const network = await this.provider.getNetwork();
        const blockNumber = await this.provider.getBlockNumber();
        const feeData = await this.provider.getFeeData();

        return {
            name: network.name,
            chainId: network.chainId,
            blockNumber,
            gasPrice: feeData.gasPrice?.toString() || '0',
            contracts: Object.fromEntries(this.contracts.entries())
        };
    }

    /**
     * Health check for the client
     */
    async healthCheck(): Promise<{ 
        status: string; 
        details: {
            connectionState: ConnectionState;
            network: {
                name: string;
                chainId: bigint;
                blockNumber: number;
                gasPrice: string;
                contracts: Record<string, ethers.Contract>;
            };
            address: string | null;
            contractsLoaded: number;
            subscriptions: number;
            websocketConnected: boolean;
        } | {
            error: string;
            connectionState: ConnectionState;
        }
    }> {
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
        } catch (error: unknown) {
            return {
                status: 'unhealthy',
                details: {
                    error: error instanceof Error ? error.message : 'Unknown error',
                    connectionState: this.connectionState
                }
            };
        }
    }
}