import { ethers } from 'ethers';

/**
 * Common types and interfaces used across the SDK
 */

export interface A2AClientConfig {
    network: string;
    rpcUrl?: string;
    websocketUrl?: string;
    provider?: ethers.providers.Provider;
    privateKey?: string;
    apiTimeout?: number;
    retryAttempts?: number;
    autoReconnect?: boolean;
    rateLimits?: {
        requestsPerSecond?: number;
        requestsPerMinute?: number;
        requestsPerHour?: number;
    };
    caching?: {
        enabled: boolean;
        ttl: number; // Time to live in seconds
        maxSize: number; // Maximum cache entries
    };
    logging?: {
        level: 'debug' | 'info' | 'warn' | 'error';
        output?: 'console' | 'file';
        filePath?: string;
    };
}

export interface NetworkConfig {
    name: string;
    chainId: number;
    rpcUrls: string[];
    blockExplorerUrls?: string[];
    nativeCurrency: {
        name: string;
        symbol: string;
        decimals: number;
    };
    testnet?: boolean;
}

export enum ConnectionState {
    DISCONNECTED = 'disconnected',
    CONNECTING = 'connecting',
    CONNECTED = 'connected',
    ERROR = 'error'
}

export interface PaginationOptions {
    limit?: number;
    offset?: number;
    sortBy?: string;
    sortOrder?: 'asc' | 'desc';
}

export interface ValidationResult {
    isValid: boolean;
    errors?: string[];
    warnings?: string[];
}

export interface TransactionOptions {
    gasLimit?: ethers.BigNumberish;
    gasPrice?: ethers.BigNumberish;
    maxFeePerGas?: ethers.BigNumberish;
    maxPriorityFeePerGas?: ethers.BigNumberish;
    nonce?: number;
    value?: ethers.BigNumberish;
}

export interface TransactionResult {
    transactionHash: string;
    blockNumber?: number;
    gasUsed?: ethers.BigNumber;
    effectiveGasPrice?: ethers.BigNumber;
    status?: number;
    confirmations?: number;
}

export interface EventSubscription {
    id: string;
    contractName: string;
    eventName: string;
    filter?: any;
    callback: (...args: any[]) => void;
    isActive: boolean;
    createdAt: Date;
}

export interface APIResponse<T = any> {
    success: boolean;
    data?: T;
    error?: {
        code: string;
        message: string;
        details?: any;
    };
    metadata?: {
        totalCount?: number;
        currentPage?: number;
        totalPages?: number;
        timestamp: Date;
        requestId: string;
    };
}

export interface RateLimitInfo {
    limit: number;
    remaining: number;
    resetTime: Date;
    retryAfter?: number;
}

export interface CacheEntry<T = any> {
    key: string;
    value: T;
    timestamp: Date;
    expiresAt: Date;
    accessCount: number;
    lastAccessed: Date;
}

export interface MetricsData {
    totalRequests: number;
    successfulRequests: number;
    failedRequests: number;
    averageResponseTime: number;
    peakResponseTime: number;
    requestsPerSecond: number;
    errorRate: number;
    cacheHitRate?: number;
    lastUpdated: Date;
}

export interface HealthStatus {
    status: 'healthy' | 'degraded' | 'unhealthy';
    uptime: number;
    version: string;
    network: {
        connected: boolean;
        chainId?: number;
        blockNumber?: number;
    };
    services: {
        [serviceName: string]: {
            status: 'healthy' | 'degraded' | 'unhealthy';
            latency?: number;
            errorRate?: number;
            lastCheck: Date;
        };
    };
    timestamp: Date;
}

export interface ContractInfo {
    name: string;
    address: string;
    abi: any[];
    deployedAt?: {
        blockNumber: number;
        transactionHash: string;
        timestamp: Date;
    };
    verified?: boolean;
    proxyContract?: boolean;
    implementationAddress?: string;
}

export interface GasEstimate {
    gasLimit: ethers.BigNumber;
    gasPrice: ethers.BigNumber;
    maxFeePerGas?: ethers.BigNumber;
    maxPriorityFeePerGas?: ethers.BigNumber;
    estimatedCost: ethers.BigNumber;
    estimatedTimeSeconds?: number;
}

export interface RetryConfig {
    attempts: number;
    delay: number; // Base delay in ms
    backoffMultiplier: number; // Exponential backoff multiplier
    maxDelay: number; // Maximum delay in ms
    jitter: boolean; // Add random jitter to prevent thundering herd
}

export interface WebSocketMessage {
    type: string;
    data: any;
    timestamp: Date;
    requestId?: string;
    channel?: string;
}

export interface SDKError {
    code: string;
    message: string;
    details?: any;
    timestamp: Date;
    requestId?: string;
    stackTrace?: string;
}

export interface BatchRequest {
    id: string;
    method: string;
    params: any[];
    timestamp: Date;
}

export interface BatchResponse<T = any> {
    id: string;
    success: boolean;
    result?: T;
    error?: SDKError;
    executionTime: number;
}

// Utility types
export type Address = string;
export type Hash = string;
export type Bytes32 = string;
export type Timestamp = number | Date;

export type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;

export type OptionalFields<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

export type EventCallback<T = any> = (event: T) => void | Promise<void>;

export type AsyncCallback<T = any, R = void> = (data: T) => Promise<R>;

// Configuration validation schemas
export interface ConfigSchema {
    required: string[];
    optional: string[];
    validation: {
        [key: string]: (value: any) => ValidationResult;
    };
}

// Performance monitoring
export interface PerformanceMetrics {
    operationName: string;
    startTime: Date;
    endTime?: Date;
    duration?: number;
    success: boolean;
    errorMessage?: string;
    metadata?: {
        [key: string]: any;
    };
}

// Security and encryption
export interface EncryptionKeyPair {
    publicKey: string;
    privateKey: string;
    algorithm: string;
    keySize: number;
    createdAt: Date;
    expiresAt?: Date;
}

export interface SignatureData {
    signature: string;
    messageHash: string;
    signer: string;
    timestamp: Date;
    nonce?: number;
}

// Network specific configurations
export interface NetworkFeatures {
    supportsEIP1559: boolean; // London hard fork gas pricing
    supportsCreate2: boolean; // Deterministic contract addresses
    supportsMetaTransactions: boolean; // Gasless transactions
    supportsMulticall: boolean; // Batch multiple calls
    maxBlockGasLimit: number;
    averageBlockTime: number; // in seconds
    finalityBlocks: number; // blocks for finality
}

// API versioning
export interface APIVersion {
    version: string;
    releaseDate: Date;
    deprecationDate?: Date;
    supportedFeatures: string[];
    breakingChanges?: string[];
}

export const DEFAULT_CONFIG: Partial<A2AClientConfig> = {
    apiTimeout: 30000,
    retryAttempts: 3,
    autoReconnect: true,
    rateLimits: {
        requestsPerSecond: 10,
        requestsPerMinute: 600,
        requestsPerHour: 36000
    },
    caching: {
        enabled: true,
        ttl: 300, // 5 minutes
        maxSize: 1000
    },
    logging: {
        level: 'info',
        output: 'console'
    }
};

export const SUPPORTED_NETWORKS = [
    'mainnet',
    'goerli',
    'sepolia',
    'polygon',
    'polygon-mumbai',
    'bsc',
    'bsc-testnet',
    'arbitrum',
    'arbitrum-goerli',
    'optimism',
    'optimism-goerli',
    'localhost'
] as const;

export type SupportedNetwork = typeof SUPPORTED_NETWORKS[number];

// Event system
export interface EventMap {
    connected: { network: string; chainId: number; address?: string };
    disconnected: {};
    error: SDKError;
    networkChanged: { chainId: number; networkName: string };
    accountChanged: { address: string };
    blockUpdated: { blockNumber: number; blockHash: string };
    transactionConfirmed: TransactionResult;
    messageReceived: any;
    agentUpdate: any;
    reputationChange: any;
    governanceProposal: any;
    networkStats: any;
}

export type EventName = keyof EventMap;
export type EventData<T extends EventName> = EventMap[T];