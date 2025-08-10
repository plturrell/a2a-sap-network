import { NETWORKS, NetworkName } from '../constants/networks';
import { CONTRACT_ADDRESSES } from '../constants/contracts';
import { A2AClientConfig } from '../types/common';
import { A2AError, ErrorCode } from './errors';

/**
 * Securely load configuration from environment variables with validation
 */
export function loadConfig(): Partial<A2AClientConfig> {
  const network = (process.env.DEFAULT_NETWORK || 'localhost') as NetworkName;
  
  // Validate network selection
  if (!Object.keys(NETWORKS).includes(network)) {
    throw new A2AError(ErrorCode.INVALID_CONFIG, `Unsupported network: ${network}`);
  }
  
  const rpcUrl = getRpcUrlForNetwork(network);
  const privateKey = getPrivateKey();
  
  return {
    network,
    rpcUrl,
    websocketUrl: process.env.WEBSOCKET_URL,
    privateKey,
    apiTimeout: parseInt(process.env.API_TIMEOUT || '30000'),
    retryAttempts: parseInt(process.env.RETRY_ATTEMPTS || '3'),
    autoReconnect: process.env.AUTO_RECONNECT !== 'false',
    logging: {
      level: (process.env.LOG_LEVEL as 'debug' | 'info' | 'warn' | 'error') || 'info',
      output: process.env.LOG_TO_FILE === 'true' ? 'file' : 'console',
      filePath: process.env.LOG_FILE_PATH
    },
    caching: {
      enabled: process.env.CACHE_ENABLED !== 'false',
      ttl: parseInt(process.env.CACHE_TTL || '300'),
      maxSize: parseInt(process.env.CACHE_MAX_SIZE || '1000')
    }
  };
}

/**
 * Securely get RPC URL for network with fallbacks
 */
function getRpcUrlForNetwork(network: NetworkName): string {
  // For localhost, always use the hardcoded URL
  if (network === 'localhost') {
    return NETWORKS.localhost.rpcUrl;
  }
  
  // Check for network-specific URL first
  const networkSpecificUrl = process.env[`${network.toUpperCase()}_RPC_URL`];
  if (networkSpecificUrl && isValidUrl(networkSpecificUrl)) {
    return networkSpecificUrl;
  }
  
  // Fall back to provider keys
  const infuraKey = process.env.INFURA_PROJECT_ID;
  const alchemyKey = process.env.ALCHEMY_API_KEY;
  
  if (infuraKey && isValidKey(infuraKey)) {
    return getInfuraUrl(network, infuraKey);
  }
  
  if (alchemyKey && isValidKey(alchemyKey)) {
    return getAlchemyUrl(network, alchemyKey);
  }
  
  // Use public endpoints as last resort (not recommended for production)
  if (process.env.NODE_ENV !== 'production') {
    return getPublicRpcUrl(network);
  }
  
  throw new A2AError(
    ErrorCode.INVALID_CONFIG, 
    `No valid RPC URL found for network ${network}. Please set INFURA_PROJECT_ID, ALCHEMY_API_KEY, or ${network.toUpperCase()}_RPC_URL`
  );
}

/**
 * Securely handle private key with validation
 */
function getPrivateKey(): string | undefined {
  const privateKey = process.env.PRIVATE_KEY;
  
  if (!privateKey) {
    return undefined;
  }
  
  // Validate private key format
  if (!isValidPrivateKey(privateKey)) {
    throw new A2AError(
      ErrorCode.INVALID_CONFIG,
      'Invalid private key format - should be 0x followed by 64 hex characters'
    );
  }
  
  return privateKey;
}

/**
 * Validate private key format
 */
function isValidPrivateKey(key: string): boolean {
  return key.startsWith('0x') && key.length === 66 && /^0x[a-fA-F0-9]{64}$/.test(key);
}

/**
 * Validate URL format
 */
function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return url.startsWith('https://') || url.startsWith('http://');
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Invalid URL';
    console.warn('Invalid URL format:', {
      url: url.substring(0, 50) + (url.length > 50 ? '...' : ''), // Truncate for security
      error: errorMessage
    });
    return false; // Return false for invalid URLs
  }
}

/**
 * Validate API key format
 */
function isValidKey(key: string): boolean {
  return key.length > 10 && /^[a-zA-Z0-9_-]+$/.test(key);
}

function getInfuraUrl(network: NetworkName, key: string): string {
  const urls: Record<string, string> = {
    mainnet: `https://mainnet.infura.io/v3/${key}`,
    sepolia: `https://sepolia.infura.io/v3/${key}`,
    polygon: `https://polygon-mainnet.infura.io/v3/${key}`
  };
  
  return urls[network] || urls.mainnet;
}

function getAlchemyUrl(network: NetworkName, key: string): string {
  const urls: Record<string, string> = {
    mainnet: `https://eth-mainnet.g.alchemy.com/v2/${key}`,
    sepolia: `https://eth-sepolia.g.alchemy.com/v2/${key}`,
    polygon: `https://polygon-mainnet.g.alchemy.com/v2/${key}`
  };
  
  return urls[network] || urls.mainnet;
}

function getPublicRpcUrl(network: NetworkName): string {
  const urls: Record<string, string> = {
    mainnet: 'https://cloudflare-eth.com',
    sepolia: 'https://rpc.sepolia.org',
    polygon: 'https://polygon-rpc.com'
  };
  
  return urls[network] || urls.mainnet;
}

/**
 * Get contract addresses with environment variable overrides and validation
 */
export function getContractAddresses(network: NetworkName): typeof CONTRACT_ADDRESSES.localhost {
  const baseAddresses = CONTRACT_ADDRESSES[network];
  const envPrefix = network.toUpperCase();
  
  const addresses = {
    AgentRegistry: process.env[`${envPrefix}_AGENT_REGISTRY`] || baseAddresses.AgentRegistry,
    MessageRouter: process.env[`${envPrefix}_MESSAGE_ROUTER`] || baseAddresses.MessageRouter,
    TokenManager: process.env[`${envPrefix}_TOKEN_MANAGER`] || baseAddresses.TokenManager,
    Governance: process.env[`${envPrefix}_GOVERNANCE`] || baseAddresses.Governance
  };
  
  // Validate contract addresses for non-localhost networks
  if (network !== 'localhost') {
    const emptyAddresses = Object.entries(addresses)
      .filter(([, address]) => !address || address === '')
      .map(([name]) => name);
    
    if (emptyAddresses.length > 0) {
      throw new A2AError(
        ErrorCode.INVALID_CONFIG,
        `Missing contract addresses for ${network}: ${emptyAddresses.join(', ')}`
      );
    }
    
    // Validate address format
    Object.entries(addresses).forEach(([name, address]) => {
      if (!isValidAddress(address)) {
        throw new A2AError(
          ErrorCode.INVALID_CONFIG,
          `Invalid contract address for ${name} on ${network}: ${address}`
        );
      }
    });
  }
  
  return addresses;
}

/**
 * Validate Ethereum address format
 */
function isValidAddress(address: string): boolean {
  return /^0x[a-fA-F0-9]{40}$/.test(address);
}

/**
 * Validate configuration with comprehensive checks
 */
export function validateConfig(config: Partial<A2AClientConfig>): { isValid: boolean; errors: string[] } {
  const errors: string[] = [];
  
  if (!config.network) {
    errors.push('Network is required');
  } else if (!Object.keys(NETWORKS).includes(config.network)) {
    errors.push(`Unsupported network: ${config.network}`);
  }
  
  if (!config.rpcUrl && !config.provider) {
    errors.push('Either rpcUrl or provider is required');
  }
  
  if (config.rpcUrl && !isValidUrl(config.rpcUrl)) {
    errors.push('Invalid RPC URL format');
  }
  
  if (config.privateKey && !isValidPrivateKey(config.privateKey)) {
    errors.push('Invalid private key format');
  }
  
  if (config.apiTimeout && (config.apiTimeout < 1000 || config.apiTimeout > 300000)) {
    errors.push('API timeout must be between 1000ms and 300000ms');
  }
  
  if (config.retryAttempts && (config.retryAttempts < 0 || config.retryAttempts > 10)) {
    errors.push('Retry attempts must be between 0 and 10');
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
}

/**
 * Get network-specific configuration with validation
 */
export function getNetworkConfig(network: NetworkName) {
  if (!Object.keys(NETWORKS).includes(network)) {
    throw new A2AError(ErrorCode.UNSUPPORTED_NETWORK, `Unsupported network: ${network}`);
  }
  
  const networkInfo = NETWORKS[network];
  const contracts = getContractAddresses(network);
  
  return {
    ...networkInfo,
    contracts
  };
}