export const NETWORKS: Record<string, {
  chainId: number;
  name: string;
  rpcUrls: string[];
  currency: {
    name: string;
    symbol: string;
    decimals: number;
  };
}> = {
  localhost: {
    chainId: 31337,
    name: 'Localhost',
    rpcUrls: ['http://localhost:8545'],
    currency: {
      name: 'Ether',
      symbol: 'ETH',
      decimals: 18
    }
  },
  mainnet: {
    chainId: 1,
    name: 'Ethereum Mainnet',
    rpcUrls: [process.env.MAINNET_RPC_URL || 'https://cloudflare-eth.com'],
    currency: {
      name: 'Ether',
      symbol: 'ETH',
      decimals: 18
    }
  },
  sepolia: {
    chainId: 11155111,
    name: 'Sepolia Testnet',
    rpcUrls: [process.env.SEPOLIA_RPC_URL || 'https://rpc.sepolia.org'],
    currency: {
      name: 'Sepolia Ether',
      symbol: 'ETH',
      decimals: 18
    }
  },
  polygon: {
    chainId: 137,
    name: 'Polygon Mainnet',
    rpcUrls: ['https://polygon-rpc.com'],
    currency: {
      name: 'MATIC',
      symbol: 'MATIC',
      decimals: 18
    }
  }
};

export type NetworkName = keyof typeof NETWORKS;

export const DEFAULT_NETWORK: NetworkName = 'localhost';