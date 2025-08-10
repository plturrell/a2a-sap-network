export interface Token {
  address: string;
  symbol: string;
  name: string;
  decimals: number;
  totalSupply: bigint;
}

export interface TokenTransfer {
  from: string;
  to: string;
  amount: bigint;
  transactionHash: string;
  blockNumber: number;
  timestamp: Date;
}

export interface TokenAllowance {
  owner: string;
  spender: string;
  amount: bigint;
}

export interface TokenMetadata {
  name: string;
  symbol: string;
  decimals: number;
  totalSupply: bigint;
  contractAddress: string;
  deployedAt: number;
}