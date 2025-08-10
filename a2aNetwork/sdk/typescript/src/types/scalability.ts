export interface ScalabilityConfig {
  maxBatchSize: number;
  batchTimeout: number;
  compressionEnabled: boolean;
  shardingEnabled: boolean;
  layerTwoEnabled: boolean;
}

export interface BatchResult {
  batchId: string;
  transactions: string[];
  status: 'pending' | 'processing' | 'completed' | 'failed';
  timestamp: Date;
  gasUsed?: bigint;
  error?: string;
}

export interface ShardInfo {
  shardId: number;
  nodeCount: number;
  transactionCount: number;
  storageUsed: bigint;
  lastBlockNumber: number;
}