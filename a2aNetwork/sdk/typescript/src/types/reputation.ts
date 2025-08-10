export interface Reputation {
  address: string;
  score: number;
  totalInteractions: number;
  positiveInteractions: number;
  negativeInteractions: number;
  disputesWon: number;
  disputesLost: number;
  lastUpdated: Date;
}

export interface ReputationChange {
  address: string;
  previousScore: number;
  newScore: number;
  change: number;
  reason: string;
  timestamp: Date;
  transactionHash: string;
}

export interface ReputationConfig {
  minScore: number;
  maxScore: number;
  decayRate: number;
  updateInterval: number;
}