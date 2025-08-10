export interface GovernanceProposal {
  id: string;
  proposer: string;
  description: string;
  startBlock: number;
  endBlock: number;
  forVotes: bigint;
  againstVotes: bigint;
  abstainVotes: bigint;
  executed: boolean;
  canceled: boolean;
  eta?: number;
  targets: string[];
  values: bigint[];
  signatures: string[];
  calldatas: string[];
}

export interface Vote {
  voter: string;
  proposalId: string;
  support: boolean;
  votes: bigint;
  reason?: string;
  timestamp: Date;
}

export interface GovernanceConfig {
  votingDelay: number;
  votingPeriod: number;
  proposalThreshold: bigint;
  quorumVotes: bigint;
}