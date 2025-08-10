// import { ethers } from 'ethers';
import type { A2AClient } from '../client/a2aClient';
import { A2AError, ErrorCode } from '../utils/errors';

export interface Proposal {
  id: string;
  proposer: string;
  description: string;
  forVotes: bigint;
  againstVotes: bigint;
  startTime: Date;
  endTime: Date;
  executed: boolean;
  status: 'pending' | 'active' | 'passed' | 'rejected' | 'executed';
}

export interface CreateProposalParams {
  description: string;
  data: string;
}

export interface VoteParams {
  proposalId: string;
  support: boolean;
}

/**
 * Governance management service for A2A Network
 */
export class GovernanceManager {
  constructor(private client: A2AClient) {}

  /**
   * Create a new governance proposal
   */
  async createProposal(params: CreateProposalParams): Promise<string> {
    try {
      const contract = await this.client.getContract('Governance');
      const tx = await contract.propose(params.description, params.data);
      const receipt = await tx.wait();
      
      // Extract proposal ID from events
      const event = receipt.logs?.find((log: ethers.Log) => 
        log.topics[0] === contract.interface.getEvent('ProposalCreated').topicHash
      );
      
      if (!event) {
        throw new Error('ProposalCreated event not found');
      }
      
      const decodedEvent = contract.interface.decodeEventLog(
        'ProposalCreated',
        event!.data,
        event!.topics
      );
      
      return decodedEvent.proposalId.toString();
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      throw new A2AError(
        ErrorCode.TRANSACTION_FAILED,
        `Failed to create proposal: ${errorMessage}`
      );
    }
  }

  /**
   * Vote on a proposal
   */
  async vote(params: VoteParams): Promise<string> {
    try {
      const contract = await this.client.getContract('Governance');
      const tx = await contract.vote(params.proposalId, params.support);
      const receipt = await tx.wait();
      
      return receipt.hash;
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      throw new A2AError(
        ErrorCode.TRANSACTION_FAILED,
        `Failed to vote: ${errorMessage}`
      );
    }
  }

  /**
   * Get proposal details
   */
  async getProposal(proposalId: string): Promise<Proposal> {
    try {
      const contract = await this.client.getContract('Governance');
      const proposal = await contract.getProposal(proposalId);
      
      return {
        id: proposalId,
        proposer: proposal.proposer,
        description: proposal.description,
        forVotes: proposal.forVotes,
        againstVotes: proposal.againstVotes,
        startTime: new Date(Number(proposal.startTime) * 1000),
        endTime: new Date(Number(proposal.endTime) * 1000),
        executed: proposal.executed,
        status: this.getProposalStatus(proposal)
      };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      throw new A2AError(
        ErrorCode.FETCH_FAILED,
        `Failed to get proposal: ${errorMessage}`
      );
    }
  }

  /**
   * Execute a passed proposal
   */
  async executeProposal(proposalId: string): Promise<string> {
    try {
      const contract = await this.client.getContract('Governance');
      const tx = await contract.execute(proposalId);
      const receipt = await tx.wait();
      
      return receipt.hash;
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      throw new A2AError(
        ErrorCode.TRANSACTION_FAILED,
        `Failed to execute proposal: ${errorMessage}`
      );
    }
  }

  private getProposalStatus(proposal: {
    executed: boolean;
    startTime: bigint;
    endTime: bigint;
    forVotes: bigint;
    againstVotes: bigint;
  }): Proposal['status'] {
    const now = Date.now() / 1000;
    
    if (proposal.executed) return 'executed';
    if (now < Number(proposal.startTime)) return 'pending';
    if (now > Number(proposal.endTime)) {
      return proposal.forVotes > proposal.againstVotes ? 'passed' : 'rejected';
    }
    return 'active';
  }
}