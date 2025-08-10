import { ethers } from 'ethers';
import type { A2AClient } from '../client/a2aClient';
import { A2AError, ErrorCode } from '../utils/errors';

export interface ScalabilityMetrics {
  transactionsPerSecond: number;
  averageBlockTime: number;
  pendingTransactions: number;
  gasPrice: bigint;
  networkCongestion: 'low' | 'medium' | 'high';
}

export interface BatchTransaction {
  to: string;
  data: string;
  value?: bigint;
}

/**
 * Scalability management service for A2A Network
 */
export class ScalabilityManager {
  constructor(private client: A2AClient) {}

  /**
   * Get current scalability metrics
   */
  async getMetrics(): Promise<ScalabilityMetrics> {
    try {
      const provider = this.client.getProvider();
      const [block, gasPrice, pendingTxs] = await Promise.all([
        provider.getBlock('latest'),
        provider.getFeeData(),
        this.getPendingTransactionCount()
      ]);

      const avgBlockTime = 15; // seconds, can be calculated from recent blocks
      const tps = block ? block.transactions.length / avgBlockTime : 0;

      return {
        transactionsPerSecond: tps,
        averageBlockTime: avgBlockTime,
        pendingTransactions: pendingTxs,
        gasPrice: gasPrice.gasPrice || 0n,
        networkCongestion: this.calculateCongestion(pendingTxs, gasPrice.gasPrice || 0n)
      };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      throw new A2AError(
        ErrorCode.FETCH_FAILED,
        `Failed to get scalability metrics: ${errorMessage}`
      );
    }
  }

  /**
   * Execute multiple transactions in a batch
   */
  async executeBatch(transactions: BatchTransaction[]): Promise<string[]> {
    try {
      const signer = await this.client.getSigner();
      if (!signer) {
        throw new A2AError(ErrorCode.NO_SIGNER, 'No signer available');
      }

      const receipts = [];
      
      // Execute transactions sequentially to maintain order
      for (const tx of transactions) {
        const transaction = await signer.sendTransaction({
          to: tx.to,
          data: tx.data,
          value: tx.value || 0n
        });
        const receipt = await transaction.wait();
        if (receipt) {
          receipts.push(receipt.hash);
        }
      }

      return receipts;
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      throw new A2AError(
        ErrorCode.TRANSACTION_FAILED,
        `Batch execution failed: ${errorMessage}`
      );
    }
  }

  /**
   * Optimize gas settings for a transaction
   */
  async optimizeGas(transaction: ethers.TransactionRequest): Promise<ethers.TransactionRequest> {
    try {
      const provider = this.client.getProvider();
      const [feeData, gasEstimate] = await Promise.all([
        provider.getFeeData(),
        provider.estimateGas(transaction)
      ]);

      return {
        ...transaction,
        gasLimit: gasEstimate * 120n / 100n, // Add 20% buffer
        maxFeePerGas: feeData.maxFeePerGas,
        maxPriorityFeePerGas: feeData.maxPriorityFeePerGas
      };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      throw new A2AError(
        ErrorCode.CALCULATION_FAILED,
        `Failed to optimize gas: ${errorMessage}`
      );
    }
  }

  private async getPendingTransactionCount(): Promise<number> {
    try {
      const provider = this.client.getProvider();
      const signer = await this.client.getSigner();
      if (!signer) return 0;
      
      const address = await signer.getAddress();
      const [nonce, pendingNonce] = await Promise.all([
        provider.getTransactionCount(address, 'latest'),
        provider.getTransactionCount(address, 'pending')
      ]);
      
      return pendingNonce - nonce;
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.warn('Failed to get pending transaction count:', error.message);
      return 0; // Return safe default when unable to fetch pending transactions
    }
  }

  private calculateCongestion(pendingTxs: number, gasPrice: bigint): 'low' | 'medium' | 'high' {
    if (pendingTxs > 5000 || gasPrice > ethers.parseUnits('50', 'gwei')) {
      return 'high';
    } else if (pendingTxs > 1000 || gasPrice > ethers.parseUnits('20', 'gwei')) {
      return 'medium';
    }
    return 'low';
  }
}