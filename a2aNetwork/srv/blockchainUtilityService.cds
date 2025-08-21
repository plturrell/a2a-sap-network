/**
 * @fileoverview Blockchain Utility Service - CDS Definition
 * @since 1.0.0
 * @module blockchainUtilityService
 * 
 * CDS service definition for blockchain utility operations
 * Replaces Express middleware with proper CAP service actions
 */

namespace a2a.blockchain;

/**
 * BlockchainUtilityService - Utility operations for blockchain
 * Provides gas price, block info, transaction details
 */
service BlockchainUtilityService @(path: '/api/v1/blockchain/utils') {
    
    /**
     * Get current gas price in various units
     */
    action getGasPrice() returns {
        wei: String;
        gwei: String;
        eth: String;
    };
    
    /**
     * Get block information by number
     * @param blockNumber - Block number or 'latest'
     */
    action getBlock(blockNumber: String) returns {
        number: Integer64;
        hash: String;
        parentHash: String;
        timestamp: Integer64;
        gasLimit: String;
        gasUsed: String;
        miner: String;
        transactionCount: Integer;
    };
    
    /**
     * Get transaction details by hash
     * @param txHash - Transaction hash
     */
    action getTransaction(txHash: String) returns {
        hash: String;
        ![from]: String;
        to: String;
        value: String;
        gas: String;
        gasPrice: String;
        nonce: Integer;
        blockNumber: Integer64;
        blockHash: String;
        status: Boolean;
    };
    
    /**
     * Get account balance
     * @param address - Ethereum address
     */
    action getBalance(address: String) returns {
        wei: String;
        ether: String;
    };
    
    /**
     * Estimate gas for a transaction
     */
    action estimateGas(params: {
        ![from]: String;
        to: String;
        value: String;
        data: String;
    }) returns {
        gas: String;
        gasPrice: String;
        totalCost: String;
    };
}