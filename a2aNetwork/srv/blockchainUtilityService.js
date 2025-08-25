/**
 * @fileoverview Blockchain Utility Service - CAP Implementation
 * @since 1.0.0
 * @module blockchainUtilityService
 *
 * CAP service handlers for blockchain utility operations
 * Replaces Express middleware with proper SAP CAP architecture
 */

const cds = require('@sap/cds');
const LOG = cds.log('blockchain-utility');

/**
 * CAP Service Handler for Blockchain Utility Actions
 */
module.exports = function() {

    // Connect to BlockchainService on startup
    let blockchainService;

    this.before('*', async () => {
        if (!blockchainService) {
            try {
                blockchainService = await cds.connect.to('BlockchainService');
            } catch (error) {
                LOG.error('Failed to connect to BlockchainService:', error);
                throw new Error('Blockchain service unavailable');
            }
        }
    });

    // Get current gas price
    this.on('getGasPrice', async (req) => {
        try {
            const gasPrice = await blockchainService.web3.eth.getGasPrice();
            return {
                wei: gasPrice.toString(),
                gwei: blockchainService.web3.utils.fromWei(gasPrice, 'gwei'),
                eth: blockchainService.web3.utils.fromWei(gasPrice, 'ether')
            };
        } catch (error) {
            LOG.error('Error getting gas price:', error);
            req.error(500, 'GAS_PRICE_ERROR', `Failed to get gas price: ${error.message}`);
        }
    });

    // Get block information
    this.on('getBlock', async (req) => {
        const { blockNumber } = req.data;

        if (!blockNumber) {
            req.error(400, 'INVALID_BLOCK_NUMBER', 'Block number is required');
            return;
        }

        try {
            const block = await blockchainService.web3.eth.getBlock(blockNumber);

            if (!block) {
                req.error(404, 'BLOCK_NOT_FOUND', `Block ${blockNumber} not found`);
                return;
            }

            return {
                number: block.number,
                hash: block.hash,
                parentHash: block.parentHash,
                timestamp: block.timestamp,
                gasLimit: block.gasLimit.toString(),
                gasUsed: block.gasUsed.toString(),
                miner: block.miner,
                transactionCount: block.transactions.length
            };
        } catch (error) {
            LOG.error('Error getting block:', error);
            req.error(500, 'BLOCK_ERROR', `Failed to get block: ${error.message}`);
        }
    });

    // Get transaction details
    this.on('getTransaction', async (req) => {
        const { txHash } = req.data;

        if (!txHash || !txHash.match(/^0x[a-fA-F0-9]{64}$/)) {
            req.error(400, 'INVALID_TX_HASH', 'Valid transaction hash is required');
            return;
        }

        try {
            const [tx, receipt] = await Promise.all([
                blockchainService.web3.eth.getTransaction(txHash),
                blockchainService.web3.eth.getTransactionReceipt(txHash)
            ]);

            if (!tx) {
                req.error(404, 'TX_NOT_FOUND', `Transaction ${txHash} not found`);
                return;
            }

            return {
                hash: tx.hash,
                from: tx.from,
                to: tx.to,
                value: tx.value,
                gas: tx.gas.toString(),
                gasPrice: tx.gasPrice.toString(),
                nonce: tx.nonce,
                blockNumber: tx.blockNumber,
                blockHash: tx.blockHash,
                status: receipt ? receipt.status : null
            };
        } catch (error) {
            LOG.error('Error getting transaction:', error);
            req.error(500, 'TX_ERROR', `Failed to get transaction: ${error.message}`);
        }
    });

    // Get account balance
    this.on('getBalance', async (req) => {
        const { address } = req.data;

        if (!address || !blockchainService.web3.utils.isAddress(address)) {
            req.error(400, 'INVALID_ADDRESS', 'Valid Ethereum address is required');
            return;
        }

        try {
            const balanceWei = await blockchainService.web3.eth.getBalance(address);

            return {
                wei: balanceWei.toString(),
                ether: blockchainService.web3.utils.fromWei(balanceWei, 'ether')
            };
        } catch (error) {
            LOG.error('Error getting balance:', error);
            req.error(500, 'BALANCE_ERROR', `Failed to get balance: ${error.message}`);
        }
    });

    // Estimate gas for transaction
    this.on('estimateGas', async (req) => {
        const { params } = req.data;

        if (!params || !params.from || !params.to) {
            req.error(400, 'INVALID_PARAMS', 'From and to addresses are required');
            return;
        }

        try {
            const estimatedGas = await blockchainService.web3.eth.estimateGas({
                from: params.from,
                to: params.to,
                value: params.value || '0',
                data: params.data || '0x'
            });

            const gasPrice = await blockchainService.web3.eth.getGasPrice();
            const totalCost = BigInt(estimatedGas) * BigInt(gasPrice);

            return {
                gas: estimatedGas.toString(),
                gasPrice: gasPrice.toString(),
                totalCost: totalCost.toString()
            };
        } catch (error) {
            LOG.error('Error estimating gas:', error);
            req.error(500, 'GAS_ESTIMATE_ERROR', `Failed to estimate gas: ${error.message}`);
        }
    });

    LOG.info('Blockchain Utility service handlers registered');
};