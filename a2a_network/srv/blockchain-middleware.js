const express = require('express');
const router = express.Router();
const cds = require('@sap/cds');

// Middleware for blockchain-specific operations
router.use(async (req, res, next) => {
    try {
        req.blockchain = await cds.connect.to('BlockchainService');
        next();
    } catch (error) {
        res.status(503).json({ error: 'Blockchain service unavailable' });
    }
});

// Get current gas price
router.get('/gas-price', async (req, res) => {
    try {
        const gasPrice = await req.blockchain.web3.eth.getGasPrice();
        res.json({
            wei: gasPrice.toString(),
            gwei: req.blockchain.web3.utils.fromWei(gasPrice, 'gwei'),
            eth: req.blockchain.web3.utils.fromWei(gasPrice, 'ether')
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Get block information
router.get('/block/:number', async (req, res) => {
    try {
        const block = await req.blockchain.web3.eth.getBlock(req.params.number);
        res.json(block);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Get transaction
router.get('/transaction/:hash', async (req, res) => {
    try {
        const tx = await req.blockchain.web3.eth.getTransaction(req.params.hash);
        const receipt = await req.blockchain.web3.eth.getTransactionReceipt(req.params.hash);
        res.json({ transaction: tx, receipt });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Get contract information
router.get('/contract/:name', async (req, res) => {
    try {
        const contract = req.blockchain.contracts[req.params.name];
        if (!contract) {
            return res.status(404).json({ error: 'Contract not found' });
        }
        
        res.json({
            address: contract.address,
            methods: contract.abi
                .filter(item => item.type === 'function')
                .map(item => ({
                    name: item.name,
                    inputs: item.inputs,
                    outputs: item.outputs,
                    stateMutability: item.stateMutability
                }))
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Execute read-only contract method
router.post('/contract/:name/call', async (req, res) => {
    try {
        const { method, params = [] } = req.body;
        const contract = req.blockchain.contracts[req.params.name];
        
        if (!contract) {
            return res.status(404).json({ error: 'Contract not found' });
        }
        
        const result = await contract.web3.methods[method](...params).call();
        res.json({ result });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Estimate gas for transaction
router.post('/estimate-gas', async (req, res) => {
    try {
        const { contract, method, params = [], from } = req.body;
        const contractInstance = req.blockchain.contracts[contract];
        
        if (!contractInstance) {
            return res.status(404).json({ error: 'Contract not found' });
        }
        
        const gasEstimate = await contractInstance.web3.methods[method](...params)
            .estimateGas({ from });
        
        res.json({
            gas: gasEstimate.toString(),
            estimatedCost: {
                wei: (gasEstimate * 20000000000).toString(), // 20 gwei
                eth: req.blockchain.web3.utils.fromWei((gasEstimate * 20000000000).toString(), 'ether')
            }
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

module.exports = router;