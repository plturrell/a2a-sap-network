#!/usr/bin/env node
/**
 * Simple A2A Contract Deployment Script
 * Deploys AgentRegistry and MessageRouter contracts using compiled artifacts
 */

require('dotenv').config();
const { ethers } = require('ethers');
const fs = require('fs');
const path = require('path');

// Load compiled artifacts
function loadCompiledContract(contractName) {
    try {
        const artifactPath = path.join(__dirname, `../out/${contractName}.sol/${contractName}.json`);
        const artifact = JSON.parse(fs.readFileSync(artifactPath, 'utf8'));
        return {
            bytecode: artifact.bytecode.object,
            abi: artifact.abi
        };
    } catch (error) {
        console.error(`Failed to load ${contractName} artifact:`, error);
        throw error;
    }
}

async function deployContract(wallet, contractName, constructorArgs = []) {
    console.log(`\n🚀 Deploying ${contractName}...`);
    
    const { bytecode, abi } = loadCompiledContract(contractName);
    
    // Get fresh nonce to avoid conflicts
    const nonce = await wallet.provider.getTransactionCount(wallet.address, 'latest');
    console.log(`📋 Using nonce: ${nonce}`);
    
    // Create contract factory
    const contractFactory = new ethers.ContractFactory(abi, bytecode, wallet);
    
    // Deploy contract with explicit nonce
    const contract = await contractFactory.deploy(...constructorArgs, { nonce });
    
    console.log(`⏳ Transaction hash: ${contract.deploymentTransaction().hash}`);
    
    // Wait for deployment
    await contract.waitForDeployment();
    const address = await contract.getAddress();
    
    console.log(`✅ ${contractName} deployed at: ${address}`);
    
    return { contract, address, abi };
}

async function main() {
    console.log("🚀 Starting A2A Contract Deployment");
    
    try {
        // Initialize provider and wallet
        const provider = new ethers.JsonRpcProvider(process.env.A2A_RPC_URL || "http://localhost:8545");
        const privateKey = process.env.A2A_PRIVATE_KEY;
        
        if (!privateKey) {
            throw new Error("A2A_PRIVATE_KEY environment variable is required");
        }
        
        const wallet = new ethers.Wallet(privateKey, provider);
        console.log(`🔑 Deployer address: ${wallet.address}`);
        
        // Check balance
        const balance = await provider.getBalance(wallet.address);
        console.log(`💰 Deployer balance: ${ethers.formatEther(balance)} ETH`);
        
        if (balance === 0n) {
            throw new Error("Deployer account has no ETH. Please fund the account first.");
        }
        
        // Deploy AgentRegistry (with 1 confirmation for testing)
        const agentRegistry = await deployContract(wallet, "AgentRegistry", [1]);
        
        // Wait a moment to ensure nonce is properly incremented
        console.log("⏳ Waiting for nonce to update...");
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Deploy MessageRouter (with AgentRegistry address and 1 second message delay)
        const messageRouter = await deployContract(wallet, "MessageRouter", [agentRegistry.address, 1]);
        
        // Save deployment info
        const deploymentInfo = {
            network: process.env.A2A_NETWORK || "localhost",
            chainId: (await provider.getNetwork()).chainId.toString(),
            deployer: wallet.address,
            deployedAt: new Date().toISOString(),
            blockNumber: await provider.getBlockNumber(),
            contracts: {
                AgentRegistry: {
                    address: agentRegistry.address,
                    deploymentTx: agentRegistry.contract.deploymentTransaction().hash
                },
                MessageRouter: {
                    address: messageRouter.address,
                    deploymentTx: messageRouter.contract.deploymentTransaction().hash
                }
            }
        };
        
        // Save to file
        const deploymentPath = path.join(__dirname, '../data/deployments/latest.json');
        fs.mkdirSync(path.dirname(deploymentPath), { recursive: true });
        fs.writeFileSync(deploymentPath, JSON.stringify(deploymentInfo, null, 2));
        
        console.log(`\n📄 Deployment Summary:`);
        console.log(`   Network: ${deploymentInfo.network} (${deploymentInfo.chainId})`);
        console.log(`   Deployer: ${deploymentInfo.deployer}`);
        console.log(`   AgentRegistry: ${agentRegistry.address}`);
        console.log(`   MessageRouter: ${messageRouter.address}`);
        console.log(`   Saved to: ${deploymentPath}`);
        
        // Update environment file
        const envContent = `# A2A Blockchain Configuration - Generated ${new Date().toISOString()}
A2A_RPC_URL=${process.env.A2A_RPC_URL || "http://localhost:8545"}
A2A_NETWORK=${process.env.A2A_NETWORK || "localhost"}
A2A_CHAIN_ID=${deploymentInfo.chainId}

# Contract Addresses
A2A_AGENT_REGISTRY_ADDRESS=${agentRegistry.address}
A2A_MESSAGE_ROUTER_ADDRESS=${messageRouter.address}

# Private Keys (update with your actual keys)
A2A_PRIVATE_KEY=${privateKey}
QC_AGENT_PRIVATE_KEY=${privateKey}
DM_AGENT_PRIVATE_KEY=${privateKey}

# Blockchain Features
BLOCKCHAIN_ENABLED=true
`;
        
        const envPath = path.join(__dirname, '../.env.deployed');
        fs.writeFileSync(envPath, envContent);
        
        console.log(`\n🔧 Environment Configuration:`);
        console.log(`   Generated: ${envPath}`);
        
        console.log(`\n🎯 Next Steps:`);
        console.log(`   1. Copy environment: cp ${envPath} .env`);
        console.log(`   2. Register agents: node scripts/registerAgentsOnBlockchain.js`);
        console.log(`   3. Start agents with BLOCKCHAIN_ENABLED=true`);
        
        return deploymentInfo;
        
    } catch (error) {
        console.error(`❌ Deployment failed:`, error);
        process.exit(1);
    }
}

// Run deployment
if (require.main === module) {
    main().catch(error => {
        console.error('❌ Main execution failed:', error);
        process.exit(1);
    });
}

module.exports = { main, deployContract };