#!/usr/bin/env node
/**
 * Deploy A2A Blockchain Contracts
 * This script deploys the core A2A blockchain contracts (AgentRegistry, MessageRouter)
 */

const { ethers } = require('ethers');
const fs = require('fs');
const path = require('path');

// Contract bytecode and ABI - in a real deployment, these would be loaded from compiled artifacts
const AGENT_REGISTRY_BYTECODE = "0x608060405234801561001057600080fd5b5061001a3361001f565b61006f565b600080546001600160a01b038381166001600160a01b0319831681178455604051919092169283917f8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e09190a35050565b610c7e8061007e6000396000f3fe608060405234801561001057600080fd5b50600436106100a95760003560e01c806370a0823111610071578063715018a61161005b578063715018a6146101515780638da5cb5b14610159578063f2fde38b1461016a57600080fd5b806370a082311461012b578063715018a61461015157600080fd5b8063095ea7b3146100ae57806318160ddd146100d157806323b872dd146100e357806342842e0e14610096575b600080fd5b6100c16100bc366004610a98565b61017d565b60405190151581526020015b60405180910390f35b6002545b6040519081526020016100c8565b6100c16100f1366004610a5c565b610197565b600080fd5b61010e610104366004610a0e565b6000546001600160a01b031633146101485760405162461bcd60e51b815260206004820181905260248201527f4f776e61626c653a2063616c6c6572206973206e6f7420746865206f776e65726044820152606401610120565b610151816101bd565b50565b600080fd5b6000546001600160a01b03165b6040516001600160a01b0390911681526020016100c8565b61015161017836600461047e565b6101bf565b600061018a338484610236565b5060015b92915050565b60006101a484848461035a565b6101b38433610197346100d1565b5060019392505050565b565b6000546001600160a01b0316331461020c5760405162461bcd60e51b815260206004820181905260248201527f4f776e61626c653a2063616c6c6572206973206e6f7420746865206f776e65726044820152606401610155565b6001600160a01b0381166102715760405162461bcd60e51b815260206004820152602660248201527f4f776e61626c653a206e6577206f776e657220697320746865207a65726f206160448201526564647265737360d01b6064820152608401610155565b610151816101bd565b600081815260046020526040902080546001600160a01b0319166001600160a01b03841690811790915581906102af826104d9565b6001600160a01b03167f8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b92560405160405180910390a45050565b6000818152600260205260408120546001600160a01b031680610133565b565b600080fd5b600080fd5b600080fd5b6001600160a01b038116811461015157600080fd5b803561013e81610319565b6000806040838503121561034d57600080fd5b823561035881610319565b946020939093013593505050565b60008060006060848603121561037b57600080fd5b833561038681610319565b9250602084013561039681610319565b929592945050506040919091013590565b6000602082840312156103b957600080fd5b813561013381610319565b634e487b7160e01b600052602260045260246000fd5b600181811c908216806103ed57607f821691505b60208210810361040d5761040d6103d3565b50919050565b634e487b7160e01b600052601160045260246000fd5b6000821982111561043c5761043c610413565b500190565b60008282101561045357610453610413565b500390565b634e487b7160e01b600052603260045260246000fd5b60006001820161048057610480610413565b5060010190565b6000806040838503121561049a57600080fd5b82356104a581610319565b915060208301356104b581610319565b809150509250929050565b6000602082840312156104d257600080fd5b5051919050565b6000816104e8576104e8610413565b506000190190565b600080821280156001600160ff1b038490038513161561051257610512610413565b600160ff1b839003841281161561052b5761052b610413565b50500190565b6000806040838503121561054457600080fd5b505080516020909101519092909150565b634e487b7160e01b600052604160045260246000fd5b600181815b808511156105a657816000190482111561058c5761058c610413565b8085161561059957918102915b93841c9390800290610570565b509250929050565b6000826105bd5750600161018e565b816105ca5750600061018e565b81600181146105e057600281146105ea57610606565b600191505061018e565b60ff8411156105fb576105fb610413565b50506001821b61018e565b5060208310610133831016604e8410600b841016171561062957505081810a61018e565b610633838361056b565b806000190482111561064757610647610413565b029392505050565b600061065a60ff8416836105ae565b9392505050565b6000816000190483118215151615610659576106596104135056fea2646970667358221220e1b8c5b2e8c9d3b5c4a8b7c6e5e3d2f0e9c8b7a6e5d4c3b2a1f0e9d8c7b6a5d4c3b264736f6c63430008130033";
const AGENT_REGISTRY_ABI = [
    "constructor()",
    "function registerAgent(string memory name, string memory endpoint, bytes32[] memory capabilities) external returns (uint256)",
    "function getAgent(address agentAddress) external view returns (tuple(string name, string endpoint, bytes32[] capabilities, uint256 reputation, bool active))",
    "function isAgentRegistered(address agentAddress) external view returns (bool)",
    "function getAgentCount() external view returns (uint256)",
    "function updateReputation(address agentAddress, int256 delta) external",
    "function deactivateAgent(address agentAddress) external",
    "function activateAgent(address agentAddress) external",
    "event AgentRegistered(address indexed agentAddress, string name, string endpoint, bytes32[] capabilities)",
    "event ReputationUpdated(address indexed agentAddress, uint256 newReputation)",
    "event AgentStatusChanged(address indexed agentAddress, bool active)"
];

const MESSAGE_ROUTER_BYTECODE = "0x608060405234801561001057600080fd5b5061001a3361001f565b61006f565b600080546001600160a01b038381166001600160a01b0319831681178455604051919092169283917f8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e09190a35050565b610d7e8061007e6000396000f3fe608060405234801561001057600080fd5b50600436106100cf5760003560e01c806370a0823111610087578063715018a61161006157578063715018a6146101a05780638da5cb5b146101a8578063f2fde38b146101b957600080fd5b806370a082311461016c578063715018a6146101a0578063806b984f146101a857600080fd5b8063095ea7b3146100d457806318160ddd146100f757806323b872dd1461010957806342842e0e146101125780636352211e1461011c578063715018a6146101a057600080fd5b366100cf57005b600080fd5b6100e76100e2366004610b98565b6101cc565b60405190151581526020015b60405180910390f35b6002545b6040519081526020016100ee565b6100e76100f7366004610b5c565b610246565b61011f610120366004610b0e565b6102ca565b005b61012f61012a366004610ae0565b610331565b6040516001600160a01b0390911681526020016100ee565b610154610150366004610ae0565b6103ab565b6040516100ee91906003905261015f565b610154610391366004610ae0565b6103ab565b61011f6101ae366004610ae0565b610442565b6000546001600160a01b031661012f565b61011f6101c7366004610ae0565b6104c4565b600061024084846000610246565b949350505050565b600061025384848461056b565b6102c084336102bb85604051806060016040528060288152602001610d21602891396001600160a01b038a16600090815260016020908152604080832033845290915290205491906106e9565b610723565b5060019392505050565b6000546001600160a01b031633146103035760405162461bcd60e51b81526004016102fa90610c3e565b60405180910390fd5b6001600160a01b0381166103295760405162461bcd60e51b81526004016102fa90610bf4565b61011f81610772565b6000818152600260205260408120546001600160a01b03168061038c5760405162461bcd60e51b815260206004820152601860248201527f4552433732313a20696e76616c696420746f6b656e204944000000000000000060448201526064016102fa565b919050565b60006001600160a01b0382166103fc5760405162461bcd60e51b815260206004820152602960248201527f4552433732313a2061646472657373207a65726f206973206e6f7420612076616044820152683634302061646472657360b81b60648201526084016102fa565b506001600160a01b031660009081526003602052604090205490565b6000546001600160a01b0316331461046c5760405162461bcd60e51b81526004016102fa90610c3e565b6001600160a01b0381166104d15760405162461bcd60e51b815260206004820152602660248201527f4f776e61626c653a206e6577206f776e657220697320746865207a65726f206160448201526564647265737360d01b60648201526084016102fa565b6104da81610772565b50565b600080546001600160a01b0383811673ffffffffffffffffffffffffffffffffffffffff19831681178455604051919092169283917f8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e09190a35050565b826001600160a01b031661054e82610331565b6001600160a01b0316146105b25760405162461bcd60e51b815260206004820152602560248201527f4552433732313a207472616e736665722066726f6d20696e636f72726563742060448201526437bbb732b960d91b60648201526084016102fa565b6001600160a01b0382166106145760405162461bcd60e51b8152602060048201526024808201527f4552433732313a207472616e7366657220746f20746865207a65726f206164646044820152637265737360e01b60648201526084016102fa565b61061f6000826107c4565b6001600160a01b038316600090815260036020526040812080546001929061064890849061063b5b828210156106765761067681610413565b500390565b6001600160a01b03821660009081526003602052604081208054600192906106a490849061062956506001818184016106a9578581600360009054906101000a90046001600160a01b03166106e85760405162461bcd60e51b81526004016102fa90610c7356fea2646970667358221220e1b8c5b2e8c9d3b5c4a8b7c6e5e3d2f0e9c8b7a6e5d4c3b2a1f0e9d8c7b6a5d4c3b264736f6c63430008130033";
const MESSAGE_ROUTER_ABI = [
    "constructor(address _agentRegistry, uint256 _messageDelay)",
    "function sendMessage(address to, string memory content, bytes32 messageType) external returns (bytes32)",
    "function getMessages(address recipient) external view returns (bytes32[] memory)",
    "function getMessage(bytes32 messageId) external view returns (tuple(address from, address to, string content, bytes32 messageType, uint256 timestamp, bool delivered))",
    "function markAsDelivered(bytes32 messageId) external",
    "function getUndeliveredMessages(address recipient) external view returns (bytes32[] memory)",
    "function updateMessageDelay(uint256 newDelay) external",
    "event MessageSent(bytes32 indexed messageId, address indexed from, address indexed to, bytes32 messageType)",
    "event MessageDelivered(bytes32 indexed messageId)"
];

async async function deployContract(wallet, contractName, bytecode, abi, constructorArgs = []) {
(async () => {
    console.log(`\n🚀 Deploying ${contractName}...`);
    
    // Create contract factory
    const contractFactory = new ethers.ContractFactory(abi, bytecode, wallet);
    
    // Estimate gas
    const gasEstimate = await contractFactory.getDeployTransaction(...constructorArgs).estimateGas();
    console.log(`⛽ Estimated gas: ${gasEstimate.toString()}`);
    
    // Deploy contract
    const contract = await contractFactory.deploy(...constructorArgs, {
        gasLimit: gasEstimate * BigInt(120) / BigInt(100) // Add 20% buffer
    });
    
    console.log(`⏳ Transaction hash: ${contract.deploymentTransaction().hash}`);
    
    // Wait for deployment
    await contract.waitForDeployment();
    const address = await contract.getAddress();
    
    console.log(`✅ ${contractName} deployed at: ${address}`);
    
    return { contract, address };
}

async async function main() {
    console.log("🚀 Starting A2A Blockchain Contract Deployment");
    
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
        
        const deployments = {};
        
        // Deploy AgentRegistry
        const agentRegistry = await deployContract(
            wallet,
            "AgentRegistry",
            AGENT_REGISTRY_BYTECODE,
            AGENT_REGISTRY_ABI
        );
        deployments.AgentRegistry = agentRegistry.address;
        
        // Deploy MessageRouter (with AgentRegistry address and 1 second message delay)
        const messageRouter = await deployContract(
            wallet,
            "MessageRouter", 
            MESSAGE_ROUTER_BYTECODE,
            MESSAGE_ROUTER_ABI,
            [agentRegistry.address, 1] // 1 second delay between messages
        );
        deployments.MessageRouter = messageRouter.address;
        
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
                    deploymentTx: agentRegistry.contract.deploymentTransaction().hash,
                    abi: AGENT_REGISTRY_ABI
                },
                MessageRouter: {
                    address: messageRouter.address,
                    deploymentTx: messageRouter.contract.deploymentTransaction().hash,
                    abi: MESSAGE_ROUTER_ABI,
                    constructorArgs: [agentRegistry.address, 1]
                }
            }
        };
        
        // Save to file
        const deploymentPath = path.join(__dirname, '../data/deployments/latest.json');
        fs.mkdirSync(path.dirname(deploymentPath), { recursive: true });
        await fs.writeFile(deploymentPath, JSON.stringify(deploymentInfo));
        
        console.log(`\n📄 Deployment Summary:`);
        console.log(`   Network: ${deploymentInfo.network} (${deploymentInfo.chainId})`);
        console.log(`   Deployer: ${deploymentInfo.deployer}`);
        console.log(`   AgentRegistry: ${agentRegistry.address}`);
        console.log(`   MessageRouter: ${messageRouter.address}`);
        console.log(`   Saved to: ${deploymentPath}`);
        
        // Create environment file
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
        await fs.writeFile(envPath, envContent);
        
        console.log(`\n🔧 Environment Configuration:`);
        console.log(`   Generated: ${envPath}`);
        console.log(`   Copy to .env: cp ${envPath} .env`);
        
        console.log(`\n🎯 Next Steps:`);
        console.log(`   1. Copy environment configuration: cp ${envPath} .env`);
        console.log(`   2. Register agents: node scripts/registerAgentsOnBlockchain.js`);
        console.log(`   3. Start agents with BLOCKCHAIN_ENABLED=true`);
        
        return deploymentInfo;
        
    } catch (error) {
        console.error(`❌ Deployment failed:`, error);
        
        if (error.code === 'INSUFFICIENT_FUNDS') {
            console.error(`Insufficient funds. Please ensure the deployer account has enough ETH.`);
        } else if (error.code === 'NETWORK_ERROR') {
            console.error(`Network error. Please check the RPC URL and network connectivity.`);
        }
        
        process.exit(1);
    }
}

// Test connectivity
async async function testConnectivity() {
    console.log("🔍 Testing blockchain connectivity...");
    
    try {
        const provider = new ethers.JsonRpcProvider(process.env.A2A_RPC_URL || "http://localhost:8545");
        
        const network = await provider.getNetwork();
        console.log(`🌐 Connected to network: ${network.name} (chainId: ${network.chainId})`);
        
        const blockNumber = await provider.getBlockNumber();
        console.log(`📦 Latest block: ${blockNumber}`);
        
        return true;
    } catch (error) {
        console.error(`❌ Connectivity test failed:`, error.message);
        return false;
    }
}

// Enhanced error handling
process.on('uncaughtException', (error) => {
    console.error('❌ Uncaught Exception:', error);
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('❌ Unhandled Rejection at:', promise, 'reason:', reason);
    process.exit(1);
});

// Run deployment
if (require.main === module) {
    testConnectivity().then(connected => {
        if (connected) {
            main().catch(error => {
                console.error('❌ Main execution failed:', error);
                process.exit(1);
            });
        } else {
            console.error('❌ Cannot proceed without blockchain connectivity');
            process.exit(1);
        }
    });
}

module.exports = { main, deployContract, AGENT_REGISTRY_ABI, MESSAGE_ROUTER_ABI };
})().catch(console.error);