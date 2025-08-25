const { ethers } = require('ethers');
const fs = require('fs');
const path = require('path');

(async () => {
    console.log('🚀 Deploying test contracts to local blockchain...');

    // Connect to local blockchain
    const provider = new ethers.JsonRpcProvider('http://localhost:8545');
    const signer = await provider.getSigner(0);

    // Simple test contract ABI and bytecode
    const TestContract = {
        abi: [
            "function register(string memory name, address addr) public",
            "function getAgent(address addr) public view returns (string memory)",
            "event AgentRegistered(address indexed agent, string name)"
        ],
        bytecode: "0x608060405234801561001057600080fd5b50610771806100206000396000f3fe..."
    };

    // Deploy multiple test contracts
    const contracts = [
        'AgentRegistry',
        'MessageRouter',
        'BusinessDataCloud',
        'ServiceMarketplace',
        'ReputationExchange'
    ];

    const deployedAddresses = {};

    for (const contractName of contracts) {
        console.log(`Deploying ${contractName}...`);
        const factory = new ethers.ContractFactory(TestContract.abi, TestContract.bytecode, signer);
        const contract = await factory.deploy();
        await contract.waitForDeployment();
        const address = await contract.getAddress();
        deployedAddresses[contractName] = address;
        console.log(`✅ ${contractName} deployed at: ${address}`);
    }

    // Save deployed addresses
    const addressesPath = path.join(__dirname, '../data/deployed-contracts.json');
    await fs.writeFile(addressesPath, JSON.stringify(deployedAddresses));

    console.log('\n✅ All test contracts deployed successfully!');
    return deployedAddresses;
}

if (require.main === module) {
    deployTestContracts().catch(console.error);
}

module.exports = { deployTestContracts };

})().catch(console.error);