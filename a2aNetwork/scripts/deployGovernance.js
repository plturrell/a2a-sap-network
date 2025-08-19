/**
 * @fileoverview Deployment script for A2A Network governance system
 * @description Deploys governance token, timelock, and governor contracts
 */

const { ethers, upgrades } = require('hardhat');
const fs = require('fs');
const path = require('path');

async function main() {
    console.log('🚀 Starting A2A Network Governance Deployment...\n');
    
    const [deployer] = await ethers.getSigners();
    console.log('Deploying with account:', deployer.address);
    console.log('Account balance:', ethers.formatEther(await deployer.provider.getBalance(deployer.address)), 'ETH\n');

    const deploymentConfig = {
        // Governance Token Configuration
        token: {
            name: 'A2A Network Token',
            symbol: 'A2A',
            initialSupply: ethers.parseEther('1000000000'), // 1 billion tokens
        },
        
        // Timelock Configuration
        timelock: {
            minDelay: 2 * 24 * 60 * 60, // 2 days
            proposers: [], // Will be set to governor
            executors: [], // Will be set to governor
        },
        
        // Governor Configuration
        governor: {
            votingDelay: 1 * 24 * 60 * 60, // 1 day
            votingPeriod: 7 * 24 * 60 * 60, // 7 days
            proposalThreshold: ethers.parseEther('10000'), // 10k tokens
            quorumFraction: 10, // 10%
        },
        
        // Emergency Council (multi-sig or trusted address)
        emergencyCouncil: deployer.address, // Replace with actual emergency council
    };

    const deployedContracts = {};

    try {
        // Step 1: Deploy Governance Token
        console.log('📄 Deploying Governance Token...');
        const GovernanceToken = await ethers.getContractFactory('GovernanceToken');
        const governanceToken = await upgrades.deployProxy(
            GovernanceToken,
            [
                deploymentConfig.token.name,
                deploymentConfig.token.symbol,
                deployer.address
            ],
            { initializer: 'initialize' }
        );
        await governanceToken.waitForDeployment();
        
        deployedContracts.governanceToken = await governanceToken.getAddress();
        console.log('✅ Governance Token deployed to:', deployedContracts.governanceToken);
        
        // Step 2: Deploy Timelock Controller
        console.log('\n🕐 Deploying Timelock Controller...');
        const A2ATimelock = await ethers.getContractFactory('A2ATimelock');
        const timelock = await upgrades.deployProxy(
            A2ATimelock,
            [
                deploymentConfig.timelock.minDelay,
                [], // proposers - will be set later
                [], // executors - will be set later
                deployer.address // admin
            ],
            { initializer: 'initialize' }
        );
        await timelock.waitForDeployment();
        
        deployedContracts.timelock = await timelock.getAddress();
        console.log('✅ Timelock Controller deployed to:', deployedContracts.timelock);
        
        // Step 3: Deploy Governor
        console.log('\n🏛️ Deploying Governor...');
        const A2AGovernor = await ethers.getContractFactory('A2AGovernor');
        const governor = await upgrades.deployProxy(
            A2AGovernor,
            [
                deployedContracts.governanceToken,
                deployedContracts.timelock,
                deploymentConfig.emergencyCouncil
            ],
            { initializer: 'initialize' }
        );
        await governor.waitForDeployment();
        
        deployedContracts.governor = await governor.getAddress();
        console.log('✅ Governor deployed to:', deployedContracts.governor);
        
        // Step 4: Configure Timelock Roles
        console.log('\n⚙️ Configuring Timelock Roles...');
        
        // Grant proposer role to governor
        const PROPOSER_ROLE = await timelock.PROPOSER_ROLE();
        await timelock.grantRole(PROPOSER_ROLE, deployedContracts.governor);
        console.log('✅ Granted PROPOSER_ROLE to Governor');
        
        // Grant executor role to governor
        const EXECUTOR_ROLE = await timelock.EXECUTOR_ROLE();
        await timelock.grantRole(EXECUTOR_ROLE, deployedContracts.governor);
        console.log('✅ Granted EXECUTOR_ROLE to Governor');
        
        // Grant executor role to zero address (anyone can execute)
        await timelock.grantRole(EXECUTOR_ROLE, ethers.ZeroAddress);
        console.log('✅ Granted EXECUTOR_ROLE to zero address (public execution)');
        
        // Revoke admin role from deployer (optional - be careful!)
        // await timelock.revokeRole(await timelock.DEFAULT_ADMIN_ROLE(), deployer.address);
        // console.log('✅ Revoked admin role from deployer');
        
        // Step 5: Configure Governor Roles
        console.log('\n⚙️ Configuring Governor Roles...');
        
        // Add initial whitelisted proposers
        await governor.addWhitelistedProposer(deployer.address);
        console.log('✅ Added deployer as whitelisted proposer');
        
        // Step 6: Initial Token Distribution
        console.log('\n💰 Setting up initial token distribution...');
        
        // Delegate voting power to self
        await governanceToken.delegate(deployer.address);
        console.log('✅ Delegated voting power to deployer');
        
        // Create vesting schedule for team (example)
        const teamAllocation = ethers.parseEther('100000000'); // 100M tokens
        await governanceToken.createVestingSchedule(
            deployer.address,
            teamAllocation,
            180 * 24 * 60 * 60, // 180 days cliff
            4 * 365 * 24 * 60 * 60, // 4 years vesting
            true // revocable
        );
        console.log('✅ Created team vesting schedule');
        
        // Step 7: Deploy Supporting Contracts (if needed)
        console.log('\n🔧 Deploying supporting contracts...');
        
        // Deploy Treasury contract (placeholder)
        // const Treasury = await ethers.getContractFactory('Treasury');
        // const treasury = await Treasury.deploy();
        // deployedContracts.treasury = await treasury.getAddress();
        
        // Step 8: Verification and Final Setup
        console.log('\n🔍 Verifying deployment...');
        
        // Verify governance token setup
        const tokenSupply = await governanceToken.totalSupply();
        console.log('Total token supply:', ethers.formatEther(tokenSupply));
        
        // Verify voting power
        const votingPower = await governanceToken.getVotes(deployer.address);
        console.log('Deployer voting power:', ethers.formatEther(votingPower));
        
        // Verify timelock roles
        const hasProposerRole = await timelock.hasRole(PROPOSER_ROLE, deployedContracts.governor);
        const hasExecutorRole = await timelock.hasRole(EXECUTOR_ROLE, deployedContracts.governor);
        console.log('Governor has proposer role:', hasProposerRole);
        console.log('Governor has executor role:', hasExecutorRole);
        
        // Save deployment information
        const deploymentInfo = {
            network: await ethers.provider.getNetwork(),
            deployer: deployer.address,
            timestamp: new Date().toISOString(),
            contracts: deployedContracts,
            configuration: deploymentConfig,
            verificationCommands: generateVerificationCommands(deployedContracts)
        };
        
        const deploymentPath = path.join(__dirname, '..', 'deployments', 'governance-deployment.json');
        await fs.promises.mkdir(path.dirname(deploymentPath), { recursive: true });
        await fs.promises.writeFile(
            deploymentPath,
            JSON.stringify(deploymentInfo, null, 2)
        );
        
        // Step 9: Create sample proposal (for testing)
        console.log('\n📝 Creating sample proposal...');
        
        const sampleProposal = {
            targets: [deployedContracts.governanceToken],
            values: [0],
            calldatas: [
                governanceToken.interface.encodeFunctionData('updateStakingRewardRate', [6]) // Change to 6%
            ],
            description: 'Proposal #1: Update staking reward rate to 6% annually'
        };
        
        try {
            const proposalTx = await governor.proposeWithMetadata(
                sampleProposal.targets,
                sampleProposal.values,
                sampleProposal.calldatas,
                sampleProposal.description,
                1, // PARAMETER_CHANGE category
                '', // IPFS hash (empty for demo)
                5   // Medium impact
            );
            const receipt = await proposalTx.wait();
            
            // Extract proposal ID from events
            const proposalCreatedEvent = receipt.logs.find(
                log => log.address === deployedContracts.governor
            );
            
            if (proposalCreatedEvent) {
                console.log('✅ Sample proposal created successfully');
            }
        } catch (error) {
            console.log('⚠️ Could not create sample proposal:', error.message);
        }
        
        // Final Summary
        console.log('\n🎉 Governance Deployment Complete!');
        console.log('==========================================');
        console.log('Governance Token:', deployedContracts.governanceToken);
        console.log('Timelock Controller:', deployedContracts.timelock);
        console.log('Governor:', deployedContracts.governor);
        console.log('Deployment file saved to:', deploymentPath);
        console.log('\n📚 Next steps:');
        console.log('1. Verify contracts on block explorer');
        console.log('2. Set up frontend integration');
        console.log('3. Configure additional governance parameters');
        console.log('4. Test proposal creation and voting');
        console.log('5. Transfer tokens to stakeholders');
        
    } catch (error) {
        console.error('\n❌ Deployment failed:', error);
        throw error;
    }
}

/**
 * Generate verification commands for deployed contracts
 */
function generateVerificationCommands(contracts) {
    return {
        governanceToken: `npx hardhat verify --network <network> ${contracts.governanceToken}`,
        timelock: `npx hardhat verify --network <network> ${contracts.timelock}`,
        governor: `npx hardhat verify --network <network> ${contracts.governor}`
    };
}

/**
 * Deployment helper functions
 */
async function waitForConfirmations(tx, confirmations = 2) {
    console.log(`Waiting for ${confirmations} confirmations...`);
    await tx.wait(confirmations);
    console.log('✅ Transaction confirmed');
}

async function estimateGasCosts() {
    // Estimate deployment costs
    const gasPrice = await ethers.provider.getGasPrice();
    console.log('Current gas price:', ethers.formatUnits(gasPrice, 'gwei'), 'gwei');
    
    // Rough estimates for contract deployments
    const estimates = {
        governanceToken: 3000000,
        timelock: 2500000,
        governor: 4000000,
        setup: 500000
    };
    
    const totalGas = Object.values(estimates).reduce((a, b) => a + b, 0);
    const totalCost = gasPrice * BigInt(totalGas);
    
    console.log('Estimated deployment cost:', ethers.formatEther(totalCost), 'ETH');
    
    return totalCost;
}

// Execute deployment
if (require.main === module) {
    main()
        .then(() => process.exit(0))
        .catch((error) => {
            console.error(error);
            process.exit(1);
        });
}

module.exports = { main };